#!/usr/bin/env python3
"""
Proof-of-concept script for fast self-play data generation using pypkmn.

This script demonstrates the pypkmn integration by:
1. Loading teams from disk
2. Running vectorized self-play battles
3. Saving trajectories to .json.lz4 format
4. Benchmarking performance vs Showdown baseline

Usage:
    python scripts/generate_selfplay_pykmn.py \
        --team_dir ~/metamon_cache/teams/modern_replays_v2 \
        --num_battles 100 \
        --num_envs 16 \
        --save_dir ~/pypkmn_selfplay_test \
        --format gen1ou \
        --benchmark

Performance targets:
    - Sim-only: 100x+ faster than Showdown subprocess
    - End-to-end: 10x+ faster including inference + serialization
"""

import argparse
import time
from pathlib import Path
import numpy as np

from metamon.env.pykmn import (
    load_random_teams,
    PyKMNVectorEnv,
    RandomPolicyRunner,
    LocalPolicyRunner,
    SelfPlayRunner,
    save_trajectories,
    precompute_mappings,
)
from metamon.interface import DefaultObservationSpace, DefaultShapedReward
from metamon.rl.pretrained import get_pretrained_model


def main():
    parser = argparse.ArgumentParser(description="PyKMN self-play data generation PoC")
    parser.add_argument(
        "--team_dir",
        type=str,
        required=True,
        help="Directory containing team files",
    )
    parser.add_argument(
        "--num_battles",
        type=int,
        default=100,
        help="Number of battles to generate",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments (tune based on CPU cores)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save trajectories",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (e.g., gen1ou)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark after generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pretrained model to use (e.g., 'SyntheticRLV2'). If not specified, uses random policy.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Model checkpoint to load (default: model's default checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (cuda/cpu)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Action sampling temperature (higher = more random)",
    )

    args = parser.parse_args()

    # Expand paths
    team_dir = Path(args.team_dir).expanduser()
    save_dir = Path(args.save_dir).expanduser()

    print("=" * 60)
    print("PyKMN Self-Play Data Generation - Proof of Concept")
    print("=" * 60)
    print(f"Team directory: {team_dir}")
    print(f"Number of battles: {args.num_battles}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Battle format: {args.format}")
    print(f"Output directory: {save_dir}")
    print("=" * 60)

    # Load teams
    print(f"\nLoading {args.num_envs * 2} teams from {team_dir}...")
    start_time = time.time()

    try:
        teams_p1 = load_random_teams(team_dir, args.format, args.num_envs)
        teams_p2 = load_random_teams(team_dir, args.format, args.num_envs)
    except Exception as e:
        print(f"Error loading teams: {e}")
        print(f"\nPlease ensure team files exist at: {team_dir}")
        print(f"Team files should have extension: .{args.format}_team")
        return

    load_time = time.time() - start_time
    print(f"Loaded {len(teams_p1) + len(teams_p2)} teams in {load_time:.2f}s")

    # Load pretrained model if specified, otherwise use random policy
    if args.model:
        print(f"\nLoading pretrained model: {args.model}")
        print(f"  Checkpoint: {args.checkpoint or 'default'}")
        print(f"  Device: {args.device}")
        print(f"  Temperature: {args.temperature}")

        start_time = time.time()
        pretrained_cls = get_pretrained_model(args.model)

        # Use model's observation space and reward function
        obs_space = pretrained_cls.observation_space
        reward_fn = pretrained_cls.reward_function

        model_load_time = time.time() - start_time
        print(f"Model config loaded in {model_load_time:.2f}s")
        print(f"  Observation space: {obs_space.__class__.__name__}")
        print(f"  Reward function: {reward_fn.__class__.__name__}")
    else:
        print("\nUsing random policy (no pretrained model specified)")
        obs_space = DefaultObservationSpace()
        reward_fn = DefaultShapedReward()

    # Create vectorized environment
    print(f"\nInitializing {args.num_envs} parallel environments...")
    start_time = time.time()

    vec_env = PyKMNVectorEnv(
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        num_envs=args.num_envs,
        obs_space=obs_space,
        reward_fn=reward_fn,
        battle_format=args.format,
        track_trajectories=True,
        use_trace=False,  # Use no-trace build for performance
    )

    init_time = time.time() - start_time
    print(f"Initialized environments in {init_time:.2f}s")

    # Create policies (need separate instances for each player!)
    if args.model:
        print(f"\nInitializing pretrained policies for both players (this may take a minute)...")
        start_time = time.time()

        print("  Loading policy for Player 1...")
        policy_p1 = LocalPolicyRunner(
            model_name=args.model,
            checkpoint=args.checkpoint,
            device=args.device,
            temperature=args.temperature,
            verbose=args.verbose,
        )

        print("  Loading policy for Player 2...")
        policy_p2 = LocalPolicyRunner(
            model_name=args.model,
            checkpoint=args.checkpoint,
            device=args.device,
            temperature=args.temperature,
            verbose=args.verbose,
        )

        policy_init_time = time.time() - start_time
        print(f"Both policies initialized in {policy_init_time:.2f}s")
    else:
        print("\nCreating random policies...")
        policy_p1 = RandomPolicyRunner()
        policy_p2 = RandomPolicyRunner()

    # Run self-play
    print(f"\nRunning {args.num_battles} self-play battles...")
    start_time = time.time()

    runner = SelfPlayRunner(vec_env, policy_p1=policy_p1, policy_p2=policy_p2)
    trajectories = runner.collect_trajectories(
        num_battles=args.num_battles,
        verbose=args.verbose,
    )

    generation_time = time.time() - start_time
    print(f"Generated {len(trajectories)} trajectories in {generation_time:.2f}s")
    print(f"Average: {generation_time / len(trajectories):.3f}s per battle")
    print(f"Throughput: {len(trajectories) / generation_time:.1f} battles/second")

    # Save trajectories
    print(f"\nSaving trajectories to {save_dir}...")
    start_time = time.time()

    mappings = precompute_mappings()
    save_trajectories(
        trajectories=trajectories,
        output_dir=save_dir,
        mappings=mappings,
        battle_format=args.format,
        verbose=args.verbose,
    )

    save_time = time.time() - start_time
    print(f"Saved {len(trajectories)} trajectories in {save_time:.2f}s")

    # Statistics
    print("\n" + "=" * 60)
    print("Generation Statistics")
    print("=" * 60)
    print(f"Total battles: {len(trajectories)}")
    print(f"Total time: {generation_time + save_time:.2f}s")
    print(f"  - Generation: {generation_time:.2f}s")
    print(f"  - Saving: {save_time:.2f}s")
    print(f"Average battle length: {np.mean([len(t.transitions) for t in trajectories]):.1f} turns")
    print(f"Throughput (end-to-end): {len(trajectories) / (generation_time + save_time):.1f} battles/s")

    # Benchmark
    if args.benchmark:
        print("\n" + "=" * 60)
        print("Performance Benchmark")
        print("=" * 60)
        print("\nRunning sim-only benchmark (no saving)...")

        # Reload env for clean benchmark
        vec_env_bench = PyKMNVectorEnv(
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            num_envs=args.num_envs,
            obs_space=obs_space,
            reward_fn=reward_fn,
            battle_format=args.format,
            track_trajectories=False,  # Disable for pure sim benchmark
            use_trace=False,
        )

        runner_bench = SelfPlayRunner(vec_env_bench, policy_p1=policy, policy_p2=policy)

        start_time = time.time()
        trajectories_bench = runner_bench.collect_trajectories(
            num_battles=min(50, args.num_battles),
            verbose=False,
        )
        sim_time = time.time() - start_time

        print(f"Sim-only throughput: {len(trajectories_bench) / sim_time:.1f} battles/s")
        print(f"\nNote: Showdown baseline is typically ~0.1-1 battles/s")
        print(f"Speedup estimate: {(len(trajectories_bench) / sim_time) / 0.5:.0f}x")

    print("\n" + "=" * 60)
    print("Proof of Concept Complete!")
    print("=" * 60)
    print(f"\nTrajectories saved to: {save_dir}/{args.format}/")
    print("\nNext steps:")
    print("1. Verify trajectories load correctly:")
    print(f"   python -c \"from metamon.env.pykmn import load_trajectory; ")
    print(f"   t = load_trajectory('{save_dir}/{args.format}/...')\"")
    print("2. Test training pipeline integration")
    print("3. Implement full pypkmn Battle state extraction in features.py")


if __name__ == "__main__":
    main()
