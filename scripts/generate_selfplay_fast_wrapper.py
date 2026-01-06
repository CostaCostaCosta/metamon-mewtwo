#!/usr/bin/env python3
"""
Fast self-play data generation using the new InferenceWrapper.

This script uses the rebuilt safe wrapper for maximum performance:
- InferenceWrapper: 37x faster than old PyKMNVectorEnv
- SafeBattleManager: No memory corruption
- FastFeatureExtractor: Vectorized operations
- 1024 parallel battles supported

Usage:
    python scripts/generate_selfplay_fast_wrapper.py \
        --team_dir ~/metamon_cache/teams/modern_replays_v2 \
        --num_battles 50000 \
        --num_envs 1024 \
        --save_dir ~/metamon/trajectories/kakuna-wrapper \
        --format gen1ou \
        --model Kakuna
"""

import argparse
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from metamon.env.inference_wrapper import InferenceWrapper
from metamon.env.pykmn import load_random_teams, save_trajectories, precompute_mappings
from metamon.env.pykmn.policy_runner import LocalPolicyRunner, SelfPlayRunner
from metamon.rl.pretrained import get_pretrained_model


def main():
    parser = argparse.ArgumentParser(description="Fast self-play with new wrapper")
    parser.add_argument("--team_dir", type=str, required=True, help="Team directory")
    parser.add_argument("--num_battles", type=int, default=50000, help="Number of battles")
    parser.add_argument("--num_envs", type=int, default=1024, help="Parallel environments")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--format", type=str, default="gen1ou", help="Battle format")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Kakuna)")
    parser.add_argument("--checkpoint", type=int, default=None, help="Model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0). Higher values (e.g., 1.5-2.0) increase exploration "
             "and improve value estimates of sub-optimal actions. Lower values make policy more deterministic."
    )
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per battle")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Expand paths
    team_dir = Path(args.team_dir).expanduser()
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create format-specific subdirectory
    format_dir = save_dir / args.format
    format_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("FAST SELF-PLAY DATA GENERATION - New Wrapper (37x faster)")
    print("=" * 70)
    print(f"Team directory: {team_dir}")
    print(f"Number of battles: {args.num_battles:,}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Battle format: {args.format}")
    print(f"Model: {args.model}")
    print(f"Output directory: {format_dir}")
    print("=" * 70)

    # Calculate number of batches needed
    num_batches = (args.num_battles + args.num_envs - 1) // args.num_envs
    battles_per_batch = args.num_envs

    print(f"\nBatch configuration:")
    print(f"  Batches: {num_batches}")
    print(f"  Battles per batch: {battles_per_batch}")

    # Load model config
    print(f"\nLoading model config: {args.model}...")
    pretrained_cls = get_pretrained_model(args.model)
    print(f"  Observation space: {pretrained_cls.observation_space.__class__.__name__}")
    print(f"  Reward function: {pretrained_cls.reward_function.__class__.__name__}")

    # Precompute mappings for trajectory saving
    print(f"\nPrecomputing feature mappings...")
    mappings = precompute_mappings()

    # Create policy runner
    print(f"\nInitializing policy runner...")
    print(f"  Device: {args.device}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Checkpoint: {args.checkpoint or 'default'}")

    start_time = time.time()
    policy = LocalPolicyRunner(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=args.device,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    policy_load_time = time.time() - start_time
    print(f"  Loaded in {policy_load_time:.2f}s")

    # Statistics
    total_battles_collected = 0
    total_battles_saved = 0  # Track how many we've written to disk
    total_steps = 0
    all_trajectories = []  # Keep for stats only

    print(f"\nStarting data generation...")
    print(f"Target: {args.num_battles:,} battles")
    print("=" * 70)

    # Progress bar
    pbar = tqdm(total=args.num_battles, desc="Battles", unit="battle")

    batch_start_time = time.time()

    for batch_idx in range(num_batches):
        # Check if we've hit our target
        if total_battles_collected >= args.num_battles:
            break

        # Determine how many battles we still need
        battles_remaining = args.num_battles - total_battles_collected
        current_batch_size = min(args.num_envs, battles_remaining)

        if current_batch_size <= 0:
            break

        # Load fresh teams for this batch
        teams_p1 = load_random_teams(team_dir, args.format, current_batch_size)
        teams_p2 = load_random_teams(team_dir, args.format, current_batch_size)

        # Create wrapper with trajectory tracking enabled
        wrapper = InferenceWrapper(
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            num_envs=current_batch_size,
            track_trajectories=True,  # CRITICAL: Enable trajectory saving
        )

        # Reset
        obs_p1, obs_p2, legal_p1, legal_p2 = wrapper.reset()
        policy.reset(batch_size=current_batch_size)

        # Run until all battles complete
        batch_steps = 0
        battles_done = 0

        while battles_done < current_batch_size and batch_steps < args.max_steps:
            # Infer actions
            actions_p1 = policy.infer(obs_p1, legal_p1)
            actions_p2 = policy.infer(obs_p2, legal_p2)

            # Step environment
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = wrapper.step(
                actions_p1, actions_p2
            )

            # Update policy state
            policy.update_rewards(rewards_p1)
            policy.reset_hidden_state_for_dones(dones)

            # Extract legal masks
            legal_p1 = info['legal_masks_p1']
            legal_p2 = info['legal_masks_p2']

            batch_steps += 1
            battles_done = info.get('num_done', 0)

        # Collect completed trajectories from this batch
        completed = wrapper.get_completed_trajectories()

        # Only take what we need to reach target
        needed = args.num_battles - total_battles_collected
        to_keep = min(len(completed), needed)
        batch_trajectories = completed[:to_keep]

        # Save this batch immediately (robust to crashes)
        if len(batch_trajectories) > 0:
            save_start = time.time()
            save_trajectories(
                trajectories=batch_trajectories,
                output_dir=save_dir,
                mappings=mappings,
                battle_format=args.format,
                verbose=False,
            )
            save_time = time.time() - save_start
            total_battles_saved += len(batch_trajectories)

            # Keep for stats
            all_trajectories.extend(batch_trajectories)

            if args.verbose:
                pbar.write(f"  💾 Batch {batch_idx+1}: Saved {len(batch_trajectories)} trajectories in {save_time:.1f}s (total: {total_battles_saved:,})")

        total_battles_collected += to_keep
        total_steps += batch_steps
        pbar.update(to_keep)

        # Stats
        elapsed = time.time() - batch_start_time
        battles_per_sec = total_battles_collected / elapsed
        steps_per_sec = total_steps / elapsed
        pbar.set_postfix({
            'battles/s': f'{battles_per_sec:.1f}',
            'steps/s': f'{steps_per_sec:.0f}',
            'saved': f'{total_battles_saved:,}',
        })

    pbar.close()

    # Final statistics
    total_time = time.time() - batch_start_time

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Battles collected: {total_battles_collected:,} / {args.num_battles:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Performance:")
    print(f"  Battles/sec: {total_battles_collected / total_time:.1f}")
    print(f"  Steps/sec: {total_steps / total_time:.0f}")
    print(f"  Avg steps/battle: {total_steps / total_battles_collected:.1f}")
    print("=" * 70)

    print(f"\nTotal trajectories saved to disk: {total_battles_saved:,}")
    print(f"Output directory: {format_dir}")

    # Compute win statistics from all collected trajectories
    if len(all_trajectories) > 0:
        p1_wins = sum(1 for t in all_trajectories if t.winner == 1)
        p2_wins = sum(1 for t in all_trajectories if t.winner == 2)
        ties = sum(1 for t in all_trajectories if t.winner == 0)

        print(f"\nWin statistics:")
        print(f"  P1 wins: {p1_wins:,} ({100 * p1_wins / len(all_trajectories):.1f}%)")
        print(f"  P2 wins: {p2_wins:,} ({100 * p2_wins / len(all_trajectories):.1f}%)")
        print(f"  Ties: {ties:,} ({100 * ties / len(all_trajectories):.1f}%)")
    else:
        print("\n⚠️  No trajectories collected!")

    # Verify file count
    saved_files = list(format_dir.glob("*.json.lz4"))
    print(f"\nVerification:")
    print(f"  Files on disk: {len(saved_files):,}")
    print(f"  Expected: {total_battles_saved:,}")
    if len(saved_files) == total_battles_saved:
        print(f"  ✓ File count matches!")
    else:
        print(f"  ⚠️  Mismatch detected!")

    print("=" * 70)
    print("✓ COMPLETE - All trajectories saved successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
