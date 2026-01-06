#!/usr/bin/env python3
"""
Self-play data generation using inference server architecture.

This version completely separates PyKMN simulation from GPU inference,
eliminating all memory corruption issues.

Usage:
    # First, start the inference server in a separate terminal:
    python -m metamon.inference.server --model SyntheticRLV2 --batch_size 64

    # Then run this script:
    python scripts/generate_selfplay_server.py \
        --num_battles 10000 \
        --batch_size 64 \
        --format gen1ou \
        --team_set smogon_pass2 \
        --save_dir ~/selfplay_data
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Set cache directory before imports
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import PyKMNVectorEnv, SelfPlayRunner, load_random_teams, save_trajectories
from metamon.inference.client import RemotePolicyRunner
from metamon.rl.pretrained import get_pretrained_model
from metamon.env.pykmn.features import precompute_mappings


def run_selfplay_with_server(
    num_battles: int,
    batch_size: int,
    format_name: str,
    team_set: str,
    save_dir: Path,
    server_url: str = "http://localhost:8080",
    model_name: str = "SyntheticRLV2",  # For getting obs_space and reward_fn
    verbose: bool = True,
):
    """
    Run self-play using the inference server architecture.

    This function runs PyKMN simulation locally and uses the remote
    inference server for GPU operations, completely avoiding memory corruption.
    """

    # Get observation space and reward function from model
    pretrained_cls = get_pretrained_model(model_name)
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    # Precompute mappings for trajectory saving
    mappings = precompute_mappings()

    # Load teams
    cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
    team_dir = cache_dir / "teams" / team_set
    teams_p1 = load_random_teams(team_dir, format_name, batch_size * 2)
    teams_p2 = load_random_teams(team_dir, format_name, batch_size * 2)

    if verbose:
        print(f"Loaded {len(teams_p1) + len(teams_p2)} teams")

    # Create environment (runs locally, no GPU)
    env = PyKMNVectorEnv(
        num_envs=batch_size,
        teams_p1=teams_p1[:batch_size],
        teams_p2=teams_p2[batch_size:],
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=True,
    )

    # Create remote policy runners (communicate with server)
    policy_p1 = RemotePolicyRunner(server_url=server_url, model_name=model_name)
    policy_p2 = RemotePolicyRunner(server_url=server_url, model_name=model_name)

    # Create self-play runner
    runner = SelfPlayRunner(
        vec_env=env,
        policy_p1=policy_p1,
        policy_p2=policy_p2,
    )

    if verbose:
        print(f"\n{'='*70}")
        print("Starting Self-Play with Inference Server")
        print(f"{'='*70}")
        print(f"Server URL: {server_url}")
        print(f"Batch size: {batch_size}")
        print(f"Target battles: {num_battles}")
        print(f"Format: {format_name}")
        print(f"Output: {save_dir}")
        print(f"{'='*70}\n")

    # Create output directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    format_dir = save_dir / format_name
    format_dir.mkdir(exist_ok=True)

    # Run self-play
    all_trajectories = []
    battles_completed = 0
    start_time = time.time()

    try:
        while battles_completed < num_battles:
            # Determine batch size for this iteration
            battles_remaining = num_battles - battles_completed
            current_batch = min(batch_size, battles_remaining)

            if verbose:
                print(f"Collecting {current_batch} battles...")

            # Collect trajectories
            trajectories = runner.collect_trajectories(
                num_battles=current_batch,
                max_steps_per_battle=500,
                verbose=False,
            )

            all_trajectories.extend(trajectories)
            battles_completed += len(trajectories)

            # Save periodically
            if len(all_trajectories) >= 128:
                if verbose:
                    print(f"  Saving {len(all_trajectories)} trajectories...")

                # Save trajectories to disk
                save_trajectories(
                    trajectories=all_trajectories,
                    output_dir=save_dir,
                    mappings=mappings,
                    battle_format=format_name,
                    verbose=False,
                )
                all_trajectories = []

            # Progress update
            if verbose and battles_completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = battles_completed / elapsed
                eta = (num_battles - battles_completed) / rate
                print(f"Progress: {battles_completed}/{num_battles} battles "
                      f"({100*battles_completed/num_battles:.1f}%) | "
                      f"Rate: {rate:.1f} battles/sec | "
                      f"ETA: {eta:.0f}s")

        # Save any remaining trajectories
        if all_trajectories:
            if verbose:
                print(f"Saving final {len(all_trajectories)} trajectories...")
            save_trajectories(
                trajectories=all_trajectories,
                output_dir=save_dir,
                mappings=mappings,
                battle_format=format_name,
                verbose=False,
            )
            all_trajectories = []

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n{'='*70}")
            print("Self-Play Complete!")
            print(f"{'='*70}")
            print(f"Battles completed: {battles_completed}/{num_battles}")
            print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            print(f"Average rate: {battles_completed/elapsed:.1f} battles/sec")
            print(f"Output directory: {save_dir}")
            print(f"{'='*70}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        elapsed = time.time() - start_time
        print(f"Completed {battles_completed} battles in {elapsed:.1f}s")

    except Exception as e:
        print(f"\nError during self-play: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play data using inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 1000 Gen1 OU battles
    %(prog)s --num_battles 1000 --format gen1ou --save_dir ~/selfplay_data

    # Use a specific server
    %(prog)s --num_battles 1000 --server_url http://192.168.1.100:8080

Notes:
    - Start the inference server first: python -m metamon.inference.server
    - The server handles all GPU operations
    - This script only runs PyKMN simulation (CPU)
    - No memory corruption issues!
        """
    )

    # Core arguments
    parser.add_argument(
        "--num_battles",
        type=int,
        required=True,
        help="Number of battles to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of parallel battles (default: 64)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (default: gen1ou)",
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="smogon_pass2",
        help="Team set to use (default: smogon_pass2)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save trajectories",
    )

    # Server arguments
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8080",
        help="Inference server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SyntheticRLV2",
        help="Model name for obs_space/reward_fn (default: SyntheticRLV2)",
    )

    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Minimal output",
    )

    args = parser.parse_args()

    # Run self-play
    run_selfplay_with_server(
        num_battles=args.num_battles,
        batch_size=args.batch_size,
        format_name=args.format,
        team_set=args.team_set,
        save_dir=Path(args.save_dir),
        server_url=args.server_url,
        model_name=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()