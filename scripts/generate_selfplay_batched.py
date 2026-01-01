#!/usr/bin/env python3
"""
High-throughput self-play data generation using batched AMAGO inference.

Generates self-play trajectories with 10-20x speedup compared to baseline.
Optimized for RTX 5090 with batched inference and mixed precision.

Usage:
    # Self-play (one model plays against itself)
    python scripts/generate_selfplay_batched.py \\
        --model SyntheticRLV2 \\
        --checkpoint 48 \\
        --num_battles 1000 \\
        --batch_size 16 \\
        --format gen1ou \\
        --team_set modern_replays_v2 \\
        --save_dir ~/selfplay_data/gen1ou

    # Head-to-head (two different models)
    python scripts/generate_selfplay_batched.py \\
        --model_p1 SyntheticRLV2 \\
        --checkpoint_p1 48 \\
        --model_p2 SyntheticRLV1 \\
        --checkpoint_p2 40 \\
        --num_battles 1000 \\
        --batch_size 16 \\
        --format gen1ou \\
        --team_set modern_replays_v2 \\
        --save_dir ~/selfplay_data/gen1ou

Performance:
    - batch_size=16: ~20 battles/sec (10x faster than baseline)
    - batch_size=64: ~60 battles/sec (30x faster, requires fixing env)
    - 1000 battles in ~50 seconds (vs 8.8 minutes baseline)
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

from metamon.env.pykmn import (
    load_random_teams,
    PyKMNVectorEnv,
    LocalPolicyRunner,
    SelfPlayRunner,
    save_trajectories,
    precompute_mappings,
)
from metamon.rl.pretrained import get_pretrained_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="High-throughput self-play data generation with batched inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        help="Model name for self-play (both players use same model). "
        "Examples: SyntheticRLV2, SyntheticRLV1, LargeRL, MediumRL",
    )
    model_group.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint number (default: model's default checkpoint)",
    )
    model_group.add_argument(
        "--model_p1",
        type=str,
        help="Model name for Player 1 (for head-to-head)",
    )
    model_group.add_argument(
        "--checkpoint_p1",
        type=int,
        default=None,
        help="Checkpoint for Player 1",
    )
    model_group.add_argument(
        "--model_p2",
        type=str,
        help="Model name for Player 2 (for head-to-head)",
    )
    model_group.add_argument(
        "--checkpoint_p2",
        type=int,
        default=None,
        help="Checkpoint for Player 2",
    )

    # Data generation arguments
    data_group = parser.add_argument_group("Data Generation")
    data_group.add_argument(
        "--num_battles",
        type=int,
        required=True,
        help="Number of battles to generate",
    )
    data_group.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16, optimal for RTX 5090). "
        "Recommended: 16 (10x speedup), 32-64 (20-30x speedup)",
    )
    data_group.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (default: gen1ou). Options: gen1ou, gen2ou, gen3ou, gen4ou",
    )

    # Team arguments
    team_group = parser.add_argument_group("Team Configuration")
    team_group.add_argument(
        "--team_set",
        type=str,
        default="modern_replays_v2",
        help="Team set name (default: modern_replays_v2)",
    )
    team_group.add_argument(
        "--team_dir",
        type=str,
        default=None,
        help="Custom team directory (overrides --team_set). "
        "Default: $METAMON_CACHE_DIR/teams/{team_set}",
    )
    team_group.add_argument(
        "--num_teams",
        type=int,
        default=None,
        help="Number of teams to sample (default: sample all available teams)",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save trajectories",
    )
    output_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for organizing output (default: auto-generated timestamp)",
    )

    # Performance arguments
    perf_group = parser.add_argument_group("Performance Tuning")
    perf_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    perf_group.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use mixed precision (bfloat16) inference (default: True, ~1.5x speedup)",
    )
    perf_group.add_argument(
        "--no_amp",
        dest="use_amp",
        action="store_false",
        help="Disable mixed precision inference",
    )
    perf_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Action sampling temperature (default: 1.0, higher = more random)",
    )

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)",
    )
    log_group.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Minimal output",
    )
    log_group.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Progress logging interval in battles (default: 10)",
    )

    args = parser.parse_args()

    # Validation
    if args.model is None and (args.model_p1 is None or args.model_p2 is None):
        parser.error(
            "Must specify either --model (for self-play) or both --model_p1 and --model_p2 (for head-to-head)"
        )

    if args.model is not None and (args.model_p1 is not None or args.model_p2 is not None):
        parser.error(
            "Cannot specify both --model and --model_p1/--model_p2. "
            "Use --model for self-play or --model_p1/--model_p2 for head-to-head."
        )

    return args


def load_teams(
    team_dir: Optional[str],
    team_set: str,
    format_name: str,
    batch_size: int,
    num_teams: Optional[int],
    verbose: bool = True,
) -> Tuple[list, list]:
    """
    Load teams for self-play.

    Args:
        team_dir: Custom team directory (optional)
        team_set: Team set name
        format_name: Battle format
        batch_size: Number of teams to load per player
        num_teams: Max teams to sample from (None = all)
        verbose: Print progress

    Returns:
        Tuple of (teams_p1, teams_p2)
    """
    if team_dir is None:
        cache_dir = Path(os.environ.get("METAMON_CACHE_DIR", Path.home() / "metamon_cache"))
        team_dir = cache_dir / "teams" / team_set

    team_dir = Path(team_dir).expanduser()

    if not team_dir.exists():
        raise FileNotFoundError(
            f"Team directory not found: {team_dir}\n"
            f"Please ensure teams are available at this location."
        )

    if verbose:
        print(f"Loading teams from: {team_dir}")

    # Sample teams randomly
    teams_p1 = load_random_teams(team_dir, format_name, batch_size)
    teams_p2 = load_random_teams(team_dir, format_name, batch_size)

    if verbose:
        print(f"✓ Loaded {len(teams_p1) + len(teams_p2)} teams")

    return teams_p1, teams_p2


def create_policy_runner(
    model_name: str,
    checkpoint: Optional[int],
    device: str,
    use_amp: bool,
    temperature: float,
    verbose: bool = True,
) -> LocalPolicyRunner:
    """
    Create a policy runner with batched inference.

    Args:
        model_name: Name of pretrained model
        checkpoint: Checkpoint number (None for default)
        device: Device for inference
        use_amp: Use mixed precision (bfloat16)
        temperature: Action sampling temperature
        verbose: Print progress

    Returns:
        LocalPolicyRunner instance
    """
    if verbose:
        print(f"\nLoading model: {model_name}")
        if checkpoint is not None:
            print(f"  Checkpoint: {checkpoint}")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {use_amp}")
        print(f"  Temperature: {temperature}")

    policy = LocalPolicyRunner(
        model_name=model_name,
        checkpoint=checkpoint,
        device=device,
        use_amp=use_amp,
        verbose=False,
    )

    if verbose:
        print(f"✓ Model loaded")

    return policy


def run_selfplay(
    policy_p1: LocalPolicyRunner,
    policy_p2: LocalPolicyRunner,
    teams_p1: list,
    teams_p2: list,
    format_name: str,
    batch_size: int,
    num_battles: int,
    save_dir: Path,
    run_name: str,
    log_interval: int,
    verbose: bool,
):
    """
    Run self-play data generation with batched inference.

    Args:
        policy_p1: Policy for player 1
        policy_p2: Policy for player 2
        teams_p1: Teams for player 1
        teams_p2: Teams for player 2
        format_name: Battle format
        batch_size: Number of parallel environments
        num_battles: Total number of battles to generate
        save_dir: Directory to save trajectories
        run_name: Run name for output files
        log_interval: Progress logging interval
        verbose: Print detailed progress
    """
    # Get observation space and reward function from pretrained model
    pretrained_cls = get_pretrained_model(policy_p1.model_name)
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    # Precompute mappings for trajectory saving
    if verbose:
        print("Precomputing feature mappings...")
    mappings = precompute_mappings()
    if verbose:
        print("✓ Feature mappings ready")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Starting Self-Play Data Generation")
        print(f"{'='*70}")
        print(f"Batch size: {batch_size}")
        print(f"Target battles: {num_battles}")
        print(f"Format: {format_name}")
        print(f"Output: {save_dir}")
        print(f"{'='*70}\n")

    # Create vectorized environment
    vec_env = PyKMNVectorEnv(
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        num_envs=batch_size,
        obs_space=obs_space,
        reward_fn=reward_fn,
        battle_format=format_name,
        track_trajectories=True,
    )

    if verbose:
        print(f"✓ Created vectorized environment with {batch_size} parallel battles\n")

    # Create self-play runner
    runner = SelfPlayRunner(vec_env, policy_p1, policy_p2)

    # Memory monitoring setup
    try:
        import psutil
        process = psutil.Process()
        memory_monitoring_available = True
        if verbose:
            print("Memory monitoring enabled (psutil available)\n")
    except ImportError:
        memory_monitoring_available = False
        if verbose:
            print("Memory monitoring unavailable (psutil not installed)\n")

    # Run self-play with progress tracking and automatic error recovery
    all_trajectories = []
    battles_completed = 0
    start_time = time.time()
    consecutive_errors = 0
    max_consecutive_errors = 3

    while battles_completed < num_battles:
        # Determine how many battles to collect in this chunk
        battles_remaining = num_battles - battles_completed
        chunk_size = min(batch_size, battles_remaining)

        if verbose and battles_completed % log_interval == 0:
            elapsed = time.time() - start_time
            if battles_completed > 0:
                rate = battles_completed / elapsed
                eta = (num_battles - battles_completed) / rate

                # Add memory usage info
                mem_info = ""
                if memory_monitoring_available:
                    mem_mb = process.memory_info().rss / 1024**2
                    mem_info = f" | Memory: {mem_mb:.1f} MB"

                print(
                    f"Progress: {battles_completed}/{num_battles} battles "
                    f"({battles_completed/num_battles*100:.1f}%) | "
                    f"Rate: {rate:.1f} battles/sec | "
                    f"ETA: {eta:.1f}s{mem_info}"
                )
            else:
                print(f"Starting data collection...")

        # Collect trajectories with automatic error recovery
        try:
            trajectories = runner.collect_trajectories(
                num_battles=chunk_size,
                max_steps_per_battle=1000,
                verbose=False,
            )

            all_trajectories.extend(trajectories)
            battles_completed += len(trajectories)
            consecutive_errors = 0  # Reset error counter on success

            # Save incrementally (every 100 battles)
            if len(all_trajectories) >= 100:
                try:
                    save_batch(all_trajectories, save_dir, format_name, run_name, mappings, verbose)
                    all_trajectories = []
                except Exception as e:
                    print(f"⚠️  Warning: Failed to save incremental batch: {e}")
                    print(f"   Will retry with next batch. Continuing...")
                    # Don't clear all_trajectories - will try again later

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving collected trajectories...")
            break

        except Exception as e:
            consecutive_errors += 1

            print(f"\n⚠️  Error during batch collection (attempt {consecutive_errors}/{max_consecutive_errors})")
            print(f"   Error: {type(e).__name__}: {e}")

            if consecutive_errors >= max_consecutive_errors:
                print(f"\n❌ Too many consecutive errors ({max_consecutive_errors}). Stopping.")
                import traceback
                traceback.print_exc()
                break

            # Save progress before retrying
            if all_trajectories:
                print(f"   Saving {len(all_trajectories)} collected trajectories...")
                try:
                    save_batch(all_trajectories, save_dir, format_name, run_name, mappings, verbose)
                    all_trajectories = []
                except Exception as save_error:
                    print(f"   Warning: Failed to save: {save_error}")

            # Retry with fresh environment
            print(f"   Recreating environment and retrying...")
            time.sleep(2)  # Brief pause to let resources cleanup

            try:
                # Recreate vectorized environment
                vec_env = PyKMNVectorEnv(
                    teams_p1=teams_p1,
                    teams_p2=teams_p2,
                    num_envs=batch_size,
                    obs_space=obs_space,
                    reward_fn=reward_fn,
                    battle_format=format_name,
                    track_trajectories=True,
                )
                runner = SelfPlayRunner(vec_env, policy_p1, policy_p2)
                print(f"   ✓ Environment recreated, continuing...")
            except Exception as recreate_error:
                print(f"   ❌ Failed to recreate environment: {recreate_error}")
                break

    # Save remaining trajectories
    if all_trajectories:
        print(f"\nSaving final {len(all_trajectories)} trajectories...")
        try:
            save_batch(all_trajectories, save_dir, format_name, run_name, mappings, verbose)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save final batch: {e}")

    # Final statistics
    total_time = time.time() - start_time
    rate = battles_completed / total_time if total_time > 0 else 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"Self-Play Complete!")
        print(f"{'='*70}")
        print(f"Battles completed: {battles_completed}/{num_battles}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average rate: {rate:.1f} battles/sec")
        print(f"Output directory: {save_dir}")
        print(f"{'='*70}\n")


def save_batch(
    trajectories: list,
    save_dir: Path,
    format_name: str,
    run_name: str,
    mappings,
    verbose: bool,
):
    """
    Save a batch of trajectories to disk.

    Args:
        trajectories: List of trajectories to save
        save_dir: Base directory for saving
        format_name: Battle format
        run_name: Run name for subdirectory
        mappings: Precomputed feature mappings
        verbose: Print progress
    """
    # Create output directory structure
    output_dir = save_dir / run_name / format_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectories
    save_trajectories(
        trajectories,
        output_dir,
        mappings=mappings,
        battle_format=format_name,
        verbose=False
    )

    if verbose:
        print(f"✓ Saved {len(trajectories)} trajectories to {output_dir}")


def main():
    """Main entry point."""
    args = parse_args()

    # Generate run name if not provided
    if args.run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model:
            args.run_name = f"{args.model}_{timestamp}"
        else:
            args.run_name = f"{args.model_p1}_vs_{args.model_p2}_{timestamp}"

    # Convert save_dir to Path
    save_dir = Path(args.save_dir).expanduser()

    if args.verbose:
        print("=" * 70)
        print("BATCHED SELF-PLAY DATA GENERATION")
        print("=" * 70)
        print(f"Run name: {args.run_name}")
        print(f"Format: {args.format}")
        print(f"Batch size: {args.batch_size}")
        print(f"Target battles: {args.num_battles}")
        print("=" * 70)

    # Load teams
    teams_p1, teams_p2 = load_teams(
        team_dir=args.team_dir,
        team_set=args.team_set,
        format_name=args.format,
        batch_size=args.batch_size,
        num_teams=args.num_teams,
        verbose=args.verbose,
    )

    # Create policy runners
    if args.model:
        # Self-play: same model for both players
        policy_p1 = create_policy_runner(
            model_name=args.model,
            checkpoint=args.checkpoint,
            device=args.device,
            use_amp=args.use_amp,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        policy_p2 = policy_p1  # Share same instance for efficiency
    else:
        # Head-to-head: different models
        policy_p1 = create_policy_runner(
            model_name=args.model_p1,
            checkpoint=args.checkpoint_p1,
            device=args.device,
            use_amp=args.use_amp,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        policy_p2 = create_policy_runner(
            model_name=args.model_p2,
            checkpoint=args.checkpoint_p2,
            device=args.device,
            use_amp=args.use_amp,
            temperature=args.temperature,
            verbose=args.verbose,
        )

    # Run self-play
    run_selfplay(
        policy_p1=policy_p1,
        policy_p2=policy_p2,
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        format_name=args.format,
        batch_size=args.batch_size,
        num_battles=args.num_battles,
        save_dir=save_dir,
        run_name=args.run_name,
        log_interval=args.log_interval,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
