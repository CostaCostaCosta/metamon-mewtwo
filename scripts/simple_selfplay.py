#!/usr/bin/env python3
"""
Simple self-play data collection using ladder matching.

Launches N instances of the same model that automatically match against
each other on the local ladder. Much simpler than the league-based scripts.

Usage:
    # 2 players (default, slower)
    python scripts/simple_selfplay.py --model SyntheticRLV2 --num_battles 50 --output_dir ~/selfplay_test

    # 100 players (faster, parallel battles)
    python scripts/simple_selfplay.py --model SyntheticRLV2 --num_battles 50 --num_players 100 --output_dir ~/selfplay_test
"""

import os
import sys
import time
import glob
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Simple self-play data collection via ladder matching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SyntheticRLV2",
        help="Model name (must be in pretrained registry)",
    )
    parser.add_argument(
        "--num_battles",
        type=int,
        default=50,
        help="Total number of battles to generate",
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=2,
        help="Number of concurrent players on ladder (more = faster, less blocking)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trajectories and results",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (e.g., gen1ou)",
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="modern_replays_v2",
        help="Team set to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint number to load (default: model default)",
    )

    args = parser.parse_args()

    # Parse format into gen and tier
    if args.format.startswith("gen"):
        gen = args.format[3]  # Extract generation number
        tier = args.format[4:]  # Extract tier (e.g., "ou")
    else:
        print(f"Error: Format must start with 'gen' (e.g., gen1ou), got: {args.format}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    team_dir = output_dir / "team_results"

    # Calculate battles per player to achieve target total
    # Each battle involves 2 players, so we need num_battles * 2 total player-battles
    battles_per_player = max(1, (args.num_battles * 2 + args.num_players - 1) // args.num_players)
    expected_total = (args.num_players * battles_per_player) // 2

    print("=" * 50)
    print("Simple Self-Play Data Collection")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'default'}")
    print(f"Players: {args.num_players}")
    print(f"Battles per player: {battles_per_player}")
    print(f"Expected total battles: ~{expected_total}")
    print(f"Format: {args.format}")
    print(f"Team set: {args.team_set}")
    print(f"Output: {output_dir}")
    print("=" * 50)
    print()

    # Base command for all players
    base_cmd = [
        sys.executable,
        "-m",
        "metamon.rl.evaluate",
        "--agent",
        args.model,
        "--eval_type",
        "ladder",
        "--gens",
        gen,
        "--formats",
        tier,
        "--team_set",
        args.team_set,
        "--total_battles",
        str(battles_per_player),
        "--save_trajectories_to",
        str(traj_dir),
        "--save_team_results_to",
        str(team_dir),
    ]

    if args.checkpoint is not None:
        base_cmd.extend(["--checkpoints", str(args.checkpoint)])

    # Launch all players
    session_id = os.getpid()
    processes = []
    log_files = []

    print(f"Launching {args.num_players} players...")
    for i in range(args.num_players):
        player_cmd = base_cmd + ["--username", f"SP{session_id}P{i+1}"]
        log_path = output_dir / f"player{i+1}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)

        proc = subprocess.Popen(player_cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append(proc)

        if (i + 1) % 10 == 0:
            print(f"  Launched {i+1}/{args.num_players} players...")

        # Small delay to stagger connections
        time.sleep(0.1)

    print(f"✓ All {args.num_players} players launched!")
    print()
    print("Monitor progress with:")
    print(f"  tail -f {output_dir / 'player1.log'}")
    print()

    # Wait for all processes with progress monitoring
    try:
        traj_path = traj_dir / args.format
        traj_path.mkdir(parents=True, exist_ok=True)

        # Monitor progress while processes are running
        last_count = 0
        while any(proc.poll() is None for proc in processes):
            # Count trajectory files (2 per battle)
            trajectory_files = glob.glob(str(traj_path / "*.json.lz4"))
            battles_completed = len(trajectory_files) // 2

            if battles_completed != last_count:
                last_count = battles_completed
                progress = min(100, int(battles_completed / expected_total * 100))
                bar_length = 40
                filled = int(bar_length * progress / 100)
                bar = '=' * filled + '>' * (1 if progress < 100 else 0) + ' ' * (bar_length - filled - (1 if progress < 100 else 0))

                print(f"\r[{bar}] {battles_completed}/{expected_total} battles ({progress}%)", end='', flush=True)

            time.sleep(2)

        print()  # New line after progress bar

        # Collect exit codes
        exit_codes = [proc.wait() for proc in processes]

    except KeyboardInterrupt:
        print("\n\nInterrupted! Terminating all players...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        for log_file in log_files:
            log_file.close()
        sys.exit(1)

    # Close all log files
    for log_file in log_files:
        log_file.close()

    print()
    print("=" * 50)
    num_success = sum(1 for code in exit_codes if code == 0)

    if num_success == args.num_players:
        print("✓ Self-play complete!")
    elif num_success > 0:
        print(f"⚠ Partial completion: {num_success}/{args.num_players} players succeeded")
    else:
        print("✗ Self-play failed - all players exited with errors!")
        print("Check logs for details.")
        sys.exit(1)

    print(f"Trajectories: {traj_dir / args.format}/")
    print(f"Team results: {team_dir}/")
    print()
    print("Next step - Train on this data:")
    print(f"  python -m metamon.rl.finetune_from_hf \\")
    print(f"    --finetune_from_model {args.model} \\")
    print(f"    --custom_replay_dir {traj_dir} \\")
    print(f"    --formats {args.format} \\")
    print(f"    --train_gin_config vanilla_selfplay_damped.gin \\")
    print(f"    --epochs 10 \\")
    print(f"    --save_dir {output_dir / 'checkpoint'}")


if __name__ == "__main__":
    main()
