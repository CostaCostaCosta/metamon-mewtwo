#!/usr/bin/env python3
"""
Collect vanilla self-play training data.

This script runs self-play matches where a checkpoint plays against itself,
collecting trajectories for offline RL training.

Usage:
    python -m metamon.rl.collect_selfplay_data \
        --run_name SelfPlay_Iteration_0 \
        --checkpoint_path ~/checkpoints/synrl_v2 \
        --save_dir ~/selfplay_data/iter_0 \
        --battle_format gen1ou \
        --team_set modern_replays_v2 \
        --num_battles 1000
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def launch_agent_on_ladder(
    checkpoint_path: str,
    battle_format: str,
    team_set_name: str,
    username: str,
    num_battles: int,
    output_dir: str,
    model_class: str = "SyntheticRLV2",
) -> subprocess.Popen:
    """Launch an agent process on the local ladder.

    Args:
        checkpoint_path: Path to model checkpoint
        battle_format: Pokemon format (e.g., gen1ou)
        team_set_name: Team set to use
        username: Username for ladder
        num_battles: Number of battles to play
        output_dir: Where to save battle replays
        model_class: Model class name for loading

    Returns:
        subprocess.Popen: Running process
    """
    matchup_script = Path(__file__).parent.parent.parent / "scripts" / "run_matchup.py"

    cmd = [
        sys.executable,
        str(matchup_script),
        "--model", model_class,
        "--format", battle_format,
        "--team_set", team_set_name,
        "--username", username,
        "--battles", str(num_battles),
        "--output_dir", output_dir,
    ]

    # Add checkpoint path if provided
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def main():
    parser = ArgumentParser(description="Collect vanilla self-play training data")
    parser.add_argument("--run_name", type=str, required=True, help="Name for this run")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint to use for self-play"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save collected trajectories"
    )
    parser.add_argument(
        "--battle_format",
        type=str,
        default="gen1ou",
        help="Pokemon battle format"
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="modern_replays_v2",
        help="Team set to use for battles"
    )
    parser.add_argument(
        "--num_battles",
        type=int,
        default=1000,
        help="Total number of battles to collect"
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="SyntheticRLV2",
        help="Model class name for loading checkpoint"
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*80)
    print("VANILLA SELF-PLAY DATA COLLECTION")
    print("="*80)
    print(f"Run name: {args.run_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Collecting {args.num_battles} self-play battles")
    print(f"Save dir: {args.save_dir}")
    print(f"Format: {args.battle_format}")
    print(f"Team set: {args.team_set}")
    print("="*80 + "\n")

    # For self-play, launch two copies of the same agent
    timestamp = int(time.time())

    # Player 1
    player1_username = f"SP1_{args.run_name[:8]}_{timestamp%10000}"
    player1_output = os.path.join(args.save_dir, "player1")
    os.makedirs(player1_output, exist_ok=True)

    # Player 2
    player2_username = f"SP2_{args.run_name[:8]}_{timestamp%10000}"
    player2_output = os.path.join(args.save_dir, "player2")
    os.makedirs(player2_output, exist_ok=True)

    print(f"Launching Player 1: {player1_username}")
    player1_proc = launch_agent_on_ladder(
        checkpoint_path=args.checkpoint_path,
        battle_format=args.battle_format,
        team_set_name=args.team_set,
        username=player1_username,
        num_battles=args.num_battles,
        output_dir=player1_output,
        model_class=args.model_class,
    )

    time.sleep(3)  # Give Player 1 time to connect to ladder

    print(f"Launching Player 2: {player2_username}")
    player2_proc = launch_agent_on_ladder(
        checkpoint_path=args.checkpoint_path,
        battle_format=args.battle_format,
        team_set_name=args.team_set,
        username=player2_username,
        num_battles=args.num_battles,
        output_dir=player2_output,
        model_class=args.model_class,
    )

    print("\nBoth players launched. Waiting for battles to complete...")
    print(f"Player 1 saving to: {player1_output}")
    print(f"Player 2 saving to: {player2_output}")
    print("\nPress Ctrl+C to interrupt and terminate processes.\n")

    try:
        # Wait for both processes to complete
        player1_returncode = player1_proc.wait()
        player2_returncode = player2_proc.wait()

        print(f"\nPlayer 1 finished with return code: {player1_returncode}")
        print(f"Player 2 finished with return code: {player2_returncode}")

        if player1_returncode == 0 and player2_returncode == 0:
            print("\n" + "="*80)
            print("✓ Self-play data collection completed successfully!")
            print("="*80)
            print(f"Trajectories saved to: {args.save_dir}")
            print(f"  - Player 1: {player1_output}")
            print(f"  - Player 2: {player2_output}")
            print("\nNext steps:")
            print(f"  1. Train on this data using metamon.rl.train")
            print(f"  2. Or run iterative self-play with train_vanilla_selfplay.py")
            print("="*80 + "\n")
        else:
            print("\n⚠ Warning: One or both processes exited with errors.")
            print("Check the process outputs for details.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating processes...")
        player1_proc.send_signal(signal.SIGINT)
        player2_proc.send_signal(signal.SIGINT)
        player1_proc.wait(timeout=10)
        player2_proc.wait(timeout=10)
        print("Processes terminated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
