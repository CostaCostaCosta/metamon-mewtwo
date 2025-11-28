#!/usr/bin/env python3
"""
Filter self-play data for quality control.

Removes low-quality battles (too many invalid actions, too short, etc.)
and balances win/loss distribution for better training.

Usage:
    python scripts/filter_selfplay_data.py \
        --input_dir ~/gen1_selfplay_data/v0/gen1ou \
        --output_dir ~/gen1_selfplay_data/v0_filtered/gen1ou \
        --max_invalid_rate 0.05 \
        --min_turns 10 \
        --balance_outcomes
"""

import os
import sys
import json
import argparse
import lz4.frame
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trajectory(filepath: str) -> Dict:
    """Load a trajectory from lz4-compressed JSON file."""
    with lz4.frame.open(filepath, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
    return data


def save_trajectory(filepath: str, data: Dict):
    """Save a trajectory to lz4-compressed JSON file."""
    # Conservative file writing to avoid partial writes
    temp_path = filepath + ".tmp"
    with lz4.frame.open(temp_path, "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))
    os.rename(temp_path, filepath)


def count_invalid_actions(trajectory: Dict) -> Tuple[int, int]:
    """
    Count invalid actions in a trajectory.

    Returns (invalid_count, total_count)
    """
    actions = trajectory.get("actions", [])
    # -1 indicates invalid or placeholder action
    invalid = sum(1 for a in actions if a == -1)
    total = len(actions)
    return invalid, total


def get_turn_count(trajectory: Dict) -> int:
    """Get number of turns in a battle."""
    return len(trajectory.get("states", [])) - 1  # First state is initial


def get_outcome(filepath: str) -> str:
    """Extract outcome (WIN/LOSS) from filename."""
    filename = Path(filepath).name
    if "WIN" in filename:
        return "WIN"
    elif "LOSS" in filename:
        return "LOSS"
    else:
        return "UNKNOWN"


def filter_trajectory(
    filepath: str,
    max_invalid_rate: float = 0.05,
    min_turns: int = 10,
    max_turns: int = 1000,
) -> Tuple[bool, str]:
    """
    Check if trajectory passes quality filters.

    Returns (passed, reason)
    """
    try:
        traj = load_trajectory(filepath)
    except Exception as e:
        return False, f"load_error: {e}"

    # Check turn count
    turns = get_turn_count(traj)
    if turns < min_turns:
        return False, f"too_short: {turns} turns"
    if turns > max_turns:
        return False, f"too_long: {turns} turns"

    # Check invalid action rate
    invalid, total = count_invalid_actions(traj)
    if total == 0:
        return False, "no_actions"

    invalid_rate = invalid / total
    if invalid_rate > max_invalid_rate:
        return False, f"invalid_actions: {invalid_rate:.1%}"

    return True, "passed"


def balance_outcomes(
    filepaths: List[str], target_balance: float = 0.5
) -> List[str]:
    """
    Balance win/loss outcomes by randomly removing excess battles.

    Args:
        filepaths: List of trajectory filepaths
        target_balance: Target win rate (0.5 = equal wins/losses)

    Returns:
        Filtered list of filepaths with balanced outcomes
    """
    outcomes = Counter([get_outcome(fp) for fp in filepaths])

    print(f"Outcome distribution before balancing:")
    print(f"  Wins: {outcomes['WIN']}")
    print(f"  Losses: {outcomes['LOSS']}")
    print(f"  Unknown: {outcomes['UNKNOWN']}")

    wins = [fp for fp in filepaths if get_outcome(fp) == "WIN"]
    losses = [fp for fp in filepaths if get_outcome(fp) == "LOSS"]
    unknown = [fp for fp in filepaths if get_outcome(fp) == "UNKNOWN"]

    # Balance to have equal wins and losses
    target_count = min(len(wins), len(losses))

    # Randomly sample if needed
    if len(wins) > target_count:
        wins = random.sample(wins, target_count)
    if len(losses) > target_count:
        losses = random.sample(losses, target_count)

    balanced = wins + losses + unknown

    print(f"\nOutcome distribution after balancing:")
    print(f"  Wins: {len(wins)}")
    print(f"  Losses: {len(losses)}")
    print(f"  Unknown: {len(unknown)}")
    print(f"  Total: {len(balanced)} (removed {len(filepaths) - len(balanced)})")

    return balanced


def filter_dataset(
    input_dir: str,
    output_dir: str,
    max_invalid_rate: float = 0.05,
    min_turns: int = 10,
    max_turns: int = 1000,
    balance: bool = False,
):
    """Filter entire dataset directory."""

    print(f"\n{'='*80}")
    print(f"SELF-PLAY DATA FILTERING")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Max invalid rate: {max_invalid_rate:.1%}")
    print(f"Min turns: {min_turns}")
    print(f"Max turns: {max_turns}")
    print(f"Balance outcomes: {balance}")
    print(f"{'='*80}\n")

    # Find all trajectory files
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    traj_files = list(input_path.glob("*.json.lz4"))
    print(f"Found {len(traj_files)} trajectory files")

    if len(traj_files) == 0:
        print("No files to filter!")
        return

    # Filter trajectories
    print("\nFiltering trajectories...")
    passed = []
    failed_reasons = Counter()

    for i, filepath in enumerate(traj_files):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(traj_files)}")

        is_valid, reason = filter_trajectory(
            str(filepath), max_invalid_rate, min_turns, max_turns
        )

        if is_valid:
            passed.append(str(filepath))
        else:
            failed_reasons[reason] += 1

    print(f"\n✓ Filtering complete: {len(passed)}/{len(traj_files)} passed")

    # Print failure reasons
    if failed_reasons:
        print("\nFailure reasons:")
        for reason, count in failed_reasons.most_common():
            print(f"  {reason}: {count} ({count/len(traj_files)*100:.1f}%)")

    # Balance outcomes if requested
    if balance and len(passed) > 0:
        print("\nBalancing outcomes...")
        passed = balance_outcomes(passed)

    # Copy passed files to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {len(passed)} files to {output_dir}...")

    for i, filepath in enumerate(passed):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(passed)}")

        filename = Path(filepath).name
        output_file = output_path / filename

        # Load and save (could also just copy, but this verifies integrity)
        try:
            traj = load_trajectory(filepath)
            save_trajectory(str(output_file), traj)
        except Exception as e:
            print(f"Warning: Failed to copy {filename}: {e}")

    print(f"\n{'='*80}")
    print(f"FILTERING COMPLETE")
    print(f"Input: {len(traj_files)} files")
    print(f"Filtered: {len(passed)} files ({len(passed)/len(traj_files)*100:.1f}%)")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Save filtering statistics
    stats = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "input_count": len(traj_files),
        "output_count": len(passed),
        "filter_rate": len(passed) / len(traj_files),
        "filters": {
            "max_invalid_rate": max_invalid_rate,
            "min_turns": min_turns,
            "max_turns": max_turns,
            "balance_outcomes": balance,
        },
        "failed_reasons": dict(failed_reasons),
    }

    stats_file = output_path / "filtering_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Filtering statistics saved to: {stats_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Filter self-play trajectory data for quality"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with trajectory files (*.json.lz4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for filtered files",
    )
    parser.add_argument(
        "--max_invalid_rate",
        type=float,
        default=0.05,
        help="Maximum invalid action rate (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--min_turns",
        type=int,
        default=10,
        help="Minimum number of turns (default: 10)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=1000,
        help="Maximum number of turns (default: 1000)",
    )
    parser.add_argument(
        "--balance_outcomes",
        action="store_true",
        help="Balance win/loss outcomes to 50/50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Expand home directories
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    # Filter dataset
    filter_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        max_invalid_rate=args.max_invalid_rate,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        balance=args.balance_outcomes,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
