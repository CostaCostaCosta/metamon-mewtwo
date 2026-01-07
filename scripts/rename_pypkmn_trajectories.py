#!/usr/bin/env python3
"""
Rename pypkmn trajectories to match metamon training format.

Converts:
    {uuid}_pypkmn.json.lz4
To:
    metamon-{format}-{id}_Unrated_{p1}_vs_{p2}_{date}_{result}.json.lz4

This allows the training script to properly load pypkmn-generated trajectories.
"""

import argparse
import json
import lz4.frame
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def parse_trajectory_result(trajectory_path: Path) -> dict:
    """Load trajectory and extract metadata (winner, etc.)."""
    try:
        with lz4.frame.open(trajectory_path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))

        winner = data.get("winner", 1)  # 1 or 2
        return {"winner": winner, "valid": True}
    except Exception as e:
        print(f"Error reading {trajectory_path.name}: {e}")
        return {"valid": False}


def generate_new_filename(
    old_path: Path,
    battle_format: str,
    counter: int,
    winner: int,
) -> str:
    """
    Generate training-compatible filename.

    Format: metamon-{format}-{id}_Unrated_{p1}_vs_{p2}_{date}_{result}.json.lz4
    """
    battle_id = f"metamon-{battle_format}-{counter:06d}"
    rating = "Unrated"
    p1 = "PyKMNP1"  # No underscore!
    p2 = "PyKMNP2"  # No underscore!
    date = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    # Winner from P1's perspective
    result = "WIN" if winner == 1 else "LOSS"

    filename = f"{battle_id}_{rating}_{p1}_vs_{p2}_{date}_{result}.json.lz4"
    return filename


def rename_trajectories(
    input_dir: Path,
    output_dir: Path,
    battle_format: str = "gen1ou",
    dry_run: bool = False,
):
    """
    Rename all _pypkmn.json.lz4 files to training-compatible format.

    Args:
        input_dir: Source directory containing {format}/*.json.lz4 files
        output_dir: Destination directory for renamed files
        battle_format: Battle format (e.g., "gen1ou")
        dry_run: If True, print actions without executing
    """
    # Find source files
    source_format_dir = input_dir / battle_format
    if not source_format_dir.exists():
        print(f"❌ Source directory not found: {source_format_dir}")
        return

    pypkmn_files = sorted(source_format_dir.glob("*_pypkmn.json.lz4"))

    if not pypkmn_files:
        print(f"❌ No *_pypkmn.json.lz4 files found in {source_format_dir}")
        return

    print(f"Found {len(pypkmn_files):,} pypkmn trajectory files")

    # Create output directory
    output_format_dir = output_dir / battle_format
    if not dry_run:
        output_format_dir.mkdir(parents=True, exist_ok=True)

    # Rename files
    success_count = 0
    error_count = 0

    for i, old_path in enumerate(tqdm(pypkmn_files, desc="Renaming files")):
        # Parse trajectory to get result
        metadata = parse_trajectory_result(old_path)

        if not metadata["valid"]:
            error_count += 1
            continue

        # Generate new filename
        new_filename = generate_new_filename(
            old_path,
            battle_format,
            counter=i,
            winner=metadata["winner"],
        )
        new_path = output_format_dir / new_filename

        # Copy/rename file
        if dry_run:
            if i < 5:  # Show first 5 examples
                print(f"  {old_path.name} → {new_filename}")
        else:
            try:
                shutil.copy2(old_path, new_path)
                success_count += 1
            except Exception as e:
                print(f"Error copying {old_path.name}: {e}")
                error_count += 1

    # Summary
    print(f"\n{'=' * 70}")
    if dry_run:
        print("DRY RUN COMPLETE (no files copied)")
        print(f"Would rename {len(pypkmn_files):,} files")
    else:
        print("RENAMING COMPLETE")
        print(f"  ✓ Successfully renamed: {success_count:,}")
        print(f"  ✗ Errors: {error_count:,}")
        print(f"  Output directory: {output_format_dir}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename pypkmn trajectories to training-compatible format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory (e.g., ~/metamon/trajectories/kakuna-wrapper1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (e.g., ~/metamon/trajectories/kakuna-wrapper-formatted)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (default: gen1ou)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be renamed without copying files",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    print("=" * 70)
    print("PYPKMN TRAJECTORY RENAMING")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Battle format: {args.format}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'COPY & RENAME'}")
    print("=" * 70)
    print()

    rename_trajectories(
        input_dir=input_dir,
        output_dir=output_dir,
        battle_format=args.format,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
