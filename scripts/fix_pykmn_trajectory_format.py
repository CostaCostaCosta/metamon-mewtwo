#!/usr/bin/env python3
"""
Fix PyKMN trajectory format by removing the duplicate final action.

PyKMN-generated trajectories incorrectly have len(actions) = len(states) + 1
due to a bug in trajectory_saver.py. This script fixes existing trajectories
to match the expected format: len(actions) = len(states).

Usage:
    python scripts/fix_pykmn_trajectory_format.py \
        --input_dir ~/metamon/trajectories/kakuna-loop1 \
        --output_dir ~/metamon/trajectories/kakuna-loop1-fixed \
        --dry_run

    python scripts/fix_pykmn_trajectory_format.py \
        --input_dir ~/metamon/trajectories/kakuna-loop1 \
        --output_dir ~/metamon/trajectories/kakuna-loop1-fixed

Options:
    --input_dir: Directory containing buggy trajectory files
    --output_dir: Directory to save fixed trajectories (can be same as input_dir for in-place)
    --dry_run: Preview changes without writing files
    --verbose: Print detailed progress
"""

import argparse
import json
import lz4.frame
import shutil
from pathlib import Path
from tqdm import tqdm


def fix_trajectory_file(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """
    Fix a single trajectory file by removing the duplicate final action.

    Args:
        input_path: Path to input .json.lz4 file
        output_path: Path to output .json.lz4 file
        dry_run: If True, don't write output file

    Returns:
        Dictionary with fix statistics
    """
    # Load compressed trajectory
    with lz4.frame.open(input_path, 'rb') as f:
        data = json.load(f)

    # Check format
    states_len = len(data['states'])
    actions_len = len(data['actions'])

    stats = {
        'processed': True,
        'needed_fix': False,
        'states_len': states_len,
        'actions_len_before': actions_len,
        'actions_len_after': actions_len,
    }

    # Check if fix is needed
    if actions_len == states_len + 1:
        # Remove duplicate final action
        data['actions'] = data['actions'][:-1]
        stats['needed_fix'] = True
        stats['actions_len_after'] = len(data['actions'])
    elif actions_len != states_len:
        # Unexpected format
        stats['processed'] = False
        stats['error'] = f"Unexpected format: {states_len} states, {actions_len} actions"
        return stats

    # Write fixed trajectory
    if not dry_run and stats['needed_fix']:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(data)
        compressed = lz4.frame.compress(json_str.encode('utf-8'))

        with open(output_path, 'wb') as f:
            f.write(compressed)
    elif not dry_run and not stats['needed_fix']:
        # Copy unchanged file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix PyKMN trajectory format by removing duplicate final action"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing buggy trajectory files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save fixed trajectories (can be same as input_dir)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    print("=" * 70)
    print("PYKMN TRAJECTORY FORMAT FIXER")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 70)
    print()

    # Find all trajectory files (supports both flat and format-subdirectory structure)
    trajectory_files = []

    # Check for format subdirectories (e.g., gen1ou/)
    format_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if format_dirs:
        print(f"Found {len(format_dirs)} format subdirectories")
        for format_dir in format_dirs:
            files = list(format_dir.glob("*.json.lz4"))
            trajectory_files.extend([(f, format_dir.name) for f in files])
            print(f"  {format_dir.name}: {len(files)} files")
    else:
        # Flat directory structure
        files = list(input_dir.glob("*.json.lz4"))
        trajectory_files.extend([(f, None) for f in files])
        print(f"Found {len(files)} trajectory files in flat directory")

    if not trajectory_files:
        print(f"\nNo trajectory files found in {input_dir}")
        return

    print(f"\nTotal files to process: {len(trajectory_files)}")
    print()

    # Process files
    total_fixed = 0
    total_unchanged = 0
    total_errors = 0

    for input_path, format_subdir in tqdm(trajectory_files, desc="Processing files"):
        # Determine output path (preserve directory structure)
        if format_subdir:
            output_path = output_dir / format_subdir / input_path.name
        else:
            output_path = output_dir / input_path.name

        try:
            stats = fix_trajectory_file(input_path, output_path, dry_run=args.dry_run)

            if not stats['processed']:
                total_errors += 1
                if args.verbose:
                    print(f"\nERROR: {input_path.name}")
                    print(f"  {stats.get('error', 'Unknown error')}")
            elif stats['needed_fix']:
                total_fixed += 1
                if args.verbose:
                    print(f"\nFIXED: {input_path.name}")
                    print(f"  states={stats['states_len']}, actions: {stats['actions_len_before']} → {stats['actions_len_after']}")
            else:
                total_unchanged += 1

        except Exception as e:
            total_errors += 1
            print(f"\nERROR processing {input_path.name}: {e}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(trajectory_files)}")
    print(f"  Fixed: {total_fixed}")
    print(f"  Unchanged (already correct): {total_unchanged}")
    print(f"  Errors: {total_errors}")

    if args.dry_run:
        print()
        print("=" * 70)
        print("DRY RUN COMPLETE (no files written)")
        print("=" * 70)
        print(f"Run without --dry_run to apply fixes to {output_dir}")
    else:
        print()
        print("=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(f"Fixed trajectories saved to: {output_dir}")

    print()


if __name__ == "__main__":
    main()
