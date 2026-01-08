#!/usr/bin/env python3
"""
Convert and split pypkmn trajectories for training.

Converts dual-perspective pypkmn format:
    {uuid}_pypkmn.json.lz4 with keys: states_p1, actions_p1, states_p2, actions_p2, etc.

To two single-perspective training files:
    metamon-{format}-{id}_..._WIN_P1.json.lz4 with keys: states, actions, rewards
    metamon-{format}-{id}_..._LOSS_P2.json.lz4 with keys: states, actions, rewards
"""

import argparse
import json
import lz4.frame
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def convert_and_split_trajectory(
    input_path: Path,
    output_dir: Path,
    battle_format: str,
    counter: int,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Load a pypkmn trajectory and split it into P1 and P2 perspectives.

    Returns:
        (success_count, error_count) tuple
    """
    try:
        # Load trajectory
        with lz4.frame.open(input_path, 'rb') as f:
            data = json.load(f)

        # Extract metadata
        winner = data.get('winner', 1)
        timestamp = data.get('timestamp', datetime.now().isoformat())
        date_str = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

        # Check if actions need fixing (old bug)
        states_p1_len = len(data.get('states_p1', []))
        actions_p1_len = len(data.get('actions_p1', []))

        fix_needed = actions_p1_len == states_p1_len + 1

        success = 0

        # Create P1 file
        p1_result = "WIN" if winner == 1 else "LOSS"
        p1_filename = f"metamon-{battle_format}-{counter:06d}_Unrated_PyKMNP1_vs_PyKMNP2_{date_str}_{p1_result}_P1.json.lz4"
        p1_path = output_dir / p1_filename

        p1_data = {
            'format': data.get('format', battle_format),
            'states': data['states_p1'],
            'actions': data['actions_p1'][:-1] if fix_needed else data['actions_p1'],
            'rewards': data.get('rewards_p1', []),
            'winner': winner,
            'num_turns': data.get('num_turns', len(data['states_p1'])),
            'timestamp': timestamp,
            'source': 'pypkmn',
        }

        if not dry_run:
            json_str = json.dumps(p1_data)
            compressed = lz4.frame.compress(json_str.encode('utf-8'))
            with open(p1_path, 'wb') as f:
                f.write(compressed)
        success += 1

        # Create P2 file
        p2_result = "WIN" if winner == 2 else "LOSS"
        p2_filename = f"metamon-{battle_format}-{counter:06d}_Unrated_PyKMNP1_vs_PyKMNP2_{date_str}_{p2_result}_P2.json.lz4"
        p2_path = output_dir / p2_filename

        p2_data = {
            'format': data.get('format', battle_format),
            'states': data['states_p2'],
            'actions': data['actions_p2'][:-1] if fix_needed else data['actions_p2'],
            'rewards': data.get('rewards_p2', []),
            'winner': 3 - winner,  # Flip winner for P2 perspective
            'num_turns': data.get('num_turns', len(data['states_p2'])),
            'timestamp': timestamp,
            'source': 'pypkmn',
        }

        if not dry_run:
            json_str = json.dumps(p2_data)
            compressed = lz4.frame.compress(json_str.encode('utf-8'))
            with open(p2_path, 'wb') as f:
                f.write(compressed)
        success += 1

        return success, 0

    except Exception as e:
        print(f"\nError processing {input_path.name}: {e}")
        return 0, 1


def main():
    parser = argparse.ArgumentParser(
        description="Convert and split pypkmn trajectories for training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing {format}/*_pypkmn.json.lz4",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for split trajectories",
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
        help="Preview without writing files",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    battle_format = args.format

    print("=" * 70)
    print("PYPKMN TRAJECTORY CONVERSION & SPLITTING")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Battle format: {battle_format}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 70)
    print()

    # Find input files
    source_format_dir = input_dir / battle_format
    if not source_format_dir.exists():
        print(f"❌ Source directory not found: {source_format_dir}")
        return

    pypkmn_files = sorted(source_format_dir.glob("*_pypkmn.json.lz4"))

    if not pypkmn_files:
        print(f"❌ No *_pypkmn.json.lz4 files found in {source_format_dir}")
        return

    print(f"Found {len(pypkmn_files):,} pypkmn trajectory files")
    print(f"Will create {len(pypkmn_files) * 2:,} training files (P1 + P2 splits)")
    print()

    # Create output directory
    output_format_dir = output_dir / battle_format
    if not args.dry_run:
        output_format_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    total_success = 0
    total_errors = 0

    for i, input_path in enumerate(tqdm(pypkmn_files, desc="Processing files")):
        success, errors = convert_and_split_trajectory(
            input_path,
            output_format_dir,
            battle_format,
            counter=i,
            dry_run=args.dry_run,
        )
        total_success += success
        total_errors += errors

    # Summary
    print()
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE (no files written)")
        print(f"Would create {len(pypkmn_files) * 2:,} files")
    else:
        print("CONVERSION COMPLETE")
        print(f"  ✓ Successfully created: {total_success:,} files")
        print(f"  ✗ Errors: {total_errors:,}")
        print(f"  Output directory: {output_format_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
