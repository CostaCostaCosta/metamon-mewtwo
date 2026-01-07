#!/usr/bin/env python3
"""
Convert pypkmn trajectories from dual-perspective format to single-perspective format.

Converts files with:
    {"states_p1": [...], "actions_p1": [...], "states_p2": [...], "actions_p2": [...]}
To:
    {"states": [...], "actions": [...]}  (P1 perspective only)
"""

import argparse
import json
import lz4.frame
from pathlib import Path
from tqdm import tqdm


def convert_file(input_path: Path, output_dir: Path):
    """
    Convert a single file from dual to single perspective.
    Creates TWO output files: one for P1 perspective, one for P2 perspective.
    """
    # Load file
    with lz4.frame.open(input_path, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))

    # Check if already in new format (single perspective)
    if "states" in data and "actions" in data and "states_p1" not in data:
        # Already converted, just copy once
        output_path = output_dir / input_path.name
        with lz4.frame.open(output_path, "wb") as f:
            f.write(lz4.frame.compress(json.dumps(data).encode("utf-8")))
        return 1

    # Convert to TWO separate files (one per player perspective)

    # P1 perspective
    p1_data = {
        "format": data.get("format", "gen1ou"),
        "states": data["states_p1"],
        "actions": data["actions_p1"] + [data["actions_p1"][-1]],  # Add terminal action
        "rewards": data.get("rewards_p1", [0] * len(data["states_p1"])),
        "winner": data["winner"],
        "num_turns": data.get("num_turns", len(data["states_p1"])),
        "timestamp": data.get("timestamp", ""),
        "source": data.get("source", "pypkmn"),
    }

    # P2 perspective (flip winner: 1->2, 2->1)
    p2_data = {
        "format": data.get("format", "gen1ou"),
        "states": data["states_p2"],
        "actions": data["actions_p2"] + [data["actions_p2"][-1]],  # Add terminal action
        "rewards": data.get("rewards_p2", [0] * len(data["states_p2"])),
        "winner": 2 if data["winner"] == 1 else 1,  # Flip winner from P2's perspective
        "num_turns": data.get("num_turns", len(data["states_p2"])),
        "timestamp": data.get("timestamp", ""),
        "source": data.get("source", "pypkmn"),
    }

    # Generate output filenames (add _P1 and _P2 suffixes)
    base_name = input_path.stem.replace(".json", "")  # Remove .json.lz4

    p1_path = output_dir / f"{base_name}_P1.json.lz4"
    p2_path = output_dir / f"{base_name}_P2.json.lz4"

    # Save P1 perspective (lz4.frame.open handles compression automatically)
    with lz4.frame.open(p1_path, "wb") as f:
        f.write(json.dumps(p1_data).encode("utf-8"))

    # Save P2 perspective
    with lz4.frame.open(p2_path, "wb") as f:
        f.write(json.dumps(p2_data).encode("utf-8"))

    return 2  # Created 2 files


def convert_directory(input_dir: Path, output_dir: Path, battle_format: str = "gen1ou"):
    """Convert all files in a directory, creating 2 files per battle (one per player)."""
    input_format_dir = input_dir / battle_format
    output_format_dir = output_dir / battle_format
    output_format_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_format_dir.glob("*.json.lz4"))
    print(f"Found {len(files):,} files to convert")
    print(f"Will create ~{len(files) * 2:,} output files (2 per battle)")

    files_created = 0
    error_count = 0

    for input_path in tqdm(files, desc="Converting files"):
        try:
            num_created = convert_file(input_path, output_format_dir)
            files_created += num_created
        except Exception as e:
            print(f"\nError converting {input_path.name}: {e}")
            error_count += 1

    print(f"\n{'=' * 70}")
    print("CONVERSION COMPLETE")
    print(f"  Input files: {len(files):,}")
    print(f"  ✓ Output files created: {files_created:,}")
    print(f"  ✗ Errors: {error_count:,}")
    print(f"  Output directory: {output_format_dir}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pypkmn trajectories to single-perspective format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with dual-perspective files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (default: gen1ou)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    print("=" * 70)
    print("PYPKMN FORMAT CONVERSION")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Battle format: {args.format}")
    print("=" * 70)
    print()

    convert_directory(input_dir, output_dir, args.format)


if __name__ == "__main__":
    main()
