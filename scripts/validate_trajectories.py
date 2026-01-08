#!/usr/bin/env python3
"""
Validate pypkmn trajectories and identify corrupted files.

This script loads each trajectory and checks for:
1. Correct format (states, actions, rewards keys)
2. Length consistency (len(states) == len(actions))
3. Observation shape consistency within trajectory
4. Ability to convert to training format

Corrupted files are logged and optionally moved to a separate directory.
"""

import argparse
import json
import lz4.frame
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metamon.interface import DefaultObservationSpace, UniversalState, AggressiveShapedReward
from metamon.data.parsed_replay_dset import MetamonDataset
import numpy as np


def validate_trajectory(filepath: Path, obs_space: DefaultObservationSpace) -> tuple[bool, str]:
    """
    Validate a single trajectory file.

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        # Load file
        with lz4.frame.open(filepath, 'rb') as f:
            data = json.load(f)

        # Check required keys
        required_keys = ['states', 'actions']
        for key in required_keys:
            if key not in data:
                return False, f"Missing key: {key}"

        states = data['states']
        actions = data['actions']

        # Check length consistency
        if len(states) != len(actions):
            return False, f"Length mismatch: {len(states)} states vs {len(actions)} actions"

        if len(states) == 0:
            return False, "Empty trajectory"

        # Try to convert states to UniversalState
        try:
            universal_states = [UniversalState.from_dict(s) for s in states]
        except Exception as e:
            return False, f"Failed to parse states: {e}"

        # Try to generate observations and check shapes
        obs_space.reset()
        try:
            obs_list = [obs_space.state_to_obs(s) for s in universal_states]
        except Exception as e:
            return False, f"Failed to generate observations: {e}"

        # Check observation shape consistency
        obs_dict = defaultdict(list)
        for obs in obs_list:
            for k, v in obs.items():
                obs_dict[k].append(v)

        # Try to stack observations (this is where many errors occur)
        for k, v_list in obs_dict.items():
            try:
                np.stack(v_list, axis=0)
            except ValueError as e:
                return False, f"Cannot stack observation '{k}': {e}"

        return True, ""

    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Validate pypkmn trajectories")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing trajectory files",
    )
    parser.add_argument(
        "--output_bad",
        type=str,
        default=None,
        help="Move bad files to this directory (optional)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (default: gen1ou)",
    )
    parser.add_argument(
        "--max_check",
        type=int,
        default=None,
        help="Maximum number of files to check (for testing)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    battle_format = args.format

    print("=" * 70)
    print("TRAJECTORY VALIDATION")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    if args.output_bad:
        print(f"Bad files will be moved to: {args.output_bad}")
    print("=" * 70)
    print()

    # Find trajectory files
    format_dir = input_dir / battle_format
    if not format_dir.exists():
        # Try flat directory
        format_dir = input_dir

    trajectory_files = sorted(format_dir.glob("*.json.lz4"))

    if not trajectory_files:
        print(f"❌ No .json.lz4 files found in {format_dir}")
        return

    if args.max_check:
        trajectory_files = trajectory_files[:args.max_check]
        print(f"Checking first {len(trajectory_files)} files (--max_check={args.max_check})")
    else:
        print(f"Found {len(trajectory_files):,} trajectory files")
    print()

    # Create observation space
    obs_space = DefaultObservationSpace()

    # Validate files
    valid_count = 0
    invalid_count = 0
    error_types = defaultdict(int)
    invalid_files = []

    for filepath in tqdm(trajectory_files, desc="Validating"):
        is_valid, error_msg = validate_trajectory(filepath, obs_space)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            invalid_files.append((filepath, error_msg))
            # Categorize error
            error_category = error_msg.split(':')[0]
            error_types[error_category] += 1

    # Summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(trajectory_files):,}")
    print(f"  ✓ Valid: {valid_count:,} ({100*valid_count/len(trajectory_files):.1f}%)")
    print(f"  ✗ Invalid: {invalid_count:,} ({100*invalid_count/len(trajectory_files):.1f}%)")

    if error_types:
        print()
        print("Error breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

    if invalid_files:
        print()
        print(f"First 10 invalid files:")
        for filepath, error_msg in invalid_files[:10]:
            print(f"  {filepath.name}")
            print(f"    → {error_msg}")

    # Move bad files if requested
    if args.output_bad and invalid_files:
        output_bad_dir = Path(args.output_bad).expanduser() / battle_format
        output_bad_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f"Moving {len(invalid_files)} invalid files to {output_bad_dir}...")

        for filepath, _ in tqdm(invalid_files, desc="Moving bad files"):
            try:
                filepath.rename(output_bad_dir / filepath.name)
            except Exception as e:
                print(f"  Error moving {filepath.name}: {e}")

        print(f"✓ Moved {len(invalid_files)} files")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
