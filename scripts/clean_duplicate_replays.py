#!/usr/bin/env python3
"""
Clean replays with duplicate Pokemon from trajectory directories.

This script scans trajectory files for duplicate Pokemon in teams
(violating Species Clause) and removes the corrupted replays.

Usage:
    python scripts/clean_duplicate_replays.py \
        --replay_dir ~/metamon/trajectories/kakuna-v2b \
        --format gen1ou \
        --dry_run  # Preview without deleting

    # Actually delete:
    python scripts/clean_duplicate_replays.py \
        --replay_dir ~/metamon/trajectories/kakuna-v2b \
        --format gen1ou
"""

import argparse
import lz4.frame
import json
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Set


def check_team_for_duplicates(states: List[dict]) -> Tuple[bool, Set[str]]:
    """
    Check if a trajectory has duplicate Pokemon in the team.

    Args:
        states: List of state dictionaries from trajectory

    Returns:
        Tuple of (has_duplicates: bool, duplicate_species: Set[str])
    """
    if not states or len(states) == 0:
        return False, set()

    # Check first state (team composition doesn't change)
    first_state = states[0]

    # Collect all team members
    team_species = []

    # Add active Pokemon
    if 'player_active_pokemon' in first_state:
        active = first_state['player_active_pokemon']
        if isinstance(active, dict) and 'base_species' in active:
            team_species.append(active['base_species'])

    # Add bench Pokemon
    if 'available_switches' in first_state:
        for pokemon in first_state['available_switches']:
            if isinstance(pokemon, dict) and 'base_species' in pokemon:
                team_species.append(pokemon['base_species'])

    # Count species occurrences
    species_counts = Counter(team_species)
    duplicates = {species for species, count in species_counts.items() if count > 1}

    return len(duplicates) > 0, duplicates


def scan_directory(
    replay_dir: Path,
    battle_format: str,
    dry_run: bool = True,
    verbose: bool = True
) -> Tuple[int, int, List[Path]]:
    """
    Scan directory for replays with duplicate Pokemon.

    Args:
        replay_dir: Root directory containing replays
        battle_format: Format subdirectory (e.g., "gen1ou")
        dry_run: If True, don't actually delete files
        verbose: Print progress updates

    Returns:
        Tuple of (total_scanned, num_duplicates, deleted_files)
    """
    format_dir = replay_dir / battle_format

    if not format_dir.exists():
        raise FileNotFoundError(f"Format directory not found: {format_dir}")

    # Find all replay files
    replay_files = list(format_dir.glob("*.json.lz4"))
    total_files = len(replay_files)

    if verbose:
        print(f"Scanning {total_files} replay files in {format_dir}...")

    duplicates_found = 0
    files_to_delete = []

    for i, replay_file in enumerate(replay_files):
        try:
            # Load trajectory
            with lz4.frame.open(replay_file, 'rb') as f:
                data = json.load(f)

            # Check for duplicates
            has_duplicates, duplicate_species = check_team_for_duplicates(data.get('states', []))

            if has_duplicates:
                duplicates_found += 1
                files_to_delete.append(replay_file)

                if verbose:
                    print(f"  ❌ {replay_file.name}: Duplicates found - {duplicate_species}")

        except Exception as e:
            print(f"  ⚠️  Error reading {replay_file.name}: {e}")
            continue

        # Progress update every 1000 files
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{total_files} files scanned, {duplicates_found} duplicates found")

    # Delete files if not dry run
    if not dry_run and files_to_delete:
        if verbose:
            print(f"\nDeleting {len(files_to_delete)} files with duplicate Pokemon...")

        for file_path in files_to_delete:
            file_path.unlink()
            if verbose:
                print(f"  Deleted: {file_path.name}")

    return total_files, duplicates_found, files_to_delete


def rebuild_index(replay_dir: Path, battle_format: str, verbose: bool = True):
    """
    Rebuild index.csv after cleaning.

    Args:
        replay_dir: Root directory containing replays
        battle_format: Format subdirectory
        verbose: Print progress
    """
    format_dir = replay_dir / battle_format
    index_path = replay_dir / "index.csv"

    if verbose:
        print(f"\nRebuilding index.csv...")

    # Find all remaining replay files
    replay_files = sorted(format_dir.glob("*.json.lz4"))

    # Write index
    with open(index_path, 'w') as f:
        f.write("filename\n")
        for replay_file in replay_files:
            relative_path = f"{battle_format}/{replay_file.name}"
            f.write(f"{relative_path}\n")

    if verbose:
        print(f"  ✓ Index rebuilt with {len(replay_files)} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Clean replays with duplicate Pokemon from trajectory directories"
    )
    parser.add_argument(
        "--replay_dir",
        type=str,
        required=True,
        help="Root directory containing replay files",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format subdirectory (default: gen1ou)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview files to delete without actually deleting",
    )
    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Rebuild index.csv after cleaning",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Expand paths
    replay_dir = Path(args.replay_dir).expanduser()

    print("=" * 70)
    print("Duplicate Pokemon Replay Cleaner")
    print("=" * 70)
    print(f"Replay directory: {replay_dir}")
    print(f"Format: {args.format}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'DELETE MODE'}")
    print("=" * 70)
    print()

    # Scan directory
    total_scanned, num_duplicates, files_to_delete = scan_directory(
        replay_dir=replay_dir,
        battle_format=args.format,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total files scanned: {total_scanned}")
    print(f"Files with duplicates: {num_duplicates} ({num_duplicates / max(total_scanned, 1) * 100:.2f}%)")

    if args.dry_run:
        print()
        print("DRY RUN: No files were deleted.")
        print("Run without --dry_run to actually delete files.")
        print()
        print("Files that would be deleted:")
        for file_path in files_to_delete[:10]:
            print(f"  - {file_path.name}")
        if len(files_to_delete) > 10:
            print(f"  ... and {len(files_to_delete) - 10} more")
    else:
        print(f"Files deleted: {len(files_to_delete)}")
        print(f"Files remaining: {total_scanned - len(files_to_delete)}")

        # Rebuild index if requested
        if args.rebuild_index:
            rebuild_index(replay_dir, args.format, verbose=not args.quiet)

    print("=" * 70)


if __name__ == "__main__":
    main()
