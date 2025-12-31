#!/usr/bin/env python3
"""
Normalize existing team files to use proper pypkmn-compatible naming.

This script reads .gen1ou_team files and rewrites them with properly
capitalized Pokemon and move names.
"""

import argparse
from pathlib import Path
import re
from typing import List


def normalize_pokemon_name(name: str) -> str:
    """Normalize Pokemon name to proper case."""
    special_cases = {
        'mr. mime': 'Mr. Mime',
        'mr-mime': 'Mr. Mime',
        'mrmime': 'Mr. Mime',
        'farfetchd': "Farfetch'd",
        "farfetch'd": "Farfetch'd",
        'nidoran-f': 'Nidoran-F',
        'nidoran-m': 'Nidoran-M',
    }

    lower_name = name.lower().strip()
    if lower_name in special_cases:
        return special_cases[lower_name]

    return name.strip().title()


def normalize_move_name(move: str) -> str:
    """Normalize move name to proper format."""
    # Import pypkmn MOVES to check against
    try:
        from pykmn.data.gen1 import MOVES
    except ImportError:
        # Fallback if pypkmn not available
        MOVES = {}

    move = move.strip()

    # Try as-is first
    if move in MOVES:
        return move

    # Replace hyphens with spaces and title case
    normalized = move.replace('-', ' ').title()
    if normalized in MOVES:
        return normalized

    # Try just title case
    title_cased = move.title()
    if title_cased in MOVES:
        return title_cased

    # Special cases
    special_cases = {
        'softboiled': 'Soft-Boiled',
        'soft-boiled': 'Soft-Boiled',
        'hi jump kick': 'High Jump Kick',
        'vicegrip': 'Vice Grip',
        'doubleedge': 'Double-Edge',
        'double-edge': 'Double-Edge',
    }

    lower_move = move.lower()
    if lower_move in special_cases:
        return special_cases[lower_move]

    # Try splitting compound words (for all-lowercase or title case)
    if move.islower() or move.istitle():
        lower_move = move.lower()
        # Try all possible splits into two parts
        for i in range(1, len(lower_move)):
            part1 = lower_move[:i].title()
            part2 = lower_move[i:].title()
            candidate = f"{part1} {part2}"
            if candidate in MOVES:
                return candidate
            # Also try with hyphen
            candidate_hyphen = f"{part1}-{part2}"
            if candidate_hyphen in MOVES:
                return candidate_hyphen

    # Fallback: replace hyphens with spaces and title case
    return move.replace('-', ' ').title()


def normalize_team_file(input_path: Path, output_path: Path = None, dry_run: bool = False) -> bool:
    """
    Normalize a single team file.

    Args:
        input_path: Path to input team file
        output_path: Path to output file (overwrites input if None)
        dry_run: If True, only print changes without writing

    Returns:
        True if file was modified, False otherwise
    """
    if output_path is None:
        output_path = input_path

    # Read file
    with open(input_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    modified = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Empty line
            new_lines.append(line)
            continue

        if stripped.startswith('-'):
            # Move line
            move = stripped[1:].strip()
            # Remove comments
            move = move.split('#')[0].strip()
            normalized_move = normalize_move_name(move)

            if normalized_move != move:
                modified = True
                if dry_run:
                    print(f"  Move: {move} -> {normalized_move}")

            new_lines.append(f"- {normalized_move}\n")

        elif stripped.startswith('Ability:') or stripped.startswith('Level:') or \
             stripped.startswith('EVs:') or stripped.startswith('IVs:') or \
             stripped.startswith('Happiness:'):
            # Keep attribute lines as-is
            new_lines.append(line)

        elif '(' in stripped and ')' in stripped:
            # Nickname (Species) format - normalize species
            match = re.search(r'\(([^)]+)\)', stripped)
            if match:
                species = match.group(1).strip()
                normalized_species = normalize_pokemon_name(species)
                if normalized_species != species:
                    modified = True
                    if dry_run:
                        print(f"  Species: {species} -> {normalized_species}")
                    new_line = stripped.replace(f"({species})", f"({normalized_species})")
                    new_lines.append(new_line + '\n')
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        elif '@' in stripped:
            # Species @ Item format - normalize species
            species = stripped.split('@')[0].strip()
            rest = stripped.split('@', 1)[1]
            normalized_species = normalize_pokemon_name(species)
            if normalized_species != species:
                modified = True
                if dry_run:
                    print(f"  Species: {species} -> {normalized_species}")
                new_lines.append(f"{normalized_species} @ {rest}\n")
            else:
                new_lines.append(line)

        else:
            # Likely a plain species name
            normalized_species = normalize_pokemon_name(stripped)
            if normalized_species != stripped:
                modified = True
                if dry_run:
                    print(f"  Species: {stripped} -> {normalized_species}")
                new_lines.append(normalized_species + '\n')
            else:
                new_lines.append(line)

    # Write output
    if modified and not dry_run:
        with open(output_path, 'w') as f:
            f.writelines(new_lines)

    return modified


def main():
    parser = argparse.ArgumentParser(
        description='Normalize Pokemon team files to pypkmn-compatible format'
    )
    parser.add_argument(
        'team_dir',
        type=str,
        help='Directory containing team files'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='gen1ou',
        help='Battle format (default: gen1ou)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show changes without modifying files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: overwrite input files)'
    )

    args = parser.parse_args()

    team_dir = Path(args.team_dir).expanduser() / args.format
    if not team_dir.exists():
        print(f"Error: Directory not found: {team_dir}")
        return

    # Find all team files
    team_files = list(team_dir.glob(f'*.{args.format}_team'))
    print(f"Found {len(team_files)} team files in {team_dir}")

    if args.dry_run:
        print("\n=== DRY RUN MODE - No files will be modified ===\n")

    # Process each file
    modified_count = 0
    for team_file in team_files:
        if args.output_dir:
            output_dir = Path(args.output_dir) / args.format
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / team_file.name
        else:
            output_path = None

        if args.dry_run:
            print(f"\nChecking: {team_file.name}")

        was_modified = normalize_team_file(team_file, output_path, args.dry_run)

        if was_modified:
            modified_count += 1
            if not args.dry_run:
                print(f"✓ Normalized: {team_file.name}")

    print(f"\n{'Would modify' if args.dry_run else 'Modified'} {modified_count}/{len(team_files)} files")


if __name__ == '__main__':
    main()
