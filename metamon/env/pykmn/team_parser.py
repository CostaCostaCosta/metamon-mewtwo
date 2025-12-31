"""
Parse Showdown team format to pypkmn Pokemon objects.

This module provides utilities to convert Showdown team export format
(the text format used by pok epast.es and saved in .gen1ou_team files)
into pypkmn Pokemon objects for battle simulation.
"""

import re
from pathlib import Path
from typing import List, Tuple

from pykmn.engine.gen1 import Pokemon


def parse_showdown_team(team_text: str) -> List[Pokemon]:
    """
    Parse a Showdown team export format into pypkmn Pokemon objects.

    Args:
        team_text: Team string in Showdown export format. Includes species,
            moves, and optionally EVs/IVs (though Gen 1 has limited support).

    Returns:
        List of 6 Pokemon objects ready for battle simulation.

    Example:
        >>> team_text = '''
        ... Tauros
        ... - Body Slam
        ... - Hyper Beam
        ... - Blizzard
        ... - Earthquake
        ...
        ... Chansey
        ... - Ice Beam
        ... - Thunderbolt
        ... - Thunder Wave
        ... - Soft-Boiled
        ... '''
        >>> team = parse_showdown_team(team_text)
        >>> len(team)
        2
    """
    # Split by double newline to separate Pokemon
    pokemon_blocks = re.split(r'\n\s*\n', team_text.strip())
    pokemon_blocks = [block.strip() for block in pokemon_blocks if block.strip()]

    pokemon_list = []
    for block in pokemon_blocks:
        pokemon = _parse_single_pokemon(block)
        if pokemon is not None:
            pokemon_list.append(pokemon)

    if len(pokemon_list) != 6:
        raise ValueError(
            f"Expected exactly 6 Pokemon in team, got {len(pokemon_list)}. "
            "Gen 1 OU requires teams of 6 Pokemon."
        )

    return pokemon_list


def _parse_single_pokemon(block: str) -> Pokemon:
    """Parse a single Pokemon block from Showdown format."""
    lines = [line.strip() for line in block.split('\n') if line.strip()]

    if not lines:
        return None

    # First line is species name (possibly with nickname)
    # Format: "Nickname (Species)" or just "Species"
    species_line = lines[0]
    species = _extract_species_name(species_line)

    # Extract moves (lines starting with "-")
    moves = []
    for line in lines[1:]:
        if line.startswith('-'):
            move = line[1:].strip()
            # Remove any comments or additional info after the move
            move = move.split('#')[0].strip()
            # Normalize move name to pypkmn format
            move = _normalize_move_name(move)
            moves.append(move)

    if len(moves) == 0:
        raise ValueError(f"Pokemon {species} has no moves")

    if len(moves) > 4:
        raise ValueError(
            f"Pokemon {species} has {len(moves)} moves, but maximum is 4"
        )

    # Pad moves to 4 if less than 4 are provided
    # PyKMN requires exactly 4 moves
    while len(moves) < 4:
        moves.append(moves[0])  # Duplicate first move as placeholder

    # Create Pokemon object
    # Note: pypkmn.engine.gen1.Pokemon constructor accepts:
    # - species (str): Species name
    # - moves (tuple of str): Exactly 4 move names
    # - level (int, optional): Pokemon level (default 100)
    # - happiness (int, optional): Happiness value
    # Gen 1 doesn't use EVs/IVs in the same way, so we use defaults
    return Pokemon(species=species, moves=tuple(moves[:4]))


def _normalize_move_name(move: str) -> str:
    """
    Normalize move name to pypkmn's expected format.

    PyKMN expects title case with spaces (e.g., "Thunder Wave", "Ice Beam").
    This function converts common variations like "thunder-wave" or "thunderbolt".
    """
    from pykmn.data.gen1 import MOVES

    # Try the name as-is first
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

    # Handle special cases and common compound words
    special_cases = {
        'u-turn': 'U-turn',
        'u turn': 'U-turn',
        'x-scissor': 'X-Scissor',
        'softboiled': 'Soft-Boiled',
        'soft boiled': 'Soft-Boiled',
        'hi jump kick': 'High Jump Kick',
        'vicegrip': 'Vice Grip',
        'nightshade': 'Night Shade',
        'extremespeed': 'Extreme Speed',
        'solarbeam': 'Solar Beam',
        'ancientpower': 'Ancient Power',
        'dynamicpunch': 'Dynamic Punch',
        'dragonbreath': 'Dragon Breath',
        'shadowball': 'Shadow Ball',
        'mudslap': 'Mud-Slap',
        'rocksmash': 'Rock Smash',
        'thunderwave': 'Thunder Wave',
        'icebeam': 'Ice Beam',
        'thunderbolt': 'Thunderbolt',
        'doubleedge': 'Double-Edge',
        'bodyslam': 'Body Slam',
        'hyperbeam': 'Hyper Beam',
        'earthquake': 'Earthquake',
        'psychic': 'Psychic',
        'recover': 'Recover',
        'rest': 'Rest',
        'explosion': 'Explosion',
        'surf': 'Surf',
        'blizzard': 'Blizzard',
    }

    lower_move = move.lower()
    if lower_move in special_cases:
        return special_cases[lower_move]

    # Try fuzzy matching: for compound words, try adding space before capital letters
    # e.g., "IceBeam" -> "Ice Beam", "Lovelykiss" -> "Lovely Kiss"
    import re

    # Method 1: Add space before capitals (except at start) for CamelCase
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', move)
    if spaced in MOVES:
        return spaced

    # Method 2: For all-lowercase compound words, try splitting at common boundaries
    # and checking all possible splits
    if move.islower() or move.istitle():
        lower_move = move.lower()
        # Try all possible ways to split the word into two parts
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

    # Last resort: return normalized version
    return normalized


def _normalize_species_name(species: str) -> str:
    """
    Normalize species name to pypkmn's expected format.

    PyKMN expects title case (e.g., "Jynx", "Tauros", "Mr. Mime").
    This function handles common variations and special cases.
    """
    from pykmn.data.gen1 import SPECIES_IDS

    # Try the name as-is first
    if species in SPECIES_IDS:
        return species

    # Try title case
    title_cased = species.title()
    if title_cased in SPECIES_IDS:
        return title_cased

    # Handle special cases
    special_cases = {
        'mr-mime': 'Mr. Mime',
        'mr mime': 'Mr. Mime',
        'mrmime': 'Mr. Mime',
        'farfetchd': "Farfetch'd",
        'farfetch-d': "Farfetch'd",
        'nidoran-f': 'Nidoran-F',
        'nidoran-m': 'Nidoran-M',
        'nidoranf': 'Nidoran-F',
        'nidoranm': 'Nidoran-M',
    }

    lower_species = species.lower()
    if lower_species in special_cases:
        return special_cases[lower_species]

    # If still not found, try common variations
    # Replace hyphens with spaces and title case
    with_spaces = species.replace('-', ' ').title()
    if with_spaces in SPECIES_IDS:
        return with_spaces

    # Last resort: just return title case and let pypkmn error if invalid
    return title_cased


def _extract_species_name(species_line: str) -> str:
    """
    Extract species name from the first line of a Pokemon block.

    Handles formats:
    - "Species" -> "Species"
    - "Nickname (Species)" -> "Species"
    - "Species @ Item" -> "Species"
    - "Nickname (Species) @ Item" -> "Species"

    Normalizes to pypkmn's expected format (title case).
    """
    # Remove item if present
    species_line = species_line.split('@')[0].strip()

    # Check for nickname format: "Nickname (Species)"
    match = re.search(r'\(([^)]+)\)', species_line)
    if match:
        species = match.group(1).strip()
    else:
        species = species_line.strip()

    # Normalize to pypkmn's expected format (title case)
    # Handle special cases like "Mr. Mime", "Farfetch'd", "Nidoran-F", etc.
    return _normalize_species_name(species)


def parse_team_file(team_file_path: str | Path) -> List[Pokemon]:
    """
    Load and parse a team from a file.

    Args:
        team_file_path: Path to team file in Showdown export format
            (typically .gen1ou_team extension).

    Returns:
        List of 6 Pokemon objects ready for battle simulation.

    Example:
        >>> team = parse_team_file("~/metamon_cache/teams/my_team.gen1ou_team")
        >>> len(team)
        6
    """
    team_file_path = Path(team_file_path).expanduser()

    if not team_file_path.exists():
        raise FileNotFoundError(f"Team file not found: {team_file_path}")

    with open(team_file_path, 'r', encoding='utf-8') as f:
        team_text = f.read()

    return parse_showdown_team(team_text)


def load_random_teams(
    team_dir: str | Path,
    battle_format: str,
    num_teams: int
) -> List[List[Pokemon]]:
    """
    Load multiple random teams from a directory.

    Args:
        team_dir: Directory containing team files
        battle_format: Format to filter (e.g., "gen1ou")
        num_teams: Number of teams to load

    Returns:
        List of teams (each team is a list of 6 Pokemon)

    Example:
        >>> teams = load_random_teams("~/metamon_cache/teams", "gen1ou", 100)
        >>> len(teams)
        100
        >>> len(teams[0])
        6
    """
    import random

    team_dir = Path(team_dir).expanduser()
    extension = f".{battle_format.lower()}_team"

    # Find all team files
    team_files = list(team_dir.rglob(f"*{extension}"))

    if len(team_files) == 0:
        raise ValueError(
            f"No team files found in {team_dir} with format {battle_format}"
        )

    if len(team_files) < num_teams:
        # Sample with replacement if not enough unique teams
        sampled_files = random.choices(team_files, k=num_teams)
    else:
        # Sample without replacement
        sampled_files = random.sample(team_files, num_teams)

    teams = []
    for team_file in sampled_files:
        try:
            team = parse_team_file(team_file)
            teams.append(team)
        except Exception as e:
            # Log error but continue
            print(f"Warning: Failed to parse team file {team_file}: {e}")
            continue

    if len(teams) == 0:
        raise ValueError(f"Failed to parse any team files from {team_dir}")

    return teams
