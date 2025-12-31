"""
Decode pypkmn choice values to understand the encoding.
"""

import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from pykmn.engine.gen1 import Battle, Player
from pykmn.engine.common import Choice, ChoiceType
from metamon.env.pykmn.team_parser import parse_showdown_team


def decode_choices():
    """Decode pypkmn choice values."""
    print("=" * 60)
    print("Decoding PyKMN Choice Values")
    print("=" * 60)

    # Load teams
    team_dir = "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou"
    team_files = [
        os.path.join(team_dir, f)
        for f in os.listdir(team_dir)
        if f.endswith(".gen1ou_team")
    ][:2]

    teams = [parse_showdown_team(open(f).read()) for f in team_files]

    # Create battle
    battle = Battle(teams[0], teams[1])
    result, _ = battle.update_raw(0, 0)  # Team preview

    # Get raw choices
    raw_choices = list(battle.possible_choices_raw(Player.P1, result))
    print(f"\nRaw choices: {raw_choices}")
    print()

    # Decode each choice
    print("Decoding each choice:")
    print("-" * 60)
    for raw in raw_choices:
        choice = Choice(raw)
        choice_type = choice.type()
        choice_data = choice.data()

        print(f"Raw {raw:3d} (0x{raw:02X}): type={choice_type.name:6s}, data={choice_data}")

    # Group by type
    print()
    print("Grouped by type:")
    print("-" * 60)

    moves = []
    switches = []
    passes = []

    for raw in raw_choices:
        choice = Choice(raw)
        choice_type = choice.type()
        choice_data = choice.data()

        if choice_type == ChoiceType.MOVE:
            moves.append((raw, choice_data))
        elif choice_type == ChoiceType.SWITCH:
            switches.append((raw, choice_data))
        elif choice_type == ChoiceType.PASS:
            passes.append((raw, choice_data))

    if moves:
        print(f"\nMoves ({len(moves)}):")
        for raw, data in moves:
            print(f"  Raw {raw:3d} → Move #{data}")

    if switches:
        print(f"\nSwitches ({len(switches)}):")
        for raw, data in switches:
            print(f"  Raw {raw:3d} → Switch to slot #{data}")

    if passes:
        print(f"\nPasses ({len(passes)}):")
        for raw, data in passes:
            print(f"  Raw {raw:3d} → Pass")

    # Now derive the encoding pattern
    print()
    print("=" * 60)
    print("Encoding Pattern Analysis:")
    print("=" * 60)

    if moves:
        print("\nMove encoding:")
        for raw, data in moves:
            print(f"  Move #{data} → raw {raw} (0x{raw:02X}, 0b{raw:08b})")

    if switches:
        print("\nSwitch encoding:")
        for raw, data in switches:
            print(f"  Switch to slot #{data} → raw {raw} (0x{raw:02X}, 0b{raw:08b})")


if __name__ == "__main__":
    decode_choices()
