"""
Investigate the pypkmn choice encoding to understand what values like 10, 14, 18 mean.
"""

import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from pykmn.engine.gen1 import Battle, Player
from metamon.env.pykmn.team_parser import parse_showdown_team


def analyze_choice_encoding():
    """Analyze what pypkmn choice values mean."""
    print("=" * 60)
    print("Analyzing PyKMN Choice Encoding")
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

    print("\nInitial state (after team preview):")
    legal_p1 = list(battle.possible_choices_raw(Player.P1, result))
    legal_p2 = list(battle.possible_choices_raw(Player.P2, result))
    print(f"  P1 legal choices: {legal_p1}")
    print(f"  P2 legal choices: {legal_p2}")
    print()

    # Take first action
    print("Taking first action (both players use first legal choice)...")
    result, _ = battle.update_raw(legal_p1[0], legal_p2[0])

    print(f"\nAfter first action:")
    print(f"  Result type: {result.type()}")
    legal_p1 = list(battle.possible_choices_raw(Player.P1, result))
    legal_p2 = list(battle.possible_choices_raw(Player.P2, result))
    print(f"  P1 legal choices: {legal_p1}")
    print(f"  P2 legal choices: {legal_p2}")
    print()

    # Analyze choice values
    print("Analyzing choice values:")
    for choice in legal_p1:
        print(f"  Choice {choice:3d} = 0x{choice:02X} = 0b{choice:08b}")

        # Try to decode
        print(f"    Possible interpretations:")
        print(f"      - As move index: {choice} (out of range 1-4)")
        print(f"      - As switch index: {choice} (out of range 5-9)")
        print(f"      - High byte: {choice >> 4}, Low byte: {choice & 0xF}")
        print()

    # Let's step through several moves and track the pattern
    print("\n" + "=" * 60)
    print("Stepping through battle and tracking choice patterns...")
    print("=" * 60)

    for step in range(20):
        legal_p1 = list(battle.possible_choices_raw(Player.P1, result))
        legal_p2 = list(battle.possible_choices_raw(Player.P2, result))

        if not legal_p1 or not legal_p2:
            print(f"\nStep {step}: Battle ended or no legal actions")
            break

        print(f"\nStep {step}:")
        print(f"  P1 choices: {legal_p1}")
        print(f"  P2 choices: {legal_p2}")

        # Analyze if these are standard moves/switches or something else
        all_choices = set(legal_p1 + legal_p2)
        standard = []
        non_standard = []

        for c in all_choices:
            if c <= 9:
                standard.append(c)
            else:
                non_standard.append(c)

        if standard:
            print(f"  Standard (0-9): {sorted(standard)}")
        if non_standard:
            print(f"  Non-standard (>9): {sorted(non_standard)}")

        # Take actions
        result, _ = battle.update_raw(legal_p1[0], legal_p2[0])


if __name__ == "__main__":
    analyze_choice_encoding()
