"""
Investigate when pypkmn returns PASS as the only legal choice.
"""

import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from pykmn.engine.gen1 import Battle, Player
from pykmn.engine.common import ResultType, Choice, ChoiceType
from metamon.env.pykmn.team_parser import parse_showdown_team
from metamon.env.pykmn.action_mapper import ActionMappings
from metamon.env.pykmn.features import precompute_mappings
import random


def investigate_pass():
    """Find when PASS is the only legal choice."""
    print("=" * 60)
    print("Investigating PASS States")
    print("=" * 60)

    team_dir = "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou"
    team_files = [
        os.path.join(team_dir, f)
        for f in os.listdir(team_dir)
        if f.endswith(".gen1ou_team")
    ][:2]

    teams = [parse_showdown_team(open(f).read()) for f in team_files]

    mappings = precompute_mappings()
    action_mappings = ActionMappings.create()

    battle = Battle(teams[0], teams[1])
    result, _ = battle.update_raw(0, 0)

    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        raw_p1 = list(battle.possible_choices_raw(Player.P1, result))
        raw_p2 = list(battle.possible_choices_raw(Player.P2, result))

        # Check if either player has only PASS
        if raw_p1 == [0] or raw_p2 == [0]:
            print(f"\n{'='*60}")
            print(f"PASS FOUND at step {step_count}!")
            print(f"{'='*60}")
            print(f"Raw P1 choices: {raw_p1}")
            print(f"Raw P2 choices: {raw_p2}")
            print(f"Result type: {result.type()}")
            print()

            # Decode all choices
            print("Decoded P1 choices:")
            for raw in raw_p1:
                choice = Choice(raw)
                print(f"  {raw:3d}: {choice}")

            print("\nDecoded P2 choices:")
            for raw in raw_p2:
                choice = Choice(raw)
                print(f"  {raw:3d}: {choice}")

            # Try to understand the game state
            print()
            print("Attempting to continue with PASS...")

            # Use PASS for P1 (or whichever has PASS), and first choice for P2
            choice_p1 = raw_p1[0]
            choice_p2 = raw_p2[0]

            result, _ = battle.update_raw(choice_p1, choice_p2)
            print(f"After step: result type = {result.type()}")

            if result.type() != ResultType.NONE:
                print(f"Battle ended!")
                return

            step_count += 1
            continue

        if not raw_p1 or not raw_p2:
            print(f"\nEmpty choices at step {step_count}")
            break

        # Normal turn
        choice_p1 = random.choice(raw_p1)
        choice_p2 = random.choice(raw_p2)

        result, _ = battle.update_raw(choice_p1, choice_p2)

        if result.type() != ResultType.NONE:
            print(f"\nBattle ended normally at step {step_count}")
            return

        step_count += 1

    print(f"\nNo PASS states found in {step_count} steps (or battle timeout)")


if __name__ == "__main__":
    for i in range(5):
        print(f"\n\nAttempt {i+1}/5:")
        print("=" * 60)
        investigate_pass()
