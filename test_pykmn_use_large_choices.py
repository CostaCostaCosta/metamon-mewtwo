"""
Test if the "non-standard" choices (>9) actually work in pypkmn.
"""

import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from pykmn.engine.gen1 import Battle, Player
from pykmn.engine.common import ResultType
from metamon.env.pykmn.team_parser import parse_showdown_team
import random


def test_using_large_choices():
    """Test if choices like 10, 14, 18 actually work."""
    print("=" * 60)
    print("Testing Large Choice Values")
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

    print("\nRunning battle using ALL legal choices (including >9)...")

    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        legal_p1 = list(battle.possible_choices_raw(Player.P1, result))
        legal_p2 = list(battle.possible_choices_raw(Player.P2, result))

        if not legal_p1 or not legal_p2:
            if result.type() == ResultType.NONE:
                print(f"\nERROR at step {step_count}: No legal choices but battle not finished")
                return False
            else:
                print(f"\nBattle completed normally at step {step_count}")
                print(f"Result: {result.type()}")
                return True

        # Pick random legal choices (including those >9)
        choice_p1 = random.choice(legal_p1)
        choice_p2 = random.choice(legal_p2)

        if step_count % 50 == 0:
            print(f"Step {step_count:4d}: P1 choice={choice_p1}, P2 choice={choice_p2}, num_legal={len(legal_p1)}")

        # Use them directly
        try:
            result, _ = battle.update_raw(choice_p1, choice_p2)
        except Exception as e:
            print(f"\nERROR at step {step_count}: Failed to use choices {choice_p1}, {choice_p2}")
            print(f"Error: {e}")
            return False

        if result.type() != ResultType.NONE:
            print(f"\nBattle completed normally at step {step_count}")
            print(f"Result: {result.type()}")
            return True

        step_count += 1

    print(f"\nWARNING: Battle timeout at {max_steps} steps")
    return False


def run_multiple_battles(num_battles=10):
    """Test multiple battles using large choices."""
    print("\n" + "=" * 60)
    print(f"Running {num_battles} battles with large choices...")
    print("=" * 60)

    team_dir = "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou"
    team_files = [
        os.path.join(team_dir, f)
        for f in os.listdir(team_dir)
        if f.endswith(".gen1ou_team")
    ]

    successes = 0
    failures = 0

    for i in range(num_battles):
        print(f"\nBattle {i+1}/{num_battles}:")

        selected = random.sample(team_files, 2)
        teams = [parse_showdown_team(open(f).read()) for f in selected]

        battle = Battle(teams[0], teams[1])
        result, _ = battle.update_raw(0, 0)

        step_count = 0
        max_steps = 1000

        while step_count < max_steps:
            legal_p1 = list(battle.possible_choices_raw(Player.P1, result))
            legal_p2 = list(battle.possible_choices_raw(Player.P2, result))

            if not legal_p1 or not legal_p2:
                if result.type() != ResultType.NONE:
                    print(f"  Completed at step {step_count}")
                    successes += 1
                else:
                    print(f"  ERROR: No choices at step {step_count}")
                    failures += 1
                break

            choice_p1 = random.choice(legal_p1)
            choice_p2 = random.choice(legal_p2)

            result, _ = battle.update_raw(choice_p1, choice_p2)

            if result.type() != ResultType.NONE:
                print(f"  Completed at step {step_count}")
                successes += 1
                break

            step_count += 1

        if step_count >= max_steps:
            print(f"  TIMEOUT at {max_steps} steps")
            failures += 1

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Successes: {successes}/{num_battles}")
    print(f"  Failures:  {failures}/{num_battles}")
    print("=" * 60)

    return failures == 0


if __name__ == "__main__":
    print("\nTest 1: Single battle with large choices")
    success = test_using_large_choices()

    print("\n" + "=" * 60)
    print("Test 2: Multiple battles")
    all_success = run_multiple_battles(10)

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if all_success:
        print("  ✓ All battles completed using large choice values!")
        print("  → The large choices (>9) are VALID")
        print("  → get_legal_mask() needs to be fixed to handle them")
    else:
        print("  ✗ Battles failed even when using raw choices")
    print("=" * 60)
