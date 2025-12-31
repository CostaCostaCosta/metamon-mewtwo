"""
Minimal test to reproduce PyKMN battle bug where legal actions become empty.

This test uses pypkmn directly without any metamon wrappers to determine
if the bug is in pypkmn itself or in our integration code.
"""

from pykmn.engine.gen1 import Battle, Player, Pokemon
from pykmn.engine.common import ResultType
import random


def create_simple_team():
    """Create a simple 6-Pokemon team for testing."""
    # Use common Gen1 OU Pokemon with standard movesets
    return [
        Pokemon(species="Tauros", moves=("Body Slam", "Hyper Beam", "Blizzard", "Earthquake")),
        Pokemon(species="Snorlax", moves=("Body Slam", "Earthquake", "Rest", "Ice Beam")),
        Pokemon(species="Chansey", moves=("Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled")),
        Pokemon(species="Exeggutor", moves=("Psychic", "Sleep Powder", "Explosion", "Stun Spore")),
        Pokemon(species="Starmie", moves=("Thunderbolt", "Blizzard", "Thunder Wave", "Recover")),
        Pokemon(species="Alakazam", moves=("Psychic", "Seismic Toss", "Thunder Wave", "Recover")),
    ]


def test_pypkmn_battle():
    """Test pypkmn battle to see if legal actions become empty."""
    print("=" * 60)
    print("PyKMN Minimal Bug Reproduction Test")
    print("=" * 60)
    print()

    # Create teams
    print("Creating teams...")
    team1 = create_simple_team()
    team2 = create_simple_team()
    print(f"  Team 1: {len(team1)} Pokemon")
    print(f"  Team 2: {len(team2)} Pokemon")
    print()

    # Initialize battle
    print("Initializing battle...")
    battle = Battle(team1, team2)
    result, _ = battle.update_raw(0, 0)  # Team preview pass
    print(f"  Battle initialized, result type: {result.type()}")
    print()

    # Run battle loop
    print("Running battle loop (max 1000 steps)...")
    print()

    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        # Get legal choices for both players using pypkmn API
        # Convert cdata to list for easier handling
        legal_p1 = list(battle.possible_choices_raw(Player.P1, result))
        legal_p2 = list(battle.possible_choices_raw(Player.P2, result))

        # Check for bug: empty legal actions while battle not finished
        if len(legal_p1) == 0 or len(legal_p2) == 0:
            print(f"\n{'!' * 60}")
            print(f"BUG REPRODUCED at step {step_count}!")
            print(f"{'!' * 60}")
            print(f"P1 legal choices: {legal_p1}")
            print(f"P2 legal choices: {legal_p2}")
            print(f"Battle result type: {result.type()}")
            print(f"Battle finished: {result.type() != ResultType.NONE}")
            print()

            if result.type() == ResultType.NONE:
                print("ERROR: Battle not finished but no legal actions available!")
                return False
            else:
                print("Battle finished normally (legal actions empty is expected)")
                return True

        # Log every 50 steps
        if step_count % 50 == 0:
            print(f"  Step {step_count:4d}: P1_legal={len(legal_p1)} choices, P2_legal={len(legal_p2)} choices")

        # Pick random legal actions
        choice_p1 = random.choice(legal_p1)
        choice_p2 = random.choice(legal_p2)

        # Update battle
        result, _ = battle.update_raw(choice_p1, choice_p2)

        # Check if battle finished
        if result.type() != ResultType.NONE:
            print()
            print(f"Battle completed normally at step {step_count}")
            print(f"Result: {result.type()}")
            return True

        step_count += 1

    # Reached max steps
    print()
    print(f"WARNING: Battle reached max steps ({max_steps}) without finishing")
    print(f"Final result type: {result.type()}")
    print(f"P1 legal choices: {battle.legal_choices(Player.P1)}")
    print(f"P2 legal choices: {battle.legal_choices(Player.P2)}")
    return False


def test_multiple_battles(num_battles=10):
    """Run multiple battles to see if bug is consistent."""
    print("\n" + "=" * 60)
    print(f"Running {num_battles} battles to check consistency...")
    print("=" * 60)
    print()

    successes = 0
    failures = 0

    for i in range(num_battles):
        print(f"\nBattle {i+1}/{num_battles}:")
        print("-" * 40)

        team1 = create_simple_team()
        team2 = create_simple_team()
        battle = Battle(team1, team2)
        result, _ = battle.update_raw(0, 0)

        step_count = 0
        max_steps = 1000
        bug_found = False

        while step_count < max_steps:
            legal_p1 = list(battle.possible_choices_raw(Player.P1, result))
            legal_p2 = list(battle.possible_choices_raw(Player.P2, result))

            if len(legal_p1) == 0 or len(legal_p2) == 0:
                if result.type() == ResultType.NONE:
                    print(f"  BUG at step {step_count}: No legal actions but battle not finished")
                    bug_found = True
                    failures += 1
                else:
                    print(f"  Completed normally at step {step_count}")
                    successes += 1
                break

            choice_p1 = random.choice(legal_p1)
            choice_p2 = random.choice(legal_p2)
            result, _ = battle.update_raw(choice_p1, choice_p2)

            if result.type() != ResultType.NONE:
                print(f"  Completed normally at step {step_count}")
                successes += 1
                break

            step_count += 1

        if step_count >= max_steps:
            print(f"  TIMEOUT at step {step_count}")
            failures += 1

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Successes: {successes}/{num_battles}")
    print(f"  Failures:  {failures}/{num_battles}")
    print("=" * 60)

    return failures == 0


if __name__ == "__main__":
    print("\nTest 1: Single battle detailed trace")
    print("=" * 60)
    success = test_pypkmn_battle()

    print("\n" + "=" * 60)
    print("Test 2: Multiple battles")
    print("=" * 60)
    all_success = test_multiple_battles(num_battles=10)

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if all_success:
        print("  ✓ All battles completed successfully")
        print("  → Bug is likely in metamon wrapper, not pypkmn")
    else:
        print("  ✗ Bug reproduced in pypkmn directly")
        print("  → This is an upstream pypkmn issue")
    print("=" * 60)
