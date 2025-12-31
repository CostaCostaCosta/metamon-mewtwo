"""
Test pypkmn with PASS handling (mirrors vector_env logic).
"""

import os
import numpy as np

os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from pykmn.engine.gen1 import Battle, Player
from pykmn.engine.common import ResultType
from metamon.env.pykmn.team_parser import parse_showdown_team
from metamon.env.pykmn.action_mapper import (
    ActionMappings,
    get_legal_mask,
    metamon_action_to_choice,
)
from metamon.env.pykmn.features import precompute_mappings


def test_with_pass_handling():
    """Test using vector_env logic with PASS handling."""
    print("=" * 60)
    print("Testing PyKMN with PASS Handling")
    print("=" * 60)

    # Load teams
    team_dir = "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou"
    team_files = [
        os.path.join(team_dir, f)
        for f in os.listdir(team_dir)
        if f.endswith(".gen1ou_team")
    ][:2]

    print("\nLoading teams...")
    teams = [parse_showdown_team(open(f).read()) for f in team_files]

    # Create mappings
    mappings = precompute_mappings()
    action_mappings = ActionMappings.create()

    # Create battle
    print("Creating battle...")
    battle = Battle(teams[0], teams[1])
    result, _ = battle.update_raw(0, 0)  # Team preview

    print("Running battle (max 1000 steps)...\n")

    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        # Extract legal masks
        mask_p1 = get_legal_mask(battle, result, Player.P1, action_mappings)
        mask_p2 = get_legal_mask(battle, result, Player.P2, action_mappings)

        legal_p1 = np.where(mask_p1)[0]
        legal_p2 = np.where(mask_p2)[0]

        # Handle PASS (like vector_env does)
        if not mask_p1.any():
            # P1 must PASS
            choice_p1 = 0
        else:
            # P1 chooses random legal action
            action_p1 = np.random.choice(legal_p1)
            choice_p1 = metamon_action_to_choice(action_p1, action_mappings)

        if not mask_p2.any():
            # P2 must PASS
            choice_p2 = 0
        else:
            # P2 chooses random legal action
            action_p2 = np.random.choice(legal_p2)
            choice_p2 = metamon_action_to_choice(action_p2, action_mappings)

        # Log every 50 steps
        if step_count % 50 == 0:
            print(
                f"  Step {step_count:4d}: P1={len(legal_p1)} legal (choice={choice_p1}), "
                f"P2={len(legal_p2)} legal (choice={choice_p2})"
            )

        # Step battle
        result, _ = battle.update_raw(choice_p1, choice_p2)

        # Check if done
        if result.type() != ResultType.NONE:
            print(f"\nBattle completed normally at step {step_count}")
            print(f"Result: {result.type()}")
            return True

        step_count += 1

    # Timeout
    print(f"\nWARNING: Battle timeout at {max_steps} steps")
    return False


def run_multiple_battles(num_battles=10):
    """Run multiple battles with PASS handling."""
    print("\n" + "=" * 60)
    print(f"Running {num_battles} battles...")
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

        import random
        selected = random.sample(team_files, 2)
        teams = [parse_showdown_team(open(f).read()) for f in selected]

        mappings = precompute_mappings()
        action_mappings = ActionMappings.create()

        battle = Battle(teams[0], teams[1])
        result, _ = battle.update_raw(0, 0)

        step_count = 0
        max_steps = 1000

        while step_count < max_steps:
            mask_p1 = get_legal_mask(battle, result, Player.P1, action_mappings)
            mask_p2 = get_legal_mask(battle, result, Player.P2, action_mappings)

            legal_p1 = np.where(mask_p1)[0]
            legal_p2 = np.where(mask_p2)[0]

            # Handle PASS
            if not mask_p1.any():
                choice_p1 = 0
            else:
                action_p1 = np.random.choice(legal_p1)
                choice_p1 = metamon_action_to_choice(action_p1, action_mappings)

            if not mask_p2.any():
                choice_p2 = 0
            else:
                action_p2 = np.random.choice(legal_p2)
                choice_p2 = metamon_action_to_choice(action_p2, action_mappings)

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
    print("\nTest 1: Single battle with PASS handling")
    success = test_with_pass_handling()

    print("\n" + "=" * 60)
    print("Test 2: Multiple battles")
    all_success = run_multiple_battles(10)

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if all_success:
        print("  ✓ All battles completed successfully!")
        print("  ✓ PyKMN wrapper is working correctly!")
    else:
        print("  ✗ Some battles failed")
    print("=" * 60)
