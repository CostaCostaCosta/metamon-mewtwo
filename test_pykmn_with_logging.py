"""
Test pypkmn with detailed logging to find where legal actions become empty.
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
from metamon.env.pykmn.features import precompute_mappings, pykmn_to_features_raw


def test_with_wrapper_logic():
    """Test using the same logic as the wrapper to see where it fails."""
    print("=" * 60)
    print("Testing PyKMN with Wrapper Logic")
    print("=" * 60)

    # Load two teams
    team_dir = "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou"
    team_files = [
        os.path.join(team_dir, f)
        for f in os.listdir(team_dir)
        if f.endswith(".gen1ou_team")
    ][:2]

    print("\nLoading teams...")
    teams = [parse_showdown_team(open(f).read()) for f in team_files]
    print(f"  Loaded {len(teams)} teams")

    # Create mappings (same as vector_env)
    mappings = precompute_mappings()
    action_mappings = ActionMappings.create()

    # Create battle
    print("\nCreating battle...")
    battle = Battle(teams[0], teams[1])
    result, _ = battle.update_raw(0, 0)  # Team preview
    print(f"  Battle created, initial result type: {result.type()}")

    # Run battle with same logic as vector_env
    print("\nRunning battle (max 1000 steps)...")
    print()

    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        # Extract legal masks using wrapper's get_legal_mask function
        mask_p1 = get_legal_mask(battle, result, Player.P1, action_mappings)
        mask_p2 = get_legal_mask(battle, result, Player.P2, action_mappings)

        legal_indices_p1 = np.where(mask_p1)[0]
        legal_indices_p2 = np.where(mask_p2)[0]

        # Check for empty legal actions
        if len(legal_indices_p1) == 0 or len(legal_indices_p2) == 0:
            print("\n" + "!" * 60)
            print(f"BUG FOUND at step {step_count}!")
            print("!" * 60)
            print(f"P1 legal mask: {mask_p1}")
            print(f"P1 legal indices: {legal_indices_p1}")
            print(f"P2 legal mask: {mask_p2}")
            print(f"P2 legal indices: {legal_indices_p2}")
            print(f"Result type: {result.type()}")
            print(f"Battle finished: {result.type() != ResultType.NONE}")

            # Also check raw pypkmn API
            print("\nChecking raw pypkmn API:")
            raw_p1 = list(battle.possible_choices_raw(Player.P1, result))
            raw_p2 = list(battle.possible_choices_raw(Player.P2, result))
            print(f"  Raw P1 choices: {raw_p1}")
            print(f"  Raw P2 choices: {raw_p2}")

            if result.type() == ResultType.NONE:
                print("\nERROR: Battle not finished but no legal actions!")
                print("This means get_legal_mask() has a bug.")
                return False
            else:
                print("\nBattle finished normally.")
                return True

        # Log every 50 steps
        if step_count % 50 == 0:
            print(
                f"  Step {step_count:4d}: P1={len(legal_indices_p1)} legal, "
                f"P2={len(legal_indices_p2)} legal, result={result.type()}"
            )

        # Pick random legal actions
        action_idx_p1 = np.random.choice(legal_indices_p1)
        action_idx_p2 = np.random.choice(legal_indices_p2)

        # Convert to pypkmn choices
        choice_p1 = metamon_action_to_choice(action_idx_p1, action_mappings)
        choice_p2 = metamon_action_to_choice(action_idx_p2, action_mappings)

        # Step battle
        result, _ = battle.update_raw(choice_p1, choice_p2)

        # Check if done
        if result.type() != ResultType.NONE:
            print()
            print(f"Battle completed normally at step {step_count}")
            print(f"Result: {result.type()}")
            return True

        step_count += 1

    # Timeout
    print()
    print(f"WARNING: Battle reached {max_steps} steps without finishing")
    print(f"Final result type: {result.type()}")
    mask_p1 = get_legal_mask(battle, result, Player.P1, action_mappings)
    mask_p2 = get_legal_mask(battle, result, Player.P2, action_mappings)
    print(f"P1 legal actions: {np.where(mask_p1)[0]}")
    print(f"P2 legal actions: {np.where(mask_p2)[0]}")
    return False


def test_multiple_battles(num_battles=10):
    """Run multiple battles with wrapper logic."""
    print("\n" + "=" * 60)
    print(f"Running {num_battles} battles with wrapper logic...")
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
        print("-" * 40)

        # Load random teams
        import random
        selected_files = random.sample(team_files, 2)
        teams = [parse_showdown_team(open(f).read()) for f in selected_files]

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

            if len(legal_p1) == 0 or len(legal_p2) == 0:
                if result.type() == ResultType.NONE:
                    print(f"  BUG at step {step_count}: No legal actions")
                    failures += 1
                else:
                    print(f"  Completed at step {step_count}")
                    successes += 1
                break

            action_p1 = np.random.choice(legal_p1)
            action_p2 = np.random.choice(legal_p2)

            choice_p1 = metamon_action_to_choice(action_p1, action_mappings)
            choice_p2 = metamon_action_to_choice(action_p2, action_mappings)

            result, _ = battle.update_raw(choice_p1, choice_p2)

            if result.type() != ResultType.NONE:
                print(f"  Completed at step {step_count}")
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
    print("\nTest 1: Single battle with detailed trace")
    success = test_with_wrapper_logic()

    print("\n" + "=" * 60)
    print("Test 2: Multiple battles")
    all_success = test_multiple_battles(num_battles=10)

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if all_success:
        print("  ✓ All battles completed successfully with wrapper logic")
        print("  → Bug is NOT in action_mapper or get_legal_mask")
    else:
        print("  ✗ Bug reproduced with wrapper logic")
        print("  → Bug IS in action_mapper or get_legal_mask")
    print("=" * 60)
