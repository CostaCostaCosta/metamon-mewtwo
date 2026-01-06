#!/usr/bin/env python3
"""
Mode 1: Raw PyKMN Stepping (No Feature Extraction)
Purpose: Test if PyKMN binding crashes without our conversion code
Tests:
  1A: Fixed battles, long stepping
  1B: Create/destroy loop (catches pool/destructor bugs)
"""

import gc
import random
import sys
import traceback
import time
from typing import List, Tuple
import numpy as np

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Player, Choice, MoveChoice, PassChoice, SwitchChoice
from metamon.data.team_repo import load_teams_for_format


def get_legal_action(battle: Battle, player: Player) -> Choice:
    """Get a random legal action for the player."""
    legal_moves = []
    for slot in range(4):
        if battle.possible(player, MoveChoice(slot)):
            legal_moves.append(MoveChoice(slot))

    legal_switches = []
    for slot in range(1, 6):
        if battle.possible(player, SwitchChoice(slot)):
            legal_switches.append(SwitchChoice(slot))

    all_legal = legal_moves + legal_switches
    if not all_legal:
        return PassChoice()

    return random.choice(all_legal)


def run_short_episode(battle: Battle, max_steps: int = 50) -> int:
    """Run a battle for a short episode."""
    steps = 0
    while steps < max_steps:
        c1 = get_legal_action(battle, Player.P1)
        c2 = get_legal_action(battle, Player.P2)
        result, _ = battle.update(c1, c2)
        steps += 1
        if result.type() != 0:  # Terminal
            break
    return steps


def test_1a_fixed_battles(batch_sizes: List[int], steps_per_size: int = 10000) -> dict:
    """Test 1A: Fixed battles, long stepping."""
    print("\nTest 1A: Fixed battles, long stepping")
    print("-" * 40)

    results = {}
    teams = load_teams_for_format("gen1ou")

    for batch_size in batch_sizes:
        print(f"\n  Testing batch_size={batch_size}...")

        # Create fixed battles
        battles = []
        for i in range(batch_size):
            team_idx = i % len(teams)
            battle = Battle(
                p1_team=teams[team_idx].to_pykmn(),
                p2_team=teams[team_idx].to_pykmn(),
                p1_seed=42 + i,
                p2_seed=142 + i
            )
            battles.append(battle)

        # Run steps
        try:
            start_time = time.time()
            for step in range(steps_per_size):
                for b in battles:
                    c1 = get_legal_action(b, Player.P1)
                    c2 = get_legal_action(b, Player.P2)
                    result, _ = b.update(c1, c2)

                    # Reset if terminal
                    if result.type() != 0:
                        team_idx = random.randint(0, len(teams) - 1)
                        b.__init__(
                            p1_team=teams[team_idx].to_pykmn(),
                            p2_team=teams[team_idx].to_pykmn(),
                            p1_seed=random.randint(0, 1000000),
                            p2_seed=random.randint(0, 1000000)
                        )

                if step % 1000 == 0 and step > 0:
                    elapsed = time.time() - start_time
                    rate = (step * batch_size) / elapsed
                    print(f"    Step {step}/{steps_per_size}, {rate:.1f} updates/sec")

            results[batch_size] = "PASS"
            print(f"  ✓ batch_size={batch_size} PASSED")

        except Exception as e:
            results[batch_size] = f"FAIL: {str(e)}"
            print(f"  ✗ batch_size={batch_size} FAILED at step {step}")
            print(f"    Error: {e}")
            if batch_size >= 128:
                print("    Note: Failed at/above 128 boundary!")

    return results


def test_1b_create_destroy(iterations: int = 100000) -> bool:
    """Test 1B: Create/destroy battles repeatedly."""
    print("\nTest 1B: Create/destroy loop")
    print("-" * 40)

    teams = load_teams_for_format("gen1ou")

    try:
        start_time = time.time()
        for i in range(iterations):
            # Create battle
            team_idx = i % len(teams)
            battle = Battle(
                p1_team=teams[team_idx].to_pykmn(),
                p2_team=teams[team_idx].to_pykmn(),
                p1_seed=42 + i,
                p2_seed=142 + i
            )

            # Run short episode
            run_short_episode(battle, max_steps=50)

            # Explicit deletion
            del battle

            # Periodic GC and progress
            if i % 1000 == 0:
                gc.collect()
                if i > 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    print(f"  Iteration {i}/{iterations}, {rate:.1f} battles/sec")

            # Check around 128 boundary
            if i in [127, 128, 129]:
                print(f"    Passed iteration {i} (128 boundary check)")

        print(f"\n  ✓ Test 1B PASSED: {iterations} create/destroy cycles")
        return True

    except Exception as e:
        print(f"\n  ✗ Test 1B FAILED at iteration {i}")
        print(f"    Error: {e}")
        if i >= 126 and i <= 130:
            print("    Note: Failed near 128 boundary!")
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("PyKMN Stability Test - Mode 1: Raw PyKMN Stepping")
    print("=" * 60)

    # Set deterministic seeds
    random.seed(42)
    np.random.seed(42)

    # Test batch sizes straddling 128
    batch_sizes = [1, 64, 96, 112, 127, 128, 129, 144, 256]

    print("\nTesting multiple batch sizes to find failure threshold...")

    # Run Test 1A
    test_1a_results = test_1a_fixed_battles(batch_sizes, steps_per_size=5000)

    # Run Test 1B
    test_1b_success = test_1b_create_destroy(iterations=10000)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nTest 1A Results (Fixed Battles):")
    for batch_size, result in test_1a_results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} batch_size={batch_size}: {result}")

    print(f"\nTest 1B Result (Create/Destroy):")
    status = "✓" if test_1b_success else "✗"
    print(f"  {status} {'PASS' if test_1b_success else 'FAIL'}")

    # Determine overall result
    all_passed = all(r == "PASS" for r in test_1a_results.values()) and test_1b_success

    if all_passed:
        print("\n✓ Mode 1 PASSED: PyKMN stepping is stable without feature extraction")
        sys.exit(0)
    else:
        # Find failure threshold
        for batch_size in batch_sizes:
            if test_1a_results.get(batch_size, "").startswith("FAIL"):
                print(f"\n✗ Mode 1 FAILED: First failure at batch_size={batch_size}")
                if batch_size == 128:
                    print("  CRITICAL: Failure at exactly 128 - likely hardcoded buffer limit")
                break
        sys.exit(1)


if __name__ == "__main__":
    main()