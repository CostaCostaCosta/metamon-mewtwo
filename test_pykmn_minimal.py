#!/usr/bin/env python3
"""
Minimal PyKMN test - identify the exact failure point
Tests PyKMN in increasing batch sizes to find the crash threshold
"""

import gc
import sys
import time
import traceback
from typing import List

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Pokemon
from pykmn.engine.common import Player, ResultType


def create_simple_team() -> List[Pokemon]:
    """Create a simple hardcoded Gen1 team."""
    # Create a basic team with common Gen1 Pokemon
    team = [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]
    return team


def test_single_battle(steps: int = 1000) -> bool:
    """Test a single battle."""
    print("Testing single battle...")
    try:
        team = create_simple_team()
        battle = Battle(p1_team=team, p2_team=team)

        # Initial update
        result, _ = battle.update_raw(0, 0)

        for step in range(steps):
            # Get legal choices (pass the previous result)
            p1_choices = battle.possible_choices_raw(Player.P1, result)
            p2_choices = battle.possible_choices_raw(Player.P2, result)

            # Pick random legal choice or PASS
            c1 = p1_choices[0] if p1_choices else 0
            c2 = p2_choices[0] if p2_choices else 0

            # Update
            result, _ = battle.update_raw(c1, c2)

            # Reset if terminal
            if result.type() != ResultType.NONE:
                battle = Battle(p1_team=team, p2_team=team)
                result, _ = battle.update_raw(0, 0)

        print(f"  ✓ Single battle PASSED ({steps} steps)")
        return True

    except Exception as e:
        print(f"  ✗ Single battle FAILED: {e}")
        traceback.print_exc()
        return False


def test_batch_size(batch_size: int, steps: int = 500) -> bool:
    """Test a specific batch size."""
    print(f"\nTesting batch_size={batch_size}...")

    try:
        # Create battles
        team = create_simple_team()
        battles = []
        for i in range(batch_size):
            b = Battle(p1_team=team, p2_team=team)
            # Initial update
            b.update_raw(0, 0)
            battles.append(b)

        # Track results for each battle
        results = [None] * batch_size

        # Run steps
        for step in range(steps):
            for i, battle in enumerate(battles):
                # Get legal choices (pass the previous result or None)
                prev_result = results[i] if results[i] is not None else battle.update_raw(0, 0)[0]
                p1_choices = battle.possible_choices_raw(Player.P1, prev_result)
                p2_choices = battle.possible_choices_raw(Player.P2, prev_result)

                # Pick first legal choice or PASS
                c1 = p1_choices[0] if p1_choices else 0
                c2 = p2_choices[0] if p2_choices else 0

                # Update
                result, _ = battle.update_raw(c1, c2)
                results[i] = result

                # Reset if terminal
                if result.type() != ResultType.NONE:
                    battles[i] = Battle(p1_team=team, p2_team=team)
                    results[i] = battles[i].update_raw(0, 0)[0]

            if step % 100 == 0 and step > 0:
                print(f"    Step {step}/{steps}")

        print(f"  ✓ batch_size={batch_size} PASSED")
        return True

    except Exception as e:
        print(f"  ✗ batch_size={batch_size} FAILED at step {step}: {e}")
        return False


def test_create_destroy(iterations: int = 1000) -> bool:
    """Test creating and destroying battles repeatedly."""
    print(f"\nTesting create/destroy ({iterations} iterations)...")

    try:
        team = create_simple_team()

        for i in range(iterations):
            # Create battle
            battle = Battle(p1_team=team, p2_team=team)
            result, _ = battle.update_raw(0, 0)

            # Run a few steps
            for _ in range(10):
                p1_choices = battle.possible_choices_raw(Player.P1, result)
                p2_choices = battle.possible_choices_raw(Player.P2, result)
                c1 = p1_choices[0] if p1_choices else 0
                c2 = p2_choices[0] if p2_choices else 0
                result, _ = battle.update_raw(c1, c2)
                if result.type() != ResultType.NONE:
                    break

            # Delete
            del battle

            if i % 100 == 0 and i > 0:
                gc.collect()
                print(f"    Iteration {i}/{iterations}")

            # Check around 128
            if i in [126, 127, 128, 129, 130]:
                print(f"    Passed iteration {i} (128 boundary)")

        print(f"  ✓ Create/destroy PASSED")
        return True

    except Exception as e:
        print(f"  ✗ Create/destroy FAILED at iteration {i}: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MINIMAL PyKMN STABILITY TEST")
    print("=" * 60)

    results = {}

    # Test 1: Single battle
    results['single'] = test_single_battle(steps=1000)

    # Test 2: Create/destroy
    results['create_destroy'] = test_create_destroy(iterations=500)

    # Test 3: Batch sizes
    batch_sizes = [1, 32, 64, 96, 127, 128, 129, 144, 256]
    for batch_size in batch_sizes:
        results[f'batch_{batch_size}'] = test_batch_size(batch_size, steps=200)

        # Stop if we hit a failure
        if not results[f'batch_{batch_size}']:
            print(f"\nStopping at first failure (batch_size={batch_size})")
            break

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    first_failure = None

    for key, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {key}")
        if not passed and first_failure is None:
            first_failure = key
            all_passed = False

    if all_passed:
        print("\n✓ ALL TESTS PASSED - PyKMN appears stable")
    else:
        print(f"\n✗ FIRST FAILURE: {first_failure}")

        # Diagnosis
        if first_failure == 'single':
            print("\nDIAGNOSIS: PyKMN unstable even for single battles")
            print("ACTION: Check PyKMN installation/version")
        elif first_failure == 'create_destroy':
            print("\nDIAGNOSIS: Memory management issue in create/destroy cycle")
            print("ACTION: Check for resource leaks or destructor bugs")
        elif 'batch_128' in first_failure:
            print("\nDIAGNOSIS: Hard limit at 128 battles")
            print("ACTION: Search for buffer size constants in PyKMN")
        elif 'batch_' in first_failure:
            batch_num = int(first_failure.split('_')[1])
            print(f"\nDIAGNOSIS: Failure at batch_size={batch_num}")
            print("ACTION: Memory or resource exhaustion")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()