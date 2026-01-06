#!/usr/bin/env python3
"""
Test rapid Battle object creation/destruction (churn test)
This simulates the worst case: constantly creating new Battle objects
"""

import gc
import sys
import time
import traceback
import psutil
import os

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Pokemon
from pykmn.engine.common import Player, ResultType


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_simple_team():
    """Create a simple hardcoded Gen1 team."""
    team = [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]
    return team


def churn_test(target_objects: int = 10000):
    """Rapidly create and destroy Battle objects."""
    print(f"\n{'='*70}")
    print(f"CHURN TEST: Creating {target_objects} Battle objects rapidly")
    print(f"{'='*70}")

    team = create_simple_team()
    created = 0
    destroyed = 0
    initial_memory = get_memory_usage()
    start_time = time.time()

    # Keep some battles alive to test mixed scenarios
    live_battles = []
    MAX_LIVE = 128  # Keep up to 128 alive at once

    try:
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Strategy: Keep up to {MAX_LIVE} battles alive, churn the rest\n")

        for i in range(target_objects):
            # Create new battle
            battle = Battle(p1_team=team, p2_team=team)
            result, _ = battle.update_raw(0, 0)
            created += 1

            # Run a few steps
            for _ in range(5):
                p1_choices = battle.possible_choices_raw(Player.P1, result)
                p2_choices = battle.possible_choices_raw(Player.P2, result)
                c1 = p1_choices[0] if p1_choices else 0
                c2 = p2_choices[0] if p2_choices else 0
                result, _ = battle.update_raw(c1, c2)
                if result.type() != ResultType.NONE:
                    break

            # Keep some alive, destroy others
            if len(live_battles) < MAX_LIVE:
                live_battles.append(battle)
            else:
                # Replace oldest
                old_battle = live_battles.pop(0)
                del old_battle
                destroyed += 1
                live_battles.append(battle)

            # Progress report
            if i % 100 == 0 and i > 0:
                current_memory = get_memory_usage()
                memory_delta = current_memory - initial_memory
                elapsed = time.time() - start_time
                rate = created / elapsed if elapsed > 0 else 0

                print(f"Progress: {created}/{target_objects} created, {destroyed} destroyed")
                print(f"  Live battles: {len(live_battles)}")
                print(f"  Memory: {current_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
                print(f"  Rate: {rate:.1f} objects/sec")

                # Check critical boundaries
                if created in [127, 128, 129]:
                    print(f"  ⚠️  At {created} - watching for 128 barrier crash...")
                if created in [255, 256, 257]:
                    print(f"  ⚠️  At {created} - watching for 256 barrier crash...")

            # Aggressive GC every 50 objects
            if i % 50 == 0:
                gc.collect()

        # Clean up remaining
        for battle in live_battles:
            del battle
            destroyed += 1
        live_battles.clear()
        gc.collect()

        # Final stats
        elapsed = time.time() - start_time
        final_memory = get_memory_usage()
        memory_growth = final_memory - initial_memory

        print(f"\n{'='*70}")
        print(f"✅ CHURN TEST PASSED")
        print(f"{'='*70}")
        print(f"  Objects created: {created}")
        print(f"  Objects destroyed: {destroyed}")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Rate: {created/elapsed:.1f} objects/sec")
        print(f"  Memory: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_growth:+.1f} MB)")
        print(f"  Memory leak: {memory_growth:.1f} MB for {created} objects")
        print(f"  Leak per object: {memory_growth/created*1024:.1f} KB" if created > 0 else "")

        return True, created

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"💥 CHURN TEST FAILED")
        print(f"{'='*70}")
        print(f"  Created: {created} objects")
        print(f"  Destroyed: {destroyed} objects")
        print(f"  Live at crash: {len(live_battles)}")
        print(f"  Error: {e}")

        # Check if it's the 128 barrier
        if created >= 126 and created <= 130:
            print(f"\n🎯 128 BARRIER CONFIRMED! Crashed at exactly {created} objects!")
        elif created >= 254 and created <= 258:
            print(f"\n🎯 256 BARRIER CONFIRMED! Crashed at exactly {created} objects!")

        print(f"\nFull traceback:")
        traceback.print_exc()

        return False, created


def parallel_battles_test(batch_size: int = 200):
    """Test many battles running in parallel."""
    print(f"\n{'='*70}")
    print(f"PARALLEL TEST: {batch_size} simultaneous battles")
    print(f"{'='*70}")

    team = create_simple_team()
    initial_memory = get_memory_usage()

    try:
        print(f"Creating {batch_size} battles simultaneously...")

        battles = []
        results = []

        # Create all battles at once
        for i in range(batch_size):
            battle = Battle(p1_team=team, p2_team=team)
            result, _ = battle.update_raw(0, 0)
            battles.append(battle)
            results.append(result)

            if i % 50 == 0:
                print(f"  Created {i}/{batch_size}...")

        print(f"✓ All {batch_size} battles created successfully!")

        # Run them for a while
        print(f"\nRunning all battles for 100 steps...")
        for step in range(100):
            for i, battle in enumerate(battles):
                result = results[i]
                p1_choices = battle.possible_choices_raw(Player.P1, result)
                p2_choices = battle.possible_choices_raw(Player.P2, result)

                c1 = p1_choices[0] if p1_choices else 0
                c2 = p2_choices[0] if p2_choices else 0

                result, _ = battle.update_raw(c1, c2)
                results[i] = result

                if result.type() != ResultType.NONE:
                    # Reset
                    battles[i] = Battle(p1_team=team, p2_team=team)
                    results[i] = battles[i].update_raw(0, 0)[0]

            if step % 20 == 0:
                current_memory = get_memory_usage()
                print(f"  Step {step}: Memory = {current_memory:.1f} MB")

        # Cleanup
        for battle in battles:
            del battle
        battles.clear()
        gc.collect()

        final_memory = get_memory_usage()
        print(f"\n✅ PARALLEL TEST PASSED with {batch_size} simultaneous battles!")
        print(f"  Memory: {initial_memory:.1f} MB → {final_memory:.1f} MB")

        return True

    except Exception as e:
        print(f"\n💥 PARALLEL TEST FAILED at batch_size={batch_size}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all churn and parallel tests."""
    print("=" * 70)
    print("PyKMN CHURN & PARALLEL TESTS")
    print("=" * 70)
    print("\nThese tests specifically target:")
    print("  1. Rapid object creation/destruction (memory leaks)")
    print("  2. Large numbers of simultaneous battles")
    print("  3. The reported 128/256 barriers")

    # Test 1: Churn test
    print("\n" + "=" * 70)
    print("TEST 1: RAPID CHURN (10000 objects)")
    success1, created = churn_test(10000)

    # Test 2: Parallel battles
    print("\n" + "=" * 70)
    print("TEST 2: PARALLEL BATTLES")

    # Try increasing batch sizes
    test_sizes = [100, 128, 150, 200, 256]
    max_success = 0

    for size in test_sizes:
        if parallel_battles_test(size):
            max_success = size
        else:
            print(f"  Failed at {size} parallel battles!")
            break

        gc.collect()
        time.sleep(1)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if success1:
        print(f"✅ Churn test: PASSED ({created} objects)")
    else:
        print(f"❌ Churn test: FAILED at {created} objects")

    if max_success > 0:
        print(f"✅ Parallel test: PASSED up to {max_success} battles")

    if success1 and max_success >= 128:
        print("\n🎉 PyKMN PASSED ALL STRESS TESTS!")
        print("No 128 barrier detected under current conditions.")
    else:
        print("\n⚠️  Some stability issues detected.")


if __name__ == "__main__":
    main()