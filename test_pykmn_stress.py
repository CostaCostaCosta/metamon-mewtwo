#!/usr/bin/env python3
"""
Stress test PyKMN with longer runs to trigger crashes
The crash seems to happen after some time/memory accumulation
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
    team = [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]
    return team


def stress_test(batch_size: int, total_steps: int = 10000) -> bool:
    """Run a longer stress test."""
    print(f"\nStress testing batch_size={batch_size} for {total_steps} steps...", flush=True)

    try:
        team = create_simple_team()
        battles = []
        results = []
        battle_count = 0  # Track total battles created

        # Create initial battles
        for i in range(batch_size):
            b = Battle(p1_team=team, p2_team=team)
            result, _ = b.update_raw(0, 0)
            battles.append(b)
            results.append(result)
            battle_count += 1

        # Run many steps
        for step in range(total_steps):
            for i, battle in enumerate(battles):
                prev_result = results[i]
                p1_choices = battle.possible_choices_raw(Player.P1, prev_result)
                p2_choices = battle.possible_choices_raw(Player.P2, prev_result)

                c1 = p1_choices[0] if p1_choices else 0
                c2 = p2_choices[0] if p2_choices else 0

                result, _ = battle.update_raw(c1, c2)
                results[i] = result

                # Reset if terminal - this creates NEW Battle objects
                if result.type() != ResultType.NONE:
                    # Create new battle (replacing old one)
                    battles[i] = Battle(p1_team=team, p2_team=team)
                    results[i] = battles[i].update_raw(0, 0)[0]
                    battle_count += 1

            # Progress and memory info
            if step % 100 == 0:
                print(f"  Step {step}/{total_steps}, Total battles created: {battle_count}", flush=True)

                # Check around potential problem boundaries
                if battle_count >= 120 and battle_count <= 135:
                    print(f"    ⚠️  Approaching 128 battles (current: {battle_count})", flush=True)

            # Periodic GC (might affect crash timing)
            if step % 50 == 0:
                gc.collect()

        print(f"  ✓ Stress test PASSED: batch_size={batch_size}, {total_steps} steps, {battle_count} total battles", flush=True)
        return True

    except Exception as e:
        print(f"  ✗ Stress test FAILED at step {step}, battle_count={battle_count}", flush=True)
        print(f"    Error: {e}", flush=True)
        traceback.print_exc()
        return False


def main():
    """Run stress tests."""
    print("=" * 60)
    print("PyKMN STRESS TEST - Trigger memory/resource crashes")
    print("=" * 60)

    # Test different batch sizes with longer runs
    test_configs = [
        (1, 5000),    # Single battle, many resets
        (16, 2000),   # Medium batch
        (32, 1000),   # Larger batch
        (64, 500),    # Large batch
        (128, 250),   # Very large batch
    ]

    results = {}

    for batch_size, steps in test_configs:
        success = stress_test(batch_size, steps)
        results[batch_size] = success

        if not success:
            print(f"\n🔴 CRASH FOUND at batch_size={batch_size}")
            break

        # Cleanup between tests
        gc.collect()
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for batch_size, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: batch_size={batch_size}")

    # Analysis
    failures = [k for k, v in results.items() if not v]
    if failures:
        min_failure = min(failures)
        print(f"\n🔴 First failure at batch_size={min_failure}")

        if min_failure in [32, 64, 128]:
            print("  DIAGNOSIS: Power-of-2 failure suggests buffer limit")
        if any(r for r in results.values() if not r):
            print("  DIAGNOSIS: Crash after extended runtime suggests memory leak or resource exhaustion")
    else:
        print("\n✓ All stress tests passed - no crashes detected")


if __name__ == "__main__":
    main()