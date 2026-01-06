#!/usr/bin/env python3
"""
Narrow down exact batch size where PyKMN crashes
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


def test_batch_size(batch_size: int, steps: int = 100) -> bool:
    """Test a specific batch size."""
    print(f"Testing batch_size={batch_size}...", flush=True)

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

        print(f"  ✓ batch_size={batch_size} PASSED", flush=True)
        return True

    except Exception as e:
        print(f"  ✗ batch_size={batch_size} FAILED: {e}", flush=True)
        return False


def main():
    """Binary search to find exact failure point."""
    print("=" * 60)
    print("NARROWING PyKMN CRASH THRESHOLD")
    print("=" * 60)
    print()

    # We know 32 works and 64 crashes, so test in between
    test_sizes = [32, 40, 48, 50, 52, 54, 56, 58, 60, 62, 63, 64, 65, 66, 68, 70, 72, 80, 96, 128]

    last_working = None
    first_failure = None

    for size in test_sizes:
        success = test_batch_size(size, steps=50)

        if success:
            last_working = size
        else:
            first_failure = size
            print(f"\n✗ CRASH AT batch_size={size}")
            break

        # Force GC between tests
        gc.collect()
        time.sleep(0.1)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if last_working:
        print(f"✓ Last working batch size: {last_working}")
    if first_failure:
        print(f"✗ First failure batch size: {first_failure}")

        if first_failure == 64:
            print("\nDIAGNOSIS: Failure at exactly 64 (power of 2)")
            print("This suggests a fixed-size buffer or alignment issue")
        elif first_failure == 128:
            print("\nDIAGNOSIS: Failure at exactly 128")
            print("Classic hardcoded buffer limit")


if __name__ == "__main__":
    main()