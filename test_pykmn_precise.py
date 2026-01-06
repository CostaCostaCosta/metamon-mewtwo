#!/usr/bin/env python3
"""
Find the precise batch size where PyKMN fails
"""

import gc
import sys
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


def test_batch_size_minimal(batch_size: int) -> bool:
    """Test a specific batch size with minimal steps."""
    try:
        # Create battles
        team = create_simple_team()
        battles = []
        results = []

        for i in range(batch_size):
            b = Battle(p1_team=team, p2_team=team)
            # Initial update
            result, _ = b.update_raw(0, 0)
            battles.append(b)
            results.append(result)

        # Just one step to test
        for i, battle in enumerate(battles):
            prev_result = results[i]
            p1_choices = battle.possible_choices_raw(Player.P1, prev_result)
            p2_choices = battle.possible_choices_raw(Player.P2, prev_result)

            c1 = p1_choices[0] if p1_choices else 0
            c2 = p2_choices[0] if p2_choices else 0

            result, _ = battle.update_raw(c1, c2)
            results[i] = result

        return True

    except Exception as e:
        print(f"    Error at {batch_size}: {str(e)[:100]}", flush=True)
        return False


def binary_search_failure():
    """Binary search to find exact failure point."""
    low = 1
    high = 64
    last_working = 1
    first_failure = None

    while low <= high:
        mid = (low + high) // 2
        print(f"Testing batch_size={mid}...", flush=True)

        # Test this size
        gc.collect()
        success = test_batch_size_minimal(mid)

        if success:
            print(f"  ✓ {mid} works", flush=True)
            last_working = mid
            low = mid + 1
        else:
            print(f"  ✗ {mid} fails", flush=True)
            first_failure = mid
            high = mid - 1

    return last_working, first_failure


def linear_search_from(start: int, end: int):
    """Linear search from start to end."""
    for size in range(start, end + 1):
        print(f"Testing batch_size={size}...", flush=True)
        gc.collect()

        success = test_batch_size_minimal(size)

        if success:
            print(f"  ✓ {size} works", flush=True)
        else:
            print(f"  ✗ {size} FAILS - Found threshold!", flush=True)
            return size - 1, size

    return end, None


def main():
    """Find precise failure point."""
    print("=" * 60)
    print("FINDING PRECISE PyKMN CRASH THRESHOLD")
    print("=" * 60)
    print()

    # Start with small linear search
    print("Linear search from 1 to 40...")
    last_working, first_failure = linear_search_from(1, 40)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if last_working is not None:
        print(f"✓ Last working batch size: {last_working}")
    if first_failure is not None:
        print(f"✗ First failure batch size: {first_failure}")
        print(f"\nCRITICAL: PyKMN fails at exactly {first_failure} simultaneous battles")

        # Check for special values
        if first_failure in [16, 32, 64, 128, 256]:
            print(f"  This is a power of 2, suggesting a fixed buffer size")
        if first_failure % 8 == 0:
            print(f"  This is a multiple of 8, suggesting alignment issues")


if __name__ == "__main__":
    main()