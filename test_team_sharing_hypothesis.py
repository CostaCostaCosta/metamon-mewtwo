"""
Test if team sharing causes heap corruption.

Hypothesis: Sharing Pokemon team objects between multiple Battle instances
causes double-free errors during cleanup.
"""
import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

import gc
import time
from pykmn.engine.gen1 import Battle, Pokemon

def create_team():
    return [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]

def run_battles(battles, steps=100):
    """Run battles for N steps."""
    for step in range(steps):
        for battle in battles:
            result, _ = battle.update_raw(1, 1)  # Simple moves
    return True

print("=" * 70)
print("TEAM SHARING CORRUPTION TEST")
print("=" * 70)

# Test 1: SHARED TEAMS (hypothesis: will crash)
print("\nTest 1: SHARED teams (64 battles, same team)")
print("-" * 70)
try:
    team = create_team()
    battles = [Battle(p1_team=team, p2_team=team) for _ in range(64)]
    print(f"  Created 64 battles with SHARED team (id={id(team)})")
    
    run_battles(battles, steps=100)
    print(f"  Ran 100 steps OK")
    
    # Force cleanup
    print(f"  Cleaning up...")
    for i, b in enumerate(battles):
        del battles[i]
    battles.clear()
    del battles
    del team
    gc.collect()
    
    print("  ✅ Test 1 PASSED (no crash)")
    test1_passed = True
except Exception as e:
    print(f"  ❌ Test 1 CRASHED: {e}")
    test1_passed = False

time.sleep(1)

# Test 2: UNIQUE TEAMS (hypothesis: will pass)
print("\nTest 2: UNIQUE teams (64 battles, different teams)")
print("-" * 70)
try:
    battles = [Battle(p1_team=create_team(), p2_team=create_team()) for _ in range(64)]
    print(f"  Created 64 battles with UNIQUE teams")
    
    run_battles(battles, steps=100)
    print(f"  Ran 100 steps OK")
    
    # Force cleanup
    print(f"  Cleaning up...")
    for i in range(len(battles)):
        del battles[i]
    battles.clear()
    del battles
    gc.collect()
    
    print("  ✅ Test 2 PASSED (no crash)")
    test2_passed = True
except Exception as e:
    print(f"  ❌ Test 2 CRASHED: {e}")
    test2_passed = False

time.sleep(1)

# Test 3: EXTREME - Many battles with shared team
print("\nTest 3: EXTREME shared teams (128 battles, same team)")
print("-" * 70)
try:
    team = create_team()
    battles = [Battle(p1_team=team, p2_team=team) for _ in range(128)]
    print(f"  Created 128 battles with SHARED team")
    
    run_battles(battles, steps=50)
    print(f"  Ran 50 steps OK")
    
    # Force cleanup
    print(f"  Cleaning up...")
    del battles
    del team
    gc.collect()
    
    print("  ✅ Test 3 PASSED (no crash)")
    test3_passed = True
except Exception as e:
    print(f"  ❌ Test 3 CRASHED: {e}")
    test3_passed = False

# Summary
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Test 1 (Shared 64):  {'PASS' if test1_passed else 'FAIL'}")
print(f"Test 2 (Unique 64):  {'PASS' if test2_passed else 'FAIL'}")
print(f"Test 3 (Shared 128): {'PASS' if test3_passed else 'FAIL'}")

if not test1_passed and test2_passed:
    print("\n🎯 HYPOTHESIS CONFIRMED: Team sharing causes corruption!")
    print("   Shared teams crash, unique teams work.")
elif test1_passed and test2_passed:
    print("\n❓ HYPOTHESIS UNCERTAIN: Both patterns work in this test")
    print("   Corruption may require more specific conditions")
else:
    print("\n⚠️  UNEXPECTED: Check test logic")
