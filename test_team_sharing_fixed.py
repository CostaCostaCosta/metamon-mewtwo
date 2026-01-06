"""
Test team sharing with proper cleanup.
"""
import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

import gc
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

print("=" * 70)
print("TEAM SHARING TEST (FIXED)")
print("=" * 70)

# Test 1: Shared teams
print("\nTest 1: SHARED teams (64 battles)")
team = create_team()
battles = [Battle(p1_team=team, p2_team=team) for _ in range(64)]
for step in range(100):
    for battle in battles:
        battle.update_raw(1, 1)
del battles
del team
gc.collect()
print("  ✅ PASSED")

# Test 2: Unique teams
print("\nTest 2: UNIQUE teams (64 battles)")
battles = [Battle(p1_team=create_team(), p2_team=create_team()) for _ in range(64)]
for step in range(100):
    for battle in battles:
        battle.update_raw(1, 1)
del battles
gc.collect()
print("  ✅ PASSED")

print("\nBoth patterns work! Team sharing is NOT the issue.")
