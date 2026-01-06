"""
Test PyKMNVectorEnv but WITHOUT observation space extraction.
This isolates if the corruption is in features.py.
"""
import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

import gc
import time
import numpy as np
from pykmn.engine.gen1 import Battle, Pokemon
from pykmn.engine.common import Result, Player

# Minimal test - just create battles and step them, no observation extraction
def create_team():
    return [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]

print("Testing vectorized battles (no observation extraction)...")
num_envs = 64
battles_completed = 0
target_battles = 1000
start_time = time.time()

try:
    while battles_completed < target_battles:
        # Create batch of battles
        battles = []
        for i in range(num_envs):
            team1 = create_team()
            team2 = create_team()
            battle = Battle(p1_team=team1, p2_team=team2)
            result, _ = battle.update_raw(0, 0)  # Initial pass
            battles.append((battle, result))
        
        # Run battles until all finish
        all_done = False
        steps = 0
        while not all_done and steps < 500:
            all_done = True
            for i in range(num_envs):
                battle, result = battles[i]
                if result.type() == 0:  # Not done
                    result, _ = battle.update_raw(1, 1)  # Random moves
                    battles[i] = (battle, result)
                    all_done = False
            steps += 1
        
        # Count completions
        completed_in_batch = sum(1 for _, result in battles if result.type() != 0)
        battles_completed += completed_in_batch
        
        # Clear battles
        for battle, _ in battles:
            del battle
        battles.clear()
        
        if battles_completed % (num_envs * 5) == 0:
            elapsed = time.time() - start_time
            rate = battles_completed / elapsed
            print(f"  {battles_completed}/{target_battles} battles, {rate:.1f} battles/sec")
            gc.collect()
    
    print(f"\n✅ SUCCESS: Completed {battles_completed} battles")
    elapsed = time.time() - start_time
    print(f"   Time: {elapsed:.1f}s, Rate: {battles_completed/elapsed:.1f} battles/sec")

except Exception as e:
    print(f"\n❌ CRASH at battle {battles_completed}")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
