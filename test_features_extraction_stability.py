"""
Test if the corruption is in pykmn_to_features_raw().
This calls the C++ accessor methods repeatedly.
"""
import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

import gc
import time
from pykmn.engine.gen1 import Battle, Pokemon
from pykmn.engine.common import Player
from metamon.env.pykmn.features import precompute_mappings, pykmn_to_features_raw

def create_team():
    return [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]

print("Testing features extraction for 10,000 calls...")
mappings = precompute_mappings()
calls_done = 0
target_calls = 10000
start_time = time.time()

try:
    # Create one battle and extract features many times
    team1 = create_team()
    team2 = create_team()
    battle = Battle(p1_team=team1, p2_team=team2)
    result, _ = battle.update_raw(0, 0)
    
    while calls_done < target_calls:
        # Extract features for both players
        features_p1 = pykmn_to_features_raw(battle, result, Player.P1, mappings)
        features_p2 = pykmn_to_features_raw(battle, result, Player.P2, mappings)
        
        # Step battle occasionally
        if calls_done % 100 == 0 and result.type() == 0:
            result, _ = battle.update_raw(1, 1)
        
        calls_done += 1
        
        if calls_done % 1000 == 0:
            elapsed = time.time() - start_time
            rate = calls_done / elapsed
            print(f"  {calls_done} feature extractions, {rate:.0f} calls/sec")
    
    print(f"\n✅ SUCCESS: Completed {calls_done} feature extractions")
    elapsed = time.time() - start_time
    print(f"   Time: {elapsed:.1f}s, Rate: {calls_done/elapsed:.0f} calls/sec")

except Exception as e:
    print(f"\n❌ CRASH at call {calls_done}")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
