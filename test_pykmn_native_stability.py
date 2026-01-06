"""
Test pure PyKMN (no metamon) for 1000 battles to isolate if corruption is in PyKMN itself.
"""
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

print("Testing pure PyKMN for 1000 battles...")
battles_completed = 0
start_time = time.time()

try:
    while battles_completed < 1000:
        team1 = create_team()
        team2 = create_team()
        
        battle = Battle(p1_team=team1, p2_team=team2)
        result, _ = battle.update_raw(0, 0)  # Initial pass
        
        steps = 0
        while result.type() == 0 and steps < 500:  # ResultType.NONE == 0
            # Random choices (simplified)
            choice_p1 = 1  # Move 1
            choice_p2 = 1  # Move 1
            result, _ = battle.update_raw(choice_p1, choice_p2)
            steps += 1
        
        battles_completed += 1
        
        # Clear battle
        del battle
        
        if battles_completed % 100 == 0:
            elapsed = time.time() - start_time
            rate = battles_completed / elapsed
            print(f"  {battles_completed} battles, {rate:.1f} battles/sec")
            gc.collect()
    
    print(f"\n✅ SUCCESS: Completed {battles_completed} battles")
    elapsed = time.time() - start_time
    print(f"   Time: {elapsed:.1f}s, Rate: {battles_completed/elapsed:.1f} battles/sec")

except Exception as e:
    print(f"\n❌ CRASH at battle {battles_completed}")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
