"""
Test if observation space state management causes corruption.

Compare:
1. DefaultObservationSpace (no state)
2. ExpandedObservationSpace (with state)
"""
import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

import gc
import time
import numpy as np
from pykmn.engine.gen1 import Pokemon
from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.interface import DefaultObservationSpace, ExpandedObservationSpace, DefaultShapedReward

def create_team():
    return [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]

def run_test(obs_space_cls, obs_space_name, batch_size=64, target_battles=500):
    print(f"\n{'='*70}")
    print(f"TEST: {obs_space_name}, batch_size={batch_size}, target={target_battles}")
    print(f"{'='*70}")
    
    battles_completed = 0
    start_time = time.time()
    
    try:
        team = create_team()
        teams_p1 = [team] * batch_size
        teams_p2 = [team] * batch_size
        
        env = PyKMNVectorEnv(
            num_envs=batch_size,
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            obs_space=obs_space_cls(),
            reward_fn=DefaultShapedReward(),
            battle_format="gen1ou",
            track_trajectories=False,
        )
        
        print(f"✓ Environment created")
        
        obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()
        
        while battles_completed < target_battles:
            # Random legal actions
            actions_p1 = []
            actions_p2 = []
            
            for i in range(batch_size):
                legal_p1 = np.where(legal_masks_p1[i])[0]
                legal_p2 = np.where(legal_masks_p2[i])[0]
                
                action_p1 = np.random.choice(legal_p1) if len(legal_p1) > 0 else 0
                action_p2 = np.random.choice(legal_p2) if len(legal_p2) > 0 else 0
                
                actions_p1.append(action_p1)
                actions_p2.append(action_p2)
            
            actions_p1 = np.array(actions_p1)
            actions_p2 = np.array(actions_p2)
            
            # Step
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)
            
            # Get legal masks
            legal_masks_p1, legal_masks_p2 = env._extract_legal_masks()
            
            # Count completions
            if dones.any():
                battles_completed += dones.sum()
                
                # Progress
                if battles_completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = battles_completed / elapsed
                    print(f"  {battles_completed}/{target_battles} battles, {rate:.1f} battles/sec")
            
            # Periodic GC
            if battles_completed % 50 == 0:
                gc.collect(0)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS: {battles_completed} battles in {elapsed:.1f}s ({battles_completed/elapsed:.1f} battles/sec)")
        return True
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ FAILURE after {battles_completed} battles ({elapsed:.1f}s)")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'env' in locals():
            env.close()
            del env
        gc.collect()

# Run tests
print("=" * 70)
print("OBSERVATION SPACE STATE CORRUPTION TEST")
print("=" * 70)

# Test 1: DefaultObservationSpace (no state)
print("\n>>> Testing DefaultObservationSpace (no state management)")
success1 = run_test(DefaultObservationSpace, "DefaultObservationSpace", batch_size=64, target_battles=500)

time.sleep(2)
gc.collect()

# Test 2: ExpandedObservationSpace (with state)
print("\n>>> Testing ExpandedObservationSpace (with state management)")
success2 = run_test(ExpandedObservationSpace, "ExpandedObservationSpace", batch_size=64, target_battles=500)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"DefaultObservationSpace: {'PASS' if success1 else 'FAIL'}")
print(f"ExpandedObservationSpace: {'PASS' if success2 else 'FAIL'}")

if success1 and not success2:
    print("\n🔴 SMOKING GUN: Observation state management causes corruption!")
elif not success1 and not success2:
    print("\n⚠️  Both failed - corruption is elsewhere")
elif success1 and success2:
    print("\n✅ Both passed - cannot reproduce corruption")
