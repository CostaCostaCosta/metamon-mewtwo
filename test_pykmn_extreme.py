#!/usr/bin/env python3
"""
EXTREME stress test - push PyKMN to its absolute limits
Run many more battles, larger batches, longer episodes
"""

import gc
import sys
import time
import numpy as np
import traceback
import psutil
import os

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Pokemon
from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.interface import ExpandedObservationSpace, DefaultShapedReward


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


def run_extreme_test(batch_size: int, target_battles: int = 10000, max_time: int = 300):
    """Run EXTREME test - many battles, no mercy."""
    print(f"\n{'='*70}")
    print(f"EXTREME TEST: batch_size={batch_size}, target={target_battles} battles")
    print(f"{'='*70}")

    # Track metrics
    battles_completed = 0
    battles_created = 0  # Track total Battle objects created
    total_steps = 0
    start_time = time.time()
    initial_memory = get_memory_usage()

    try:
        # Create teams
        team = create_simple_team()
        teams_p1 = [team] * batch_size
        teams_p2 = [team] * batch_size

        # Create environment
        env = PyKMNVectorEnv(
            num_envs=batch_size,
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            obs_space=ExpandedObservationSpace(),
            reward_fn=DefaultShapedReward(),
            battle_format="gen1ou",
            track_trajectories=False,
        )
        battles_created = batch_size

        # Reset environment
        obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()
        print(f"✓ Environment created with batch_size={batch_size}")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print()

        # Run until we hit target battles or timeout
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

            actions_p1 = np.array(actions_p1, dtype=np.int32)
            actions_p2 = np.array(actions_p2, dtype=np.int32)

            # Step environment
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)

            # Update legal masks
            legal_masks_p1 = obs_p1.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))
            legal_masks_p2 = obs_p2.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))

            total_steps += 1

            # Count completed battles (these trigger resets internally)
            new_completions = dones.sum()
            if new_completions > 0:
                battles_completed += new_completions
                battles_created += new_completions  # Each reset creates a new Battle

            # Detailed progress report
            if total_steps % 50 == 0:
                current_memory = get_memory_usage()
                memory_delta = current_memory - initial_memory
                elapsed = time.time() - start_time
                battles_per_sec = battles_completed / elapsed if elapsed > 0 else 0
                steps_per_sec = total_steps / elapsed if elapsed > 0 else 0

                print(f"Step {total_steps:5d}: {battles_completed:5d}/{target_battles} battles completed")
                print(f"  Total Battle objects created: {battles_created}")
                print(f"  Memory: {current_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
                print(f"  Rates: {battles_per_sec:.1f} battles/sec, {steps_per_sec:.1f} steps/sec")

                # Critical boundaries
                critical_boundaries = [128, 256, 512, 1024, 2048, 4096, 8192]
                for boundary in critical_boundaries:
                    if battles_created >= boundary - 5 and battles_created <= boundary + 5:
                        print(f"  🚨 CRITICAL: Near {boundary} boundary! (created={battles_created})")

            # Aggressive GC every 25 steps
            if total_steps % 25 == 0:
                gc.collect()

            # Check timeout
            if time.time() - start_time > max_time:
                print(f"\n⏱️  Reached time limit ({max_time}s)")
                break

        # Final stats
        elapsed = time.time() - start_time
        final_memory = get_memory_usage()
        memory_growth = final_memory - initial_memory

        print(f"\n{'='*70}")
        print(f"✅ TEST COMPLETED WITHOUT CRASH")
        print(f"{'='*70}")
        print(f"  Battles completed: {battles_completed}")
        print(f"  Battle objects created: {battles_created}")
        print(f"  Total steps: {total_steps}")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Rate: {battles_completed/elapsed:.1f} battles/sec")
        print(f"  Memory: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_growth:+.1f} MB)")
        print(f"  Memory per battle: {memory_growth/battles_created:.3f} MB" if battles_created > 0 else "")

        return True, battles_created

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"💥 CRASH DETECTED! PyKMN FAILED!")
        print(f"{'='*70}")
        print(f"  Battles completed: {battles_completed}")
        print(f"  Battle objects created: {battles_created}")
        print(f"  Failed at step: {total_steps}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")

        print(f"\n📊 CRASH ANALYSIS:")
        # Check if near a power of 2
        powers_of_2 = [2**i for i in range(5, 15)]
        for p in powers_of_2:
            if abs(battles_created - p) <= 5:
                print(f"  ⚠️  Crashed near power of 2: {p}")
                break

        # Memory state
        crash_memory = get_memory_usage()
        print(f"  Memory at crash: {crash_memory:.1f} MB")
        print(f"  Memory growth: {crash_memory - initial_memory:.1f} MB")

        print(f"\n🔍 FULL TRACEBACK:")
        traceback.print_exc()

        return False, battles_created

    finally:
        # Cleanup
        if 'env' in locals():
            try:
                env._cleanup_battles_incremental()
            except:
                pass
            del env
        gc.collect()


def main():
    """Run EXTREME tests to find the breaking point."""
    print("=" * 70)
    print("PyKMN EXTREME STRESS TEST - FIND THE BREAKING POINT")
    print("=" * 70)
    print()
    print("WARNING: This will push PyKMN to its absolute limits!")
    print("We're looking for crashes, memory leaks, and the 128 barrier.")
    print()

    # Extreme test configurations
    test_configs = [
        (128, 5000, 120),   # Push the 128 boundary hard
        (256, 5000, 120),   # Even bigger batch
        (128, 10000, 180),  # MANY battles at 128
    ]

    crash_found = False
    crash_point = None

    for batch_size, target, max_time in test_configs:
        print(f"\n🔥 Testing batch_size={batch_size}, target={target} battles...")

        success, battles_created = run_extreme_test(batch_size, target, max_time)

        if not success:
            crash_found = True
            crash_point = battles_created
            print(f"\n💥💥💥 CRASH CONFIRMED at {battles_created} total Battle objects!")
            break

        # Clean up between tests
        gc.collect()
        time.sleep(3)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    if crash_found:
        print(f"❌ CRASH DETECTED after creating {crash_point} Battle objects")
        print(f"\nThis confirms PyKMN has stability issues with:")

        if crash_point >= 126 and crash_point <= 130:
            print("  • The 128 battle barrier (exact match!)")
        elif crash_point >= 254 and crash_point <= 258:
            print("  • The 256 battle barrier")
        else:
            print(f"  • Resource exhaustion around {crash_point} battles")

        print("\nRECOMMENDATION: Use subprocess isolation or limit batch sizes!")
    else:
        print("✅ NO CRASHES DETECTED even under extreme load")
        print("\nPyKMN appears stable for production use.")
        print("The '128 barrier' may be a historical issue or require")
        print("specific conditions not tested here.")


if __name__ == "__main__":
    main()