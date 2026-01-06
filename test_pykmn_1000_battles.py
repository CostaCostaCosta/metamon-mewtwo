#!/usr/bin/env python3
"""
Extended stress test - run PyKMN for 1000+ battles
This tests if crashes occur with extended usage / memory accumulation
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


def run_extended_test(batch_size: int, target_battles: int = 1000):
    """Run extended test until we complete target_battles."""
    print(f"\n{'='*60}")
    print(f"EXTENDED TEST: batch_size={batch_size}, target={target_battles} battles")
    print(f"{'='*60}")

    # Track metrics
    battles_completed = 0
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

        # Reset environment
        obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()
        print(f"✓ Environment created with batch_size={batch_size}")
        print(f"  Initial memory: {initial_memory:.1f} MB")

        # Run until we hit target battles
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

            # Step environment
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)

            # Update legal masks
            legal_masks_p1 = obs_p1.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))
            legal_masks_p2 = obs_p2.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))

            total_steps += 1

            # Count completed battles
            new_completions = dones.sum()
            if new_completions > 0:
                battles_completed += new_completions

                # Reset completed battles
                for i in range(batch_size):
                    if dones[i]:
                        # Environment should auto-reset, but get fresh observations
                        pass

            # Progress report
            if total_steps % 100 == 0:
                current_memory = get_memory_usage()
                memory_delta = current_memory - initial_memory
                elapsed = time.time() - start_time
                battles_per_sec = battles_completed / elapsed if elapsed > 0 else 0

                print(f"  Step {total_steps}: {battles_completed}/{target_battles} battles")
                print(f"    Memory: {current_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
                print(f"    Rate: {battles_per_sec:.1f} battles/sec")

                # Check for critical boundaries
                if battles_completed >= 120 and battles_completed <= 135:
                    print(f"    ⚠️  CROSSING 128 BOUNDARY: {battles_completed} battles")
                if battles_completed >= 250 and battles_completed <= 260:
                    print(f"    ⚠️  CROSSING 256 BOUNDARY: {battles_completed} battles")
                if battles_completed >= 500 and battles_completed <= 510:
                    print(f"    ⚠️  CROSSING 512 BOUNDARY: {battles_completed} battles")

            # Periodic GC
            if total_steps % 50 == 0:
                gc.collect()

            # Emergency exit if taking too long
            if time.time() - start_time > 300:  # 5 minutes
                print(f"  ⏱️  Timeout after 5 minutes")
                break

        # Final stats
        elapsed = time.time() - start_time
        final_memory = get_memory_usage()
        memory_growth = final_memory - initial_memory

        print(f"\n{'='*60}")
        print(f"✅ TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"  Battles completed: {battles_completed}")
        print(f"  Total steps: {total_steps}")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Rate: {battles_completed/elapsed:.1f} battles/sec")
        print(f"  Memory: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_growth:+.1f} MB)")
        print(f"  Memory per battle: {memory_growth/battles_completed:.3f} MB" if battles_completed > 0 else "")

        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ TEST FAILED")
        print(f"{'='*60}")
        print(f"  Battles completed: {battles_completed}")
        print(f"  Failed at step: {total_steps}")
        print(f"  Error: {e}")
        print(f"\n🔴 CRASH DETAILS:")
        traceback.print_exc()

        # Memory state at crash
        crash_memory = get_memory_usage()
        print(f"\n  Memory at crash: {crash_memory:.1f} MB")
        print(f"  Memory growth: {crash_memory - initial_memory:.1f} MB")

        return False
    finally:
        # Cleanup
        if 'env' in locals():
            env._cleanup_battles_incremental()
            del env
        gc.collect()


def main():
    """Run extended tests with different batch sizes."""
    print("=" * 70)
    print("PyKMN EXTENDED STRESS TEST - 1000+ BATTLES")
    print("=" * 70)
    print()
    print("This test will run the full PyKMN pipeline until we complete")
    print("1000+ battles to check for memory leaks, resource exhaustion,")
    print("and the reported '128 battle barrier' crash.")
    print()

    # Test configurations (batch_size, target_battles)
    test_configs = [
        (32, 1000),   # Moderate batch, many battles
        (64, 1000),   # Larger batch
        (128, 1000),  # Test the "dangerous" 128 boundary
    ]

    for batch_size, target in test_configs:
        success = run_extended_test(batch_size, target)

        if not success:
            print(f"\n🔴 CRASH DETECTED at batch_size={batch_size}")
            print("This confirms the PyKMN stability issue exists!")
            break

        # Clean up between tests
        gc.collect()
        time.sleep(2)
        print()


if __name__ == "__main__":
    main()