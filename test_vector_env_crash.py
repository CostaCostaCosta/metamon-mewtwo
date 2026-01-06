#!/usr/bin/env python3
"""
Test the actual PyKMN vector environment to reproduce the crash
This uses the full metamon stack with feature extraction
"""

import gc
import sys
import numpy as np
import traceback

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Pokemon
from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.interface import ExpandedObservationSpace, DefaultShapedReward


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


def test_vector_env(batch_size: int, steps: int = 100):
    """Test the vector environment."""
    print(f"\nTesting PyKmnVectorEnv with batch_size={batch_size}...", flush=True)

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
        print(f"  Environment created and reset successfully", flush=True)

        # Run steps
        for step in range(steps):
            # Random actions (respecting legal masks)
            actions_p1 = []
            actions_p2 = []

            for i in range(batch_size):
                # Get legal actions
                legal_p1 = np.where(legal_masks_p1[i])[0]
                legal_p2 = np.where(legal_masks_p2[i])[0]

                # Pick random legal action
                action_p1 = np.random.choice(legal_p1) if len(legal_p1) > 0 else 0
                action_p2 = np.random.choice(legal_p2) if len(legal_p2) > 0 else 0

                actions_p1.append(action_p1)
                actions_p2.append(action_p2)

            actions_p1 = np.array(actions_p1)
            actions_p2 = np.array(actions_p2)

            # Step environment
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)

            # Get new legal masks from observations
            legal_masks_p1 = obs_p1.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))
            legal_masks_p2 = obs_p2.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))

            # Progress report
            if step % 20 == 0:
                print(f"    Step {step}/{steps}, dones: {dones.sum()}/{batch_size}", flush=True)

            # Reset if all done
            if dones.all():
                obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()
                print(f"    All battles done, environment reset", flush=True)

        print(f"  ✓ Vector environment test PASSED: batch_size={batch_size}", flush=True)
        return True

    except Exception as e:
        print(f"  ✗ Vector environment test FAILED: {e}", flush=True)
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if 'env' in locals():
            env._cleanup_battles_incremental()


def main():
    """Test various batch sizes."""
    print("=" * 60)
    print("PyKMN VECTOR ENVIRONMENT CRASH TEST")
    print("=" * 60)

    # Test increasing batch sizes
    batch_sizes = [1, 16, 32, 64, 80, 96, 112, 127, 128, 129, 144, 256]

    results = {}
    for batch_size in batch_sizes:
        gc.collect()
        success = test_vector_env(batch_size, steps=50)
        results[batch_size] = success

        if not success:
            print(f"\n🔴 CRASH at batch_size={batch_size}")
            break

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for size, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} batch_size={size}")

    # Analysis
    failures = [k for k, v in results.items() if not v]
    if failures:
        first_failure = min(failures)
        print(f"\n🔴 First failure at batch_size={first_failure}")

        if first_failure == 128:
            print("  CRITICAL: Failure at exactly 128 - hardcoded buffer limit")
            print("  This matches the known '128 battle barrier' issue")
        elif first_failure in [32, 64, 256]:
            print(f"  Failure at power-of-2 ({first_failure}) suggests buffer/alignment issue")
    else:
        print("\n✓ All tests passed - no crashes detected")
        print("  Note: Crashes may be non-deterministic or require longer runs")


if __name__ == "__main__":
    main()