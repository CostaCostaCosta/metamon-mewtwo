#!/usr/bin/env python3
"""
Precise test to identify exactly where memory corruption occurs.
"""

import os
import sys
import gc
import tracemalloc
import numpy as np
from pathlib import Path

# Set cache directory before imports
os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward

def test_memory_leak(num_battles=2000, batch_size=64):
    """Test for memory leaks with progressive diagnostics."""

    print(f"Testing with {num_battles} battles, batch_size={batch_size}")

    # Load teams
    cache_dir = Path(os.environ.get("METAMON_CACHE_DIR", Path.home() / "metamon_cache"))
    team_dir = cache_dir / "teams" / "smogon_pass2"
    teams_p1 = load_random_teams(team_dir, "gen1ou", batch_size)
    teams_p2 = load_random_teams(team_dir, "gen1ou", batch_size)

    # Create environment
    obs_space = DefaultObservationSpace()
    reward_fn = DefaultShapedReward()
    env = PyKMNVectorEnv(
        num_envs=batch_size,
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=True,  # Enable trajectory tracking to trigger the bug
    )

    print("Environment created")

    # Start memory tracking
    tracemalloc.start()

    battles_completed = 0
    steps_taken = 0

    try:
        while battles_completed < num_battles:
            # Reset environment
            obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()

            batch_steps = 0
            batch_done = False

            # Run batch until all battles complete
            while not batch_done and batch_steps < 500:
                # Random actions
                actions_p1 = np.random.randint(0, 9, size=batch_size)
                actions_p2 = np.random.randint(0, 9, size=batch_size)

                # Step
                obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(
                    actions_p1, actions_p2
                )

                batch_steps += 1
                steps_taken += 1

                if info["num_done"] == batch_size:
                    batch_done = True

            # Get completed trajectories (this should trigger cleanup)
            trajectories = env.get_completed_trajectories()
            battles_completed += len(trajectories)

            # Clear trajectories immediately to test if this helps
            del trajectories

            # Progress report
            if battles_completed % 128 == 0:
                current, peak = tracemalloc.get_traced_memory()
                print(f"Battles: {battles_completed}/{num_battles}, "
                      f"Memory: {current / 1024 / 1024:.1f} MB (peak: {peak / 1024 / 1024:.1f} MB), "
                      f"Steps: {steps_taken}")

                # Force garbage collection periodically
                gc.collect()

    except Exception as e:
        print(f"\n❌ CRASH at {battles_completed} battles, {steps_taken} steps")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        env.close()
        tracemalloc.stop()

    print(f"\n✅ SUCCESS: Completed {battles_completed} battles without crash")
    return True


if __name__ == "__main__":
    import traceback

    # Test with increasing battle counts
    for num_battles in [256, 512, 1024, 2048]:
        print(f"\n{'='*60}")
        print(f"Testing {num_battles} battles...")
        print('='*60)

        success = test_memory_leak(num_battles=num_battles)

        if not success:
            print(f"\nFailed at {num_battles} battles")
            break

        # Force full cleanup between tests
        gc.collect()

    print("\nTest complete")