#!/usr/bin/env python3
"""
Test if observations from PyKMN are causing memory issues.
"""

import os
import gc
import sys

os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward
from pathlib import Path
import numpy as np

def test_observation_extraction():
    """Test if repeatedly extracting observations causes memory issues."""

    print("Testing observation extraction from PyKMN...")

    # Setup
    cache_dir = Path("/home/eddie/metamon_cache")
    team_dir = cache_dir / "teams" / "smogon_pass2"
    batch_size = 64

    teams = load_random_teams(team_dir, "gen1ou", batch_size * 2)

    # Create environment
    obs_space = DefaultObservationSpace()
    reward_fn = DefaultShapedReward()

    print(f"Creating environment with batch_size={batch_size}")
    env = PyKMNVectorEnv(
        num_envs=batch_size,
        teams_p1=teams[:batch_size],
        teams_p2=teams[batch_size:],
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=True,  # Enable to match production
    )

    try:
        # Store observations to check if they're being properly copied
        all_observations = []

        for iteration in range(50):
            print(f"\nIteration {iteration + 1}...")

            # Reset environment
            obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()

            # Store observations
            all_observations.append({
                "obs_p1": obs_p1,
                "obs_p2": obs_p2,
                "masks_p1": masks_p1,
                "masks_p2": masks_p2,
            })

            # Run some steps
            for step in range(100):
                # Random actions
                actions_p1 = np.random.randint(0, 9, batch_size)
                actions_p2 = np.random.randint(0, 9, batch_size)

                obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(
                    actions_p1, actions_p2
                )

                if info["num_done"] == batch_size:
                    break

            # Get trajectories (this triggers cleanup)
            trajectories = env.get_completed_trajectories()
            print(f"  Collected {len(trajectories)} trajectories")

            # Check if old observations are still valid
            if iteration > 0:
                try:
                    # Try to access old observation data
                    old_obs = all_observations[0]["obs_p1"]
                    if isinstance(old_obs, dict) and "numbers" in old_obs:
                        # Check if we can still access the numpy array
                        shape = old_obs["numbers"].shape
                        mean = old_obs["numbers"].mean()
                        print(f"  Old observation still accessible: shape={shape}, mean={mean:.3f}")
                except Exception as e:
                    print(f"  ⚠️ Old observation corrupted: {e}")

            # Periodically clear old observations to free memory
            if iteration % 10 == 0 and iteration > 0:
                print(f"  Clearing old observations...")
                all_observations = all_observations[-5:]  # Keep only last 5
                gc.collect()

        print("\n✅ SUCCESS: No crash after 50 iterations")
        return True

    except Exception as e:
        print(f"\n❌ CRASH at iteration {iteration + 1}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        env.close()


def test_observation_types():
    """Check what types of data are in observations."""

    cache_dir = Path("/home/eddie/metamon_cache")
    team_dir = cache_dir / "teams" / "smogon_pass2"
    teams = load_random_teams(team_dir, "gen1ou", 4)

    obs_space = DefaultObservationSpace()
    reward_fn = DefaultShapedReward()

    env = PyKMNVectorEnv(
        num_envs=2,
        teams_p1=teams[:2],
        teams_p2=teams[2:],
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=False,
    )

    obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()

    print("\n=== Observation Structure ===")
    print(f"obs_p1 type: {type(obs_p1)}")
    if isinstance(obs_p1, dict):
        for key, value in obs_p1.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: ndarray {value.dtype} {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

    print(f"\nmasks_p1 type: {type(masks_p1)}")
    if isinstance(masks_p1, np.ndarray):
        print(f"  shape: {masks_p1.shape}, dtype: {masks_p1.dtype}")

    env.close()


if __name__ == "__main__":
    # First check observation structure
    test_observation_types()

    # Then test memory issues
    success = test_observation_extraction()

    sys.exit(0 if success else 1)