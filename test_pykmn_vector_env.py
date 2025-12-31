#!/usr/bin/env python3
"""Test the pypkmn vector environment with random policies."""

import numpy as np
from metamon.env.pykmn import (
    PyKMNVectorEnv,
    RandomPolicyRunner,
    SelfPlayRunner,
    load_random_teams,
)
from metamon.interface import get_observation_space, get_reward_function

def test_vector_env():
    """Test vectorized environment with random policies."""
    print("=" * 60)
    print("Testing PyKMN Vector Environment")
    print("=" * 60)

    # Configuration
    num_envs = 4
    battle_format = "gen1ou"
    team_dir = "~/metamon_cache/teams/modern_replays_v2"

    # Load teams
    print(f"\n1. Loading {num_envs} random teams...")
    teams_p1 = load_random_teams(team_dir, battle_format, num_envs)
    teams_p2 = load_random_teams(team_dir, battle_format, num_envs)
    print(f"   ✓ Loaded {len(teams_p1)} vs {len(teams_p2)} teams")

    # Create observation space and reward function
    print("\n2. Initializing observation space and reward function...")
    obs_space = get_observation_space("ExpandedObservationSpace")
    reward_fn = get_reward_function("AggressiveShapedRewardSleep")
    print(f"   ✓ Using {obs_space.__class__.__name__}")
    print(f"   ✓ Using {reward_fn.__class__.__name__}")

    # Create vector environment
    print(f"\n3. Creating vector environment with {num_envs} battles...")
    vec_env = PyKMNVectorEnv(
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        num_envs=num_envs,
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=True,
    )
    print(f"   ✓ Environment created")

    # Reset environment
    print("\n4. Resetting environment...")
    obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()
    print(f"   ✓ Observations shape: {obs_p1['numbers'].shape}")
    print(f"   ✓ Text shape: {obs_p1['text'].shape}")
    print(f"   ✓ Legal masks shape: {masks_p1.shape}")

    # Create random policies
    print("\n5. Creating random policies...")
    policy_p1 = RandomPolicyRunner()
    policy_p2 = RandomPolicyRunner()
    print(f"   ✓ Policies created")

    # Run a few steps
    print("\n6. Running 10 steps...")
    for step in range(10):
        # Infer actions
        actions_p1 = policy_p1.infer(obs_p1, masks_p1)
        actions_p2 = policy_p2.infer(obs_p2, masks_p2)

        # Step environment
        obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = vec_env.step(
            actions_p1, actions_p2
        )

        # Extract legal masks
        masks_p1, masks_p2 = vec_env._extract_legal_masks()

        num_done = info["num_done"]
        print(f"   Step {step + 1}: {num_done} battles done, "
              f"mean reward: {rewards_p1.mean():.2f}")

        if num_done > 0:
            print(f"      Completed {num_done} battles!")
            break

    # Get completed trajectories
    print("\n7. Collecting trajectories...")
    trajectories = vec_env.get_completed_trajectories()
    print(f"   ✓ Collected {len(trajectories)} trajectories")

    if len(trajectories) > 0:
        traj = trajectories[0]
        print(f"   ✓ First trajectory: {len(traj.transitions)} transitions")
        print(f"   ✓ Winner: P{traj.winner if traj.winner > 0 else 'Tie'}")

    # Clean up
    vec_env.close()
    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_vector_env()
