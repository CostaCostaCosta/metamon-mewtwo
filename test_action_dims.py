#!/usr/bin/env python3
"""Test action dimension mismatch between model and environment."""

import os
import sys
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward
from pathlib import Path
import numpy as np

# Load teams
cache_dir = Path("/home/eddie/metamon_cache")
team_dir = cache_dir / "teams" / "smogon_pass2"
teams_p1 = load_random_teams(team_dir, "gen1ou", 2)
teams_p2 = load_random_teams(team_dir, "gen1ou", 2)

# Create environment
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
env = PyKMNVectorEnv(
    num_envs=2,
    teams_p1=teams_p1,
    teams_p2=teams_p2,
    obs_space=obs_space,
    reward_fn=reward_fn,
    track_trajectories=False,
)

print("Testing action dimensions...")

# Reset and check mask dimensions
obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()

print(f"Environment masks shape: {masks_p1.shape}")
print(f"Number of actions in environment: {masks_p1.shape[1]}")

# Now test with Kakuna model
from metamon.env.pykmn import LocalPolicyRunner

policy = LocalPolicyRunner(
    model_name="Kakuna",
    device="cuda",
)

print(f"\nKakuna model action_dim: {policy.action_dim}")

# Test inference
try:
    actions = policy.infer(obs_p1, masks_p1)
    print(f"Actions from model: {actions}")
    print(f"Max action value: {actions.max()}")

    # Check if any actions exceed environment's action space
    if actions.max() >= masks_p1.shape[1]:
        print(f"\n⚠️  PROBLEM: Model outputs action {actions.max()} but environment only has {masks_p1.shape[1]} actions!")
    else:
        print("\n✓ Actions within valid range")

except Exception as e:
    print(f"\n❌ Error during inference: {e}")

env.close()