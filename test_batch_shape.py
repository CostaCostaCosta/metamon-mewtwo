#!/usr/bin/env python3
"""Test batch shapes."""

import os
import numpy as np
from pathlib import Path

os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams
from metamon.inference.client import RemotePolicyRunner
from metamon.rl.pretrained import get_pretrained_model

BATCH_SIZE = 4

# Get model config
pretrained_cls = get_pretrained_model("SyntheticRLV2")
obs_space = pretrained_cls.observation_space
reward_fn = pretrained_cls.reward_function

# Load teams
cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
team_dir = cache_dir / "teams" / "smogon_pass2"
teams = load_random_teams(team_dir, "gen1ou", BATCH_SIZE * 2)

# Create environment
env = PyKMNVectorEnv(
    num_envs=BATCH_SIZE,
    teams_p1=teams[:BATCH_SIZE],
    teams_p2=teams[BATCH_SIZE:],
    obs_space=obs_space,
    reward_fn=reward_fn,
    track_trajectories=False,
)

# Create policy
policy = RemotePolicyRunner(
    server_url="http://localhost:8080",
    model_name="SyntheticRLV2",
    client_id="test_shape"
)

# Reset and get obs
obs_p1, obs_p2, mask_p1, mask_p2 = env.reset()

print(f"Batch size: {BATCH_SIZE}")
print(f"obs_p1 shapes:")
for k, v in obs_p1.items():
    print(f"  {k}: {v.shape}")
print(f"mask_p1 shape: {mask_p1.shape}")

# Run inference
print(f"\nCalling policy.infer()...")
actions = policy.infer(obs_p1, mask_p1)

print(f"Returned actions shape: {actions.shape}")
print(f"Returned actions dtype: {actions.dtype}")
print(f"Returned actions: {actions}")

env.close()

if actions.shape == (BATCH_SIZE,):
    print(f"\n✅ SUCCESS: Actions have correct shape [{BATCH_SIZE}]")
else:
    print(f"\n❌ FAIL: Expected shape [{BATCH_SIZE}], got {actions.shape}")
