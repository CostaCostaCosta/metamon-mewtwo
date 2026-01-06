#!/usr/bin/env python3
"""Simple test to debug server inference."""

import os
import sys
import numpy as np
from pathlib import Path

# Set cache directory
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams
from metamon.inference.client import InferenceClient
from metamon.rl.pretrained import get_pretrained_model

print("Testing basic inference...")

# Get model config
pretrained_cls = get_pretrained_model("SyntheticRLV2")
obs_space = pretrained_cls.observation_space
reward_fn = pretrained_cls.reward_function

# Load teams
cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
team_dir = cache_dir / "teams" / "smogon_pass2"
teams = load_random_teams(team_dir, "gen1ou", 2)
print(f"Loaded {len(teams)} teams")

# Create environment
env = PyKMNVectorEnv(
    num_envs=1,
    teams_p1=[teams[0]],
    teams_p2=[teams[1]],
    obs_space=obs_space,
    reward_fn=reward_fn,
    track_trajectories=False,
)
print("Created environment")

# Get initial observation
obs_p1, obs_p2, legal_mask_p1, legal_mask_p2 = env.reset()
print(f"Reset environment")
print(f"  obs_p1 type: {type(obs_p1)}")
print(f"  obs_p1 keys: {list(obs_p1.keys()) if isinstance(obs_p1, dict) else 'NOT A DICT'}")
print(f"  legal_mask_p1 shape: {legal_mask_p1.shape}")

# Extract single observation (batch size = 1)
obs_single = {k: v[0] for k, v in obs_p1.items()}
mask_single = legal_mask_p1[0]

print(f"\nSingle obs keys: {list(obs_single.keys())}")
for k, v in obs_single.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: type={type(v)}, value={v}")

print(f"\nMask shape: {mask_single.shape}")
print(f"Mask sum (num legal actions): {mask_single.sum()}")

# Create client
print("\nConnecting to server...")
client = InferenceClient("http://localhost:8080", client_id="test_debug")
print("Connected!")

# Try inference
print("\nRunning inference...")
try:
    actions = client.infer(obs_single, mask_single, reset_state=True)
    print(f"✓ Success! Actions: {actions}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

env.close()
print("\n✓ Test passed!")
