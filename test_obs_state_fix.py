#!/usr/bin/env python3
"""Minimal test of observation state fix."""

import os
from pathlib import Path

os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import load_random_teams, PyKMNVectorEnv
from metamon.rl.pretrained import get_pretrained_model

# Load model and observation space
model = get_pretrained_model('Kakuna')
obs_space = model.observation_space
reward_fn = model.reward_function

# Load teams
cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
team_dir = cache_dir / "teams" / "smogon_pass2"
teams_p1 = load_random_teams(team_dir, "gen1ou", 4)
teams_p2 = load_random_teams(team_dir, "gen1ou", 4)

print("Creating vectorized environment...")
vec_env = PyKMNVectorEnv(
    teams_p1=teams_p1,
    teams_p2=teams_p2,
    num_envs=4,
    obs_space=obs_space,
    reward_fn=reward_fn,
    battle_format="gen1ou",
    track_trajectories=False,
)

print("Resetting environment...")
obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()

print(f"P1 observations shape: {obs_p1['numbers'].shape}")
print(f"P2 observations shape: {obs_p2['numbers'].shape}")

print("\nTaking 10 steps...")
for i in range(10):
    # Sample random legal actions
    actions_p1 = [masks_p1[j].nonzero()[0][0] if masks_p1[j].any() else 0 for j in range(4)]
    actions_p2 = [masks_p2[j].nonzero()[0][0] if masks_p2[j].any() else 0 for j in range(4)]

    obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = vec_env.step(actions_p1, actions_p2)
    masks_p1, masks_p2 = vec_env._extract_legal_masks()

    print(f"Step {i+1}: done={dones.sum()}, rewards_p1={rewards_p1.mean():.3f}, rewards_p2={rewards_p2.mean():.3f}")

    if dones.all():
        break

print("\nTest completed successfully!")
vec_env.close()
