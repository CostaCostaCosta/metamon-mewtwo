import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward
from pathlib import Path
import numpy as np

# Load teams
cache_dir = Path("/home/eddie/metamon_cache")
team_dir = cache_dir / "teams" / "smogon_pass2"
teams_p1 = load_random_teams(team_dir, "gen1ou", 64)
teams_p2 = load_random_teams(team_dir, "gen1ou", 64)

# Create environment WITHOUT trajectory tracking
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
env = PyKMNVectorEnv(
    num_envs=64,
    teams_p1=teams_p1,
    teams_p2=teams_p2,
    obs_space=obs_space,
    reward_fn=reward_fn,
    track_trajectories=False,  # DISABLED
)

print("Running 1000 battles WITHOUT trajectory tracking...")
battles_done = 0
while battles_done < 1000:
    obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()
    for step in range(500):
        actions_p1 = np.random.randint(0, 9, 64)
        actions_p2 = np.random.randint(0, 9, 64)
        obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)
        if info["num_done"] == 64:
            break
    battles_done += 64
    print(f"Completed {battles_done} battles")

print("SUCCESS: No crash!")
