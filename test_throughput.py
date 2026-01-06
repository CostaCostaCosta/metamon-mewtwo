#!/usr/bin/env python3
"""Test throughput with batched inference."""

import os
import time
from pathlib import Path

os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import PyKMNVectorEnv, SelfPlayRunner, load_random_teams
from metamon.inference.client import RemotePolicyRunner
from metamon.rl.pretrained import get_pretrained_model

# Config
BATCH_SIZE = 64
NUM_BATTLES = 128  # 2 full batches

print(f"\n{'='*70}")
print(f"THROUGHPUT TEST")
print(f"{'='*70}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Num battles: {NUM_BATTLES}")
print(f"{'='*70}\n")

# Get model config
pretrained_cls = get_pretrained_model("SyntheticRLV2")
obs_space = pretrained_cls.observation_space
reward_fn = pretrained_cls.reward_function

# Load teams
cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
team_dir = cache_dir / "teams" / "smogon_pass2"
teams_p1 = load_random_teams(team_dir, "gen1ou", BATCH_SIZE)
teams_p2 = load_random_teams(team_dir, "gen1ou", BATCH_SIZE)
print(f"Loaded {len(teams_p1) + len(teams_p2)} teams")

# Create environment
env = PyKMNVectorEnv(
    num_envs=BATCH_SIZE,
    teams_p1=teams_p1,
    teams_p2=teams_p2,
    obs_space=obs_space,
    reward_fn=reward_fn,
    track_trajectories=True,
)
print(f"Created environment with {BATCH_SIZE} parallel battles")

# Create remote policy runners
policy_p1 = RemotePolicyRunner(
    server_url="http://localhost:8080",
    model_name="SyntheticRLV2",
    client_id="p1_perf"
)
policy_p2 = RemotePolicyRunner(
    server_url="http://localhost:8080",
    model_name="SyntheticRLV2",
    client_id="p2_perf"
)
print("Connected to inference server")

# Create self-play runner
runner = SelfPlayRunner(
    vec_env=env,
    policy_p1=policy_p1,
    policy_p2=policy_p2,
)

# Run battles and measure time
print(f"\nRunning {NUM_BATTLES} battles...")
start_time = time.time()

trajectories = runner.collect_trajectories(
    num_battles=NUM_BATTLES,
    max_steps_per_battle=500,
    verbose=False,
)

elapsed = time.time() - start_time
rate = NUM_BATTLES / elapsed

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Battles completed: {len(trajectories)}")
print(f"Time: {elapsed:.2f}s")
print(f"Throughput: {rate:.2f} battles/sec")
print(f"{'='*70}\n")

# Cleanup
env.close()

# Performance assessment
if rate >= 50:
    print("✅ EXCELLENT: Throughput >= 50 battles/sec")
elif rate >= 20:
    print("✓ GOOD: Throughput >= 20 battles/sec")
elif rate >= 5:
    print("⚠ FAIR: Throughput >= 5 battles/sec (can be improved)")
else:
    print("❌ POOR: Throughput < 5 battles/sec (needs optimization)")
