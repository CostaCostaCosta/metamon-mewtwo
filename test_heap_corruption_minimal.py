#!/usr/bin/env python3
"""Minimal test to reproduce heap corruption."""

import os
import gc
import sys
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams
from metamon.interface import ExpandedObservationSpace, DefaultShapedReward, TokenizedObservationSpace
from metamon.tokenizer import PokemonTokenizer
from pathlib import Path
import numpy as np

print("Creating tokenized observation space for Kakuna...")
tokenizer = PokemonTokenizer()
vocab_path = Path(os.environ["METAMON_CACHE_DIR"]) / "vocab.json"
if vocab_path.exists():
    tokenizer.load_tokens_from_disk(str(vocab_path))
base_obs_space = ExpandedObservationSpace()
obs_space = TokenizedObservationSpace(base_obs_space, tokenizer)

print("Running heap corruption test with sequential environments...")

def run_batch(batch_num, num_envs=64, battles_per_batch=64):
    """Run a single batch of battles."""
    print(f"\n=== Batch {batch_num}: {num_envs} envs ===")

    # Load teams
    cache_dir = Path("/home/eddie/metamon_cache")
    team_dir = cache_dir / "teams" / "smogon_pass2"
    teams_p1 = load_random_teams(team_dir, "gen1ou", num_envs)
    teams_p2 = load_random_teams(team_dir, "gen1ou", num_envs)

    # Create environment
    reward_fn = DefaultShapedReward()
    env = PyKMNVectorEnv(
        num_envs=num_envs,
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=True,  # Enable to trigger the bug
    )

    battles_done = 0
    while battles_done < battles_per_batch:
        obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()

        for step in range(500):
            # Random actions
            actions_p1 = np.random.randint(0, 9, num_envs)  # Gen1 only has 9 actions
            actions_p2 = np.random.randint(0, 9, num_envs)

            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(
                actions_p1, actions_p2
            )

            if info["num_done"] == num_envs:
                break

        # Get trajectories to trigger cleanup
        trajectories = env.get_completed_trajectories()
        battles_done += len(trajectories)
        print(f"  Completed {len(trajectories)} battles")

        # Clear trajectories
        del trajectories

    # Close environment
    env.close()

    # Force cleanup
    gc.collect()

    print(f"  ✓ Batch {batch_num} complete")
    return True


# Run multiple batches sequentially to trigger heap corruption
try:
    # Start with different batch sizes to stress memory management
    batch_configs = [
        (1, 64, 64),   # batch 1: 64 envs
        (2, 32, 32),   # batch 2: 32 envs (different size)
        (3, 64, 64),   # batch 3: back to 64
        (4, 16, 16),   # batch 4: 16 envs
        (5, 64, 128),  # batch 5: 64 envs, 128 battles
        (6, 64, 128),  # batch 6: 64 envs, 128 battles
    ]

    for batch_num, num_envs, battles in batch_configs:
        success = run_batch(batch_num, num_envs, battles)
        if not success:
            print(f"\n❌ Failed at batch {batch_num}")
            sys.exit(1)

    print("\n✅ SUCCESS: All batches completed without heap corruption!")

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)