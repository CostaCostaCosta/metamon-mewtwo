#!/usr/bin/env python3
"""
Minimal test to reproduce the crash with model inference.
This simulates exactly what generate_selfplay_batched.py does.
"""

import os
import sys
import gc

os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn import PyKMNVectorEnv, LocalPolicyRunner, SelfPlayRunner, load_random_teams
from metamon.rl.pretrained import get_pretrained_model
from pathlib import Path

def run_test(model_name="SyntheticRLV2", total_battles=1000, batch_size=64):
    """Reproduce the exact crash from generate_selfplay_batched.py."""

    print(f"Testing {model_name} with {total_battles} battles, batch_size={batch_size}")

    # Get observation space and reward function from the pretrained model (match the script exactly)
    pretrained_cls = get_pretrained_model(model_name)
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    # Load teams
    cache_dir = Path("/home/eddie/metamon_cache")
    team_dir = cache_dir / "teams" / "smogon_pass2"
    all_teams = load_random_teams(team_dir, "gen1ou", batch_size * 4)  # Extra teams for variety

    # Create environment
    env = PyKMNVectorEnv(
        num_envs=batch_size,
        teams_p1=all_teams[:batch_size],
        teams_p2=all_teams[batch_size:batch_size*2],
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=True,
    )

    # Create policy
    print(f"Loading {model_name}...")
    policy = LocalPolicyRunner(
        model_name=model_name,
        device="cuda",
        temperature=1.0,
    )

    # Create self-play runner
    runner = SelfPlayRunner(
        vec_env=env,
        policy_p1=policy,
        policy_p2=policy,
    )

    battles_completed = 0

    try:
        while battles_completed < total_battles:
            battles_remaining = total_battles - battles_completed
            current_batch = min(batch_size, battles_remaining)

            print(f"\nBatch: collecting {current_batch} battles (completed: {battles_completed}/{total_battles})")

            # Collect trajectories (this is where it crashes)
            trajectories = runner.collect_trajectories(
                num_battles=current_batch,
                max_steps_per_battle=500,
                verbose=False,
            )

            battles_completed += len(trajectories)
            print(f"  Collected {len(trajectories)} trajectories")

            # Clear trajectories to free memory
            del trajectories
            gc.collect()

            # This simulates what happens when we save batches
            if battles_completed % 128 == 0:
                print(f"  Checkpoint: {battles_completed} battles completed")

        print(f"\n✅ SUCCESS: Completed {battles_completed} battles without crash!")
        return True

    except Exception as e:
        print(f"\n❌ CRASH after {battles_completed} battles")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        env.close()
        del policy
        del runner
        gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SyntheticRLV2", help="Model to test")
    parser.add_argument("--battles", type=int, default=1000, help="Number of battles")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    success = run_test(
        model_name=args.model,
        total_battles=args.battles,
        batch_size=args.batch_size
    )

    sys.exit(0 if success else 1)