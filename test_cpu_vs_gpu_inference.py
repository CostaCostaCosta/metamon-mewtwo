#!/usr/bin/env python3
"""
Test to isolate whether the heap corruption is GPU-related.
"""

import os
import sys
import argparse
import time

os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

def test_inference(device="cuda", num_battles=1000, batch_size=64):
    """Test inference with specified device."""

    from metamon.env.pykmn import PyKMNVectorEnv, LocalPolicyRunner, SelfPlayRunner, load_random_teams
    from metamon.interface import DefaultObservationSpace, DefaultShapedReward
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"Testing {num_battles} battles with {device.upper()} inference")
    print(f"Batch size: {batch_size}")
    print('='*60)

    # Load teams
    cache_dir = Path("/home/eddie/metamon_cache")
    team_dir = cache_dir / "teams" / "smogon_pass2"
    teams = load_random_teams(team_dir, "gen1ou", batch_size * 2)

    # Create environment
    obs_space = DefaultObservationSpace()
    reward_fn = DefaultShapedReward()
    env = PyKMNVectorEnv(
        num_envs=batch_size,
        teams_p1=teams[:batch_size],
        teams_p2=teams[batch_size:],
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=True,
    )

    # Create policy runner with specified device
    print(f"Loading model on {device}...")

    # Force CUDA to be disabled for CPU testing
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    policy = LocalPolicyRunner(
        model_name="SyntheticRLV2",
        device=device,
        temperature=1.0,
    )

    # Create self-play runner
    runner = SelfPlayRunner(
        vec_env=env,
        policy_p1=policy,
        policy_p2=policy,
    )

    start_time = time.time()

    try:
        # Collect trajectories
        print("Starting trajectory collection...")
        trajectories = runner.collect_trajectories(
            num_battles=num_battles,
            max_steps_per_battle=500,
            verbose=True,
        )

        elapsed = time.time() - start_time

        print(f"\n✅ SUCCESS: Completed {len(trajectories)} battles in {elapsed:.1f}s")
        print(f"Rate: {len(trajectories)/elapsed:.1f} battles/sec")

        # Clean up
        env.close()
        del policy
        del runner

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ CRASH after {elapsed:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CPU vs GPU inference stability")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--num_battles", type=int, default=1000,
                        help="Number of battles to run")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for parallel battles")

    args = parser.parse_args()

    success = test_inference(
        device=args.device,
        num_battles=args.num_battles,
        batch_size=args.batch_size
    )

    sys.exit(0 if success else 1)