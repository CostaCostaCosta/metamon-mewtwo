#!/usr/bin/env python3
"""
Test script for the GPU inference server.

This script verifies that the inference server is working correctly
and can handle battles end-to-end.

Usage:
    # First, start the server in one terminal:
    python -m metamon.inference.server --model SyntheticRLV2 --batch_size 128

    # Then run this test in another terminal:
    python test_inference_server.py
"""

import os
import sys
from pathlib import Path

# Set cache directory
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import PyKMNVectorEnv, SelfPlayRunner, load_random_teams
from metamon.inference.client import RemotePolicyRunner
from metamon.rl.pretrained import get_pretrained_model


def test_basic_inference():
    """Test basic inference with the server."""
    print("\n" + "="*70)
    print("TEST 1: Basic Inference")
    print("="*70)

    try:
        # Create remote policy runner
        print("Connecting to inference server...")
        policy = RemotePolicyRunner(
            server_url="http://localhost:8080",
            model_name="SyntheticRLV2"
        )
        print("✓ Successfully connected to server")

        # Get model config
        pretrained_cls = get_pretrained_model("SyntheticRLV2")
        obs_space = pretrained_cls.observation_space

        # Load teams
        cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
        team_dir = cache_dir / "teams" / "smogon_pass2"
        teams = load_random_teams(team_dir, "gen1ou", 2)
        print(f"✓ Loaded {len(teams)} teams")

        # Create environment
        env = PyKMNVectorEnv(
            num_envs=1,
            teams_p1=[teams[0]],
            teams_p2=[teams[1]],
            obs_space=obs_space,
            reward_fn=pretrained_cls.reward_function,
            track_trajectories=False,
        )
        print("✓ Created environment")

        # Get initial observation
        obs_p1, obs_p2, legal_mask_p1, legal_mask_p2 = env.reset()
        # Use P1 observations for test
        obs_dict = obs_p1
        legal_mask = legal_mask_p1

        # Run single inference
        print("Running inference...")
        actions = policy.infer(obs_dict, legal_mask)
        print(f"✓ Inference successful! Actions: {actions}")

        # Clean up
        env.close()

        print("\n✅ TEST 1 PASSED: Basic inference works!")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_battle():
    """Test a complete battle with the server."""
    print("\n" + "="*70)
    print("TEST 2: Full Battle")
    print("="*70)

    try:
        # Get model config
        pretrained_cls = get_pretrained_model("SyntheticRLV2")
        obs_space = pretrained_cls.observation_space
        reward_fn = pretrained_cls.reward_function

        # Load teams
        cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
        team_dir = cache_dir / "teams" / "smogon_pass2"
        teams_p1 = load_random_teams(team_dir, "gen1ou", 1)
        teams_p2 = load_random_teams(team_dir, "gen1ou", 1)
        print(f"✓ Loaded {len(teams_p1) + len(teams_p2)} teams")

        # Create environment
        env = PyKMNVectorEnv(
            num_envs=1,
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            obs_space=obs_space,
            reward_fn=reward_fn,
            track_trajectories=True,
        )
        print("✓ Created environment")

        # Create remote policy runners
        print("Connecting to inference server...")
        policy_p1 = RemotePolicyRunner(
            server_url="http://localhost:8080",
            model_name="SyntheticRLV2",
            client_id="p1_test"
        )
        policy_p2 = RemotePolicyRunner(
            server_url="http://localhost:8080",
            model_name="SyntheticRLV2",
            client_id="p2_test"
        )
        print("✓ Connected both players to server")

        # Create self-play runner
        runner = SelfPlayRunner(
            vec_env=env,
            policy_p1=policy_p1,
            policy_p2=policy_p2,
        )
        print("✓ Created self-play runner")

        # Run a single battle
        print("\nRunning battle...")
        trajectories = runner.collect_trajectories(
            num_battles=1,
            max_steps_per_battle=500,
            verbose=True,
        )
        print(f"✓ Battle complete! Generated {len(trajectories)} trajectories")

        if trajectories:
            traj = trajectories[0]
            print(f"  Winner: Player {traj.winner}")
            print(f"  Num transitions: {len(traj.transitions)}")

        # Clean up
        env.close()

        print("\n✅ TEST 2 PASSED: Full battle works!")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_battles():
    """Test multiple battles in batch."""
    print("\n" + "="*70)
    print("TEST 3: Batch Battles (4 battles)")
    print("="*70)

    try:
        # Get model config
        pretrained_cls = get_pretrained_model("SyntheticRLV2")
        obs_space = pretrained_cls.observation_space
        reward_fn = pretrained_cls.reward_function

        # Load teams
        cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
        team_dir = cache_dir / "teams" / "smogon_pass2"
        teams_p1 = load_random_teams(team_dir, "gen1ou", 4)
        teams_p2 = load_random_teams(team_dir, "gen1ou", 4)
        print(f"✓ Loaded {len(teams_p1) + len(teams_p2)} teams")

        # Create environment
        env = PyKMNVectorEnv(
            num_envs=4,
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            obs_space=obs_space,
            reward_fn=reward_fn,
            track_trajectories=True,
        )
        print("✓ Created environment with 4 parallel battles")

        # Create remote policy runners
        print("Connecting to inference server...")
        policy_p1 = RemotePolicyRunner(
            server_url="http://localhost:8080",
            model_name="SyntheticRLV2",
            client_id="p1_batch"
        )
        policy_p2 = RemotePolicyRunner(
            server_url="http://localhost:8080",
            model_name="SyntheticRLV2",
            client_id="p2_batch"
        )
        print("✓ Connected both players to server")

        # Create self-play runner
        runner = SelfPlayRunner(
            vec_env=env,
            policy_p1=policy_p1,
            policy_p2=policy_p2,
        )

        # Run battles
        import time
        start_time = time.time()
        print("\nRunning 4 battles in parallel...")
        trajectories = runner.collect_trajectories(
            num_battles=4,
            max_steps_per_battle=500,
            verbose=True,
        )
        elapsed = time.time() - start_time

        print(f"✓ Battles complete! Generated {len(trajectories)} trajectories")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {len(trajectories)/elapsed:.2f} battles/sec")

        # Show results
        for i, traj in enumerate(trajectories):
            print(f"  Battle {i+1}: Winner = Player {traj.winner}, Transitions = {len(traj.transitions)}")

        # Clean up
        env.close()

        print("\n✅ TEST 3 PASSED: Batch battles work!")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("INFERENCE SERVER TEST SUITE")
    print("="*70)
    print("\nMake sure the inference server is running in another terminal:")
    print("  source .venv/bin/activate")
    print("  export METAMON_CACHE_DIR=/home/eddie/metamon_cache")
    print("  python -m metamon.inference.server --model SyntheticRLV2 --batch_size 128")
    print("\nStarting tests in 3 seconds...")

    import time
    time.sleep(3)

    results = []

    # Run tests
    results.append(("Basic Inference", test_basic_inference()))
    results.append(("Full Battle", test_full_battle()))
    results.append(("Batch Battles", test_batch_battles()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Inference server is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
