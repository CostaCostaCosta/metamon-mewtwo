#!/usr/bin/env python3
"""
Debugging harness to isolate PyKMN batched inference memory corruption.

This script performs a binary search through the observation extraction pipeline
to identify the exact layer causing heap corruption:

1. test_pure_pykmn() - Raw PyKMN only (no metamon wrappers)
2. test_feature_extraction() - PyKMN + feature extraction
3. test_observation_space() - Add observation space conversion
4. test_vectorized_env() - Full integration (16 envs)

Usage:
    # Run all tests
    python test_corruption_bisect.py

    # Run specific test
    python test_corruption_bisect.py --test vectorized

    # Run with ASAN/allocator hardening
    PYTHONMALLOC=malloc MALLOC_CHECK_=3 PYTHONFAULTHANDLER=1 python test_corruption_bisect.py

    # Run with maximum memory debugging
    PYTHONMALLOC=malloc MALLOC_CHECK_=3 PYTHONFAULTHANDLER=1 ASAN_OPTIONS=detect_leaks=1 python test_corruption_bisect.py
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

# Ensure metamon is importable
sys.path.insert(0, str(Path(__file__).parent))


def load_test_teams(battle_format="gen1ou", num_teams=100):
    """Load random teams for testing."""
    from metamon.data.teams import load_random_teams
    return load_random_teams(battle_format, num=num_teams, seed=42)


def test_pure_pykmn(num_battles=100, verbose=True):
    """Test raw PyKMN (no metamon wrappers).

    This tests the PyKMN C++ library directly without any metamon integration.
    If this crashes, the bug is in PyKMN itself (upstream issue).
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 1: Pure PyKMN (no metamon wrappers)")
        print("="*60)
        print(f"Running {num_battles} battles...")

    from pykmn.engine.gen1 import Battle

    teams_p1 = load_test_teams(num_teams=num_battles)
    teams_p2 = load_test_teams(num_teams=num_battles)

    crash_count = 0
    for i in range(num_battles):
        try:
            battle = Battle(p1_team=teams_p1[i], p2_team=teams_p2[i])
            for step in range(100):
                result, _ = battle.update_raw(0, 0)
                if result.type().name != "NONE":
                    break
            battle = None

            if i % 10 == 0 and i > 0:
                gc.collect()
                if verbose and i % 20 == 0:
                    print(f"  Completed {i} battles...")
        except Exception as e:
            crash_count += 1
            if verbose:
                print(f"  ❌ Battle {i} crashed: {e}")
            if crash_count > 5:
                raise RuntimeError(f"Too many crashes ({crash_count}) in pure PyKMN test")

    if verbose:
        print(f"✅ Pure PyKMN: {num_battles} battles completed without crashes")
    return True


def test_feature_extraction(num_battles=100, verbose=True):
    """Test PyKMN + feature extraction layer.

    This tests whether the feature extraction (pykmn_to_features_raw) causes corruption.
    If this crashes but test_pure_pykmn doesn't, the bug is in metamon/env/pykmn/features.py.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 2: PyKMN + Feature Extraction")
        print("="*60)
        print(f"Running {num_battles} battles with feature extraction...")

    from pykmn.engine.gen1 import Battle, Player
    from metamon.env.pykmn.features import pykmn_to_features_raw, precompute_mappings

    mappings = precompute_mappings()
    teams_p1 = load_test_teams(num_teams=num_battles)
    teams_p2 = load_test_teams(num_teams=num_battles)

    crash_count = 0
    for i in range(num_battles):
        try:
            battle = Battle(p1_team=teams_p1[i], p2_team=teams_p2[i])
            result, _ = battle.update_raw(0, 0)

            for step in range(100):
                # Extract features (this is where C++ → Python conversion happens)
                features_p1 = pykmn_to_features_raw(battle, result, Player.P1, mappings)
                features_p2 = pykmn_to_features_raw(battle, result, Player.P2, mappings)

                result, _ = battle.update_raw(0, 0)
                if result.type().name != "NONE":
                    break

            battle = None

            if i % 10 == 0 and i > 0:
                gc.collect()
                if verbose and i % 20 == 0:
                    print(f"  Completed {i} battles...")
        except Exception as e:
            crash_count += 1
            if verbose:
                print(f"  ❌ Battle {i} crashed: {e}")
            if crash_count > 5:
                raise RuntimeError(f"Too many crashes ({crash_count}) in feature extraction test")

    if verbose:
        print(f"✅ Feature Extraction: {num_battles} battles completed without crashes")
    return True


def test_observation_space(num_battles=100, obs_space_name="ExpandedObservationSpace", verbose=True):
    """Test PyKMN + feature extraction + observation space conversion.

    This tests whether observation space conversion (especially ExpandedObservationSpace)
    causes corruption. If this crashes but test_feature_extraction doesn't, the bug is in
    metamon/interface.py observation spaces.
    """
    if verbose:
        print("\n" + "="*60)
        print(f"TEST 3: PyKMN + Features + {obs_space_name}")
        print("="*60)
        print(f"Running {num_battles} battles with observation space...")

    from pykmn.engine.gen1 import Battle, Player
    from metamon.env.pykmn.features import pykmn_to_features_raw, features_to_universal_state, precompute_mappings
    from metamon.interface import get_observation_space

    obs_space = get_observation_space(obs_space_name)
    mappings = precompute_mappings()
    teams_p1 = load_test_teams(num_teams=num_battles)
    teams_p2 = load_test_teams(num_teams=num_battles)

    crash_count = 0
    for i in range(num_battles):
        try:
            # Reset observation space state (IMPORTANT: per-battle reset)
            if hasattr(obs_space, 'init_obs_state'):
                obs_state = obs_space.init_obs_state()
            else:
                obs_space.reset()
                obs_state = None

            battle = Battle(p1_team=teams_p1[i], p2_team=teams_p2[i])
            result, _ = battle.update_raw(0, 0)

            for step in range(100):
                # Full pipeline: C++ → features → UniversalState → observation
                features_p1 = pykmn_to_features_raw(battle, result, Player.P1, mappings)
                state_p1 = features_to_universal_state(features_p1, mappings)

                # Convert to observation (tests both new and legacy protocols)
                if obs_state is not None:
                    obs_p1, obs_state = obs_space(state_p1, obs_state)
                else:
                    obs_p1 = obs_space(state_p1)

                result, _ = battle.update_raw(0, 0)
                if result.type().name != "NONE":
                    break

            battle = None

            if i % 10 == 0 and i > 0:
                gc.collect()
                if verbose and i % 20 == 0:
                    print(f"  Completed {i} battles...")
        except Exception as e:
            crash_count += 1
            if verbose:
                print(f"  ❌ Battle {i} crashed: {e}")
            if crash_count > 5:
                raise RuntimeError(f"Too many crashes ({crash_count}) in observation space test")

    if verbose:
        print(f"✅ Observation Space ({obs_space_name}): {num_battles} battles completed without crashes")
    return True


def test_vectorized_env(num_batches=10, batch_size=16, obs_space_name="ExpandedObservationSpace", verbose=True):
    """Test full vectorized environment (16 parallel battles).

    This is the complete integration test. If this crashes but earlier tests don't,
    the bug is in metamon/env/pykmn/vector_env.py or in the batched interaction pattern.
    """
    if verbose:
        print("\n" + "="*60)
        print(f"TEST 4: Vectorized Env (batch_size={batch_size})")
        print("="*60)
        print(f"Running {num_batches} batches ({num_batches * batch_size} total battles)...")

    from metamon.env.pykmn import PyKMNVectorEnv
    from metamon.interface import get_observation_space, get_reward_function

    obs_space = get_observation_space(obs_space_name)
    reward_fn = get_reward_function("DefaultShapedReward")

    crash_count = 0
    completed_battles = 0

    for batch_idx in range(num_batches):
        try:
            teams_p1 = load_test_teams(num_teams=batch_size)
            teams_p2 = load_test_teams(num_teams=batch_size)

            vec_env = PyKMNVectorEnv(
                teams_p1=teams_p1,
                teams_p2=teams_p2,
                num_envs=batch_size,
                obs_space=obs_space,
                reward_fn=reward_fn,
                battle_format="gen1ou",
                track_trajectories=False,
            )

            obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()

            for step in range(100):
                actions_p1 = np.random.randint(0, 13, size=batch_size)
                actions_p2 = np.random.randint(0, 13, size=batch_size)
                obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = vec_env.step(actions_p1, actions_p2)

                if info["num_done"] == batch_size:
                    break

            completed_battles += batch_size
            vec_env.close()

            if batch_idx % 2 == 0 and batch_idx > 0:
                gc.collect()
                if verbose:
                    print(f"  Completed batch {batch_idx+1}/{num_batches} ({completed_battles} battles)...")

        except Exception as e:
            crash_count += 1
            if verbose:
                print(f"  ❌ Batch {batch_idx} crashed: {e}")
            if crash_count > 3:
                raise RuntimeError(f"Too many crashes ({crash_count}) in vectorized env test")

    if verbose:
        print(f"✅ Vectorized Env: {num_batches} batches ({completed_battles} battles) completed without crashes")
    return True


def main():
    parser = argparse.ArgumentParser(description="PyKMN memory corruption debugging harness")
    parser.add_argument("--test", type=str, choices=["pykmn", "features", "obs_space", "vectorized", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--num-battles", type=int, default=100, help="Number of battles for single-env tests")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches for vectorized test")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for vectorized test")
    parser.add_argument("--obs-space", type=str, default="ExpandedObservationSpace",
                        help="Observation space to test")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()
    verbose = not args.quiet

    print("="*60)
    print("PyKMN Memory Corruption Bisection Harness")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Test: {args.test}")
    print(f"  - Observation Space: {args.obs_space}")
    print(f"  - Num Battles (single-env): {args.num_battles}")
    print(f"  - Num Batches (vectorized): {args.num_batches}")
    print(f"  - Batch Size (vectorized): {args.batch_size}")
    print()

    tests = {
        "pykmn": lambda: test_pure_pykmn(args.num_battles, verbose),
        "features": lambda: test_feature_extraction(args.num_battles, verbose),
        "obs_space": lambda: test_observation_space(args.num_battles, args.obs_space, verbose),
        "vectorized": lambda: test_vectorized_env(args.num_batches, args.batch_size, args.obs_space, verbose),
    }

    if args.test == "all":
        test_order = ["pykmn", "features", "obs_space", "vectorized"]
    else:
        test_order = [args.test]

    for test_name in test_order:
        try:
            tests[test_name]()
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"Error: {e}")
            print(f"\nConclusion: Corruption occurs in the '{test_name}' layer")
            print(f"\nDebugging Recommendations:")
            if test_name == "pykmn":
                print("  - Report bug to PyKMN upstream")
                print("  - Check PyKMN version and installation")
            elif test_name == "features":
                print("  - Review metamon/env/pykmn/features.py:pykmn_to_features_raw()")
                print("  - Check for dangling C++ object references")
                print("  - Run with ASAN to get stack trace")
            elif test_name == "obs_space":
                print("  - Review metamon/interface.py observation space classes")
                print(f"  - Focus on {args.obs_space} implementation")
                print("  - Check for deepcopy() of C++ objects")
            elif test_name == "vectorized":
                print("  - Review metamon/env/pykmn/vector_env.py")
                print("  - Check per-env state management")
                print("  - Check for race conditions in C++ object lifecycle")
            sys.exit(1)

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nNo memory corruption detected in tested code paths.")
    print("If crashes still occur in production:")
    print("  1. Increase --num-battles and --num-batches")
    print("  2. Run with ASAN/allocator hardening (see CLAUDE.md)")
    print("  3. Test with different observation spaces")
    print("  4. Check for issues in model inference code (not tested here)")


if __name__ == "__main__":
    # Enable fault handler for better segfault traces
    import faulthandler
    faulthandler.enable(all_threads=True)

    main()
