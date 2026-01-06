#!/usr/bin/env python3
"""
Test script to reproduce the batch size mismatch bug when reset() is not called.

The actual bug: buffers are initialized with first batch size,
but subsequent batches with different sizes don't get properly resized.
"""

import os
from pathlib import Path
os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

import numpy as np
from metamon.env.pykmn import load_random_teams, PyKMNVectorEnv, LocalPolicyRunner
from metamon.rl.pretrained import get_pretrained_model

def test_batch_size_mismatch_no_reset():
    """Reproduce the bug where batch size changes but reset() is not called properly."""

    print("="*70)
    print("Testing batch size mismatch WITHOUT proper reset()")
    print("="*70)

    # Load teams
    team_dir = Path.home() / 'metamon_cache' / 'teams' / 'smogon_pass2'

    # Get obs/reward from pretrained
    pretrained_cls = get_pretrained_model('SyntheticRLV2')
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    # Create policy runner
    print("\n1. Creating policy runner...")
    policy = LocalPolicyRunner(
        model_name='SyntheticRLV2',
        checkpoint=48,
        device='cuda',
        use_amp=True,
        verbose=False,
    )

    # Simulate first batch with 64 envs
    print("\n2. First batch: 64 environments")
    teams_p1_large = load_random_teams(team_dir, 'gen1ou', 64)
    teams_p2_large = load_random_teams(team_dir, 'gen1ou', 64)
    env_large = PyKMNVectorEnv(teams_p1_large, teams_p2_large, num_envs=64,
                               obs_space=obs_space, reward_fn=reward_fn)
    obs_p1, obs_p2, masks_p1, masks_p2 = env_large.reset()

    print(f"   Observation shape: {obs_p1['numbers'].shape}")
    policy.reset(batch_size=64)

    # Do one inference step to initialize buffers
    actions = policy.infer(obs_p1, masks_p1)
    print(f"   Actions shape: {actions.shape}")
    print(f"   ✓ First inference succeeded")

    # Simulate second batch with only 4 envs BUT DON'T CALL RESET
    print("\n3. Second batch: 4 environments (WITHOUT reset() call)")
    teams_p1_small = load_random_teams(team_dir, 'gen1ou', 4)
    teams_p2_small = load_random_teams(team_dir, 'gen1ou', 4)
    env_small = PyKMNVectorEnv(teams_p1_small, teams_p2_small, num_envs=4,
                               obs_space=obs_space, reward_fn=reward_fn)
    obs_p1, obs_p2, masks_p1, masks_p2 = env_small.reset()

    print(f"   Observation shape: {obs_p1['numbers'].shape}")
    # NOTE: NOT calling policy.reset(batch_size=4) here!

    # This should fail with the bug
    print("   Attempting inference WITHOUT calling reset()...")
    try:
        actions = policy.infer(obs_p1, masks_p1)
        print(f"   Actions shape: {actions.shape}")
        print(f"   ✗ Inference succeeded when it should have failed!")
        print("\n" + "="*70)
        print("UNEXPECTED: Bug not reproduced (maybe auto-detection?)")
        print("="*70)
        return False
    except RuntimeError as e:
        print(f"   ✓ Inference FAILED as expected!")
        print(f"   Error: {e}")
        print("\n" + "="*70)
        print("BUG REPRODUCED: Batch size mismatch when reset() not called")
        print("="*70)
        return True

if __name__ == "__main__":
    success = test_batch_size_mismatch_no_reset()
    exit(0 if success else 1)
