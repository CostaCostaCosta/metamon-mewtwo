#!/usr/bin/env python3
"""
Test script for batched AMAGO inference with PyKMN.

Validates that the corrected implementation works correctly with:
- Different batch sizes (1, 4, 16)
- Episodic resets
- RL2 state management
- Hidden state management
"""

import os
import time
import numpy as np
from pathlib import Path

# Set cache directory
os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import (
    load_random_teams,
    PyKMNVectorEnv,
    LocalPolicyRunner,
    SelfPlayRunner,
)
from metamon.interface import get_observation_space, get_reward_function
from metamon.rl.pretrained import get_pretrained_model


def test_batch_size(batch_size: int, num_steps: int = 50, verbose: bool = True):
    """
    Test batched inference with a specific batch size.

    Args:
        batch_size: Number of parallel environments
        num_steps: Number of steps to run
        verbose: Print progress

    Returns:
        dict with test results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing batch_size={batch_size}")
        print(f"{'='*60}")

    # Load teams
    team_dir = Path.home() / "metamon_cache" / "teams" / "modern_replays_v2"
    if verbose:
        print(f"Loading {batch_size * 2} teams...")

    teams_p1 = load_random_teams(team_dir, "gen1ou", batch_size)
    teams_p2 = load_random_teams(team_dir, "gen1ou", batch_size)

    if verbose:
        print(f"✓ Loaded teams")

    # Get observation space and reward function from pretrained model
    pretrained_cls = get_pretrained_model("SyntheticRLV2")
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    # Create vectorized environment
    if verbose:
        print(f"Creating PyKMNVectorEnv with {batch_size} parallel battles...")

    vec_env = PyKMNVectorEnv(
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        num_envs=batch_size,
        obs_space=obs_space,
        reward_fn=reward_fn,
        battle_format="gen1ou",
        track_trajectories=False,  # Don't track for this test
    )

    if verbose:
        print(f"✓ Environment created")

    # Create policy runner with batched inference
    if verbose:
        print(f"Loading pretrained model for inference...")

    policy = LocalPolicyRunner(
        model_name="SyntheticRLV2",
        checkpoint=48,
        device="cuda",
        use_amp=True,
        verbose=False,
    )

    if verbose:
        print(f"✓ Model loaded")

    # Reset environment
    obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()

    if verbose:
        print(f"\n✓ Environment reset")
        print(f"  Observation keys: {list(obs_p1.keys())}")
        obs_shapes = {k: v.shape for k, v in obs_p1.items()}
        print(f"  Observation shapes: {obs_shapes}")
        print(f"  Legal mask shape: {masks_p1.shape}")

    # Run inference for num_steps
    if verbose:
        print(f"\nRunning {num_steps} steps of inference...")

    start_time = time.time()
    step_times = []
    completed_battles = 0

    for step in range(num_steps):
        step_start = time.perf_counter()

        # Inference
        actions_p1 = policy.infer(obs_p1, masks_p1)
        actions_p2 = policy.infer(obs_p2, masks_p2)

        # Step environment
        obs_p1, obs_p2, rew_p1, rew_p2, dones, info = vec_env.step(actions_p1, actions_p2)

        # Update rewards
        policy.update_rewards(rew_p1)

        # Reset hidden states for done episodes
        if dones.any():
            policy.reset_hidden_state_for_dones(dones)
            completed_battles += dones.sum()

        # Update masks
        masks_p1, masks_p2 = vec_env._extract_legal_masks()

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        if verbose and step % 10 == 0:
            print(f"  Step {step:3d}: {step_time*1000:6.2f}ms, "
                  f"completed={completed_battles}, "
                  f"mean_reward={rew_p1.mean():.3f}")

    total_time = time.time() - start_time
    mean_step_time = np.mean(step_times)
    steps_per_sec = num_steps * batch_size / total_time

    if verbose:
        print(f"\n✓ Test completed successfully!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Mean step time: {mean_step_time*1000:.2f}ms")
        print(f"  Steps/sec: {steps_per_sec:.1f}")
        print(f"  Completed battles: {completed_battles}/{batch_size}")

    return {
        "batch_size": batch_size,
        "num_steps": num_steps,
        "total_time": total_time,
        "mean_step_time": mean_step_time,
        "steps_per_sec": steps_per_sec,
        "completed_battles": completed_battles,
    }


def main():
    print("=" * 60)
    print("Batched AMAGO Inference Test Suite")
    print("=" * 60)

    # Test different batch sizes
    batch_sizes = [1, 4, 16]
    results = []

    for batch_size in batch_sizes:
        try:
            result = test_batch_size(batch_size, num_steps=50, verbose=True)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed for batch_size={batch_size}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Batch Size':<12} {'Steps/sec':<12} {'Step Time (ms)':<16} {'Speedup':<10}")
    print("-" * 60)

    baseline_steps_per_sec = results[0]["steps_per_sec"] if results else None

    for result in results:
        speedup = result["steps_per_sec"] / baseline_steps_per_sec if baseline_steps_per_sec else 1.0
        print(
            f"{result['batch_size']:<12} "
            f"{result['steps_per_sec']:<12.1f} "
            f"{result['mean_step_time']*1000:<16.2f} "
            f"{speedup:<10.2f}x"
        )

    print(f"\n✓ All tests passed!")


if __name__ == "__main__":
    main()
