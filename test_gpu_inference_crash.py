#!/usr/bin/env python3
"""
Test the actual GPU inference pipeline with PyKMN
This tests the full handoff: PyKMN -> NumPy -> Torch -> GPU -> Model
"""

import gc
import os
import sys
import time
import numpy as np
import torch
import traceback
import psutil

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Pokemon
from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.env.pykmn.policy_runner import LocalPolicyRunner
from metamon.interface import ExpandedObservationSpace, DefaultShapedReward


def get_memory_usage():
    """Get memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return allocated, reserved
    return 0, 0


def create_simple_team():
    """Create a simple hardcoded Gen1 team."""
    team = [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]
    return team


def test_with_model_inference(batch_size: int, num_steps: int = 100, use_gpu: bool = True):
    """Test the full pipeline with actual model inference."""
    print(f"\n{'='*70}")
    print(f"TESTING FULL PIPELINE: batch_size={batch_size}, GPU={use_gpu}")
    print(f"{'='*70}")

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    initial_mem = get_memory_usage()
    initial_gpu_alloc, initial_gpu_res = get_gpu_memory_usage()

    print(f"Initial memory: CPU={initial_mem:.1f} MB")
    if use_gpu:
        print(f"Initial GPU: Allocated={initial_gpu_alloc:.1f} MB, Reserved={initial_gpu_res:.1f} MB")

    try:
        # Create environment
        team = create_simple_team()
        teams = [team] * batch_size

        env = PyKMNVectorEnv(
            num_envs=batch_size,
            teams_p1=teams,
            teams_p2=teams,
            obs_space=ExpandedObservationSpace(),
            reward_fn=DefaultShapedReward(),
            battle_format="gen1ou",
            track_trajectories=False,
        )

        print(f"✓ Environment created with batch_size={batch_size}")

        # Create policy runner with actual model
        print(f"\nLoading model for inference on {device}...")
        policy_p1 = LocalPolicyRunner(
            model_name="MediumRL",  # Use smaller model for testing
            checkpoint=40,
            device=device,
            temperature=1.0,
            use_amp=(device == "cuda"),
            verbose=False,
        )
        policy_p2 = LocalPolicyRunner(
            model_name="MediumRL",
            checkpoint=40,
            device=device,
            temperature=1.0,
            use_amp=(device == "cuda"),
            verbose=False,
        )

        print(f"✓ Models loaded successfully")

        # Reset environment
        obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()

        # Reset policies
        policy_p1.reset(batch_size)
        policy_p2.reset(batch_size)

        print(f"\n{'='*50}")
        print(f"Running {num_steps} steps with GPU inference...")
        print(f"{'='*50}\n")

        battles_completed = 0

        for step in range(num_steps):
            # CRITICAL: This is where PyKMN -> GPU happens
            # obs_p1 is dict with numpy arrays -> converted to torch tensors -> GPU

            # Get actions from models
            actions_p1 = policy_p1.infer(obs_p1, legal_masks_p1)
            actions_p2 = policy_p2.infer(obs_p2, legal_masks_p2)

            # Step environment
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)

            # Update legal masks
            legal_masks_p1 = obs_p1.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))
            legal_masks_p2 = obs_p2.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))

            # Update policy rewards
            policy_p1.update_rewards(rewards_p1)
            policy_p2.update_rewards(rewards_p2)

            # Reset hidden states for done episodes
            policy_p1.reset_hidden_state_for_dones(dones)
            policy_p2.reset_hidden_state_for_dones(dones)

            battles_completed += dones.sum()

            # Progress report
            if step % 20 == 0:
                mem = get_memory_usage()
                gpu_alloc, gpu_res = get_gpu_memory_usage()
                mem_delta = mem - initial_mem
                gpu_alloc_delta = gpu_alloc - initial_gpu_alloc

                print(f"Step {step:3d}: Battles={battles_completed:4d}, "
                      f"CPU Mem={mem:.1f} MB (Δ{mem_delta:+.1f}), "
                      f"GPU={gpu_alloc:.1f}/{gpu_res:.1f} MB (Δ{gpu_alloc_delta:+.1f})")

                # Check critical boundaries
                if battles_completed >= 126 and battles_completed <= 130:
                    print(f"  ⚠️  CROSSING 128 BOUNDARY: {battles_completed} battles")

            # Periodic cleanup
            if step % 50 == 0:
                gc.collect()
                if use_gpu:
                    torch.cuda.empty_cache()

        print(f"\n{'='*70}")
        print(f"✅ TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"  Steps: {num_steps}")
        print(f"  Battles completed: {battles_completed}")
        print(f"  Final CPU memory: {get_memory_usage():.1f} MB")
        if use_gpu:
            gpu_alloc, gpu_res = get_gpu_memory_usage()
            print(f"  Final GPU memory: {gpu_alloc:.1f}/{gpu_res:.1f} MB")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ TEST FAILED")
        print(f"{'='*70}")
        print(f"  Failed at step: {step if 'step' in locals() else 'initialization'}")
        print(f"  Battles completed: {battles_completed if 'battles_completed' in locals() else 0}")
        print(f"  Error: {e}")
        print(f"\nMemory at crash:")
        print(f"  CPU: {get_memory_usage():.1f} MB")
        if use_gpu:
            gpu_alloc, gpu_res = get_gpu_memory_usage()
            print(f"  GPU: {gpu_alloc:.1f}/{gpu_res:.1f} MB")

        print(f"\nFull traceback:")
        traceback.print_exc()

        return False
    finally:
        # Cleanup
        if 'env' in locals():
            env._cleanup_battles_incremental()
        if 'policy_p1' in locals():
            del policy_p1
        if 'policy_p2' in locals():
            del policy_p2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Test various batch sizes with GPU inference."""
    print("=" * 70)
    print("GPU INFERENCE PIPELINE TEST")
    print("=" * 70)
    print("\nThis tests the full handoff:")
    print("  PyKMN -> NumPy arrays -> Torch tensors -> GPU -> Model -> Actions")
    print("\nLooking for crashes at the 128 boundary or memory issues.")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name()}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("\n⚠️  No GPU available, will test CPU inference")

    # Test configurations
    test_configs = [
        (32, 100, True),   # Small batch
        (64, 100, True),   # Medium batch
        (128, 100, True),  # The critical 128
        (256, 50, True),   # Large batch (shorter to avoid timeout)
    ]

    for batch_size, steps, use_gpu in test_configs:
        if not torch.cuda.is_available():
            use_gpu = False

        success = test_with_model_inference(batch_size, steps, use_gpu)

        if not success:
            print(f"\n🔴 CRASH at batch_size={batch_size}!")
            print("This confirms an issue in the GPU inference pipeline.")
            break

        # Clean up between tests
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(2)

    if success:
        print("\n✅ All tests passed! GPU inference pipeline is stable.")
    else:
        print("\n❌ GPU inference pipeline has stability issues.")


if __name__ == "__main__":
    # Set environment variable for cache
    if "METAMON_CACHE_DIR" not in os.environ:
        os.environ["METAMON_CACHE_DIR"] = os.path.expanduser("~/metamon_cache")

    main()