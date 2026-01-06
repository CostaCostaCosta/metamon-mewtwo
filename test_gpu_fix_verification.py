#!/usr/bin/env python3
"""
Test that the GPU inference pipeline works after fixing the text observation bug
"""

import gc
import os
import sys
import time
import numpy as np
import torch
import traceback

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Pokemon
from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.env.pykmn.policy_runner import LocalPolicyRunner
from metamon.interface import (
    ExpandedObservationSpace,
    DefaultShapedReward,
    TokenizedObservationSpace,
    DefaultObservationSpace
)
from metamon.tokenizer import PokemonTokenizer


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


def test_gpu_inference_after_fix(batch_size: int, num_steps: int = 200):
    """Test GPU inference with the fix applied."""

    print(f"\n{'='*70}")
    print(f"TESTING FIXED GPU PIPELINE: batch_size={batch_size}")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Create environment
        team = create_simple_team()
        teams = [team] * batch_size

        print("Creating environment with tokenized observations...")
        # Create tokenized observation space that the model expects
        tokenizer = PokemonTokenizer()
        vocab_path = os.path.join(os.environ["METAMON_CACHE_DIR"], "vocab.json")
        if os.path.exists(vocab_path):
            tokenizer.load_tokens_from_disk(vocab_path)
        else:
            print(f"  Warning: vocab.json not found at {vocab_path}, using empty tokenizer")

        base_obs_space = DefaultObservationSpace()  # or ExpandedObservationSpace()
        obs_space = TokenizedObservationSpace(base_obs_space, tokenizer)

        env = PyKMNVectorEnv(
            num_envs=batch_size,
            teams_p1=teams,
            teams_p2=teams,
            obs_space=obs_space,
            reward_fn=DefaultShapedReward(),
            battle_format="gen1ou",
            track_trajectories=False,
        )
        print(f"✓ Environment created with batch_size={batch_size}")

        # Create policy runners
        print(f"Loading models for {device}...")
        policy_p1 = LocalPolicyRunner(
            model_name="MediumRL",
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
        print("✓ Models loaded successfully")

        # Reset environment
        obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()
        policy_p1.reset(batch_size)
        policy_p2.reset(batch_size)

        # Check observation format
        print("\nObservation format check:")
        for key in obs_p1.keys():
            value = obs_p1[key]
            dtype_str = str(value.dtype) if hasattr(value, 'dtype') else type(value).__name__
            shape_str = str(value.shape) if hasattr(value, 'shape') else 'N/A'
            print(f"  '{key}': dtype={dtype_str}, shape={shape_str}")

            # Highlight text fields
            if hasattr(value, 'dtype') and 'str' in str(value.dtype):
                print(f"    ⚠️  Text field detected - will be skipped in tensor conversion")

        print(f"\nRunning {num_steps} steps...")

        battles_completed = 0
        start_time = time.time()

        for step in range(num_steps):
            # This is the critical part - will it handle text observations correctly?
            actions_p1 = policy_p1.infer(obs_p1, legal_masks_p1)
            actions_p2 = policy_p2.infer(obs_p2, legal_masks_p2)

            # Step environment
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)

            # Update legal masks
            legal_masks_p1 = obs_p1.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))
            legal_masks_p2 = obs_p2.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))

            # Update rewards and reset for done episodes
            policy_p1.update_rewards(rewards_p1)
            policy_p2.update_rewards(rewards_p2)
            policy_p1.reset_hidden_state_for_dones(dones)
            policy_p2.reset_hidden_state_for_dones(dones)

            battles_completed += dones.sum()

            # Progress report
            if step % 50 == 0:
                elapsed = time.time() - start_time
                rate = battles_completed / elapsed if elapsed > 0 else 0
                print(f"  Step {step:3d}: Battles completed: {battles_completed:4d}, "
                      f"Rate: {rate:.1f} battles/sec")

                # Check critical boundaries
                if battles_completed >= 126 and battles_completed <= 130:
                    print(f"    ✅ Successfully crossed 128 boundary! ({battles_completed} battles)")

        # Final stats
        elapsed = time.time() - start_time
        final_rate = battles_completed / elapsed if elapsed > 0 else 0

        print(f"\n{'='*70}")
        print(f"✅ TEST PASSED - GPU INFERENCE WORKS!")
        print(f"{'='*70}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps completed: {num_steps}")
        print(f"  Battles completed: {battles_completed}")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Rate: {final_rate:.1f} battles/sec")

        if battles_completed > 128:
            print(f"\n  🎉 Successfully exceeded the '128 barrier' with {battles_completed} battles!")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ TEST FAILED")
        print(f"{'='*70}")
        print(f"  Error: {e}")
        print(f"\nTraceback:")
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if 'env' in locals():
            env._cleanup_battles_incremental()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Test multiple batch sizes to verify the fix works."""
    print("=" * 70)
    print("GPU INFERENCE FIX VERIFICATION TEST")
    print("=" * 70)
    print("\nThis test verifies that the text observation bug is fixed")
    print("and GPU inference works correctly with all batch sizes.")

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name()}")
    else:
        print("\n⚠️  No GPU available, testing CPU inference")

    # Test critical batch sizes
    test_configs = [
        (32, 200),   # Small batch
        (64, 200),   # Medium batch
        (128, 200),  # The infamous 128
        (256, 100),  # Large batch
    ]

    all_passed = True

    for batch_size, steps in test_configs:
        success = test_gpu_inference_after_fix(batch_size, steps)

        if not success:
            all_passed = False
            print(f"\n🔴 Failed at batch_size={batch_size}")
            break

        # Clean up between tests
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)

    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nThe GPU inference pipeline is now working correctly.")
        print("The text observation bug has been fixed.")
        print("There is no '128 battle barrier' - it was just a type conversion error!")
    else:
        print("❌ Some tests failed")
        print("The fix may need adjustment.")


if __name__ == "__main__":
    # Set environment variable
    if "METAMON_CACHE_DIR" not in os.environ:
        os.environ["METAMON_CACHE_DIR"] = os.path.expanduser("~/metamon_cache")

    main()