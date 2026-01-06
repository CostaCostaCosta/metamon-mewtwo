#!/usr/bin/env python3
"""
Test to identify if the memory corruption is in the CPU->GPU->CPU transfer pipeline.
"""

import os
import gc
import torch
import numpy as np
import tracemalloc

os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn import PyKMNVectorEnv, load_random_teams, LocalPolicyRunner
from metamon.interface import DefaultObservationSpace, DefaultShapedReward
from pathlib import Path

def test_inference_leak():
    """Test if there's a memory leak in the inference pipeline."""

    print("Testing inference memory leak...")

    # Setup
    cache_dir = Path("/home/eddie/metamon_cache")
    team_dir = cache_dir / "teams" / "smogon_pass2"
    batch_size = 64

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
        track_trajectories=False,  # Disable to isolate inference
    )

    # Create policy
    print("Loading model...")
    policy = LocalPolicyRunner(
        model_name="SyntheticRLV2",
        device="cuda",
    )

    print("Starting memory tracking...")
    tracemalloc.start()

    try:
        # Test repeated inferences with environment resets
        for iteration in range(20):
            print(f"\n=== Iteration {iteration + 1} ===")

            # Reset environment
            obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()

            # Reset policy with same batch size
            policy.reset(batch_size)

            # Run some steps
            for step in range(100):
                # Infer actions
                actions_p1 = policy.infer(obs_p1, masks_p1)
                actions_p2 = policy.infer(obs_p2, masks_p2)

                # Step environment
                obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(
                    actions_p1, actions_p2
                )

                # Update policy rewards
                policy.update_rewards(rewards_p1)

                # Reset hidden states for done episodes
                if dones.any():
                    policy.reset_hidden_state_for_dones(dones)

                if info["num_done"] == batch_size:
                    break

            # Check memory
            current, peak = tracemalloc.get_traced_memory()
            print(f"Memory: {current / 1024 / 1024:.1f} MB (peak: {peak / 1024 / 1024:.1f} MB)")

            # Check GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                print(f"GPU: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

        print("\n✅ SUCCESS: No crash after 20 iterations")

    except Exception as e:
        print(f"\n❌ CRASH at iteration {iteration + 1}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        tracemalloc.stop()
        env.close()

    return True


def test_tensor_lifecycle():
    """Test the lifecycle of tensors between CPU and GPU."""

    print("\nTesting tensor lifecycle...")

    # Create some fake observations like PyKMN would produce
    batch_size = 64
    obs_dict = {
        "numbers": np.random.randn(batch_size, 200).astype(np.float32),
        "text": np.array(["test"] * batch_size),
    }
    legal_mask = np.random.randint(0, 2, (batch_size, 9)).astype(bool)

    print("Creating tensors on GPU...")

    # Simulate what happens in LocalPolicyRunner.infer()
    device = "cuda"

    # Convert to torch (like in infer())
    obs_torch = {}
    for k, v in obs_dict.items():
        if k == "text" or (hasattr(v, 'dtype') and 'str' in str(v.dtype)):
            continue
        elif isinstance(v, np.ndarray):
            obs_torch[k] = torch.from_numpy(v).to(device, non_blocking=True)

    illegal_mask = ~legal_mask
    illegal_mask_trimmed = illegal_mask[:, :9]
    obs_torch["illegal_actions"] = torch.from_numpy(illegal_mask_trimmed).to(
        device, non_blocking=True
    ).bool()

    print(f"Tensors created: {list(obs_torch.keys())}")

    # Add sequence dimension
    obs_torch_seq = {k: v.unsqueeze(1) for k, v in obs_torch.items()}

    # Create RL2 buffer
    action_dim = 9
    rl2_buffer = torch.zeros((batch_size, action_dim + 1), dtype=torch.float32, device=device)
    rl2s_seq = rl2_buffer.unsqueeze(1)

    # Time indices
    time_idxs = torch.zeros((batch_size,), dtype=torch.long, device=device)
    time_idxs_seq = time_idxs.unsqueeze(1).unsqueeze(2)

    print("Tensors prepared for inference")

    # Simulate inference output
    actions = torch.randint(0, 9, (batch_size, 1, 1), device=device)

    # Convert back to numpy
    actions_np = actions.squeeze(-1).squeeze(1).cpu().numpy().astype(np.int32)

    print(f"Actions converted back to numpy: shape={actions_np.shape}")

    # Clean up
    del obs_torch
    del obs_torch_seq
    del rl2_buffer
    del rl2s_seq
    del time_idxs
    del time_idxs_seq
    del actions
    del actions_np

    gc.collect()
    torch.cuda.empty_cache()

    print("✅ Tensor lifecycle test passed")


if __name__ == "__main__":
    # First test tensor lifecycle
    test_tensor_lifecycle()

    # Then test inference memory leak
    success = test_inference_leak()

    import sys
    sys.exit(0 if success else 1)