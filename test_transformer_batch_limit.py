"""
Test transformer model inference with various batch sizes to find limitations.

This bypasses the pypkmn Battle objects entirely and directly tests the
AMAGO transformer model with synthetic observations.
"""

import sys
import torch
import numpy as np
from metamon.rl.pretrained import get_pretrained_model


def test_batch_inference(model_name: str, batch_size: int, device: str = "cuda", verbose: bool = True):
    """Test inference with a specific batch size.

    Args:
        model_name: Pretrained model name
        batch_size: Batch size to test
        device: Device to use
        verbose: Print progress

    Returns:
        True if successful, False if failed
    """
    if verbose:
        print(f"\n=== Testing batch size: {batch_size} on {device} ===")

    try:
        # Load model
        if verbose:
            print(f"Loading model: {model_name}")
        pretrained_cls = get_pretrained_model(model_name)
        experiment = pretrained_cls.initialize_agent(checkpoint=None, log=False)
        agent = experiment.policy
        agent.eval()
        agent = agent.to(device)

        # Get obs space dimensions
        obs_space = pretrained_cls.observation_space
        action_dim = pretrained_cls.action_space.gym_space.n

        if verbose:
            print(f"Action dim: {action_dim}")
            print(f"Observation space: {type(obs_space).__name__}")

        # Create synthetic batch observations
        if verbose:
            print(f"Creating synthetic observations for batch_size={batch_size}...")

        # Create dummy observations matching the observation space format
        # Most metamon models use numeric + text observations
        obs_batch = {}

        # Numeric features (vary by obs space, but typically ~100-200 dims)
        if hasattr(obs_space, 'numbers_dim'):
            numeric_dim = obs_space.numbers_dim()
            obs_batch['numbers'] = torch.randn(batch_size, numeric_dim, dtype=torch.float32, device=device)
            if verbose:
                print(f"  numbers: {obs_batch['numbers'].shape}")

        # Text features (tokenized)
        if hasattr(obs_space, 'text_tokenizer'):
            # Create dummy tokenized text (max_len=512 typical)
            max_text_len = 512
            obs_batch['text'] = torch.randint(0, 100, (batch_size, max_text_len), dtype=torch.long, device=device)
            if verbose:
                print(f"  text: {obs_batch['text'].shape}")

        # Legal action mask
        legal_mask = torch.ones(batch_size, action_dim, dtype=torch.bool, device=device)

        # RL2 features (prev action + prev reward)
        rl2_input = torch.zeros(batch_size, 1, action_dim + 1, dtype=torch.float32, device=device)  # Add seq dim

        # Time indices (per-env counters)
        time_idxs = torch.zeros(batch_size, 1, 1, dtype=torch.long, device=device)  # (batch, seq=1, 1)

        # Legal action mask (illegal actions = inverted)
        obs_batch["illegal_actions"] = torch.zeros(batch_size, action_dim, dtype=torch.bool, device=device)

        # Add sequence dimension to obs: (N, ...) -> (N, 1, ...)
        obs_batch_seq = {k: v.unsqueeze(1) if v.ndim >= 1 else v for k, v in obs_batch.items()}

        # Initialize hidden state
        hidden_state = agent.traj_encoder.init_hidden_state(batch_size, device)

        if verbose:
            print(f"Running inference with get_actions()...")

        # Run inference using get_actions (the proper AMAGO interface)
        with torch.inference_mode():
            actions, new_hidden = agent.get_actions(
                obs=obs_batch_seq,
                rl2s=rl2_input,
                time_idxs=time_idxs,
                hidden_state=hidden_state,
                sample=True,
            )

            if verbose:
                print(f"✓ Inference successful! Actions shape: {actions.shape}")

        # Cleanup
        del agent, experiment, obs_batch, legal_mask, rl2_input, time_idxs, actions, new_hidden, hidden_state
        torch.cuda.empty_cache()

        if verbose:
            print("✓ Cleanup successful")

        return True

    except Exception as e:
        print(f"✗ FAILED with batch_size={batch_size}")
        print(f"  Error: {e}")
        print(f"  Type: {type(e).__name__}")

        # Cleanup
        try:
            torch.cuda.empty_cache()
        except:
            pass

        return False


def main():
    """Run progressive batch size tests."""
    model_name = "SyntheticRLV2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(sys.argv) > 1:
        # Test specific batch size
        batch_size = int(sys.argv[1])
        if len(sys.argv) > 2:
            model_name = sys.argv[2]
        success = test_batch_inference(model_name, batch_size, device, verbose=True)
        sys.exit(0 if success else 1)

    # Progressive testing
    print("="*70)
    print(f"TRANSFORMER BATCH SIZE LIMIT TEST")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("="*70)

    test_sizes = [1, 4, 8, 16, 32, 64, 80, 96, 112, 128, 144, 160, 192, 256]

    for size in test_sizes:
        success = test_batch_inference(model_name, size, device, verbose=True)
        if not success:
            print(f"\n⚠️  Found limit around batch size {size}")

            # Binary search for exact limit
            if test_sizes.index(size) > 0:
                low = test_sizes[test_sizes.index(size) - 1]
                high = size

                print(f"\nBinary searching between {low} and {high}...")
                while low < high - 1:
                    mid = (low + high) // 2
                    if test_batch_inference(model_name, mid, device, verbose=False):
                        print(f"  ✓ {mid} works")
                        low = mid
                    else:
                        print(f"  ✗ {mid} fails")
                        high = mid

                print(f"\n🎯 Maximum working batch size: {low}")
            break

        print()


if __name__ == "__main__":
    main()
