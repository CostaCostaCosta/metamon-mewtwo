"""
Test LocalPolicyRunner with various batch sizes to isolate transformer vs Battle limit.

This uses the actual LocalPolicyRunner (which works) with synthetic observations
to test if the transformer model has batch size limits independent of Battle objects.
"""

import sys
import numpy as np
import torch
from metamon.env.pykmn.policy_runner import LocalPolicyRunner


def test_policy_runner_batch(model_name: str, batch_size: int, device: str = "cuda", verbose: bool = True):
    """Test LocalPolicyRunner with a specific batch size using synthetic observations.

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
        # Create policy runner
        if verbose:
            print(f"Loading LocalPolicyRunner with {model_name}...")

        runner = LocalPolicyRunner(
            model_name=model_name,
            checkpoint=None,
            device=device,
            temperature=1.0,
            use_amp=True,
            verbose=False,
        )

        # Reset for this batch size
        runner.reset(batch_size=batch_size)

        action_dim = runner.action_dim
        if verbose:
            print(f"Action dim: {action_dim}")

        # Create synthetic observations
        # TokenizedObservationSpace uses "text_tokens"
        if verbose:
            print(f"Creating synthetic observations for batch_size={batch_size}...")

        obs_dict = {
            "text_tokens": np.random.randint(0, 100, (batch_size, 512), dtype=np.int64),
        }

        # Legal action mask (all actions legal)
        legal_mask = np.ones((batch_size, action_dim), dtype=bool)

        # Run inference
        if verbose:
            print(f"Running inference...")

        actions = runner.infer(obs_dict, legal_mask)

        if verbose:
            print(f"✓ Inference successful! Actions shape: {actions.shape}")

        # Cleanup
        del runner, obs_dict, legal_mask, actions
        torch.cuda.empty_cache()

        if verbose:
            print("✓ Cleanup successful")

        return True

    except Exception as e:
        print(f"✗ FAILED with batch_size={batch_size}")
        print(f"  Error: {e}")
        print(f"  Type: {type(e).__name__}")

        # Print traceback for first failure
        if verbose:
            import traceback
            traceback.print_exc()

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
        success = test_policy_runner_batch(model_name, batch_size, device, verbose=True)
        sys.exit(0 if success else 1)

    # Progressive testing
    print("="*70)
    print(f"LOCAL POLICY RUNNER BATCH SIZE LIMIT TEST")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("="*70)

    test_sizes = [1, 4, 8, 16, 32, 64, 80, 96, 112, 128, 144, 160, 192, 256]

    for size in test_sizes:
        success = test_policy_runner_batch(model_name, size, device, verbose=True)
        if not success:
            print(f"\n⚠️  Found limit around batch size {size}")

            # Binary search for exact limit
            if test_sizes.index(size) > 0:
                low = test_sizes[test_sizes.index(size) - 1]
                high = size

                print(f"\nBinary searching between {low} and {high}...")
                while low < high - 1:
                    mid = (low + high) // 2
                    if test_policy_runner_batch(model_name, mid, device, verbose=False):
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
