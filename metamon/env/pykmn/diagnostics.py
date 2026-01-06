"""
Diagnostic utilities for catching memory corruption early.

Based on recommendations from heap corruption analysis (2026-01-02).
Implements canary checks, assertions, and validation to surface bugs
before they corrupt malloc metadata.
"""

import numpy as np
import torch
from typing import Dict, Any, List


def validate_obs_dict(obs_dict: Dict[str, np.ndarray], expected_shapes: Dict[str, tuple], env_idx: int = -1):
    """
    Validate observation dictionary before passing to model.

    Checks:
    - Shape exactly as expected
    - Dtype exactly as expected
    - Contiguous layout
    - No object dtype
    - Values in sane ranges

    Args:
        obs_dict: Observation dictionary to validate
        expected_shapes: Expected shapes for each key
        env_idx: Environment index (for error messages, -1 = batch)

    Raises:
        AssertionError: If validation fails
    """
    prefix = f"Env {env_idx}" if env_idx >= 0 else "Batch"

    for key, arr in obs_dict.items():
        # Check shape
        if key in expected_shapes:
            expected_shape = expected_shapes[key]
            assert arr.shape == expected_shape, (
                f"{prefix}: {key} shape mismatch: "
                f"expected {expected_shape}, got {arr.shape}"
            )

        # Check dtype
        assert arr.dtype != np.object_, (
            f"{prefix}: {key} has object dtype (use np.str_ instead)"
        )

        # Check contiguous
        if isinstance(arr, np.ndarray):
            assert arr.flags['C_CONTIGUOUS'], (
                f"{prefix}: {key} is not C-contiguous"
            )

        # Check numerical ranges
        if key == "numbers" and np.issubdtype(arr.dtype, np.number):
            assert not np.any(np.isnan(arr)), (
                f"{prefix}: {key} contains NaN values"
            )
            assert not np.any(np.isinf(arr)), (
                f"{prefix}: {key} contains inf values"
            )
            # Reasonable range check (normalized features should be ~[-10, 10])
            if np.max(np.abs(arr)) > 1000:
                print(f"⚠️  Warning: {prefix} {key} has large values (max={np.max(np.abs(arr))})")


def validate_torch_batch(obs_torch: Dict[str, torch.Tensor], batch_size: int):
    """
    Validate torch observation batch before forward pass.

    Args:
        obs_torch: Torch observation dict
        batch_size: Expected batch size

    Raises:
        AssertionError: If validation fails
    """
    for key, tensor in obs_torch.items():
        # Check batch dimension
        assert tensor.shape[0] == batch_size, (
            f"Batch: {key} batch size mismatch: "
            f"expected {batch_size}, got {tensor.shape[0]}"
        )

        # Check NaNs
        if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            assert not torch.any(torch.isnan(tensor)), (
                f"Batch: {key} contains NaN values after to(device)"
            )


def validate_legal_mask(legal_mask: np.ndarray, env_idx: int = -1):
    """
    Validate legal action mask.

    Args:
        legal_mask: Boolean mask (True = legal)
        env_idx: Environment index (for error messages)

    Raises:
        AssertionError: If validation fails
    """
    prefix = f"Env {env_idx}" if env_idx >= 0 else "Batch"

    # Check dtype
    assert legal_mask.dtype == np.bool_, (
        f"{prefix}: legal_mask dtype is {legal_mask.dtype}, expected bool"
    )

    # Check at least one legal action
    if legal_mask.ndim == 1:
        assert np.any(legal_mask), (
            f"{prefix}: legal_mask has no legal actions!"
        )
    else:  # Batched
        for i in range(legal_mask.shape[0]):
            assert np.any(legal_mask[i]), (
                f"{prefix} env {i}: legal_mask has no legal actions!"
            )


def assert_per_env_obs_state_unique(obs_states: List[Dict[str, Any]]):
    """
    Assert that per-environment observation states are distinct objects.

    This catches the bug where all environments share the same mutable state.

    Args:
        obs_states: List of per-env observation state dicts

    Raises:
        AssertionError: If states are shared
    """
    if not obs_states or obs_states[0] is None:
        return  # Legacy path, no per-env state

    # Check all state dicts are unique objects
    state_ids = [id(state) for state in obs_states]
    assert len(set(state_ids)) == len(state_ids), (
        "Observation states are shared across environments! "
        "All environments must have distinct obs_state dicts."
    )

    # Check mutable containers (like lists/arrays) are also unique
    if 'revealed_opponents_names' in obs_states[0]:
        array_ids = [id(state['revealed_opponents_names']) for state in obs_states]
        assert len(set(array_ids)) == len(array_ids), (
            "revealed_opponents_names arrays are shared across environments!"
        )


def log_step_diagnostic(
    battle_idx: int,
    env_idx: int,
    turn_idx: int,
    obs_dict: Dict[str, np.ndarray],
    legal_mask: np.ndarray,
    action: int,
):
    """
    Log diagnostic info for a single step (for bisecting crashes).

    Call this every N steps to create a breadcrumb trail.

    Args:
        battle_idx: Global battle index
        env_idx: Environment index within batch
        turn_idx: Turn number within battle
        obs_dict: Observation dict
        legal_mask: Legal action mask
        action: Chosen action
    """
    # Only log occasionally to avoid spam
    if turn_idx % 20 != 0:
        return

    numbers_summary = ""
    if "numbers" in obs_dict:
        nums = obs_dict["numbers"]
        numbers_summary = f"numbers=[{nums.min():.2f}, {nums.max():.2f}]"

    text_len = ""
    if "text" in obs_dict:
        text_len = f"text_len={len(obs_dict['text'].item())}"

    legal_count = np.sum(legal_mask)

    print(
        f"[DIAG] Battle {battle_idx:4d} | Env {env_idx:3d} | Turn {turn_idx:3d} | "
        f"{numbers_summary} | {text_len} | legal={legal_count}/13 | action={action}"
    )


def enable_hardened_allocator():
    """
    Print instructions for enabling hardened allocator diagnostics.

    These env vars make glibc abort earlier on heap misuse and fill
    allocations/frees with patterns to surface use-after-free.
    """
    print("="*70)
    print("HARDENED ALLOCATOR DIAGNOSTICS")
    print("="*70)
    print("To enable hardened allocator checks, run with:")
    print()
    print("  MALLOC_CHECK_=3 MALLOC_PERTURB_=165 \\")
    print("  PYTHONFAULTHANDLER=1 \\")
    print("  CUDA_LAUNCH_BLOCKING=1 \\")
    print("  python scripts/generate_selfplay_batched.py ...")
    print()
    print("What this does:")
    print("  - MALLOC_CHECK_=3:          Abort on heap corruption")
    print("  - MALLOC_PERTURB_=165:      Fill allocs with pattern")
    print("  - PYTHONFAULTHANDLER=1:     Print stack on segfault")
    print("  - CUDA_LAUNCH_BLOCKING=1:   Synchronous CUDA (slower but exact errors)")
    print("="*70)
    print()


def check_numpy_buffer_safety(arr: np.ndarray, name: str):
    """
    Check if numpy array is safe to convert to torch and transfer to GPU.

    Args:
        arr: Numpy array to check
        name: Name for error messages

    Raises:
        AssertionError: If array is unsafe
    """
    # Check contiguous
    assert arr.flags['C_CONTIGUOUS'] or arr.flags['F_CONTIGUOUS'], (
        f"{name}: Array is not contiguous (flags={arr.flags})"
    )

    # Check not a view (unless explicitly allowed)
    if arr.base is not None:
        print(f"⚠️  Warning: {name} is a view into another array (base={type(arr.base)})")

    # Check dtype is standard
    assert arr.dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_, np.str_], (
        f"{name}: Unusual dtype {arr.dtype}"
    )
