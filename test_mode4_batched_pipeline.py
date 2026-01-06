#!/usr/bin/env python3
"""
Mode 4: Full Batched Pipeline with Chunking Equivalence
Purpose: Test complete vectorized batching operations
Critical: Verify chunking equivalence (batch 128 = 2x batch 64)
"""

import gc
import hashlib
import random
import sys
import traceback
import time
from typing import Any, Dict, List
import numpy as np
import torch

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Player, Choice, MoveChoice, PassChoice, SwitchChoice
from metamon.data.team_repo import load_teams_for_format
from metamon.env.pykmn.feature_extractor import (
    pykmn_to_features_raw,
    features_to_universal_state,
    create_gen1_mappings
)
from metamon.interface import ExpandedObservationSpace


def get_legal_action(battle: Battle, player: Player) -> Choice:
    """Get a random legal action for the player."""
    legal_moves = []
    for slot in range(4):
        if battle.possible(player, MoveChoice(slot)):
            legal_moves.append(MoveChoice(slot))

    legal_switches = []
    for slot in range(1, 6):
        if battle.possible(player, SwitchChoice(slot)):
            legal_switches.append(SwitchChoice(slot))

    all_legal = legal_moves + legal_switches
    if not all_legal:
        return PassChoice()

    return random.choice(all_legal)


def validate_array_properties(arrays: List[np.ndarray], prefix: str = "") -> None:
    """Validate array properties before stacking."""
    if not arrays:
        return

    first = arrays[0]
    expected_dtype = first.dtype
    expected_shape = first.shape

    for i, arr in enumerate(arrays):
        # Check dtype consistency
        if arr.dtype != expected_dtype:
            raise ValueError(f"{prefix} Array {i} has dtype {arr.dtype}, expected {expected_dtype}")

        # Check shape consistency
        if arr.shape != expected_shape:
            raise ValueError(f"{prefix} Array {i} has shape {arr.shape}, expected {expected_shape}")

        # Check contiguity
        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError(f"{prefix} Array {i} is not C-contiguous")

        # Check memory sharing
        for j in range(i):
            if np.shares_memory(arr, arrays[j]):
                raise ValueError(f"{prefix} Arrays {i} and {j} share memory")


def test_chunking_equivalence(batch_size: int, obs_list: List[Dict]) -> bool:
    """Test that different chunking strategies produce identical results."""
    if batch_size < 2:
        return True

    # Extract arrays
    arrays = [obs["numbers"] for obs in obs_list[:batch_size]]

    # Strategy 1: Stack all at once
    batch_full = np.stack(arrays)

    # Strategy 2: Stack in halves
    mid = batch_size // 2
    batch_first_half = np.stack(arrays[:mid])
    batch_second_half = np.stack(arrays[mid:batch_size])
    batch_concat = np.concatenate([batch_first_half, batch_second_half], axis=0)

    # Verify identical
    if not np.array_equal(batch_full, batch_concat):
        diff = np.abs(batch_full - batch_concat).max()
        raise ValueError(f"Chunking not equivalent! Max diff: {diff}")

    # Strategy 3: Stack individually then vstack
    individual_stacks = [arr[np.newaxis, ...] for arr in arrays]
    batch_vstack = np.vstack(individual_stacks)

    if not np.array_equal(batch_full, batch_vstack):
        diff = np.abs(batch_full - batch_vstack).max()
        raise ValueError(f"Vstack not equivalent! Max diff: {diff}")

    return True


def test_batched_pipeline(batch_sizes: List[int], steps_per_size: int = 2000) -> dict:
    """Test full batched pipeline with validation."""
    print("\nTesting full batched pipeline with chunking equivalence")
    print("-" * 50)

    results = {}
    teams = load_teams_for_format("gen1ou")
    mappings = create_gen1_mappings()
    obs_space = ExpandedObservationSpace()

    for batch_size in batch_sizes:
        print(f"\n  Testing batch_size={batch_size}...")

        # Create battles
        battles = []
        obs_states = []
        for i in range(batch_size):
            team_idx = i % len(teams)
            battle = Battle(
                p1_team=teams[team_idx].to_pykmn(),
                p2_team=teams[team_idx].to_pykmn(),
                p1_seed=42 + i,
                p2_seed=142 + i
            )
            battles.append(battle)
            obs_states.append(None)

        try:
            start_time = time.time()

            for step in range(steps_per_size):
                obs_list = []

                # Collect observations
                for i, battle in enumerate(battles):
                    # Get actions and update
                    c1 = get_legal_action(battle, Player.P1)
                    c2 = get_legal_action(battle, Player.P2)
                    result, _ = battle.update(c1, c2)

                    # Extract features
                    features_p1 = pykmn_to_features_raw(battle, result, Player.P1, mappings)

                    # Convert to universal state
                    state_p1 = features_to_universal_state(features_p1, mappings)

                    # Get observation
                    obs_p1 = obs_space(state_p1, obs_states[i])
                    obs_states[i] = obs_p1

                    obs_list.append(obs_p1)

                    # Reset if terminal
                    if result.type() != 0:
                        team_idx = random.randint(0, len(teams) - 1)
                        battles[i] = Battle(
                            p1_team=teams[team_idx].to_pykmn(),
                            p2_team=teams[team_idx].to_pykmn(),
                            p1_seed=random.randint(0, 1000000),
                            p2_seed=random.randint(0, 1000000)
                        )
                        obs_states[i] = None

                # Validate array properties before batching
                arrays = [obs["numbers"] for obs in obs_list]
                validate_array_properties(arrays, f"Step {step}: ")

                # Test batching operations
                batched_obs = {}
                for key in obs_list[0].keys():
                    if key == "numbers":
                        # Numeric arrays - stack them
                        values = [obs[key] for obs in obs_list]
                        batched_obs[key] = np.stack(values)
                    elif key == "words":
                        # Word arrays - stack them
                        values = [obs[key] for obs in obs_list]
                        batched_obs[key] = np.stack(values)
                    elif key == "legal_actions_mask":
                        # Boolean masks - stack them
                        values = [obs[key] for obs in obs_list]
                        batched_obs[key] = np.stack(values)
                    else:
                        # Other types - just collect
                        batched_obs[key] = [obs[key] for obs in obs_list]

                # Test chunking equivalence every 100 steps
                if step % 100 == 0:
                    test_chunking_equivalence(batch_size, obs_list)

                # Convert to torch tensors (full pipeline)
                torch_batched = {}
                with torch.no_grad():
                    for key, value in batched_obs.items():
                        if isinstance(value, np.ndarray):
                            # Test both copy and view creation
                            tensor_copy = torch.tensor(value, dtype=torch.float32)
                            tensor_view = torch.from_numpy(value.astype(np.float32))

                            # Verify shapes
                            assert tensor_copy.shape[0] == batch_size, f"Batch dim mismatch: {tensor_copy.shape}"
                            assert tensor_view.shape[0] == batch_size, f"Batch dim mismatch: {tensor_view.shape}"

                            torch_batched[key] = tensor_copy

                # Progress report
                if step % 500 == 0 and step > 0:
                    elapsed = time.time() - start_time
                    rate = (step * batch_size) / elapsed
                    print(f"    Step {step}/{steps_per_size}, {rate:.1f} env steps/sec")

                    # Memory check
                    if step % 1000 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            results[batch_size] = "PASS"
            print(f"  ✓ batch_size={batch_size} PASSED (batching stable, chunking equivalent)")

        except Exception as e:
            results[batch_size] = f"FAIL: {str(e)}"
            print(f"  ✗ batch_size={batch_size} FAILED at step {step}")
            print(f"    Error: {e}")

            if "chunking" in str(e).lower():
                print("    CRITICAL: Chunking not equivalent - batching is non-deterministic!")
            if "memory" in str(e).lower():
                print("    CRITICAL: Memory sharing detected in batch!")
            if batch_size >= 128:
                print("    Note: Failed at/above 128 boundary!")

            traceback.print_exc()

    return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("PyKMN Stability Test - Mode 4: Full Batched Pipeline")
    print("=" * 60)

    # Set deterministic seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Test batch sizes
    batch_sizes = [1, 32, 64, 96, 127, 128, 129, 144, 256]

    print("\nTesting full batched pipeline with chunking equivalence...")
    print("Validates: array properties, memory sharing, chunking determinism")

    # Run tests
    results = test_batched_pipeline(batch_sizes, steps_per_size=1000)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nResults:")
    all_passed = True
    first_failure = None

    for batch_size in batch_sizes:
        result = results.get(batch_size, "NOT RUN")
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} batch_size={batch_size}: {result}")

        if result != "PASS" and first_failure is None:
            first_failure = batch_size
            all_passed = False

    if all_passed:
        print("\n✓ Mode 4 PASSED: Full batched pipeline stable, chunking deterministic")
        sys.exit(0)
    else:
        print(f"\n✗ Mode 4 FAILED: First failure at batch_size={first_failure}")
        if first_failure == 128:
            print("  CRITICAL: Failure at exactly 128 - batching/stacking issue")
        print("\nRecommendation: Check Mode 1-3 results to isolate the layer")
        sys.exit(1)


if __name__ == "__main__":
    main()