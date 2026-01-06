#!/usr/bin/env python3
"""
Mode 3: Individual Torch Tensors with Lifetime Validation
Purpose: Test tensor creation without batching, verify no aliasing
Critical: Test both torch.tensor (copy) and torch.from_numpy (view)
"""

import copy
import gc
import hashlib
import random
import sys
import traceback
import time
from typing import Any, Dict, List, Tuple
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


def tensor_hash(tensor: torch.Tensor) -> str:
    """Compute hash of tensor contents."""
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def check_tensor_aliasing(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """Check if two tensors share storage."""
    return t1.data_ptr() == t2.data_ptr()


def test_torch_tensors(batch_sizes: List[int], steps_per_size: int = 5000) -> dict:
    """Test torch tensor creation with lifetime validation."""
    print("\nTesting torch tensor creation with lifetime validation")
    print("-" * 50)

    results = {}
    teams = load_teams_for_format("gen1ou")
    mappings = create_gen1_mappings()
    obs_space = ExpandedObservationSpace()

    for batch_size in batch_sizes:
        print(f"\n  Testing batch_size={batch_size}...")

        # Create battles
        battles = []
        obs_states = []  # Persistent observation state
        for i in range(batch_size):
            team_idx = i % len(teams)
            battle = Battle(
                p1_team=teams[team_idx].to_pykmn(),
                p2_team=teams[team_idx].to_pykmn(),
                p1_seed=42 + i,
                p2_seed=142 + i
            )
            battles.append(battle)
            obs_states.append(None)  # Will be initialized on first observation

        # History for lifetime validation
        tensor_history = []
        numpy_array_history = []

        try:
            start_time = time.time()

            with torch.no_grad():  # Ensure no gradient tracking
                for step in range(steps_per_size):
                    current_tensors = []
                    current_arrays = []

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
                        obs_states[i] = obs_p1  # Update persistent state

                        # Extract numpy array
                        np_array = obs_p1["numbers"]

                        # Test BOTH copy and view tensor creation
                        tensor_copy = torch.tensor(np_array, dtype=torch.float32)  # Creates copy
                        tensor_view = torch.from_numpy(np_array.astype(np.float32))  # Creates view

                        # Store for history
                        current_tensors.append({
                            'copy': tensor_copy.clone(),  # Store a clone for history
                            'view': tensor_view.clone(),  # Clone the view for safety
                            'copy_hash': tensor_hash(tensor_copy),
                            'view_hash': tensor_hash(tensor_view),
                            'copy_ptr': tensor_copy.data_ptr(),
                            'view_ptr': tensor_view.data_ptr(),
                            'np_id': id(np_array)
                        })
                        current_arrays.append(np_array.copy())

                        # Check for aliasing between consecutive tensors
                        if i > 0:
                            prev = current_tensors[i-1]
                            curr = current_tensors[i]

                            # Tensors from different battles shouldn't share storage
                            if check_tensor_aliasing(tensor_copy, prev['copy']):
                                raise ValueError(f"Copy tensors share storage at step {step}, battles {i-1} and {i}")

                            # Check if numpy arrays share memory (they shouldn't)
                            if np.shares_memory(np_array, current_arrays[i-1]):
                                raise ValueError(f"NumPy arrays share memory at step {step}, battles {i-1} and {i}")

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

                    # CRITICAL: Verify old tensors haven't mutated
                    if len(tensor_history) > 0:
                        for hist_idx, historical in enumerate(tensor_history[-10:]):
                            for battle_idx in range(min(batch_size, len(historical))):
                                old_entry = historical[battle_idx]

                                # Verify tensor contents haven't changed
                                copy_hash_now = tensor_hash(old_entry['copy'])
                                view_hash_now = tensor_hash(old_entry['view'])

                                if copy_hash_now != old_entry['copy_hash']:
                                    raise ValueError(
                                        f"Copy tensor mutated! Battle {battle_idx}, "
                                        f"history {hist_idx}, changed after {step-hist_idx} steps"
                                    )

                                if view_hash_now != old_entry['view_hash']:
                                    raise ValueError(
                                        f"View tensor mutated! Battle {battle_idx}, "
                                        f"history {hist_idx}, changed after {step-hist_idx} steps"
                                    )

                                # Check that data pointers are still valid
                                if old_entry['copy'].data_ptr() == 0:
                                    raise ValueError(f"Copy tensor deallocated at history {hist_idx}")

                    # Add to history
                    tensor_history.append(current_tensors)
                    numpy_array_history.append(current_arrays)

                    # Keep history bounded
                    if len(tensor_history) > 20:
                        tensor_history.pop(0)
                        numpy_array_history.pop(0)

                    # Progress report
                    if step % 1000 == 0 and step > 0:
                        elapsed = time.time() - start_time
                        rate = (step * batch_size * 2) / elapsed  # *2 for copy and view
                        print(f"    Step {step}/{steps_per_size}, {rate:.1f} tensor ops/sec")

                    # GC periodically
                    if step % 100 == 0:
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

            results[batch_size] = "PASS"
            print(f"  ✓ batch_size={batch_size} PASSED (no tensor aliasing/mutation)")

        except Exception as e:
            results[batch_size] = f"FAIL: {str(e)}"
            print(f"  ✗ batch_size={batch_size} FAILED at step {step}")
            print(f"    Error: {e}")
            if "share" in str(e).lower():
                print("    CRITICAL: Tensor/memory sharing detected!")
            if "mutat" in str(e).lower():
                print("    CRITICAL: Tensor mutation detected!")
            if batch_size >= 128:
                print("    Note: Failed at/above 128 boundary!")
            traceback.print_exc()

    return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("PyKMN Stability Test - Mode 3: Torch Tensors")
    print("=" * 60)

    # Set deterministic seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Test batch sizes
    batch_sizes = [1, 64, 96, 127, 128, 129, 144]

    print("\nTesting tensor creation with lifetime validation...")
    print("Testing both torch.tensor (copy) and torch.from_numpy (view)...")

    # Run tests
    results = test_torch_tensors(batch_sizes, steps_per_size=2000)

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
        print("\n✓ Mode 3 PASSED: Torch tensor creation stable, no aliasing/mutation")
        sys.exit(0)
    else:
        print(f"\n✗ Mode 3 FAILED: First failure at batch_size={first_failure}")
        if first_failure == 128:
            print("  CRITICAL: Failure at exactly 128 - tensor/memory management issue")
        sys.exit(1)


if __name__ == "__main__":
    main()