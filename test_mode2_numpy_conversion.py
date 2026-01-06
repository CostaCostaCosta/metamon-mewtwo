#!/usr/bin/env python3
"""
Mode 2: PyKMN → NumPy Conversion with Aliasing Detection
Purpose: Test if feature extraction holds views into mutable buffers
Critical: Verify that old features don't mutate after new steps
"""

import copy
import gc
import hashlib
import json
import random
import sys
import traceback
import time
from typing import Any, Dict, List
import numpy as np

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Player, Choice, MoveChoice, PassChoice, SwitchChoice
from metamon.data.team_repo import load_teams_for_format
from metamon.env.pykmn.feature_extractor import (
    pykmn_to_features_raw,
    features_to_universal_state,
    create_gen1_mappings
)


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


def compute_feature_hash(features: Dict[str, Any]) -> str:
    """Compute a deterministic hash of feature dictionary."""
    # Convert to stable string representation
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tobytes()
        elif isinstance(obj, dict):
            return json.dumps({k: serialize(v) for k, v in sorted(obj.items())}, sort_keys=True)
        elif isinstance(obj, (list, tuple)):
            return json.dumps([serialize(x) for x in obj])
        else:
            return str(obj)

    serialized = serialize(features)
    if isinstance(serialized, str):
        serialized = serialized.encode('utf-8')
    return hashlib.md5(serialized).hexdigest()


def check_memory_aliasing(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """Check if two numpy arrays share memory."""
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        return False
    return np.shares_memory(arr1, arr2)


def test_feature_extraction(batch_sizes: List[int], steps_per_size: int = 5000) -> dict:
    """Test feature extraction with aliasing detection."""
    print("\nTesting feature extraction with aliasing detection")
    print("-" * 50)

    results = {}
    teams = load_teams_for_format("gen1ou")
    mappings = create_gen1_mappings()

    for batch_size in batch_sizes:
        print(f"\n  Testing batch_size={batch_size}...")

        # Create battles
        battles = []
        for i in range(batch_size):
            team_idx = i % len(teams)
            battle = Battle(
                p1_team=teams[team_idx].to_pykmn(),
                p2_team=teams[team_idx].to_pykmn(),
                p1_seed=42 + i,
                p2_seed=142 + i
            )
            battles.append(battle)

        # Initialize result tracking
        results_list = [None] * batch_size

        # History for aliasing detection
        feature_history = []
        state_history = []

        try:
            start_time = time.time()
            for step in range(steps_per_size):
                # Store current features
                current_features = []
                current_states = []

                for i, battle in enumerate(battles):
                    # Get actions and update
                    c1 = get_legal_action(battle, Player.P1)
                    c2 = get_legal_action(battle, Player.P2)
                    result, _ = battle.update(c1, c2)
                    results_list[i] = result

                    # Extract features
                    features_p1 = pykmn_to_features_raw(battle, result, Player.P1, mappings)
                    features_p2 = pykmn_to_features_raw(battle, result, Player.P2, mappings)

                    # Convert to universal state
                    state_p1 = features_to_universal_state(features_p1, mappings)
                    state_p2 = features_to_universal_state(features_p2, mappings)

                    # Store for history
                    current_features.append({
                        'p1': copy.deepcopy(features_p1),
                        'p2': copy.deepcopy(features_p2),
                        'p1_hash': compute_feature_hash(features_p1),
                        'p2_hash': compute_feature_hash(features_p2)
                    })
                    current_states.append({
                        'p1': copy.deepcopy(state_p1),
                        'p2': copy.deepcopy(state_p2)
                    })

                    # Check for memory aliasing between battles
                    if i > 0:
                        prev_features = current_features[i-1]['p1']
                        curr_features = features_p1

                        # Check if any arrays share memory
                        for key in features_p1:
                            if isinstance(features_p1[key], np.ndarray) and key in prev_features:
                                if isinstance(prev_features[key], np.ndarray):
                                    if check_memory_aliasing(features_p1[key], prev_features[key]):
                                        raise ValueError(f"Memory aliasing detected between battles at step {step}, key {key}")

                    # Reset if terminal
                    if result.type() != 0:
                        team_idx = random.randint(0, len(teams) - 1)
                        battles[i] = Battle(
                            p1_team=teams[team_idx].to_pykmn(),
                            p2_team=teams[team_idx].to_pykmn(),
                            p1_seed=random.randint(0, 1000000),
                            p2_seed=random.randint(0, 1000000)
                        )

                # CRITICAL: Verify old features haven't mutated
                if len(feature_history) > 0:
                    # Check last 10 historical entries
                    for hist_idx, historical in enumerate(feature_history[-10:]):
                        for battle_idx in range(min(batch_size, len(historical))):
                            old_entry = historical[battle_idx]

                            # Recompute hash of stored features
                            new_p1_hash = compute_feature_hash(old_entry['p1'])
                            new_p2_hash = compute_feature_hash(old_entry['p2'])

                            if new_p1_hash != old_entry['p1_hash']:
                                raise ValueError(
                                    f"Feature mutation detected! Battle {battle_idx}, "
                                    f"history {hist_idx}, P1 features changed after {step-hist_idx} steps"
                                )
                            if new_p2_hash != old_entry['p2_hash']:
                                raise ValueError(
                                    f"Feature mutation detected! Battle {battle_idx}, "
                                    f"history {hist_idx}, P2 features changed after {step-hist_idx} steps"
                                )

                # Add to history
                feature_history.append(current_features)
                state_history.append(current_states)

                # Keep history bounded
                if len(feature_history) > 20:
                    feature_history.pop(0)
                    state_history.pop(0)

                # Progress report
                if step % 1000 == 0 and step > 0:
                    elapsed = time.time() - start_time
                    rate = (step * batch_size) / elapsed
                    print(f"    Step {step}/{steps_per_size}, {rate:.1f} extractions/sec")

                # GC periodically to ensure cleanup
                if step % 100 == 0:
                    gc.collect()

            results[batch_size] = "PASS"
            print(f"  ✓ batch_size={batch_size} PASSED (no aliasing/mutation detected)")

        except Exception as e:
            results[batch_size] = f"FAIL: {str(e)}"
            print(f"  ✗ batch_size={batch_size} FAILED at step {step}")
            print(f"    Error: {e}")
            if "aliasing" in str(e).lower():
                print("    CRITICAL: Memory aliasing detected!")
            if "mutation" in str(e).lower():
                print("    CRITICAL: Feature mutation detected!")
            if batch_size >= 128:
                print("    Note: Failed at/above 128 boundary!")

    return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("PyKMN Stability Test - Mode 2: NumPy Conversion")
    print("=" * 60)

    # Set deterministic seeds
    random.seed(42)
    np.random.seed(42)

    # Test batch sizes
    batch_sizes = [1, 64, 96, 127, 128, 129, 144]

    print("\nTesting feature extraction with mutation detection...")

    # Run tests
    results = test_feature_extraction(batch_sizes, steps_per_size=2000)

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
        print("\n✓ Mode 2 PASSED: Feature extraction stable, no aliasing/mutation detected")
        sys.exit(0)
    else:
        print(f"\n✗ Mode 2 FAILED: First failure at batch_size={first_failure}")
        if first_failure == 128:
            print("  CRITICAL: Failure at exactly 128 - buffer limit in feature extraction")
        sys.exit(1)


if __name__ == "__main__":
    main()