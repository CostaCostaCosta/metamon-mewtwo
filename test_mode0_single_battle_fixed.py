#!/usr/bin/env python3
"""
Mode 0: Single Battle Baseline (FIXED)
Purpose: Establish if PyKMN is stable without any batching
Expected: Should run 100K steps without crashes or invariant violations
"""

import random
import sys
import traceback
from typing import List, Tuple
import numpy as np

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle
from pykmn.engine.common import Player, ResultType
from metamon.data.team_repo import load_teams_for_format


def get_random_legal_choice(battle: Battle, player: Player) -> int:
    """Get a random legal choice for the player."""
    choices = battle.possible_choices_raw(player)
    if not choices:
        return 0  # PASS
    return random.choice(choices)


def validate_invariants(battle: Battle, step: int, prev_turn: int = None) -> int:
    """Validate battle invariants."""
    # Get current turn
    current_turn = battle.turn()

    # Check turn counter monotonicity
    if prev_turn is not None:
        assert current_turn >= prev_turn, f"Turn went backward: {prev_turn} -> {current_turn}"
        assert current_turn - prev_turn <= 2, f"Turn jumped too much: {prev_turn} -> {current_turn}"

    # Check HP bounds for each pokemon
    for player in [Player.P1, Player.P2]:
        hp_values = battle.current_hp(player)
        stats = battle.stats(player)

        for i in range(6):  # 6 pokemon
            current_hp = hp_values[i]
            max_hp = stats[i][0] if i < len(stats) else 0  # HP is first stat

            if max_hp > 0:  # Pokemon exists
                assert 0 <= current_hp <= max_hp, \
                    f"HP invariant violated: {current_hp}/{max_hp} at step {step}"

    return current_turn


def run_single_battle_test(max_steps: int = 100000) -> bool:
    """Run a single battle for many steps to establish baseline stability."""
    print(f"Mode 0: Testing single battle for {max_steps} steps...")

    # Set deterministic seeds
    random.seed(42)
    np.random.seed(42)

    # Load teams
    teams = load_teams_for_format("gen1ou")
    team_idx = random.randint(0, len(teams) - 1)
    team1 = teams[team_idx]
    team2 = teams[team_idx]  # Mirror match for consistency

    print(f"Using team {team_idx} (mirror match)")

    # Create battle
    battle = Battle(
        p1_team=team1.to_pykmn(),
        p2_team=team2.to_pykmn(),
    )

    # Initial update (team preview)
    result, _ = battle.update_raw(0, 0)  # PASS for both

    # Run battle steps
    step = 0
    episodes = 0
    prev_turn = None

    try:
        while step < max_steps:
            # Get legal actions
            c1 = get_random_legal_choice(battle, Player.P1)
            c2 = get_random_legal_choice(battle, Player.P2)

            # Update battle
            result, _ = battle.update_raw(c1, c2)

            # Validate invariants
            prev_turn = validate_invariants(battle, step, prev_turn)

            # Check if battle ended
            if result.type() != ResultType.NONE:  # Terminal
                episodes += 1
                if episodes % 100 == 0:
                    print(f"  Completed {episodes} episodes, {step} steps")

                # Reset for new battle
                battle = Battle(
                    p1_team=team1.to_pykmn(),
                    p2_team=team2.to_pykmn(),
                )
                # Initial update
                result, _ = battle.update_raw(0, 0)
                prev_turn = None

            step += 1

            # Progress report
            if step % 10000 == 0:
                print(f"  Progress: {step}/{max_steps} steps, {episodes} episodes completed")

        print(f"\n✓ Mode 0 PASSED: {step} steps, {episodes} episodes, no crashes or violations")
        return True

    except Exception as e:
        print(f"\n✗ Mode 0 FAILED at step {step}, episode {episodes}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("PyKMN Stability Test - Mode 0: Single Battle Baseline")
    print("=" * 60)
    print()

    success = run_single_battle_test(max_steps=100000)

    if success:
        print("\nMode 0 baseline established - PyKMN is stable for single battles")
        sys.exit(0)
    else:
        print("\nMode 0 failed - PyKMN has issues even with single battles")
        sys.exit(1)


if __name__ == "__main__":
    main()