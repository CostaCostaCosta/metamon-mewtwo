#!/usr/bin/env python3
"""
Mode 0: Single Battle Baseline
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

from pykmn.engine.gen1 import Battle, Player, Choice, MoveChoice, PassChoice, SwitchChoice
from metamon.data.team_repo import load_teams_for_format


def get_legal_action(battle: Battle, player: Player) -> Choice:
    """Get a random legal action for the player."""
    # Get legal moves
    legal_moves = []
    for slot in range(4):
        if battle.possible(player, MoveChoice(slot)):
            legal_moves.append(MoveChoice(slot))

    # Get legal switches
    legal_switches = []
    for slot in range(1, 6):  # slots 1-5 are switches
        if battle.possible(player, SwitchChoice(slot)):
            legal_switches.append(SwitchChoice(slot))

    # Combine all legal actions
    all_legal = legal_moves + legal_switches

    if not all_legal:
        # No legal actions, return pass
        return PassChoice()

    return random.choice(all_legal)


def validate_invariants(battle: Battle, step: int, prev_turn: int = None) -> int:
    """Validate battle invariants."""
    # Get battle state
    p1_side = battle.sides[0]
    p2_side = battle.sides[1]

    # Check HP bounds
    for side_idx, side in enumerate([p1_side, p2_side]):
        for poke_idx, pokemon in enumerate(side.pokemon):
            if pokemon is not None:
                hp = pokemon.hp
                max_hp = pokemon.max_hp
                assert 0 <= hp <= max_hp, f"HP invariant violated: {hp}/{max_hp} at step {step}"

                # Check PP bounds
                for move_idx, move in enumerate(pokemon.moves):
                    if move is not None:
                        pp = move.pp
                        max_pp = move.maxpp
                        assert 0 <= pp <= max_pp, f"PP invariant violated: {pp}/{max_pp} at step {step}"

    # Check turn counter monotonicity
    current_turn = battle.turn
    if prev_turn is not None:
        assert current_turn >= prev_turn, f"Turn went backward: {prev_turn} -> {current_turn}"
        assert current_turn - prev_turn <= 2, f"Turn jumped too much: {prev_turn} -> {current_turn}"

    # Check that non-terminal battles have legal moves
    result = battle.result()
    if result.type() == 0:  # Not terminal
        p1_has_legal = False
        p2_has_legal = False

        for slot in range(6):  # Check moves and switches
            if slot < 4:
                if battle.possible(Player.P1, MoveChoice(slot)):
                    p1_has_legal = True
                if battle.possible(Player.P2, MoveChoice(slot)):
                    p2_has_legal = True
            else:
                if battle.possible(Player.P1, SwitchChoice(slot-3)):
                    p1_has_legal = True
                if battle.possible(Player.P2, SwitchChoice(slot-3)):
                    p2_has_legal = True

        # At least one player should have legal moves
        assert p1_has_legal or p2_has_legal, f"No legal moves but battle not terminal at step {step}"

    return current_turn


def run_single_battle_test(max_steps: int = 100000) -> None:
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
        p1_seed=42,
        p2_seed=43
    )

    # Run battle steps
    step = 0
    episodes = 0
    prev_turn = None

    try:
        while step < max_steps:
            # Get legal actions
            c1 = get_legal_action(battle, Player.P1)
            c2 = get_legal_action(battle, Player.P2)

            # Update battle
            result, _ = battle.update(c1, c2)

            # Validate invariants
            prev_turn = validate_invariants(battle, step, prev_turn)

            # Check if battle ended
            if result.type() != 0:  # Terminal
                episodes += 1
                if episodes % 100 == 0:
                    print(f"  Completed {episodes} episodes, {step} steps")

                # Reset for new battle
                battle = Battle(
                    p1_team=team1.to_pykmn(),
                    p2_team=team2.to_pykmn(),
                    p1_seed=42 + episodes,
                    p2_seed=43 + episodes
                )
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