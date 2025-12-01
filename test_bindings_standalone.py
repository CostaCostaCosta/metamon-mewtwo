"""
Standalone test of engine bindings - doesn't import metamon package.
"""

import sys
import random
from pathlib import Path

# Add metamon to path but don't import the package
sys.path.insert(0, str(Path(__file__).parent))

# Import bindings module directly
from metamon.engine import bindings

def main():
    print("Testing engine bindings...")
    print(f"Library path: {bindings.ENGINE_LIB_PATH}")

    # Constants
    PKMN_CHOICE_PASS = bindings.PKMN_CHOICE_PASS
    PKMN_RESULT_NONE = bindings.PKMN_RESULT_NONE
    PKMN_PLAYER_P1 = bindings.PKMN_PLAYER_P1
    PKMN_PLAYER_P2 = bindings.PKMN_PLAYER_P2

    print("\nInitializing Gen 1 battle from example bytes...")
    battle = bindings.Gen1Battle(bindings.EXAMPLE_BATTLE_BYTES)
    print(f"Battle state: {len(battle.get_battle_bytes())} bytes")

    print("\nStarting battle with random choices...")

    # First update: both players pass to send out their first Pokemon
    result_type, request_p1, request_p2 = battle.update(PKMN_CHOICE_PASS, PKMN_CHOICE_PASS)
    print(f"Initial result: type={result_type}, P1 request={request_p1}, P2 request={request_p2}")

    turn = 1
    while not battle.done and turn < 1000:  # Safety limit
        # Get legal choices for both players
        choices_p1 = battle.get_legal_choices(PKMN_PLAYER_P1, request_p1)
        choices_p2 = battle.get_legal_choices(PKMN_PLAYER_P2, request_p2)

        if not choices_p1 or not choices_p2:
            print(f"No legal choices available (P1: {len(choices_p1)}, P2: {len(choices_p2)})")
            break

        # Pick random choices
        choice_p1 = random.choice(choices_p1)
        choice_p2 = random.choice(choices_p2)

        if turn <= 5 or turn % 10 == 0:  # Print first few turns and every 10th
            print(f"Turn {turn}: P1={choice_p1} P2={choice_p2}")

        # Update battle
        result_type, request_p1, request_p2 = battle.update(choice_p1, choice_p2)

        if result_type != PKMN_RESULT_NONE:
            print(f"Battle ended: result={result_type}")
            break

        turn += 1

    if battle.done:
        result_map = {1: "P1 WIN", 2: "P1 LOSE", 3: "TIE", 4: "ERROR"}
        print(f"\n✓ Battle finished after {turn} turns!")
        print(f"✓ Result: {result_map.get(battle.result, 'UNKNOWN')}")
        return 0
    else:
        print(f"\n✗ Battle stopped after {turn} turns (safety limit or no choices)")
        return 1


if __name__ == "__main__":
    random.seed(42)
    sys.exit(main())
