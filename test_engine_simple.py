"""
Simple test of engine bindings - run one battle with random choices.
"""

import random
from metamon.engine.bindings import (
    Gen1Battle,
    EXAMPLE_BATTLE_BYTES,
    PKMN_CHOICE_PASS,
    PKMN_RESULT_NONE,
    PKMN_PLAYER_P1,
    PKMN_PLAYER_P2,
)

def main():
    print("Initializing Gen 1 battle from example bytes...")
    battle = Gen1Battle(EXAMPLE_BATTLE_BYTES)

    print("Starting battle with random choices...")

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

        print(f"Turn {turn}: P1 chooses {choice_p1}, P2 chooses {choice_p2}")

        # Update battle
        result_type, request_p1, request_p2 = battle.update(choice_p1, choice_p2)

        if result_type != PKMN_RESULT_NONE:
            print(f"Battle ended: result={result_type}")
            break

        turn += 1

    if battle.done:
        result_map = {1: "P1 WIN", 2: "P1 LOSE", 3: "TIE", 4: "ERROR"}
        print(f"\nBattle finished after {turn} turns!")
        print(f"Result: {result_map.get(battle.result, 'UNKNOWN')}")
    else:
        print(f"\nBattle stopped after {turn} turns (safety limit or no choices)")

    return battle.result if battle.done else None


if __name__ == "__main__":
    random.seed(42)
    result = main()
    exit(0 if result is not None else 1)
