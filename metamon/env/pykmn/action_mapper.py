"""
Action mapping between metamon action space and pypkmn choices.

This module provides:
1. Legal action mask generation from pypkmn possible_choices_raw()
2. Fast action index → Choice conversion (no branching in hot loop)
3. Handling of forced switches and disabled moves

Performance: All lookups must be O(1) using precomputed tables.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from pykmn.engine.gen1 import Battle
from pykmn.engine.common import Result, Player, ResultType


# Metamon action space constants
NUM_MOVES = 4  # Actions 0-3
NUM_SWITCHES = 5  # Actions 4-8 (switch to slots 2-6)
NUM_TERA = 4  # Actions 9-12 (not used in Gen1)
TOTAL_ACTIONS = NUM_MOVES + NUM_SWITCHES + NUM_TERA  # 13 total


@dataclass
class ActionMappings:
    """
    Precomputed action index → raw choice integer mappings.

    Built once at initialization to avoid repeated lookups.

    PyKMN raw choice encoding (discovered via testing):
    - Encoding: raw = (data << 2) | type
    - type (low 2 bits): 0 = PASS, 1 = MOVE, 2 = SWITCH
    - data (high 6 bits): move index (1-4) or slot number (2-6)

    Examples:
    - Move #1: (1 << 2) | 1 = 5
    - Move #4: (4 << 2) | 1 = 17
    - Switch to slot #2: (2 << 2) | 2 = 10
    - Switch to slot #6: (6 << 2) | 2 = 26
    - Pass: 0
    """

    action_to_choice: Dict[int, int]  # Maps metamon action idx to pypkmn raw choice int

    @staticmethod
    def create() -> "ActionMappings":
        """Create precomputed action mappings using correct pypkmn encoding."""
        action_to_choice = {}

        # Actions 0-3: Move 1-4 → pypkmn encoded as (move_index << 2) | 1
        for i in range(NUM_MOVES):
            move_index = i + 1  # Moves are 1-indexed (1-4)
            action_to_choice[i] = (move_index << 2) | 1

        # Actions 4-8: Switch to slots 2-6 → pypkmn encoded as (slot << 2) | 2
        for i in range(NUM_SWITCHES):
            action_idx = NUM_MOVES + i
            slot = i + 2  # Slots are 2-6 (slot 1 is active, so switch targets 2-6)
            action_to_choice[action_idx] = (slot << 2) | 2

        # Actions 9-12: Tera moves (not valid in Gen1, fallback to move 1)
        for i in range(NUM_TERA):
            action_idx = NUM_MOVES + NUM_SWITCHES + i
            # Fallback to move 1 encoding (should be masked out in Gen1)
            action_to_choice[action_idx] = (1 << 2) | 1  # Move #1

        return ActionMappings(action_to_choice=action_to_choice)


def get_legal_mask(
    battle: Battle,
    result: Result,
    player: Player,
    action_mappings: ActionMappings,
) -> np.ndarray:
    """
    Generate boolean mask of legal actions from pypkmn battle state.

    This function is called in the hot loop but infrequently enough
    (once per player per turn) that some overhead is acceptable.

    Args:
        battle: PyKMN Battle object
        result: Current Result from last update
        player: Which player (P1 or P2)
        action_mappings: Precomputed action → Choice mappings

    Returns:
        Boolean array of shape (TOTAL_ACTIONS,) where True = legal action.
        For Gen1: actions 9-12 (tera) are always masked to False.
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)

    # If battle is over, no actions are legal
    if result.type() != ResultType.NONE:
        return mask

    # Get legal choices from pypkmn (raw format for speed)
    legal_choices_raw = battle.possible_choices_raw(player, result)

    # Parse legal choices and update mask
    # PyKMN encoding (discovered via testing):
    # - raw = (data << 2) | type
    # - type (low 2 bits): 0 = PASS, 1 = MOVE, 2 = SWITCH
    # - data (high 6 bits): move index (1-4) or slot number (2-6)
    #
    # Examples:
    # - Move #1: raw = (1 << 2) | 1 = 5
    # - Move #2: raw = (2 << 2) | 1 = 9
    # - Switch to slot #2: raw = (2 << 2) | 2 = 10
    # - Switch to slot #3: raw = (3 << 2) | 2 = 14
    # - Pass: raw = 0
    for choice_byte in legal_choices_raw:
        choice_type = choice_byte & 0x03  # Low 2 bits
        choice_data = choice_byte >> 2     # High 6 bits

        if choice_type == 0:  # PASS
            # PASS is not a normal action in metamon, skip
            continue
        elif choice_type == 1:  # MOVE
            # choice_data is move index (1-4)
            move_idx = choice_data - 1  # Convert to 0-indexed (0-3)
            if 0 <= move_idx < NUM_MOVES:
                mask[move_idx] = True
        elif choice_type == 2:  # SWITCH
            # choice_data is slot number (2-6)
            slot = choice_data
            if 2 <= slot <= 6:
                action_idx = NUM_MOVES + (slot - 2)  # Map to actions 4-8
                mask[action_idx] = True

    # Tera moves (actions 9-12) are never legal in Gen1
    # Already initialized to False, no need to explicitly mask

    return mask


def metamon_action_to_choice(
    action_idx: int,
    action_mappings: ActionMappings,
) -> int:
    """
    Convert metamon action index to pypkmn raw choice integer (FAST PATH).

    This function MUST be O(1) with no branching. Uses precomputed lookup.

    Args:
        action_idx: Action index in range [0, 12]
        action_mappings: Precomputed action → raw choice mappings

    Returns:
        PyKMN raw choice integer (1-4 for moves, 5-9 for switches).

    Note:
        Does NOT validate legality - that's done via masks.
        Illegal actions will be rejected by pypkmn engine, not here.
    """
    return action_mappings.action_to_choice[action_idx]


def get_legal_mask_batch(
    battles: List[Battle],
    results: List[Result],
    players: List[Player],
    action_mappings: ActionMappings,
) -> np.ndarray:
    """
    Vectorized version of get_legal_mask for multiple battles.

    Args:
        battles: List of N Battle objects
        results: List of N Result objects
        players: List of N Player objects (typically all P1 or all P2)
        action_mappings: Precomputed action → Choice mappings

    Returns:
        Boolean array of shape (N, TOTAL_ACTIONS).
    """
    n = len(battles)
    masks = np.zeros((n, TOTAL_ACTIONS), dtype=bool)

    for i in range(n):
        masks[i] = get_legal_mask(battles[i], results[i], players[i], action_mappings)

    return masks


def is_forced_switch(result: Result) -> bool:
    """
    Check if the current result requires a forced switch.

    In pypkmn, forced switches occur when:
    - Active Pokemon fainted
    - Certain moves force switches (e.g., Whirlwind, Roar)

    Args:
        result: Current Result from battle.update()

    Returns:
        True if this is a forced switch state.
    """
    # TODO: Implement proper detection based on result type
    # This requires understanding pypkmn's result types for forced switches
    # For now, return False as placeholder
    return False


def filter_illegal_actions(
    actions: np.ndarray,
    legal_masks: np.ndarray,
    fallback: str = "first_legal",
) -> np.ndarray:
    """
    Filter out illegal actions from a batch of action selections.

    This is a safety mechanism for when policies select invalid actions
    (shouldn't happen if masks are used correctly, but useful for debugging).

    Args:
        actions: Array of shape (N,) with selected action indices
        legal_masks: Array of shape (N, TOTAL_ACTIONS) with legal action masks
        fallback: Strategy for illegal actions:
            - "first_legal": Replace with first legal action
            - "random_legal": Replace with random legal action

    Returns:
        Array of shape (N,) with legal actions.
    """
    filtered_actions = actions.copy()

    for i in range(len(actions)):
        action = actions[i]
        if not legal_masks[i, action]:
            # Action is illegal, use fallback
            legal_indices = np.where(legal_masks[i])[0]
            if len(legal_indices) == 0:
                # No legal actions (shouldn't happen), default to action 0
                filtered_actions[i] = 0
            elif fallback == "first_legal":
                filtered_actions[i] = legal_indices[0]
            elif fallback == "random_legal":
                filtered_actions[i] = np.random.choice(legal_indices)
            else:
                raise ValueError(f"Unknown fallback strategy: {fallback}")

    return filtered_actions
