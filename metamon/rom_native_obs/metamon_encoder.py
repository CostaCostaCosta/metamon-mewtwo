"""
Metamon-side encoder: converts UniversalState -> RomBattleState.

This encoder produces the canonical compact structured representation
without using Metamon's text-token observation pipeline. It works
directly from the UniversalState dataclass and the string-based names
are converted to integer IDs via the mappings module.
"""
from __future__ import annotations

import copy
from typing import Optional, List, Set

from metamon.interface import UniversalState, UniversalPokemon, UniversalMove, UniversalAction
from metamon.backend.replay_parser.str_parsing import pokemon_name, move_name, clean_name

from .schema import (
    RomBattleState, GlobalFeatures, PokemonFeatures,
    NUM_POKEMON_SLOTS, NUM_MOVES_PER_POKEMON, NUM_ACTIONS,
    SLOT_PLAYER_ACTIVE, SLOT_SWITCH_0, SLOT_OPPONENT_ACTIVE, SLOT_REVEALED_OPP_0,
    SPECIES_UNKNOWN, MOVE_UNKNOWN, MOVE_NONE,
    TYPE_NONE, STATUS_NONE, STATUS_FAINT, STATUS_UNKNOWN,
    WEATHER_NONE, SIDE_COND_NONE, FIELD_NONE, EFFECT_NONE,
    CATEGORY_NONE, CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
)
from .mappings import (
    species_name_to_id, move_name_to_id,
    TYPE_NAME_TO_ID, STATUS_NAME_TO_ID, WEATHER_NAME_TO_ID,
    SIDE_COND_NAME_TO_ID, FIELD_NAME_TO_ID, EFFECT_NAME_TO_ID,
    CATEGORY_NAME_TO_ID,
)


def _type_str_to_id(type_str: str) -> int:
    """Convert a type string (e.g. "ghost poison" or single word) to type ID.

    UniversalPokemon.types is a space-separated string of 1-2 type names.
    This function returns the first type. Use _types_str_to_ids for both.
    """
    if not type_str or type_str == "notype":
        return TYPE_NONE
    parts = type_str.strip().split()
    if not parts:
        return TYPE_NONE
    return TYPE_NAME_TO_ID.get(clean_name(parts[0]), TYPE_NONE)


def _types_str_to_ids(type_str: str) -> tuple[int, int]:
    """Convert a type string to (type_1_id, type_2_id)."""
    if not type_str or type_str == "notype":
        return (TYPE_NONE, TYPE_NONE)
    parts = type_str.strip().split()
    t1 = TYPE_NAME_TO_ID.get(clean_name(parts[0]), TYPE_NONE) if len(parts) > 0 else TYPE_NONE
    t2 = TYPE_NAME_TO_ID.get(clean_name(parts[1]), TYPE_NONE) if len(parts) > 1 else TYPE_NONE
    return (t1, t2)


def _status_str_to_id(status_str: str) -> int:
    """Convert a status string to status ID."""
    if not status_str:
        return STATUS_NONE
    return STATUS_NAME_TO_ID.get(clean_name(status_str), STATUS_NONE)


def _weather_str_to_id(weather_str: str) -> int:
    """Convert a weather string to weather ID."""
    if not weather_str:
        return WEATHER_NONE
    return WEATHER_NAME_TO_ID.get(clean_name(weather_str), WEATHER_NONE)


def _side_cond_str_to_id(cond_str: str) -> int:
    """Convert a side condition string to side condition ID."""
    if not cond_str:
        return SIDE_COND_NONE
    return SIDE_COND_NAME_TO_ID.get(clean_name(cond_str), SIDE_COND_NONE)


def _field_str_to_id(field_str: str) -> int:
    """Convert a field effect string to field effect ID."""
    if not field_str:
        return FIELD_NONE
    return FIELD_NAME_TO_ID.get(clean_name(field_str), FIELD_NONE)


def _effect_str_to_id(effect_str: str) -> int:
    """Convert an effect string to effect ID."""
    if not effect_str:
        return EFFECT_NONE
    return EFFECT_NAME_TO_ID.get(clean_name(effect_str), EFFECT_NONE)


def _category_str_to_id(cat_str: str) -> int:
    """Convert a move category string to category ID."""
    if not cat_str:
        return CATEGORY_NONE
    return CATEGORY_NAME_TO_ID.get(clean_name(cat_str), CATEGORY_NONE)


def _encode_move(move: UniversalMove, is_revealed: bool = True) -> dict:
    """Encode a UniversalMove into move feature components.

    Returns a dict with: move_id, category, type, bp, acc, pri, pp
    """
    if move is None:
        return {
            "move_id": MOVE_UNKNOWN,
            "category": CATEGORY_NONE,
            "type": TYPE_NONE,
            "bp": -2.0,
            "acc": -2.0,
            "pri": -2.0,
            "pp": -2.0,
        }

    move_id = move_name_to_id(move.name) if is_revealed else MOVE_UNKNOWN
    category = _category_str_to_id(move.category) if is_revealed else CATEGORY_NONE
    move_type = _type_str_to_id(move.move_type) if is_revealed else TYPE_NONE

    if is_revealed:
        bp = move.base_power / 200.0
        acc = float(move.accuracy) if move.accuracy is not None else -2.0
        pri = move.priority / 5.0
        if move.max_pp > 0:
            pp = move.current_pp / move.max_pp
        else:
            pp = -2.0
    else:
        bp = -2.0
        acc = -2.0
        pri = -2.0
        pp = -2.0

    return {
        "move_id": move_id,
        "category": category,
        "type": move_type,
        "bp": bp,
        "acc": acc,
        "pri": pri,
        "pp": pp,
    }


def _encode_pokemon(pokemon: UniversalPokemon, is_active: bool, 
                     is_opponent: bool, moves_revealed: bool,
                     hp_known: bool = True) -> PokemonFeatures:
    """Encode a UniversalPokemon into PokemonFeatures.

    Information visibility rules:
    - Player's own Pokémon: full information (moves, HP, stats, boosts)
    - Opponent active: HP known (from HP bar), species/types known,
      status/effect known, boosts known, but moves only if revealed
    - Opponent bench (revealed): species known, HP may be unknown,
      moves only if revealed, no stat details

    Args:
        pokemon: The UniversalPokemon to encode
        is_active: Whether this is the active Pokémon
        is_opponent: Whether this is an opponent Pokémon
        moves_revealed: Whether the moveset has been revealed
        hp_known: Whether the HP fraction is observable
    """
    if pokemon is None:
        return PokemonFeatures()

    pf = PokemonFeatures()
    pf.valid = True

    # Species
    pf.species = species_name_to_id(pokemon.name)

    # Types
    pf.type_1, pf.type_2 = _types_str_to_ids(pokemon.types)

    # Status
    pf.status = _status_str_to_id(pokemon.status)

    # Effect (volatile)
    pf.effect = _effect_str_to_id(pokemon.effect)

    # HP fraction
    if hp_known:
        pf.hp_fraction = float(pokemon.hp_pct)
        pf.hp_known = True
    else:
        pf.hp_fraction = -1.0
        pf.hp_known = False

    # Fainted
    pf.fainted = (pf.status == STATUS_FAINT) or (hp_known and pokemon.hp_pct <= 0.0)

    # Level
    pf.level_norm = float(pokemon.lvl) / 100.0

    # Base stats (always available from pokedex)
    pf.base_atk_norm = float(pokemon.base_atk) / 255.0
    pf.base_spa_norm = float(pokemon.base_spa) / 255.0
    pf.base_def_norm = float(pokemon.base_def) / 255.0
    pf.base_spd_norm = float(pokemon.base_spd) / 255.0
    pf.base_spe_norm = float(pokemon.base_spe) / 255.0
    pf.base_hp_norm = float(pokemon.base_hp) / 255.0

    # Boosts (only for active Pokémon; opponent active boosts are visible)
    if is_active:
        pf.boosts = [
            float(pokemon.atk_boost) / 6.0,
            float(pokemon.spa_boost) / 6.0,
            float(pokemon.def_boost) / 6.0,
            float(pokemon.spd_boost) / 6.0,
            float(pokemon.spe_boost) / 6.0,
            float(pokemon.accuracy_boost) / 6.0,
            float(pokemon.evasion_boost) / 6.0,
        ]
    else:
        pf.boosts = [0.0] * 7

    # Moves
    pf.moves_revealed = moves_revealed
    from metamon.interface import consistent_move_order
    sorted_moves = consistent_move_order(list(pokemon.moves))[:NUM_MOVES_PER_POKEMON]

    for i in range(NUM_MOVES_PER_POKEMON):
        if i < len(sorted_moves):
            move = sorted_moves[i]
            move_data = _encode_move(move, is_revealed=moves_revealed)
            pf.move_ids[i] = move_data["move_id"]
            pf.move_categories[i] = move_data["category"]
            pf.move_types[i] = move_data["type"]
            pf.move_bp[i] = move_data["bp"]
            pf.move_acc[i] = move_data["acc"]
            pf.move_pri[i] = move_data["pri"]
            pf.move_pp[i] = move_data["pp"]
        else:
            pf.move_ids[i] = MOVE_UNKNOWN
            pf.move_categories[i] = CATEGORY_NONE
            pf.move_types[i] = TYPE_NONE
            pf.move_bp[i] = -2.0
            pf.move_acc[i] = -2.0
            pf.move_pri[i] = -2.0
            pf.move_pp[i] = -2.0

    return pf


class RomObservationEncoder:
    """Encoder that converts UniversalState into RomBattleState.

    This encoder maintains state across timesteps to track:
    - Revealed opponent species
    - Sleep/freeze clause flags
    - Opponent move reveals

    It does NOT use any text-token observation space.
    """

    def __init__(self, gen: int = 1):
        self.gen = gen
        self.reset()

    def reset(self):
        """Reset internal state for a new battle."""
        self.revealed_opponents: dict[str, UniversalPokemon] = {}
        self.revealed_opponent_order: list[str] = []
        self.any_opponent_asleep = False
        self.any_opponent_frozen = False
        self._turn_count = 0

    def encode(self, state: UniversalState) -> RomBattleState:
        """Convert a UniversalState into a RomBattleState.

        Args:
            state: The UniversalState from Metamon's replay/env pipeline

        Returns:
            RomBattleState with compact structured features
        """
        self._turn_count += 1

        # Track revealed opponents
        opponent = state.opponent_active_pokemon
        opp_key = pokemon_name(opponent.base_species or opponent.name)
        if opp_key and opp_key not in self.revealed_opponents:
            self.revealed_opponent_order.append(opp_key)
        self.revealed_opponents[opp_key] = copy.deepcopy(opponent)

        # Track sleep/freeze
        if _status_str_to_id(opponent.status) == 1:  # SLP
            self.any_opponent_asleep = True
        if _status_str_to_id(opponent.status) == 4:  # FRZ
            self.any_opponent_frozen = True

        # Build global features
        global_feats = GlobalFeatures(
            weather=_weather_str_to_id(state.weather),
            field_effect=_field_str_to_id(state.battle_field),
            player_side_cond=_side_cond_str_to_id(state.player_conditions),
            opponent_side_cond=_side_cond_str_to_id(state.opponent_conditions),
            player_prev_move=move_name_to_id(state.player_prev_move.name) if state.player_prev_move else MOVE_NONE,
            opponent_prev_move=move_name_to_id(state.opponent_prev_move.name) if state.opponent_prev_move else MOVE_NONE,
            turn_norm=min(self._turn_count / 200.0, 1.0),
            opponents_remaining=float(state.opponents_remaining) / 6.0,
            forced_switch=1.0 if state.forced_switch else 0.0,
        )

        # Build Pokémon slots
        pokemon_list: list[PokemonFeatures] = [PokemonFeatures() for _ in range(NUM_POKEMON_SLOTS)]

        # Slot 0: Player active Pokémon (full info)
        pokemon_list[SLOT_PLAYER_ACTIVE] = _encode_pokemon(
            state.player_active_pokemon,
            is_active=True, is_opponent=False,
            moves_revealed=True, hp_known=True,
        )

        # Slots 1-5: Available switches (player's bench - full info)
        from metamon.interface import consistent_pokemon_order
        switches = consistent_pokemon_order(state.available_switches)
        for i in range(5):
            if i < len(switches):
                pokemon_list[SLOT_SWITCH_0 + i] = _encode_pokemon(
                    switches[i],
                    is_active=False, is_opponent=False,
                    moves_revealed=True, hp_known=True,
                )

        # Slot 6: Opponent active Pokémon
        # HP is known from the HP bar, species/types/status/boosts are visible,
        # but moves are only known if revealed (we track revealed opponents)
        opp_moves_revealed = len(opponent.moves) > 0
        pokemon_list[SLOT_OPPONENT_ACTIVE] = _encode_pokemon(
            opponent,
            is_active=True, is_opponent=True,
            moves_revealed=opp_moves_revealed, hp_known=True,
        )

        # Slots 7-12: Revealed opponent Pokémon (from memory)
        revealed = [
            self.revealed_opponents[name]
            for name in self.revealed_opponent_order[:6]
            if name in self.revealed_opponents
        ]
        # The active opponent is already in slot 6, skip it
        revealed = [p for p in revealed if pokemon_name(p.base_species or p.name) != opp_key]

        for i in range(6):
            if i < len(revealed):
                rev_pokemon = revealed[i]
                rev_key = pokemon_name(rev_pokemon.base_species or rev_pokemon.name)
                # For revealed opponents on bench: species known, HP may not be current
                # Moves are revealed only if they were seen
                rev_moves_revealed = len(rev_pokemon.moves) > 0
                pokemon_list[SLOT_REVEALED_OPP_0 + i] = _encode_pokemon(
                    rev_pokemon,
                    is_active=False, is_opponent=True,
                    moves_revealed=rev_moves_revealed, hp_known=False,
                )

        # Build legal action mask
        legal_mask = [False] * NUM_ACTIONS
        if not state.forced_switch:
            num_moves = len(state.player_active_pokemon.moves)
            for i in range(min(num_moves, 4)):
                legal_mask[i] = True
        num_switches = len(switches)
        for i in range(min(num_switches, 5)):
            legal_mask[4 + i] = True

        return RomBattleState(
            global_features=global_feats,
            pokemon=pokemon_list,
            legal_action_mask=legal_mask,
        )

    def encode_from_dict(self, state_dict: dict) -> RomBattleState:
        """Encode a state from a dictionary (e.g., from a replay file)."""
        state = UniversalState.from_dict(state_dict)
        return self.encode(state)
