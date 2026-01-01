"""
Two-tier state representation for pypkmn integration.

This module provides:
1. Fast numeric feature extraction (hot loop, no allocations)
2. Slow UniversalState conversion (save time only)
3. Precomputed mappings for efficient lookups

Performance is critical - all string lookups must be precomputed.
Use no-trace builds and _raw methods only in the hot loop.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from pykmn.engine.gen1 import Battle, Pokemon
from pykmn.engine.common import Result, Player, ResultType
from pykmn.data.gen1 import (
    SPECIES,
    MOVES,
    SPECIES_IDS,
    MOVE_IDS,
    SPECIES_ID_LOOKUP,
    MOVE_ID_LOOKUP,
)

from metamon.interface import UniversalState, UniversalPokemon, UniversalMove


@dataclass
class Mappings:
    """
    Precomputed mappings for fast feature extraction.

    All string lookups must be done via these mappings to avoid
    performance bottlenecks in the hot loop.
    """

    # Species mappings
    species_id_to_name: Dict[int, str]
    species_name_to_id: Dict[str, int]

    # Move mappings
    move_id_to_name: Dict[int, str]
    move_name_to_id: Dict[str, int]

    # Status/effect enums (if needed)
    # TODO: Add more mappings as needed


def precompute_mappings() -> Mappings:
    """
    Precompute all string↔ID mappings once at initialization.

    This function should be called once when creating environments
    to avoid repeated string lookups during simulation.

    Returns:
        Mappings object with all precomputed lookup tables.
    """
    # Build species mappings
    species_id_to_name = {}
    species_name_to_id = {}

    for species_name, species_data in SPECIES.items():
        species_id = SPECIES_IDS[species_name]
        species_id_to_name[species_id] = species_name
        species_name_to_id[species_name] = species_id

    # Build move mappings
    move_id_to_name = {}
    move_name_to_id = {}

    for move_name, move_data in MOVES.items():
        move_id = MOVE_IDS[move_name]
        move_id_to_name[move_id] = move_name
        move_name_to_id[move_name] = move_id

    return Mappings(
        species_id_to_name=species_id_to_name,
        species_name_to_id=species_name_to_id,
        move_id_to_name=move_id_to_name,
        move_name_to_id=move_name_to_id,
    )


def pykmn_to_features_raw(
    battle: Battle,
    result: Result,
    player: Player,
    mappings: Mappings,
) -> Dict[str, np.ndarray]:
    """
    Extract numeric features from pypkmn Battle (FAST PATH).

    Uses pypkmn's Python accessor methods (no need to parse binary state).
    Still fast because it avoids object allocation and string operations.

    Args:
        battle: PyKMN Battle object
        result: Current Result from last update
        player: Which player's perspective (P1 or P2)
        mappings: Precomputed lookup tables

    Returns:
        Dictionary of numeric feature arrays for efficient batching.
    """
    from pykmn.engine.gen1 import Slot
    from pykmn.engine.common import Player as PyKMNPlayer

    # Determine opponent
    opponent = PyKMNPlayer.P2 if player == PyKMNPlayer.P1 else PyKMNPlayer.P1

    # === Active Pokemon (player) ===
    active_species = battle.active_pokemon_species(player)
    active_species_id = mappings.species_name_to_id.get(active_species, 0)

    active_stats = battle.active_pokemon_stats(player)
    active_hp_pct = active_stats['hp'] / max(active_stats['hp'], 1)  # Avoid div by 0

    active_level = 100  # Gen 1 OU is always level 100

    # Active Pokemon moves and PP
    active_moves_with_pp = battle.moves_with_pp(player, "Active")
    active_moves = np.zeros(4, dtype=np.int32)
    active_move_pp = np.zeros(4, dtype=np.int32)
    active_move_max_pp = np.zeros(4, dtype=np.int32)

    import math
    for i, (move_name, pp) in enumerate(active_moves_with_pp[:4]):
        active_moves[i] = mappings.move_name_to_id.get(move_name, 0)
        active_move_pp[i] = pp
        # Calculate max PP with PP Ups (Gen 1 formula: base_pp * 8/5, max 61)
        if move_name in MOVES:
            base_pp = MOVES[move_name]
            active_move_max_pp[i] = min(math.floor(base_pp * 8 / 5), 61)

    # Active Pokemon boosts (with defensive defaults for missing keys)
    active_boosts = battle.boosts(player)
    # PyKMN sometimes returns incomplete boost dictionaries in edge cases
    active_boosts_safe = {
        'atk': active_boosts.get('atk', 0),
        'def': active_boosts.get('def', 0),
        'spc': active_boosts.get('spc', 0),
        'spe': active_boosts.get('spe', 0),
        'accuracy': active_boosts.get('accuracy', 0),
        'evasion': active_boosts.get('evasion', 0),
    }
    active_boosts = active_boosts_safe

    # Active Pokemon status
    # Get status from slot 1 (active Pokemon is always in slot 1)
    try:
        from pykmn.engine.gen1 import Slot
        active_status_obj = battle.status(player, Slot.ONE)
        active_status = _status_to_int(active_status_obj)
    except:
        active_status = 0

    # === Opponent Active Pokemon ===
    opp_species = battle.active_pokemon_species(opponent)
    opp_species_id = mappings.species_name_to_id.get(opp_species, 0)

    opp_stats = battle.active_pokemon_stats(opponent)
    opp_hp_pct = opp_stats['hp'] / max(opp_stats['hp'], 1)

    opp_moves_with_pp = battle.moves_with_pp(opponent, "Active")
    opp_moves = np.zeros(4, dtype=np.int32)
    opp_move_pp = np.zeros(4, dtype=np.int32)
    opp_move_max_pp = np.zeros(4, dtype=np.int32)

    for i, (move_name, pp) in enumerate(opp_moves_with_pp[:4]):
        opp_moves[i] = mappings.move_name_to_id.get(move_name, 0)
        opp_move_pp[i] = pp
        if move_name in MOVES:
            base_pp = MOVES[move_name]
            opp_move_max_pp[i] = min(math.floor(base_pp * 8 / 5), 61)

    # Opponent boosts (with defensive defaults for missing keys)
    opp_boosts = battle.boosts(opponent)
    # PyKMN sometimes returns incomplete boost dictionaries in edge cases
    opp_boosts_safe = {
        'atk': opp_boosts.get('atk', 0),
        'def': opp_boosts.get('def', 0),
        'spc': opp_boosts.get('spc', 0),
        'spe': opp_boosts.get('spe', 0),
        'accuracy': opp_boosts.get('accuracy', 0),
        'evasion': opp_boosts.get('evasion', 0),
    }
    opp_boosts = opp_boosts_safe

    # Opponent active Pokemon status
    try:
        opp_status_obj = battle.status(opponent, Slot.ONE)
        opp_status = _status_to_int(opp_status_obj)
    except:
        opp_status = 0

    # === Team Pokemon (benched) ===
    team_species_ids = np.zeros(5, dtype=np.int32)
    team_hp_pct = np.ones(5, dtype=np.float32)
    team_status = np.zeros(5, dtype=np.int32)

    for i in range(5):  # Slots 2-6 (benched Pokemon)
        slot = Slot(i + 2)
        try:
            species = battle.species(player, slot)
            species_id = mappings.species_name_to_id.get(species, 0)
            if species_id != 0:  # Pokemon exists
                team_species_ids[i] = species_id

                current_hp = battle.current_hp(player, slot)
                max_hp = battle.stats(player, slot)['hp']
                team_hp_pct[i] = current_hp / max(max_hp, 1)

                status_obj = battle.status(player, slot)
                team_status[i] = _status_to_int(status_obj)
        except:
            # Slot might be empty
            pass

    # === Opponent Team Pokemon ===
    opp_team_species_ids = np.zeros(5, dtype=np.int32)
    opp_team_hp_pct = np.ones(5, dtype=np.float32)

    for i in range(5):
        slot = Slot(i + 2)
        try:
            species = battle.species(opponent, slot)
            species_id = mappings.species_name_to_id.get(species, 0)
            if species_id != 0:
                opp_team_species_ids[i] = species_id

                current_hp = battle.current_hp(opponent, slot)
                max_hp = battle.stats(opponent, slot)['hp']
                opp_team_hp_pct[i] = current_hp / max(max_hp, 1)
        except:
            pass

    # === Previous moves ===
    player_prev_move_name = battle.last_used_move(player)
    player_prev_move_id = mappings.move_name_to_id.get(player_prev_move_name, 0)

    opp_prev_move_name = battle.last_used_move(opponent)
    opp_prev_move_id = mappings.move_name_to_id.get(opp_prev_move_name, 0)

    # === Weather and conditions ===
    # Check for side conditions using volatile flags
    # Gen 1 has Reflect and Light Screen as side-wide effects
    from pykmn.engine.gen1 import VolatileFlag

    player_has_reflect = battle.volatile(player, VolatileFlag.Reflect)
    player_has_light_screen = battle.volatile(player, VolatileFlag.LightScreen)
    opp_has_reflect = battle.volatile(opponent, VolatileFlag.Reflect)
    opp_has_light_screen = battle.volatile(opponent, VolatileFlag.LightScreen)

    # Encode side conditions (0=none, 1=reflect, 2=light_screen, 3=both)
    player_side_condition = (
        (1 if player_has_reflect else 0) +
        (2 if player_has_light_screen else 0)
    )
    opp_side_condition = (
        (1 if opp_has_reflect else 0) +
        (2 if opp_has_light_screen else 0)
    )

    # === Forced switch detection ===
    # Check if result indicates a forced switch (choice type is Switch)
    from pykmn.engine.common import ChoiceType
    forced_switch = False
    try:
        p1_choice_type = result.p1_choice_type()
        p2_choice_type = result.p2_choice_type()
        if player == PyKMNPlayer.P1:
            forced_switch = (p1_choice_type == ChoiceType.SWITCH)
        else:
            forced_switch = (p2_choice_type == ChoiceType.SWITCH)
    except:
        pass

    # Assemble features
    features = {
        # Result info
        "result_type": np.array(result.type(), dtype=np.int32),

        # Active Pokemon (player)
        "active_species_id": np.array(active_species_id, dtype=np.int32),
        "active_hp_pct": np.array(active_hp_pct, dtype=np.float32),
        "active_status": np.array(active_status, dtype=np.int32),
        "active_level": np.array(active_level, dtype=np.int32),

        "active_moves": active_moves,
        "active_move_pp": active_move_pp,
        "active_move_max_pp": active_move_max_pp,

        "active_atk_boost": np.array(active_boosts['atk'], dtype=np.int32),
        "active_def_boost": np.array(active_boosts['def'], dtype=np.int32),
        "active_spa_boost": np.array(active_boosts['spc'], dtype=np.int32),  # Gen1 has 'spc' not 'spa'/'spd'
        "active_spd_boost": np.array(active_boosts['spc'], dtype=np.int32),  # Same as spa in Gen1
        "active_spe_boost": np.array(active_boosts['spe'], dtype=np.int32),
        "active_accuracy_boost": np.array(active_boosts['accuracy'], dtype=np.int32),
        "active_evasion_boost": np.array(active_boosts['evasion'], dtype=np.int32),

        # Opponent active Pokemon
        "opponent_active_species_id": np.array(opp_species_id, dtype=np.int32),
        "opponent_active_hp_pct": np.array(opp_hp_pct, dtype=np.float32),
        "opponent_active_status": np.array(opp_status, dtype=np.int32),
        "opponent_active_level": np.array(100, dtype=np.int32),

        "opponent_active_moves": opp_moves,
        "opponent_active_move_pp": opp_move_pp,
        "opponent_active_move_max_pp": opp_move_max_pp,

        "opponent_active_atk_boost": np.array(opp_boosts['atk'], dtype=np.int32),
        "opponent_active_def_boost": np.array(opp_boosts['def'], dtype=np.int32),
        "opponent_active_spa_boost": np.array(opp_boosts['spc'], dtype=np.int32),
        "opponent_active_spd_boost": np.array(opp_boosts['spc'], dtype=np.int32),
        "opponent_active_spe_boost": np.array(opp_boosts['spe'], dtype=np.int32),

        # Team Pokemon (benched)
        "team_species_ids": team_species_ids,
        "team_hp_pct": team_hp_pct,
        "team_status": team_status,

        # Opponent team
        "opponent_team_species_ids": opp_team_species_ids,
        "opponent_team_hp_pct": opp_team_hp_pct,

        # Game state
        # Gen 1 doesn't have weather in the traditional sense
        "weather": np.array(0, dtype=np.int32),
        "side_condition": np.array(player_side_condition, dtype=np.int32),
        "opponent_side_condition": np.array(opp_side_condition, dtype=np.int32),
        # Gen 1 doesn't have field-wide conditions
        "field_condition": np.array(0, dtype=np.int32),

        # Flags
        "forced_switch": np.array(forced_switch, dtype=bool),
        "can_tera": np.array(False, dtype=bool),  # Not in Gen 1

        # Previous moves
        "player_prev_move": np.array(player_prev_move_id, dtype=np.int32),
        "opponent_prev_move": np.array(opp_prev_move_id, dtype=np.int32),
    }

    return features


def _status_to_int(status) -> int:
    """Convert pypkmn Status object to integer."""
    if status.healthy():
        return 0
    elif status.burned():
        return 1
    elif status.frozen():
        return 2
    elif status.paralyzed():
        return 3
    elif status.poisoned():
        return 4
    elif status.asleep():
        return 5
    else:
        return 0


def features_to_universal_state(
    features: Dict[str, np.ndarray],
    mappings: Mappings,
    battle_format: str = "gen1ou",
) -> UniversalState:
    """
    Convert numeric features to UniversalState (SLOW PATH).

    This function is only called when saving trajectories, not in the hot loop.
    It can do string lookups, object construction, etc.

    Args:
        features: Dictionary of numeric feature arrays from pykmn_to_features_raw
        mappings: Precomputed lookup tables
        battle_format: Battle format string (e.g., "gen1ou")

    Returns:
        UniversalState object compatible with metamon training pipeline.
    """
    # Convert active Pokemon
    player_active = _features_to_universal_pokemon(
        species_id=int(features["active_species_id"]),
        hp_pct=float(features["active_hp_pct"]),
        status=int(features["active_status"]),
        level=int(features["active_level"]),
        moves=features["active_moves"],
        move_pp=features["active_move_pp"],
        move_max_pp=features["active_move_max_pp"],
        atk_boost=int(features["active_atk_boost"]),
        def_boost=int(features["active_def_boost"]),
        spa_boost=int(features["active_spa_boost"]),
        spd_boost=int(features["active_spd_boost"]),
        spe_boost=int(features["active_spe_boost"]),
        accuracy_boost=int(features["active_accuracy_boost"]),
        evasion_boost=int(features["active_evasion_boost"]),
        mappings=mappings,
    )

    opponent_active = _features_to_universal_pokemon(
        species_id=int(features["opponent_active_species_id"]),
        hp_pct=float(features["opponent_active_hp_pct"]),
        status=int(features["opponent_active_status"]),
        level=int(features["opponent_active_level"]),
        moves=features["opponent_active_moves"],
        move_pp=features["opponent_active_move_pp"],
        move_max_pp=np.zeros(4, dtype=np.int32),  # May not know max PP for opponent
        atk_boost=int(features["opponent_active_atk_boost"]),
        def_boost=int(features["opponent_active_def_boost"]),
        spa_boost=int(features["opponent_active_spa_boost"]),
        spd_boost=int(features["opponent_active_spd_boost"]),
        spe_boost=int(features["opponent_active_spe_boost"]),
        accuracy_boost=0,
        evasion_boost=0,
        mappings=mappings,
    )

    # Convert benched Pokemon
    available_switches = []
    for i in range(5):  # Gen1 has 5 benched Pokemon (6 total - 1 active)
        species_id = int(features["team_species_ids"][i])
        if species_id == 0:  # No Pokemon in this slot
            continue

        switch_pokemon = _features_to_universal_pokemon(
            species_id=species_id,
            hp_pct=float(features["team_hp_pct"][i]),
            status=int(features["team_status"][i]),
            level=100,
            moves=np.zeros(4, dtype=np.int32),  # Don't know moves for benched
            move_pp=np.zeros(4, dtype=np.int32),
            move_max_pp=np.zeros(4, dtype=np.int32),
            atk_boost=0,
            def_boost=0,
            spa_boost=0,
            spd_boost=0,
            spe_boost=0,
            accuracy_boost=0,
            evasion_boost=0,
            mappings=mappings,
        )
        available_switches.append(switch_pokemon)

    # Convert previous moves
    player_prev_move = _move_id_to_universal_move(
        int(features["player_prev_move"]), mappings
    )
    opponent_prev_move = _move_id_to_universal_move(
        int(features["opponent_prev_move"]), mappings
    )

    # Count opponents remaining
    opponents_remaining = 0
    for i in range(5):
        if features["opponent_team_hp_pct"][i] > 0:
            opponents_remaining += 1
    opponents_remaining += 1  # Add the active Pokemon

    # Determine battle outcome
    result_type = int(features["result_type"])
    battle_won = result_type == ResultType.PLAYER_1_WIN
    battle_lost = result_type == ResultType.PLAYER_2_WIN

    # Construct UniversalState
    return UniversalState(
        format=battle_format,
        player_active_pokemon=player_active,
        opponent_active_pokemon=opponent_active,
        available_switches=available_switches,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=opponents_remaining,
        player_conditions=_decode_side_condition(int(features["side_condition"])),
        opponent_conditions=_decode_side_condition(int(features["opponent_side_condition"])),
        weather=_decode_weather(int(features["weather"])),
        battle_field=_decode_field(int(features["field_condition"])),
        forced_switch=bool(features["forced_switch"]),
        battle_won=battle_won,
        battle_lost=battle_lost,
        can_tera=bool(features["can_tera"]),
        opponent_teampreview=[],  # TODO: Extract if available
    )


def _features_to_universal_pokemon(
    species_id: int,
    hp_pct: float,
    status: int,
    level: int,
    moves: np.ndarray,
    move_pp: np.ndarray,
    move_max_pp: np.ndarray,
    atk_boost: int,
    def_boost: int,
    spa_boost: int,
    spd_boost: int,
    spe_boost: int,
    accuracy_boost: int,
    evasion_boost: int,
    mappings: Mappings,
) -> UniversalPokemon:
    """Convert numeric Pokemon features to UniversalPokemon."""
    # Look up species name
    species_name = mappings.species_id_to_name.get(species_id, "missingno")

    # Get base stats from species data
    # Gen 1 structure: {'stats': {...}, 'types': [...]}
    species_data = SPECIES.get(species_name, None)
    if species_data:
        stats = species_data['stats']
        base_hp = stats['hp']
        base_atk = stats['atk']
        base_def = stats['def']
        base_spe = stats['spe']
        # Gen 1 has only 'spc' (special), not separate spa/spd
        base_spc = stats['spc']
        base_spa = base_spc  # Use spc for both spa and spd
        base_spd = base_spc
        types_str = "_".join(species_data['types'])
    else:
        base_hp = base_atk = base_def = base_spa = base_spd = base_spe = 100
        types_str = "normal"

    # Convert moves
    # Note: pypkmn's MOVES dict only contains base PP values (int), not full move data
    # We create minimal UniversalMove objects with placeholder data
    universal_moves = []
    for i in range(4):
        move_id = int(moves[i])
        if move_id == 0:
            universal_moves.append(UniversalMove.blank_move())
        else:
            move_name = mappings.move_id_to_name.get(move_id, "nomove")
            # MOVES[move_name] is just an int (base PP), not detailed move data
            # Create minimal UniversalMove with placeholders
            universal_moves.append(
                UniversalMove(
                    name=move_name,
                    move_type="normal",  # Placeholder - not available from pypkmn
                    category="physical",  # Placeholder - not available from pypkmn
                    base_power=50,  # Placeholder - not available from pypkmn
                    accuracy=1.0,  # Placeholder - not available from pypkmn
                    priority=0,  # Placeholder - not available from pypkmn
                    current_pp=int(move_pp[i]),
                    max_pp=int(move_max_pp[i]),
                )
            )

    return UniversalPokemon(
        name=species_name,
        hp_pct=hp_pct,
        types=types_str,
        item="noitem",  # Gen1 doesn't have held items
        ability="noability",  # Gen1 doesn't have abilities
        lvl=level,
        status=_decode_status(status),
        effect="noeffect",  # TODO: Decode volatile statuses
        moves=universal_moves,
        atk_boost=atk_boost,
        spa_boost=spa_boost,
        def_boost=def_boost,
        spd_boost=spd_boost,
        spe_boost=spe_boost,
        accuracy_boost=accuracy_boost,
        evasion_boost=evasion_boost,
        base_atk=base_atk,
        base_spa=base_spa,
        base_def=base_def,
        base_spd=base_spd,
        base_spe=base_spe,
        base_hp=base_hp,
        tera_type="",  # Gen1 doesn't have tera
        base_species=species_name,
    )


def _move_id_to_universal_move(move_id: int, mappings: Mappings) -> UniversalMove:
    """Convert move ID to UniversalMove.

    Note: pypkmn's MOVES dict only contains base PP values (integers), not full
    move data. We use placeholder values for type/power/accuracy since these
    aren't needed by observation spaces (they only use move names as tokens).
    """
    if move_id == 0:
        return UniversalMove.blank_move()

    move_name = mappings.move_id_to_name.get(move_id, "nomove")
    base_pp = MOVES.get(move_name, 0)  # MOVES contains only PP values (integers)

    if not base_pp:
        return UniversalMove.blank_move()

    # Calculate max PP using Gen1 formula
    max_pp = min(math.floor(base_pp * 8 / 5), 61)

    return UniversalMove(
        name=move_name,
        move_type="normal",  # Placeholder - not available in pypkmn
        category="physical",  # Placeholder - not available in pypkmn
        base_power=50,  # Placeholder - not available in pypkmn
        accuracy=1.0,  # Placeholder - not available in pypkmn
        priority=0,  # Placeholder - not available in pypkmn
        current_pp=0,  # Don't know PP for previous move
        max_pp=max_pp,
    )


def _decode_status(status: int) -> str:
    """Decode numeric status to string."""
    status_map = {
        0: "nostatus",
        1: "brn",
        2: "frz",
        3: "par",
        4: "psn",
        5: "slp",
        6: "tox",
    }
    return status_map.get(status, "nostatus")


def _decode_side_condition(condition: int) -> str:
    """Decode numeric side condition to string."""
    condition_map = {
        0: "noconditions",
        1: "reflect",
        2: "light_screen",
        3: "mist",
        4: "safeguard",
        # Add more as needed
    }
    return condition_map.get(condition, "noconditions")


def _decode_weather(weather: int) -> str:
    """Decode numeric weather to string."""
    weather_map = {
        0: "noweather",
        1: "raindance",
        2: "sunnyday",
        3: "sandstorm",
        4: "hail",
    }
    return weather_map.get(weather, "noweather")


def _decode_field(field: int) -> str:
    """Decode numeric field condition to string."""
    field_map = {
        0: "nofield",
        1: "trickroom",
        2: "gravity",
        # Add more as needed
    }
    return field_map.get(field, "nofield")
