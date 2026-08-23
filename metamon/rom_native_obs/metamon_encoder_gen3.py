"""
Gen 3 Metamon-side encoder: UniversalState -> Gen3RomBattleState.

Mirrors ``metamon_encoder.RomObservationEncoder`` (gen1) but for the gen3
schema v2: wider vocab, per-Pokemon item + ability categoricals, and two extra
reveal masks. Ability/item use ROM-canonical IDs (see mappings_gen3).

Information visibility (gen3):
- Player side: full info (moves, item, ability, HP, boosts, PP).
- Opponent active: species/types/status/boosts/HP visible; moves only once
  revealed; item/ability only once revealed (gen3 has no team preview and
  abilities/items are hidden until they trigger / are observed).
- Revealed opponent bench: species known; HP/moves/item/ability only as
  previously revealed (tracked across timesteps like the gen1 moves memory).
"""
from __future__ import annotations

import copy
from typing import Optional

from metamon.interface import (
    UniversalState, UniversalPokemon, UniversalMove, consistent_move_order,
    consistent_pokemon_order,
)
from metamon.backend.replay_parser.str_parsing import pokemon_name, clean_name

from .schema_gen3 import (
    Gen3RomBattleState, Gen3PokemonFeatures, GlobalFeatures,
    NUM_POKEMON_SLOTS, NUM_MOVES_PER_POKEMON, NUM_ACTIONS,
    SLOT_PLAYER_ACTIVE, SLOT_SWITCH_0, SLOT_OPPONENT_ACTIVE, SLOT_REVEALED_OPP_0,
    SPECIES_UNKNOWN, MOVE_UNKNOWN, MOVE_NONE,
    TYPE_NONE, STATUS_NONE, STATUS_FAINT,
    WEATHER_NONE, SIDE_COND_NONE, FIELD_NONE, EFFECT_NONE,
    CATEGORY_NONE, ITEM_NONE, ABILITY_NONE, SPIKES_LAYER_MAX,
)
from .mappings import (
    TYPE_NAME_TO_ID, STATUS_NAME_TO_ID, WEATHER_NAME_TO_ID,
    FIELD_NAME_TO_ID, EFFECT_NAME_TO_ID, CATEGORY_NAME_TO_ID,
)
from .mappings_gen3 import (
    species_name_to_id, move_name_to_id, ability_name_to_id, item_name_to_id,
    GEN3_SIDE_COND_NAME_TO_ID,
)


def _type_str_to_id(type_str: str) -> int:
    if not type_str or type_str == "notype":
        return TYPE_NONE
    parts = type_str.strip().split()
    return TYPE_NAME_TO_ID.get(clean_name(parts[0]), TYPE_NONE) if parts else TYPE_NONE


def _types_str_to_ids(type_str: str) -> tuple[int, int]:
    if not type_str or type_str == "notype":
        return (TYPE_NONE, TYPE_NONE)
    parts = type_str.strip().split()
    t1 = TYPE_NAME_TO_ID.get(clean_name(parts[0]), TYPE_NONE) if len(parts) > 0 else TYPE_NONE
    t2 = TYPE_NAME_TO_ID.get(clean_name(parts[1]), TYPE_NONE) if len(parts) > 1 else TYPE_NONE
    return (t1, t2)


def _status_str_to_id(s: str) -> int:
    return STATUS_NAME_TO_ID.get(clean_name(s), STATUS_NONE) if s else STATUS_NONE


def _weather_str_to_id(s: str) -> int:
    return WEATHER_NAME_TO_ID.get(clean_name(s), WEATHER_NONE) if s else WEATHER_NONE


def _side_cond_str_to_id(s: str) -> int:
    return GEN3_SIDE_COND_NAME_TO_ID.get(clean_name(s), SIDE_COND_NONE) if s else SIDE_COND_NONE


def _field_str_to_id(s: str) -> int:
    return FIELD_NAME_TO_ID.get(clean_name(s), FIELD_NONE) if s else FIELD_NONE


def _effect_str_to_id(s: str) -> int:
    return EFFECT_NAME_TO_ID.get(clean_name(s), EFFECT_NONE) if s else EFFECT_NONE


def _category_str_to_id(s: str) -> int:
    return CATEGORY_NAME_TO_ID.get(clean_name(s), CATEGORY_NONE) if s else CATEGORY_NONE


def _encode_move(move: UniversalMove, is_revealed: bool = True) -> dict:
    if move is None:
        return {"move_id": MOVE_UNKNOWN, "category": CATEGORY_NONE, "type": TYPE_NONE,
                "bp": -2.0, "acc": -2.0, "pri": -2.0, "pp": -2.0}
    if is_revealed:
        bp = move.base_power / 200.0
        acc = float(move.accuracy) if move.accuracy is not None else -2.0
        pri = move.priority / 5.0
        pp = (move.current_pp / move.max_pp) if move.max_pp > 0 else -2.0
        return {
            "move_id": move_name_to_id(move.name),
            "category": _category_str_to_id(move.category),
            "type": _type_str_to_id(move.move_type),
            "bp": bp, "acc": acc, "pri": pri, "pp": pp,
        }
    return {"move_id": MOVE_UNKNOWN, "category": CATEGORY_NONE, "type": TYPE_NONE,
            "bp": -2.0, "acc": -2.0, "pri": -2.0, "pp": -2.0}


def _encode_pokemon(
    pokemon: UniversalPokemon,
    is_active: bool,
    is_opponent: bool,
    moves_revealed: bool,
    hp_known: bool = True,
    item_revealed: bool = True,
    ability_revealed: bool = True,
) -> Gen3PokemonFeatures:
    if pokemon is None:
        return Gen3PokemonFeatures()

    pf = Gen3PokemonFeatures()
    pf.valid = True
    pf.species = species_name_to_id(pokemon.name)
    pf.type_1, pf.type_2 = _types_str_to_ids(pokemon.types)
    pf.status = _status_str_to_id(pokemon.status)
    pf.effect = _effect_str_to_id(pokemon.effect)

    if hp_known:
        pf.hp_fraction = float(pokemon.hp_pct)
        pf.hp_known = True
    else:
        pf.hp_fraction = -1.0
        pf.hp_known = False

    pf.fainted = (pf.status == STATUS_FAINT) or (hp_known and pokemon.hp_pct <= 0.0)
    pf.level_norm = float(pokemon.lvl) / 100.0
    pf.base_atk_norm = float(pokemon.base_atk) / 255.0
    pf.base_spa_norm = float(pokemon.base_spa) / 255.0
    pf.base_def_norm = float(pokemon.base_def) / 255.0
    pf.base_spd_norm = float(pokemon.base_spd) / 255.0
    pf.base_spe_norm = float(pokemon.base_spe) / 255.0
    pf.base_hp_norm = float(pokemon.base_hp) / 255.0

    if is_active:
        pf.boosts = [
            float(pokemon.atk_boost) / 6.0, float(pokemon.spa_boost) / 6.0,
            float(pokemon.def_boost) / 6.0, float(pokemon.spd_boost) / 6.0,
            float(pokemon.spe_boost) / 6.0, float(pokemon.accuracy_boost) / 6.0,
            float(pokemon.evasion_boost) / 6.0,
        ]
    else:
        pf.boosts = [0.0] * 7

    # Item / ability (only surfaced when revealed)
    pf.item_revealed = bool(item_revealed)
    pf.ability_revealed = bool(ability_revealed)
    pf.item = item_name_to_id(pokemon.item) if item_revealed else ITEM_NONE
    pf.ability = ability_name_to_id(pokemon.ability) if ability_revealed else ABILITY_NONE

    # Moves
    pf.moves_revealed = moves_revealed
    sorted_moves = consistent_move_order(list(pokemon.moves))[:NUM_MOVES_PER_POKEMON]
    for i in range(NUM_MOVES_PER_POKEMON):
        if i < len(sorted_moves):
            md = _encode_move(sorted_moves[i], is_revealed=moves_revealed)
            pf.move_ids[i] = md["move_id"]
            pf.move_categories[i] = md["category"]
            pf.move_types[i] = md["type"]
            pf.move_bp[i] = md["bp"]
            pf.move_acc[i] = md["acc"]
            pf.move_pri[i] = md["pri"]
            pf.move_pp[i] = md["pp"]
        else:
            pf.move_ids[i] = MOVE_UNKNOWN
            pf.move_categories[i] = CATEGORY_NONE
            pf.move_types[i] = TYPE_NONE
            pf.move_bp[i] = -2.0
            pf.move_acc[i] = -2.0
            pf.move_pri[i] = -2.0
            pf.move_pp[i] = -2.0
    return pf


class Gen3RomObservationEncoder:
    """UniversalState -> Gen3RomBattleState with cross-timestep reveal memory.

    Tracks revealed opponent species, moves, items, and abilities across the
    battle (same pattern as the gen1 encoder's revealed-opponent memory).
    """

    def __init__(self, gen: int = 3):
        self.gen = gen
        self.reset()

    def reset(self):
        self.revealed_opponents: dict[str, UniversalPokemon] = {}
        self.revealed_opponent_order: list[str] = []
        # per-species reveal memory for item/ability
        self.revealed_items: dict[str, int] = {}
        self.revealed_abilities: dict[str, int] = {}
        self._turn_count = 0

    @staticmethod
    def _is_known_item(p: UniversalPokemon) -> bool:
        return item_name_to_id(getattr(p, "item", "")) != 0

    @staticmethod
    def _is_known_ability(p: UniversalPokemon) -> bool:
        return ability_name_to_id(getattr(p, "ability", "")) != 0

    def _update_reveal_memory(self, key: str, p: UniversalPokemon):
        if self._is_known_item(p):
            self.revealed_items[key] = item_name_to_id(p.item)
        if self._is_known_ability(p):
            self.revealed_abilities[key] = ability_name_to_id(p.ability)

    def encode(self, state: UniversalState) -> Gen3RomBattleState:
        self._turn_count += 1
        opponent = state.opponent_active_pokemon
        opp_key = pokemon_name(opponent.base_species or opponent.name)
        if opp_key and opp_key not in self.revealed_opponents:
            self.revealed_opponent_order.append(opp_key)
        self.revealed_opponents[opp_key] = copy.deepcopy(opponent)
        self._update_reveal_memory(opp_key, opponent)

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
            # v6.1 parsed data populates these (0-3); clamp/normalize to [0,1].
            player_spikes_layers=min(float(getattr(state, "player_spikes_layers", 0)) / SPIKES_LAYER_MAX, 1.0),
            opponent_spikes_layers=min(float(getattr(state, "opponent_spikes_layers", 0)) / SPIKES_LAYER_MAX, 1.0),
        )

        pokemon_list = [Gen3PokemonFeatures() for _ in range(NUM_POKEMON_SLOTS)]

        # Slot 0: player active (full info)
        pokemon_list[SLOT_PLAYER_ACTIVE] = _encode_pokemon(
            state.player_active_pokemon, is_active=True, is_opponent=False,
            moves_revealed=True, hp_known=True, item_revealed=True, ability_revealed=True,
        )

        # Slots 1-5: player bench (full info)
        switches = consistent_pokemon_order(state.available_switches)
        for i in range(5):
            if i < len(switches):
                pokemon_list[SLOT_SWITCH_0 + i] = _encode_pokemon(
                    switches[i], is_active=False, is_opponent=False,
                    moves_revealed=True, hp_known=True, item_revealed=True, ability_revealed=True,
                )

        # Slot 6: opponent active (HP/species/types/status/boosts visible; moves/item/ability only if revealed)
        opp_moves_revealed = len(opponent.moves) > 0
        pokemon_list[SLOT_OPPONENT_ACTIVE] = _encode_pokemon(
            opponent, is_active=True, is_opponent=True,
            moves_revealed=opp_moves_revealed, hp_known=True,
            item_revealed=(opp_key in self.revealed_items),
            ability_revealed=(opp_key in self.revealed_abilities),
        )

        # Slots 7-12: revealed opponent bench (from memory)
        revealed = [
            self.revealed_opponents[n] for n in self.revealed_opponent_order[:6]
            if n in self.revealed_opponents
        ]
        revealed = [p for p in revealed if pokemon_name(p.base_species or p.name) != opp_key]
        for i in range(6):
            if i < len(revealed):
                rp = revealed[i]
                rk = pokemon_name(rp.base_species or rp.name)
                pokemon_list[SLOT_REVEALED_OPP_0 + i] = _encode_pokemon(
                    rp, is_active=False, is_opponent=True,
                    moves_revealed=(len(rp.moves) > 0), hp_known=False,
                    item_revealed=(rk in self.revealed_items),
                    ability_revealed=(rk in self.revealed_abilities),
                )

        legal_mask = [False] * NUM_ACTIONS
        if not state.forced_switch:
            for i in range(min(len(state.player_active_pokemon.moves), 4)):
                legal_mask[i] = True
        for i in range(min(len(switches), 5)):
            legal_mask[4 + i] = True

        return Gen3RomBattleState(
            global_features=global_feats,
            pokemon=pokemon_list,
            legal_action_mask=legal_mask,
        )

    def encode_from_dict(self, state_dict: dict) -> Gen3RomBattleState:
        return self.encode(UniversalState.from_dict(state_dict))
