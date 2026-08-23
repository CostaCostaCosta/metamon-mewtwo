"""
Gen 3 ROM-native battle state schema ("schema v2").

Extends the Gen 1 canonical schema (``schema.py``) for Gen 3:
- wider vocab: species 1-386, moves 1-354, abilities 1-76, gen3-legal items
- per-Pokemon categoricals 9 -> 11 (APPEND item, ability after move_4_id)
- per-Pokemon masks 4 -> 6 (APPEND item_revealed, ability_revealed)
- side-condition enum gains SPIKES = 8 (single-enum stays lossy for concurrent
  screens -- a documented known mismatch)

Layout is APPEND-ONLY relative to the gen1 tensor order so the two encoders stay
structurally aligned (and the C encoder in poke-plastic-ox mirrors the same
ordering). 13 slots / 9 actions / 31 per-mon numerical / 6 global cat + 3 global
num are all unchanged from gen1.

Tensor shapes (gen3):
    global_cat:        (6,)      int32
    global_num:        (3,)      float32
    pokemon_cat:       (13, 11)  int32
    pokemon_move_cat:  (13, 4)   int32
    pokemon_move_type: (13, 4)   int32
    pokemon_num:       (13, 31)  float32
    pokemon_mask:      (13, 6)   int32
    legal_action_mask: (9,)      int32
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

# Reuse every gen-agnostic enum/constant from the gen1 schema.
from .schema import (  # noqa: F401
    UNKNOWN_ID,
    NONE_ID,
    SPECIES_UNKNOWN,
    MOVE_UNKNOWN,
    MOVE_NONE,
    TYPE_NONE,
    TYPE_MAX,
    STATUS_NONE,
    STATUS_FAINT,
    STATUS_UNKNOWN,
    STATUS_MAX,
    WEATHER_NONE,
    WEATHER_MAX,
    FIELD_NONE,
    FIELD_MAX,
    CATEGORY_NONE,
    CATEGORY_PHYSICAL,
    CATEGORY_SPECIAL,
    CATEGORY_STATUS,
    CATEGORY_UNKNOWN,
    EFFECT_NONE,
    EFFECT_UNKNOWN,
    NUM_ACTIONS,
    NUM_POKEMON_SLOTS,
    NUM_MOVES_PER_POKEMON,
    POKEMON_NUM_FEATURES,
    POKEMON_NUM_LEN,
    POKEMON_MOVE_CAT_FEATURES,
    POKEMON_MOVE_CAT_LEN,
    POKEMON_MOVE_TYPE_FEATURES,
    POKEMON_MOVE_TYPE_LEN,
    GLOBAL_CAT_FEATURES,
    GLOBAL_CAT_LEN,
    SLOT_PLAYER_ACTIVE,
    SLOT_SWITCH_0,
    SLOT_SWITCH_1,
    SLOT_SWITCH_2,
    SLOT_SWITCH_3,
    SLOT_SWITCH_4,
    SLOT_OPPONENT_ACTIVE,
    SLOT_REVEALED_OPP_0,
    SLOT_REVEALED_OPP_1,
    SLOT_REVEALED_OPP_2,
    SLOT_REVEALED_OPP_3,
    SLOT_REVEALED_OPP_4,
    SLOT_REVEALED_OPP_5,
)

# ---- Gen 3 vocab bounds ----
SPECIES_MAX_GEN3 = 386
MOVE_MAX_GEN3 = 354
ABILITY_MAX_GEN3 = 76
# Max pokeemerald-expansion ITEM_* enum value used by the gen3-legal set
# (ITEM_BERSERK_GENE = 798). Embedding table is sized ITEM_VOCAB_SIZE = max+1.
ITEM_MAX_GEN3 = 798
ITEM_VOCAB_SIZE = ITEM_MAX_GEN3 + 1

ABILITY_NONE = 0
ITEM_NONE = 0

# Side condition enum: gen1 values + SPIKES=8 (kept SIDE_COND_UNKNOWN=7).
SIDE_COND_NONE = 0
SIDE_COND_REFLECT = 1
SIDE_COND_LIGHTSCREEN = 2
SIDE_COND_SAFEGUARD = 3
SIDE_COND_MIST = 4
SIDE_COND_TAILWIND = 5
SIDE_COND_AURORA_VEIL = 6
SIDE_COND_UNKNOWN = 7
SIDE_COND_SPIKES = 8
SIDE_COND_MAX_GEN3 = 8

# ---- Gen 3 global numerical features (5 = gen1 3 + 2 spikes-layer counts) ----
# APPEND-ONLY relative to gen1 (turn_norm, opponents_remaining, forced_switch
# stay first). v6.1 parsed data populates UniversalState.player/opponent_spikes_layers
# (0-3); the ROM schema finally surfaces them (the single side-cond enum is lossy).
SPIKES_LAYER_MAX = 3.0
GLOBAL_NUM_FEATURES = [
    "turn_norm",  # turn / 200.0 (clipped to 1.0)
    "opponents_remaining",  # / 6.0
    "forced_switch",  # 0.0 or 1.0
    "player_spikes_layers",  # appended (gen3): 0-3 / 3.0
    "opponent_spikes_layers",  # appended (gen3): 0-3 / 3.0
]
GLOBAL_NUM_LEN = len(GLOBAL_NUM_FEATURES)  # 5

# ---- Gen 3 per-Pokemon categorical features (11 = gen1 9 + item + ability) ----
POKEMON_CAT_FEATURES = [
    "species",
    "type_1",
    "type_2",
    "status",
    "effect",
    "move_1_id",
    "move_2_id",
    "move_3_id",
    "move_4_id",
    "item",  # appended (gen3)
    "ability",  # appended (gen3)
]
POKEMON_CAT_LEN = len(POKEMON_CAT_FEATURES)  # 11

# ---- Gen 3 per-Pokemon masks (6 = gen1 4 + item_revealed + ability_revealed) ----
POKEMON_MASK_FEATURES = [
    "valid",
    "fainted",
    "moves_revealed",
    "hp_known",
    "item_revealed",  # appended (gen3)
    "ability_revealed",  # appended (gen3)
]
POKEMON_MASK_LEN = len(POKEMON_MASK_FEATURES)  # 6

TOTAL_CAT = GLOBAL_CAT_LEN + NUM_POKEMON_SLOTS * (
    POKEMON_CAT_LEN + POKEMON_MOVE_CAT_LEN + POKEMON_MOVE_TYPE_LEN
)
TOTAL_NUM = GLOBAL_NUM_LEN + NUM_POKEMON_SLOTS * POKEMON_NUM_LEN
TOTAL_MASK = NUM_POKEMON_SLOTS * POKEMON_MASK_LEN + NUM_ACTIONS


@dataclass
class Gen3PokemonFeatures:
    species: int = SPECIES_UNKNOWN
    type_1: int = TYPE_NONE
    type_2: int = TYPE_NONE
    status: int = STATUS_NONE
    effect: int = EFFECT_NONE
    move_ids: List[int] = field(
        default_factory=lambda: [MOVE_UNKNOWN] * NUM_MOVES_PER_POKEMON
    )
    move_categories: List[int] = field(
        default_factory=lambda: [CATEGORY_NONE] * NUM_MOVES_PER_POKEMON
    )
    move_types: List[int] = field(
        default_factory=lambda: [TYPE_NONE] * NUM_MOVES_PER_POKEMON
    )
    item: int = ITEM_NONE
    ability: int = ABILITY_NONE

    hp_fraction: float = -1.0
    level_norm: float = 0.0
    base_atk_norm: float = 0.0
    base_spa_norm: float = 0.0
    base_def_norm: float = 0.0
    base_spd_norm: float = 0.0
    base_spe_norm: float = 0.0
    base_hp_norm: float = 0.0
    boosts: List[float] = field(default_factory=lambda: [0.0] * 7)
    move_bp: List[float] = field(default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON)
    move_acc: List[float] = field(
        default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON
    )
    move_pri: List[float] = field(
        default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON
    )
    move_pp: List[float] = field(default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON)

    valid: bool = False
    fainted: bool = False
    moves_revealed: bool = False
    hp_known: bool = False
    item_revealed: bool = False
    ability_revealed: bool = False


from .schema import GlobalFeatures  # noqa: E402  (unchanged in gen3)


@dataclass
class Gen3RomBattleState:
    global_features: GlobalFeatures = field(default_factory=GlobalFeatures)
    pokemon: List[Gen3PokemonFeatures] = field(
        default_factory=lambda: [
            Gen3PokemonFeatures() for _ in range(NUM_POKEMON_SLOTS)
        ]
    )
    legal_action_mask: List[bool] = field(default_factory=lambda: [False] * NUM_ACTIONS)

    def to_tensors(self) -> dict:
        global_cat = np.array(
            [
                self.global_features.weather,
                self.global_features.field_effect,
                self.global_features.player_side_cond,
                self.global_features.opponent_side_cond,
                self.global_features.player_prev_move,
                self.global_features.opponent_prev_move,
            ],
            dtype=np.int32,
        )
        global_num = np.array(
            [
                self.global_features.turn_norm,
                self.global_features.opponents_remaining,
                self.global_features.forced_switch,
                self.global_features.player_spikes_layers,
                self.global_features.opponent_spikes_layers,
            ],
            dtype=np.float32,
        )

        pokemon_cat = np.zeros((NUM_POKEMON_SLOTS, POKEMON_CAT_LEN), dtype=np.int32)
        pokemon_move_cat = np.zeros(
            (NUM_POKEMON_SLOTS, POKEMON_MOVE_CAT_LEN), dtype=np.int32
        )
        pokemon_move_type = np.zeros(
            (NUM_POKEMON_SLOTS, POKEMON_MOVE_TYPE_LEN), dtype=np.int32
        )
        pokemon_num = np.zeros((NUM_POKEMON_SLOTS, POKEMON_NUM_LEN), dtype=np.float32)
        pokemon_mask = np.zeros((NUM_POKEMON_SLOTS, POKEMON_MASK_LEN), dtype=np.int32)

        for i, p in enumerate(self.pokemon):
            pokemon_cat[i] = [
                p.species,
                p.type_1,
                p.type_2,
                p.status,
                p.effect,
                p.move_ids[0],
                p.move_ids[1],
                p.move_ids[2],
                p.move_ids[3],
                p.item,
                p.ability,
            ]
            pokemon_move_cat[i] = p.move_categories
            pokemon_move_type[i] = p.move_types
            pokemon_num[i] = [
                p.hp_fraction,
                p.level_norm,
                p.base_atk_norm,
                p.base_spa_norm,
                p.base_def_norm,
                p.base_spd_norm,
                p.base_spe_norm,
                p.base_hp_norm,
                p.boosts[0],
                p.boosts[1],
                p.boosts[2],
                p.boosts[3],
                p.boosts[4],
                p.boosts[5],
                p.boosts[6],
                p.move_bp[0],
                p.move_acc[0],
                p.move_pri[0],
                p.move_pp[0],
                p.move_bp[1],
                p.move_acc[1],
                p.move_pri[1],
                p.move_pp[1],
                p.move_bp[2],
                p.move_acc[2],
                p.move_pri[2],
                p.move_pp[2],
                p.move_bp[3],
                p.move_acc[3],
                p.move_pri[3],
                p.move_pp[3],
            ]
            pokemon_mask[i] = [
                int(p.valid),
                int(p.fainted),
                int(p.moves_revealed),
                int(p.hp_known),
                int(p.item_revealed),
                int(p.ability_revealed),
            ]

        legal_mask = np.array(self.legal_action_mask, dtype=np.int32)
        return {
            "global_cat": global_cat,
            "global_num": global_num,
            "pokemon_cat": pokemon_cat,
            "pokemon_move_cat": pokemon_move_cat,
            "pokemon_move_type": pokemon_move_type,
            "pokemon_num": pokemon_num,
            "pokemon_mask": pokemon_mask,
            "legal_action_mask": legal_mask,
        }
