"""
Canonical ROM-native battle state representation.

This module defines the logical schema for a compact, structured, ROM-friendly
battle observation that can be reproduced from both Metamon's UniversalState
and pokeemerald-expansion's battle engine state.

Design principles:
- Fixed-width categorical IDs (no strings, no tokens)
- Normalized numerical values (0.0-1.0 or small integers)
- Explicit masking for unknown/unrevealed information
- No dependency on text-token observation encoders
- Deterministic and documented

The schema is organized as:
  - Global features (weather, field effects, side conditions, turn, forced switch)
  - Per-Pokémon features for up to 13 slots (1 player active + 5 switches + 1 opponent active + 6 revealed opponents)
  - Per-move features for up to 4 moves per Pokémon
  - Legal action mask (9 actions: 4 moves + 5 switches)

All categorical values use stable integer IDs. Unknown/unrevealed values
use an explicit UNKNOWN sentinel rather than arbitrary zeros.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ============================================================================
# Sentinel values
# ============================================================================

UNKNOWN_ID = 0       # Used for any unknown/unrevealed categorical value
NONE_ID = 0          # Used for "none" / "no status" / "no weather" etc.
                     # NOTE: UNKNOWN and NONE share 0 because in practice
                     # "unknown" and "no effect present" are indistinguishable
                     # from the player's perspective for many fields.
                     # For fields where they must be distinguished, a separate
                     # validity mask is used.


# ============================================================================
# ID Space Definitions
# ============================================================================

# Species: National Dex number (1-151 for Gen1). 0 = unknown/unrevealed.
SPECIES_UNKNOWN = 0
SPECIES_MAX_GEN1 = 151

# Moves: Showdown move num (1-165 for Gen1). 0 = unknown/no move.
MOVE_UNKNOWN = 0
MOVE_NONE = 0
MOVE_MAX_GEN1 = 165

# Types: Match pokeemerald-expansion Type enum values for direct compatibility.
TYPE_NONE = 0
TYPE_NORMAL = 1
TYPE_FIGHTING = 2
TYPE_FLYING = 3
TYPE_POISON = 4
TYPE_GROUND = 5
TYPE_ROCK = 6
TYPE_BIRD = 7
TYPE_BUG = 8
TYPE_GHOST = 9
TYPE_STEEL = 10
TYPE_FIRE = 11
TYPE_WATER = 12
TYPE_GRASS = 13
TYPE_ELECTRIC = 14
TYPE_PSYCHIC = 15
TYPE_ICE = 16
TYPE_DRAGON = 17
TYPE_DARK = 18
TYPE_FAIRY = 19
TYPE_MAX = 19

# Status conditions (major/volatile status on the Pokémon)
STATUS_NONE = 0
STATUS_SLEEP = 1
STATUS_POISON = 2
STATUS_BURN = 3
STATUS_FREEZE = 4
STATUS_PARALYSIS = 5
STATUS_TOXIC = 6
STATUS_FAINT = 7
STATUS_UNKNOWN = 8
STATUS_MAX = 8

# Weather (Gen1 only has rain, sun, sandstorm - but we define all)
WEATHER_NONE = 0
WEATHER_RAIN = 1
WEATHER_SUN = 2
WEATHER_SANDSTORM = 3
WEATHER_HAIL = 4
WEATHER_SNOW = 5
WEATHER_FOG = 6
WEATHER_UNKNOWN = 7
WEATHER_MAX = 7

# Side conditions (bitfield-compatible IDs)
SIDE_COND_NONE = 0
SIDE_COND_REFLECT = 1
SIDE_COND_LIGHTSCREEN = 2
SIDE_COND_SAFEGUARD = 3
SIDE_COND_MIST = 4
SIDE_COND_TAILWIND = 5
SIDE_COND_AURORA_VEIL = 6
SIDE_COND_UNKNOWN = 7
SIDE_COND_MAX = 7

# Field effects
FIELD_NONE = 0
FIELD_GRAVITY = 1
FIELD_TRICK_ROOM = 2
FIELD_WONDER_ROOM = 3
FIELD_MAGIC_ROOM = 4
FIELD_MUD_SPORT = 5
FIELD_WATER_SPORT = 6
FIELD_UNKNOWN = 7
FIELD_MAX = 7

# Move category
CATEGORY_NONE = 0
CATEGORY_PHYSICAL = 1
CATEGORY_SPECIAL = 2
CATEGORY_STATUS = 3
CATEGORY_UNKNOWN = 4

# Effects (volatile battle effects on Pokémon)
EFFECT_NONE = 0
EFFECT_CONFUSION = 1
EFFECT_INFATUATION = 2
EFFECT_LEECH_SEED = 3
EFFECT_LOCK = 4
EFFECT_NIGHTMARE = 5
EFFECT_CURSE = 6
EFFECT_UNKNOWN = 7

# Action space: 9 actions (Gen1, MinimalActionSpace)
# 0-3: Move slots (sorted alphabetically as in Metamon)
# 4-8: Switch slots (sorted alphabetically as in Metamon)
NUM_ACTIONS = 9


# ============================================================================
# Tensor Layout Definitions
# ============================================================================

NUM_POKEMON_SLOTS = 13  # 1 player active + 5 switches + 1 opponent active + 6 revealed
NUM_MOVES_PER_POKEMON = 4

# Per-Pokémon categorical features (int IDs)
POKEMON_CAT_FEATURES = [
    "species",        # National Dex ID (0=unknown)
    "type_1",         # Type enum
    "type_2",         # Type enum
    "status",         # Status enum
    "effect",         # Volatile effect enum
    "move_1_id",      # Move ID (0=unknown)
    "move_2_id",
    "move_3_id",
    "move_4_id",
]
POKEMON_CAT_LEN = len(POKEMON_CAT_FEATURES)  # 9

# Per-Pokémon numerical features (normalized floats)
POKEMON_NUM_FEATURES = [
    "hp_fraction",       # current_hp / max_hp (0.0-1.0), -1.0 = unknown
    "level_norm",        # level / 100.0
    "base_atk_norm",     # base_atk / 255.0
    "base_spa_norm",     # base_spa / 255.0
    "base_def_norm",     # base_def / 255.0
    "base_spd_norm",     # base_spd / 255.0
    "base_spe_norm",     # base_spe / 255.0
    "base_hp_norm",      # base_hp / 255.0
    "atk_boost_norm",    # boost / 6.0 (-1.0 to 1.0)
    "spa_boost_norm",
    "def_boost_norm",
    "spd_boost_norm",
    "spe_boost_norm",
    "accuracy_boost_norm",
    "evasion_boost_norm",
    "move_1_bp_norm",    # base_power / 200.0
    "move_1_acc_norm",   # accuracy (0.0-1.0)
    "move_1_pri_norm",   # priority / 5.0
    "move_1_pp_norm",    # current_pp / max_pp (0.0-1.0), -1.0 = unknown
    "move_2_bp_norm",
    "move_2_acc_norm",
    "move_2_pri_norm",
    "move_2_pp_norm",
    "move_3_bp_norm",
    "move_3_acc_norm",
    "move_3_pri_norm",
    "move_3_pp_norm",
    "move_4_bp_norm",
    "move_4_acc_norm",
    "move_4_pri_norm",
    "move_4_pp_norm",
]
POKEMON_NUM_LEN = len(POKEMON_NUM_FEATURES)  # 31

# Per-Pokémon mask features
POKEMON_MASK_FEATURES = [
    "valid",            # 1 if this slot has a Pokémon, 0 if padding
    "fainted",          # 1 if fainted, 0 if alive
    "moves_revealed",   # 1 if moves are known, 0 if hidden
    "hp_known",         # 1 if HP is observable, 0 if unknown
]
POKEMON_MASK_LEN = len(POKEMON_MASK_FEATURES)  # 4

# Per-Pokémon move category features (categorical)
POKEMON_MOVE_CAT_FEATURES = [
    "move_1_category",
    "move_2_category",
    "move_3_category",
    "move_4_category",
]
POKEMON_MOVE_CAT_LEN = len(POKEMON_MOVE_CAT_FEATURES)  # 4

# Per-Pokémon move type features (categorical)
POKEMON_MOVE_TYPE_FEATURES = [
    "move_1_type",
    "move_2_type",
    "move_3_type",
    "move_4_type",
]
POKEMON_MOVE_TYPE_LEN = len(POKEMON_MOVE_TYPE_FEATURES)  # 4

# Global categorical features
GLOBAL_CAT_FEATURES = [
    "weather",           # Weather enum
    "field_effect",      # Field effect enum
    "player_side_cond",  # Side condition enum
    "opponent_side_cond",# Side condition enum
    "player_prev_move",  # Move ID (0=none/unknown)
    "opponent_prev_move",# Move ID (0=none/unknown)
]
GLOBAL_CAT_LEN = len(GLOBAL_CAT_FEATURES)  # 6

# Global numerical features
GLOBAL_NUM_FEATURES = [
    "turn_norm",           # turn / 200.0 (clipped to 1.0)
    "opponents_remaining", # / 6.0
    "forced_switch",       # 0.0 or 1.0
]
GLOBAL_NUM_LEN = len(GLOBAL_NUM_FEATURES)  # 3

# Total categorical and numerical dimensions for flat representations
TOTAL_CAT = GLOBAL_CAT_LEN + NUM_POKEMON_SLOTS * (POKEMON_CAT_LEN + POKEMON_MOVE_CAT_LEN + POKEMON_MOVE_TYPE_LEN)
TOTAL_NUM = GLOBAL_NUM_LEN + NUM_POKEMON_SLOTS * POKEMON_NUM_LEN
TOTAL_MASK = NUM_POKEMON_SLOTS * POKEMON_MASK_LEN + NUM_ACTIONS  # pokemon masks + legal action mask


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class PokemonFeatures:
    """Features for a single Pokémon slot."""
    species: int = SPECIES_UNKNOWN
    type_1: int = TYPE_NONE
    type_2: int = TYPE_NONE
    status: int = STATUS_NONE
    effect: int = EFFECT_NONE

    # Move IDs
    move_ids: List[int] = field(default_factory=lambda: [MOVE_UNKNOWN] * NUM_MOVES_PER_POKEMON)
    # Move categories
    move_categories: List[int] = field(default_factory=lambda: [CATEGORY_NONE] * NUM_MOVES_PER_POKEMON)
    # Move types
    move_types: List[int] = field(default_factory=lambda: [TYPE_NONE] * NUM_MOVES_PER_POKEMON)

    # Numerical
    hp_fraction: float = -1.0     # -1.0 = unknown
    level_norm: float = 0.0
    base_atk_norm: float = 0.0
    base_spa_norm: float = 0.0
    base_def_norm: float = 0.0
    base_spd_norm: float = 0.0
    base_spe_norm: float = 0.0
    base_hp_norm: float = 0.0
    boosts: List[float] = field(default_factory=lambda: [0.0] * 7)  # atk, spa, def, spd, spe, acc, eva
    move_bp: List[float] = field(default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON)
    move_acc: List[float] = field(default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON)
    move_pri: List[float] = field(default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON)
    move_pp: List[float] = field(default_factory=lambda: [-2.0] * NUM_MOVES_PER_POKEMON)

    # Masks
    valid: bool = False
    fainted: bool = False
    moves_revealed: bool = False
    hp_known: bool = False


@dataclass
class GlobalFeatures:
    """Global battle state features."""
    weather: int = WEATHER_NONE
    field_effect: int = FIELD_NONE
    player_side_cond: int = SIDE_COND_NONE
    opponent_side_cond: int = SIDE_COND_NONE
    player_prev_move: int = MOVE_NONE
    opponent_prev_move: int = MOVE_NONE
    turn_norm: float = 0.0
    opponents_remaining: float = 1.0
    forced_switch: float = 0.0
    # Gen 3 (schema v2) APPEND-ONLY additions. Default 0.0 so gen1 construction
    # sites and the gen1 to_tensors() layout (which ignores these) are unchanged.
    # Spikes layers on each side, 0-3 (normalized to [0,1] by /3 in the encoder).
    player_spikes_layers: float = 0.0
    opponent_spikes_layers: float = 0.0


@dataclass
class RomBattleState:
    """The canonical ROM-native battle state representation.

    This is a structured, fixed-size representation that can be produced
    from both Metamon's UniversalState and pokeemerald-expansion's
    battle engine state.
    """
    global_features: GlobalFeatures = field(default_factory=GlobalFeatures)
    pokemon: List[PokemonFeatures] = field(default_factory=lambda: [PokemonFeatures() for _ in range(NUM_POKEMON_SLOTS)])
    legal_action_mask: List[bool] = field(default_factory=lambda: [False] * NUM_ACTIONS)

    def to_tensors(self) -> dict:
        """Convert to contiguous numpy tensors suitable for model input."""
        # Global categorical
        global_cat = np.array([
            self.global_features.weather,
            self.global_features.field_effect,
            self.global_features.player_side_cond,
            self.global_features.opponent_side_cond,
            self.global_features.player_prev_move,
            self.global_features.opponent_prev_move,
        ], dtype=np.int32)

        # Global numerical
        global_num = np.array([
            self.global_features.turn_norm,
            self.global_features.opponents_remaining,
            self.global_features.forced_switch,
        ], dtype=np.float32)

        # Per-Pokémon features
        pokemon_cat = np.zeros((NUM_POKEMON_SLOTS, POKEMON_CAT_LEN), dtype=np.int32)
        pokemon_move_cat = np.zeros((NUM_POKEMON_SLOTS, POKEMON_MOVE_CAT_LEN), dtype=np.int32)
        pokemon_move_type = np.zeros((NUM_POKEMON_SLOTS, POKEMON_MOVE_TYPE_LEN), dtype=np.int32)
        pokemon_num = np.zeros((NUM_POKEMON_SLOTS, POKEMON_NUM_LEN), dtype=np.float32)
        pokemon_mask = np.zeros((NUM_POKEMON_SLOTS, POKEMON_MASK_LEN), dtype=np.int32)

        for i, p in enumerate(self.pokemon):
            pokemon_cat[i] = [
                p.species, p.type_1, p.type_2, p.status, p.effect,
                p.move_ids[0], p.move_ids[1], p.move_ids[2], p.move_ids[3],
            ]
            pokemon_move_cat[i] = p.move_categories
            pokemon_move_type[i] = p.move_types
            pokemon_num[i] = [
                p.hp_fraction, p.level_norm,
                p.base_atk_norm, p.base_spa_norm, p.base_def_norm,
                p.base_spd_norm, p.base_spe_norm, p.base_hp_norm,
                p.boosts[0], p.boosts[1], p.boosts[2], p.boosts[3],
                p.boosts[4], p.boosts[5], p.boosts[6],
                p.move_bp[0], p.move_acc[0], p.move_pri[0], p.move_pp[0],
                p.move_bp[1], p.move_acc[1], p.move_pri[1], p.move_pp[1],
                p.move_bp[2], p.move_acc[2], p.move_pri[2], p.move_pp[2],
                p.move_bp[3], p.move_acc[3], p.move_pri[3], p.move_pp[3],
            ]
            pokemon_mask[i] = [
                int(p.valid), int(p.fainted), int(p.moves_revealed), int(p.hp_known),
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

    def to_flat(self) -> dict:
        """Convert to flat contiguous tensors (all features concatenated)."""
        t = self.to_tensors()
        cat = np.concatenate([
            t["global_cat"],
            t["pokemon_cat"].flatten(),
            t["pokemon_move_cat"].flatten(),
            t["pokemon_move_type"].flatten(),
        ])
        num = np.concatenate([
            t["global_num"],
            t["pokemon_num"].flatten(),
        ])
        mask = np.concatenate([
            t["pokemon_mask"].flatten(),
            t["legal_action_mask"],
        ])
        return {
            "categorical": cat,
            "numerical": num,
            "masks": mask,
        }

    def to_json(self) -> dict:
        """Convert to a JSON-serializable dictionary for debugging/export."""
        return {
            "global": {
                "weather": self.global_features.weather,
                "field_effect": self.global_features.field_effect,
                "player_side_cond": self.global_features.player_side_cond,
                "opponent_side_cond": self.global_features.opponent_side_cond,
                "player_prev_move": self.global_features.player_prev_move,
                "opponent_prev_move": self.global_features.opponent_prev_move,
                "turn_norm": self.global_features.turn_norm,
                "opponents_remaining": self.global_features.opponents_remaining,
                "forced_switch": self.global_features.forced_switch,
            },
            "pokemon": [
                {
                    "species": p.species,
                    "type_1": p.type_1,
                    "type_2": p.type_2,
                    "status": p.status,
                    "effect": p.effect,
                    "move_ids": p.move_ids,
                    "move_categories": p.move_categories,
                    "move_types": p.move_types,
                    "hp_fraction": p.hp_fraction,
                    "level_norm": p.level_norm,
                    "base_stats": [
                        p.base_atk_norm, p.base_spa_norm, p.base_def_norm,
                        p.base_spd_norm, p.base_spe_norm, p.base_hp_norm,
                    ],
                    "boosts": p.boosts,
                    "move_bp": p.move_bp,
                    "move_acc": p.move_acc,
                    "move_pri": p.move_pri,
                    "move_pp": p.move_pp,
                    "valid": p.valid,
                    "fainted": p.fainted,
                    "moves_revealed": p.moves_revealed,
                    "hp_known": p.hp_known,
                }
                for p in self.pokemon
            ],
            "legal_action_mask": self.legal_action_mask,
        }


# Slot ordering convention:
SLOT_PLAYER_ACTIVE = 0
SLOT_SWITCH_0 = 1
SLOT_SWITCH_1 = 2
SLOT_SWITCH_2 = 3
SLOT_SWITCH_3 = 4
SLOT_SWITCH_4 = 5
SLOT_OPPONENT_ACTIVE = 6
SLOT_REVEALED_OPP_0 = 7
SLOT_REVEALED_OPP_1 = 8
SLOT_REVEALED_OPP_2 = 9
SLOT_REVEALED_OPP_3 = 10
SLOT_REVEALED_OPP_4 = 11
SLOT_REVEALED_OPP_5 = 12
