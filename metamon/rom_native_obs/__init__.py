"""
ROM-native observation representation for Metamon.

This module provides a compact, structured battle state representation
that can be produced from both Metamon's UniversalState and
pokeemerald-expansion's battle engine state, suitable for training
small distilled policies for eventual GBA deployment.
"""

from .schema import (
    RomBattleState,
    GlobalFeatures,
    PokemonFeatures,
    NUM_POKEMON_SLOTS,
    NUM_MOVES_PER_POKEMON,
    NUM_ACTIONS,
    POKEMON_CAT_LEN,
    POKEMON_NUM_LEN,
    POKEMON_MASK_LEN,
    POKEMON_MOVE_CAT_LEN,
    POKEMON_MOVE_TYPE_LEN,
    GLOBAL_CAT_LEN,
    GLOBAL_NUM_LEN,
    TOTAL_CAT,
    TOTAL_NUM,
    TOTAL_MASK,
)
from .mappings import (
    species_name_to_id,
    move_name_to_id,
    TYPE_NAME_TO_ID,
    STATUS_NAME_TO_ID,
    WEATHER_NAME_TO_ID,
    SIDE_COND_NAME_TO_ID,
    FIELD_NAME_TO_ID,
    EFFECT_NAME_TO_ID,
)
from .metamon_encoder import RomObservationEncoder
from .student_model import (
    RomStudentPolicy,
    RomStudentGRUPolicy,
    RomStudentEncoder,
    StudentConfig,
    preset_config,
    build_model,
)
