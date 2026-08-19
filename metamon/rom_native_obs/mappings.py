"""
Stable integer mappings between Metamon/Showdown names and canonical IDs.

For Gen1, National Dex species IDs and Showdown move numbers happen to match
the pokeemerald-expansion enum values. This file builds explicit translation
tables from Metamon's string-based names to the canonical integer IDs.

All mappings are deterministic and generated from the Showdown data files
where possible, with hand-maintained entries for status, weather, etc.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

from metamon.backend.replay_parser.str_parsing import pokemon_name, move_name, clean_name, clean_no_numbers


# ============================================================================
# Type name -> canonical type ID (matches pokeemerald-expansion Type enum)
# ============================================================================

TYPE_NAME_TO_ID: Dict[str, int] = {
    "none": 0,
    "normal": 1,
    "fighting": 2,
    "flying": 3,
    "poison": 4,
    "ground": 5,
    "rock": 6,
    "bird": 7,
    "bug": 8,
    "ghost": 9,
    "steel": 10,
    "fire": 11,
    "water": 12,
    "grass": 13,
    "electric": 14,
    "psychic": 15,
    "ice": 16,
    "dragon": 17,
    "dark": 18,
    "fairy": 19,
    "???": 0,
    "typeless": 0,
}

TYPE_ID_TO_NAME: Dict[int, str] = {v: k for k, v in TYPE_NAME_TO_ID.items()}


# ============================================================================
# Status name -> canonical status ID
# ============================================================================

STATUS_NAME_TO_ID: Dict[str, int] = {
    "nostatus": 0,
    "none": 0,
    "no": 0,
    "slp": 1,
    "sleep": 1,
    "psn": 2,
    "poison": 2,
    "brn": 3,
    "burn": 3,
    "frz": 4,
    "freeze": 4,
    "frozen": 4,
    "par": 5,
    "paralysis": 5,
    "paralyzed": 5,
    "tox": 6,
    "toxic": 6,
    "fnt": 7,
    "faint": 7,
    "fainted": 7,
}

STATUS_ID_TO_NAME: Dict[int, str] = {
    0: "nostatus",
    1: "slp",
    2: "psn",
    3: "brn",
    4: "frz",
    5: "par",
    6: "tox",
    7: "fnt",
    8: "unknown",
}


# ============================================================================
# Weather name -> canonical weather ID
# ============================================================================

WEATHER_NAME_TO_ID: Dict[str, int] = {
    "noweather": 0,
    "none": 0,
    "raindance": 1,
    "rain": 1,
    "sunnyday": 2,
    "sun": 2,
    "sandstorm": 3,
    "hail": 4,
    "snow": 5,
    "fog": 6,
    "deltastream": 0,  # strong winds - treat as none for Gen1
}

WEATHER_ID_TO_NAME: Dict[int, str] = {
    0: "noweather",
    1: "rain",
    2: "sun",
    3: "sandstorm",
    4: "hail",
    5: "snow",
    6: "fog",
    7: "unknown",
}


# ============================================================================
# Side condition name -> canonical ID
# ============================================================================

SIDE_COND_NAME_TO_ID: Dict[str, int] = {
    "noconditions": 0,
    "none": 0,
    "reflect": 1,
    "lightscreen": 2,
    "safeguard": 3,
    "mist": 4,
    "tailwind": 5,
    "auroraveil": 6,
}

SIDE_COND_ID_TO_NAME: Dict[int, str] = {
    0: "noconditions",
    1: "reflect",
    2: "lightscreen",
    3: "safeguard",
    4: "mist",
    5: "tailwind",
    6: "auroraveil",
    7: "unknown",
}


# ============================================================================
# Field effect name -> canonical ID
# ============================================================================

FIELD_NAME_TO_ID: Dict[str, int] = {
    "nofield": 0,
    "none": 0,
    "gravity": 1,
    "trickroom": 2,
    "wonderroom": 3,
    "magicroom": 4,
    "mudsport": 5,
    "watersport": 6,
}

FIELD_ID_TO_NAME: Dict[int, str] = {
    0: "nofield",
    1: "gravity",
    2: "trickroom",
    3: "wonderroom",
    4: "magicroom",
    5: "mudsport",
    6: "watersport",
    7: "unknown",
}


# ============================================================================
# Effect (volatile) name -> canonical ID
# ============================================================================

EFFECT_NAME_TO_ID: Dict[str, int] = {
    "noeffect": 0,
    "none": 0,
    "confusion": 1,
    "infatuation": 2,
    "leechseed": 3,
    "lock": 4,
    "nightmare": 5,
    "curse": 6,
    # Gen1-specific volatiles that affect decisions
    "flinch": 1,  # map to confusion-like (temporary incapacitation)
    "thrash": 4,  # locked into move
    "petaldance": 4,
    "outrage": 4,
}

EFFECT_ID_TO_NAME: Dict[int, str] = {
    0: "noeffect",
    1: "confusion",
    2: "infatuation",
    3: "leechseed",
    4: "lock",
    5: "nightmare",
    6: "curse",
    7: "unknown",
}


# ============================================================================
# Move category name -> canonical ID
# ============================================================================

CATEGORY_NAME_TO_ID: Dict[str, int] = {
    "nomove": 0,
    "none": 0,
    "physical": 1,
    "special": 2,
    "status": 3,
}


# ============================================================================
# Species name -> National Dex ID
# ============================================================================

# Build from the Showdown Gen1 pokedex
_STATIC_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "backend", "showdown_dex", "static")

def _build_species_map() -> Dict[str, int]:
    """Build species name -> National Dex ID mapping from Showdown data."""
    with open(os.path.join(_STATIC_ROOT, "pokemon", "gen1pokedex.json")) as f:
        pokedex = json.load(f)

    mapping = {}
    for key, val in pokedex.items():
        num = val.get("num", 0)
        if num > 0:
            # Map the cleaned name
            mapping[pokemon_name(val.get("name", key))] = num
            # Also map the key itself
            mapping[pokemon_name(key)] = num
            # Map base species
            if "baseSpecies" in val:
                mapping[pokemon_name(val["baseSpecies"])] = num
    return mapping

SPECIES_NAME_TO_ID = _build_species_map()


def species_name_to_id(name: str) -> int:
    """Convert a Metamon/Showdown species name to a canonical species ID."""
    if not name or name == "unknown":
        return 0
    cleaned = pokemon_name(name)
    return SPECIES_NAME_TO_ID.get(cleaned, 0)


# ============================================================================
# Move name -> Showdown move number
# ============================================================================

def _build_move_map() -> Dict[str, int]:
    """Build move name -> move number mapping from Showdown data."""
    with open(os.path.join(_STATIC_ROOT, "moves", "gen1moves.json")) as f:
        moves = json.load(f)

    mapping = {}
    for key, val in moves.items():
        num = val.get("num", 0)
        if num > 0:
            # Map the cleaned name (using move_name which handles hidden power etc.)
            mapping[move_name(val.get("name", key))] = num
            # Also map the key
            mapping[move_name(key)] = num
    return mapping

MOVE_NAME_TO_ID = _build_move_map()


def move_name_to_id(name: str) -> int:
    """Convert a Metamon/Showdown move name to a canonical move ID."""
    if not name or name == "nomove" or name == "unknown":
        return 0
    cleaned = move_name(name)
    return MOVE_NAME_TO_ID.get(cleaned, 0)


# ============================================================================
# Reverse mappings (ID -> name) for debugging
# ============================================================================

SPECIES_ID_TO_NAME: Dict[int, str] = {}
for name, id in SPECIES_NAME_TO_ID.items():
    if id not in SPECIES_ID_TO_NAME:
        SPECIES_ID_TO_NAME[id] = name

MOVE_ID_TO_NAME: Dict[int, str] = {}
for name, id in MOVE_NAME_TO_ID.items():
    if id not in MOVE_ID_TO_NAME:
        MOVE_ID_TO_NAME[id] = name
