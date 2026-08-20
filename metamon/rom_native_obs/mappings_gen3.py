"""
Gen 3 integer mappings between Metamon/Showdown names and canonical ROM IDs.

Canonical ID spaces (see docs/gen3_regi_plan.md SS3c and ROM_NATIVE_OBSERVATION.md
"Gen3 schema v2"):

- Species : National Dex number, 1-386 (matches pokeemerald-expansion enum; this
  fork has NO gen3 gap -- Skitty=300, Rayquaza=384, Deoxys=386 on both sides).
- Moves   : Showdown move num, 1-354 (== expansion enum; MOVE_PSYCHO_BOOST=354).
- Abilities: pokeemerald-expansion ABILITY_* enum, 1-76. NOTE the lone exception:
  Showdown assigns lightningrod num=32, but the expansion enum is
  ABILITY_LIGHTNING_ROD=31 (serenegrace=32 both sides). The canonical space is the
  ROM enum, so lightningrod -> 31 here.
- Items   : pokeemerald-expansion ITEM_* enum values for the gen3-legal set
  (96 held items). The expansion reorders items, so these are NOT Showdown nums.

Types / status / weather / side-conditions / field / effect / category enums are
inherited unchanged from the gen1 `mappings.py` (imported, not duplicated); gen3
adds SPIKES (8) to the side-condition enum (see `GEN3_SIDE_COND_NAME_TO_ID`).
"""
from __future__ import annotations

import json
import os
from typing import Dict

from metamon.backend.replay_parser.str_parsing import pokemon_name, move_name, clean_name

# Reuse the gen-agnostic categorical tables from the gen1 mappings module.
from .mappings import (  # noqa: F401
    TYPE_NAME_TO_ID,
    STATUS_NAME_TO_ID,
    WEATHER_NAME_TO_ID,
    FIELD_NAME_TO_ID,
    EFFECT_NAME_TO_ID,
    CATEGORY_NAME_TO_ID,
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "gen3_static")


def _load(name: str) -> Dict[str, int]:
    with open(os.path.join(_STATIC_DIR, name)) as f:
        return json.load(f)


# Species / moves / abilities / items (clean lowercase no-space name -> ID)
SPECIES_NAME_TO_ID: Dict[str, int] = _load("gen3species.json")
MOVE_NAME_TO_ID: Dict[str, int] = _load("gen3moves.json")
ABILITY_NAME_TO_ID: Dict[str, int] = _load("gen3abilities.json")
ITEM_NAME_TO_ID: Dict[str, int] = _load("gen3items.json")

# Gen3 side-condition enum = gen1 table + SPIKES=8 (single-enum stays lossy for
# concurrent screens -- documented known mismatch in ROM_NATIVE_OBSERVATION.md).
def _build_gen3_side_cond() -> Dict[str, int]:
    from .mappings import SIDE_COND_NAME_TO_ID as base
    t = dict(base)
    t["spikes"] = 8
    return t


GEN3_SIDE_COND_NAME_TO_ID = _build_gen3_side_cond()


def species_name_to_id(name: str) -> int:
    if not name or name == "unknown":
        return 0
    return SPECIES_NAME_TO_ID.get(pokemon_name(name), 0)


def move_name_to_id(name: str) -> int:
    if not name or name in ("nomove", "unknown"):
        return 0
    return MOVE_NAME_TO_ID.get(move_name(name), 0)


def ability_name_to_id(name: str) -> int:
    if not name or name in ("unknown", "unknownability", "noability", "none"):
        return 0
    return ABILITY_NAME_TO_ID.get(clean_name(name), 0)


def item_name_to_id(name: str) -> int:
    if not name or name in ("unknown", "unknownitem", "noitem", "none"):
        return 0
    return ITEM_NAME_TO_ID.get(clean_name(name), 0)


# Reverse maps for debugging
SPECIES_ID_TO_NAME: Dict[int, str] = {}
for _n, _i in SPECIES_NAME_TO_ID.items():
    SPECIES_ID_TO_NAME.setdefault(_i, _n)
MOVE_ID_TO_NAME: Dict[int, str] = {}
for _n, _i in MOVE_NAME_TO_ID.items():
    MOVE_ID_TO_NAME.setdefault(_i, _n)
ABILITY_ID_TO_NAME: Dict[int, str] = {}
for _n, _i in ABILITY_NAME_TO_ID.items():
    ABILITY_ID_TO_NAME.setdefault(_i, _n)
ITEM_ID_TO_NAME: Dict[int, str] = {}
for _n, _i in ITEM_NAME_TO_ID.items():
    ITEM_ID_TO_NAME.setdefault(_i, _n)
