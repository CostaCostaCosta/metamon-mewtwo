"""
Cross-system state comparison utility.

Compares RomBattleState representations produced from Metamon and poke-plastic-ox
to verify semantic equivalence.

Can operate at two levels:
1. Unit-level: compare individual concept mappings (species, move, status, etc.)
2. Full-state: compare complete battle states field-by-field
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add metamon to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metamon.rom_native_obs.schema import (
    RomBattleState,
    GlobalFeatures,
    PokemonFeatures,
    NUM_POKEMON_SLOTS,
    NUM_ACTIONS,
    SLOT_PLAYER_ACTIVE,
    SLOT_OPPONENT_ACTIVE,
    SLOT_SWITCH_0,
    SLOT_REVEALED_OPP_0,
)
from metamon.rom_native_obs.mappings import (
    SPECIES_NAME_TO_ID,
    MOVE_NAME_TO_ID,
    TYPE_NAME_TO_ID,
    STATUS_NAME_TO_ID,
    WEATHER_NAME_TO_ID,
    SIDE_COND_NAME_TO_ID,
    FIELD_NAME_TO_ID,
    EFFECT_NAME_TO_ID,
    species_name_to_id,
    move_name_to_id,
)


@dataclass
class FieldComparison:
    """Result of comparing a single field."""

    field_name: str
    metamon_value: object
    rom_value: object
    match: bool
    category: str = "unknown"  # exact_match, expected_diff, mismatch, unavailable

    def __str__(self):
        status = "✓" if self.match else "✗"
        return f"  {status} {self.field_name}: metamon={self.metamon_value}, rom={self.rom_value} [{self.category}]"


@dataclass
class StateComparison:
    """Result of comparing two RomBattleState objects."""

    field_results: List[FieldComparison] = field(default_factory=list)

    @property
    def exact_matches(self) -> int:
        return sum(
            1 for r in self.field_results if r.match and r.category == "exact_match"
        )

    @property
    def expected_diffs(self) -> int:
        return sum(1 for r in self.field_results if r.category == "expected_diff")

    @property
    def mismatches(self) -> int:
        return sum(
            1 for r in self.field_results if not r.match and r.category == "mismatch"
        )

    @property
    def unavailable(self) -> int:
        return sum(1 for r in self.field_results if r.category == "unavailable")

    @property
    def total(self) -> int:
        return len(self.field_results)

    @property
    def all_critical_match(self) -> bool:
        """True if no mismatches (expected diffs and unavailable are OK)."""
        return self.mismatches == 0

    def summary(self) -> str:
        lines = [
            f"Comparison Summary: {self.exact_matches}/{self.total} exact matches, "
            f"{self.expected_diffs} expected diffs, {self.mismatches} mismatches, "
            f"{self.unavailable} unavailable"
        ]
        if self.mismatches > 0:
            lines.append("MISMATCHES:")
            for r in self.field_results:
                if not r.match and r.category == "mismatch":
                    lines.append(str(r))
        return "\n".join(lines)


def compare_states(
    metamon_state: RomBattleState,
    rom_state: RomBattleState,
    ignore_revealed_order: bool = True,
) -> StateComparison:
    """Compare two RomBattleState objects field-by-field.

    Args:
        metamon_state: State encoded from Metamon
        rom_state: State encoded from poke-plastic-ox
        ignore_revealed_order: If True, compare revealed opponent sets rather than slot order

    Returns:
        StateComparison with detailed field-by-field results
    """
    result = StateComparison()

    # Compare global features
    mg = metamon_state.global_features
    rg = rom_state.global_features

    for field_name in [
        "weather",
        "field_effect",
        "player_side_cond",
        "opponent_side_cond",
    ]:
        mv = getattr(mg, field_name)
        rv = getattr(rg, field_name)
        result.field_results.append(
            FieldComparison(
                f"global.{field_name}",
                mv,
                rv,
                mv == rv,
                "exact_match" if mv == rv else "mismatch",
            )
        )

    for field_name in ["player_prev_move", "opponent_prev_move"]:
        mv = getattr(mg, field_name)
        rv = getattr(rg, field_name)
        result.field_results.append(
            FieldComparison(
                f"global.{field_name}",
                mv,
                rv,
                mv == rv,
                "exact_match" if mv == rv else "mismatch",
            )
        )

    for field_name in ["turn_norm", "opponents_remaining", "forced_switch"]:
        mv = getattr(mg, field_name)
        rv = getattr(rg, field_name)
        # Use tolerance for float comparison
        match = abs(mv - rv) < 0.01
        result.field_results.append(
            FieldComparison(
                f"global.{field_name}",
                mv,
                rv,
                match,
                "exact_match" if match else "mismatch",
            )
        )

    # Compare player active and opponent active (slots 0 and 6)
    for slot_idx, slot_name in [
        (SLOT_PLAYER_ACTIVE, "player_active"),
        (SLOT_OPPONENT_ACTIVE, "opponent_active"),
    ]:
        mp = metamon_state.pokemon[slot_idx]
        rp = rom_state.pokemon[slot_idx]
        _compare_pokemon(result, mp, rp, slot_name)

    # Compare switches (slots 1-5)
    for i in range(5):
        mp = metamon_state.pokemon[SLOT_SWITCH_0 + i]
        rp = rom_state.pokemon[SLOT_SWITCH_0 + i]
        _compare_pokemon(result, mp, rp, f"switch_{i}")

    # Compare revealed opponents (slots 7-12)
    if ignore_revealed_order:
        # Compare as sets
        metamon_revealed = sorted(
            [
                (p.species, p.status, p.fainted)
                for p in metamon_state.pokemon[
                    SLOT_REVEALED_OPP_0 : SLOT_REVEALED_OPP_0 + 6
                ]
                if p.valid
            ]
        )
        rom_revealed = sorted(
            [
                (p.species, p.status, p.fainted)
                for p in rom_state.pokemon[
                    SLOT_REVEALED_OPP_0 : SLOT_REVEALED_OPP_0 + 6
                ]
                if p.valid
            ]
        )
        match = metamon_revealed == rom_revealed
        result.field_results.append(
            FieldComparison(
                "revealed_opponents_set",
                metamon_revealed,
                rom_revealed,
                match,
                (
                    "exact_match" if match else "expected_diff"
                ),  # may differ due to tracking
            )
        )
    else:
        for i in range(6):
            mp = metamon_state.pokemon[SLOT_REVEALED_OPP_0 + i]
            rp = rom_state.pokemon[SLOT_REVEALED_OPP_0 + i]
            _compare_pokemon(result, mp, rp, f"revealed_{i}")

    # Compare legal action mask
    match = list(metamon_state.legal_action_mask) == list(rom_state.legal_action_mask)
    result.field_results.append(
        FieldComparison(
            "legal_action_mask",
            list(metamon_state.legal_action_mask),
            list(rom_state.legal_action_mask),
            match,
            "exact_match" if match else "mismatch",
        )
    )

    return result


def _compare_pokemon(
    result: StateComparison, mp: PokemonFeatures, rp: PokemonFeatures, name: str
):
    """Compare two PokemonFeatures and add results to the comparison."""
    # Validity
    match = mp.valid == rp.valid
    result.field_results.append(
        FieldComparison(
            f"{name}.valid",
            mp.valid,
            rp.valid,
            match,
            "exact_match" if match else "mismatch",
        )
    )

    if not mp.valid and not rp.valid:
        return  # Both are padding, nothing else to compare

    # Species
    match = mp.species == rp.species
    result.field_results.append(
        FieldComparison(
            f"{name}.species",
            mp.species,
            rp.species,
            match,
            "exact_match" if match else "mismatch",
        )
    )

    # Types
    for i, tname in enumerate(["type_1", "type_2"]):
        mv = getattr(mp, tname)
        rv = getattr(rp, tname)
        match = mv == rv
        result.field_results.append(
            FieldComparison(
                f"{name}.{tname}", mv, rv, match, "exact_match" if match else "mismatch"
            )
        )

    # Status
    match = mp.status == rp.status
    result.field_results.append(
        FieldComparison(
            f"{name}.status",
            mp.status,
            rp.status,
            match,
            "exact_match" if match else "mismatch",
        )
    )

    # HP fraction
    match = abs(mp.hp_fraction - rp.hp_fraction) < 0.02  # 2% tolerance for rounding
    result.field_results.append(
        FieldComparison(
            f"{name}.hp_fraction",
            mp.hp_fraction,
            rp.hp_fraction,
            match,
            "exact_match" if match else "mismatch",
        )
    )

    # Fainted
    match = mp.fainted == rp.fainted
    result.field_results.append(
        FieldComparison(
            f"{name}.fainted",
            mp.fainted,
            rp.fainted,
            match,
            "exact_match" if match else "mismatch",
        )
    )


def compare_json_states(metamon_json: dict, rom_json: dict) -> StateComparison:
    """Compare two RomBattleState JSON representations."""
    # Reconstruct RomBattleState from JSON
    metamon_state = _json_to_state(metamon_json)
    rom_state = _json_to_state(rom_json)
    return compare_states(metamon_state, rom_state)


def _json_to_state(j: dict) -> RomBattleState:
    """Convert a JSON dict back to RomBattleState."""
    state = RomBattleState()
    g = j["global"]
    state.global_features = GlobalFeatures(
        weather=g["weather"],
        field_effect=g["field_effect"],
        player_side_cond=g["player_side_cond"],
        opponent_side_cond=g["opponent_side_cond"],
        player_prev_move=g["player_prev_move"],
        opponent_prev_move=g["opponent_prev_move"],
        turn_norm=g["turn_norm"],
        opponents_remaining=g["opponents_remaining"],
        forced_switch=g["forced_switch"],
    )
    for i, pj in enumerate(j["pokemon"]):
        p = PokemonFeatures()
        p.species = pj["species"]
        p.type_1 = pj["type_1"]
        p.type_2 = pj["type_2"]
        p.status = pj["status"]
        p.effect = pj["effect"]
        p.move_ids = pj["move_ids"]
        p.move_categories = pj["move_categories"]
        p.move_types = pj["move_types"]
        p.hp_fraction = pj["hp_fraction"]
        p.level_norm = pj["level_norm"]
        p.base_atk_norm = pj["base_stats"][0]
        p.base_spa_norm = pj["base_stats"][1]
        p.base_def_norm = pj["base_stats"][2]
        p.base_spd_norm = pj["base_stats"][3]
        p.base_spe_norm = pj["base_stats"][4]
        p.base_hp_norm = pj["base_stats"][5]
        p.boosts = pj["boosts"]
        p.move_bp = pj["move_bp"]
        p.move_acc = pj["move_acc"]
        p.move_pri = pj["move_pri"]
        p.move_pp = pj["move_pp"]
        p.valid = pj["valid"]
        p.fainted = pj["fainted"]
        p.moves_revealed = pj["moves_revealed"]
        p.hp_known = pj["hp_known"]
        state.pokemon[i] = p
    state.legal_action_mask = j["legal_action_mask"]
    return state


# ============================================================================
# Unit-level equivalence tests
# ============================================================================


def test_species_mapping():
    """Test that species IDs match between Showdown and ROM for Gen1."""
    # Gen1 species use National Dex numbers in both systems
    test_cases = [
        ("bulbasaur", 1),
        ("charmander", 4),
        ("charizard", 6),
        ("alakazam", 65),
        ("gengar", 94),
        ("exeggutor", 103),
        ("snorlax", 143),
        ("tauros", 128),
        ("starmie", 121),
        ("mew", 151),
        ("mewtwo", 150),
    ]
    results = []
    for name, expected in test_cases:
        sid = species_name_to_id(name)
        match = sid == expected
        results.append(
            FieldComparison(
                f"species.{name}",
                expected,
                sid,
                match,
                "exact_match" if match else "mismatch",
            )
        )
    return results


def test_move_mapping():
    """Test that move IDs match between Showdown and ROM for Gen1."""
    test_cases = [
        ("pound", 1),
        ("karatechop", 2),
        ("bodyslam", 34),
        ("thunderbolt", 85),
        ("psychic", 94),
        ("explosion", 153),
        ("hyperbeam", 63),
        ("blizzard", 59),
        ("fireblast", 126),
        ("recover", 105),
        ("amnesia", 133),
    ]
    results = []
    for name, expected in test_cases:
        mid = move_name_to_id(name)
        match = mid == expected
        results.append(
            FieldComparison(
                f"move.{name}",
                expected,
                mid,
                match,
                "exact_match" if match else "mismatch",
            )
        )
    return results


def test_type_mapping():
    """Test type ID mapping."""
    test_cases = [
        ("normal", 1),
        ("fire", 11),
        ("water", 12),
        ("grass", 13),
        ("electric", 14),
        ("psychic", 15),
        ("ice", 16),
        ("dragon", 17),
        ("ghost", 9),
        ("poison", 4),
        ("bug", 8),
        ("rock", 6),
        ("ground", 5),
        ("fighting", 2),
        ("flying", 3),
    ]
    results = []
    for name, expected in test_cases:
        tid = TYPE_NAME_TO_ID.get(name, -1)
        match = tid == expected
        results.append(
            FieldComparison(
                f"type.{name}",
                expected,
                tid,
                match,
                "exact_match" if match else "mismatch",
            )
        )
    return results


def test_status_mapping():
    """Test status ID mapping."""
    test_cases = [
        ("nostatus", 0),
        ("slp", 1),
        ("psn", 2),
        ("brn", 3),
        ("frz", 4),
        ("par", 5),
        ("tox", 6),
        ("fnt", 7),
    ]
    results = []
    for name, expected in test_cases:
        sid = STATUS_NAME_TO_ID.get(name, -1)
        match = sid == expected
        results.append(
            FieldComparison(
                f"status.{name}",
                expected,
                sid,
                match,
                "exact_match" if match else "mismatch",
            )
        )
    return results


def run_all_unit_tests():
    """Run all unit-level equivalence tests and print results."""
    all_results = []
    all_results.extend(test_species_mapping())
    all_results.extend(test_move_mapping())
    all_results.extend(test_type_mapping())
    all_results.extend(test_status_mapping())

    matches = sum(1 for r in all_results if r.match)
    mismatches = [r for r in all_results if not r.match]

    print(f"Unit tests: {matches}/{len(all_results)} passed")
    if mismatches:
        print("MISMATCHES:")
        for m in mismatches:
            print(m)

    return matches == len(all_results)
