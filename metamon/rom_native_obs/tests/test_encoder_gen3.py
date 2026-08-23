"""
Tests for the Gen 3 ROM-native observation encoder (schema v2).

Covers: tensor shapes/dtypes, gen3 vocab ID ranges, item/ability encoding,
information-visibility (opponent item/ability/moves hidden until revealed),
spikes side-condition mapping, forced-switch legal mask, reveal memory across
encode() calls, and reset().
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metamon.rom_native_obs.metamon_encoder_gen3 import Gen3RomObservationEncoder
from metamon.rom_native_obs.schema_gen3 import (
    Gen3RomBattleState,
    NUM_POKEMON_SLOTS,
    NUM_ACTIONS,
    POKEMON_CAT_LEN,
    POKEMON_MASK_LEN,
    POKEMON_NUM_LEN,
    SLOT_PLAYER_ACTIVE,
    SLOT_OPPONENT_ACTIVE,
    SLOT_SWITCH_0,
    SPECIES_MAX_GEN3,
    MOVE_MAX_GEN3,
    ABILITY_MAX_GEN3,
    ITEM_MAX_GEN3,
    SIDE_COND_SPIKES,
)
from metamon.rom_native_obs import mappings_gen3 as mg
from metamon.interface import UniversalState, UniversalPokemon, UniversalMove


def _mv(name="thunderbolt", t="electric", cat="special", bp=95):
    return UniversalMove(
        name=name,
        move_type=t,
        category=cat,
        base_power=bp,
        accuracy=1.0,
        priority=0,
        current_pp=24,
        max_pp=24,
    )


def _mon(
    name="salamence",
    types="dragon flying",
    item="leftovers",
    ability="intimidate",
    hp=1.0,
    status="nostatus",
    moves=None,
    lvl=100,
):
    if moves is None:
        moves = [_mv()]
    return UniversalPokemon(
        name=name,
        hp_pct=hp,
        types=types,
        item=item,
        ability=ability,
        lvl=lvl,
        status=status,
        effect="noeffect",
        moves=moves,
        atk_boost=0,
        spa_boost=0,
        def_boost=0,
        spd_boost=0,
        spe_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        base_atk=135,
        base_spa=110,
        base_def=80,
        base_spd=80,
        base_spe=100,
        base_hp=95,
        hp_stat=-1,
        atk_stat=-1,
        def_stat=-1,
        spa_stat=-1,
        spd_stat=-1,
        spe_stat=-1,
        tera_type="notype",
        base_species=name,
    )


def _state(
    opp_item="leftovers",
    opp_ability="intimidate",
    opp_moves=None,
    player_conditions="noconditions",
    forced_switch=False,
    nswitch=2,
):
    player = _mon(
        "metagross", types="steel psychic", item="choiceband", ability="clearbody"
    )
    opponent = _mon("tyranitar", types="rock dark", item=opp_item, ability=opp_ability)
    opponent.moves = [] if opp_moves is None else opp_moves
    switches = [
        _mon("skarmory", types="steel flying"),
        _mon("blissey", types="normal"),
    ][:nswitch]
    return UniversalState(
        format="gen3ou",
        player_active_pokemon=player,
        opponent_active_pokemon=opponent,
        available_switches=switches,
        player_prev_move=UniversalMove.blank_move(),
        opponent_prev_move=UniversalMove.blank_move(),
        opponents_remaining=6,
        player_conditions=player_conditions,
        opponent_conditions="noconditions",
        weather="noweather",
        battle_field="nofield",
        forced_switch=forced_switch,
        battle_won=False,
        battle_lost=False,
        can_tera=False,
        opponent_teampreview=[],
    )


class TestGen3Schema(unittest.TestCase):
    def test_tensor_shapes(self):
        t = Gen3RomBattleState().to_tensors()
        self.assertEqual(t["global_cat"].shape, (6,))
        self.assertEqual(t["pokemon_cat"].shape, (NUM_POKEMON_SLOTS, 11))
        self.assertEqual(t["pokemon_mask"].shape, (NUM_POKEMON_SLOTS, 6))
        self.assertEqual(t["pokemon_num"].shape, (NUM_POKEMON_SLOTS, 31))
        self.assertEqual(t["pokemon_move_cat"].shape, (NUM_POKEMON_SLOTS, 4))
        self.assertEqual(t["legal_action_mask"].shape, (NUM_ACTIONS,))

    def test_dtypes(self):
        t = Gen3RomBattleState().to_tensors()
        import numpy as np

        self.assertEqual(t["pokemon_cat"].dtype, np.int32)
        self.assertEqual(t["pokemon_num"].dtype, np.float32)
        self.assertEqual(t["pokemon_mask"].dtype, np.int32)


class TestGen3Mappings(unittest.TestCase):
    def test_species_bounds(self):
        self.assertEqual(mg.species_name_to_id("Rayquaza"), 384)
        self.assertEqual(mg.species_name_to_id("Skitty"), 300)  # no gen3 gap
        self.assertEqual(mg.species_name_to_id("Deoxys"), 386)
        self.assertEqual(mg.species_name_to_id("Turtwig"), 0)  # gen4 excluded
        self.assertTrue(
            all(1 <= v <= SPECIES_MAX_GEN3 for v in mg.SPECIES_NAME_TO_ID.values())
        )

    def test_move_bounds(self):
        self.assertEqual(mg.move_name_to_id("Psycho Boost"), 354)
        self.assertEqual(mg.move_name_to_id("Pound"), 1)
        self.assertEqual(mg.move_name_to_id("Roost"), 0)  # gen4 excluded
        self.assertTrue(
            all(1 <= v <= MOVE_MAX_GEN3 for v in mg.MOVE_NAME_TO_ID.values())
        )

    def test_ability_rom_canonical(self):
        # lightningrod is ROM-canonical 31 (Showdown num 32) -- the lone swap.
        self.assertEqual(mg.ability_name_to_id("Lightning Rod"), 31)
        self.assertEqual(mg.ability_name_to_id("Air Lock"), 76)
        self.assertEqual(mg.ability_name_to_id("unknownability"), 0)
        self.assertTrue(
            all(1 <= v <= ABILITY_MAX_GEN3 for v in mg.ABILITY_NAME_TO_ID.values())
        )

    def test_item_expansion_enum(self):
        self.assertEqual(mg.item_name_to_id("Leftovers"), 472)
        self.assertEqual(mg.item_name_to_id("Choice Band"), 442)
        self.assertEqual(mg.item_name_to_id("unknownitem"), 0)
        self.assertTrue(
            all(1 <= v <= ITEM_MAX_GEN3 for v in mg.ITEM_NAME_TO_ID.values())
        )

    def test_spikes_side_cond(self):
        self.assertEqual(mg.GEN3_SIDE_COND_NAME_TO_ID["spikes"], SIDE_COND_SPIKES)


class TestGen3Encoder(unittest.TestCase):
    def test_player_full_info(self):
        enc = Gen3RomObservationEncoder()
        t = enc.encode(_state()).to_tensors()
        pa = t["pokemon_cat"][SLOT_PLAYER_ACTIVE]
        self.assertEqual(pa[9], 442)  # choiceband
        self.assertEqual(pa[10], mg.ability_name_to_id("clearbody"))
        # valid, moves_revealed, hp_known, item_revealed, ability_revealed = 1; fainted = 0
        m = t["pokemon_mask"][SLOT_PLAYER_ACTIVE]
        self.assertEqual(list(m), [1, 0, 1, 1, 1, 1])

    def test_bench_full_info(self):
        enc = Gen3RomObservationEncoder()
        t = enc.encode(_state()).to_tensors()
        sw = t["pokemon_mask"][SLOT_SWITCH_0]
        self.assertEqual(list(sw), [1, 0, 1, 1, 1, 1])  # player bench full info

    def test_opp_item_ability_revealed_in_full_obs_replay(self):
        # Parsed replays are full-observability, so opponent item/ability surface.
        enc = Gen3RomObservationEncoder()
        t = enc.encode(
            _state(opp_item="leftovers", opp_ability="intimidate")
        ).to_tensors()
        oa = t["pokemon_cat"][SLOT_OPPONENT_ACTIVE]
        self.assertEqual(oa[9], 472)  # leftovers
        self.assertEqual(oa[10], mg.ability_name_to_id("intimidate"))
        self.assertEqual(t["pokemon_mask"][SLOT_OPPONENT_ACTIVE][4], 1)  # item_revealed
        self.assertEqual(
            t["pokemon_mask"][SLOT_OPPONENT_ACTIVE][5], 1
        )  # ability_revealed

    def test_opp_hidden_then_revealed_memory(self):
        enc = Gen3RomObservationEncoder()
        # first: opponent item/ability unknown
        t1 = enc.encode(
            _state(opp_item="unknownitem", opp_ability="unknownability")
        ).to_tensors()
        self.assertEqual(t1["pokemon_cat"][SLOT_OPPONENT_ACTIVE][9], 0)
        self.assertEqual(t1["pokemon_cat"][SLOT_OPPONENT_ACTIVE][10], 0)
        self.assertEqual(t1["pokemon_mask"][SLOT_OPPONENT_ACTIVE][4], 0)
        # then: item revealed -> persists in memory
        t2 = enc.encode(
            _state(opp_item="leftovers", opp_ability="unknownability")
        ).to_tensors()
        self.assertEqual(t2["pokemon_cat"][SLOT_OPPONENT_ACTIVE][9], 472)
        # back to unknown display, but memory retains the reveal
        t3 = enc.encode(
            _state(opp_item="unknownitem", opp_ability="unknownability")
        ).to_tensors()
        self.assertEqual(t3["pokemon_mask"][SLOT_OPPONENT_ACTIVE][4], 1)

    def test_opp_moves_hidden_until_revealed(self):
        enc = Gen3RomObservationEncoder()
        t1 = enc.encode(_state(opp_moves=[])).to_tensors()
        self.assertEqual(
            t1["pokemon_mask"][SLOT_OPPONENT_ACTIVE][2], 0
        )  # moves_revealed=0
        t2 = enc.encode(
            _state(opp_moves=[_mv("crunch", "dark", "physical", 80)])
        ).to_tensors()
        self.assertEqual(t2["pokemon_mask"][SLOT_OPPONENT_ACTIVE][2], 1)
        self.assertGreater(
            t2["pokemon_cat"][SLOT_OPPONENT_ACTIVE][5], 0
        )  # move id present

    def test_spikes_mapping(self):
        enc = Gen3RomObservationEncoder()
        t = enc.encode(_state(player_conditions="spikes")).to_tensors()
        self.assertEqual(t["global_cat"][2], SIDE_COND_SPIKES)

    def test_forced_switch_legal_mask(self):
        enc = Gen3RomObservationEncoder()
        t = enc.encode(_state(forced_switch=True)).to_tensors()
        # no move actions legal when forced to switch
        self.assertTrue(all(t["legal_action_mask"][:4] == 0))
        self.assertTrue(any(t["legal_action_mask"][4:] == 1))

    def test_reset_clears_memory(self):
        enc = Gen3RomObservationEncoder()
        enc.encode(_state(opp_item="leftovers"))
        self.assertTrue(enc.revealed_items)
        enc.reset()
        self.assertFalse(enc.revealed_items)
        self.assertFalse(enc.revealed_opponents)


if __name__ == "__main__":
    unittest.main()
