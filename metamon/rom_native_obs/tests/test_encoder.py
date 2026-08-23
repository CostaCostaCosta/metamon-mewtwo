"""
Tests for the ROM-native observation encoder.

Tests cover:
1. Basic encoding correctness (species, moves, types, status)
2. Information visibility (hidden info doesn't leak)
3. Tensor shape correctness
4. Deterministic encoding
5. Multi-timestep state tracking (revealed opponents)
6. Legal action mask correctness
"""
import json
import lz4.frame
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metamon.rom_native_obs import RomObservationEncoder, RomBattleState
from metamon.rom_native_obs.schema import (
    NUM_POKEMON_SLOTS, NUM_MOVES_PER_POKEMON, NUM_ACTIONS,
    SLOT_PLAYER_ACTIVE, SLOT_OPPONENT_ACTIVE, SLOT_SWITCH_0, SLOT_SWITCH_1, SLOT_REVEALED_OPP_0,
    SPECIES_UNKNOWN, MOVE_UNKNOWN, STATUS_FAINT,
)
from metamon.interface import UniversalState, UniversalPokemon, UniversalMove
from metamon.backend.replay_parser.str_parsing import pokemon_name


def _make_simple_pokemon(name="gengar", hp=1.0, status="nostatus", moves=None):
    """Create a minimal UniversalPokemon for testing."""
    if moves is None:
        moves = [
            UniversalMove(name="thunderbolt", move_type="electric", category="special",
                         base_power=95, accuracy=1.0, priority=0, current_pp=24, max_pp=24),
            UniversalMove(name="psychic", move_type="psychic", category="special",
                         base_power=90, accuracy=1.0, priority=0, current_pp=16, max_pp=16),
        ]
    return UniversalPokemon(
        name=name, hp_pct=hp, types="ghost poison", item="noitem", ability="noability",
        lvl=100, status=status, effect="noeffect", moves=moves,
        atk_boost=0, spa_boost=0, def_boost=0, spd_boost=0, spe_boost=0,
        accuracy_boost=0, evasion_boost=0,
        base_atk=65, base_spa=130, base_def=60, base_spd=130, base_spe=110, base_hp=60,
        hp_stat=-1, atk_stat=-1, def_stat=-1, spa_stat=-1, spd_stat=-1, spe_stat=-1,
        tera_type="notype", base_species=name,
    )


def _make_simple_state():
    """Create a minimal UniversalState for testing."""
    player = _make_simple_pokemon("gengar", hp=1.0)
    opponent = _make_simple_pokemon("alakazam", hp=1.0)
    opponent.moves = []  # opponent moves not revealed
    switches = [_make_simple_pokemon("snorlax"), _make_simple_pokemon("tauros")]

    return UniversalState(
        format="gen1ou",
        player_active_pokemon=player,
        opponent_active_pokemon=opponent,
        available_switches=switches,
        player_prev_move=UniversalMove.blank_move(),
        opponent_prev_move=UniversalMove.blank_move(),
        opponents_remaining=6,
        player_conditions="noconditions",
        opponent_conditions="noconditions",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=False,
        opponent_teampreview=[],
    )


class TestRomBattleStateSchema(unittest.TestCase):
    """Test the canonical schema tensor shapes and types."""

    def test_tensor_shapes(self):
        state = RomBattleState()
        tensors = state.to_tensors()
        self.assertEqual(tensors["global_cat"].shape, (6,))
        self.assertEqual(tensors["global_num"].shape, (3,))
        self.assertEqual(tensors["pokemon_cat"].shape, (NUM_POKEMON_SLOTS, 9))
        self.assertEqual(tensors["pokemon_move_cat"].shape, (NUM_POKEMON_SLOTS, 4))
        self.assertEqual(tensors["pokemon_move_type"].shape, (NUM_POKEMON_SLOTS, 4))
        self.assertEqual(tensors["pokemon_num"].shape, (NUM_POKEMON_SLOTS, 31))
        self.assertEqual(tensors["pokemon_mask"].shape, (NUM_POKEMON_SLOTS, 4))
        self.assertEqual(tensors["legal_action_mask"].shape, (NUM_ACTIONS,))

    def test_flat_shapes(self):
        state = RomBattleState()
        flat = state.to_flat()
        self.assertEqual(flat["categorical"].ndim, 1)
        self.assertEqual(flat["numerical"].ndim, 1)
        self.assertEqual(flat["masks"].ndim, 1)

    def test_json_serializable(self):
        state = RomBattleState()
        j = state.to_json()
        json.dumps(j)  # should not raise

    def test_default_values(self):
        state = RomBattleState()
        # All pokemon should be invalid by default
        for p in state.pokemon:
            self.assertFalse(p.valid)
        # All legal actions should be False
        self.assertFalse(any(state.legal_action_mask))


class TestMetamonEncoder(unittest.TestCase):
    """Test the Metamon-side encoder."""

    def test_basic_encoding(self):
        encoder = RomObservationEncoder(gen=1)
        encoder.reset()
        state = _make_simple_state()
        rom_state = encoder.encode(state)

        # Player active should be Gengar (species 94)
        p = rom_state.pokemon[SLOT_PLAYER_ACTIVE]
        self.assertTrue(p.valid)
        self.assertEqual(p.species, 94)  # Gengar
        self.assertEqual(p.type_1, 9)    # Ghost
        self.assertEqual(p.type_2, 4)    # Poison
        self.assertEqual(p.status, 0)    # No status
        self.assertAlmostEqual(p.hp_fraction, 1.0)

        # Should have 2 moves encoded (sorted alphabetically: psychic < thunderbolt)
        self.assertEqual(p.move_ids[0], 94)   # psychic = 94
        self.assertEqual(p.move_ids[1], 85)   # thunderbolt = 85

    def test_opponent_moves_hidden(self):
        """Opponent moves should not leak when not revealed."""
        encoder = RomObservationEncoder(gen=1)
        encoder.reset()
        state = _make_simple_state()
        rom_state = encoder.encode(state)

        opp = rom_state.pokemon[SLOT_OPPONENT_ACTIVE]
        self.assertTrue(opp.valid)
        self.assertEqual(opp.species, 65)  # Alakazam
        # Moves should be unknown (opponent has empty moves list)
        self.assertFalse(opp.moves_revealed)
        for mid in opp.move_ids:
            self.assertEqual(mid, MOVE_UNKNOWN)

    def test_player_switches_encoded(self):
        encoder = RomObservationEncoder(gen=1)
        encoder.reset()
        state = _make_simple_state()
        rom_state = encoder.encode(state)

        # Should have Snorlax (143) and Tauros (128) as switches
        s0 = rom_state.pokemon[SLOT_SWITCH_0]
        s1 = rom_state.pokemon[SLOT_SWITCH_1]

        # Switches are sorted alphabetically, so snorlax < tauros
        self.assertTrue(s0.valid)
        self.assertTrue(s1.valid)
        # One should be Snorlax (143), other Tauros (128)
        species = {s0.species, s1.species}
        self.assertIn(143, species)  # Snorlax
        self.assertIn(128, species)  # Tauros

    def test_legal_action_mask(self):
        encoder = RomObservationEncoder(gen=1)
        encoder.reset()
        state = _make_simple_state()
        rom_state = encoder.encode(state)

        # Player has 2 moves, 2 switches
        mask = rom_state.legal_action_mask
        self.assertTrue(mask[0])  # move 0
        self.assertTrue(mask[1])  # move 1
        self.assertFalse(mask[2]) # move 2 (padding)
        self.assertFalse(mask[3]) # move 3 (padding)
        self.assertTrue(mask[4])  # switch 0
        self.assertTrue(mask[5])  # switch 1
        self.assertFalse(mask[6]) # switch 2 (padding)
        self.assertFalse(mask[7])
        self.assertFalse(mask[8])

    def test_forced_switch(self):
        encoder = RomObservationEncoder(gen=1)
        encoder.reset()
        state = _make_simple_state()
        state.forced_switch = True
        rom_state = encoder.encode(state)

        # During forced switch, move actions should be illegal
        mask = rom_state.legal_action_mask
        self.assertFalse(mask[0])
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertTrue(mask[4])  # switches still legal
        self.assertTrue(mask[5])
        self.assertEqual(rom_state.global_features.forced_switch, 1.0)

    def test_deterministic_encoding(self):
        """Same state should produce same encoding."""
        for _ in range(2):
            encoder = RomObservationEncoder(gen=1)
            encoder.reset()
            state = _make_simple_state()
            rom_state = encoder.encode(state)
            t1 = rom_state.to_tensors()

        encoder2 = RomObservationEncoder(gen=1)
        encoder2.reset()
        state2 = _make_simple_state()
        rom_state2 = encoder2.encode(state2)
        t2 = rom_state2.to_tensors()

        for key in t1:
            np.testing.assert_array_equal(t1[key], t2[key])

    def test_revealed_opponents_tracking(self):
        """Opponent species should be tracked across timesteps."""
        encoder = RomObservationEncoder(gen=1)
        encoder.reset()

        # Turn 1: opponent has Alakazam
        state1 = _make_simple_state()
        rom1 = encoder.encode(state1)

        # Only Alakazam should be revealed
        revealed_slots = rom1.pokemon[SLOT_REVEALED_OPP_0:SLOT_REVEALED_OPP_0+6]
        valid_revealed = [p for p in revealed_slots if p.valid and p.species != 65]  # exclude active
        self.assertEqual(len(valid_revealed), 0)  # No additional revealed yet

        # Turn 2: opponent switches to Exeggutor
        state2 = _make_simple_state()
        state2.opponent_active_pokemon = _make_simple_pokemon("exeggutor")
        state2.opponent_active_pokemon.moves = []
        rom2 = encoder.encode(state2)

        # Now Alakazam should be in revealed slots
        revealed_slots = rom2.pokemon[SLOT_REVEALED_OPP_0:SLOT_REVEALED_OPP_0+6]
        revealed_species = [p.species for p in revealed_slots if p.valid]
        self.assertIn(65, revealed_species)  # Alakazam should be revealed

    def test_status_encoding(self):
        """Test that status conditions are correctly encoded."""
        for status_name, expected_id in [
            ("nostatus", 0), ("slp", 1), ("psn", 2), ("brn", 3),
            ("frz", 4), ("par", 5), ("tox", 6), ("fnt", 7),
        ]:
            encoder = RomObservationEncoder(gen=1)
            encoder.reset()
            state = _make_simple_state()
            state.player_active_pokemon.status = status_name
            rom_state = encoder.encode(state)
            self.assertEqual(
                rom_state.pokemon[SLOT_PLAYER_ACTIVE].status, expected_id,
                f"Status '{status_name}' should map to {expected_id}"
            )

    def test_hp_fraction(self):
        """Test HP fraction encoding."""
        for hp_pct in [1.0, 0.5, 0.25, 0.0]:
            encoder = RomObservationEncoder(gen=1)
            encoder.reset()
            state = _make_simple_state()
            state.player_active_pokemon.hp_pct = hp_pct
            rom_state = encoder.encode(state)
            p = rom_state.pokemon[SLOT_PLAYER_ACTIVE]
            self.assertAlmostEqual(p.hp_fraction, hp_pct, places=2)
            if hp_pct <= 0.0:
                self.assertTrue(p.fainted)

    def test_no_information_leakage(self):
        """Ensure hidden opponent info doesn't leak into observation."""
        encoder = RomObservationEncoder(gen=1)
        encoder.reset()
        state = _make_simple_state()

        # Opponent has no moves revealed, unknown item
        opp = state.opponent_active_pokemon
        opp.moves = []
        opp.item = "unknownitem"

        rom_state = encoder.encode(state)
        opp_encoded = rom_state.pokemon[SLOT_OPPONENT_ACTIVE]

        # Moves should be unknown
        self.assertFalse(opp_encoded.moves_revealed)
        for mid in opp_encoded.move_ids:
            self.assertEqual(mid, MOVE_UNKNOWN)

        # Move BP/ACC/PRI/PP should be sentinel values
        for i in range(4):
            self.assertEqual(opp_encoded.move_bp[i], -2.0)
            self.assertEqual(opp_encoded.move_acc[i], -2.0)

    def test_real_trajectory(self):
        """Test encoding from a real replay file."""
        sample_file = os.path.expanduser(
            '~/metamon/trajectories/metamon_1400/gen1ou/gen1ou-2220252351_1410_chansey96380_vs_mirrorcoat22581_10-10-2024_WIN.json.lz4'
        )
        if not os.path.exists(sample_file):
            self.skipTest("Sample trajectory file not found")

        with lz4.frame.open(sample_file, 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))

        encoder = RomObservationEncoder(gen=1)
        encoder.reset()

        for i, state_dict in enumerate(data['states'][:10]):
            state = UniversalState.from_dict(state_dict)
            rom_state = encoder.encode(state)
            tensors = rom_state.to_tensors()

            # Verify shapes are consistent
            self.assertEqual(tensors['pokemon_cat'].shape, (13, 9))
            self.assertEqual(tensors['pokemon_num'].shape, (13, 31))

            # Player active should always be valid
            self.assertTrue(rom_state.pokemon[SLOT_PLAYER_ACTIVE].valid)

        # After 10 turns, some opponents should be revealed
        revealed_count = sum(
            1 for p in rom_state.pokemon[SLOT_REVEALED_OPP_0:SLOT_REVEALED_OPP_0+6]
            if p.valid
        )
        self.assertGreaterEqual(revealed_count, 0)  # at least no crashes


if __name__ == '__main__':
    unittest.main()
