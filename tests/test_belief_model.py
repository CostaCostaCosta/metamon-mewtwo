import numpy as np
import pytest
import torch

from metamon.interface import UniversalAction, UniversalMove, UniversalPokemon, UniversalState
from metamon.rl.belief import (
    BELIEF_MOVES_KEY,
    BELIEF_MOVES_MASK_KEY,
    BELIEF_SPECIES_KEY,
    BELIEF_SPECIES_MASK_KEY,
    Gen1OpponentTeamBeliefHead,
    MetamonBeliefMultiTaskAgent,
    build_gen1_belief_targets,
)
from metamon.rl.metamon_to_amago import MetamonAMAGODataset
from metamon.tokenizer import get_tokenizer


def make_move(name="tackle", base_power=40):
    return UniversalMove(
        name=name,
        move_type="normal",
        category="physical",
        base_power=base_power,
        accuracy=1.0,
        priority=0,
        current_pp=10,
        max_pp=10,
    )


def make_pokemon(name, moves=None):
    return UniversalPokemon(
        name=name,
        base_species=name,
        hp_pct=1.0,
        types="normal notype",
        item="noitem",
        ability="noability",
        lvl=100,
        status="nostatus",
        effect="noeffect",
        moves=moves if moves is not None else [make_move()],
        atk_boost=0,
        spa_boost=0,
        def_boost=0,
        spd_boost=0,
        spe_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        base_atk=100,
        base_spa=100,
        base_def=100,
        base_spd=100,
        base_spe=100,
        base_hp=100,
        tera_type="notype",
    )


def make_state(opponent_name, opponent_moves):
    return UniversalState(
        format="gen1ou",
        player_active_pokemon=make_pokemon("alakazam"),
        opponent_active_pokemon=make_pokemon(opponent_name, moves=opponent_moves),
        available_switches=[make_pokemon("chansey")],
        player_prev_move=make_move("psychic"),
        opponent_prev_move=opponent_moves[0],
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


def test_gen1_belief_targets_are_set_style_and_masked():
    tokenizer = get_tokenizer("DefaultObservationSpace-v1")
    states = [
        make_state("starmie", [make_move("surf"), make_move("recover")]),
        make_state("tauros", [make_move("bodyslam"), make_move("hyperbeam")]),
    ]

    targets = build_gen1_belief_targets(states, tokenizer)

    assert targets[BELIEF_SPECIES_KEY].shape == (2, len(tokenizer))
    assert targets[BELIEF_MOVES_KEY].shape == (2, len(tokenizer))
    assert targets[BELIEF_SPECIES_MASK_KEY].tolist() == [[1.0], [1.0]]
    assert targets[BELIEF_MOVES_MASK_KEY].tolist() == [[1.0], [1.0]]

    for species in ["starmie", "tauros"]:
        idx = tokenizer[species]
        assert idx >= 0
        assert targets[BELIEF_SPECIES_KEY][:, idx].tolist() == [1.0, 1.0]
    for move in ["surf", "recover", "bodyslam", "hyperbeam"]:
        idx = tokenizer[move]
        assert idx >= 0
        assert targets[BELIEF_MOVES_KEY][:, idx].tolist() == [1.0, 1.0]


def test_belief_head_forward_and_loss_shapes():
    head = Gen1OpponentTeamBeliefHead(
        state_dim=12,
        vocab_size=17,
        belief_dim=5,
        hidden_dim=16,
        n_layers=1,
        include_moves=True,
    )
    state = torch.randn(2, 3, 12)
    outputs = head(state)

    assert outputs.species_logits.shape == (2, 3, 17)
    assert outputs.move_logits.shape == (2, 3, 17)
    assert outputs.actor_embedding.shape == (2, 3, 5)

    obs = {
        BELIEF_SPECIES_KEY: torch.zeros(2, 3, 17),
        BELIEF_SPECIES_MASK_KEY: torch.ones(2, 3, 1),
        BELIEF_MOVES_KEY: torch.zeros(2, 3, 17),
        BELIEF_MOVES_MASK_KEY: torch.ones(2, 3, 1),
    }
    obs[BELIEF_SPECIES_KEY][..., 4] = 1.0
    obs[BELIEF_MOVES_KEY][..., 7] = 1.0
    loss, metrics = head.compute_loss(
        outputs,
        obs,
        valid_timestep_mask=torch.ones(2, 3, 1, dtype=torch.bool),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["Belief Species Mask Count"].item() == 6
    assert metrics["Belief Moves Mask Count"].item() == 6


def test_amago_dataset_propagates_belief_targets_outside_public_obs():
    class DummyActionSpace:
        gym_space = type("Space", (), {"n": 3})()

        def action_to_agent_output(self, state, action: UniversalAction):
            return action.action_idx

    class DummyReplayDataset:
        action_space = DummyActionSpace()

    wrapper = MetamonAMAGODataset(parsed_replay_dset=DummyReplayDataset())
    obs = {
        "numbers": [np.zeros(2, dtype=np.float32) for _ in range(3)],
        "text_tokens": [np.zeros(4, dtype=np.int32) for _ in range(3)],
    }
    action_infos = {
        "chosen": [0, 1],
        "legal": [{0, 1}, {1, 2}],
        "missing": [False, False],
    }
    targets = {
        BELIEF_SPECIES_KEY: torch.zeros(3, 11),
        BELIEF_SPECIES_MASK_KEY: torch.ones(3, 1),
    }

    rl_data = wrapper._process_data(
        (
            obs,
            action_infos,
            np.zeros(2, dtype=np.float32),
            np.array([False, True]),
            targets,
        )
    )

    assert BELIEF_SPECIES_KEY in rl_data.obs
    assert BELIEF_SPECIES_MASK_KEY in rl_data.obs
    assert rl_data.obs[BELIEF_SPECIES_KEY].shape == (3, 11)
    assert "numbers" in rl_data.obs
    assert "text_tokens" in rl_data.obs


def test_actor_pass_through_rejects_belief_keys():
    MetamonBeliefMultiTaskAgent._check_no_belief_actor_keys(["illegal_actions"])
    with pytest.raises(ValueError, match="must not be passed directly"):
        MetamonBeliefMultiTaskAgent._check_no_belief_actor_keys(
            ["illegal_actions", BELIEF_SPECIES_KEY]
        )
