from typing import Optional, Any, Type
from collections import deque
from contextlib import contextmanager
import copy
import math
import os
import warnings

import gin
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import einops


from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    ActionSpace,
    UniversalAction,
)
from metamon.il.model import (
    TransformerTurnEmbedding,
    PerceiverTurnEmbedding,
    TokenEmbedding,
    MultiModalEmbedding,
    LearnablePosEmb,
    PerceiverEncoder,
)
from metamon.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.data import ParsedReplayDataset
from metamon.env import (
    TeamSet,
    PokeEnvWrapper,
    BattleAgainstBaseline,
    QueueOnLocalLadder,
    ChallengeByUsername,
    PokeAgentLadder,
)

try:
    import amago
except ImportError:
    raise ImportError(
        "Must install `amago` RL package. Visit: https://ut-austin-rpl.github.io/amago/ "
    )
else:
    assert (
        hasattr(amago, "__version__") and amago.__version__ >= "3.4.0"
    ), f"AMAGO v3.4.0+ required; found {getattr(amago, '__version__', 'unknown')}."
    from amago.envs import AMAGOEnv
    from amago.nets.utils import symlog, add_activation_log
    from amago.loading import RLData, RLDataset, Batch, MAGIC_PAD_VAL
    from amago.envs.amago_env import AMAGO_ENV_LOG_PREFIX
    from amago.nets.ff import Normalization


def _block_warnings():
    """Suppress common gymnasium warnings during environment creation."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=amago.utils.AmagoWarning)


@gin.configurable
class BatchNormalizedExpFilter:
    """Batch-normalized exponential weighting for filtered behavior cloning.

    Z-scores advantages over *unmasked* positions before applying the
    exponential, making ``beta`` invariant to the absolute scale of
    Q-values / rewards.  Inspired by GRPO-style relative advantage
    normalization.

    Because amago's ``fbc_filter_func`` interface only passes the advantage
    tensor, the mask must be injected externally via :meth:`set_mask` before
    the agent forward pass.  :class:`MetamonAMAGOExperiment` handles this
    automatically in :meth:`train_step`.

    Args:
        beta: Scale applied after normalization.  With unit-variance inputs,
            values in [1, 3] give a stable curriculum.
        eps: Small constant for numerical stability in std computation.
        clip_weights_low: Floor for output weights.
        clip_weights_high: Ceiling for output weights.
    """

    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-8,
        clip_weights_low: Optional[float] = 1e-7,
        clip_weights_high: Optional[float] = 100.0,
    ):
        self.beta = beta
        self.eps = eps
        self.clip_weights_low = clip_weights_low
        self.clip_weights_high = clip_weights_high
        self._mask: Optional[torch.Tensor] = None

    def set_mask(self, mask: Optional[torch.Tensor]):
        """Set the boolean mask for the next ``__call__``.

        Args:
            mask: (Batch, Length, 1) or broadcastable bool tensor. ``True``
                where the advantage is valid.  Cleared after each call.
        """
        self._mask = mask

    def __call__(self, adv: torch.Tensor) -> torch.Tensor:
        mask = self._mask
        self._mask = None

        if mask is not None:
            mask = mask[:, : adv.shape[1], ...]
            while mask.ndim < adv.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(adv)
            valid = adv[mask]
            mu = valid.mean()
            sigma = valid.std() + self.eps
        else:
            mu = adv.mean()
            sigma = adv.std() + self.eps

        adv_norm = (adv - mu) / sigma
        weights = torch.exp(self.beta * adv_norm)
        if self.clip_weights_low is not None or self.clip_weights_high is not None:
            weights = torch.clamp(
                weights, min=self.clip_weights_low, max=self.clip_weights_high
            )
        return weights


def make_placeholder_env(
    observation_space: ObservationSpace, action_space: ActionSpace
) -> AMAGOEnv:
    """
    Create an environment that does nothing. Can be used to initialize a policy
    """
    _block_warnings()

    class _PlaceholderShowdown(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = observation_space.gym_space
            self.metamon_action_space = action_space
            self.action_space = action_space.gym_space
            self.observation_space["illegal_actions"] = gym.spaces.Box(
                low=0, high=1, shape=(self.action_space.n,), dtype=bool
            )
            self.metamon_battle_format = "PlaceholderShowdown"
            self.metamon_opponent_name = "PlaceholderOpponent"

        def reset(self, *args, **kwargs):
            obs = {
                key: np.zeros(value.shape, dtype=value.dtype)
                for key, value in self.observation_space.items()
            }
            return obs, {"legal_actions": []}

        def take_long_break(self):
            pass

        def resume_from_break(self):
            pass

    penv = _PlaceholderShowdown()
    return MetamonAMAGOWrapper(penv)


def make_local_ladder_env(*args, **kwargs):
    """
    Battle on the local Showdown ladder!
    """
    _block_warnings()
    menv = QueueOnLocalLadder(*args, **kwargs)
    print("Made Local Ladder Env")
    return PSLadderAMAGOWrapper(menv)


def make_pokeagent_ladder_env(*args, **kwargs):
    """
    Battle on the NeurIPS 2025 PokéAgent Challenge ladder!
    """
    _block_warnings()
    menv = PokeAgentLadder(*args, **kwargs)
    print("Made PokeAgent Ladder Env")
    return PSLadderAMAGOWrapper(menv)


def make_challenge_env(*args, **kwargs):
    """
    Battle a specific opponent by username (head-to-head challenge mode).
    """
    _block_warnings()
    menv = ChallengeByUsername(*args, **kwargs)
    print(
        f"Made Challenge Env ({menv._role}): {menv.player_username} vs {menv._opponent_username}"
    )
    return PSLadderAMAGOWrapper(menv)


def make_baseline_env(*args, **kwargs):
    """
    Battle against a built-in baseline opponent
    """
    _block_warnings()
    menv = BattleAgainstBaseline(*args, **kwargs)
    print("Made Baseline Env")
    return MetamonAMAGOWrapper(menv)


def make_placeholder_experiment(
    ckpt_base_dir: str,
    run_name: str,
    log: bool,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    experiment_type: type = None,
):
    """
    Initialize an AMAGO experiment that will be used to load a pretrained checkpoint
    and manage agent/env interaction.

    Args:
        experiment_type: Experiment class to instantiate. Defaults to MetamonAMAGOExperiment.
    """
    if experiment_type is None:
        experiment_type = MetamonAMAGOExperiment
    penv = make_placeholder_env(
        observation_space=observation_space,
        action_space=action_space,
    )
    dummy_dset = amago.loading.DoNothingDataset()
    dummy_env = lambda: penv
    experiment = experiment_type(
        # assumes that positional args
        # agent_type, tstep_encoder_type,
        # traj_encoder_type, and max_seq_len
        # are set in the gin file
        ckpt_base_dir=ckpt_base_dir,
        run_name=run_name,
        dataset=dummy_dset,
        make_train_env=dummy_env,
        make_val_env=dummy_env,
        env_mode="sync",
        async_env_mp_context="spawn",
        parallel_actors=1,
        exploration_wrapper_type=None,
        epochs=0,
        start_learning_at_epoch=float("inf"),
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        traj_save_len=10_000_000_000,
        stagger_traj_file_lengths=False,
        train_batches_per_epoch=0,
        val_interval=None,
        val_timesteps_per_epoch=0,
        ckpt_interval=None,
        always_save_latest=False,
        always_load_latest=False,
        log_interval=1,
        batch_size=1,
        dloader_workers=0,
        log_to_wandb=log,
        wandb_project=os.environ.get("METAMON_WANDB_PROJECT"),
        wandb_entity=os.environ.get("METAMON_WANDB_ENTITY"),
        verbose=True,
    )
    return experiment


class MetamonAMAGOWrapper(amago.envs.AMAGOEnv):
    """AMAGOEnv wrapper for poke-env gymnasium environments.

    - Extends the observation space with an illegal action mask, which will
        be passed along to the actor network.
    - Adds success rate and valid action rate logging.
    """

    def __init__(self, metamon_env: PokeEnvWrapper):
        self.metamon_action_space = metamon_env.metamon_action_space
        super().__init__(
            env=metamon_env,
            env_name="metamon",
            batched_envs=1,
        )
        assert isinstance(self.action_space, gym.spaces.Discrete)
        self.observation_space["illegal_actions"] = gym.spaces.Box(
            low=0, high=1, shape=(self.action_space.n,), dtype=bool
        )

    def add_illegal_action_mask_to_obs(self, obs: dict, info: dict):
        # move legal action from info to obs
        legal_actions = info["legal_actions"]
        illegal_actions = np.ones((self.action_space.n,), dtype=bool)
        for agent_legal_action in legal_actions:
            illegal_actions[agent_legal_action] = False
        obs["illegal_actions"] = illegal_actions

    def inner_reset(self, *args, **kwargs):
        # move legal action from info to obs
        obs, info = self.env.reset(*args, **kwargs)
        self.add_illegal_action_mask_to_obs(obs, info)
        return obs, info

    def inner_step(self, action):
        # move legal action from info to obs
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.add_illegal_action_mask_to_obs(obs, info)
        return obs, reward, terminated, truncated, info

    def step(self, action):
        try:
            next_tstep, reward, terminated, truncated, info = super().step(action)
            # amago will average these stats over episodes, devices, and parallel actors.
            if "won" in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Win Rate"] = info["won"]
            if "valid_action_count" in info and "invalid_action_count" in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Valid Actions"] = info[
                    "valid_action_count"
                ] / (info["valid_action_count"] + info["invalid_action_count"])
            return next_tstep, reward, terminated, truncated, info
        except Exception as e:
            print(e)
            print("Force resetting due to long-tail error")
            self.reset()
            next_tstep, reward, terminated, truncated, info = self.step(action)
            reward *= 0.0
            terminated[:] = False
            truncated[:] = True  # force a proper reset asap
            return next_tstep, reward, terminated, truncated, info

    @property
    def env_name(self):
        return f"{self.env.metamon_battle_format}_vs_{self.env.metamon_opponent_name}"


@gin.configurable
class MetamonDiscrete(amago.nets.policy_dists.Discrete):
    """Discrete policy with temperature-based sampling.

    Extends AMAGO's Discrete PolicyOutput to add temperature scaling to the logits.
    High-temperature sampling is a better alternative to epsilon-greedy exploration
    for self-play in metamon due to illegal action masking.

    Args:
        d_action: Dimension of the action space.
        temperature: Temperature for scaling logits. Default is 1.0 (no scaling).
        clip_prob_low: Clips action probabilities to this value before
            renormalizing. Default is 0.001.
        clip_prob_high: Clips action probabilities to this value before
            renormalizing. Default is 0.99.
    """

    def __init__(
        self,
        d_action: int,
        clip_prob_low: float = 0.001,
        clip_prob_high: float = 0.99,
        temperature: float = 1.0,
    ):
        super().__init__(
            d_action=d_action,
            clip_prob_low=clip_prob_low,
            clip_prob_high=clip_prob_high,
        )
        self.temperature = temperature

    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> amago.nets.policy_dists._Categorical:
        scaled_logits = vec / self.temperature

        dist = amago.nets.policy_dists._Categorical(logits=scaled_logits)
        probs = dist.probs
        clip_probs = probs.clamp(self.clip_prob_low, self.clip_prob_high)
        safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
        safe_dist = amago.nets.policy_dists._Categorical(probs=safe_probs)

        if log_dict is not None:
            from amago.nets.utils import add_activation_log

            add_activation_log("MetamonDiscrete-probs", probs, log_dict)
            add_activation_log(
                "MetamonDiscrete-temperature", torch.tensor(self.temperature), log_dict
            )

        return safe_dist


@gin.configurable
class MetamonMaskedActor(amago.nets.actor_critic.Actor):
    """
    Default AMAGO Actor with optional logit masking of illegal actions.

    Note that all the original models were trained with the equivalent of
    mask_illegal_actions=False... the dataset would not have illegal actions,
    and in self-play data an illegal action triggers a random one to be taken,
    so it's always a bad idea, and critic nets have no problem learning this.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        n_layers: int = 2,
        d_hidden: int = 256,
        activation: str = "leaky_relu",
        dropout_p: float = 0.0,
        continuous_dist_type=None,
        mask_illegal_actions: bool = True,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            n_layers=n_layers,
            d_hidden=d_hidden,
            activation=activation,
            dropout_p=dropout_p,
            continuous_dist_type=continuous_dist_type,
            discrete_dist_type=MetamonDiscrete,
        )
        self.mask_illegal_actions = mask_illegal_actions

    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict[str, Any]] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ):
        dist_params = super().actor_network_forward(
            state, log_dict=log_dict, straight_from_obs=straight_from_obs
        )
        if self.mask_illegal_actions:
            Batch, Len, Gammas, N = dist_params.shape
            mask = straight_from_obs["illegal_actions"]
            no_options = mask.all(dim=-1, keepdim=True)
            # TODO: having no legal options should be considered a problem
            # with action masking / action space, but seems to happen
            # for two reasons: 1) battle is over and there's nothing left to do
            # (harmless) and 2) gen 9 revival blessing edge case (need to revisit).
            # prevent crash by letting agent pick its own action and dealing with
            # legality on the env side (probably falling back to a default choice).
            mask = torch.logical_and(mask, ~no_options)
            mask = einops.repeat(mask, f"b l n -> b l {Gammas} n")
            dist_params.masked_fill_(mask, -float("inf"))
        return dist_params


@gin.configurable
class MetamonMaskedResidualActor(amago.nets.actor_critic.ResidualActor):
    """ResidualActor with optional masking of illegal actions in logits.

    Mirrors `MetamonMaskedActor` but for AMAGO's ResidualActor head.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        feature_dim: int = 256,
        residual_ff_dim: int = 512,
        residual_blocks: int = 2,
        activation: str = "leaky_relu",
        normalization: str = "layer",
        dropout_p: float = 0.0,
        continuous_dist_type=None,
        mask_illegal_actions: bool = True,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            feature_dim=feature_dim,
            residual_ff_dim=residual_ff_dim,
            residual_blocks=residual_blocks,
            activation=activation,
            normalization=normalization,
            dropout_p=dropout_p,
            continuous_dist_type=continuous_dist_type,
            discrete_dist_type=MetamonDiscrete,
        )
        self.mask_illegal_actions = mask_illegal_actions

    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict[str, Any]] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        dist_params = super().actor_network_forward(
            state, log_dict=log_dict, straight_from_obs=straight_from_obs
        )
        if self.mask_illegal_actions and straight_from_obs is not None:
            Batch, Len, Gammas, N = dist_params.shape
            mask = straight_from_obs["illegal_actions"]
            no_options = mask.all(dim=-1, keepdim=True)
            mask = torch.logical_and(mask, ~no_options)
            mask = einops.repeat(mask, f"b l n -> b l {Gammas} n")
            dist_params.masked_fill_(mask, -float("inf"))
        return dist_params


class PSLadderAMAGOWrapper(MetamonAMAGOWrapper):
    """AMAGO wrapper for envs with a fixed number of battles (ladder or challenge mode).

    Blocks auto-resets after num_battles to avoid creating battles that won't be completed.
    Works with both QueueOnLocalLadder and ChallengeByUsername.
    """

    def __init__(self, env):
        assert isinstance(env, (QueueOnLocalLadder, ChallengeByUsername))
        self.placeholder_obs = None
        self.battle_counter = 0
        super().__init__(env)

    def inner_reset(self, *args, **kwargs):
        if self.battle_counter >= self.env.num_battles:
            # quirk of amago's parallel actor auto-resets that matters
            # for online ladder and challenge mode.
            warnings.warn(
                "Blocking auto-reset to avoid creating a battle that will not be completed!"
            )
            return self.placeholder_obs, {}
        obs, info = self.env.reset(*args, **kwargs)
        self.battle_counter += 1
        if self.placeholder_obs is None:
            self.placeholder_obs = obs
        # move legal action from info to obs
        self.add_illegal_action_mask_to_obs(obs, info)
        return obs, info

    @property
    def env_name(self):
        return f"psladder_{self.env.env.username}"


def unknown_token_mask(tokens, skip_prob: float = 0.5, batch_max_prob: float = 0.2):
    """Randomly set entries in the text component of the observation space to UNKNOWN_TOKEN.

    Args:
        skip_prob: Probability of entirely skipping the mask for any given sequence
        batch_max_prob: For each sequence, randomly mask tokens with [0, batch_max_prob) prob
            (if not skipped).
    """
    B, L, tok = tokens.shape
    dev = tokens.device
    batch_mask = torch.rand(B) < (1.0 - skip_prob)  # mask tokens from this batch index
    batch_thresh = (
        torch.rand(B) * batch_max_prob
    )  # mask this % of tokens from the sequence
    thresh = (
        batch_mask * batch_thresh
    )  # 0 if batch index isn't masked, % to mask otherwise
    mask = torch.rand(tokens.shape) < thresh.view(-1, 1, 1)
    tokens[mask.to(dev)] = UNKNOWN_TOKEN
    return tokens.to(dev)


@gin.configurable
class MetamonTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    """
    Token + numerical embedding for Metamon.

    Fuses multi-modal input with attention and summary tokens.
    Visualized on the README and in the paper architecture figure.
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        extra_emb_dim: int = 18,
        d_model: int = 100,
        n_layers: int = 3,
        n_heads: int = 5,
        scratch_tokens: int = 4,
        numerical_tokens: int = 6,
        token_mask_aug: bool = False,
        dropout: float = 0.05,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        base_numerical_features = obs_space["numbers"].shape[0]
        base_text_features = obs_space["text_tokens"].shape[0]
        self.turn_embedding = TransformerTurnEmbedding(
            tokenizer=tokenizer,
            token_embedding_dim=d_model,
            text_features=base_text_features,
            numerical_features=base_numerical_features + extra_emb_dim,
            numerical_tokens=numerical_tokens,
            scratch_tokens=scratch_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    @property
    def emb_dim(self):
        return self.turn_embedding.output_dim

    @torch.compile
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        add_activation_log("MetamonTstepEncoder/extra_emb", extras, log_dict)
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        add_activation_log("MetamonTstepEncoder/numerical", numerical, log_dict)
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
        add_activation_log("MetamonTstepEncoder/turn_emb", turn_emb, log_dict)
        return turn_emb


@gin.configurable
class MetamonPerceiverTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    """
    Efficient attention scheme for processing turn token inputs.

    Uses latent cross-/self-attention with learnable positional embeddings.
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        extra_emb_dim: int = 18,
        d_model: int = 100,
        n_layers: int = 3,
        n_heads: int = 5,
        latent_tokens: int = 8,
        numerical_tokens: int = 6,
        token_mask_aug: bool = False,
        dropout: float = 0.05,
        max_tokens_per_turn: int = 128,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        base_numerical_features = obs_space["numbers"].shape[0]
        base_text_features = obs_space["text_tokens"].shape[0]
        self.turn_embedding = PerceiverTurnEmbedding(
            tokenizer=tokenizer,
            token_embedding_dim=d_model,
            text_features=base_text_features,
            numerical_features=base_numerical_features + extra_emb_dim,
            numerical_tokens=numerical_tokens,
            latent_tokens=latent_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_tokens_per_turn=max_tokens_per_turn,
        )

    @property
    def emb_dim(self):
        return self.turn_embedding.output_dim

    @torch.compile
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        add_activation_log("MetamonPerceiverTstepEncoder/extra_emb", extras, log_dict)
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        add_activation_log(
            "MetamonPerceiverTstepEncoder/numerical", numerical, log_dict
        )
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
        add_activation_log("MetamonPerceiverTstepEncoder/turn_emb", turn_emb, log_dict)
        return turn_emb


class PokemonSlotTurnEmbedding(nn.Module):
    """
    Encode fixed Pokemon slots independently, then merge slot and global context
    tokens with a second Perceiver block.
    """

    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        slot_count: int,
        pokemon_text_features: int,
        pokemon_numerical_features: int,
        global_text_features: int,
        global_numerical_features: int,
        token_embedding_dim: int,
        d_model: int,
        n_heads: int,
        slot_layers: int,
        team_layers: int,
        slot_latent_tokens: int,
        team_latent_tokens: int,
        pokemon_numerical_tokens: int,
        global_numerical_tokens: int,
        dropout: float,
        max_pokemon_tokens: int,
        max_team_tokens: int,
    ):
        super().__init__()
        self.slot_count = slot_count
        self.pokemon_text_features = pokemon_text_features
        self.pokemon_numerical_features = pokemon_numerical_features
        self.global_text_features = global_text_features
        self.global_numerical_features = global_numerical_features
        self.token_embedding = TokenEmbedding(tokenizer, emb_dim=token_embedding_dim)

        self.pokemon_multimodal_fuse = MultiModalEmbedding(
            token_emb_dim=self.token_embedding.output_dim,
            numerical_d_inp=pokemon_numerical_features,
            output_dim=d_model,
            numerical_tokens=pokemon_numerical_tokens,
            dropout=dropout,
        )
        self.pokemon_pos = LearnablePosEmb(
            max_len=max_pokemon_tokens,
            d_model=d_model,
        )
        self.pokemon_perceiver = PerceiverEncoder(
            latent_tokens=slot_latent_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=slot_layers,
            dropout=dropout,
        )
        self.slot_projection = nn.Sequential(
            nn.Linear(self.pokemon_perceiver.output_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.slot_role_embedding = nn.Embedding(slot_count, d_model)

        self.global_multimodal_fuse = MultiModalEmbedding(
            token_emb_dim=self.token_embedding.output_dim,
            numerical_d_inp=global_numerical_features,
            output_dim=d_model,
            numerical_tokens=global_numerical_tokens,
            dropout=dropout,
        )
        self.team_pos = LearnablePosEmb(max_len=max_team_tokens, d_model=d_model)
        self.team_perceiver = PerceiverEncoder(
            latent_tokens=team_latent_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=team_layers,
            dropout=dropout,
        )

    @property
    def output_dim(self):
        return self.team_perceiver.output_dim

    def _add_pos(self, seq: torch.Tensor, pos_emb: LearnablePosEmb) -> torch.Tensor:
        pos = (
            torch.arange(0, seq.shape[1], device=seq.device)
            .long()
            .unsqueeze(0)
            .expand(seq.shape[0], -1)
        )
        return seq + pos_emb(pos)

    def forward(
        self,
        pokemon_token_inputs: torch.Tensor,
        pokemon_numerical_inputs: torch.Tensor,
        global_token_inputs: torch.Tensor,
        global_numerical_inputs: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = pokemon_token_inputs.shape
        pokemon_tokens = einops.rearrange(
            pokemon_token_inputs,
            "b t (s p) -> (b t s) 1 p",
            s=self.slot_count,
            p=self.pokemon_text_features,
        )
        pokemon_numbers = einops.rearrange(
            pokemon_numerical_inputs,
            "b t (s n) -> (b t s) 1 n",
            s=self.slot_count,
            n=self.pokemon_numerical_features,
        )
        pokemon_text_emb = self.token_embedding(pokemon_tokens)
        pokemon_seq = self.pokemon_multimodal_fuse(
            pokemon_text_emb,
            numerical_features=pokemon_numbers,
        )
        pokemon_seq = einops.rearrange(pokemon_seq, "b 1 l d -> b l d")
        pokemon_seq = self._add_pos(pokemon_seq, self.pokemon_pos)
        pokemon_slots = self.pokemon_perceiver(pokemon_seq)
        pokemon_slots = einops.rearrange(
            pokemon_slots,
            "(b t s) 1 d -> b t s d",
            b=B,
            t=T,
            s=self.slot_count,
        )
        pokemon_slots = self.slot_projection(pokemon_slots)
        role_idxs = torch.arange(
            0,
            self.slot_count,
            device=pokemon_slots.device,
        )
        pokemon_slots = pokemon_slots + self.slot_role_embedding(role_idxs).view(
            1, 1, self.slot_count, -1
        )

        global_text_emb = self.token_embedding(global_token_inputs)
        global_seq = self.global_multimodal_fuse(
            global_text_emb,
            numerical_features=global_numerical_inputs,
        )

        team_seq = torch.cat((pokemon_slots, global_seq), dim=-2)
        team_seq = einops.rearrange(team_seq, "b t l d -> (b t) l d")
        team_seq = self._add_pos(team_seq, self.team_pos)
        team_emb = self.team_perceiver(team_seq)
        team_emb = einops.rearrange(team_emb, "(b t) 1 d -> b t d", b=B, t=T)
        return team_emb


@gin.configurable
class MetamonPokemonSlotTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    """
    AMAGO timestep encoder for Gen1PokemonSlotObservationSpace.
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        extra_emb_dim: int = 18,
        d_model: int = 168,
        n_heads: int = 8,
        slot_layers: int = 2,
        team_layers: int = 8,
        slot_latent_tokens: int = 2,
        team_latent_tokens: int = 8,
        pokemon_numerical_tokens: int = 4,
        global_numerical_tokens: int = 2,
        token_mask_aug: bool = False,
        dropout: float = 0.05,
        max_pokemon_tokens: int = 16,
        max_team_tokens: int = 32,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)

        pokemon_text_features = obs_space["pokemon_text_tokens"].shape[0]
        pokemon_numerical_features = obs_space["pokemon_numbers"].shape[0]
        global_text_features = obs_space["global_text_tokens"].shape[0]
        global_numerical_features = obs_space["global_numbers"].shape[0] + extra_emb_dim

        slot_count = 13
        if pokemon_text_features % slot_count != 0:
            raise ValueError(
                "pokemon_text_tokens length must be divisible by the Pokemon slot count"
            )
        if pokemon_numerical_features % slot_count != 0:
            raise ValueError(
                "pokemon_numbers length must be divisible by the Pokemon slot count"
            )

        self.turn_embedding = PokemonSlotTurnEmbedding(
            tokenizer=tokenizer,
            slot_count=slot_count,
            pokemon_text_features=pokemon_text_features // slot_count,
            pokemon_numerical_features=pokemon_numerical_features // slot_count,
            global_text_features=global_text_features,
            global_numerical_features=global_numerical_features,
            token_embedding_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            slot_layers=slot_layers,
            team_layers=team_layers,
            slot_latent_tokens=slot_latent_tokens,
            team_latent_tokens=team_latent_tokens,
            pokemon_numerical_tokens=pokemon_numerical_tokens,
            global_numerical_tokens=global_numerical_tokens,
            dropout=dropout,
            max_pokemon_tokens=max_pokemon_tokens,
            max_team_tokens=max_team_tokens,
        )

    @property
    def emb_dim(self):
        return self.turn_embedding.output_dim

    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["pokemon_text_tokens"] = unknown_token_mask(obs["pokemon_text_tokens"])
            obs["global_text_tokens"] = unknown_token_mask(obs["global_text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        add_activation_log("MetamonPokemonSlotTstepEncoder/extra_emb", extras, log_dict)
        global_numbers = torch.cat((obs["global_numbers"], extras), dim=-1)
        add_activation_log(
            "MetamonPokemonSlotTstepEncoder/global_numbers",
            global_numbers,
            log_dict,
        )
        turn_emb = self.turn_embedding(
            pokemon_token_inputs=obs["pokemon_text_tokens"],
            pokemon_numerical_inputs=obs["pokemon_numbers"],
            global_token_inputs=obs["global_text_tokens"],
            global_numerical_inputs=global_numbers,
        )
        add_activation_log("MetamonPokemonSlotTstepEncoder/turn_emb", turn_emb, log_dict)
        return turn_emb


class _PerceiverLayer(nn.Module):
    """Cross-attention + self-attention with fused projections and F.scaled_dot_product_attention.

    Drop-in replacement for the PerceiverEncoder's paired CrossAttentionBlock +
    SelfAttentionBlock.  Same parameter count and semantics, but uses a single
    fused KV projection (cross) or QKV projection (self) and calls
    F.scaled_dot_product_attention directly instead of nn.MultiheadAttention.

    Optional ``cross_mask`` / ``self_mask`` boolean tensors (``True`` = masked
    out) enable block-diagonal attention for grouped independent processing.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        normformer_norms: bool = False,
        qk_norm: bool = False,
        ff_mult: int = 4,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self._dp = dropout
        self._normformer = normformer_norms
        self._qk_norm = qk_norm

        d_ff = d_model * ff_mult

        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_kv = nn.Linear(d_model, 2 * d_model)
        self.cross_out = nn.Linear(d_model, d_model)
        self.cross_ff_norm = nn.LayerNorm(d_model)
        self.cross_ff1 = nn.Linear(d_model, d_ff)
        self.cross_ff2 = nn.Linear(d_ff, d_model)
        self.cross_ff_drop = nn.Dropout(dropout)

        self.self_norm = nn.LayerNorm(d_model)
        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)
        self.self_ff_norm = nn.LayerNorm(d_model)
        self.self_ff1 = nn.Linear(d_model, d_ff)
        self.self_ff2 = nn.Linear(d_ff, d_model)
        self.self_ff_drop = nn.Dropout(dropout)

        if normformer_norms:
            self.cross_post_attn_norm = nn.LayerNorm(d_model)
            self.cross_mid_ff_norm = nn.LayerNorm(d_ff)
            self.self_post_attn_norm = nn.LayerNorm(d_model)
            self.self_mid_ff_norm = nn.LayerNorm(d_ff)

        if qk_norm:
            hd = self.head_dim
            self.cross_q_norm = nn.LayerNorm(hd)
            self.cross_k_norm = nn.LayerNorm(hd)
            self.self_q_norm = nn.LayerNorm(hd)
            self.self_k_norm = nn.LayerNorm(hd)

    def forward(
        self,
        latents: torch.Tensor,
        kv_input: torch.Tensor,
        cross_mask: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_block_mask=None,
        self_block_mask=None,
    ) -> torch.Tensor:
        H, HD, D = self.n_heads, self.head_dim, self.d_model
        dp = self._dp if self.training else 0.0
        B, Lq = latents.shape[:2]

        q = self.cross_q(self.cross_norm_q(latents))
        q = q.unflatten(-1, (H, HD)).transpose(1, 2)  # (B, H, Lq, HD)
        kv = self.cross_kv(self.cross_norm_kv(kv_input))
        kv = kv.unflatten(-1, (2, H, HD))
        k = kv[:, :, 0].transpose(1, 2)  # (B, H, Lkv, HD)
        v = kv[:, :, 1].transpose(1, 2)  # (B, H, Lkv, HD)
        if self._qk_norm:
            q = self.cross_q_norm(q)
            k = self.cross_k_norm(k)
        if cross_block_mask is not None:
            attn = flex_attention(q, k, v, block_mask=cross_block_mask)
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=cross_mask, dropout_p=dp
            )
        cross_out = self.cross_out(attn.transpose(1, 2).reshape(B, Lq, D))
        if self._normformer:
            cross_out = self.cross_post_attn_norm(cross_out)
        latents = latents + cross_out
        h = F.silu(self.cross_ff1(self.cross_ff_norm(latents)))
        if self._normformer:
            h = self.cross_mid_ff_norm(h)
        latents = latents + self.cross_ff_drop(self.cross_ff2(h))

        qkv = self.self_qkv(self.self_norm(latents))
        qkv = qkv.unflatten(-1, (3, H, HD))
        sq = qkv[:, :, 0].transpose(1, 2)  # (B, H, Lq, HD)
        sk = qkv[:, :, 1].transpose(1, 2)  # (B, H, Lq, HD)
        sv = qkv[:, :, 2].transpose(1, 2)  # (B, H, Lq, HD)
        if self._qk_norm:
            sq = self.self_q_norm(sq)
            sk = self.self_k_norm(sk)
        if self_block_mask is not None:
            attn = flex_attention(sq, sk, sv, block_mask=self_block_mask)
        else:
            attn = F.scaled_dot_product_attention(
                sq, sk, sv, attn_mask=self_mask, dropout_p=dp
            )
        self_out = self.self_out(attn.transpose(1, 2).reshape(B, Lq, D))
        if self._normformer:
            self_out = self.self_post_attn_norm(self_out)
        latents = latents + self_out
        h = F.silu(self.self_ff1(self.self_ff_norm(latents)))
        if self._normformer:
            h = self.self_mid_ff_norm(h)
        latents = latents + self.self_ff_drop(self.self_ff2(h))

        return latents


class _FastPerceiverEncoder(nn.Module):
    """Perceiver encoder with fused attention projections.

    Functionally identical to :class:`PerceiverEncoder` from ``metamon.il.model``
    but replaces ``nn.MultiheadAttention`` with fused QKV/KV linear projections
    and direct ``F.scaled_dot_product_attention`` calls.
    """

    def __init__(
        self,
        latent_tokens: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        normformer_norms: bool = False,
        qk_norm: bool = False,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_tokens, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                _PerceiverLayer(
                    d_model, n_heads, dropout, normformer_norms, qk_norm, ff_mult
                )
                for _ in range(n_layers)
            ]
        )
        self.output_dim = latent_tokens * d_model

    def forward(self, x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, x)
        if flatten:
            return latents.reshape(B, 1, -1)
        return latents


class _BlockDiagPerceiverEncoder(nn.Module):
    """Perceiver for *N* independent groups via block-diagonal attention masking.

    Tiles the shared learnable latent queries *N* times and pre-computes
    block-diagonal masks so each group's latents only attend to their own
    input tokens (cross-attention) and to each other (self-attention).

    This is **semantically identical** to running a perceiver *N* times with
    shared weights on *N* separate inputs, but everything happens in a single
    attention call (batch = B, seq = N * group_seq_len) so the GPU sees fewer,
    larger kernels.

    When *use_flex_attention* is True, uses ``flex_attention`` with compiled
    block-sparse masks — this produces a Triton kernel whose backward pass is
    significantly faster than the memory-efficient SDPA backward triggered by
    boolean masks.
    """

    def __init__(
        self,
        latent_tokens: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        n_groups: int,
        group_seq_len: int,
        use_flex_attention: bool = False,
        normformer_norms: bool = False,
        qk_norm: bool = False,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.latent_tokens = latent_tokens
        self.use_flex_attention = use_flex_attention
        self.latents = nn.Parameter(torch.randn(latent_tokens, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                _PerceiverLayer(
                    d_model, n_heads, dropout, normformer_norms, qk_norm, ff_mult
                )
                for _ in range(n_layers)
            ]
        )
        self.output_dim = latent_tokens * d_model

        total_q = n_groups * latent_tokens
        total_kv = n_groups * group_seq_len

        if use_flex_attention:
            lt = latent_tokens
            gs = group_seq_len

            def cross_mask_mod(b, h, q_idx, kv_idx):
                return (q_idx // lt) == (kv_idx // gs)

            def self_mask_mod(b, h, q_idx, kv_idx):
                return (q_idx // lt) == (kv_idx // lt)

            self._cross_block_mask = create_block_mask(
                cross_mask_mod,
                B=None,
                H=None,
                Q_LEN=total_q,
                KV_LEN=total_kv,
                device="cuda",
            )
            self._self_block_mask = create_block_mask(
                self_mask_mod,
                B=None,
                H=None,
                Q_LEN=total_q,
                KV_LEN=total_q,
                device="cuda",
            )
            self._cross_mask = None
            self._self_mask = None
        else:
            # SDPA bool convention: True = allowed to attend, False = masked out
            cross_mask = torch.zeros(total_q, total_kv, dtype=torch.bool)
            self_mask = torch.zeros(total_q, total_q, dtype=torch.bool)
            for i in range(n_groups):
                qs, qe = i * latent_tokens, (i + 1) * latent_tokens
                kvs, kve = i * group_seq_len, (i + 1) * group_seq_len
                cross_mask[qs:qe, kvs:kve] = True
                self_mask[qs:qe, qs:qe] = True

            self.register_buffer("_cross_mask", cross_mask)
            self.register_buffer("_self_mask", self_mask)
            self._cross_block_mask = None
            self._self_block_mask = None

    def forward(self, x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """
        Args:
            x: ``(B, n_groups * group_seq_len, d_model)`` — all groups concatenated.
        Returns:
            If *flatten*: ``(B, n_groups, latent_tokens * d_model)``
            Else: ``(B, n_groups, latent_tokens, d_model)``
        """
        B = x.shape[0]
        latents = self.latents.repeat(self.n_groups, 1)
        latents = latents.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            latents = layer(
                latents,
                x,
                cross_mask=self._cross_mask,
                self_mask=self._self_mask,
                cross_block_mask=self._cross_block_mask,
                self_block_mask=self._self_block_mask,
            )

        latents = latents.unflatten(1, (self.n_groups, self.latent_tokens))
        if flatten:
            return latents.flatten(2)
        return latents


@gin.configurable
class MetamonGroupedTstepEncoderV2(amago.nets.tstep_encoders.TstepEncoder):
    """Timestep encoder for GroupedObservationSpace.

    Three-stage architecture:
        1. Pokemon perceiver (shared): encodes each of 7 Pokemon independently
        2. Global perceiver: encodes misc features (format, conditions, etc.) + rl2
        3. Fusion perceiver: combines 8 entity embeddings into final representation

    Slightly optimized by some fancy attention masking tricks.
    """

    POKEMON_TEXT_LEN = 12
    POKEMON_NUM_LEN = 31
    MISC_TEXT_LEN = 20
    MISC_NUM_LEN = 4
    NUM_POKEMON = 7

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        # Pokemon encoder
        d_pokemon: int = 64,
        n_heads_pokemon: int = 4,
        n_layers_pokemon: int = 2,
        latent_tokens_pokemon: int = 4,
        numerical_tokens_pokemon: int = 4,
        pokemon_out_norm: str = "layer",
        # Global encoder
        d_global: int = 64,
        n_heads_global: int = 4,
        n_layers_global: int = 2,
        latent_tokens_global: int = 4,
        numerical_tokens_global: int = 2,
        global_out_norm: str = "layer",
        # Fusion encoder
        d_fusion: int = 128,
        n_heads_fusion: int = 4,
        n_layers_fusion: int = 2,
        latent_tokens_fusion: int = 4,
        fusion_out_norm: str = "layer",
        # General
        extra_emb_dim: int = 16,
        dropout: float = 0.05,
        use_flex_attention: bool = False,
        normformer_norms: bool = False,
        qk_norm: bool = False,
        ff_mult: int = 4,
        pokemon_role_emb: bool = False,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)

        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)

        # --- Pokemon encoder (shared for all 7, block-diagonal masking) ---
        self.pokemon_token_emb = TokenEmbedding(tokenizer, d_pokemon)
        self.pokemon_fuse = MultiModalEmbedding(
            token_emb_dim=d_pokemon,
            numerical_d_inp=self.POKEMON_NUM_LEN,
            output_dim=d_pokemon,
            numerical_tokens=numerical_tokens_pokemon,
            dropout=dropout,
        )
        pokemon_seq_len = self.POKEMON_TEXT_LEN + numerical_tokens_pokemon
        self.pokemon_pos = LearnablePosEmb(max_len=pokemon_seq_len, d_model=d_pokemon)
        self.pokemon_perceiver = _BlockDiagPerceiverEncoder(
            latent_tokens=latent_tokens_pokemon,
            d_model=d_pokemon,
            n_heads=n_heads_pokemon,
            n_layers=n_layers_pokemon,
            dropout=dropout,
            n_groups=self.NUM_POKEMON,
            group_seq_len=pokemon_seq_len,
            use_flex_attention=use_flex_attention,
            normformer_norms=normformer_norms,
            qk_norm=qk_norm,
            ff_mult=ff_mult,
        )
        self.pokemon_out_norm = Normalization(pokemon_out_norm, d_pokemon)
        self.pokemon_proj = nn.Linear(latent_tokens_pokemon * d_pokemon, d_fusion)
        self.register_buffer(
            "_pokemon_pos_ids",
            torch.arange(pokemon_seq_len, dtype=torch.long),
        )
        self._pokemon_role_emb = (
            nn.Embedding(3, d_pokemon) if pokemon_role_emb else None
        )
        if pokemon_role_emb:
            # 0 = player active, 1 = bench/switch, 2 = opponent active
            self.register_buffer(
                "_pokemon_role_ids",
                torch.tensor([0, 1, 1, 1, 1, 1, 2], dtype=torch.long),
            )

        # --- Global encoder ---
        self.global_token_emb = TokenEmbedding(tokenizer, d_global)
        self.global_fuse = MultiModalEmbedding(
            token_emb_dim=d_global,
            numerical_d_inp=self.MISC_NUM_LEN + extra_emb_dim,
            output_dim=d_global,
            numerical_tokens=numerical_tokens_global,
            dropout=dropout,
        )
        global_seq_len = self.MISC_TEXT_LEN + numerical_tokens_global
        self.global_pos = LearnablePosEmb(max_len=global_seq_len, d_model=d_global)
        self.global_perceiver = _FastPerceiverEncoder(
            latent_tokens=latent_tokens_global,
            d_model=d_global,
            n_heads=n_heads_global,
            n_layers=n_layers_global,
            dropout=dropout,
            normformer_norms=normformer_norms,
            qk_norm=qk_norm,
            ff_mult=ff_mult,
        )
        self.global_out_norm = Normalization(global_out_norm, d_global)
        self.global_proj = nn.Linear(latent_tokens_global * d_global, d_fusion)
        self.register_buffer(
            "_global_pos_ids", torch.arange(global_seq_len, dtype=torch.long)
        )

        # --- Fusion encoder ---
        self.entity_type_emb = nn.Embedding(self.NUM_POKEMON + 1, d_fusion)
        self.fusion = _FastPerceiverEncoder(
            latent_tokens=latent_tokens_fusion,
            d_model=d_fusion,
            n_heads=n_heads_fusion,
            n_layers=n_layers_fusion,
            dropout=dropout,
            normformer_norms=normformer_norms,
            qk_norm=qk_norm,
            ff_mult=ff_mult,
        )
        self.fusion_out_norm = Normalization(fusion_out_norm, d_fusion)
        self.register_buffer(
            "_entity_type_ids", torch.arange(self.NUM_POKEMON + 1, dtype=torch.long)
        )

        self._emb_dim = self.fusion.output_dim

    @property
    def emb_dim(self):
        return self._emb_dim

    def inner_forward(self, obs, rl2s, log_dict=None):
        pokemon_text = torch.stack(
            [
                obs["text_active_pokemon_tokens"],
                obs["text_switch_0_tokens"],
                obs["text_switch_1_tokens"],
                obs["text_switch_2_tokens"],
                obs["text_switch_3_tokens"],
                obs["text_switch_4_tokens"],
                obs["text_opponent_active_pokemon_tokens"],
            ],
            dim=2,
        )
        pokemon_nums = torch.stack(
            [
                obs["numbers_active_pokemon"],
                obs["numbers_switch_0"],
                obs["numbers_switch_1"],
                obs["numbers_switch_2"],
                obs["numbers_switch_3"],
                obs["numbers_switch_4"],
                obs["numbers_opponent_active_pokemon"],
            ],
            dim=2,
        )

        B, L = pokemon_text.shape[:2]
        pokemon_text = pokemon_text.flatten(0, 1)
        pokemon_nums = pokemon_nums.flatten(0, 1)
        rl2s_flat = rl2s.flatten(0, 1)
        global_nums_flat = obs["numbers_misc"].flatten(0, 1)
        global_text_flat = obs["text_misc_tokens"].flatten(0, 1)

        emb = self._inner_forward_impl(
            pokemon_text,
            pokemon_nums,
            rl2s_flat,
            global_nums_flat,
            global_text_flat,
            log_dict,
        )
        return emb.unflatten(0, (B, L))

    def _encode_pokemon(
        self, text_tokens: torch.Tensor, numerical: torch.Tensor, log_dict=None
    ) -> torch.Tensor:
        B = text_tokens.size(0)

        # Embed each pokemon independently (shared weights)
        text_flat = text_tokens.flatten(0, 1)
        nums_flat = numerical.flatten(0, 1)

        tok_emb = self.pokemon_token_emb(text_flat)
        tok_emb = tok_emb.unsqueeze(1)
        nums_flat = nums_flat.unsqueeze(1)
        seq = self.pokemon_fuse(tok_emb, nums_flat).squeeze(1)

        seq = seq + self.pokemon_pos(self._pokemon_pos_ids)

        # Concatenate all 7 pokemon into one sequence for block-diagonal attn
        seq = seq.unflatten(0, (-1, self.NUM_POKEMON)).flatten(1, 2)

        if self._pokemon_role_emb is not None:
            role = self._pokemon_role_emb(self._pokemon_role_ids)  # (7, d_pokemon)
            tokens_per_pokemon = seq.shape[1] // self.NUM_POKEMON
            idx = torch.arange(self.NUM_POKEMON, device=seq.device) * tokens_per_pokemon
            role_signal = torch.zeros(
                seq.shape[1], seq.shape[2], device=seq.device, dtype=seq.dtype
            )
            role_signal[idx] = role
            seq = seq + role_signal

        # Block-diagonal perceiver → (B, 7, latent_tokens, d_pokemon)
        emb = self.pokemon_perceiver(seq, flatten=False)
        add_activation_log(
            "MetamonGroupedTstepEncoderV2/pokemon_perceiver", emb, log_dict
        )

        emb = self.pokemon_out_norm(emb)
        emb = emb.flatten(2)
        emb = self.pokemon_proj(emb)
        add_activation_log("MetamonGroupedTstepEncoderV2/pokemon_proj", emb, log_dict)

        return emb

    def _encode_global(
        self, text_tokens: torch.Tensor, numerical: torch.Tensor, log_dict=None
    ) -> torch.Tensor:
        tok_emb = self.global_token_emb(text_tokens)
        tok_emb = tok_emb.unsqueeze(1)
        numerical = numerical.unsqueeze(1)
        seq = self.global_fuse(tok_emb, numerical).squeeze(1)

        seq = seq + self.global_pos(self._global_pos_ids)

        emb = self.global_perceiver(seq, flatten=False)
        add_activation_log(
            "MetamonGroupedTstepEncoderV2/global_perceiver", emb, log_dict
        )

        emb = self.global_out_norm(emb)
        emb = emb.flatten(1)
        emb = self.global_proj(emb)
        add_activation_log("MetamonGroupedTstepEncoderV2/global_proj", emb, log_dict)

        return emb

    @torch.compile
    def _inner_forward_impl(
        self,
        pokemon_text,
        pokemon_nums,
        rl2s_flat,
        global_nums_flat,
        global_text_flat,
        log_dict=None,
    ):
        pokemon_embs = self._encode_pokemon(pokemon_text, pokemon_nums, log_dict)

        extras = F.leaky_relu(self.extra_emb(symlog(rl2s_flat)))
        global_nums = torch.cat([global_nums_flat, extras], dim=-1)
        global_emb = self._encode_global(global_text_flat, global_nums, log_dict)
        all_embs = torch.cat([pokemon_embs, global_emb.unsqueeze(1)], dim=1)

        all_embs = all_embs + self.entity_type_emb(self._entity_type_ids)

        emb = self.fusion(all_embs, flatten=False)
        add_activation_log("MetamonGroupedTstepEncoderV2/fusion", emb, log_dict)

        emb = self.fusion_out_norm(emb)
        add_activation_log(
            "MetamonGroupedTstepEncoderV2/fusion_out_norm", emb, log_dict
        )

        return emb.flatten(1)


class MetamonAMAGODataset(RLDataset):
    """A wrapper around the ParsedReplayDataset that converts to an AMAGO RLDataset.

    Args:
        parsed_replay_dset: The ParsedReplayDataset to wrap.
        dset_name: Give the dataset an arbitrary name for logging. Defaults to class name.
        refresh_files_every_epoch: Whether to find newly written replay files at the end of each epoch.
            This imitates the behavior of the main AMAGO disk replay buffer. Would be necessary for
            online RL. Defaults to False.
    """

    def __init__(
        self,
        parsed_replay_dset: ParsedReplayDataset,
        dset_name: Optional[str] = None,
        refresh_files_every_epoch: bool = False,
    ):
        super().__init__(dset_name=dset_name)
        self.parsed_replay_dset = parsed_replay_dset
        self.refresh_files_every_epoch = refresh_files_every_epoch

    @property
    def save_new_trajs_to(self):
        # disables AMAGO's trajetory saving; metamon
        # will handle this in its own replay format.
        return None

    def on_end_of_collection(self, experiment) -> dict[str, Any]:
        # TODO: implement FIFO replay buffer
        if self.refresh_files_every_epoch:
            self.parsed_replay_dset.refresh_files()
        return {"Num Replays": len(self.parsed_replay_dset)}

    def get_description(self) -> str:
        return f"Metamon Replay Dataset ({self.dset_name})"

    def sample_random_trajectory(self) -> RLData:
        data = self.parsed_replay_dset.random_sample()
        return self._process_data(data)

    def _process_data(self, data):
        if len(data) == 5:
            obs, action_infos, rewards, dones, belief_targets = data
        else:
            obs, action_infos, rewards, dones = data
            belief_targets = None
        # amago expects discrete actions to be one-hot encoded
        num_actions = self.parsed_replay_dset.action_space.gym_space.n
        actions_torch = F.one_hot(
            torch.tensor(action_infos["chosen"]).long().clamp(min=0),
            num_classes=num_actions,
        ).float()

        # set all illegal. needs to be one timestep longer than the actions to match the size of observations
        illegal_actions = torch.ones(
            (len(action_infos["chosen"]) + 1, num_actions)
        ).bool()
        for i, legal_actions in enumerate(action_infos["legal"]):
            for legal_action in legal_actions:
                legal_universal_action = UniversalAction(action_idx=legal_action)
                # discrete action spaces don't need a state input...
                legal_agent_action = (
                    self.parsed_replay_dset.action_space.action_to_agent_output(
                        state=None, action=legal_universal_action
                    )
                )
                # set the action legal
                illegal_actions[i, legal_agent_action] = False

        # a bit of a hack: put action info in the amago observation dict, let the network ignore it,
        # and make it accessible to mask the actor/critic loss later on.
        obs_torch = {k: torch.from_numpy(np.stack(v, axis=0)) for k, v in obs.items()}
        if belief_targets is not None:
            obs_torch.update({k: v for k, v in belief_targets.items()})
        # add a final missing action to match the size of observations
        missing_acts = torch.tensor(action_infos["missing"] + [True]).unsqueeze(-1)
        obs_torch["missing_action_mask"] = missing_acts
        # the environment wrappers also add illegal_actions to the obs
        obs_torch["illegal_actions"] = illegal_actions
        rewards_torch = torch.from_numpy(rewards).unsqueeze(-1)
        dones_torch = torch.from_numpy(dones).unsqueeze(-1)
        time_idxs = torch.arange(len(action_infos["chosen"]) + 1).long().unsqueeze(-1)
        rl_data = RLData(
            obs=obs_torch,
            actions=actions_torch,
            rews=rewards_torch,
            dones=dones_torch,
            time_idxs=time_idxs,
        )
        return rl_data


@gin.configurable
class MetamonAMAGOExperiment(amago.Experiment):
    """
    Adds actions masking to the main AMAGO experiment, and leaves room for further tweaks.
    """

    def __init__(
        self,
        *args,
        critic_loss_weight: Optional[float] = None,
        use_dynamic_damping: bool = False,
        kl_coef_init: float = 0.05,
        kl_coef_max: float = 0.5,
        kl_power_alpha: float = 0.0,
        kl_schedule_steps: int = 1_000_000,
        ent_coef_init: float = 0.0,
        ent_coef_min: float = 0.0,
        ent_power_alpha: float = 0.0,
        ent_schedule_steps: int = 1_000_000,
        target_kl_per_step: float = 0.02,
        target_kl_final: Optional[float] = None,
        target_kl_schedule_steps: int = 0,
        kl_tolerance: float = 1.5,
        dd_adapt_interval: int = 100,
        lr_shrink_factor: float = 0.5,
        lr_grow_factor: float = 1.05,
        lr_multiplier_min: Optional[float] = None,
        lr_multiplier_max: Optional[float] = 1_000_000.0,
        kl_coef_growth_factor: float = 1.5,
        kl_coef_decay_factor: float = 0.95,
        kl_multiplier_max: Optional[float] = 1_000_000.0,
        min_lr: float = 0.0,
        max_lr: Optional[float] = None,
        lr_decay_steps: int = 0,
        lr_final_multiplier: float = 1.0,
        grad_clip_final: Optional[float] = None,
        grad_clip_schedule_steps: int = 0,
        ratio_clip_low: Optional[float] = None,
        ratio_clip_high: Optional[float] = None,
        ratio_clip_low_final: Optional[float] = None,
        ratio_clip_high_final: Optional[float] = None,
        ratio_clip_schedule_steps: int = 0,
        ratio_clip_penalty_coeff: float = 0.0,
        dd_kl_reference_mode: str = "epoch",
        dd_kl_reference_update_interval: int = 1,
        dd_controller_kl_metric: str = "local_window_kl",
        log_global_anchor_kl: bool = False,
        global_anchor_kl_controls_lr: bool = False,
        global_anchor_kl_controls_coef: bool = False,
        log_policy_health_metrics: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_update_interval: int = 1,
        ema_warmup_steps: int = 0,
        ema_eval_only: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.critic_loss_weight_override = critic_loss_weight

        self.use_dynamic_damping = use_dynamic_damping
        self.kl_coef_init = kl_coef_init
        self.kl_coef_max = kl_coef_max
        self.kl_power_alpha = kl_power_alpha
        self.kl_schedule_steps = kl_schedule_steps
        self.ent_coef_init = ent_coef_init
        self.ent_coef_min = ent_coef_min
        self.ent_power_alpha = ent_power_alpha
        self.ent_schedule_steps = ent_schedule_steps
        self.target_kl_per_step = target_kl_per_step
        self.target_kl_final = target_kl_final
        self.target_kl_schedule_steps = target_kl_schedule_steps
        self.kl_tolerance = kl_tolerance
        self.dd_adapt_interval = max(1, int(dd_adapt_interval))
        self.lr_shrink_factor = lr_shrink_factor
        self.lr_grow_factor = lr_grow_factor
        self.lr_multiplier_min = lr_multiplier_min
        self.lr_multiplier_max = lr_multiplier_max
        self.kl_coef_growth_factor = kl_coef_growth_factor
        self.kl_coef_decay_factor = kl_coef_decay_factor
        self.kl_multiplier_max = kl_multiplier_max
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay_steps = lr_decay_steps
        self.lr_final_multiplier = lr_final_multiplier
        self.grad_clip_final = grad_clip_final
        self.grad_clip_schedule_steps = grad_clip_schedule_steps
        self.ratio_clip_low = ratio_clip_low
        self.ratio_clip_high = ratio_clip_high
        self.ratio_clip_low_final = ratio_clip_low_final
        self.ratio_clip_high_final = ratio_clip_high_final
        self.ratio_clip_schedule_steps = ratio_clip_schedule_steps
        self.ratio_clip_penalty_coeff = ratio_clip_penalty_coeff
        self.dd_kl_reference_mode = dd_kl_reference_mode.lower()
        if self.dd_kl_reference_mode not in {"epoch", "interval"}:
            raise ValueError(
                "dd_kl_reference_mode must be one of {'epoch', 'interval'}; "
                f"got {dd_kl_reference_mode!r}"
            )
        self.dd_kl_reference_update_interval = max(
            1, int(dd_kl_reference_update_interval)
        )
        self.dd_controller_kl_metric = dd_controller_kl_metric.lower()
        if self.dd_controller_kl_metric not in {
            "local_kl",
            "local_window_kl",
            "global_anchor_kl",
        }:
            raise ValueError(
                "dd_controller_kl_metric must be one of "
                "{'local_kl', 'local_window_kl', 'global_anchor_kl'}; "
                f"got {dd_controller_kl_metric!r}"
            )
        self.log_global_anchor_kl = log_global_anchor_kl
        self.global_anchor_kl_controls_lr = global_anchor_kl_controls_lr
        self.global_anchor_kl_controls_coef = global_anchor_kl_controls_coef
        self.log_policy_health_metrics = log_policy_health_metrics

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_interval = max(1, int(ema_update_interval))
        self.ema_warmup_steps = ema_warmup_steps
        self.ema_eval_only = ema_eval_only

        self._dd_kl_window = deque(maxlen=self.dd_adapt_interval)
        self._dd_global_anchor_kl_window = deque(maxlen=self.dd_adapt_interval)
        self._dd_kl_multiplier = 1.0
        self._dd_lr_multiplier = 1.0
        self._dd_last_window_kl: Optional[float] = None
        self._dd_last_local_kl: Optional[float] = None
        self._dd_last_local_window_kl: Optional[float] = None
        self._dd_last_global_anchor_kl: Optional[float] = None
        self._dd_last_global_anchor_window_kl: Optional[float] = None
        self._dd_last_action = "init"
        self._dd_lr_multiplier_clip_count = 0
        self._dd_kl_multiplier_clip_count = 0
        self._dd_nonfinite_kl_count = 0
        self._update_control_ref: Optional[nn.ModuleDict] = None
        self._global_anchor_ref: Optional[nn.ModuleDict] = None
        self._fixed_base_anchor_ref: Optional[nn.ModuleDict] = None
        self._dd_local_ref_refresh_count = 0
        self._dd_global_anchor_refresh_count = 0
        self._dd_fixed_base_anchor_refresh_count = 0
        self._dd_last_local_ref_step = 0
        self._dd_last_global_anchor_step = 0
        self._dd_last_fixed_base_anchor_step = 0
        self._dd_last_fixed_base_anchor_kl: Optional[float] = None
        self._dd_last_fixed_base_anchor_eval_kl: Optional[float] = None
        self._ema_policy_heads: Optional[nn.ModuleDict] = None

    def start(self):
        super().start()

    @staticmethod
    def _linear_schedule(start: float, end: float, steps: int, step: int) -> float:
        if steps <= 0:
            return start
        pct = min(max(step / steps, 0.0), 1.0)
        return start + pct * (end - start)

    def _current_target_kl(self) -> float:
        target_final = (
            self.target_kl_per_step
            if self.target_kl_final is None
            else self.target_kl_final
        )
        return self._linear_schedule(
            self.target_kl_per_step,
            target_final,
            self.target_kl_schedule_steps,
            self.grad_update_counter,
        )

    def _current_kl_coef(self) -> float:
        self._sanitize_dynamic_damping_multipliers()
        step = self.grad_update_counter
        scale = (1.0 + step / max(1, self.kl_schedule_steps)) ** (
            -self.kl_power_alpha
        )
        return min(self.kl_coef_init * scale * self._dd_kl_multiplier, self.kl_coef_max)

    def _current_ent_coef(self) -> float:
        step = self.grad_update_counter
        scale = (1.0 + step / max(1, self.ent_schedule_steps)) ** (
            -self.ent_power_alpha
        )
        return max(self.ent_coef_init * scale, self.ent_coef_min)

    def _current_ratio_clip(self) -> Optional[tuple[float, float]]:
        if self.ratio_clip_low is None or self.ratio_clip_high is None:
            return None
        low_final = (
            self.ratio_clip_low
            if self.ratio_clip_low_final is None
            else self.ratio_clip_low_final
        )
        high_final = (
            self.ratio_clip_high
            if self.ratio_clip_high_final is None
            else self.ratio_clip_high_final
        )
        return (
            self._linear_schedule(
                self.ratio_clip_low,
                low_final,
                self.ratio_clip_schedule_steps,
                self.grad_update_counter,
            ),
            self._linear_schedule(
                self.ratio_clip_high,
                high_final,
                self.ratio_clip_schedule_steps,
                self.grad_update_counter,
            ),
        )

    def _current_grad_clip(self) -> float:
        if self.grad_clip_final is None:
            return self.grad_clip
        return self._linear_schedule(
            self.grad_clip,
            self.grad_clip_final,
            self.grad_clip_schedule_steps,
            self.grad_update_counter,
        )

    def _current_lr_decay_multiplier(self) -> float:
        return self._linear_schedule(
            1.0,
            self.lr_final_multiplier,
            self.lr_decay_steps,
            self.grad_update_counter,
        )

    def _sanitize_multiplier(
        self,
        value: float,
        *,
        min_value: Optional[float],
        max_value: Optional[float],
        fallback: float,
    ) -> tuple[float, bool]:
        clipped = False
        if not math.isfinite(value):
            if value == float("inf") and max_value is not None:
                value = max_value
            elif value == float("-inf") and min_value is not None:
                value = min_value
            else:
                value = fallback
            clipped = True
        if min_value is not None and value < min_value:
            value = min_value
            clipped = True
        if max_value is not None and value > max_value:
            value = max_value
            clipped = True
        return value, clipped

    def _sanitize_dynamic_damping_multipliers(self):
        if not self.use_dynamic_damping:
            return
        lr_multiplier, lr_clipped = self._sanitize_multiplier(
            float(self._dd_lr_multiplier),
            min_value=self.lr_multiplier_min,
            max_value=self.lr_multiplier_max,
            fallback=(
                self.lr_multiplier_min
                if self.lr_multiplier_min is not None
                else 1.0
            ),
        )
        kl_multiplier, kl_clipped = self._sanitize_multiplier(
            float(self._dd_kl_multiplier),
            min_value=0.0,
            max_value=self.kl_multiplier_max,
            fallback=1.0,
        )
        if lr_clipped:
            self._dd_lr_multiplier_clip_count += 1
            self._dd_last_action = f"{self._dd_last_action}+lr_clip"
        if kl_clipped:
            self._dd_kl_multiplier_clip_count += 1
            self._dd_last_action = f"{self._dd_last_action}+kl_clip"
        self._dd_lr_multiplier = lr_multiplier
        self._dd_kl_multiplier = kl_multiplier

    @staticmethod
    def _clone_policy_heads(policy) -> nn.ModuleDict:
        modules = {
            "tstep_encoder": copy.deepcopy(policy.tstep_encoder),
            "traj_encoder": copy.deepcopy(policy.traj_encoder),
            "actor": copy.deepcopy(policy.actor),
        }
        belief_head = getattr(policy, "belief_head", None)
        if belief_head is not None:
            modules["belief_head"] = copy.deepcopy(belief_head)
        heads = nn.ModuleDict(modules)
        heads.eval()
        heads.requires_grad_(False)
        return heads

    def _load_policy_heads_from(self, heads: nn.ModuleDict):
        policy = self.policy
        policy.tstep_encoder.load_state_dict(heads["tstep_encoder"].state_dict())
        policy.traj_encoder.load_state_dict(heads["traj_encoder"].state_dict())
        policy.actor.load_state_dict(heads["actor"].state_dict())
        if "belief_head" in heads and getattr(policy, "belief_head", None) is not None:
            policy.belief_head.load_state_dict(heads["belief_head"].state_dict())

    @contextmanager
    def _policy_heads_eval_mode(self):
        modules = [
            self.policy.tstep_encoder,
            self.policy.traj_encoder,
            self.policy.actor,
        ]
        belief_head = getattr(self.policy, "belief_head", None)
        if belief_head is not None:
            modules.append(belief_head)
        previous_modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, was_training in zip(modules, previous_modes):
                module.train(was_training)

    def _refresh_update_control_reference(self):
        if not self.use_dynamic_damping:
            return
        self._update_control_ref = self._clone_policy_heads(self.policy)
        self._dd_local_ref_refresh_count += 1
        self._dd_last_local_ref_step = int(self.grad_update_counter)
        self._dd_kl_window.clear()

    def _refresh_global_anchor_reference(self):
        if not self.use_dynamic_damping:
            return
        if not (
            self.log_global_anchor_kl
            or self.global_anchor_kl_controls_lr
            or self.global_anchor_kl_controls_coef
            or self.dd_controller_kl_metric == "global_anchor_kl"
        ):
            return
        self._global_anchor_ref = self._clone_policy_heads(self.policy)
        self._dd_global_anchor_refresh_count += 1
        self._dd_last_global_anchor_step = int(self.grad_update_counter)
        self._dd_global_anchor_kl_window.clear()

    def _refresh_fixed_base_anchor_reference(self):
        self._fixed_base_anchor_ref = self._clone_policy_heads(self.policy)
        self._dd_fixed_base_anchor_refresh_count += 1
        self._dd_last_fixed_base_anchor_step = int(self.grad_update_counter)
        self._dd_last_fixed_base_anchor_kl = None
        self._dd_last_fixed_base_anchor_eval_kl = None

    def _maybe_refresh_update_control_reference_after_update(self):
        if not self.use_dynamic_damping:
            return
        if self.dd_kl_reference_mode != "interval":
            return
        if self.grad_update_counter <= 0:
            return
        if self.grad_update_counter % self.dd_kl_reference_update_interval == 0:
            self._refresh_update_control_reference()

    def _init_ema_policy_heads(self):
        if self.use_ema and self._ema_policy_heads is None:
            self._ema_policy_heads = self._clone_policy_heads(self.policy)

    @torch.no_grad()
    def _update_ema_policy_heads(self):
        if not self.use_ema:
            return
        if self.grad_update_counter < self.ema_warmup_steps:
            return
        if self.grad_update_counter % self.ema_update_interval != 0:
            return
        self._init_ema_policy_heads()
        assert self._ema_policy_heads is not None
        policy_heads = {
            "tstep_encoder": self.policy.tstep_encoder,
            "traj_encoder": self.policy.traj_encoder,
            "actor": self.policy.actor,
        }
        alpha = 1.0 - self.ema_decay
        for name, ema_module in self._ema_policy_heads.items():
            current_state = policy_heads[name].state_dict()
            ema_state = ema_module.state_dict()
            for key, ema_tensor in ema_state.items():
                current_tensor = current_state[key].detach()
                if ema_tensor.is_floating_point():
                    ema_tensor.lerp_(current_tensor, alpha)
                else:
                    ema_tensor.copy_(current_tensor)

    @contextmanager
    def _maybe_ema_eval_weights(self):
        if (
            not self.use_ema
            or not self.ema_eval_only
            or self._ema_policy_heads is None
        ):
            yield
            return
        current_heads = self._clone_policy_heads(self.policy)
        try:
            self._load_policy_heads_from(self._ema_policy_heads)
            yield
        finally:
            self._load_policy_heads_from(current_heads)

    def _save_ema_checkpoint(self):
        if not self.use_ema or self._ema_policy_heads is None:
            return
        if not self.accelerator.is_main_process:
            return
        ema_dir = os.path.join(self.ckpt_dir, "ema_policy_heads")
        os.makedirs(ema_dir, exist_ok=True)
        torch.save(
            self._ema_policy_heads.state_dict(),
            os.path.join(ema_dir, f"policy_epoch_{self.epoch}.pt"),
        )

    def _apply_lr_controls(self):
        if not hasattr(self, "optimizer") or not hasattr(self, "lr_schedule"):
            return
        self._sanitize_dynamic_damping_multipliers()
        scheduled_lrs = self.lr_schedule.get_last_lr()
        decay = self._current_lr_decay_multiplier()
        for group, scheduled_lr in zip(self.optimizer.param_groups, scheduled_lrs):
            lr = scheduled_lr * self._dd_lr_multiplier * decay
            lr = max(lr, self.min_lr)
            if self.max_lr is not None:
                lr = min(lr, self.max_lr)
            group["lr"] = lr

    def _control_action_for_kl(self, window_kl: float) -> str:
        target = max(self._current_target_kl(), 1e-12)
        if window_kl > self.kl_tolerance * target:
            return "brake"
        if window_kl < target / self.kl_tolerance:
            return "relax"
        return "hold"

    def _adapt_update_control(
        self,
        observed_local_kl: Optional[torch.Tensor],
        observed_global_anchor_kl: Optional[torch.Tensor] = None,
    ):
        if not self.use_dynamic_damping or observed_local_kl is None:
            return
        local_kl_value = float(observed_local_kl.detach().float().cpu())
        global_kl_value = (
            None
            if observed_global_anchor_kl is None
            else float(observed_global_anchor_kl.detach().float().cpu())
        )
        self._dd_last_local_kl = local_kl_value if math.isfinite(local_kl_value) else None
        self._dd_last_global_anchor_kl = (
            global_kl_value
            if global_kl_value is not None and math.isfinite(global_kl_value)
            else None
        )

        controller_kl_value = (
            global_kl_value
            if self.dd_controller_kl_metric == "global_anchor_kl"
            else local_kl_value
        )
        if controller_kl_value is None or not math.isfinite(controller_kl_value):
            self._dd_nonfinite_kl_count += 1
            self._dd_last_action = "invalid_kl"
            self._sanitize_dynamic_damping_multipliers()
            return
        if not math.isfinite(local_kl_value):
            self._dd_nonfinite_kl_count += 1
            self._dd_last_action = "invalid_kl"
            self._sanitize_dynamic_damping_multipliers()
            return

        self._dd_kl_window.append(local_kl_value)
        if global_kl_value is not None and math.isfinite(global_kl_value):
            self._dd_global_anchor_kl_window.append(global_kl_value)

        if len(self._dd_kl_window) < self.dd_adapt_interval:
            return
        if self.grad_update_counter % self.dd_adapt_interval != 0:
            return

        local_window_kl = sum(self._dd_kl_window) / len(self._dd_kl_window)
        self._dd_last_local_window_kl = local_window_kl
        global_window_kl = None
        if self._dd_global_anchor_kl_window:
            global_window_kl = sum(self._dd_global_anchor_kl_window) / len(
                self._dd_global_anchor_kl_window
            )
            self._dd_last_global_anchor_window_kl = global_window_kl

        controller_window_kl = (
            global_window_kl
            if self.dd_controller_kl_metric == "global_anchor_kl"
            and global_window_kl is not None
            else local_window_kl
        )
        self._dd_last_window_kl = controller_window_kl
        controller_action = self._control_action_for_kl(controller_window_kl)

        lr_window_kl = (
            global_window_kl
            if self.global_anchor_kl_controls_lr and global_window_kl is not None
            else controller_window_kl
        )
        kl_coef_window_kl = (
            global_window_kl
            if self.global_anchor_kl_controls_coef and global_window_kl is not None
            else controller_window_kl
        )
        lr_action = self._control_action_for_kl(lr_window_kl)
        kl_coef_action = self._control_action_for_kl(kl_coef_window_kl)

        if lr_action == "brake":
            self._dd_lr_multiplier *= self.lr_shrink_factor
        elif lr_action == "relax":
            self._dd_lr_multiplier *= self.lr_grow_factor

        if kl_coef_action == "brake":
            self._dd_kl_multiplier *= self.kl_coef_growth_factor
        elif kl_coef_action == "relax":
            self._dd_kl_multiplier *= self.kl_coef_decay_factor

        self._dd_last_action = controller_action
        self._sanitize_dynamic_damping_multipliers()

    def _update_control_state_info(self) -> dict[str, Any]:
        if not self.use_dynamic_damping:
            return {
                "Damping/Enabled": 0.0,
                "Damping/LR Decay Multiplier": self._current_lr_decay_multiplier(),
            }
        self._sanitize_dynamic_damping_multipliers()
        ratio_clip = self._current_ratio_clip()
        base_action = self._dd_last_action.split("+", maxsplit=1)[0]
        action_code = {
            "relax": -1.0,
            "hold": 0.0,
            "brake": 1.0,
            "invalid_kl": 2.0,
        }.get(base_action, 0.0)
        info = {
            "Damping/Enabled": 1.0,
            "Damping/KL Coefficient": self._current_kl_coef(),
            "Damping/Entropy Coefficient": self._current_ent_coef(),
            "Damping/Target KL": self._current_target_kl(),
            "Damping/Local KL Target": self._current_target_kl(),
            "Damping/LR Multiplier": self._dd_lr_multiplier,
            "Damping/LR Multiplier Min": (
                0.0 if self.lr_multiplier_min is None else self.lr_multiplier_min
            ),
            "Damping/LR Decay Multiplier": self._current_lr_decay_multiplier(),
            "Damping/KL Multiplier": self._dd_kl_multiplier,
            "Damping/LR Multiplier Clips": self._dd_lr_multiplier_clip_count,
            "Damping/KL Multiplier Clips": self._dd_kl_multiplier_clip_count,
            "Damping/Nonfinite KL Count": self._dd_nonfinite_kl_count,
            "Damping/Window KL": (
                0.0 if self._dd_last_window_kl is None else self._dd_last_window_kl
            ),
            "Damping/Local Window KL": (
                0.0
                if self._dd_last_local_window_kl is None
                else self._dd_last_local_window_kl
            ),
            "Damping/Last Local KL": (
                0.0 if self._dd_last_local_kl is None else self._dd_last_local_kl
            ),
            "Damping/Global Anchor Window KL": (
                0.0
                if self._dd_last_global_anchor_window_kl is None
                else self._dd_last_global_anchor_window_kl
            ),
            "Damping/Last Global Anchor KL": (
                0.0
                if self._dd_last_global_anchor_kl is None
                else self._dd_last_global_anchor_kl
            ),
            "Damping/Local Reference Refreshes": self._dd_local_ref_refresh_count,
            "Damping/Local Reference Step": self._dd_last_local_ref_step,
            "Damping/Global Anchor Refreshes": self._dd_global_anchor_refresh_count,
            "Damping/Global Anchor Step": self._dd_last_global_anchor_step,
            "Retention/Base Anchor Refreshes": self._dd_fixed_base_anchor_refresh_count,
            "Retention/Base Anchor Step": self._dd_last_fixed_base_anchor_step,
            "Retention/Last Base Anchor KL": (
                0.0
                if self._dd_last_fixed_base_anchor_kl is None
                else self._dd_last_fixed_base_anchor_kl
            ),
            "Retention/Last Base Anchor Eval KL": (
                0.0
                if self._dd_last_fixed_base_anchor_eval_kl is None
                else self._dd_last_fixed_base_anchor_eval_kl
            ),
            "Retention/Fixed Base Anchor Refreshes": (
                self._dd_fixed_base_anchor_refresh_count
            ),
            "Retention/Fixed Base Anchor Step": self._dd_last_fixed_base_anchor_step,
            "Retention/Last Fixed Base Anchor KL": (
                0.0
                if self._dd_last_fixed_base_anchor_kl is None
                else self._dd_last_fixed_base_anchor_kl
            ),
            "Retention/Last Fixed Base Anchor Eval KL": (
                0.0
                if self._dd_last_fixed_base_anchor_eval_kl is None
                else self._dd_last_fixed_base_anchor_eval_kl
            ),
            "Damping/Reference Mode Interval": (
                1.0 if self.dd_kl_reference_mode == "interval" else 0.0
            ),
            "Damping/Reference Update Interval": (
                self.dd_kl_reference_update_interval
            ),
            "Damping/Global Anchor Controls LR": (
                1.0 if self.global_anchor_kl_controls_lr else 0.0
            ),
            "Damping/Global Anchor Controls Coef": (
                1.0 if self.global_anchor_kl_controls_coef else 0.0
            ),
            "Damping/Action Code": action_code,
        }
        if ratio_clip is not None:
            info["Damping/Ratio Clip Low"] = ratio_clip[0]
            info["Damping/Ratio Clip High"] = ratio_clip[1]
        return info

    def _masked_zero(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)

    @staticmethod
    def _renormalize_probs_over_legal_actions(
        probs: torch.Tensor, legal_action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tiny = torch.finfo(probs.dtype).tiny
        legal = legal_action_mask.to(dtype=probs.dtype)
        masked_probs = probs * legal
        masked_probs = masked_probs / masked_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(tiny)
        log_probs = torch.log(masked_probs.clamp_min(tiny))
        masked_probs = torch.where(
            legal_action_mask, masked_probs, torch.zeros_like(masked_probs)
        )
        log_probs = torch.where(
            legal_action_mask, log_probs, torch.zeros_like(log_probs)
        )
        return masked_probs, log_probs

    @staticmethod
    def _reverse_kl_from_log_probs(
        current_probs: torch.Tensor,
        current_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        return (
            current_probs * (current_log_probs - ref_log_probs)
        ).sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def _reference_policy_probs(
        self,
        ref_heads: nn.ModuleDict,
        batch: Batch,
        straight_from_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ref_o = ref_heads["tstep_encoder"](obs=batch.obs, rl2s=batch.rl2s)
        ref_s, _ = ref_heads["traj_encoder"](
            seq=ref_o, time_idxs=batch.time_idxs, hidden_state=None
        )
        if "belief_head" in ref_heads:
            belief_outputs = ref_heads["belief_head"](ref_s)
            ref_s = torch.cat((ref_s, belief_outputs.actor_embedding), dim=-1)
        ref_dist = ref_heads["actor"](
            ref_s,
            straight_from_obs=straight_from_obs,
        )
        return ref_dist.probs

    @torch.no_grad()
    def _current_policy_eval_probs(
        self,
        batch: Batch,
        straight_from_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        with self._policy_heads_eval_mode():
            eval_o = self.policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            eval_s, _ = self.policy.traj_encoder(
                seq=eval_o, time_idxs=batch.time_idxs, hidden_state=None
            )
            if hasattr(self.policy, "actor_state_for_policy"):
                eval_s = self.policy.actor_state_for_policy(eval_s)
            eval_dist = self.policy.actor(
                eval_s,
                straight_from_obs=straight_from_obs,
            )
        return eval_dist.probs

    def _compute_update_control_loss(
        self, batch: Batch, log_step: bool
    ) -> tuple[
        torch.Tensor,
        dict[str, Any],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        should_run = self.use_dynamic_damping or (
            log_step and self.log_policy_health_metrics
        )
        device = batch.rl2s.device
        if not should_run:
            return self._masked_zero(device), {}, None, None
        if self.use_dynamic_damping and self._update_control_ref is None:
            self._refresh_update_control_reference()
        if (
            self.use_dynamic_damping
            and self._global_anchor_ref is None
            and (
                self.log_global_anchor_kl
                or self.global_anchor_kl_controls_lr
                or self.global_anchor_kl_controls_coef
                or self.dd_controller_kl_metric == "global_anchor_kl"
            )
        ):
            self._refresh_global_anchor_reference()

        policy = self.policy
        straight_from_obs = {k: batch.obs[k] for k in policy.pass_obs_keys_to_actor}
        current_o = policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        current_s, _ = policy.traj_encoder(
            seq=current_o, time_idxs=batch.time_idxs, hidden_state=None
        )
        current_actor_s = (
            policy.actor_state_for_policy(current_s)
            if hasattr(policy, "actor_state_for_policy")
            else current_s
        )
        current_dist = policy.actor(
            current_actor_s,
            straight_from_obs=straight_from_obs,
        )

        current_probs = current_dist.probs
        current_log_probs = torch.log(
            current_probs.clamp_min(torch.finfo(current_probs.dtype).tiny)
        )
        B, L, G, N = current_probs.shape
        state_mask = (~((batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True))).bool()[
            :, 1:, ...
        ]
        valid_mask = einops.repeat(state_mask, "b l 1 -> b l g 1", g=G)
        valid_mask = policy.edit_actor_mask(
            batch,
            torch.zeros((B, L - 1, G, 1), device=device),
            valid_mask,
        )

        entropy_elems = -(current_probs * current_log_probs).sum(
            dim=-1, keepdim=True
        )[:, :-1, ...]
        entropy_mean = amago.utils.masked_avg(entropy_elems, valid_mask)
        sorted_log_probs = torch.topk(current_log_probs[:, :-1, ...], 2, dim=-1).values
        logit_margin = (sorted_log_probs[..., 0] - sorted_log_probs[..., 1]).unsqueeze(
            -1
        )
        top1_prob = current_probs[:, :-1, ...].max(dim=-1, keepdim=True).values
        no_options = batch.obs["illegal_actions"][:, :-1, :].all(
            dim=-1, keepdim=True
        )
        raw_illegal = batch.obs["illegal_actions"][:, :-1, :].unsqueeze(-2).expand(
            B, L - 1, G, N
        )
        no_options_expanded = no_options.unsqueeze(-2)
        illegal = torch.logical_and(raw_illegal, ~no_options_expanded)
        legal_action_mask = torch.logical_or(~raw_illegal, no_options_expanded)
        invalid_mass = (current_probs[:, :-1, ...] * illegal.float()).sum(
            dim=-1, keepdim=True
        )

        metrics: dict[str, Any] = {}
        total_control_loss = self._masked_zero(device)
        observed_local_kl = None
        observed_global_anchor_kl = None
        observed_fixed_base_anchor_kl = None
        fixed_base_kl_elems = None
        local_current_probs = None
        local_current_log_probs = None

        needs_legal_current_probs = (
            self.use_dynamic_damping and self._update_control_ref is not None
        ) or self._fixed_base_anchor_ref is not None
        if needs_legal_current_probs:
            local_current_probs, local_current_log_probs = (
                self._renormalize_probs_over_legal_actions(
                    current_probs[:, :-1, ...], legal_action_mask
                )
            )

        if self.use_dynamic_damping and self._update_control_ref is not None:
            ref_probs = self._reference_policy_probs(
                self._update_control_ref, batch, straight_from_obs
            )
            assert local_current_probs is not None
            assert local_current_log_probs is not None
            _, local_ref_log_probs = (
                self._renormalize_probs_over_legal_actions(
                    ref_probs[:, :-1, ...], legal_action_mask
                )
            )
            kl_elems = self._reverse_kl_from_log_probs(
                local_current_probs, local_current_log_probs, local_ref_log_probs
            )
            kl_mean = amago.utils.masked_avg(kl_elems, valid_mask)
            observed_local_kl = kl_mean.detach()
            total_control_loss = total_control_loss + self._current_kl_coef() * kl_mean
            total_control_loss = total_control_loss - self._current_ent_coef() * entropy_mean

            global_kl_elems = None
            if self._global_anchor_ref is not None:
                global_ref_probs = self._reference_policy_probs(
                    self._global_anchor_ref, batch, straight_from_obs
                )
                _, global_ref_log_probs = self._renormalize_probs_over_legal_actions(
                    global_ref_probs[:, :-1, ...], legal_action_mask
                )
                global_kl_elems = self._reverse_kl_from_log_probs(
                    local_current_probs,
                    local_current_log_probs,
                    global_ref_log_probs,
                )
                observed_global_anchor_kl = amago.utils.masked_avg(
                    global_kl_elems, valid_mask
                ).detach()

            ratio_clip = self._current_ratio_clip()
            if (
                ratio_clip is not None
                and self.ratio_clip_penalty_coeff > 0.0
                and batch.actions.shape[-1] == N
            ):
                actions = batch.actions.clamp(0, 1.0).unsqueeze(-2)
                current_action_prob = (
                    current_probs[:, :-1, ...] * actions
                ).sum(dim=-1, keepdim=True)
                ref_action_prob = (ref_probs[:, :-1, ...] * actions).sum(
                    dim=-1, keepdim=True
                )
                ratio = current_action_prob / ref_action_prob.clamp_min(1e-8)
                low, high = ratio_clip
                ratio_excess = F.relu(ratio - high) + F.relu(low - ratio)
                ratio_clip_loss = self.ratio_clip_penalty_coeff * amago.utils.masked_avg(
                    ratio_excess, valid_mask
                )
                total_control_loss = total_control_loss + ratio_clip_loss
            if log_step:
                    metrics["Ratio Clip Loss"] = ratio_clip_loss.detach()
                    metrics["Ratio Mean"] = amago.utils.masked_avg(
                        ratio, valid_mask
                    ).detach()
                    metrics["Ratio Pct Low"] = amago.utils.masked_avg(
                        (ratio < low).float(), valid_mask
                    ).detach()
                    metrics["Ratio Pct High"] = amago.utils.masked_avg(
                        (ratio > high).float(), valid_mask
                    ).detach()

            if log_step:
                metrics["KL Divergence"] = kl_mean.detach()
                metrics["Damping/Local KL"] = kl_mean.detach()
                metrics["Damping/Local KL Target"] = torch.tensor(
                    self._current_target_kl(), device=device
                )
                if observed_global_anchor_kl is not None:
                    metrics["Damping/Global Anchor KL"] = observed_global_anchor_kl
                metrics["KL P95"] = self._masked_quantile(
                    kl_elems, valid_mask, 0.95
                ).detach()
                metrics["KL P99"] = self._masked_quantile(
                    kl_elems, valid_mask, 0.99
                ).detach()
                if global_kl_elems is not None:
                    metrics["Damping/Global Anchor KL P95"] = self._masked_quantile(
                        global_kl_elems, valid_mask, 0.95
                    ).detach()
                    metrics["Damping/Global Anchor KL P99"] = self._masked_quantile(
                        global_kl_elems, valid_mask, 0.99
                    ).detach()
                metrics["Update Control Loss"] = total_control_loss.detach()

        if self._fixed_base_anchor_ref is not None:
            assert local_current_probs is not None
            assert local_current_log_probs is not None
            fixed_base_ref_probs = self._reference_policy_probs(
                self._fixed_base_anchor_ref, batch, straight_from_obs
            )
            _, fixed_base_ref_log_probs = self._renormalize_probs_over_legal_actions(
                fixed_base_ref_probs[:, :-1, ...], legal_action_mask
            )
            fixed_base_kl_elems = self._reverse_kl_from_log_probs(
                local_current_probs,
                local_current_log_probs,
                fixed_base_ref_log_probs,
            )
            observed_fixed_base_anchor_kl = amago.utils.masked_avg(
                fixed_base_kl_elems, valid_mask
            ).detach()
            fixed_base_kl_value = float(
                observed_fixed_base_anchor_kl.detach().float().cpu()
            )
            self._dd_last_fixed_base_anchor_kl = (
                fixed_base_kl_value if math.isfinite(fixed_base_kl_value) else None
            )
            if log_step:
                metrics["Retention/Base Anchor KL"] = observed_fixed_base_anchor_kl
                metrics["Retention/Fixed Base Anchor KL"] = (
                    observed_fixed_base_anchor_kl
                )
                metrics["Retention/Base Anchor Step"] = (
                    self._dd_last_fixed_base_anchor_step
                )
                metrics["Retention/Fixed Base Anchor Step"] = (
                    self._dd_last_fixed_base_anchor_step
                )
                metrics["Retention/Base Anchor Refreshes"] = (
                    self._dd_fixed_base_anchor_refresh_count
                )
                metrics["Retention/Fixed Base Anchor Refreshes"] = (
                    self._dd_fixed_base_anchor_refresh_count
                )
                metrics["Retention/Base Anchor KL P95"] = self._masked_quantile(
                    fixed_base_kl_elems, valid_mask, 0.95
                ).detach()
                metrics["Retention/Fixed Base Anchor KL P95"] = (
                    self._masked_quantile(fixed_base_kl_elems, valid_mask, 0.95).detach()
                )
                metrics["Retention/Base Anchor KL P99"] = self._masked_quantile(
                    fixed_base_kl_elems, valid_mask, 0.99
                ).detach()
                metrics["Retention/Fixed Base Anchor KL P99"] = (
                    self._masked_quantile(fixed_base_kl_elems, valid_mask, 0.99).detach()
                )
                eval_current_probs = self._current_policy_eval_probs(
                    batch, straight_from_obs
                )
                eval_current_probs, eval_current_log_probs = (
                    self._renormalize_probs_over_legal_actions(
                        eval_current_probs[:, :-1, ...], legal_action_mask
                    )
                )
                eval_base_kl_elems = self._reverse_kl_from_log_probs(
                    eval_current_probs,
                    eval_current_log_probs,
                    fixed_base_ref_log_probs,
                )
                eval_base_kl = amago.utils.masked_avg(
                    eval_base_kl_elems, valid_mask
                ).detach()
                eval_base_kl_value = float(eval_base_kl.detach().float().cpu())
                self._dd_last_fixed_base_anchor_eval_kl = (
                    eval_base_kl_value if math.isfinite(eval_base_kl_value) else None
                )
                metrics["Retention/Base Anchor Eval KL"] = eval_base_kl
                metrics["Retention/Fixed Base Anchor Eval KL"] = eval_base_kl
                metrics["Retention/Base Anchor Eval KL P95"] = self._masked_quantile(
                    eval_base_kl_elems, valid_mask, 0.95
                ).detach()
                metrics["Retention/Fixed Base Anchor Eval KL P95"] = (
                    self._masked_quantile(eval_base_kl_elems, valid_mask, 0.95).detach()
                )
                metrics["Retention/Base Anchor Eval KL P99"] = self._masked_quantile(
                    eval_base_kl_elems, valid_mask, 0.99
                ).detach()
                metrics["Retention/Fixed Base Anchor Eval KL P99"] = (
                    self._masked_quantile(eval_base_kl_elems, valid_mask, 0.99).detach()
                )

        if log_step:
            metrics["Policy Entropy"] = entropy_mean.detach()
            metrics["Policy Entropy P10"] = self._masked_quantile(
                entropy_elems, valid_mask, 0.10
            ).detach()
            metrics["Policy Effective Support"] = entropy_mean.detach().exp()
            metrics["Policy Top-1 Prob"] = amago.utils.masked_avg(
                top1_prob, valid_mask
            ).detach()
            metrics["Policy Top-1 Logit Margin"] = amago.utils.masked_avg(
                logit_margin, valid_mask
            ).detach()
            metrics["Invalid Action Prob Mass"] = amago.utils.masked_avg(
                invalid_mass, valid_mask
            ).detach()

        return total_control_loss, metrics, observed_local_kl, observed_global_anchor_kl

    def log_post_load_pre_train_diagnostics(self) -> dict[str, Any]:
        """Log one KL/health row after external weight load and before updates."""
        if not hasattr(self, "train_dloader"):
            self.init_dloaders()
        if self._fixed_base_anchor_ref is None:
            self._refresh_fixed_base_anchor_reference()
        if self.use_dynamic_damping and self._update_control_ref is None:
            self._refresh_update_control_reference()
        if (
            self.use_dynamic_damping
            and self._global_anchor_ref is None
            and (
                self.log_global_anchor_kl
                or self.global_anchor_kl_controls_lr
                or self.global_anchor_kl_controls_coef
                or self.dd_controller_kl_metric == "global_anchor_kl"
            )
        ):
            self._refresh_global_anchor_reference()

        was_training = self.policy_aclr.training
        self.policy_aclr.eval()
        try:
            batch = next(iter(self.train_dloader))
            with torch.no_grad():
                _, metrics, _, _ = self._compute_update_control_loss(
                    batch, log_step=True
                )
        finally:
            self.policy_aclr.train(was_training)

        def metric_float(name: str) -> float:
            value = metrics.get(name, 0.0)
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().float())
            return float(value)

        diagnostics = {
            "post_load_pre_train_local_kl": metric_float("Damping/Local KL"),
            "post_load_pre_train_global_kl": metric_float(
                "Damping/Global Anchor KL"
            ),
            "post_load_pre_train_base_anchor_kl": metric_float(
                "Retention/Fixed Base Anchor KL"
            ),
            "post_load_pre_train_local_reference_step": self._dd_last_local_ref_step,
            "post_load_pre_train_global_anchor_step": self._dd_last_global_anchor_step,
            "post_load_pre_train_base_anchor_step": (
                self._dd_last_fixed_base_anchor_step
            ),
            "post_load_pre_train_local_reference_refreshes": (
                self._dd_local_ref_refresh_count
            ),
            "post_load_pre_train_global_anchor_refreshes": (
                self._dd_global_anchor_refresh_count
            ),
            "post_load_pre_train_base_anchor_refreshes": (
                self._dd_fixed_base_anchor_refresh_count
            ),
        }
        self.accelerator.print("Post-load/pre-train diagnostics:")
        for name, value in diagnostics.items():
            if isinstance(value, float):
                self.accelerator.print(f"  {name}: {value:.8g}")
            else:
                self.accelerator.print(f"  {name}: {value}")
        self.log(diagnostics, key="post-load-pre-train")
        return diagnostics

    @staticmethod
    def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bool_mask = mask.bool()
        while bool_mask.ndim < values.ndim:
            bool_mask = bool_mask.unsqueeze(-1)
        bool_mask = bool_mask.expand_as(values)
        return values.detach()[bool_mask].float()

    @classmethod
    def _masked_quantile(
        cls, values: torch.Tensor, mask: torch.Tensor, q: float
    ) -> torch.Tensor:
        masked = cls._masked_values(values, mask)
        if masked.numel() == 0:
            return torch.tensor(float("nan"), device=values.device)
        return torch.quantile(masked, q)

    def init_dloaders(self):
        if hasattr(self, "policy_aclr"):
            policy = self.policy
            if hasattr(policy, "refresh_epoch_start_anchor"):
                policy.refresh_epoch_start_anchor()
            self._refresh_global_anchor_reference()
            self._refresh_update_control_reference()
        return super().init_dloaders()

    def _lapras_tauros_eval_enabled(self) -> bool:
        return bool(getattr(self, "lapras_tauros_eval", False))

    def _maybe_eval_lapras_vs_taurosv0(self):
        if not self._lapras_tauros_eval_enabled():
            return
        interval = int(getattr(self, "lapras_tauros_eval_interval_epochs", 1) or 1)
        if interval < 1 or (self.epoch + 1) % interval != 0:
            return
        if not self.accelerator.is_main_process:
            return

        base_model = getattr(self, "lapras_tauros_eval_base_model", None)
        train_gin_config = getattr(
            self, "lapras_tauros_eval_train_gin_config", None
        )
        if base_model is None or train_gin_config is None:
            self.accelerator.print(
                "Skipping Lapras-vs-TaurosV0 eval: local base model or "
                "train gin config was not provided."
            )
            return

        from metamon.rl.evaluate.common import MatchupSpec, PolicySpec, run_matchup_pair
        from metamon.rl.evaluate.results import ResultsTracker

        epoch = int(self.epoch)
        battles = int(getattr(self, "lapras_tauros_eval_battles", 100))
        output_dir = getattr(self, "lapras_tauros_eval_output_dir", None)
        if output_dir is None:
            output_dir = os.path.join(
                self.ckpt_base_dir, self.run_name, "lapras_taurosv0_eval"
            )
        os.makedirs(output_dir, exist_ok=True)

        agent = PolicySpec(
            name=f"{self.run_name}-epoch{epoch}",
            model_name=self.run_name,
            checkpoint=epoch,
            temperature=1.0,
            team_set=getattr(self, "lapras_tauros_eval_agent_team_set", "lapras"),
            battle_backend="metamon",
            local_base_model=base_model,
            local_ckpt_dir=self.ckpt_base_dir,
            local_run_name=self.run_name,
            local_train_gin_config=train_gin_config,
            local_reward_function=getattr(
                self, "lapras_tauros_eval_reward_function", None
            ),
        )
        tauros = PolicySpec(
            name="TaurosV0",
            model_name="TaurosV0",
            checkpoint=None,
            temperature=1.0,
            team_set=getattr(
                self, "lapras_tauros_eval_opponent_team_set", "competitive"
            ),
            battle_backend="metamon",
        )
        matchup = MatchupSpec(
            policy_a=agent,
            policy_b=tauros,
            n_battles=battles,
            battle_format="gen1ou",
        )

        self.accelerator.print(
            f"Running Lapras-vs-TaurosV0 eval after epoch {epoch}: "
            f"{battles} battles"
        )
        pair = run_matchup_pair(
            matchup=matchup,
            gpu_a=0,
            gpu_b=0,
            output_dir=output_dir,
            timeout=int(getattr(self, "lapras_tauros_eval_timeout", 7200)),
            acceptor_startup_delay=float(
                getattr(self, "lapras_tauros_eval_acceptor_startup_delay", 10.0)
            ),
            verbose=False,
            save_trajectories=False,
        )
        if pair.challenger_proc.returncode != 0 or pair.acceptor_proc.returncode != 0:
            self.accelerator.print(
                "Lapras-vs-TaurosV0 eval failed. "
                f"challenger={pair.challenger_proc.returncode}, "
                f"acceptor={pair.acceptor_proc.returncode}"
            )
            self.log(
                {
                    "completed": 0,
                    "epoch": epoch,
                    "requested_battles": battles,
                },
                key="lapras-taurosv0-eval",
            )
            return

        tracker = ResultsTracker(output_dir)
        result = tracker.record_from_results_dir(
            matchup_id=matchup.matchup_id,
            policy_a_name=matchup.policy_a.short_label,
            policy_b_name=matchup.policy_b.short_label,
            results_dir=os.path.join(pair.matchup_dir, "results"),
            challenger_username=pair.challenger_username,
        )
        if result is None:
            self.accelerator.print("Lapras-vs-TaurosV0 eval produced no results.")
            self.log(
                {
                    "completed": 0,
                    "epoch": epoch,
                    "requested_battles": battles,
                },
                key="lapras-taurosv0-eval",
            )
            return

        metrics = {
            "completed": 1,
            "epoch": epoch,
            "requested_battles": battles,
            "total_battles": result.total_battles,
            "lapras_wins": result.policy_a_wins,
            "taurosv0_wins": result.policy_b_wins,
            "lapras_win_rate": result.policy_a_win_rate,
            "taurosv0_win_rate": 1.0 - result.policy_a_win_rate,
        }
        self.log(metrics, key="lapras-taurosv0-eval")
        self.accelerator.print(
            f"Lapras-vs-TaurosV0 eval: {result.policy_a_wins}-"
            f"{result.policy_b_wins} ({result.policy_a_win_rate:.1%})"
        )

    def save_checkpoint(self):
        super().save_checkpoint()
        self._save_ema_checkpoint()
        self._maybe_eval_lapras_vs_taurosv0()
        self.accelerator.wait_for_everyone()

    def _refresh_policy_snapshots_after_load(self) -> None:
        self._refresh_fixed_base_anchor_reference()
        self._refresh_global_anchor_reference()
        self._refresh_update_control_reference()
        if self.use_ema:
            self._ema_policy_heads = self._clone_policy_heads(self.policy)

    def load_checkpoint(self, epoch: int, resume_training_state: bool = True) -> None:
        super().load_checkpoint(epoch, resume_training_state=resume_training_state)
        self._refresh_policy_snapshots_after_load()

    def load_checkpoint_from_path(
        self, path: str, is_accelerate_state: bool = True
    ) -> None:
        super().load_checkpoint_from_path(
            path, is_accelerate_state=is_accelerate_state
        )
        self._refresh_policy_snapshots_after_load()

    def init_logger(self):
        if self.log_to_wandb:
            super().init_logger()

    def init_envs(self):
        out = super().init_envs()
        amago.utils.call_async_env(self.val_envs, "take_long_break")
        return out

    def evaluate_val(self):
        amago.utils.call_async_env(self.val_envs, "resume_from_break")
        try:
            with self._maybe_ema_eval_weights():
                out = super().evaluate_val()
        finally:
            amago.utils.call_async_env(self.val_envs, "take_long_break")
        return out

    def init_model(self):
        super().init_model()
        policy = self.policy

        def _edit_actor_mask(batch, actor_loss, pad_mask):
            B, L, G, _ = actor_loss.shape
            missing_action_mask = einops.repeat(
                ~batch.obs["missing_action_mask"][:, :-1], "b l 1 -> b l g 1", g=G
            )
            return pad_mask & missing_action_mask

        def _edit_critic_mask(batch, critic_loss, pad_mask):
            if pad_mask is None:
                return pad_mask
            B, L, C, G, _ = pad_mask.shape
            missing_action_mask = einops.repeat(
                ~batch.obs["missing_action_mask"][:, :-1],
                "b l 1 -> b l c g 1",
                g=G,
                c=C,
            )
            return pad_mask & missing_action_mask

        policy.edit_actor_mask = _edit_actor_mask
        policy.edit_critic_mask = _edit_critic_mask
        if self.critic_loss_weight_override is not None:
            policy.critic_loss_weight = self.critic_loss_weight_override
        self._refresh_global_anchor_reference()
        self._refresh_update_control_reference()
        self._init_ema_policy_heads()

    def train_step(self, batch: Batch, log_step: bool):
        fbc_filter = self.policy.fbc_filter_func
        if hasattr(fbc_filter, "set_mask"):
            state_mask = ~(batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True)
            action_mask = ~batch.obs["missing_action_mask"]
            fbc_filter.set_mask(state_mask & action_mask)
        if hasattr(fbc_filter, "set_seq_mask") and getattr(
            fbc_filter, "seq_enabled", False
        ):
            seq_mask = (~(batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True)).bool()
            fbc_filter.set_seq_mask(seq_mask)

        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad()
            base_loss = self.policy_aclr(batch, log_step=log_step)
            (
                control_loss,
                control_info,
                observed_local_kl,
                observed_global_anchor_kl,
            ) = self._compute_update_control_loss(batch, log_step=log_step)
            loss = base_loss + control_loss
            l = (
                {"Loss": loss, "Base Loss": base_loss}
                | self.policy.update_info
                | control_info
            )
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                grad_clip = self._current_grad_clip()
                self.accelerator.clip_grad_norm_(
                    self.policy_aclr.parameters(), grad_clip
                )
                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    l.update({"Grad Clip": grad_clip} | self.policy.get_grad_norms())
            self.optimizer.step()
            self.lr_schedule.step()
            if self.accelerator.sync_gradients:
                self._adapt_update_control(observed_local_kl, observed_global_anchor_kl)
                self._apply_lr_controls()
                self._update_ema_policy_heads()
                self._maybe_refresh_update_control_reference_after_update()
            if log_step:
                l.update(
                    {"Learning Rate": self.optimizer.param_groups[0]["lr"]}
                    | self._update_control_state_info()
                )
        return l
