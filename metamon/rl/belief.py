"""Actor-side belief model utilities for offline RL.

The first target is intentionally simple: predict set-style opponent species
and move labels from the public trajectory representation. Labels are carried
as auxiliary ``belief_*`` observation tensors so AMAGO's normal collation can
pad them, but timestep encoders and actors should never consume those tensors
directly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Optional, Type

import gin
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

import amago
from amago.agent import MultiTaskAgent, binary_filter
from amago.loading import Batch
from amago.nets import actor_critic
from amago.nets.policy_dists import DiscreteLikeContinuous
from amago.nets.tstep_encoders import TstepEncoder
from amago.nets.traj_encoders import TrajEncoder

from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.interface import UniversalState
from metamon.tokenizer import PokemonTokenizer


BELIEF_PREFIX = "belief_"
BELIEF_SPECIES_KEY = f"{BELIEF_PREFIX}opp_species_set"
BELIEF_SPECIES_MASK_KEY = f"{BELIEF_PREFIX}opp_species_mask"
BELIEF_MOVES_KEY = f"{BELIEF_PREFIX}opp_move_set"
BELIEF_MOVES_MASK_KEY = f"{BELIEF_PREFIX}opp_move_mask"


def is_belief_key(key: str) -> bool:
    return key.startswith(BELIEF_PREFIX)


def belief_target_keys(target_type: str = "gen1_species_moves_set") -> tuple[str, ...]:
    if target_type == "gen1_species_set":
        return (BELIEF_SPECIES_KEY, BELIEF_SPECIES_MASK_KEY)
    if target_type == "gen1_species_moves_set":
        return (
            BELIEF_SPECIES_KEY,
            BELIEF_SPECIES_MASK_KEY,
            BELIEF_MOVES_KEY,
            BELIEF_MOVES_MASK_KEY,
        )
    raise ValueError(f"Unknown belief target_type: {target_type!r}")


def _token_id(tokenizer: PokemonTokenizer, value: str) -> Optional[int]:
    idx = tokenizer[value]
    return idx if idx >= 0 else None


def _species_token_ids(
    states: list[UniversalState], tokenizer: PokemonTokenizer
) -> set[int]:
    species: set[int] = set()
    for state in states:
        names = [state.opponent_active_pokemon.base_species]
        names.extend(getattr(state, "opponent_teampreview", []) or [])
        for name in names:
            clean = pokemon_name(name)
            if clean and clean != "<blank>":
                idx = _token_id(tokenizer, clean)
                if idx is not None:
                    species.add(idx)
    return species


def _move_token_ids(states: list[UniversalState], tokenizer: PokemonTokenizer) -> set[int]:
    moves: set[int] = set()
    for state in states:
        for move in state.opponent_active_pokemon.moves:
            clean = move_name(move.name)
            if clean and clean not in {"nomove", "<blank>"}:
                idx = _token_id(tokenizer, clean)
                if idx is not None:
                    moves.add(idx)
        clean_prev = move_name(state.opponent_prev_move.name)
        if clean_prev and clean_prev not in {"nomove", "<blank>"}:
            idx = _token_id(tokenizer, clean_prev)
            if idx is not None:
                moves.add(idx)
    return moves


def _repeated_multihot(
    timesteps: int, vocab_size: int, token_ids: set[int]
) -> torch.Tensor:
    target = torch.zeros((timesteps, vocab_size), dtype=torch.float32)
    valid_ids = [idx for idx in token_ids if 0 <= idx < vocab_size]
    if valid_ids:
        target[:, valid_ids] = 1.0
    return target


def build_gen1_belief_targets(
    states: list[UniversalState],
    tokenizer: PokemonTokenizer,
    target_type: str = "gen1_species_moves_set",
) -> dict[str, torch.Tensor]:
    """Build set-style opponent team targets from a completed trajectory.

    The labels are noisy by construction: for Gen 1 we only know what the parser
    can infer from the replay. Species and moves are aggregated over the whole
    trajectory, then repeated at every timestep as a hidden-team prediction
    target. They must remain auxiliary training labels, not actor observations.
    """

    if not states:
        raise ValueError("Cannot build belief targets from an empty state sequence")
    vocab_size = len(tokenizer)
    timesteps = len(states)

    species_ids = _species_token_ids(states, tokenizer)
    targets = {
        BELIEF_SPECIES_KEY: _repeated_multihot(timesteps, vocab_size, species_ids),
        BELIEF_SPECIES_MASK_KEY: torch.full(
            (timesteps, 1), float(bool(species_ids)), dtype=torch.float32
        ),
    }
    if target_type == "gen1_species_set":
        return targets
    if target_type != "gen1_species_moves_set":
        raise ValueError(f"Unknown belief target_type: {target_type!r}")

    move_ids = _move_token_ids(states, tokenizer)
    targets.update(
        {
            BELIEF_MOVES_KEY: _repeated_multihot(timesteps, vocab_size, move_ids),
            BELIEF_MOVES_MASK_KEY: torch.full(
                (timesteps, 1), float(bool(move_ids)), dtype=torch.float32
            ),
        }
    )
    return targets


@dataclass
class BeliefForwardOutput:
    species_logits: torch.Tensor
    move_logits: Optional[torch.Tensor]
    actor_embedding: torch.Tensor


@gin.configurable
class Gen1OpponentTeamBeliefHead(nn.Module):
    """Predict opponent species/move sets and summarize predictions for the actor."""

    def __init__(
        self,
        state_dim: int,
        vocab_size: int,
        belief_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
        include_moves: bool = True,
        species_loss_weight: float = 1.0,
        move_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.belief_dim = belief_dim
        self.include_moves = include_moves
        self.species_loss_weight = species_loss_weight
        self.move_loss_weight = move_loss_weight

        layers: list[nn.Module] = []
        inp = state_dim
        for _ in range(max(n_layers, 1)):
            layers.extend(
                [
                    nn.Linear(inp, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                ]
            )
            inp = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.species_head = nn.Linear(hidden_dim, vocab_size)
        self.move_head = nn.Linear(hidden_dim, vocab_size) if include_moves else None
        summary_dim = vocab_size * (2 if include_moves else 1)
        self.actor_projection = nn.Sequential(
            nn.Linear(summary_dim, belief_dim),
            nn.LayerNorm(belief_dim),
            nn.LeakyReLU(),
        )

    def forward(self, state: torch.Tensor) -> BeliefForwardOutput:
        hidden = self.trunk(state)
        species_logits = self.species_head(hidden)
        move_logits = self.move_head(hidden) if self.move_head is not None else None
        summaries = [torch.sigmoid(species_logits)]
        if self.include_moves:
            if move_logits is None:
                raise RuntimeError("include_moves=True but move_head is missing")
            summaries.append(torch.sigmoid(move_logits))
        actor_embedding = self.actor_projection(torch.cat(summaries, dim=-1))
        return BeliefForwardOutput(
            species_logits=species_logits,
            move_logits=move_logits,
            actor_embedding=actor_embedding,
        )

    def compute_loss(
        self,
        outputs: BeliefForwardOutput,
        obs: dict[str, torch.Tensor],
        valid_timestep_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = outputs.species_logits.device
        zero = outputs.species_logits.sum() * 0.0
        total = zero
        metrics: dict[str, torch.Tensor] = {}

        species_loss, species_metrics = self._masked_bce(
            logits=outputs.species_logits,
            target=obs.get(BELIEF_SPECIES_KEY),
            mask=obs.get(BELIEF_SPECIES_MASK_KEY),
            valid_timestep_mask=valid_timestep_mask,
            metric_prefix="Belief Species",
        )
        total = total + self.species_loss_weight * species_loss
        metrics.update(species_metrics)

        if self.include_moves and outputs.move_logits is not None:
            move_loss, move_metrics = self._masked_bce(
                logits=outputs.move_logits,
                target=obs.get(BELIEF_MOVES_KEY),
                mask=obs.get(BELIEF_MOVES_MASK_KEY),
                valid_timestep_mask=valid_timestep_mask,
                metric_prefix="Belief Moves",
            )
            total = total + self.move_loss_weight * move_loss
            metrics.update(move_metrics)

        if not metrics:
            metrics["Belief Mask Count"] = torch.tensor(0.0, device=device)
        metrics["Belief Loss"] = total
        return total, metrics

    def _masked_bce(
        self,
        logits: torch.Tensor,
        target: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        valid_timestep_mask: Optional[torch.Tensor],
        metric_prefix: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if target is None or mask is None:
            return logits.sum() * 0.0, {}

        target = target.to(device=logits.device, dtype=logits.dtype)
        mask = mask.to(device=logits.device)
        mask = mask > 0.5
        if valid_timestep_mask is not None:
            mask = mask & valid_timestep_mask.to(device=logits.device).bool()
        mask_f = mask.to(dtype=logits.dtype)
        elem_loss = F.binary_cross_entropy_with_logits(
            logits, target.clamp(0.0, 1.0), reduction="none"
        )
        denom = mask_f.expand_as(elem_loss).sum().clamp_min(1.0)
        loss = (elem_loss * mask_f).sum() / denom

        positive_count = (target.clamp(0.0, 1.0).sum(-1, keepdim=True) * mask_f).sum()
        mask_count = mask_f.sum()
        top1 = logits.argmax(dim=-1, keepdim=True)
        top1_hit = target.gather(-1, top1).clamp(0.0, 1.0) * mask_f
        topk = min(6, logits.shape[-1])
        topk_idx = logits.topk(topk, dim=-1).indices
        topk_hits = target.gather(-1, topk_idx).clamp(0.0, 1.0).sum(
            -1, keepdim=True
        )
        recall_denom = (target.clamp(0.0, 1.0).sum(-1, keepdim=True) * mask_f).sum()
        metrics = {
            f"{metric_prefix} Loss": loss.detach(),
            f"{metric_prefix} Mask Count": mask_count.detach(),
            f"{metric_prefix} Positives": positive_count.detach(),
            f"{metric_prefix} Top1 Hit": (
                top1_hit.sum() / mask_count.clamp_min(1.0)
            ).detach(),
            f"{metric_prefix} Top6 Recall": (
                (topk_hits * mask_f).sum() / recall_denom.clamp_min(1.0)
            ).detach(),
        }
        return loss, metrics


def _infer_vocab_size_from_encoder(encoder: nn.Module) -> Optional[int]:
    candidates = [
        ("turn_embedding", "token_embedding", "tokenizer"),
        ("pokemon_token_emb", "tokenizer"),
        ("global_token_emb", "tokenizer"),
    ]
    for path in candidates:
        obj: Any = encoder
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            try:
                return len(obj)
            except TypeError:
                pass
    return None


@gin.configurable
class MetamonBeliefMultiTaskAgent(MultiTaskAgent):
    """MultiTaskAgent with an actor-side auxiliary opponent-team belief head."""

    def __init__(
        self,
        obs_space,
        rl2_space,
        action_space,
        tstep_encoder_type: Type[TstepEncoder],
        traj_encoder_type: Type[TrajEncoder],
        max_seq_len: int,
        num_critics: int = 4,
        num_critics_td: int = 2,
        online_coeff: float = 0.0,
        offline_coeff: float = 1.0,
        critic_loss_weight: float = 10.0,
        gamma: float = 0.999,
        reward_multiplier: float = 10.0,
        tau: float = 0.003,
        fake_filter: bool = False,
        num_actions_for_value_in_critic_loss: int = 1,
        num_actions_for_value_in_actor_loss: int = 3,
        fbc_filter_func: callable = binary_filter,
        popart: bool = True,
        use_target_actor: bool = True,
        use_multigamma: bool = True,
        n_step: int = 1,
        actor_type: Type[actor_critic.BaseActorHead] = actor_critic.Actor,
        critic_type: Type[actor_critic.BaseCriticHead] = actor_critic.NCriticsTwoHot,
        pass_obs_keys_to_actor: Optional[list[str]] = None,
        belief_enabled: bool = True,
        belief_head_type: Type[Gen1OpponentTeamBeliefHead] = Gen1OpponentTeamBeliefHead,
        belief_vocab_size: Optional[int] = None,
        belief_dim: int = 64,
        belief_loss_coeff: float = 0.1,
        belief_loss_backprop_to_encoder: bool = True,
        detach_actor_belief: bool = False,
    ):
        self.belief_enabled = belief_enabled
        self.belief_head_type = belief_head_type
        self.belief_vocab_size = belief_vocab_size
        self.belief_dim = belief_dim
        self.belief_loss_coeff = belief_loss_coeff
        self.belief_loss_backprop_to_encoder = belief_loss_backprop_to_encoder
        self.detach_actor_belief = detach_actor_belief
        self._check_no_belief_actor_keys(pass_obs_keys_to_actor or [])
        super().__init__(
            obs_space=obs_space,
            rl2_space=rl2_space,
            action_space=action_space,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            max_seq_len=max_seq_len,
            num_critics=num_critics,
            num_critics_td=num_critics_td,
            online_coeff=online_coeff,
            offline_coeff=offline_coeff,
            critic_loss_weight=critic_loss_weight,
            gamma=gamma,
            reward_multiplier=reward_multiplier,
            tau=tau,
            fake_filter=fake_filter,
            num_actions_for_value_in_critic_loss=num_actions_for_value_in_critic_loss,
            num_actions_for_value_in_actor_loss=num_actions_for_value_in_actor_loss,
            fbc_filter_func=fbc_filter_func,
            popart=popart,
            use_target_actor=use_target_actor,
            use_multigamma=use_multigamma,
            n_step=n_step,
            actor_type=actor_type,
            critic_type=critic_type,
            pass_obs_keys_to_actor=pass_obs_keys_to_actor,
        )

    @staticmethod
    def _check_no_belief_actor_keys(keys: list[str]) -> None:
        leaked = [key for key in keys if is_belief_key(key)]
        if leaked:
            raise ValueError(
                f"Belief target keys must not be passed directly to the actor: {leaked}"
            )

    def init_actor_critic(self) -> None:
        critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "gammas": self.gammas,
        }
        self.critics = self.critic_type(**critic_kwargs, num_critics=self.num_critics)
        self.target_critics = self.critic_type(
            **critic_kwargs, num_critics=self.num_critics
        )
        self.maximized_critics = self.critic_type(
            **critic_kwargs, num_critics=self.num_critics
        )

        self.belief_head = None
        actor_state_dim = self.state_dim
        if self.belief_enabled:
            vocab_size = self.belief_vocab_size or _infer_vocab_size_from_encoder(
                self.tstep_encoder
            )
            if vocab_size is None:
                raise ValueError(
                    "Could not infer belief_vocab_size from the timestep encoder; "
                    "set MetamonBeliefMultiTaskAgent.belief_vocab_size in gin."
                )
            self.belief_vocab_size = vocab_size
            self.belief_head = self.belief_head_type(
                state_dim=self.state_dim,
                vocab_size=vocab_size,
                belief_dim=self.belief_dim,
            )
            actor_state_dim = self.state_dim + self.belief_dim

        actor_kwargs = {
            "state_dim": actor_state_dim,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "gammas": self.gammas,
        }
        self.actor = self.actor_type(**actor_kwargs)
        self.target_actor = self.actor_type(**actor_kwargs)

    @property
    def trainable_params(self):
        params = [
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.critics.parameters(),
            self.actor.parameters(),
        ]
        if self.belief_head is not None:
            params.append(self.belief_head.parameters())
        return itertools.chain(*params)

    def get_grad_norms(self) -> dict[str, float]:
        norms = super().get_grad_norms()
        if self.belief_head is not None:
            norms["Belief Head Grad Norm"] = amago.utils.get_grad_norm(
                self.belief_head
            )
        return norms

    def _belief_actor_state(
        self,
        s_rep: torch.Tensor,
        log_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, Optional[BeliefForwardOutput]]:
        if self.belief_head is None:
            return s_rep, None
        outputs = self.belief_head(s_rep)
        belief_emb = outputs.actor_embedding
        if self.detach_actor_belief:
            belief_emb = belief_emb.detach()
        actor_state = torch.cat((s_rep, belief_emb), dim=-1)
        if log_dict is not None:
            log_dict["Belief Embedding Norm"] = belief_emb.norm(dim=-1).mean().detach()
        return actor_state, outputs

    def actor_state_for_policy(
        self, s_rep: torch.Tensor, log_dict: Optional[dict[str, Any]] = None
    ) -> torch.Tensor:
        actor_state, _ = self._belief_actor_state(s_rep, log_dict=log_dict)
        return actor_state

    def _belief_loss_outputs(
        self, s_rep: torch.Tensor, actor_outputs: Optional[BeliefForwardOutput]
    ) -> Optional[BeliefForwardOutput]:
        if self.belief_head is None:
            return None
        if self.belief_loss_backprop_to_encoder and actor_outputs is not None:
            return actor_outputs
        return self.belief_head(s_rep.detach())

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[Any] = None,
        sample: bool = True,
    ):
        tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
        s_rep, hidden_state = self.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )
        actor_state, _ = self._belief_actor_state(s_rep)
        action_dists = self.actor(
            actor_state,
            straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor},
        )
        if sample:
            actions = action_dists.sample()
        else:
            if self.discrete:
                actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
            else:
                actions = action_dists.mean
        actions = actions[..., -1, :]
        dtype = torch.int64 if (self.discrete or self.multibinary) else torch.float32
        return actions.to(dtype=dtype), hidden_state

    def forward(self, batch: Batch, log_step: bool):
        # fmt: off
        self.update_info = {}
        active_log_dict = self.update_info if log_step else None

        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)
        straight_from_obs = {k: batch.obs[k] for k in self.pass_obs_keys_to_actor}

        B, L, _D_o = o.shape
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        K_c = self.num_actions_for_value_in_critic_loss
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        a_buffer = self.actor.policy_dist.action_from_buffer(a_buffer)
        C = len(self.critics)
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)
        r = repeat((self.reward_multiplier * batch.rews).float(), f"b l r -> b l 1 {G} r")
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        gamma = self.gammas.to(r.device).unsqueeze(-1)
        D_emb = self.traj_encoder.emb_dim
        Bins = self.critics.num_bins
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).bool()[:, 1:, ...]
        actor_mask = F.pad(state_mask.float(), (0, 0, 0, 1), "constant", 0.0)
        actor_mask = repeat(actor_mask, f"b l 1 -> b l {G} 1")
        critic_mask = repeat(state_mask.float(), f"b l 1 -> b l {C} {G} 1")
        full_state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).bool()

        s_rep, _hidden_state = self.traj_encoder(
            seq=o, time_idxs=batch.time_idxs, hidden_state=None, log_dict=active_log_dict
        )
        assert s_rep.shape == (B, L, D_emb)
        actor_s_rep, belief_outputs = self._belief_actor_state(s_rep, active_log_dict)

        a_dist = self.actor(actor_s_rep, log_dict=active_log_dict, straight_from_obs=straight_from_obs)
        if self.discrete:
            a_dist = DiscreteLikeContinuous(a_dist)
        if log_step:
            self.update_info.update(self._policy_stats(actor_mask, a_dist))

        critic_loss = None
        if not self.fake_filter or self.online_coeff > 0:
            with torch.no_grad():
                if self.use_target_actor:
                    target_actor_s_rep, _ = self._belief_actor_state(s_rep)
                    a_prime_dist = self.target_actor(target_actor_s_rep, straight_from_obs=straight_from_obs)
                    if self.discrete:
                        a_prime_dist = DiscreteLikeContinuous(a_prime_dist)
                else:
                    a_prime_dist = a_dist
                ap = a_prime_dist.sample((K_c,))
                assert ap.shape == (K_c, B, L, G, D_action)
                sp_ap_gp = (s_rep[:, 1:, ...].detach(), ap[:, :, 1:, ...].detach())
                q_targ_sp_ap_gp = self.target_critics(*sp_ap_gp)
                assert q_targ_sp_ap_gp.probs.shape == (K_c, B, L - 1, C, G, Bins)
                q_targ_sp_ap_gp = self.target_critics.bin_dist_to_raw_vals(q_targ_sp_ap_gp).mean(0)
                assert q_targ_sp_ap_gp.shape == (B, L - 1, C, G, 1)
                q_reduced = self._reduce_critic_ensemble(q_targ_sp_ap_gp)
                assert q_reduced.shape == (B, L - 1, 1, G, 1)
                nstep_mask = state_mask.float().unsqueeze(-1).unsqueeze(-1)
                td_target = self._nstep_fn(r, d, q_reduced, gamma, mask=nstep_mask)
                assert td_target.shape == (B, L - 1, 1, G, 1)
                self.popart.update_stats(td_target, mask=critic_mask.all(2, keepdim=True))
                td_target_labels = self.target_critics.raw_vals_to_labels(td_target)
                td_target_labels = repeat(td_target_labels, f"b l 1 g bins -> b l {C} g bins")
                assert td_target_labels.shape == (B, L - 1, C, G, Bins)

            s_a_g = (s_rep, a_buffer.unsqueeze(0))
            q_s_a_g = self.critics(*s_a_g, log_dict=active_log_dict)
            assert q_s_a_g.probs.shape == (1, B, L, C, G, Bins)
            critic_loss = F.cross_entropy(
                rearrange(q_s_a_g.logits[0, :, :-1, ...], "b l c g u -> (b l c g) u"),
                rearrange(td_target_labels, "b l c g u -> (b l c g) u"),
                reduction="none",
            )
            critic_loss = rearrange(critic_loss, "(b l c g) -> b l c g 1", b=B, l=L - 1, c=C, g=G)
            assert critic_loss.shape == (B, L - 1, C, G, 1)
            scalar_q_s_a_g = self.critics.bin_dist_to_raw_vals(q_s_a_g).squeeze(0)
            if log_step:
                td_stats = self._td_stats(
                    critic_mask,
                    self.popart.normalize_values(scalar_q_s_a_g)[:, :-1, ...],
                    scalar_q_s_a_g[:, :-1, ...],
                    r=r,
                    d=d,
                    td_target=td_target,
                    raw_q_bins=q_s_a_g.probs[0, :, :-1],
                )
                self.update_info.update(td_stats | self._popart_stats())

        actor_loss = 0.0
        K_a = self.num_actions_for_value_in_actor_loss
        if self.offline_coeff > 0:
            if not self.fake_filter:
                with torch.no_grad():
                    a_agent = a_dist.sample((K_a,))
                    q_s_a_agent = self.critics(s_rep.detach(), a_agent)
                    assert q_s_a_agent.probs.shape == (K_a, B, L, C, G, Bins)
                    val_s = self.critics.bin_dist_to_raw_vals(q_s_a_agent)
                    assert val_s.shape == (K_a, B, L, C, G, 1)
                    advantage_s_a = scalar_q_s_a_g.mean(2) - val_s.mean((0, 3))
                    assert advantage_s_a.shape == (B, L, G, 1)
                    filter_ = self.fbc_filter_func(advantage_s_a)[:, :-1, ...].float()
            else:
                filter_ = binary_filter(
                    torch.zeros((B, L - 1, G, 1), device=s_rep.device)
                ).float()
            if self.discrete:
                logp_a = a_dist.log_prob(a_buffer).unsqueeze(-1)
            elif self.multibinary:
                logp_a = a_dist.log_prob(a_buffer).mean(-1, keepdim=True)
            else:
                logp_a = a_dist.log_prob(a_buffer).sum(-1, keepdim=True)
            logp_a = logp_a[:, :-1, ...]
            actor_loss += self.offline_coeff * -(filter_.detach() * logp_a)
            if log_step:
                self.update_info.update(self._filter_stats(actor_mask, logp_a, filter_))

        if self.online_coeff > 0:
            assert self.actor.actions_differentiable, "online-style actor loss is not compatible with action distribution"
            a_agent_dpg = torch.stack([a_dist.rsample() for _ in range(K_a)], dim=0)
            q_s_a_agent = self.maximized_critics(s_rep.detach(), a_agent_dpg)
            q_s_a_agent = self.popart.normalize_values(
                self.maximized_critics.bin_dist_to_raw_vals(q_s_a_agent).mean(0).min(2).values
            )
            actor_loss += self.online_coeff * -(q_s_a_agent[:, :-1, ...])

        total_loss = self._compute_loss(
            batch=batch,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            state_mask=state_mask,
        )
        if self.belief_head is not None and self.belief_loss_coeff > 0.0:
            loss_outputs = self._belief_loss_outputs(s_rep, belief_outputs)
            belief_loss, belief_metrics = self.belief_head.compute_loss(
                loss_outputs,
                batch.obs,
                valid_timestep_mask=full_state_mask,
            )
            total_loss = total_loss + self.belief_loss_coeff * belief_loss
            if log_step:
                self.update_info.update(
                    {k: v.detach() for k, v in belief_metrics.items()}
                    | {"Belief Loss Weighted": (self.belief_loss_coeff * belief_loss).detach()}
                )
        return total_loss
        # fmt: on
