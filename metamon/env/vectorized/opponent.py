"""Batched in-the-loop opponents for the vectorized Showdown env.

The env synchronizes all lanes at each decision cycle and asks the opponent for
one action per *active* lane in a single call, so opponent inference is amortized
across the batch. Two implementations are provided:

  * :class:`RandomBatchedOpponent` — no NN; returns random action indices (the env
    repairs any illegal pick against the legal mask). Useful for smoke tests.
  * :class:`AmagoBatchedOpponent` — wraps a metamon policy via
    :class:`~metamon.env.vectorized.amago_policy.AmagoLadderPolicyDriver`, using
    the same ``rl2`` / ``time_idx`` bookkeeping as ``QueueOnLocalLadder``.
  * :class:`ConfigBatchedOpponent` — one shared policy for all lanes; on full env
    ``reset()``, sample an opponent from an :class:`~metamon.rl.evaluate.opponent_pool.OpponentPoolConfig`.
    Pool entries may mix action dimensions: each cached policy bundle owns its own
    ``rl2`` buffer and hidden state; ``configure()`` swaps bundles and reinitializes.
"""

from __future__ import annotations

import gc
import json
import math
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from .amago_policy import AmagoLadderPolicyDriver

if TYPE_CHECKING:
    from metamon.env.wrappers import TeamSet
    from metamon.rl.evaluate.common import PolicySpec


class BatchedOpponent(ABC):
    """Interface the env uses to drive the in-the-loop opponent."""

    @abstractmethod
    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        """Return an action index per lane.

        Only entries where ``active[i]`` is True are guaranteed meaningful (and
        only those lanes advance any internal recurrent state). ``obs_list`` has
        one obs dict per lane (the env supplies a cached obs for inactive lanes).
        """

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        """Record the reward/action for ``lane_idx`` (rl2 bookkeeping)."""

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        """Reset per-lane recurrent state for finished lanes."""

    def reset_all(self) -> None:
        """Reset all per-lane state (called on env.reset)."""


class RandomBatchedOpponent(BatchedOpponent):
    def __init__(
        self, num_lanes: int, action_dim: int, rng: Optional[np.random.Generator] = None
    ):
        self.num_lanes = num_lanes
        self.action_dim = action_dim
        self.rng = rng or np.random.default_rng()

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self.rng.integers(0, self.action_dim, size=self.num_lanes).astype(
            np.int64
        )


class AmagoBatchedOpponent(BatchedOpponent):
    """Wrap an AMAGO policy with ladder-identical rollout bookkeeping."""

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        num_lanes: int,
        action_dim: int,
        hidden_state=None,
        sample: bool = True,
    ):
        self._driver = AmagoLadderPolicyDriver(
            policy=policy,
            device=device,
            num_lanes=num_lanes,
            action_dim=action_dim,
            hidden_state=hidden_state,
            sample=sample,
        )
        self.action_dim = int(action_dim)

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self._driver.act(active, obs_list)

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        self._driver.observe(lane_idx, reward, action_idx)

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        self._driver.reset_lanes(done_mask)

    def reset_all(self) -> None:
        self._driver.reset_all()



def _build_type_chart() -> np.ndarray:
    """Standard 18-type effectiveness chart (Gen 3).

    Returns an 18x18 matrix where ``chart[attacker][defender]`` is the damage
    multiplier (0.0, 0.5, 1.0, 2.0).
    """
    T = 18  # Normal=0 .. Fairy=17 (Gen 3 has no Fairy, but index 17 is unused)
    chart = np.ones((T, T), dtype=np.float32)
    # 0=Normal, 1=Fire, 2=Water, 3=Electric, 4=Grass, 5=Ice,
    # 6=Fighting, 7=Poison, 8=Ground, 9=Flying, 10=Psychic, 11=Bug,
    # 12=Rock, 13=Ghost, 14=Dragon, 15=Dark, 16=Steel, 17=Fairy(unused in gen3)
    # fmt: off
    # Normal
    chart[0][12] = 0.5; chart[0][14] = 0.0
    # Fire
    chart[1][1] = 0.5; chart[1][2] = 0.5; chart[1][4] = 2.0; chart[1][5] = 2.0
    chart[1][8] = 2.0; chart[1][11] = 2.0; chart[1][12] = 0.5; chart[1][14] = 0.5
    chart[1][16] = 2.0
    # Water
    chart[2][1] = 2.0; chart[2][2] = 0.5; chart[2][4] = 0.5; chart[2][14] = 0.5
    # Electric
    chart[3][2] = 2.0; chart[3][3] = 0.5; chart[3][4] = 0.5; chart[3][8] = 0.0
    chart[3][9] = 2.0; chart[3][14] = 0.5
    # Grass
    chart[4][1] = 0.5; chart[4][2] = 2.0; chart[4][3] = 2.0; chart[4][4] = 0.5
    chart[4][9] = 0.5; chart[4][11] = 0.5; chart[4][12] = 2.0; chart[4][14] = 0.5
    chart[4][16] = 0.5
    # Ice
    chart[5][1] = 0.5; chart[5][2] = 0.5; chart[5][4] = 2.0; chart[5][5] = 0.5
    chart[5][8] = 2.0; chart[5][9] = 2.0; chart[5][12] = 2.0; chart[5][14] = 2.0
    chart[5][16] = 0.5
    # Fighting
    chart[6][0] = 2.0; chart[6][5] = 2.0; chart[6][12] = 0.5; chart[6][7] = 0.5
    chart[6][9] = 0.5; chart[6][10] = 0.5; chart[6][11] = 0.5; chart[6][14] = 2.0
    chart[6][13] = 0.0; chart[6][16] = 2.0; chart[6][15] = 2.0
    # Poison
    chart[7][4] = 2.0; chart[7][7] = 0.5; chart[7][8] = 0.5; chart[7][13] = 0.5
    chart[7][16] = 0.0
    # Ground
    chart[8][1] = 2.0; chart[8][3] = 2.0; chart[8][4] = 0.5; chart[8][9] = 0.0
    chart[8][11] = 0.5; chart[8][12] = 0.5; chart[8][14] = 2.0
    chart[8][12] = 0.5; chart[8][16] = 2.0
    # Flying
    chart[9][4] = 2.0; chart[9][11] = 2.0; chart[9][12] = 0.5; chart[9][3] = 0.5
    chart[9][14] = 1.0; chart[9][9] = 1.0
    # Psychic
    chart[10][6] = 2.0; chart[10][7] = 2.0; chart[10][13] = 1.0; chart[10][15] = 0.0
    chart[10][16] = 0.5; chart[10][10] = 0.5
    # Bug
    chart[11][1] = 0.5; chart[11][4] = 2.0; chart[11][6] = 0.5; chart[11][7] = 2.0
    chart[11][13] = 0.5; chart[11][10] = 2.0; chart[11][15] = 2.0; chart[11][16] = 0.5
    # Rock
    chart[12][1] = 2.0; chart[12][2] = 2.0; chart[12][3] = 2.0; chart[12][5] = 2.0
    chart[12][4] = 2.0; chart[12][11] = 2.0; chart[12][9] = 2.0; chart[12][12] = 0.5
    chart[12][14] = 2.0
    # Ghost
    chart[13][0] = 0.0; chart[13][10] = 0.0; chart[13][13] = 2.0; chart[13][13] = 2.0
    chart[13][15] = 0.5
    # Dragon
    chart[14][14] = 2.0; chart[14][16] = 0.5
    # Dark
    chart[15][10] = 2.0; chart[15][13] = 0.5; chart[15][15] = 0.5; chart[15][16] = 0.5
    # Steel
    chart[16][1] = 0.5; chart[16][2] = 0.5; chart[16][3] = 0.5; chart[16][5] = 0.5
    chart[16][12] = 0.5; chart[16][13] = 1.0; chart[16][16] = 0.5; chart[16][14] = 0.5
    chart[16][4] = 0.5; chart[16][7] = 0.0; chart[16][11] = 1.0; chart[16][15] = 1.0
    # Fairy (unused in gen3, but keep for safety)
    chart[17][1] = 0.5; chart[17][14] = 2.0; chart[17][15] = 2.0; chart[17][16] = 0.5
    # fmt: on
    return chart

class HeuristicBatchedOpponent(BatchedOpponent):
    """Rule-based batched opponent that decodes ROM-native observation dicts.

    Avoids loading a neural network: the heuristic logic works directly on the
    parsed observation tensors (move types, HP fractions, legal-action mask) that
    the vectorized environment already produces for the opponent side.

    Heuristic levels (mirrors the eval-suite baselines):
      - ``RandomBaseline``: uniform random legal action.
      - ``Grunt``: random legal move; switch to the healthiest bench Pokemon
        when the active Pokemon's HP falls below 33%.
      - ``GymLeader``: prefer the legal move with the highest type-effectiveness
        multiplier against the opponent's active Pokemon; switch when active HP
        is below 25% *and* a healthier teammate exists.
      - ``Gen1BossAI``: same as GymLeader but also considers move base power
        (picks the highest BP * type-effectiveness product among legal moves).

    All heuristics fall back to a random legal action when no "smart" choice is
    available (e.g. no moves are legal, or all switches are fainted).  The env's
    ``_resolve_action`` repairs any illegal pick, so a wrong index is safe.
    """

    # Standard Pokemon type-chart multipliers (attacking_type -> defending_type).
    # 18x18 matrix, rows = attacker, cols = defender. 0 = immune.
    _TYPE_CHART = _build_type_chart()

    def __init__(
        self,
        heuristic_name: str,
        num_lanes: int,
        action_dim: int,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.heuristic_name = heuristic_name
        self.num_lanes = num_lanes
        self.action_dim = action_dim
        self.temperature = temperature
        self.rng = rng or np.random.default_rng()

    # -- obs decoding helpers (ROM-native schema v2) ----------------------

    @staticmethod
    def _active_move_types(obs: dict) -> np.ndarray:
        """Return shape-(4,) array of move-type enum IDs for the *opponent's*
        active Pokemon (slot 0 in the opponent's observation)."""
        return obs["pokemon_move_type"][0]

    @staticmethod
    def _opponent_pokemon_types(obs: dict) -> tuple:
        """Return (type_1, type_2) enum IDs for the *learner's* active Pokemon
        (slot 6 in the opponent's observation = the opponent-of-the-opponent)."""
        cat = obs["pokemon_cat"][6]
        return int(cat[1]), int(cat[2])

    @staticmethod
    def _active_hp_fraction(obs: dict) -> float:
        """HP fraction (0-1, -1=unknown) of the opponent's active Pokemon."""
        return float(obs["pokemon_num"][0][0])

    @staticmethod
    def _bench_hp_fractions(obs: dict) -> list:
        """HP fractions for the opponent's bench Pokemon (slots 1-5)."""
        return [float(obs["pokemon_num"][i][0]) for i in range(1, 6)]

    @staticmethod
    def _legal_actions(obs: dict) -> np.ndarray:
        """Boolean mask of legal actions (shape-(9,))."""
        return obs["legal_action_mask"].astype(bool)

    @staticmethod
    def _move_base_powers(obs: dict) -> np.ndarray:
        """Return shape-(4,) array of base-power normals for the opponent's
        active Pokemon's four move slots."""
        nums = obs["pokemon_num"][0]
        # move_1_bp_norm .. move_4_bp_norm at indices 15, 19, 23, 27
        return np.array([nums[15], nums[19], nums[23], nums[27]], dtype=np.float32)

    @classmethod
    def _type_effectiveness(cls, atk_type: int, def_type1: int, def_type2: int) -> float:
        """Product of type-chart multipliers for ``atk_type`` vs both defender types."""
        if atk_type <= 0 or atk_type >= len(cls._TYPE_CHART):
            return 1.0
        mult = 1.0
        for dt in (def_type1, def_type2):
            if dt > 0 and dt < len(cls._TYPE_CHART):
                mult *= cls._TYPE_CHART[atk_type][dt]
        return mult

    # -- action selection --------------------------------------------------

    def _pick_random_legal(self, legal: np.ndarray) -> int:
        """Uniform random choice among legal action indices."""
        legal_ids = np.where(legal)[0]
        if len(legal_ids) == 0:
            return int(self.rng.integers(0, self.action_dim))
        return int(self.rng.choice(legal_ids))

    def _pick_smart_move(self, obs: dict, legal: np.ndarray) -> int:
        """Pick the best legal move (actions 0-3) by type effectiveness (and BP
        for Gen1BossAI).  Falls back to random if no move is legal."""
        move_types = self._active_move_types(obs)
        opp_t1, opp_t2 = self._opponent_pokemon_types(obs)
        move_bps = self._move_base_powers(obs)

        move_legal = legal[:4]
        if not move_legal.any():
            return -1  # caller falls back to random

        scores = []
        for m in range(4):
            if not move_legal[m]:
                scores.append(-1.0)
                continue
            eff = self._type_effectiveness(int(move_types[m]), opp_t1, opp_t2)
            bp = float(move_bps[m]) if self.heuristic_name == "Gen1BossAI" else 1.0
            scores.append(eff * max(bp, 0.01))

        best = int(np.argmax(scores))
        if scores[best] <= 0:
            return -1
        return best

    def _maybe_switch(self, obs: dict, legal: np.ndarray, hp_threshold: float) -> int:
        """Switch to the healthiest bench Pokemon if active HP < threshold and
        a switch is legal and the target has more HP.  Returns -1 if no switch
        is warranted."""
        if not legal[4:].any():
            return -1
        hp = self._active_hp_fraction(obs)
        if hp >= 0 and hp > hp_threshold:
            return -1
        bench_hp = self._bench_hp_fractions(obs)
        switch_legal = legal[4:]
        best_switch = -1
        best_hp = hp
        for i, legal_s in enumerate(switch_legal):
            if legal_s and bench_hp[i] > best_hp:
                best_hp = bench_hp[i]
                best_switch = 4 + i
        return best_switch

    def _pick_kaizo_move(self, obs: dict, legal: np.ndarray) -> int:
        """Pick the best legal move like Gen1BossAI but also prefer moves that
        can KO the opponent (estimated damage > opponent HP).  Falls back to
        the smart-move heuristic if no KO candidate is found."""
        move_types = self._active_move_types(obs)
        opp_t1, opp_t2 = self._opponent_pokemon_types(obs)
        move_bps = self._move_base_powers(obs)
        opp_hp = self._active_hp_fraction(obs)

        move_legal = legal[:4]
        if not move_legal.any():
            return -1

        scores = []
        for m in range(4):
            if not move_legal[m]:
                scores.append(-1.0)
                continue
            eff = self._type_effectiveness(int(move_types[m]), opp_t1, opp_t2)
            bp = float(move_bps[m])
            # Estimated damage proxy: eff * bp (both normalised 0-1 range).
            dmg_proxy = eff * max(bp, 0.01)
            # Boost moves that look like they could KO.
            if opp_hp > 0 and dmg_proxy >= (opp_hp * 2.0):
                dmg_proxy *= 3.0
            scores.append(dmg_proxy)

        best = int(np.argmax(scores))
        if scores[best] <= 0:
            return -1
        return best

    def _act_single(self, obs: dict) -> int:
        """Choose one action for one lane based on the heuristic level."""
        legal = self._legal_actions(obs)
        name = self.heuristic_name

        if name == "RandomBaseline":
            return self._pick_random_legal(legal)

        if name == "Grunt":
            # Switch at low HP; otherwise random legal move.
            sw = self._maybe_switch(obs, legal, hp_threshold=0.33)
            if sw >= 0:
                return sw
            move = self._pick_smart_move(obs, legal)
            if move >= 0:
                return move
            return self._pick_random_legal(legal)

        if name in ("GymLeader", "Gen1BossAI"):
            # Prefer super-effective move; switch at very low HP.
            move = self._pick_smart_move(obs, legal)
            if move >= 0:
                return move
            sw = self._maybe_switch(obs, legal, hp_threshold=0.25)
            if sw >= 0:
                return sw
            return self._pick_random_legal(legal)

        if name == "EmeraldKaizo":
            # Smartest heuristic: prefer KO moves, then type-eff * BP; switch
            # when at type disadvantage and low HP.
            move = self._pick_kaizo_move(obs, legal)
            if move >= 0:
                return move
            sw = self._maybe_switch(obs, legal, hp_threshold=0.30)
            if sw >= 0:
                return sw
            move = self._pick_smart_move(obs, legal)
            if move >= 0:
                return move
            return self._pick_random_legal(legal)

        # Unknown heuristic → random.
        return self._pick_random_legal(legal)

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        out = np.zeros(self.num_lanes, dtype=np.int64)
        for i in range(self.num_lanes):
            if active[i]:
                out[i] = self._act_single(obs_list[i])
            else:
                out[i] = 0  # placeholder for inactive lanes (env ignores)
        return out

class ConfigBatchedOpponent(BatchedOpponent):
    """One opponent shared by all lanes; resample from config on env reset() only."""

    def __init__(
        self,
        config: "OpponentPoolConfig",
        num_lanes: int,
        device: torch.device,
        sample: bool = True,
        cache_size: int = 1,
        weights_path: Optional[str] = None,
        quota_min_games: Optional[int] = None,
        quota_window: int = 128,
    ):
        from metamon.rl.evaluate.opponent_pool import OpponentPoolConfig

        if not isinstance(config, OpponentPoolConfig):
            raise TypeError(f"config must be OpponentPoolConfig, got {type(config)}")
        self.config = config
        self.num_lanes = int(num_lanes)
        self.device = device
        self.sample = sample
        # PSRO-Lite sidecar: optional ``meta_weights.json`` keyed by agent row
        # name. Re-read only when mtime changes; fall back to uniform on any
        # error. ``None`` disables the reader entirely (val/ladder unchanged).
        self._weights_path = weights_path
        self._weights_mtime: Optional[float] = None
        # Quota-based diversification: a rolling window of recent ``configure()``
        # assignments (one per env reset = ``num_lanes`` games against one shared
        # opponent). Each agent row is guaranteed at least ``quota_min_games``
        # games over the window — dominated, ladder-strong policies can never
        # fall to ~0 games played (which previously triggered the cold-fallback
        # weight spike). The surplus (window slots beyond all quotas) is sampled
        # by the PSRO-Lite weights, so prioritization still tilts toward weaker
        # matchups on the margin. ``quota_min_games=None`` / ``<= 0`` disables
        # the quota (pure weighted sampling, the previous behavior).
        self._quota_window = max(1, int(quota_window))
        min_assignments = 0
        if quota_min_games is not None and quota_min_games > 0 and self.num_lanes > 0:
            min_assignments = max(1, int(math.ceil(quota_min_games / self.num_lanes)))
        self._quota_min_assignments = min_assignments
        self._quota_recent: "deque[str]" = deque(maxlen=self._quota_window)
        # Bound how many opponent policies stay resident on the GPU. Each cached
        # bundle holds a full policy (60-200M params) plus per-lane KV caches, so
        # an unbounded cache OOMs collectors that resample a new opponent every
        # epoch. LRU-evict and free GPU memory beyond this many distinct opponents.
        self._cache_size = max(1, int(cache_size))
        self.current_spec: Optional["PolicySpec"] = None
        self.current_team: Optional["TeamSet"] = None
        self._active_key: Optional[str] = None
        self._bundle: Optional[BatchedOpponent] = None
        self._cache: "OrderedDict[str, BatchedOpponent]" = OrderedDict()

    def _make_bundle(self, spec: "PolicySpec") -> BatchedOpponent:
        from metamon.rl.pretrained import get_pretrained_model

        model = get_pretrained_model(spec.model_name)

        # Heuristic baselines: no neural network, just rule-based action selection.
        if getattr(model, "is_heuristic", False):
            # Strip "Heuristic" prefix so _act_single sees "RandomBaseline" etc.
            h_name = spec.model_name.replace("Heuristic", "", 1)
            return HeuristicBatchedOpponent(
                heuristic_name=h_name,
                num_lanes=self.num_lanes,
                action_dim=model.action_space.gym_space.n,
                temperature=spec.temperature,
            )

        agent = model.initialize_agent(
            checkpoint=spec.checkpoint,
            log=False,
            action_temperature=spec.temperature,
        )
        action_dim = model.action_space.gym_space.n
        agent.policy.to(self.device)
        agent.policy.eval()
        return AmagoBatchedOpponent(
            policy=agent.policy,
            device=self.device,
            num_lanes=self.num_lanes,
            action_dim=action_dim,
            sample=self.sample,
        )

    def _free_bundle(self, bundle: BatchedOpponent) -> None:
        """Drop a bundle's GPU tensors (policy weights + per-lane KV caches)."""
        driver = getattr(bundle, "_driver", None)
        if driver is not None:
            driver.hidden_state = None
            driver.policy = None

    def _maybe_refresh_weights(self) -> None:
        """Re-read the PSRO-Lite sidecar if it changed; apply to the pool."""
        if self._weights_path is None:
            return
        try:
            mtime = os.path.getmtime(self._weights_path)
        except OSError:
            # No sidecar yet (e.g. before psro_start_epoch) → stay uniform.
            self._weights_mtime = None
            return
        if mtime == self._weights_mtime:
            return
        try:
            with open(self._weights_path, "r") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        # Align sidecar weights (keyed by agent row name) to ``self.config.agents``
        # rows. Missing agents and non-finite values fall back to uniform via
        # ``set_weights`` all-zero/None handling.
        names = [row[0] for row in self.config.agents]
        aligned: List[float] = []
        for name in names:
            try:
                aligned.append(float(raw.get(name, 0.0)))
            except (TypeError, ValueError):
                aligned.append(0.0)
        try:
            self.config.set_weights(aligned)
        except ValueError:
            self.config.set_weights(None)
        self._weights_mtime = mtime

    def _sample_with_quota(self) -> "PolicySpec":
        """Two-phase opponent draw: guaranteed representation, then weighted surplus.

        One ``configure()`` call assigns one shared opponent to all ``num_lanes``
        lanes for one battle, so the quota is tracked in units of *assignments*
        (``quota_min_games`` is converted to assignments via ``num_lanes``).

        Over the rolling ``quota_window`` most recent assignments, every agent
        row is guaranteed at least ``quota_min_assignments`` assignments — i.e.
        at least ``quota_min_assignments * num_lanes`` games. Any window slots
        beyond the union of quotas are filled by the PSRO-Lite weighted sample
        (or uniform when no sidecar/weights), so prioritization still tilts the
        *surplus* toward weaker matchups without ever starving a policy.

        If the window is too small to satisfy every agent's quota
        (``n_agents * min > window``), the quota is infeasible and we fall back
        to pure weighted sampling rather than starving the surplus entirely.
        """
        names = [row[0] for row in self.config.agents]
        n_agents = len(names)
        min_a = self._quota_min_assignments
        if min_a <= 0 or n_agents == 0:
            return self.config.sample_opponent()
        if n_agents * min_a > self._quota_window:
            # Window can't hold all quotas → can't guarantee representation; fall
            # back to weighted sampling (raise the window or lower the floor).
            return self.config.sample_opponent()
        # Per-agent assignment counts within the current rolling window.
        counts: Dict[str, int] = {nm: 0 for nm in names}
        for nm in self._quota_recent:
            if nm in counts:
                counts[nm] += 1
        under = [nm for nm in names if counts[nm] < min_a]
        if under:
            # Pick the agent furthest below its quota (largest deficit); random
            # tie-break so identical deficits don't lock onto one row.
            max_deficit = max(min_a - counts[nm] for nm in under)
            tied = [nm for nm in under if (min_a - counts[nm]) == max_deficit]
            pick = self.config.rng.choice(tied)
            spec = self.config.sample_opponent_for_agent(pick)
        else:
            # All quotas satisfied → spend the surplus on the weighted sample.
            spec = self.config.sample_opponent()
        self._quota_recent.append(spec.name)
        return spec

    def configure(self, spec: Optional["PolicySpec"] = None) -> "PolicySpec":
        """Activate one sampled (or explicit) opponent for all lanes."""
        self._maybe_refresh_weights()
        sampled = spec is None
        if sampled:
            spec = self._sample_with_quota()
        self.current_spec = spec
        self.current_team = self.config.team_set_for(spec.team_set)
        key = spec.unique_key
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = self._make_bundle(spec)
            # LRU-evict the oldest opponents and reclaim their GPU memory.
            evicted = False
            while len(self._cache) > self._cache_size:
                _, old_bundle = self._cache.popitem(last=False)
                self._free_bundle(old_bundle)
                del old_bundle
                evicted = True
            if evicted:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._bundle = self._cache[key]
        self._active_key = key
        self._bundle.reset_all()
        return spec

    def _require_bundle(self) -> BatchedOpponent:
        if self._bundle is None:
            raise RuntimeError(
                "ConfigBatchedOpponent.configure() must run before act()"
            )
        return self._bundle

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self._require_bundle().act(active, obs_list)

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        self._require_bundle().observe(lane_idx, reward, action_idx)

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        if self._bundle is not None:
            self._bundle.reset_lanes(done_mask)

    def reset_all(self) -> None:
        if self._bundle is not None:
            self._bundle.reset_all()
