"""
Generate a distillation dataset for ROM-native student training from Metamon
replay trajectories.

Pipeline (per trajectory)
--------------------------
1. Load a ``*.json.lz4`` replay: ``states`` (list of ``UniversalState``-compatible
   dicts) + ``actions`` (list of int action indices, ``0-8`` = MinimalActionSpace,
   ``-1`` = missing).
2. Encode every state with :class:`RomObservationEncoder` into the canonical
   compact ROM-native tensor representation (no text tokens).
3. Produce a teacher signal over the 9 MinimalActionSpace actions:
     * ``pseudo`` (default): derived from the human ground-truth action. The
       taken action gets ``chosen_mass`` probability mass, the remaining legal
       actions split the rest uniformly, and illegal actions get zero mass.
       For missing actions (``-1``) the distribution is uniform over the legal
       actions.
     * ``amago``: loads a local AMAGO checkpoint (e.g. the
       ``plastic-tauros-15m-belief`` / ``grouped_belief_control`` family) and
       runs it over the whole trajectory (chunked, teacher-forced on the human
       actions) to produce action logits, masked to the legal actions and
       renormalized.
4. Save as ``.npz`` — either one file per trajectory (default), one concatenated
   ``single`` file, or ``sharded`` files of ``--shard-size`` trajectories each.

Output keys (per trajectory, ``T`` = number of states)
------------------------------------------------------
- ``global_cat``          (T, 6)    int32
- ``global_num``          (T, 3)    float32
- ``pokemon_cat``         (T, 13, 9)   int32
- ``pokemon_move_cat``    (T, 13, 4)   int32
- ``pokemon_move_type``   (T, 13, 4)   int32
- ``pokemon_num``         (T, 13, 31)  float32
- ``pokemon_mask``        (T, 13, 4)   int32
- ``legal_action_mask``   (T, 9)    int32      (1 = legal)
- ``actions``             (T,)      int32      (ground truth, -1 = missing)
- ``teacher_logits``      (T, 9)    float32    (log-probs; softmax ~= teacher
                                                distribution; illegal entries
                                                floored at log(floor))
- ``format``              str       (e.g. "gen1ou", from the first state)
- ``source``              str       (input filename)
- ``teacher_type``        str       ("pseudo" or "amago")

``teacher_logits`` stores *log* probabilities with a small floor (1e-8) on
illegal actions so the logits are finite everywhere — required by the existing
student trainer (``metamon/rom_native_obs/train_student.py``) which computes
``F.kl_div(student_log_probs, teacher_log_probs, log_target=True)``.

Missing actions (``-1``) are stored as-is in ``actions`` and produce a uniform
pseudo-teacher over legal actions. Time steps with missing actions are still
included in the dataset; the student trainer filters ``-1`` actions out of its
behavioral-cloning loss.

AMAGO teacher prerequisites
---------------------------
The local belief checkpoint (``grouped_belief_control``) was trained with a
metamon revision that contains ``metamon.rl.belief`` (MetamonBeliefMultiTaskAgent)
and a richer ``MetamonAMAGOExperiment``. Loading it needs BOTH:
  1. ``metamon/rl/belief.py`` present in the repo (it lives on the
     ``lapras work`` commit, ``2c17dfc8e``):
         git show 2c17dfc8e:metamon/rl/belief.py > metamon/rl/belief.py
  2. the duplicate ``traj_save_len`` keyword fixed in
     ``metamon/rl/metamon_to_amago.py`` (a committed syntax error that breaks
     ``import metamon.rl``).

The loader parses the checkpoint's own ``ckpts/config.txt`` and drops bindings
whose parameter does not exist on the *local* metamon classes (training-only
hyperparams drifted between revisions). The model architecture is then verified
by a strict state-dict load, so a mismatch fails loudly instead of silently
producing garbage.

Usage
-----
# Pseudo teacher (default), first 64 trajectories, one .npz each
python metamon/rom_native_obs/generate_dataset.py \
    --data-dir ~/metamon/trajectories/metamon_1400/gen1ou \
    --output-dir /tmp/distill_data --limit 64

# Same, but with a random sample and a single concatenated npz
python metamon/rom_native_obs/generate_dataset.py \
    --data-dir ~/metamon/trajectories/metamon_1400/gen1ou \
    --output-dir /tmp/distill_data --limit 64 --shuffle --seed 0 \
    --save-mode single

# AMAGO teacher from a local checkpoint
python metamon/rom_native_obs/generate_dataset.py \
    --data-dir ~/metamon/trajectories/metamon_1400/gen1ou \
    --output-dir /tmp/distill_data --limit 8 --teacher amago \
    --amago-ckpt-dir ~/metamon/models/plastic-tauros-15m-belief \
    --amago-run-name grouped_belief_control --amago-epoch 7
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import lz4.frame
import numpy as np

# metamon.config captures METAMON_CACHE_DIR at first import; provide a default
# before importing anything from metamon so the (optional) AMAGO teacher path
# can import metamon.rl without requiring the user to export the variable.
os.environ.setdefault("METAMON_CACHE_DIR", os.path.expanduser("~/metamon_cache"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metamon.rom_native_obs import RomObservationEncoder
from metamon.interface import UniversalState
from metamon.rom_native_obs.schema import NUM_ACTIONS

# ============================================================================
# Pseudo teacher
# ============================================================================


def make_pseudo_teacher_distribution(
    actions: np.ndarray,
    legal_action_mask: np.ndarray,
    chosen_mass: float = 0.5,
) -> np.ndarray:
    """Pseudo-teacher probability distribution from ground-truth actions.

    For each timestep ``t``:
      * if ``actions[t]`` is a legal action, it receives ``chosen_mass`` (0.5
        by default) and the remaining legal actions split ``1 - chosen_mass``
        uniformly (one-hot when only one legal action exists);
      * if ``actions[t] == -1`` (missing) or not legal, the distribution is
        uniform over the legal actions;
      * illegal actions always get zero probability.

    Args:
        actions: (T,) int array of MinimalActionSpace indices (-1 = missing).
        legal_action_mask: (T, NUM_ACTIONS) array, nonzero = legal.
        chosen_mass: probability mass on the taken action.

    Returns:
        (T, NUM_ACTIONS) float32 probability matrix.
    """
    actions = np.asarray(actions, dtype=np.int64)
    legal = np.asarray(legal_action_mask) != 0
    T = len(actions)
    probs = np.zeros((T, NUM_ACTIONS), dtype=np.float32)
    for t in range(T):
        legal_idx = np.flatnonzero(legal[t])
        n_legal = len(legal_idx)
        if n_legal == 0:
            # No legal actions (should not happen for a valid state); keep zero.
            continue
        a = int(actions[t])
        if 0 <= a < NUM_ACTIONS and legal[t, a]:
            probs[t, a] = chosen_mass
            others = legal_idx[legal_idx != a]
            if len(others) > 0:
                probs[t, others] = (1.0 - chosen_mass) / len(others)
        else:
            probs[t, legal_idx] = 1.0 / n_legal
    return probs


def pseudo_teacher_logits(
    actions: np.ndarray,
    legal_action_mask: np.ndarray,
    chosen_mass: float = 0.5,
    floor: float = 1e-8,
) -> np.ndarray:
    """Log-probability version of :func:`make_pseudo_teacher_distribution`.

    The ``floor`` keeps logits finite everywhere (``log(0)`` would otherwise
    produce ``-inf``, which breaks ``F.kl_div(..., log_target=True)`` in
    ``train_student.py``). Illegal actions therefore carry probability ~1e-8.
    """
    probs = make_pseudo_teacher_distribution(actions, legal_action_mask, chosen_mass)
    return np.log(probs + floor).astype(np.float32)


# ============================================================================
# Teacher policy interface
# ============================================================================


class TeacherPolicy:
    """Produces (T, NUM_ACTIONS) teacher log-probs for a trajectory of states."""

    name = "teacher"

    def logits_for_trajectory(
        self,
        states: list,
        actions: np.ndarray,
        legal_action_mask: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class PseudoTeacher(TeacherPolicy):
    """Ground-truth pseudo-teacher: one-hot on the taken action, uniform on the
    other legal actions; uniform over legal actions for missing (-1) actions."""

    name = "pseudo"

    def __init__(self, chosen_mass: float = 0.5):
        self.chosen_mass = chosen_mass

    def logits_for_trajectory(self, states, actions, legal_action_mask):
        return pseudo_teacher_logits(
            actions, legal_action_mask, chosen_mass=self.chosen_mass
        )


AMAGO_SETUP_MSG = (
    "The AMAGO teacher could not be imported. The local belief checkpoints "
    "(plastic-tauros-15m-belief/grouped_belief_control) were trained with a "
    "metamon revision that includes `metamon.rl.belief` and expects a richer "
    "`MetamonAMAGOExperiment`. Make sure the repo is set up for them:\n"
    "  1. restore the belief module (exists on commit 2c17dfc8e 'lapras work'):\n"
    "       git show 2c17dfc8e:metamon/rl/belief.py > metamon/rl/belief.py\n"
    "  2. fix the duplicate `traj_save_len` keyword in "
    "metamon/rl/metamon_to_amago.py (a committed SyntaxError that breaks "
    "`import metamon.rl`).\n"
    "Alternatively use --teacher pseudo."
)


class AmagoTeacher(TeacherPolicy):
    """Teacher from a local AMAGO checkpoint (GroupedObservationSpace-based).

    Loads the checkpoint through its own ``ckpts/config.txt`` (dropping
    bindings that do not exist in the local metamon classes), builds a
    placeholder AMAGO experiment, and strict-loads the policy weights.
    Trajectories are encoded into the teacher's native observation space
    (``GroupedObservationSpace`` tokenized with the ``DefaultObservationSpace-v1``
    tokenizer) and run through the policy in windows with hidden-state carry.
    """

    name = "amago"

    def __init__(
        self,
        ckpt_dir: str,
        run_name: str,
        epoch: int | str = 7,
        tokenizer_name: str = "DefaultObservationSpace-v1",
        obs_space_name: str = "GroupedObservationSpace",
        action_space_name: str = "MinimalActionSpace",
        window: int = 120,
        verbose: bool = True,
    ):
        self.ckpt_dir = os.path.expanduser(ckpt_dir)
        self.run_name = run_name
        self.epoch = epoch
        self.tokenizer_name = tokenizer_name
        self.obs_space_name = obs_space_name
        self.action_space_name = action_space_name
        self.window = max(1, int(window))
        self.verbose = verbose
        self._policy = None  # lazily loaded (policy, obs_space, device)
        self._ckpt_path = None

    # ------------------------------------------------------------------ load

    def _checkpoint_path(self) -> str:
        ckpts = os.path.join(self.ckpt_dir, self.run_name, "ckpts")
        if self.epoch == "latest" or self.epoch in (-1, "LATEST"):
            p = os.path.join(ckpts, "latest", "policy.pt")
        else:
            p = os.path.join(
                ckpts, "policy_weights", f"policy_epoch_{int(self.epoch)}.pt"
            )
        if not os.path.exists(p):
            raise FileNotFoundError(f"Teacher checkpoint not found: {p}")
        return p

    def _load(self):
        if self._policy is not None:
            return self._policy
        if self.verbose:
            print(f"[AmagoTeacher] loading {self._checkpoint_path()}")

        os.environ.setdefault(
            "METAMON_CACHE_DIR", os.path.expanduser("~/metamon_cache")
        )
        try:
            import gin
            import torch
            import amago.agent
            import amago.experiment
            import amago.nets.actor_critic
            import amago.nets.traj_encoders
            import amago.nets.transformer
            import metamon.rl.belief
            import metamon.rl.custom_agent
            import metamon.rl.metamon_to_amago as M
            from amago.loading import RLData, RLData_pad_collate
            from metamon.interface import (
                TokenizedObservationSpace,
                get_observation_space,
                get_action_space,
            )
            from metamon.tokenizer import get_tokenizer
        except ImportError as e:  # pragma: no cover - env-dependent
            raise RuntimeError(AMAGO_SETUP_MSG + f"\n  (import error: {e})") from e

        ckpts = os.path.join(self.ckpt_dir, self.run_name, "ckpts")
        config_txt = os.path.join(ckpts, "config.txt")
        if not os.path.exists(config_txt):
            raise FileNotFoundError(
                f"No config.txt found in {ckpts}; expected an AMAGO checkpoint dir."
            )

        # ---- 1. filter the checkpoint's operative config to the params the
        # local metamon classes actually accept (training-only hyperparams
        # drifted between revisions; architecture params must all stay).
        scope_map = self._build_scope_map(M)
        filtered_text = self._filter_config(open(config_txt).read(), scope_map)

        # ---- 2. parse gin + bind the tokenizer
        gin.clear_config()
        gin.parse_config(filtered_text, skip_unknown=True)
        tokenizer = get_tokenizer(self.tokenizer_name)
        gin.bind_parameter("MetamonGroupedTstepEncoderV2.tokenizer", tokenizer)

        # ---- 3. build the (placeholder) experiment
        tok_obs_space = TokenizedObservationSpace(
            base_obs_space=get_observation_space(self.obs_space_name),
            tokenizer=tokenizer,
        )
        experiment = M.make_placeholder_experiment(
            ckpt_base_dir=ckpts,
            run_name=self.run_name,
            log=False,
            observation_space=tok_obs_space,
            action_space=get_action_space(self.action_space_name),
        )
        experiment.start()

        # ---- 4. strict load of the weights (validates the architecture)
        ckpt_path = self._checkpoint_path()
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
        ckpt_state = self._normalize_checkpoint_keys(ckpt_state)
        policy = experiment.policy
        missing = set(policy.state_dict()) - set(ckpt_state)
        unexpected = set(ckpt_state) - set(policy.state_dict())
        if missing or unexpected:
            raise RuntimeError(
                f"Teacher checkpoint does not match the constructed policy: "
                f"{len(missing)} missing / {len(unexpected)} unexpected keys. "
                f"Try --teacher pseudo."
            )
        policy.load_state_dict(ckpt_state, strict=True)
        policy.on_checkpoint_loaded(is_resume=False)
        policy.eval()

        self._policy = (policy, tok_obs_space, next(policy.parameters()).device)
        self._ckpt_path = ckpt_path
        if self.verbose:
            n_params = sum(p.numel() for p in policy.parameters())
            print(
                f"[AmagoTeacher] loaded {self._ckpt_path} "
                f"({type(policy).__name__}, {n_params:,} params)"
            )
        return self._policy

    @staticmethod
    def _build_scope_map(M):
        import amago.agent
        import amago.experiment
        import amago.nets.actor_critic
        import amago.nets.traj_encoders
        import amago.nets.transformer
        import amago.experiment as amago2
        import metamon.rl.belief
        import metamon.rl.custom_agent as metamon2

        return {
            "Experiment": amago.experiment.Experiment,
            "FlashAttention": amago.nets.transformer.FlashAttention,
            "Gen1OpponentTeamBeliefHead": metamon.rl.belief.Gen1OpponentTeamBeliefHead,
            "ISAdvantageFilter": metamon2.ISAdvantageFilter,
            "MetamonAMAGOExperiment": M.MetamonAMAGOExperiment,
            "MetamonBeliefMultiTaskAgent": metamon.rl.belief.MetamonBeliefMultiTaskAgent,
            "MetamonDiscrete": M.MetamonDiscrete,
            "MetamonGroupedTstepEncoderV2": M.MetamonGroupedTstepEncoderV2,
            "MetamonMaskedResidualActor": M.MetamonMaskedResidualActor,
            "Multigammas": amago.agent.Multigammas,
            "MultiTaskAgent": amago.agent.MultiTaskAgent,
            "NCriticsTwoHot": amago.nets.actor_critic.NCriticsTwoHot,
            "TformerTrajEncoder": amago.nets.traj_encoders.TformerTrajEncoder,
        }

    @staticmethod
    def _filter_config(raw: str, scope_map: dict) -> str:
        """Drop ``Selector.param = value`` bindings whose param does not exist
        on the local class (version drift). Returns filtered gin text."""
        import inspect

        def param_exists(selector, param):
            cls = scope_map.get(selector)
            if cls is None:
                return True  # unknown selector: leave it to skip_unknown
            try:
                return param in inspect.signature(cls.__init__).parameters
            except (ValueError, TypeError):
                return True

        out, dropped = [], []
        for line in raw.splitlines():
            stripped = line.strip()
            if " = " in stripped and not stripped.startswith("#"):
                lhs = stripped.split(" = ")[0].strip()
                parts = lhs.split(".")
                if len(parts) >= 2 and not param_exists(parts[-2], parts[-1]):
                    dropped.append(lhs)
                    continue
            out.append(line)
        if dropped:
            print(
                f"[AmagoTeacher] dropped {len(dropped)} version-drifted gin "
                f"bindings (first few: {dropped[:5]} ...)"
            )
        return "\n".join(out)

    @staticmethod
    def _normalize_checkpoint_keys(ckpt_state: dict) -> dict:
        """Map older Perceiver FF module names to the current ones."""
        replacements = (
            (".cross_ff.0.", ".cross_ff1."),
            (".cross_ff.2.", ".cross_ff2."),
            (".self_ff.0.", ".self_ff1."),
            (".self_ff.2.", ".self_ff2."),
        )
        out = {}
        for k, v in ckpt_state.items():
            nk = k
            for old, new in replacements:
                nk = nk.replace(old, new)
            out[nk] = v
        return out

    # -------------------------------------------------------------- inference

    def logits_for_trajectory(self, states, actions, legal_action_mask):
        policy, obs_space, device = self._load()
        acts = np.asarray(actions, dtype=np.int64)
        legal = np.asarray(legal_action_mask) != 0
        T = len(states)
        if T == 0:
            return np.zeros((0, NUM_ACTIONS), dtype=np.float32)

        import torch
        from amago.loading import RLData, RLData_pad_collate

        # Native teacher observation (GroupedObservationSpace, tokenized).
        obs_space.reset()
        obs_list = [obs_space.state_to_obs(s) for s in states]
        obs = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0].keys()}
        # AMAGO convention: illegal_actions = 1 for illegal.
        obs["illegal_actions"] = ~legal

        # rl2s[t] = [prev_reward, prev_action_onehot]; reward is unknown here.
        T = len(states)
        rl2s = np.zeros((T, 1 + NUM_ACTIONS), dtype=np.float32)
        for t in range(1, T):
            a = int(acts[t - 1])
            if 0 <= a < NUM_ACTIONS:
                rl2s[t, 1 + a] = 1.0
        time_idxs = np.arange(T, dtype=np.int64).reshape(T, 1)

        probs = np.zeros((T, NUM_ACTIONS), dtype=np.float32)
        hidden = None
        W = self.window
        for start in range(0, T, W):
            end = min(start + W, T)
            win_len = end - start
            rldata = RLData(
                obs={k: torch.from_numpy(v[start:end]) for k, v in obs.items()},
                rews=torch.zeros((max(win_len - 1, 1), 1)),
                dones=torch.zeros((max(win_len - 1, 1), 1), dtype=torch.bool),
                actions=torch.zeros((max(win_len - 1, 1), NUM_ACTIONS)),
                time_idxs=torch.from_numpy(time_idxs[start:end]),
                rl2s=torch.from_numpy(rl2s[start:end]),
            )
            batch = RLData_pad_collate([rldata]).to(device)
            with torch.no_grad():
                tstep_emb = policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
                s_rep, hidden = policy.traj_encoder(
                    tstep_emb, time_idxs=batch.time_idxs, hidden_state=hidden
                )
                actor_state_fn = getattr(policy, "actor_state_for_policy", None)
                actor_state = (
                    actor_state_fn(s_rep) if actor_state_fn is not None else s_rep
                )
                actor_kw = {
                    k: batch.obs[k]
                    for k in getattr(policy, "pass_obs_keys_to_actor", [])
                    if k in batch.obs
                }
                dists = policy.actor(actor_state, straight_from_obs=actor_kw)
            # (1, win, n_gammas, n_actions) -> primary gamma (last)
            if dists.probs.ndim == 4:
                win_probs = dists.probs[0, :, -1, :].cpu().numpy()
            else:
                win_probs = dists.probs[0].cpu().numpy()
            probs[start:end] = win_probs

        # Mask to legal actions and renormalize (the AMAGO Discrete dist leaves
        # tiny probability on illegal actions via its clip floor).
        probs[~legal] = 0.0
        row_sums = probs.sum(-1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        probs /= row_sums
        return np.log(probs + 1e-8).astype(np.float32)


# ============================================================================
# Trajectory loading / encoding
# ============================================================================


def load_trajectory(path: str) -> tuple[dict, np.ndarray, dict]:
    """Load a ``*.json.lz4`` trajectory.

    Returns (raw_data, actions, meta) where ``raw_data`` has ``states`` /
    ``actions`` keys and ``meta`` carries the source filename + battle format.
    """
    with lz4.frame.open(path, "rt") as f:
        data = json.load(f)
    if "states" not in data or "actions" not in data:
        raise ValueError(f"{path}: missing 'states' or 'actions' keys")
    actions = np.asarray(data["actions"], dtype=np.int32)
    meta = {"source": os.path.basename(path)}
    if data["states"]:
        meta["format"] = str(data["states"][0].get("format", ""))
    return data, actions, meta


def encode_trajectory(encoder: RomObservationEncoder, states: list) -> dict:
    """Encode UniversalState objects with the ROM encoder into stacked tensors.

    Returns a dict of (T, ...) numpy arrays with the RomBattleState tensor keys.
    """
    rom_states = [encoder.encode(s).to_tensors() for s in states]
    return {k: np.stack([rs[k] for rs in rom_states]) for k in rom_states[0]}


def process_trajectory(
    data: dict,
    encoder: RomObservationEncoder,
    teacher: TeacherPolicy,
) -> dict | None:
    """Encode one trajectory; returns the full output dict or None if malformed."""
    states = [UniversalState.from_dict(s) for s in data["states"]]
    actions = np.asarray(data["actions"], dtype=np.int32)
    T = len(actions)
    if T == 0 or len(states) < T:
        return None
    states = states[:T]

    encoder.reset()
    tensors = encode_trajectory(encoder, states)
    tensors["actions"] = actions
    tensors["legal_action_mask"] = tensors["legal_action_mask"].astype(np.int32)
    tensors["teacher_logits"] = teacher.logits_for_trajectory(
        states, actions, tensors["legal_action_mask"]
    )
    tensors["teacher_type"] = teacher.name
    return tensors


def add_meta(arrays: dict, meta: dict) -> dict:
    arrays = dict(arrays)
    arrays["source"] = meta.get("source", "")
    arrays["format"] = meta.get("format", "")
    return arrays


def save_trajectory_npz(output_path: str, arrays: dict) -> None:
    np.savez_compressed(output_path, **arrays)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a ROM-native distillation dataset from Metamon replay "
            "trajectories (JSON.lz4)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "--input-dir",
        dest="data_dir",
        required=True,
        help="Directory containing *.json.lz4 trajectory files.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to write .npz files."
    )
    parser.add_argument(
        "--glob", default="*.json.lz4", help="Glob pattern for trajectory files."
    )
    parser.add_argument(
        "--limit",
        "--max-files",
        dest="limit",
        type=int,
        default=-1,
        help="Process at most this many trajectories (-1 = all).",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Randomly sample --limit trajectories."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --shuffle.")
    parser.add_argument("--gen", type=int, default=1, help="Game generation (1).")
    parser.add_argument(
        "--teacher",
        choices=["pseudo", "amago"],
        default="pseudo",
        help="Teacher signal for 'teacher_logits'.",
    )
    parser.add_argument(
        "--chosen-mass",
        type=float,
        default=0.5,
        help="Pseudo-teacher probability mass on the taken action.",
    )
    parser.add_argument(
        "--save-mode",
        choices=["per_trajectory", "single", "sharded"],
        default="per_trajectory",
        help=(
            "per_trajectory: one .npz per replay; single: one concatenated "
            ".npz with 'trajectory_ids'; sharded: concatenated shards of "
            "--shard-size trajectories."
        ),
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--output-name", default="dataset")

    # AMAGO teacher options
    parser.add_argument(
        "--amago-ckpt-dir",
        default=os.path.expanduser("~/metamon/models/plastic-tauros-15m-belief"),
        help="Base dir containing <run_name>/ckpts for the local AMAGO checkpoint.",
    )
    parser.add_argument("--amago-run-name", default="grouped_belief_control")
    parser.add_argument(
        "--amago-epoch",
        default=7,
        help="Checkpoint epoch to load, or 'latest' for ckpts/latest/policy.pt.",
    )
    parser.add_argument("--amago-tokenizer", default="DefaultObservationSpace-v1")
    parser.add_argument("--amago-obs-space", default="GroupedObservationSpace")
    parser.add_argument("--amago-action-space", default="MinimalActionSpace")
    parser.add_argument(
        "--amago-window",
        type=int,
        default=120,
        help="Max sequence length per teacher forward pass (hidden state carries).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(files)
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"No files matched {os.path.join(args.data_dir, args.glob)}")
        sys.exit(1)
    print(
        f"[generate_dataset] {len(files)} trajectories | teacher={args.teacher} "
        f"| save_mode={args.save_mode}"
    )

    if args.teacher == "pseudo":
        teacher: TeacherPolicy = PseudoTeacher(chosen_mass=args.chosen_mass)
    else:
        teacher = AmagoTeacher(
            ckpt_dir=args.amago_ckpt_dir,
            run_name=args.amago_run_name,
            epoch=args.amago_epoch,
            tokenizer_name=args.amago_tokenizer,
            obs_space_name=args.amago_obs_space,
            action_space_name=args.amago_action_space,
            window=args.amago_window,
        )

    encoder = RomObservationEncoder(gen=args.gen)
    success, failed = 0, 0
    total_timesteps = 0
    total_with_action = 0
    t0 = time.time()

    collected: list[dict] = []  # for single / sharded modes
    shard_idx = 0

    for i, filepath in enumerate(files):
        try:
            data, actions, meta = load_trajectory(filepath)
            arrays = process_trajectory(data, encoder, teacher)
            if arrays is None:
                failed += 1
                continue
            arrays = add_meta(arrays, meta)
            T = len(actions)
            total_timesteps += T
            total_with_action += int((actions >= 0).sum())

            if args.save_mode == "per_trajectory":
                stem = os.path.basename(filepath)
                for suffix in (".json.lz4", ".lz4", ".json"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                out_path = output_dir / f"{stem}.npz"
                save_trajectory_npz(str(out_path), arrays)
            else:
                arrays["trajectory_ids"] = np.full(T, success, dtype=np.int32)
                collected.append(arrays)
                if args.save_mode == "sharded" and len(collected) >= args.shard_size:
                    _write_shard(collected, output_dir, args.output_name, shard_idx)
                    collected = []
                    shard_idx += 1
            success += 1
        except Exception as e:  # noqa: BLE001 - keep going on bad files
            failed += 1
            if failed <= 5:
                print(f"  ERROR {filepath}: {e!r}")

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(
                f"  {i + 1}/{len(files)} | ok={success} fail={failed} "
                f"| {time.time() - t0:.1f}s"
            )

    # flush remaining collected trajectories
    if args.save_mode == "single" and collected:
        _write_shard(collected, output_dir, args.output_name, 0, final=True)
    elif args.save_mode == "sharded" and collected:
        _write_shard(collected, output_dir, args.output_name, shard_idx)

    print(
        f"[generate_dataset] done: {success} ok / {failed} failed in "
        f"{time.time() - t0:.1f}s"
    )
    if total_timesteps:
        print(
            f"[generate_dataset] {total_timesteps} timesteps "
            f"({100.0 * total_with_action / total_timesteps:.1f}% have actions)"
        )


def _write_shard(
    collected: list[dict], output_dir: Path, name: str, idx: int, final=False
):
    """Concatenate trajectory arrays along T and write one .npz.

    Scalar (0-d) meta keys such as ``source`` / ``format`` / ``teacher_type``
    are kept from the first trajectory; per-timestep arrays are concatenated
    along the time axis and ``trajectory_ids`` marks each trajectory's rows.
    """
    first = collected[0]
    out: dict = {}
    # 0-d (scalar) keys: keep from the first trajectory
    for k, v in first.items():
        if k in ("trajectory_ids", "n_trajectories"):
            continue
        if np.ndim(v) == 0:
            out[k] = v
    # per-timestep array keys: concatenate along T
    array_keys = [k for k, v in first.items() if k not in out and np.ndim(v) > 0]
    for k in array_keys:
        out[k] = np.concatenate([c[k] for c in collected], axis=0)
    out["trajectory_ids"] = np.concatenate(
        [c["trajectory_ids"] for c in collected], axis=0
    )
    out["n_trajectories"] = np.int32(len(collected))
    suffix = "" if final else f"_{idx:04d}"
    path = output_dir / f"{name}{suffix}.npz"
    np.savez_compressed(path, **out)
    print(f"  wrote {path} ({len(collected)} trajectories)")


if __name__ == "__main__":
    main()
