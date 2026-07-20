import copy
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from metamon.interface import ActionSpace
from metamon.rl.showdown_preview import (
    _agent_observation_preview,
    _as_policy_tensors,
    _current_battle_tag,
    _current_turn,
    _describe_action,
    _sample_actions_for_preview,
    _unwrap_attr,
    showdown_preview_url,
)


ACTION_COLUMNS = [
    "rank",
    "agent_action",
    "action",
    "selected",
    "anchor",
    "consensus",
    "default_anchor",
    "shortlist",
    "proposal_support",
    "final_score",
    "stall_penalty",
    "member_top_votes",
    "mean_member_prob",
    "max_member_prob",
    "candidate_members",
    "candidate_roles",
]

MEMBER_COLUMNS = [
    "idx",
    "model",
    "checkpoint",
    "gxe",
    "top_action",
    "top_action_label",
    "top_prob",
    "margin",
    "entropy",
    "proposer_weight",
    "judge_weight",
    "selected_judge",
]

PROPOSER_COLUMNS = [
    "member_idx",
    "model",
    "checkpoint",
    "role",
    "weight",
    "shortlist_k",
    "allowed_actions",
]

JUDGE_COLUMNS = ["member_idx", "model", "checkpoint", "score", "weight"]


@dataclass
class EnsemblePreviewSnapshot:
    status: str = "Waiting for the first ensemble policy decision..."
    updated_at: float = field(default_factory=time.time)
    battle: str = ""
    turn: Optional[int] = None
    selected_action: Optional[int] = None
    selected_action_label: str = ""
    action_rows: list[list[Any]] = field(default_factory=list)
    member_rows: list[list[Any]] = field(default_factory=list)
    proposer_rows: list[list[Any]] = field(default_factory=list)
    judge_rows: list[list[Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    agent_observation: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class EnsemblePreviewStateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = EnsemblePreviewSnapshot()

    def update(self, snapshot: EnsemblePreviewSnapshot):
        with self._lock:
            self._snapshot = snapshot

    def get(self) -> EnsemblePreviewSnapshot:
        with self._lock:
            return copy.deepcopy(self._snapshot)


def _round(value: Any, digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _join_ints(values: Any) -> str:
    return ", ".join(str(int(value)) for value in values or [])


def _join_strings(values: Any) -> str:
    return ", ".join(str(value) for value in values or [])


def _action_label(
    action_space: ActionSpace,
    state: Any,
    action_idx: Optional[int],
) -> str:
    if action_idx is None:
        return ""
    if state is None:
        return f"Action {action_idx}"
    try:
        return _describe_action(action_space, state, int(action_idx))
    except Exception:
        return f"Action {action_idx}"


def launch_gradio_ensemble_preview(
    store: EnsemblePreviewStateStore,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "The ensemble preview UI requires gradio. "
            "Install it with `pip install gradio`."
        ) from exc

    def read_snapshot():
        snapshot = store.get()
        age = max(0.0, time.time() - snapshot.updated_at)
        summary = [
            "# Metamon Ensemble Preview",
            f"**Status:** {snapshot.status}",
            f"**Last update:** {age:.1f}s ago",
        ]
        if snapshot.battle:
            summary.append(f"**Battle:** `{snapshot.battle}`")
        if snapshot.turn is not None:
            summary.append(f"**Turn:** {snapshot.turn}")
        if snapshot.selected_action is not None:
            summary.append(
                f"**Selected action:** `{snapshot.selected_action}` "
                f"{snapshot.selected_action_label}"
            )
        decision = snapshot.decision.get("decision", {})
        if decision:
            summary.append(f"**Decision reason:** `{decision.get('reason', '')}`")
            summary.append(
                f"**Disagreement:** `{decision.get('disagreement', 0.0):.3f}`"
            )
            if decision.get("full_rerank"):
                summary.append("**Full rerank:** `true`")
        if snapshot.error:
            summary.append(f"**Preview error:** `{snapshot.error}`")

        state_json = json.dumps(
            {
                "universal_state": snapshot.state,
                "agent_observation": snapshot.agent_observation,
            },
            indent=2,
            sort_keys=True,
        )
        return (
            "\n\n".join(summary),
            snapshot.action_rows,
            snapshot.member_rows,
            snapshot.proposer_rows,
            snapshot.judge_rows,
            snapshot.decision,
            state_json,
        )

    with gr.Blocks(title="Metamon Ensemble Preview") as demo:
        with gr.Row():
            summary = gr.Markdown()
            decision_json = gr.JSON(label="Decision")
        with gr.Tabs():
            with gr.Tab("Actions"):
                actions = gr.Dataframe(
                    headers=ACTION_COLUMNS,
                    interactive=False,
                    label="Actions",
                )
            with gr.Tab("Members"):
                members = gr.Dataframe(
                    headers=MEMBER_COLUMNS,
                    interactive=False,
                    label="Members",
                )
            with gr.Tab("Routing"):
                proposers = gr.Dataframe(
                    headers=PROPOSER_COLUMNS,
                    interactive=False,
                    label="Proposers",
                )
                judges = gr.Dataframe(
                    headers=JUDGE_COLUMNS,
                    interactive=False,
                    label="Judges",
                )
            with gr.Tab("State"):
                state = gr.Code(
                    label="State / Agent Observation", language="json", lines=24
                )
        refresh = gr.Button("Refresh")
        outputs = [
            summary,
            actions,
            members,
            proposers,
            judges,
            decision_json,
            state,
        ]
        refresh.click(read_snapshot, outputs=outputs)
        if hasattr(gr, "Timer"):
            timer = gr.Timer(1.0)
            timer.tick(read_snapshot, outputs=outputs)
        demo.load(read_snapshot, outputs=outputs)

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        prevent_thread_lock=True,
        quiet=True,
    )
    return demo


def _build_snapshot(
    env: Any,
    observation_space: Any,
    action_space: ActionSpace,
    debug: dict[str, Any],
    selected_action: Optional[int],
    error: Optional[str] = None,
) -> EnsemblePreviewSnapshot:
    state = _unwrap_attr(env, "_most_recent_state")
    if state is None:
        return EnsemblePreviewSnapshot(
            status="Waiting for battle state...", error=error
        )

    decision = debug.get("decision", {})
    selected_action = selected_action if selected_action is not None else debug.get(
        "selected_action"
    )
    selected_label = _action_label(action_space, state, selected_action)

    action_rows = []
    for rank, row in enumerate(debug.get("actions", []), start=1):
        action_idx = int(row["action"])
        action_rows.append(
            [
                rank,
                action_idx,
                _action_label(action_space, state, action_idx),
                bool(row.get("selected")),
                bool(row.get("anchor")),
                bool(row.get("consensus")),
                bool(row.get("default_anchor")),
                bool(row.get("in_shortlist")),
                _round(row.get("proposal_support")),
                _round(row.get("final_score")),
                _round(row.get("stall_penalty")),
                int(row.get("member_top_votes", 0)),
                _round(row.get("mean_member_prob")),
                _round(row.get("max_member_prob")),
                _join_ints(row.get("candidate_members")),
                _join_strings(row.get("candidate_roles")),
            ]
        )

    member_rows = []
    for row in debug.get("members", []):
        top_action = row.get("top_action")
        member_rows.append(
            [
                int(row["idx"]),
                row.get("model_name", ""),
                row.get("checkpoint"),
                _round(row.get("gxe")),
                top_action,
                _action_label(action_space, state, top_action),
                _round(row.get("top_prob")),
                _round(row.get("margin")),
                _round(row.get("entropy")),
                _round(row.get("proposer_weight")),
                _round(row.get("judge_weight")),
                bool(row.get("selected_judge")),
            ]
        )

    proposer_rows = [
        [
            int(row["member_idx"]),
            row.get("model_name", ""),
            row.get("checkpoint"),
            row.get("role", ""),
            _round(row.get("weight")),
            row.get("shortlist_k"),
            _join_ints(row.get("allowed_actions")),
        ]
        for row in debug.get("proposer_variants", [])
    ]

    judge_rows = [
        [
            int(row["member_idx"]),
            row.get("model_name", ""),
            row.get("checkpoint"),
            _round(row.get("score")),
            _round(row.get("weight")),
        ]
        for row in debug.get("judges", [])
    ]

    status = "Running"
    if debug.get("status") and debug.get("status") != "running":
        status = str(debug["status"])
    return EnsemblePreviewSnapshot(
        status=status,
        battle=_current_battle_tag(env),
        turn=_current_turn(env),
        selected_action=selected_action,
        selected_action_label=selected_label,
        action_rows=action_rows,
        member_rows=member_rows,
        proposer_rows=proposer_rows,
        judge_rows=judge_rows,
        decision={
            "decision": decision,
            "state_summary": debug.get("state_summary", {}),
            "candidates": debug.get("candidates", []),
        },
        state=state.to_dict(),
        agent_observation=_agent_observation_preview(observation_space, state),
        error=error,
    )


def run_showdown_with_ensemble_preview(
    experiment,
    make_env,
    observation_space: Any,
    action_space: ActionSpace,
    timesteps: int,
    episodes: Optional[int],
    server_name: str,
    server_port: int,
    share: bool,
) -> dict[str, float]:
    from amago.envs.amago_env import SequenceWrapper

    policy = experiment.policy
    if not hasattr(policy, "get_last_decision_debug"):
        raise ValueError("--ensemble_step requires an ensemble policy.")

    store = EnsemblePreviewStateStore()
    launch_gradio_ensemble_preview(
        store=store,
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
    print(
        f"Ensemble preview UI: {showdown_preview_url(server_name, server_port)}",
        flush=True,
    )

    policy.eval()
    env = SequenceWrapper(make_env(), save_trajs_to=None, save_every=None)
    env.reset()
    hidden_state = policy.traj_encoder.init_hidden_state(1, experiment.DEVICE)

    episodes_finished = 0
    try:
        for _ in range(timesteps):
            obs, rl2s, time_idxs = _as_policy_tensors(
                env.current_timestep, experiment.DEVICE
            )
            with torch.no_grad():
                with experiment.caster():
                    actions, next_hidden_state = policy.get_actions(
                        obs=obs,
                        rl2s=rl2s,
                        time_idxs=time_idxs,
                        sample=_sample_actions_for_preview(experiment),
                        hidden_state=hidden_state,
                    )

            action_np = actions[0, 0].cpu().numpy()
            selected_action = None
            if action_np.size:
                selected_action = int(np.asarray(action_np).reshape(-1)[0])
            debug = policy.get_last_decision_debug(batch_idx=0)
            store.update(
                _build_snapshot(
                    env=env,
                    observation_space=observation_space,
                    action_space=action_space,
                    debug=debug,
                    selected_action=selected_action,
                )
            )

            *_, terminated, truncated, _ = env.step(action_np)
            done = np.logical_or(terminated, truncated)
            hidden_state = policy.traj_encoder.reset_hidden_state(
                next_hidden_state, done
            )
            if done.any():
                episodes_finished += int(done.sum())
                if episodes is not None and episodes_finished >= episodes:
                    break
                env.reset()
    finally:
        env.close()

    logs = experiment.policy_metrics([env.return_history], [env.special_history])
    experiment.log(logs, key="test")
    return logs
