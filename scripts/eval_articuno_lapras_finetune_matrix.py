#!/usr/bin/env python3
"""Evaluate Articuno->Lapras finetune matrix checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


os.environ.setdefault("METAMON_CACHE_DIR", "/home/eddie/metamon_cache")


RUN_CONFIG = {
    "articuno-lapras-naive-control-no-kl": {
        "train_gin": "articuno_lapras_finetune_no_kl_control.gin",
        "has_ema": True,
    },
    "articuno-lapras-v3-local32": {
        "train_gin": "articuno_lapras_finetune_v3_local32.gin",
        "has_ema": True,
    },
    "articuno-lapras-v3-local16": {
        "train_gin": "articuno_lapras_finetune_v3_local16.gin",
        "has_ema": True,
    },
    "articuno-lapras-v3-local24-targetkl004": {
        "train_gin": "articuno_lapras_finetune_v3_local24_targetkl004.gin",
        "has_ema": True,
    },
    "articuno-lapras-v3-local16-targetkl004": {
        "train_gin": "articuno_lapras_finetune_v3_local16_targetkl004.gin",
        "has_ema": True,
    },
    "articuno-lapras-v3-local32-targetkl005": {
        "train_gin": "articuno_lapras_finetune_v3_local32_targetkl005.gin",
        "has_ema": True,
    },
    "articuno-lapras-v3-local32-tiny-global-anchor": {
        "train_gin": "articuno_lapras_finetune_v3_local32_tiny_global_anchor.gin",
        "has_ema": True,
    },
}


@dataclass(frozen=True)
class PolicyConfig:
    label: str
    model_name: str
    checkpoint: int | None
    team_set: str
    temperature: float = 1.0
    local_run_name: str | None = None
    local_ckpt_dir: str | None = None
    local_train_gin: str | None = None
    ema: bool = False


@dataclass
class H2HResult:
    label: str
    policy_a: str
    policy_b: str
    checkpoint: int | None
    ema: bool
    team_a: str
    team_b: str
    battles: int
    wins: int
    losses: int
    win_rate: float
    binomial_se: float
    output_dir: str


def _load_ema_heads(agent, ema_path: Path) -> None:
    import torch

    state = torch.load(ema_path, map_location="cpu")
    modules = {
        "tstep_encoder": agent.policy.tstep_encoder,
        "traj_encoder": agent.policy.traj_encoder,
        "actor": agent.policy.actor,
    }
    for prefix, module in modules.items():
        sub_state = {
            key[len(prefix) + 1 :]: value
            for key, value in state.items()
            if key.startswith(prefix + ".")
        }
        if not sub_state:
            raise RuntimeError(f"No EMA keys found for {prefix} in {ema_path}")
        module.load_state_dict(sub_state, strict=True)


def worker(args: argparse.Namespace) -> int:
    from functools import partial

    from metamon.env import get_metamon_teams
    from metamon.interface import get_reward_function
    from metamon.rl.metamon_to_amago import make_challenge_env
    from metamon.rl.pretrained import (
        ALL_PRETRAINED_MODELS,
        LocalFinetunedModel,
        get_pretrained_model,
    )

    if args.local_run_name:
        base_model = ALL_PRETRAINED_MODELS[args.model_name]
        reward_function = (
            get_reward_function(args.reward_function) if args.reward_function else None
        )
        pretrained = LocalFinetunedModel(
            base_model=base_model,
            amago_ckpt_dir=args.local_ckpt_dir,
            model_name=args.local_run_name,
            default_checkpoint=args.checkpoint,
            train_gin_config=args.local_train_gin,
            reward_function=reward_function,
            battle_backend="metamon",
        )
        checkpoint = args.checkpoint
    else:
        pretrained = get_pretrained_model(args.model_name)
        checkpoint = args.checkpoint

    agent = pretrained.initialize_agent(
        checkpoint=checkpoint,
        log=False,
        action_temperature=args.temperature,
    )
    if args.ema_path:
        _load_ema_heads(agent, Path(args.ema_path))

    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    make_env = partial(
        make_challenge_env,
        battle_format=args.format,
        num_battles=args.battles,
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
        player_team_set=get_metamon_teams(args.format, args.team_set),
        player_username=args.username,
        opponent_username=args.opponent_username,
        role=args.role,
        battle_backend="metamon",
        save_results_to=args.results_dir,
        save_trajectories_to=None,
        print_battle_bar=False,
    )
    results = agent.evaluate_test(
        [make_env],
        timesteps=args.battles * 1000,
        episodes=args.battles,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _count_challenger_results(
    results_dir: Path, challenger_username: str
) -> tuple[int, int]:
    wins = 0
    losses = 0
    for csv_path in results_dir.glob("*.csv"):
        with csv_path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 4 or row[0].strip() != challenger_username:
                    continue
                result = row[3].strip()
                if result == "WIN":
                    wins += 1
                elif result == "LOSS":
                    losses += 1
    return wins, losses


def _policy_key(policy: PolicyConfig) -> str:
    local = f"-{policy.local_run_name}" if policy.local_run_name else ""
    ema = "-ema" if policy.ema else ""
    ckpt = "default" if policy.checkpoint is None else f"ckpt{policy.checkpoint}"
    return f"{policy.label}{local}{ema}-{ckpt}-{policy.team_set}"


def _cmd_for_worker(
    args: argparse.Namespace,
    policy: PolicyConfig,
    username: str,
    opponent_username: str,
    role: str,
    results_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model-name",
        policy.model_name,
        "--format",
        args.format,
        "--battles",
        str(args.battles),
        "--team-set",
        policy.team_set,
        "--temperature",
        str(policy.temperature),
        "--username",
        username,
        "--opponent-username",
        opponent_username,
        "--role",
        role,
        "--results-dir",
        str(results_dir),
    ]
    if policy.checkpoint is not None:
        cmd.extend(["--checkpoint", str(policy.checkpoint)])
    if policy.local_run_name:
        cmd.extend(
            [
                "--local-run-name",
                policy.local_run_name,
                "--local-ckpt-dir",
                policy.local_ckpt_dir or args.save_dir,
                "--local-train-gin",
                policy.local_train_gin or "",
            ]
        )
    if policy.ema:
        ema_path = (
            Path(policy.local_ckpt_dir or args.save_dir)
            / str(policy.local_run_name)
            / "ckpts"
            / "ema_policy_heads"
            / f"policy_epoch_{policy.checkpoint}.pt"
        )
        cmd.extend(["--ema-path", str(ema_path)])
    return cmd


def run_matchup(
    args: argparse.Namespace,
    label: str,
    policy_a: PolicyConfig,
    policy_b: PolicyConfig,
) -> H2HResult:
    matchup_key = f"{label}__{_policy_key(policy_a)}__vs__{_policy_key(policy_b)}"
    short_hash = hashlib.md5(matchup_key.encode()).hexdigest()[:8]
    challenger_username = f"al-A-{short_hash}"
    acceptor_username = f"al-B-{short_hash}"
    matchup_dir = Path(args.output_dir) / matchup_key
    results_dir = matchup_dir / "results"
    matchup_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    wins, losses = _count_challenger_results(results_dir, challenger_username)
    if wins + losses >= args.battles:
        total = wins + losses
        p = wins / total
        return H2HResult(
            label=label,
            policy_a=policy_a.label,
            policy_b=policy_b.label,
            checkpoint=policy_a.checkpoint,
            ema=policy_a.ema,
            team_a=policy_a.team_set,
            team_b=policy_b.team_set,
            battles=total,
            wins=wins,
            losses=losses,
            win_rate=p,
            binomial_se=(p * (1.0 - p) / total) ** 0.5,
            output_dir=str(matchup_dir),
        )

    acceptor_cmd = _cmd_for_worker(
        args, policy_b, acceptor_username, challenger_username, "acceptor", results_dir
    )
    challenger_cmd = _cmd_for_worker(
        args,
        policy_a,
        challenger_username,
        acceptor_username,
        "challenger",
        results_dir,
    )

    env_acceptor = os.environ.copy()
    env_acceptor["CUDA_VISIBLE_DEVICES"] = str(args.gpus[1] if len(args.gpus) > 1 else args.gpus[0])
    acceptor_stdout = None
    acceptor_stderr = None
    challenger_stdout = None
    challenger_stderr = None
    if not args.verbose:
        acceptor_stdout = (matchup_dir / "acceptor.stdout").open("w")
        acceptor_stderr = (matchup_dir / "acceptor.stderr").open("w")
        challenger_stdout = (matchup_dir / "challenger.stdout").open("w")
        challenger_stderr = (matchup_dir / "challenger.stderr").open("w")
    acceptor_proc = None
    try:
        acceptor_proc = subprocess.Popen(
            acceptor_cmd,
            env=env_acceptor,
            text=True,
            stdout=None if args.verbose else acceptor_stdout,
            stderr=None if args.verbose else acceptor_stderr,
        )
        time.sleep(args.acceptor_startup_delay)

        env_challenger = os.environ.copy()
        env_challenger["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        challenger_proc = subprocess.run(
            challenger_cmd,
            env=env_challenger,
            text=True,
            timeout=args.timeout,
            stdout=None if args.verbose else challenger_stdout,
            stderr=None if args.verbose else challenger_stderr,
        )
    except BaseException:
        if acceptor_proc is not None and acceptor_proc.poll() is None:
            acceptor_proc.kill()
            acceptor_proc.wait()
        raise
    finally:
        for handle in (
            acceptor_stdout,
            acceptor_stderr,
            challenger_stdout,
            challenger_stderr,
        ):
            if handle is not None:
                handle.close()
    try:
        acceptor_proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        acceptor_proc.kill()
        acceptor_proc.wait()

    if challenger_proc.returncode != 0:
        raise RuntimeError(
            f"Challenger failed for {matchup_key}. See logs in {matchup_dir}"
        )
    if acceptor_proc.returncode != 0:
        raise RuntimeError(f"Acceptor failed for {matchup_key}. See logs in {matchup_dir}")

    wins, losses = _count_challenger_results(results_dir, challenger_username)
    total = wins + losses
    if total == 0:
        raise RuntimeError(f"No completed battle rows found in {results_dir}")
    p = wins / total
    return H2HResult(
        label=label,
        policy_a=policy_a.label,
        policy_b=policy_b.label,
        checkpoint=policy_a.checkpoint,
        ema=policy_a.ema,
        team_a=policy_a.team_set,
        team_b=policy_b.team_set,
        battles=total,
        wins=wins,
        losses=losses,
        win_rate=p,
        binomial_se=(p * (1.0 - p) / total) ** 0.5,
        output_dir=str(matchup_dir),
    )


def _local_policy(
    args: argparse.Namespace, run_name: str, checkpoint: int, team_set: str, ema: bool
) -> PolicyConfig:
    run_cfg = RUN_CONFIG[run_name]
    return PolicyConfig(
        label=run_name,
        model_name=args.base_model,
        checkpoint=checkpoint,
        team_set=team_set,
        local_run_name=run_name,
        local_ckpt_dir=args.save_dir,
        local_train_gin=run_cfg["train_gin"],
        ema=ema,
    )


def _articuno(checkpoint: int, team_set: str) -> PolicyConfig:
    return PolicyConfig(
        label=f"Articuno{checkpoint}",
        model_name="Articuno",
        checkpoint=checkpoint,
        team_set=team_set,
    )


def _tauros(team_set: str) -> PolicyConfig:
    return PolicyConfig(
        label="TaurosV0",
        model_name="TaurosV0",
        checkpoint=None,
        team_set=team_set,
    )


def build_matchups(args: argparse.Namespace) -> list[tuple[str, PolicyConfig, PolicyConfig]]:
    matchups: list[tuple[str, PolicyConfig, PolicyConfig]] = []
    if args.include_baselines:
        for ckpt in args.baseline_checkpoints:
            matchups.append(
                (
                    f"baseline_articuno{ckpt}_lapras_vs_taurosv0",
                    _articuno(ckpt, "lapras"),
                    _tauros("competitive"),
                )
            )
            matchups.append(
                (
                    f"baseline_articuno{ckpt}_lapras_vs_articuno36",
                    _articuno(ckpt, "lapras"),
                    _articuno(36, "competitive"),
                )
            )
            matchups.append(
                (
                    f"baseline_articuno{ckpt}_retention_vs_taurosv0",
                    _articuno(ckpt, "competitive"),
                    _tauros("competitive"),
                )
            )

    eval_kinds = ["raw"]
    if args.include_ema:
        eval_kinds.append("ema")

    for run_name in args.runs:
        if run_name not in RUN_CONFIG:
            raise ValueError(f"Unknown run {run_name!r}; choices: {sorted(RUN_CONFIG)}")
        for eval_kind in eval_kinds:
            ema = eval_kind == "ema"
            if ema and not RUN_CONFIG[run_name]["has_ema"]:
                continue
            if "specialization" in args.suites:
                matchups.append(
                    (
                        f"{run_name}_{eval_kind}_lapras_vs_taurosv0",
                        _local_policy(args, run_name, args.checkpoint, "lapras", ema),
                        _tauros("competitive"),
                    )
                )
                matchups.append(
                    (
                        f"{run_name}_{eval_kind}_lapras_vs_articuno36",
                        _local_policy(args, run_name, args.checkpoint, "lapras", ema),
                        _articuno(36, "competitive"),
                    )
                )
            if "retention" in args.suites:
                matchups.append(
                    (
                        f"{run_name}_{eval_kind}_retention_vs_taurosv0",
                        _local_policy(
                            args, run_name, args.checkpoint, "competitive", ema
                        ),
                        _tauros("competitive"),
                    )
                )
    return matchups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-name", default="Articuno")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--local-run-name", default=None)
    parser.add_argument("--local-ckpt-dir", default=None)
    parser.add_argument("--local-train-gin", default=None)
    parser.add_argument("--ema-path", default=None)
    parser.add_argument("--reward-function", default=None)
    parser.add_argument("--format", default="gen1ou")
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--team-set", default="competitive")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--username", default=None)
    parser.add_argument("--opponent-username", default=None)
    parser.add_argument("--role", choices=["acceptor", "challenger"], default=None)
    parser.add_argument("--results-dir", default=None)

    parser.add_argument(
        "--save-dir",
        default="/home/eddie/metamon/models/articuno_lapras_v3_finetune_matrix_fixed_anchor_targetkl",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/eddie/metamon/evals/articuno_lapras_v3_finetune_matrix_fixed_anchor_targetkl",
    )
    parser.add_argument("--base-model", default="Articuno")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "articuno-lapras-naive-control-no-kl",
            "articuno-lapras-v3-local24-targetkl004",
            "articuno-lapras-v3-local16-targetkl004",
            "articuno-lapras-v3-local32-targetkl005",
        ],
    )
    parser.add_argument(
        "--suites", nargs="+", default=["specialization", "retention"]
    )
    parser.add_argument("--include-ema", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--baseline-checkpoints", nargs="+", type=int, default=[36, 40])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--acceptor-startup-delay", type=float, default=10.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.worker:
        required = [args.username, args.opponent_username, args.role, args.results_dir]
        if any(value is None for value in required):
            parser.error("--worker requires username, opponent-username, role, results-dir")
    return args


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"

    results = []
    for label, policy_a, policy_b in build_matchups(args):
        if policy_a.ema:
            ema_path = (
                Path(policy_a.local_ckpt_dir or args.save_dir)
                / str(policy_a.local_run_name)
                / "ckpts"
                / "ema_policy_heads"
                / f"policy_epoch_{policy_a.checkpoint}.pt"
            )
            if not ema_path.exists():
                print(f"Skipping missing EMA checkpoint: {ema_path}", file=sys.stderr)
                continue
        result = run_matchup(args, label, policy_a, policy_b)
        results.append(result)
        print(
            f"{label}: {result.wins}-{result.losses} "
            f"({result.win_rate:.1%} +/- {result.binomial_se:.1%} SE)"
        )

    with summary_path.open("w") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
