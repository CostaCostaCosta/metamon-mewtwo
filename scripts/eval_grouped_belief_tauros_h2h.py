#!/usr/bin/env python3
"""Run a local grouped belief checkpoint against TaurosV0."""

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


@dataclass
class H2HResult:
    run: str
    checkpoint: int
    battles: int
    wins: int
    losses: int
    win_rate: float
    output_dir: str


def worker(args: argparse.Namespace) -> int:
    from functools import partial

    from metamon.env import get_metamon_teams
    from metamon.interface import get_action_space, get_observation_space, get_reward_function
    from metamon.rl.metamon_to_amago import make_challenge_env
    from metamon.rl.pretrained import LocalPretrainedModel, get_pretrained_model
    from metamon.tokenizer import get_tokenizer

    if args.side == "tauros":
        pretrained = get_pretrained_model("TaurosV0")
        checkpoint = None
    else:
        pretrained = LocalPretrainedModel(
            amago_ckpt_dir=args.ckpt_root,
            model_name=args.run,
            model_gin_config=args.model_gin_config,
            train_gin_config=args.train_gin_config,
            default_checkpoint=args.checkpoint,
            observation_space=get_observation_space(args.obs_space),
            action_space=get_action_space(args.action_space),
            reward_function=get_reward_function(args.reward_function),
            tokenizer=get_tokenizer(args.tokenizer),
            battle_backend=args.battle_backend,
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    args.tokenizer
                ),
            },
        )
        checkpoint = args.checkpoint

    agent = pretrained.initialize_agent(
        checkpoint=checkpoint,
        log=False,
        action_temperature=args.temperature,
    )
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    make_env = partial(
        make_challenge_env,
        battle_format="gen1ou",
        num_battles=args.battles,
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
        player_team_set=get_metamon_teams("gen1ou", args.team_set),
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


def count_challenger_results(results_dir: Path, challenger_username: str) -> tuple[int, int]:
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


def run_one(args: argparse.Namespace) -> H2HResult:
    matchup_label = f"{args.run}-ckpt{args.checkpoint}-vs-TaurosV0"
    short_hash = hashlib.md5(matchup_label.encode()).hexdigest()[:8]
    challenger_username = f"gb-A-{short_hash}"
    acceptor_username = f"gb-B-{short_hash}"
    matchup_dir = Path(args.output_dir) / matchup_label
    results_dir = matchup_dir / "results"
    matchup_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--run",
        args.run,
        "--checkpoint",
        str(args.checkpoint),
        "--battles",
        str(args.battles),
        "--ckpt-root",
        args.ckpt_root,
        "--model-gin-config",
        args.model_gin_config,
        "--train-gin-config",
        args.train_gin_config,
        "--obs-space",
        args.obs_space,
        "--action-space",
        args.action_space,
        "--reward-function",
        args.reward_function,
        "--tokenizer",
        args.tokenizer,
        "--battle-backend",
        args.battle_backend,
        "--team-set",
        args.team_set,
        "--temperature",
        str(args.temperature),
        "--results-dir",
        str(results_dir),
    ]

    acceptor_cmd = base_cmd + [
        "--side",
        "tauros",
        "--role",
        "acceptor",
        "--username",
        acceptor_username,
        "--opponent-username",
        challenger_username,
    ]
    challenger_cmd = base_cmd + [
        "--side",
        "local",
        "--role",
        "challenger",
        "--username",
        challenger_username,
        "--opponent-username",
        acceptor_username,
    ]

    env = os.environ.copy()
    env.setdefault("METAMON_CACHE_DIR", "/home/eddie/metamon_cache")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    acceptor = subprocess.Popen(
        acceptor_cmd,
        cwd=str(Path.cwd()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(args.acceptor_startup_delay)
    challenger = subprocess.run(
        challenger_cmd,
        cwd=str(Path.cwd()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )
    try:
        acceptor.wait(timeout=60)
    except subprocess.TimeoutExpired:
        acceptor.kill()
        acceptor.wait()

    (matchup_dir / "challenger.stdout").write_text(challenger.stdout or "")
    (matchup_dir / "challenger.stderr").write_text(challenger.stderr or "")
    (matchup_dir / "acceptor.stdout").write_text(
        acceptor.stdout.read() if acceptor.stdout else ""
    )
    (matchup_dir / "acceptor.stderr").write_text(
        acceptor.stderr.read() if acceptor.stderr else ""
    )

    if challenger.returncode != 0 or acceptor.returncode != 0:
        raise RuntimeError(
            f"H2H failed for {matchup_label}: challenger={challenger.returncode}, "
            f"acceptor={acceptor.returncode}. See {matchup_dir}"
        )

    wins, losses = count_challenger_results(results_dir, challenger_username)
    total = wins + losses
    if total == 0:
        raise RuntimeError(f"No battle results found for {matchup_label} in {results_dir}")
    return H2HResult(
        run=args.run,
        checkpoint=args.checkpoint,
        battles=total,
        wins=wins,
        losses=losses,
        win_rate=wins / total,
        output_dir=str(matchup_dir),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--side", choices=["local", "tauros"])
    parser.add_argument("--role", choices=["challenger", "acceptor"])
    parser.add_argument("--username")
    parser.add_argument("--opponent-username")
    parser.add_argument("--results-dir")
    parser.add_argument("--run", default="grouped_belief_control_150k")
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument(
        "--ckpt-root",
        default="/home/eddie/metamon/models/plastic-tauros-15m-belief",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/eddie/metamon/evals/plastic_tauros_15m_belief_vs_taurosv0",
    )
    parser.add_argument(
        "--model-gin-config",
        default="smaller_multitaskagent_grouped_v2_belief_arch.gin",
    )
    parser.add_argument(
        "--train-gin-config",
        default="plastic_tauros_15m_belief_control.gin",
    )
    parser.add_argument("--obs-space", default="GroupedObservationSpace")
    parser.add_argument("--action-space", default="MinimalActionSpace")
    parser.add_argument("--reward-function", default="AggressiveShapedReward")
    parser.add_argument("--tokenizer", default="DefaultObservationSpace-v1")
    parser.add_argument("--battle-backend", default="metamon")
    parser.add_argument("--team-set", default="competitive")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--acceptor-startup-delay", type=float, default=10.0)
    args = parser.parse_args()

    if args.worker:
        required = [
            args.side,
            args.role,
            args.username,
            args.opponent_username,
            args.results_dir,
        ]
        if any(value is None for value in required):
            parser.error(
                "--worker requires --side, --role, --username, "
                "--opponent-username, and --results-dir"
            )
        return worker(args)

    result = run_one(args)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    out_path = Path(args.output_dir) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(result)], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
