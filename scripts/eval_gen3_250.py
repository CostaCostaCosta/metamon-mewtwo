#!/usr/bin/env python3
"""250-battle evaluation harness for gen3 ROM-native models (goal task 1).

Measures a registered model against:
  * the 6 heuristic baselines (the paper's "Heuristic Composite Score"), and
  * the pretrained SyntheticRLV2 policy (a strong external reference),

each for `--battles` (default 250) battles on the `competitive` gen3ou team
set, with a fixed `--seed` so results are reproducible across checkpoints/runs.

Requires the local Showdown server (server/pokemon-showdown) on port 8000.

Examples
--------
# final model of the online run (default checkpoint = its registration default)
uv run python scripts/eval_gen3_250.py --model Gen3RomOnlineV0 --checkpoint 200

# any registered model
uv run python scripts/eval_gen3_250.py --model Gen3RomNative15M --checkpoint 145

# only the SyntheticRLV2 reference matchup, more battles
uv run python scripts/eval_gen3_250.py --model Gen3RomOnlineV0 --checkpoint 200 \
    --skip_heuristics --battles 250

Outputs a JSON summary (win rate per opponent + composite) to --output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import metamon.rl.pretrained  # noqa: F401  (registers all models incl. gen3)
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from metamon.rl.evaluate import pretrained_vs_baselines, pretrained_vs_metamon
from metamon.env import get_metamon_teams

HEURISTICS = [
    "PokeEnvHeuristic",
    "Gen1BossAI",
    "Grunt",
    "GymLeader",
    "EmeraldKaizo",
    "RandomBaseline",
]


def _win_rates(res):
    """Pull `{opponent: win_rate}` out of an evaluate_test results dict."""
    out = {}
    if not isinstance(res, dict):
        return out
    for k, v in res.items():
        if isinstance(k, str) and k.startswith("Average Win Rate in "):
            opp = k.split("_vs_", 1)[-1]
            try:
                out[opp] = round(float(v), 4)
            except (TypeError, ValueError):
                pass
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="250-battle gen3 eval harness")
    p.add_argument("--model", required=True, help="Registered model name")
    p.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint epoch (default = model's default; -1 = latest)",
    )
    p.add_argument("--format", default="gen3ou")
    p.add_argument("--team_set", default="competitive")
    p.add_argument("--battles", type=int, default=250)
    p.add_argument("--seed", type=int, default=0, help="Fixed seed for reproducibility")
    p.add_argument("--srv2_checkpoint", type=int, default=48)
    p.add_argument("--skip_heuristics", action="store_true")
    p.add_argument("--skip_srv2", action="store_true")
    p.add_argument(
        "--heuristic_actors",
        type=int,
        default=5,
        help="Parallel actors per heuristic baseline",
    )
    p.add_argument("--srv2_parallel", type=int, default=8)
    p.add_argument("--output", default=None, help="Where to write the JSON summary")
    p.add_argument(
        "--log_wandb",
        action="store_true",
        help="Log per-opponent win rates to wandb (group gen3-eval250).",
    )
    args = p.parse_args()

    model = get_pretrained_model(args.model)
    team_set = get_metamon_teams(args.format, args.team_set)

    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "format": args.format,
        "team_set": args.team_set,
        "battles": args.battles,
        "seed": args.seed,
        "heuristics": {},
        "vs_SyntheticRLV2": None,
    }

    # --- 6 heuristic baselines ---
    if not args.skip_heuristics:
        print(
            f"[eval] {args.model} ckpt={args.checkpoint} vs 6 heuristics "
            f"({args.battles} battles each, seed={args.seed})"
        )
        # NOTE: pretrained_vs_baselines has no seed plumbed to the heuristic envs
        # (BattleAgainstBaseline has no seed param). Reproducibility here comes
        # from the fixed battle count + fixed competitive team set; the agent is
        # seeded below. The SyntheticRLV2 path (pretrained_vs_metamon) does seed.
        import random, numpy as np, torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        hres = pretrained_vs_baselines(
            pretrained_model=model,
            battle_format=args.format,
            team_set=team_set,
            checkpoint=args.checkpoint,
            total_battles=args.battles,
            parallel_actors_per_baseline=args.heuristic_actors,
            baselines=HEURISTICS,
        )
        summary["heuristics"] = _win_rates(hres) or hres

    # --- SyntheticRLV2 reference ---
    if not args.skip_srv2:
        print(
            f"[eval] {args.model} ckpt={args.checkpoint} vs SyntheticRLV2 "
            f"ckpt={args.srv2_checkpoint} ({args.battles} battles, seed={args.seed})"
        )
        sres = pretrained_vs_metamon(
            pretrained_model=model,
            battle_format=args.format,
            team_set=team_set,
            team_set_name=args.team_set,
            checkpoint=args.checkpoint,
            total_battles=args.battles,
            num_parallel=args.srv2_parallel,
            opponent_agent="SyntheticRLV2",
            opponent_checkpoint=args.srv2_checkpoint,
            seed=args.seed,
        )
        summary["vs_SyntheticRLV2"] = _win_rates(sres) or sres

    if args.log_wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("METAMON_WANDB_PROJECT", "metamon"),
            entity=os.environ.get("METAMON_WANDB_ENTITY"),
            group="gen3-eval250",
            name=f"eval250_{args.model}_ckpt{args.checkpoint}_seed{args.seed}",
            config={
                "model": args.model,
                "checkpoint": args.checkpoint,
                "team_set": args.team_set,
                "battles": args.battles,
                "seed": args.seed,
            },
            reinit=True,
        )
        flat = {}
        for opp, wr in (summary.get("heuristics") or {}).items():
            flat[f"eval250/heuristic/{opp}"] = wr
        for opp, wr in (summary.get("vs_SyntheticRLV2") or {}).items():
            flat[f"eval250/SyntheticRLV2/{opp}"] = wr
        wandb.log(flat)
        run.finish()
        print(f"[eval] logged {len(flat)} metrics to wandb group gen3-eval250")

    out = args.output or os.path.join(
        os.path.expanduser("~/metamon_runs"),
        f"eval250_{args.model}_ckpt{args.checkpoint}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[eval] wrote {out}")
    print(json.dumps(summary, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
