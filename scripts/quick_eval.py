#!/usr/bin/env python3
"""
Quick evaluation script - simpler alternative to full tournament.

Evaluates a single model against heuristic baselines.
Useful for quick testing without the complexity of ladder matchups.

Usage:
    python scripts/quick_eval.py \
        --model Gen1BinaryV0_Epoch2 \
        --battles 50 \
        --format gen1ou
"""

import os
import sys
import argparse
from pathlib import Path

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metamon.rl.gen1_binary_models import *
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from metamon.rl.evaluate import pretrained_vs_baselines
from metamon.env import get_metamon_teams


def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation against heuristic baselines"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format",
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="competitive",
        help="Team set",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=50,
        help="Total battles (divided among baselines)",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in sorted(get_pretrained_model_names()):
            print(f"  - {model}")
        return 0

    print(f"\n{'='*80}")
    print(f"QUICK EVALUATION")
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")
    print(f"Battles: {args.battles}")
    print(f"{'='*80}\n")

    # Load model
    model = get_pretrained_model(args.model)
    team_set = get_metamon_teams(args.format, args.team_set)

    # Evaluate against heuristics
    results = pretrained_vs_baselines(
        pretrained_model=model,
        battle_format=args.format,
        team_set=team_set,
        total_battles=args.battles,
        parallel_actors_per_baseline=1,  # Keep it simple
        baselines=["PokeEnvHeuristic", "MaxDamage"],  # Just 2 quick baselines
    )

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")

    import json
    print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
