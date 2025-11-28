#!/usr/bin/env python3
"""
Run a single head-to-head matchup between two models.

This script should be run TWICE in separate terminals - once for each model.
Both will connect to the local ladder and battle each other.

Usage (Terminal 1):
    python scripts/run_matchup.py \
        --model Gen1BinaryV0_Epoch0 \
        --username Player1 \
        --battles 100 \
        --output_dir ~/gen1_tournament_results

Usage (Terminal 2):
    python scripts/run_matchup.py \
        --model Gen1BinaryV0_Epoch2 \
        --username Player2 \
        --battles 100 \
        --output_dir ~/gen1_tournament_results

The agents will automatically find and battle each other on the ladder.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metamon.rl.gen1_binary_models import *
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from metamon.rl.evaluate import pretrained_vs_local_ladder
from metamon.env import get_metamon_teams


def main():
    parser = argparse.ArgumentParser(
        description="Run one side of a head-to-head matchup"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (mutually exclusive with --baseline)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline to use (mutually exclusive with --model)",
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Unique username for this agent (must be different from opponent)",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=100,
        help="Number of battles to play",
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
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help="Battle backend",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model is None and args.baseline is None:
        parser.error("Must specify either --model or --baseline")
    if args.model is not None and args.baseline is not None:
        parser.error("Cannot specify both --model and --baseline")

    output_dir = os.path.expanduser(args.output_dir)
    trajectories_dir = os.path.join(output_dir, "trajectories")
    team_results_dir = os.path.join(output_dir, "team_results")
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(team_results_dir, exist_ok=True)

    agent_type = args.model if args.model else args.baseline
    print(f"\n{'='*80}")
    print(f"HEAD-TO-HEAD MATCHUP")
    print(f"Agent: {agent_type} ({'Model' if args.model else 'Baseline'})")
    print(f"Username: {args.username}")
    print(f"Battles: {args.battles}")
    print(f"Format: {args.format}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    print("Connecting to local ladder...")
    print("Waiting for opponent to connect...\n")

    # Load team set
    team_set = get_metamon_teams(args.format, args.team_set)

    if args.model:
        # Load pretrained model
        model = get_pretrained_model(args.model)

        # Run on ladder
        results = pretrained_vs_local_ladder(
            pretrained_model=model,
            username=args.username,
            battle_format=args.format,
            team_set=team_set,
            total_battles=args.battles,
            battle_backend=args.battle_backend,
            save_trajectories_to=trajectories_dir,
            save_team_results_to=team_results_dir,
            log_to_wandb=False,
        )
    else:
        # Use baseline (heuristic)
        from metamon.baselines import get_baseline
        from metamon.rl.evaluate import baseline_vs_local_ladder

        baseline_class = get_baseline(args.baseline)

        # Run on ladder
        results = baseline_vs_local_ladder(
            baseline_class=baseline_class,
            username=args.username,
            battle_format=args.format,
            team_set=team_set,
            total_battles=args.battles,
            save_trajectories_to=trajectories_dir,
            save_team_results_to=team_results_dir,
        )

    print(f"\n{'='*80}")
    print(f"MATCHUP COMPLETE")
    print(f"{'='*80}")

    import json
    print(json.dumps(results, indent=2))

    print(f"\nResults saved to: {output_dir}")
    print(f"Run calculate_elo.py to compute ratings\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
