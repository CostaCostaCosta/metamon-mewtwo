#!/usr/bin/env python3
"""
Round-robin tournament script for evaluating multiple model checkpoints.

Runs all models against each other on the local ladder and collects battle results
for ELO calculation and analysis.

Usage:
    python scripts/self_play_tournament.py \
        --models Gen1BinaryV0_Epoch0 Gen1BinaryV0_Epoch2 SyntheticRLV2 \
        --battles_per_matchup 200 \
        --output_dir ~/gen1_tournament_results \
        --battle_format gen1ou \
        --team_set competitive

This will run a 3x3 round-robin (each model vs every other model).
"""

import os
import sys
import json
import time
import argparse
import subprocess
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from itertools import combinations

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path is set
import metamon
from metamon.rl.gen1_binary_models import *  # Register custom models
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from metamon.rl.evaluate import pretrained_vs_local_ladder
from metamon.env import get_metamon_teams


def get_matchups(model_names: List[str]) -> List[Tuple[str, str]]:
    """Generate all unique pairings for round-robin tournament."""
    return list(combinations(model_names, 2))


def run_matchup(
    model1_name: str,
    model2_name: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    output_dir: str,
    battle_backend: str = "poke-env",
) -> Dict:
    """
    Run a single matchup between two models.

    Both models connect to the local ladder and battle each other.
    Results are saved to output_dir for later ELO calculation.
    """
    print(f"\n{'='*80}")
    print(f"MATCHUP: {model1_name} vs {model2_name}")
    print(f"Format: {battle_format} | Battles: {num_battles}")
    print(f"{'='*80}\n")

    # Create output directories
    trajectories_dir = os.path.join(output_dir, "trajectories")
    team_results_dir = os.path.join(output_dir, "team_results")
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(team_results_dir, exist_ok=True)

    # Load models
    model1 = get_pretrained_model(model1_name)
    model2 = get_pretrained_model(model2_name)

    # Load team set
    team_set = get_metamon_teams(battle_format, team_set_name)

    # Create unique usernames with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%H%M%S")
    username1 = f"{model1_name[:10]}_{timestamp}_A"
    username2 = f"{model2_name[:10]}_{timestamp}_B"

    # Launch both agents in subprocess to run concurrently
    # Model 1 process
    print(f"Launching {model1_name} as {username1}...")
    proc1 = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, '{str(Path(__file__).parent.parent)}')
from metamon.rl.gen1_binary_models import *
from metamon.rl.evaluate import pretrained_vs_local_ladder
from metamon.rl.pretrained import get_pretrained_model
from metamon.env import get_metamon_teams

model = get_pretrained_model('{model1_name}')
team_set = get_metamon_teams('{battle_format}', '{team_set_name}')

pretrained_vs_local_ladder(
    pretrained_model=model,
    username='{username1}',
    battle_format='{battle_format}',
    team_set=team_set,
    total_battles={num_battles},
    battle_backend='{battle_backend}',
    save_trajectories_to='{trajectories_dir}',
    save_team_results_to='{team_results_dir}',
    log_to_wandb=False,
)
""",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give first model time to connect
    time.sleep(5)

    # Model 2 process
    print(f"Launching {model2_name} as {username2}...")
    proc2 = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, '{str(Path(__file__).parent.parent)}')
from metamon.rl.gen1_binary_models import *
from metamon.rl.evaluate import pretrained_vs_local_ladder
from metamon.rl.pretrained import get_pretrained_model
from metamon.env import get_metamon_teams

model = get_pretrained_model('{model2_name}')
team_set = get_metamon_teams('{battle_format}', '{team_set_name}')

pretrained_vs_local_ladder(
    pretrained_model=model,
    username='{username2}',
    battle_format='{battle_format}',
    team_set=team_set,
    total_battles={num_battles},
    battle_backend='{battle_backend}',
    save_trajectories_to='{trajectories_dir}',
    save_team_results_to='{team_results_dir}',
    log_to_wandb=False,
)
""",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for both to complete
    print(f"Battles in progress... (this may take 1-2 hours)")

    try:
        proc1.wait(timeout=7200)  # 2 hour timeout
        proc2.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        print("WARNING: Matchup timed out after 2 hours")
        proc1.kill()
        proc2.kill()
        return {"status": "timeout"}

    # Check for errors
    if proc1.returncode != 0:
        stderr = proc1.stderr.read().decode()
        print(f"ERROR in {model1_name}: {stderr}")
        return {"status": "error", "model": model1_name}

    if proc2.returncode != 0:
        stderr = proc2.stderr.read().decode()
        print(f"ERROR in {model2_name}: {stderr}")
        return {"status": "error", "model": model2_name}

    print(f"✓ Matchup complete: {model1_name} vs {model2_name}")

    return {
        "status": "success",
        "model1": model1_name,
        "model2": model2_name,
        "usernames": [username1, username2],
        "num_battles": num_battles,
    }


def run_tournament(
    models: List[str],
    battle_format: str,
    team_set: str,
    battles_per_matchup: int,
    output_dir: str,
    battle_backend: str = "poke-env",
):
    """Run full round-robin tournament."""

    print(f"\n{'#'*80}")
    print(f"# TOURNAMENT START")
    print(f"# Models: {', '.join(models)}")
    print(f"# Format: {battle_format}")
    print(f"# Battles per matchup: {battles_per_matchup}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all matchups
    matchups = get_matchups(models)
    print(f"Total matchups: {len(matchups)}\n")

    # Track results
    results = []

    # Run each matchup
    for i, (model1, model2) in enumerate(matchups, 1):
        print(f"\n[{i}/{len(matchups)}] Starting matchup...")

        result = run_matchup(
            model1_name=model1,
            model2_name=model2,
            battle_format=battle_format,
            team_set_name=team_set,
            num_battles=battles_per_matchup,
            output_dir=output_dir,
            battle_backend=battle_backend,
        )

        results.append(result)

        # Save intermediate results
        results_file = os.path.join(output_dir, "tournament_results.json")
        with open(results_file, "w") as f:
            json.dump(
                {
                    "models": models,
                    "battle_format": battle_format,
                    "team_set": team_set,
                    "battles_per_matchup": battles_per_matchup,
                    "timestamp": datetime.now().isoformat(),
                    "matchups": results,
                },
                f,
                indent=2,
            )

        print(f"Progress: {i}/{len(matchups)} matchups complete\n")

    print(f"\n{'#'*80}")
    print(f"# TOURNAMENT COMPLETE")
    print(f"# Results saved to: {results_file}")
    print(f"# Run calculate_elo.py to compute rankings")
    print(f"{'#'*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run round-robin tournament between model checkpoints"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to compete (must be registered in pretrained.py or gen1_binary_models.py)",
    )
    parser.add_argument(
        "--battles_per_matchup",
        type=int,
        default=200,
        help="Number of battles for each head-to-head matchup",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/gen1_tournament_results",
        help="Directory to save results and trajectories",
    )
    parser.add_argument(
        "--battle_format",
        type=str,
        default="gen1ou",
        help="Showdown battle format",
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="competitive",
        help="Team set name (competitive, modern_replays, etc.)",
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help="Battle state backend",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("Available models:")
        for model in sorted(get_pretrained_model_names()):
            print(f"  - {model}")
        return

    # Expand home directory
    output_dir = os.path.expanduser(args.output_dir)

    # Validate models exist
    available_models = get_pretrained_model_names()
    for model in args.models:
        if model not in available_models:
            print(f"ERROR: Model '{model}' not found")
            print(f"Available models: {', '.join(sorted(available_models))}")
            print("Run with --list_models to see all options")
            return 1

    # Run tournament
    run_tournament(
        models=args.models,
        battle_format=args.battle_format,
        team_set=args.team_set,
        battles_per_matchup=args.battles_per_matchup,
        output_dir=output_dir,
        battle_backend=args.battle_backend,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
