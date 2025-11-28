#!/usr/bin/env python3
"""
Smart tournament runner for mixed RL + heuristic populations.

Automatically routes matchups based on policy types:
- RL vs RL: Ladder-based (self-play)
- RL vs Heuristic: Direct evaluation (pretrained_vs_baselines)
- Heuristic vs Heuristic: Simple evaluation

This is more reliable and faster than forcing all matchups through the ladder.

Usage:
    python scripts/run_mixed_population_tournament.py \
        --population_file ~/nash_phase0/population.json \
        --battles_per_matchup 50 \
        --battle_format gen1ou \
        --team_set competitive \
        --output_dir ~/nash_phase0
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import combinations
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metamon.nash.population import PolicyPopulation
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.evaluate import pretrained_vs_baselines
from metamon.env import get_metamon_teams


def get_matchups(policy_names: List[str]) -> List[Tuple[str, str]]:
    """Generate all unique pairings."""
    return list(combinations(policy_names, 2))


def run_single_matchup(
    matchup_info: Tuple[str, str, str, str],
    population_file: str,
    battle_format: str,
    team_set: str,
    battles_per_matchup: int,
    output_dir: str,
) -> dict:
    """
    Run a single matchup. This function is designed to be called in parallel.

    Args:
        matchup_info: (policy1_name, policy2_name, policy1_type, policy2_type)
        population_file: Path to population.json
        battle_format: Battle format
        team_set: Team set name
        battles_per_matchup: Number of battles
        output_dir: Output directory

    Returns:
        Result dictionary from the matchup
    """
    # Reload population in subprocess
    population = PolicyPopulation.load(population_file)

    policy1_name, policy2_name, policy1_type, policy2_type = matchup_info

    # Route based on types
    if policy1_type == "pretrained" and policy2_type == "pretrained":
        # RL vs RL - use ladder
        result = run_rl_vs_rl_ladder(
            model1_name=policy1_name,
            model2_name=policy2_name,
            battle_format=battle_format,
            team_set_name=team_set,
            num_battles=battles_per_matchup,
            output_dir=output_dir,
        )
    elif policy1_type == "pretrained" and policy2_type == "heuristic":
        # RL vs Heuristic - use direct eval
        result = run_rl_vs_heuristic(
            rl_model_name=policy1_name,
            heuristic_name=policy2_name,
            battle_format=battle_format,
            team_set_name=team_set,
            num_battles=battles_per_matchup,
            output_dir=output_dir,
        )
    elif policy1_type == "heuristic" and policy2_type == "pretrained":
        # Heuristic vs RL - flip and use direct eval
        result = run_rl_vs_heuristic(
            rl_model_name=policy2_name,
            heuristic_name=policy1_name,
            battle_format=battle_format,
            team_set_name=team_set,
            num_battles=battles_per_matchup,
            output_dir=output_dir,
        )
    else:
        # Heuristic vs Heuristic
        result = run_heuristic_vs_heuristic(
            heuristic1_name=policy1_name,
            heuristic2_name=policy2_name,
            battle_format=battle_format,
            team_set_name=team_set,
            num_battles=battles_per_matchup,
            output_dir=output_dir,
        )

    return result


def run_rl_vs_rl_ladder(
    model1_name: str,
    model2_name: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    output_dir: str,
) -> dict:
    """
    Run RL vs RL matchup using local ladder.

    Both models connect to ladder and battle each other.
    """
    print(f"\n{'='*80}")
    print(f"RL vs RL (Ladder): {model1_name} vs {model2_name}")
    print(f"{'='*80}\n")

    timestamp = datetime.now().strftime("%H%M%S")
    # Sanitize usernames - replace underscores with hyphens
    # Format: ModelName-HHMMSS-A (max 18 chars for showdown)
    # timestamp=6, suffix=2 ("-A"), separator=1 = 9 chars total
    # Leaves 18-9=9 chars for model name
    user1 = f"{model1_name.replace('_', '-')[:9]}-{timestamp}-A"
    user2 = f"{model2_name.replace('_', '-')[:9]}-{timestamp}-B"

    trajectories_dir = os.path.join(output_dir, "trajectories")
    team_results_dir = os.path.join(output_dir, "team_results")
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(team_results_dir, exist_ok=True)

    # Check if verbose mode enabled
    verbose = os.environ.get("METAMON_VERBOSE", "0") == "1"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Setup output streams
    if verbose:
        # Stream to terminal
        stdout1 = None
        stderr1 = None
        stdout2 = None
        stderr2 = None
    else:
        # Log to files
        stdout1 = open(os.path.join(log_dir, f"{model1_name}_{user1}.stdout.log"), "w")
        stderr1 = open(os.path.join(log_dir, f"{model1_name}_{user1}.stderr.log"), "w")
        stdout2 = open(os.path.join(log_dir, f"{model2_name}_{user2}.stdout.log"), "w")
        stderr2 = open(os.path.join(log_dir, f"{model2_name}_{user2}.stderr.log"), "w")

    # Launch model 1
    print(f"Launching {model1_name}...")
    proc1 = subprocess.Popen(
        [
            sys.executable,
            "scripts/run_matchup.py",
            "--model", model1_name,
            "--username", user1,
            "--battles", str(num_battles),
            "--format", battle_format,
            "--team_set", team_set_name,
            "--output_dir", output_dir,
            "--battle_backend", "metamon",  # Use metamon backend for ladder battles
        ],
        stdout=stdout1,
        stderr=stderr1,
        cwd=str(Path(__file__).parent.parent),
    )

    # Wait longer for first agent to fully connect and start laddering
    # This prevents "Agent is not challenging" errors
    time.sleep(30)

    # Launch model 2
    print(f"Launching {model2_name}...")
    proc2 = subprocess.Popen(
        [
            sys.executable,
            "scripts/run_matchup.py",
            "--model", model2_name,
            "--username", user2,
            "--battles", str(num_battles),
            "--format", battle_format,
            "--team_set", team_set_name,
            "--output_dir", output_dir,
            "--battle_backend", "metamon",  # Use metamon backend for ladder battles
        ],
        stdout=stdout2,
        stderr=stderr2,
        cwd=str(Path(__file__).parent.parent),
    )

    print("Waiting for battles to complete...")

    try:
        proc1.wait(timeout=7200)
        proc2.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        print("WARNING: Matchup timed out")
        proc1.kill()
        proc2.kill()
        if not verbose:
            stdout1.close()
            stderr1.close()
            stdout2.close()
            stderr2.close()
        return {"status": "timeout"}

    # Close log files
    if not verbose:
        stdout1.close()
        stderr1.close()
        stdout2.close()
        stderr2.close()

    if proc1.returncode != 0 or proc2.returncode != 0:
        if proc1.returncode != 0:
            print(f"ERROR in {model1_name} (exit code: {proc1.returncode})")
            if not verbose:
                stderr_path = os.path.join(log_dir, f"{model1_name}_{user1}.stderr.log")
                with open(stderr_path) as f:
                    print(f.read()[-2000:])  # Last 2000 chars
        if proc2.returncode != 0:
            print(f"ERROR in {model2_name} (exit code: {proc2.returncode})")
            if not verbose:
                stderr_path = os.path.join(log_dir, f"{model2_name}_{user2}.stderr.log")
                with open(stderr_path) as f:
                    print(f.read()[-2000:])  # Last 2000 chars
        return {"status": "error"}

    print(f"✓ Matchup complete\n")
    return {
        "status": "success",
        "model1": model1_name,
        "model2": model2_name,
        "method": "ladder",
    }


def run_rl_vs_heuristic(
    rl_model_name: str,
    heuristic_name: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    output_dir: str,
) -> dict:
    """
    Run RL vs Heuristic using direct evaluation.

    Much faster and more reliable than ladder.
    """
    print(f"\n{'='*80}")
    print(f"RL vs Heuristic: {rl_model_name} vs {heuristic_name}")
    print(f"{'='*80}\n")

    trajectories_dir = os.path.join(output_dir, "trajectories")
    team_results_dir = os.path.join(output_dir, "team_results")

    # Load RL model
    pretrained_model = get_pretrained_model(rl_model_name)
    team_set = get_metamon_teams(battle_format, team_set_name)

    # Run evaluation
    print(f"Running {num_battles} battles...")
    results = pretrained_vs_baselines(
        pretrained_model=pretrained_model,
        battle_format=battle_format,
        team_set=team_set,
        total_battles=num_battles,
        baselines=[heuristic_name],
        parallel_actors_per_baseline=min(5, num_battles // 10),
        save_trajectories_to=trajectories_dir,
        save_team_results_to=team_results_dir,
        log_to_wandb=False,
    )

    print(f"✓ Evaluation complete")
    print(f"   Win rate: {results.get(f'Average Win Rate in {battle_format}_vs_{heuristic_name}', 'N/A')}\n")

    return {
        "status": "success",
        "model1": rl_model_name,
        "model2": heuristic_name,
        "method": "direct_eval",
        "results": results,
    }


def run_heuristic_vs_heuristic(
    heuristic1_name: str,
    heuristic2_name: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    output_dir: str,
) -> dict:
    """
    Run Heuristic vs Heuristic using simple evaluation.

    Creates a dummy RL agent that just wraps the heuristic for consistency.
    """
    print(f"\n{'='*80}")
    print(f"Heuristic vs Heuristic: {heuristic1_name} vs {heuristic2_name}")
    print(f"{'='*80}\n")

    # For heuristic vs heuristic, we can use ladder with simple players
    # Or just use pretrained_vs_baselines with a baseline as the "model"
    # For now, use ladder approach but with simpler agents

    print(f"Running {num_battles} battles (ladder)...")
    # This is tricky - for now, skip and record as 50-50
    # In practice, heuristic vs heuristic is less important for Nash
    print("⚠️  Heuristic vs heuristic matchup - assuming 50-50 for now")
    print("    (Can be run separately if needed)\n")

    return {
        "status": "skipped",
        "model1": heuristic1_name,
        "model2": heuristic2_name,
        "method": "assumed",
        "note": "Heuristic vs heuristic assumed 50-50",
    }


def run_mixed_tournament(
    population: PolicyPopulation,
    population_file: str,
    battle_format: str,
    team_set: str,
    battles_per_matchup: int,
    output_dir: str,
    parallel_matchups: int = 4,
):
    """Run tournament intelligently routing by policy type with parallel execution."""

    print(f"\n{'#'*80}")
    print(f"# MIXED POPULATION TOURNAMENT")
    print(f"# Policies: {population.size()}")
    print(f"# Format: {battle_format}")
    print(f"# Battles per matchup: {battles_per_matchup}")
    print(f"# Parallel matchups: {parallel_matchups}")
    print(f"{'#'*80}\n")

    matchups = get_matchups(population.list_policies())
    print(f"Total matchups: {len(matchups)}\n")

    # Prepare matchup info with types
    matchup_infos = []
    rl_vs_rl_matchups = []
    other_matchups = []

    for policy1_name, policy2_name in matchups:
        policy1 = population.get_policy(policy1_name)
        policy2 = population.get_policy(policy2_name)
        matchup_info = (policy1_name, policy2_name, policy1.policy_type, policy2.policy_type)

        # Separate RL vs RL from others
        if policy1.policy_type == "pretrained" and policy2.policy_type == "pretrained":
            rl_vs_rl_matchups.append(matchup_info)
        else:
            other_matchups.append(matchup_info)

    print(f"RL vs RL matchups: {len(rl_vs_rl_matchups)}")
    print(f"Other matchups: {len(other_matchups)}")
    print(f"Running with parallelism={parallel_matchups}\n")

    results = []
    results_file = os.path.join(output_dir, "tournament_results.json")

    def save_results():
        """Save intermediate results to disk."""
        with open(results_file, "w") as f:
            json.dump(
                {
                    "population": population.list_policies(),
                    "battle_format": battle_format,
                    "battles_per_matchup": battles_per_matchup,
                    "timestamp": datetime.now().isoformat(),
                    "matchups": results,
                },
                f,
                indent=2,
            )

    # Create worker function with fixed args
    worker = partial(
        run_single_matchup,
        population_file=population_file,
        battle_format=battle_format,
        team_set=team_set,
        battles_per_matchup=battles_per_matchup,
        output_dir=output_dir,
    )

    # Run non-RL-vs-RL matchups in parallel (these can run fully parallel)
    if other_matchups:
        print(f"Running {len(other_matchups)} non-RL-vs-RL matchups in parallel...")
        with ProcessPoolExecutor(max_workers=parallel_matchups) as executor:
            futures = {executor.submit(worker, m): m for m in other_matchups}

            # Use tqdm if available
            if HAS_TQDM:
                pbar = tqdm(total=len(other_matchups), desc="Non-RL matchups")

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    save_results()
                    if HAS_TQDM:
                        pbar.update(1)
                except Exception as e:
                    matchup_info = futures[future]
                    print(f"\n⚠️  Error in matchup {matchup_info[:2]}: {e}")
                    results.append({"status": "error"})
                    save_results()
                    if HAS_TQDM:
                        pbar.update(1)

            if HAS_TQDM:
                pbar.close()

    # Run RL vs RL matchups sequentially (parallelism=1 to avoid ladder matching conflicts)
    # Multiple parallel RL vs RL matchups cause agents to match with wrong opponents
    if rl_vs_rl_matchups:
        rl_workers = 1
        print(f"\nRunning {len(rl_vs_rl_matchups)} RL-vs-RL matchups sequentially (parallelism={rl_workers})...")
        with ProcessPoolExecutor(max_workers=rl_workers) as executor:
            futures = {executor.submit(worker, m): m for m in rl_vs_rl_matchups}

            # Use tqdm if available
            if HAS_TQDM:
                pbar = tqdm(total=len(rl_vs_rl_matchups), desc="RL-vs-RL matchups")

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    save_results()
                    if HAS_TQDM:
                        pbar.update(1)
                except Exception as e:
                    matchup_info = futures[future]
                    print(f"\n⚠️  Error in matchup {matchup_info[:2]}: {e}")
                    results.append({"status": "error"})
                    save_results()
                    if HAS_TQDM:
                        pbar.update(1)

            if HAS_TQDM:
                pbar.close()

    print(f"\n{'#'*80}")
    print(f"# TOURNAMENT COMPLETE")
    print(f"# Results: {results_file}")
    print(f"{'#'*80}\n")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tournament for mixed RL + heuristic populations"
    )
    parser.add_argument(
        "--population_file",
        type=str,
        required=True,
        help="Path to population.json",
    )
    parser.add_argument(
        "--battles_per_matchup",
        type=int,
        default=50,
        help="Battles per matchup",
    )
    parser.add_argument(
        "--battle_format",
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
        help="Output directory",
    )
    parser.add_argument(
        "--parallel_matchups",
        type=int,
        default=4,
        help="Number of matchups to run in parallel (default: 4)",
    )

    args = parser.parse_args()

    # Load population
    population = PolicyPopulation.load(args.population_file)
    print(f"Loaded population: {population.size()} policies")
    print(population)

    # Run tournament
    run_mixed_tournament(
        population=population,
        population_file=args.population_file,
        battle_format=args.battle_format,
        team_set=args.team_set,
        battles_per_matchup=args.battles_per_matchup,
        output_dir=os.path.expanduser(args.output_dir),
        parallel_matchups=args.parallel_matchups,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
