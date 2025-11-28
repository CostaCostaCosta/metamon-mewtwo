#!/usr/bin/env python3
"""
Collect PSRO training data by running sequential matchups vs population.

This script:
1. Loads population and meta-strategy
2. For each battle, samples opponent from σ
3. Runs 1v1 matchup on ladder (BR agent vs opponent)
4. Saves trajectories to disk

Usage:
    python -m metamon.nash.collect_psro_data \
        --run_name PSRO_BR0 \
        --population_file ~/nash_phase0/population.json \
        --meta_strategy_file ~/nash_phase0/meta_strategy.json \
        --save_dir ~/nash_phase1/iteration_0/trajectories \
        --battle_format gen1ou \
        --team_set modern_replays_v2 \
        --num_battles 1000 \
        --init_from_checkpoint SyntheticRLV2
"""

import os
import sys
import json
import subprocess
import time
import signal
from pathlib import Path
from typing import List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metamon.nash.population import PolicyPopulation
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.evaluate import pretrained_vs_local_ladder
from metamon.env import get_metamon_teams
from metamon.interface import (
    TokenizedObservationSpace,
    get_observation_space,
    get_action_space,
    get_reward_function,
)
from metamon.tokenizer import get_tokenizer


def sample_opponent_from_population(
    population: PolicyPopulation,
    meta_strategy: np.ndarray,
) -> str:
    """Sample opponent name from population according to meta-strategy."""
    policy_names = population.list_policies()
    return np.random.choice(policy_names, p=meta_strategy)


def launch_opponent_on_ladder(
    policy_name: str,
    population: PolicyPopulation,
    battle_format: str,
    team_set_name: str,
    username: str,
    num_battles: int,
    output_dir: str,
) -> subprocess.Popen:
    """Launch a single opponent process on ladder."""
    policy = population.get_policy(policy_name)
    opponent_script = Path(__file__).parent.parent.parent / "scripts" / "run_matchup.py"

    if policy.policy_type == "pretrained":
        cmd = [
            sys.executable,
            str(opponent_script),
            "--model", policy.model_class,
            "--format", battle_format,
            "--team_set", team_set_name,
            "--username", username,
            "--battles", str(num_battles),
            "--output_dir", output_dir,
        ]
    elif policy.policy_type == "heuristic":
        cmd = [
            sys.executable,
            str(opponent_script),
            "--baseline", policy.baseline_class,
            "--format", battle_format,
            "--team_set", team_set_name,
            "--username", username,
            "--battles", str(num_battles),
            "--output_dir", output_dir,
        ]
    else:
        raise ValueError(f"Unknown policy type: {policy.policy_type}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Collect PSRO training data via sequential matchups")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--population_file", type=str, required=True)
    parser.add_argument("--meta_strategy_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--battle_format", type=str, default="gen1ou")
    parser.add_argument("--team_set", type=str, default="modern_replays_v2")
    parser.add_argument("--num_battles", type=int, default=1000, help="Total battles to collect")
    parser.add_argument("--init_from_checkpoint", type=str, default="SyntheticRLV2")
    parser.add_argument("--obs_space", type=str, default="TeamPreviewObservationSpace")
    parser.add_argument("--reward_function", type=str, default="DefaultShapedReward")
    parser.add_argument("--action_space", type=str, default="DefaultActionSpace")
    parser.add_argument("--tokenizer", type=str, default="DefaultObservationSpace-v1")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*80)
    print("PSRO DATA COLLECTION (Sequential Matchups)")
    print("="*80)
    print(f"Run name: {args.run_name}")
    print(f"Population: {args.population_file}")
    print(f"Meta-strategy: {args.meta_strategy_file}")
    print(f"Collecting {args.num_battles} battles")
    print(f"Save dir: {args.save_dir}")
    print("="*80 + "\n")

    # Load population and meta-strategy
    population = PolicyPopulation.load(args.population_file)

    with open(args.meta_strategy_file, "r") as f:
        meta_data = json.load(f)
    meta_strategy = np.array(meta_data["meta_strategy"])
    policy_names = meta_data["policy_names"]

    print(f"Population: {policy_names}")
    print(f"Meta-strategy: {dict(zip(policy_names, meta_strategy))}\n")

    # Sample opponents for all battles upfront
    opponent_samples = [
        sample_opponent_from_population(population, meta_strategy)
        for _ in range(args.num_battles)
    ]

    # Count distribution
    from collections import Counter
    sample_counts = Counter(opponent_samples)
    print("Sampled opponent distribution:")
    for name, count in sample_counts.items():
        print(f"  {name}: {count} battles ({count/args.num_battles:.1%})")
    print()

    # Group consecutive same-opponent battles for efficiency
    battle_groups = []
    current_opponent = opponent_samples[0]
    current_count = 1

    for opponent in opponent_samples[1:]:
        if opponent == current_opponent:
            current_count += 1
        else:
            battle_groups.append((current_opponent, current_count))
            current_opponent = opponent
            current_count = 1
    battle_groups.append((current_opponent, current_count))

    print(f"Running {len(battle_groups)} matchup groups (batching consecutive same-opponent battles)\n")

    # Run matchups sequentially
    total_battles_collected = 0

    for group_idx, (opponent_name, num_battles_in_group) in enumerate(battle_groups):
        print(f"[{group_idx+1}/{len(battle_groups)}] Running {num_battles_in_group} battles vs {opponent_name}")

        # Launch opponent
        opponent_username = f"OPP_{opponent_name[:6]}_{int(time.time())%10000}"
        opponent_output = f"/tmp/psro_opponent_{int(time.time())}"
        os.makedirs(opponent_output, exist_ok=True)

        opponent_proc = launch_opponent_on_ladder(
            policy_name=opponent_name,
            population=population,
            battle_format=args.battle_format,
            team_set_name=args.team_set,
            username=opponent_username,
            num_battles=num_battles_in_group,
            output_dir=opponent_output,
        )

        print(f"  Launched opponent: {opponent_username}")
        time.sleep(2)  # Give opponent time to connect

        # Launch BR agent
        br_username = f"BR_{args.run_name}_{int(time.time())%10000}"
        br_model = get_pretrained_model(args.init_from_checkpoint)

        print(f"  Launching BR agent: {br_username}")
        print(f"  Saving trajectories to: {args.save_dir}")

        # Run BR agent on ladder
        try:
            pretrained_vs_local_ladder(
                pretrained_model=br_model,
                username=br_username,
                battle_format=args.battle_format,
                team_set=get_metamon_teams(args.battle_format, args.team_set),
                total_battles=num_battles_in_group,
                battle_backend="poke-env",
                save_trajectories_to=args.save_dir,
            )
            total_battles_collected += num_battles_in_group
            print(f"  ✓ Completed {num_battles_in_group} battles")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        finally:
            # Kill opponent
            try:
                opponent_proc.terminate()
                opponent_proc.wait(timeout=5)
            except:
                try:
                    opponent_proc.kill()
                except:
                    pass

        print()

    print("="*80)
    print(f"✓ Data collection complete!")
    print(f"  Total battles collected: {total_battles_collected}/{args.num_battles}")
    print(f"  Trajectories saved to: {args.save_dir}")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
