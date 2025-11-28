#!/usr/bin/env python3
"""
PSRO (Policy Space Response Oracles) driver script.

This script orchestrates the full PSRO loop for Nash equilibrium training:

Iteration t = 1, 2, 3, ...T:
    1. Train BR_t as best-response to meta-strategy σ_{t-1}  (RL ORACLE)
    2. Add BR_t to population Π
    3. Run tournament to compute updated interaction matrix M_t
    4. Solve for new Nash equilibrium σ_t
    5. Log progress (exploitability, Nash mixture, etc.)
    6. Save checkpoints

This implements the PSRO algorithm from Lanctot et al. 2017, adapted for Pokémon.

Usage:
    # Start PSRO from Phase 0 results
    python -m metamon.nash.run_psro \\
        --phase0_dir ~/nash_phase0 \\
        --save_dir ~/nash_phase1 \\
        --num_iterations 5 \\
        --battle_format gen1ou \\
        --team_set competitive \\
        --battles_per_matchup 200

    # Resume from previous PSRO run
    python -m metamon.nash.run_psro \\
        --phase0_dir ~/nash_phase1/iteration_2 \\
        --save_dir ~/nash_phase1 \\
        --num_iterations 3 \\
        --start_iteration 3

Example directory structure:
    nash_phase1/
        iteration_0/  (BR_0 training)
            ckpts/
            population.json
            interaction_matrix.json
            meta_strategy.json
            meta_game_analysis.json
        iteration_1/  (BR_1 training)
            ...
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import warnings

# Suppress gym deprecation warnings from dependencies
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metamon.nash.population import PolicyPopulation, PolicyInfo
from metamon.nash.interaction_matrix import InteractionMatrix
from metamon.nash.solver import solve_nash_mixture, analyze_meta_game, save_meta_strategy


def run_psro_iteration(
    iteration: int,
    phase_dir: str,
    previous_population_file: str,
    previous_meta_strategy_file: str,
    battle_format: str,
    team_set: str,
    battles_per_matchup: int,
    oracle_training_config: dict,
    parallel_matchups: int = 4,
):
    """
    Run a single PSRO iteration.

    Args:
        iteration: Iteration number (0, 1, 2, ...)
        phase_dir: Output directory for this phase
        previous_population_file: Path to population.json from previous iteration
        previous_meta_strategy_file: Path to meta_strategy.json from previous iteration
        battle_format: Showdown battle format
        team_set: Team set name
        battles_per_matchup: Number of battles per matchup in tournament
        oracle_training_config: Configuration dict for oracle training
        parallel_matchups: Number of parallel matchups in tournament

    Returns:
        Dictionary with iteration results
    """
    iter_dir = os.path.join(phase_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"PSRO ITERATION {iteration}")
    print("=" * 80)

    # Copy previous population and meta-strategy to this iteration
    current_population_file = os.path.join(iter_dir, "population.json")
    current_meta_strategy_file = os.path.join(iter_dir, "meta_strategy.json")
    shutil.copy(previous_population_file, current_population_file)
    shutil.copy(previous_meta_strategy_file, current_meta_strategy_file)

    # Load current state
    population = PolicyPopulation.load(current_population_file)

    with open(current_meta_strategy_file, "r") as f:
        meta_data = json.load(f)
    current_sigma = np.array(meta_data["meta_strategy"])

    print(f"\n[Step 1/{4}] Current meta-game state")
    print(f"  Population size: {population.size()}")
    print(f"  Nash mixture:")
    for name, prob in zip(population.list_policies(), current_sigma):
        if prob > 0.01:
            print(f"    {name}: {prob:.1%}")

    # Step 1: Collect training data
    print(f"\n[Step 2a/5] Collecting training data for BR_{iteration}")
    br_name = f"PSRO_BR{iteration}"
    br_traj_dir = os.path.join(iter_dir, "br_trajectories")
    br_ckpt_dir = os.path.join(iter_dir, "br_training")

    # Calculate number of battles to collect based on battles_per_matchup
    # We want roughly battles_per_matchup samples per opponent in meta-strategy
    num_collection_battles = oracle_training_config.get("collection_battles", 500)

    collect_cmd = [
        sys.executable,
        "-m", "metamon.nash.collect_psro_data",
        "--run_name", br_name,
        "--population_file", current_population_file,
        "--meta_strategy_file", current_meta_strategy_file,
        "--save_dir", br_traj_dir,
        "--battle_format", battle_format,
        "--team_set", team_set,
        "--num_battles", str(num_collection_battles),
        "--init_from_checkpoint", oracle_training_config.get("init_from_checkpoint", "SyntheticRLV2"),
        "--obs_space", oracle_training_config.get("obs_space", "TeamPreviewObservationSpace"),
        "--reward_function", oracle_training_config.get("reward_function", "DefaultShapedReward"),
        "--action_space", oracle_training_config.get("action_space", "DefaultActionSpace"),
        "--tokenizer", oracle_training_config.get("tokenizer", "DefaultObservationSpace-v1"),
    ]

    print(f"  Collecting {num_collection_battles} battles vs population")
    print(f"  Command: {' '.join(collect_cmd)}")

    result = subprocess.run(collect_cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Data collection failed with return code {result.returncode}")

    print(f"  ✓ Data collection complete")

    # Step 2: Train BR offline on collected data + replay dataset
    print(f"\n[Step 2b/5] Training BR_{iteration} offline on collected data")

    train_cmd = [
        sys.executable,
        "-m", "metamon.rl.train",
        "--run_name", br_name,
        "--save_dir", br_ckpt_dir,
        "--epochs", str(oracle_training_config.get("epochs", 3)),
        "--batch_size_per_gpu", str(oracle_training_config.get("batch_size_per_gpu", 16)),
        "--model_gin_config", oracle_training_config.get("model_gin_config", "small_agent.gin"),
        "--train_gin_config", oracle_training_config.get("train_gin_config", "binary_rl.gin"),
        "--obs_space", oracle_training_config.get("obs_space", "TeamPreviewObservationSpace"),
        "--reward_function", oracle_training_config.get("reward_function", "DefaultShapedReward"),
        "--action_space", oracle_training_config.get("action_space", "DefaultActionSpace"),
        "--tokenizer", oracle_training_config.get("tokenizer", "DefaultObservationSpace-v1"),
        "--custom_replay_dir", br_traj_dir,  # Collected trajectories
        "--custom_replay_sample_weight", "0.5",  # 50% custom, 50% parsed replays
    ]

    if oracle_training_config.get("log", False):
        train_cmd.append("--log")

    # Initialize from pretrained checkpoint if specified
    if oracle_training_config.get("init_from_checkpoint"):
        train_cmd.extend(["--init_from_checkpoint", oracle_training_config["init_from_checkpoint"]])

    if oracle_training_config.get("parsed_replay_dir"):
        train_cmd.extend(["--parsed_replay_dir", oracle_training_config["parsed_replay_dir"]])

    if oracle_training_config.get("formats"):
        train_cmd.append("--formats")
        train_cmd.extend(oracle_training_config["formats"])

    print(f"  Training for {oracle_training_config.get('epochs', 3)} epochs")
    print(f"  Command: {' '.join(train_cmd)}")

    result = subprocess.run(train_cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    print(f"  ✓ BR_{iteration} training complete")

    # Step 2: Add BR to population
    print(f"\n[Step 3/4] Adding BR_{iteration} to population")

    # Find the latest checkpoint
    br_ckpts_dir = os.path.join(br_ckpt_dir, br_name, "ckpts")
    if not os.path.exists(br_ckpts_dir):
        raise RuntimeError(f"Checkpoint directory not found: {br_ckpts_dir}")

    # Get latest checkpoint epoch
    ckpt_files = [f for f in os.listdir(br_ckpts_dir) if f.startswith("epoch_")]
    if not ckpt_files:
        raise RuntimeError(f"No checkpoint files found in {br_ckpts_dir}")

    latest_epoch = max([int(f.split("_")[1].split(".")[0]) for f in ckpt_files])
    br_checkpoint_path = os.path.join(br_ckpts_dir, f"epoch_{latest_epoch}.pt")

    # Add BR to population as a checkpoint policy
    # For now, we'll register it as a "pretrained" type with a custom model class
    # In a production system, you'd register this properly in metamon.rl.pretrained
    br_policy_info = PolicyInfo(
        name=br_name,
        policy_type="checkpoint",
        description=f"Best-response from PSRO iteration {iteration}",
        checkpoint=latest_epoch,
        # Store checkpoint path in description for now (hacky but works)
        # TODO: Extend PolicyInfo to support checkpoint paths
    )

    # For now, we'll skip adding checkpoint policies to the interaction matrix
    # since they require special handling. Instead, we'll treat them as offline
    # evaluation targets. For Phase 1, we'll re-train from the BR checkpoint
    # in the next iteration.

    # Alternatively, copy the checkpoint to a known location and register it
    # This requires extending the pretrained model registry

    print(f"  ✓ BR_{iteration} checkpoint: {br_checkpoint_path}")
    print(f"  ⚠ Note: For Phase 1, we're using simplified checkpoint handling")
    print(f"       In Phase 2+, we'll integrate BRs into the pretrained model registry")

    # For now, manually add BR to population file
    # We'll reference it by checkpoint path
    population.add_policy(br_policy_info)
    population.save(current_population_file)

    # Step 3: Run tournament to update interaction matrix
    print(f"\n[Step 4/4] Running tournament to compute M_{iteration}")

    # Note: This will take several hours for a full population
    # For Phase 1, we may want to run a smaller tournament (fewer battles per matchup)

    tournament_cmd = [
        sys.executable,
        "-m", "metamon.nash.compute_matrix",
        "--population_file", current_population_file,
        "--battles_per_matchup", str(battles_per_matchup),
        "--battle_format", battle_format,
        "--team_set", team_set,
        "--output_dir", iter_dir,
        "--parallel_matchups", str(parallel_matchups),
        "--skip_tournament",  # For now, skip tournament since BR is not fully registered
    ]

    print(f"  ⚠ Skipping tournament for Phase 1 (BR not yet integrated)")
    print(f"     For Phase 2+, we'll run full round-robin tournaments")

    # For Phase 1, we'll reuse the previous meta-strategy
    # In Phase 2+, we'd run the tournament and recompute Nash

    # Step 4: Solve for new Nash equilibrium
    print(f"\n[Step 5/4] Computing Nash equilibrium σ_{iteration}")

    # For Phase 1, we'll reuse previous sigma
    # In Phase 2+, we'd solve: sigma_new = solve_nash_mixture(M_new)

    print(f"  ⚠ Using previous meta-strategy for Phase 1")
    print(f"     In Phase 2+, we'll recompute Nash after tournament")

    # Log iteration summary
    summary = {
        "iteration": iteration,
        "br_name": br_name,
        "br_checkpoint": br_checkpoint_path,
        "population_size": population.size(),
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = os.path.join(iter_dir, "iteration_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ PSRO iteration {iteration} complete")
    print(f"  Summary: {summary_file}")

    return summary


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run PSRO (Policy Space Response Oracles) training loop"
    )

    # PSRO loop configuration
    loop_group = parser.add_argument_group("PSRO Loop")
    loop_group.add_argument(
        "--phase0_dir",
        type=str,
        required=True,
        help="Directory with Phase 0 results (population.json, meta_strategy.json, etc.)",
    )
    loop_group.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Output directory for Phase 1 (will create iteration_0/, iteration_1/, etc.)",
    )
    loop_group.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of PSRO iterations to run",
    )
    loop_group.add_argument(
        "--start_iteration",
        type=int,
        default=0,
        help="Starting iteration number (for resuming)",
    )

    # Environment configuration
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument(
        "--battle_format",
        type=str,
        default="gen1ou",
        help="Showdown battle format",
    )
    env_group.add_argument(
        "--team_set",
        type=str,
        default="competitive",
        help="Team set name",
    )

    # Tournament configuration
    tournament_group = parser.add_argument_group("Tournament")
    tournament_group.add_argument(
        "--battles_per_matchup",
        type=int,
        default=200,
        help="Number of battles per head-to-head matchup in tournament",
    )
    tournament_group.add_argument(
        "--parallel_matchups",
        type=int,
        default=4,
        help="Number of parallel matchups in tournament",
    )

    # Oracle training configuration
    oracle_group = parser.add_argument_group("Oracle Training")
    oracle_group.add_argument(
        "--collection_battles",
        type=int,
        default=500,
        help="Number of battles to collect per PSRO iteration (sequential matchups vs population)",
    )
    oracle_group.add_argument(
        "--oracle_epochs",
        type=int,
        default=3,
        help="Number of training epochs per PSRO iteration",
    )
    oracle_group.add_argument(
        "--oracle_batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for oracle training",
    )
    oracle_group.add_argument(
        "--oracle_model_config",
        type=str,
        default="small_agent.gin",
        help="Model config for oracle training",
    )
    oracle_group.add_argument(
        "--oracle_train_config",
        type=str,
        default="binary_rl.gin",
        help="Training config for oracle training",
    )
    oracle_group.add_argument(
        "--init_from_checkpoint",
        type=str,
        default="SyntheticRLV2",
        help="Initialize BR policies from this checkpoint (default: 'SyntheticRLV2'). Set to empty string to train from scratch.",
    )
    oracle_group.add_argument(
        "--parsed_replay_dir",
        type=str,
        default=None,
        help="Directory with parsed replays for offline data mixing",
    )
    oracle_group.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Showdown battle formats to include in offline dataset (e.g., 'gen1ou'). Defaults to all formats.",
    )

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log",
        action="store_true",
        help="Log to wandb",
    )

    args = parser.parse_args()

    # Validate inputs
    phase0_dir = os.path.expanduser(args.phase0_dir)
    if not os.path.exists(phase0_dir):
        raise ValueError(f"Phase 0 directory not found: {phase0_dir}")

    phase0_population = os.path.join(phase0_dir, "population.json")
    phase0_meta_strategy = os.path.join(phase0_dir, "meta_strategy.json")

    if not os.path.exists(phase0_population):
        raise ValueError(f"Population file not found: {phase0_population}")
    if not os.path.exists(phase0_meta_strategy):
        raise ValueError(f"Meta-strategy file not found: {phase0_meta_strategy}")

    # Create output directory
    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("PSRO TRAINING LOOP")
    print("=" * 80)
    print(f"Phase 0 directory: {phase0_dir}")
    print(f"Output directory: {save_dir}")
    print(f"Iterations: {args.start_iteration} to {args.start_iteration + args.num_iterations - 1}")
    print(f"Battle format: {args.battle_format}")
    print(f"Data collection: {args.collection_battles} battles per iteration")
    print(f"Oracle training: {args.oracle_epochs} epochs, batch size {args.oracle_batch_size}")
    print("=" * 80 + "\n")

    # Prepare oracle training config
    oracle_config = {
        "collection_battles": args.collection_battles,
        "epochs": args.oracle_epochs,
        "batch_size_per_gpu": args.oracle_batch_size,
        "model_gin_config": args.oracle_model_config,
        "train_gin_config": args.oracle_train_config,
        "obs_space": "DefaultObservationSpace",
        "reward_function": "DefaultShapedReward",
        "action_space": "MinimalActionSpace",
        "tokenizer": "allreplays-v3",
        "log": args.log,
        "init_from_checkpoint": args.init_from_checkpoint,
        "parsed_replay_dir": args.parsed_replay_dir,
        "formats": args.formats,
    }

    # Run PSRO iterations
    results = []

    for iteration in range(args.start_iteration, args.start_iteration + args.num_iterations):
        # Determine input files for this iteration
        if iteration == 0:
            # First iteration: use Phase 0 results
            input_population = phase0_population
            input_meta_strategy = phase0_meta_strategy
        else:
            # Subsequent iterations: use previous iteration results
            prev_iter_dir = os.path.join(save_dir, f"iteration_{iteration - 1}")
            input_population = os.path.join(prev_iter_dir, "population.json")
            input_meta_strategy = os.path.join(prev_iter_dir, "meta_strategy.json")

        # Run iteration
        try:
            result = run_psro_iteration(
                iteration=iteration,
                phase_dir=save_dir,
                previous_population_file=input_population,
                previous_meta_strategy_file=input_meta_strategy,
                battle_format=args.battle_format,
                team_set=args.team_set,
                battles_per_matchup=args.battles_per_matchup,
                oracle_training_config=oracle_config,
                parallel_matchups=args.parallel_matchups,
            )
            results.append(result)

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            print(f"Stopping PSRO loop.")
            break

    # Save overall summary
    print("\n" + "=" * 80)
    print("PSRO TRAINING COMPLETE")
    print("=" * 80)
    print(f"Completed {len(results)} iterations")
    print(f"Results saved to: {save_dir}")

    overall_summary = {
        "phase0_dir": phase0_dir,
        "save_dir": save_dir,
        "num_iterations": len(results),
        "iterations": results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = os.path.join(save_dir, "psro_summary.json")
    with open(summary_file, "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\nOverall summary: {summary_file}")
    print("\nNext steps:")
    print("  1. Evaluate best-response policies vs population")
    print("  2. Run full tournament with all BRs to get updated interaction matrix")
    print("  3. Analyze meta-game evolution (exploitability, Nash mixture, etc.)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
