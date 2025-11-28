#!/usr/bin/env python3
"""
Iterative vanilla self-play training loop with optional dynamic damping.

This script implements a simple self-play loop:
    1. Collect self-play data from current checkpoint
    2. Train on collected data (with optional dynamic damping)
    3. Save new checkpoint
    4. Evaluate against fixed opponents
    5. Repeat

This is useful for testing whether dynamic damping alone is sufficient to improve
agent performance through self-play, before moving to more complex PSRO setups.

Usage:
    # Baseline (no damping)
    python -m metamon.rl.train_vanilla_selfplay \
        --run_name VanillaSelfPlay_Baseline \
        --init_checkpoint SyntheticRLV2 \
        --num_iterations 10 \
        --epochs_per_iteration 20 \
        --episodes_per_iteration 5000 \
        --save_dir ~/selfplay_baseline \
        --train_gin_config vanilla_selfplay_baseline.gin

    # With dynamic damping
    python -m metamon.rl.train_vanilla_selfplay \
        --run_name VanillaSelfPlay_Damped \
        --init_checkpoint SyntheticRLV2 \
        --num_iterations 10 \
        --epochs_per_iteration 20 \
        --episodes_per_iteration 5000 \
        --save_dir ~/selfplay_damped \
        --train_gin_config vanilla_selfplay_damped.gin
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def collect_selfplay_data(
    iteration: int,
    checkpoint_path: str,
    save_dir: str,
    num_battles: int,
    battle_format: str,
    team_set: str,
    model_class: str,
    run_name: str,
) -> str:
    """Collect self-play data from current checkpoint.

    Returns:
        Path to collected data directory
    """
    data_dir = os.path.join(save_dir, f"iteration_{iteration}", "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}: COLLECTING SELF-PLAY DATA")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num battles: {num_battles}")
    print(f"Saving to: {data_dir}\n")

    cmd = [
        sys.executable,
        "-m", "metamon.rl.collect_selfplay_data",
        "--run_name", f"{run_name}_iter{iteration}",
        "--checkpoint_path", checkpoint_path,
        "--save_dir", data_dir,
        "--battle_format", battle_format,
        "--team_set", team_set,
        "--num_battles", str(num_battles),
        "--model_class", model_class,
    ]

    result = subprocess.run(cmd, check=True)

    if result.returncode != 0:
        raise RuntimeError(f"Data collection failed with return code {result.returncode}")

    print(f"\n✓ Data collection completed: {data_dir}\n")
    return data_dir


def train_on_data(
    iteration: int,
    data_dir: str,
    save_dir: str,
    init_checkpoint: str,
    epochs: int,
    model_gin_config: str,
    train_gin_config: str,
    run_name: str,
    obs_space: str,
    reward_function: str,
    action_space: str,
    tokenizer: str,
    battle_format: str,
    team_set: str,
) -> str:
    """Train on collected self-play data.

    Returns:
        Path to trained checkpoint
    """
    checkpoint_dir = os.path.join(save_dir, f"iteration_{iteration}", "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}: TRAINING")
    print(f"{'='*80}")
    print(f"Data dir: {data_dir}")
    print(f"Training for {epochs} epochs")
    print(f"Train config: {train_gin_config}")
    print(f"Saving to: {checkpoint_dir}\n")

    cmd = [
        sys.executable,
        "-m", "metamon.rl.train",
        "--run_name", f"{run_name}_iter{iteration}",
        "--save_dir", checkpoint_dir,
        "--data_dir", data_dir,
        "--model_gin_config", model_gin_config,
        "--train_gin_config", train_gin_config,
        "--epochs", str(epochs),
        "--obs_space", obs_space,
        "--reward_function", reward_function,
        "--action_space", action_space,
        "--tokenizer", tokenizer,
        "--battle_format", battle_format,
        "--team_set", team_set,
    ]

    # Add checkpoint to initialize from
    if init_checkpoint:
        cmd.extend(["--init_from_checkpoint", init_checkpoint])

    result = subprocess.run(cmd, check=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    print(f"\n✓ Training completed: {checkpoint_dir}\n")
    return checkpoint_dir


def evaluate_checkpoint(
    iteration: int,
    checkpoint_path: str,
    save_dir: str,
    battle_format: str,
    team_set: str,
    num_eval_battles: int,
) -> dict:
    """Evaluate checkpoint against fixed opponents.

    Returns:
        Dict of evaluation metrics
    """
    eval_dir = os.path.join(save_dir, f"iteration_{iteration}", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}: EVALUATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval battles: {num_eval_battles}\n")

    # TODO: Implement actual evaluation logic
    # For now, just create placeholder results
    results = {
        "iteration": iteration,
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "eval_battles": num_eval_battles,
        # Placeholder metrics - should be filled by actual evaluation
        "win_rate_vs_synrl_v2": None,
        "win_rate_vs_maxbasepower": None,
        "win_rate_vs_random": None,
    }

    # Save results
    results_file = os.path.join(eval_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Evaluation results saved: {results_file}\n")
    return results


def main():
    parser = ArgumentParser(description="Iterative vanilla self-play training")

    # Run configuration
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name for this self-play run"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Root directory to save all results"
    )

    # Iteration parameters
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of self-play iterations"
    )
    parser.add_argument(
        "--epochs_per_iteration",
        type=int,
        default=20,
        help="Training epochs per iteration"
    )
    parser.add_argument(
        "--episodes_per_iteration",
        type=int,
        default=5000,
        help="Self-play episodes to collect per iteration"
    )
    parser.add_argument(
        "--num_eval_battles",
        type=int,
        default=100,
        help="Number of evaluation battles per iteration"
    )

    # Model configuration
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="SyntheticRLV2",
        help="Initial checkpoint to start from (model class or path)"
    )
    parser.add_argument(
        "--model_gin_config",
        type=str,
        default="small_agent.gin",
        help="Model architecture gin config"
    )
    parser.add_argument(
        "--train_gin_config",
        type=str,
        default="vanilla_selfplay_baseline.gin",
        help="Training hyperparameters gin config"
    )

    # Environment configuration
    parser.add_argument(
        "--battle_format",
        type=str,
        default="gen1ou",
        help="Pokemon battle format"
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="modern_replays_v2",
        help="Team set for battles"
    )
    parser.add_argument(
        "--obs_space",
        type=str,
        default="TeamPreviewObservationSpace",
        help="Observation space class"
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default="DefaultShapedReward",
        help="Reward function class"
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="DefaultActionSpace",
        help="Action space class"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="DefaultObservationSpace-v1",
        help="Tokenizer version"
    )

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save configuration
    config_file = os.path.join(args.save_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "="*80)
    print("VANILLA SELF-PLAY TRAINING")
    print("="*80)
    print(f"Run name: {args.run_name}")
    print(f"Save directory: {args.save_dir}")
    print(f"Initial checkpoint: {args.init_checkpoint}")
    print(f"Num iterations: {args.num_iterations}")
    print(f"Episodes per iteration: {args.episodes_per_iteration}")
    print(f"Epochs per iteration: {args.epochs_per_iteration}")
    print(f"Train config: {args.train_gin_config}")
    print(f"Model config: {args.model_gin_config}")
    print("="*80 + "\n")

    # Track all results
    all_results = []

    # Current checkpoint starts as init checkpoint
    current_checkpoint = args.init_checkpoint

    # Iterative self-play loop
    for iteration in range(args.num_iterations):
        print(f"\n\n{'#'*80}")
        print(f"# ITERATION {iteration} / {args.num_iterations}")
        print(f"{'#'*80}\n")

        try:
            # Step 1: Collect self-play data
            data_dir = collect_selfplay_data(
                iteration=iteration,
                checkpoint_path=current_checkpoint,
                save_dir=args.save_dir,
                num_battles=args.episodes_per_iteration,
                battle_format=args.battle_format,
                team_set=args.team_set,
                model_class="SyntheticRLV2",  # Model class for loading
                run_name=args.run_name,
            )

            # Step 2: Train on collected data
            checkpoint_dir = train_on_data(
                iteration=iteration,
                data_dir=data_dir,
                save_dir=args.save_dir,
                init_checkpoint=current_checkpoint,
                epochs=args.epochs_per_iteration,
                model_gin_config=args.model_gin_config,
                train_gin_config=args.train_gin_config,
                run_name=args.run_name,
                obs_space=args.obs_space,
                reward_function=args.reward_function,
                action_space=args.action_space,
                tokenizer=args.tokenizer,
                battle_format=args.battle_format,
                team_set=args.team_set,
            )

            # Step 3: Evaluate new checkpoint
            eval_results = evaluate_checkpoint(
                iteration=iteration,
                checkpoint_path=checkpoint_dir,
                save_dir=args.save_dir,
                battle_format=args.battle_format,
                team_set=args.team_set,
                num_eval_battles=args.num_eval_battles,
            )

            all_results.append(eval_results)

            # Update current checkpoint for next iteration
            current_checkpoint = checkpoint_dir

            print(f"\n{'='*80}")
            print(f"✓ ITERATION {iteration} COMPLETED")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n✗ ITERATION {iteration} FAILED: {e}")
            print("Stopping self-play loop.")
            break

    # Save aggregate results
    summary_file = os.path.join(args.save_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "config": vars(args),
            "iterations_completed": len(all_results),
            "results": all_results,
        }, f, indent=2)

    print("\n" + "="*80)
    print("VANILLA SELF-PLAY COMPLETED")
    print("="*80)
    print(f"Iterations completed: {len(all_results)} / {args.num_iterations}")
    print(f"Results saved to: {args.save_dir}")
    print(f"Summary: {summary_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
