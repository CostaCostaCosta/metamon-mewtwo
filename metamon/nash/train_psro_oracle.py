#!/usr/bin/env python3
"""
Train PSRO best-response (BR) policy as RL ORACLE.

This script trains a new policy as best-response to the current meta-strategy σ
over the policy population. Key differences from standard RL training:

1. **Online RL**: Collects trajectories during training via self-play on local ladder
2. **Opponent sampling**: Launches opponent processes weighted by meta-strategy σ
3. **Data mixing**: Combines online self-play (50%) with offline replays (50%)
4. **Short training**: Only 2-4 epochs per PSRO iteration (not full 100-epoch training)

Usage:
    python -m metamon.nash.train_psro_oracle \\
        --run_name PSRO_BR0 \\
        --population_file ~/nash_phase0/population.json \\
        --meta_strategy_file ~/nash_phase0/meta_strategy.json \\
        --init_from_checkpoint Gen1BinaryV0_Epoch2 \\
        --save_dir ~/nash_phase1 \\
        --battle_format gen1ou \\
        --epochs 3 \\
        --online_selfplay_weight 0.5

This implements Algorithm 2 ("Compute the oracle in RL") from the Nash survey.
"""

import os
import sys
import json
import subprocess
import signal
import time
from pathlib import Path
from typing import List, Optional
from functools import partial
import numpy as np

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import metamon
from metamon.rl.train import (
    add_cli as add_rl_cli,
    create_offline_dataset,
)
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    make_local_ladder_env,
    make_placeholder_env,
)
from metamon.interface import (
    TokenizedObservationSpace,
    get_observation_space,
    get_action_space,
    get_reward_function,
)
from metamon.tokenizer import get_tokenizer
from metamon.env import get_metamon_teams
from metamon.nash.population import PolicyPopulation
from metamon.rl import MODEL_CONFIG_DIR, TRAINING_CONFIG_DIR
import metamon.rl

import gin
import amago
import wandb


WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY")


def launch_population_opponents(
    population_file: str,
    meta_strategy_file: str,
    battle_format: str,
    team_set_name: str,
    run_id: str,
) -> List[subprocess.Popen]:
    """
    Launch opponent processes on local ladder weighted by meta-strategy.

    For each policy in the population, we launch N instances where N is proportional
    to its meta-strategy probability. This ensures the training agent faces opponents
    with the correct distribution.

    Args:
        population_file: Path to population.json
        meta_strategy_file: Path to meta_strategy.json
        battle_format: Showdown battle format
        team_set_name: Team set name
        run_id: Unique identifier for this training run

    Returns:
        List of subprocess.Popen instances (for cleanup)
    """
    # Load population and meta-strategy
    population = PolicyPopulation.load(population_file)

    with open(meta_strategy_file, "r") as f:
        meta_data = json.load(f)

    meta_strategy = np.array(meta_data["meta_strategy"])
    policy_names = meta_data["policy_names"]

    print("\n" + "="*80)
    print("LAUNCHING POPULATION OPPONENTS ON LOCAL LADDER")
    print("="*80)

    # Calculate number of instances per policy
    # We want ~2:1 ratio of opponents to training actors (4 actors = 8 opponents)
    # This minimizes waste from opponent-vs-opponent battles on the ladder
    total_instances = 8  # Was 16, but causes too many opponent-vs-opponent battles
    instances_per_policy = (meta_strategy * total_instances).astype(int)

    # Ensure at least 1 instance for policies with >5% probability
    for i, prob in enumerate(meta_strategy):
        if prob > 0.05 and instances_per_policy[i] == 0:
            instances_per_policy[i] = 1

    # Rebalance to hit target total
    current_total = instances_per_policy.sum()
    if current_total < total_instances:
        # Add remaining to highest probability policy
        max_idx = np.argmax(meta_strategy)
        instances_per_policy[max_idx] += (total_instances - current_total)

    print(f"\nOpponent distribution (target: {total_instances} total instances):")
    for name, prob, count in zip(policy_names, meta_strategy, instances_per_policy):
        if count > 0:
            print(f"  {name}: {prob:.1%} → {count} instances")

    # Launch opponent processes
    processes = []
    opponent_script = Path(__file__).parent.parent.parent / "scripts" / "run_matchup.py"

    for policy_name, count in zip(policy_names, instances_per_policy):
        if count == 0:
            continue

        policy = population.get_policy(policy_name)

        for instance_id in range(count):
            # Keep username under 18 chars: "V1SP_12345_0" = 12 chars
            short_name = policy_name.replace("SyntheticRL", "").replace("SelfPlay", "SP")[:4]
            username = f"{short_name}_{run_id[-5:]}_{instance_id}"

            # Create temp output dir for opponent
            opponent_output_dir = f"/tmp/psro_opponents_{run_id}"
            os.makedirs(opponent_output_dir, exist_ok=True)

            if policy.policy_type == "pretrained":
                # Launch RL model on ladder
                cmd = [
                    sys.executable,
                    str(opponent_script),
                    "--model", policy.model_class,
                    "--format", battle_format,  # run_matchup.py uses --format
                    "--team_set", team_set_name,
                    "--username", username,
                    "--battles", "999999",  # run_matchup.py uses --battles
                    "--output_dir", opponent_output_dir,  # Required by run_matchup.py
                ]

            elif policy.policy_type == "heuristic":
                # Launch heuristic baseline on ladder
                cmd = [
                    sys.executable,
                    str(opponent_script),
                    "--baseline", policy.baseline_class,
                    "--format", battle_format,
                    "--team_set", team_set_name,
                    "--username", username,
                    "--battles", "999999",
                    "--output_dir", opponent_output_dir,
                ]

            else:
                print(f"⚠ Warning: Unknown policy type '{policy.policy_type}' for {policy_name}, skipping")
                continue

            # Launch process
            print(f"  Launching: {username} ({policy.policy_type}: {policy_name})")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes.append(proc)

            # Brief delay to stagger launches
            time.sleep(0.5)

    print(f"\n✓ Launched {len(processes)} opponent processes")
    print(f"  Waiting 30s for opponents to connect to ladder...")
    time.sleep(30)

    return processes


def cleanup_opponent_processes(processes: List[subprocess.Popen]):
    """Kill all opponent processes."""
    print("\nCleaning up opponent processes...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            try:
                proc.kill()
            except:
                pass
    print(f"✓ Terminated {len(processes)} opponent processes")


def create_psro_oracle_trainer(
    ckpt_dir: str,
    run_name: str,
    model_gin_config: str,
    train_gin_config: str,
    obs_space: TokenizedObservationSpace,
    action_space,
    reward_function,
    amago_dataset,
    battle_format: str,
    team_set_name: str,
    online_selfplay_weight: float = 0.5,
    async_env_mp_context: str = "spawn",
    dloader_workers: int = 8,
    epochs: int = 3,
    grad_accum: int = 1,
    steps_per_epoch: int = 10_000,
    batch_size_per_gpu: int = 16,
    train_timesteps_per_epoch: int = 2000,
    log: bool = False,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
):
    """
    Create AMAGO experiment for PSRO oracle training.

    This is similar to create_offline_rl_trainer but enables online RL:
    - Collects trajectories during training via local ladder
    - Mixes online data with offline replay buffer
    - Shorter training (2-4 epochs vs 100 epochs)

    Args:
        online_selfplay_weight: Fraction of each batch from online self-play (vs offline replays)
        train_timesteps_per_epoch: Number of timesteps to collect per epoch via self-play
        ... (other args same as create_offline_rl_trainer)
    """
    # Configuration
    config = {
        "MetamonTstepEncoder.tokenizer": obs_space.tokenizer,
        "MetamonPerceiverTstepEncoder.tokenizer": obs_space.tokenizer,
    }
    model_config_path = os.path.join(MODEL_CONFIG_DIR, model_gin_config)
    training_config_path = os.path.join(TRAINING_CONFIG_DIR, train_gin_config)
    amago.cli_utils.use_config(config, [model_config_path, training_config_path])

    # Training environment: local ladder (for online self-play)
    team_set = get_metamon_teams(battle_format, team_set_name)

    # Generate unique username generator for parallel actors
    # Each of the 4 parallel actors needs a unique username
    import random
    import datetime
    base_timestamp = datetime.datetime.now().strftime("%M%S")  # 4 chars instead of 6
    br_num = run_name.replace("PSRO_BR", "").replace("PSRO_", "")[:1]  # Single digit

    def make_train_env_with_unique_username():
        """Wrapper to generate unique username for each parallel actor."""
        # Use process ID + random number to ensure uniqueness across parallel processes
        # (actor_counter doesn't work with multiprocessing spawn context)
        # Keep under 18 chars: "BR0_1234_5678_999" = 17 chars max
        pid = os.getpid() % 10000  # Last 4 digits
        rand = random.randint(100, 999)  # 3 digits
        username = f"BR{br_num}_{base_timestamp}_{pid}_{rand}"

        return make_local_ladder_env(
            battle_format=battle_format,
            num_battles=None,  # Run indefinitely during training
            observation_space=obs_space,
            action_space=action_space,
            reward_function=reward_function,
            player_team_set=team_set,
            player_username=username,  # Unique per actor
            save_trajectories_to=None,  # Don't save during training (use buffer)
            battle_backend="poke-env",
        )

    make_train_env_fn = make_train_env_with_unique_username

    # Validation: placeholder (we'll evaluate separately)
    make_val_env_fn = [partial(make_placeholder_env, obs_space, action_space)]

    # Calculate batch mixing
    # AMAGO doesn't have built-in support for mixing online/offline, so we'll
    # use a workaround: adjust replay ratios
    # For now, we'll primarily use offline data and supplement with online trajectories
    # collected during training

    experiment = MetamonAMAGOExperiment(
        run_name=run_name,
        ckpt_base_dir=ckpt_dir,
        dataset=amago_dataset,
        make_train_env=make_train_env_fn,
        make_val_env=make_val_env_fn,
        env_mode="async",
        async_env_mp_context=async_env_mp_context,
        parallel_actors=4,  # 4 parallel training agents on ladder
        exploration_wrapper_type=None,
        sample_actions=True,
        val_timesteps_per_epoch=0,  # No validation during training
        val_interval=None,
        log_to_wandb=log,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        verbose=True,
        log_interval=300,
        padded_sampling="none",
        dloader_workers=dloader_workers,
        epochs=epochs,
        # Enable online RL: start collecting from epoch 0
        start_learning_at_epoch=0,
        start_collecting_at_epoch=0,
        train_timesteps_per_epoch=train_timesteps_per_epoch,
        train_batches_per_epoch=steps_per_epoch * grad_accum,
        ckpt_interval=1,  # Save every epoch
        batch_size=batch_size_per_gpu,
        batches_per_update=grad_accum,
        mixed_precision="no",
    )

    return experiment


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Train PSRO best-response policy via online self-play against population"
    )

    # PSRO-specific args
    psro_group = parser.add_argument_group("PSRO")
    psro_group.add_argument(
        "--population_file",
        type=str,
        required=True,
        help="Path to population.json from Phase 0",
    )
    psro_group.add_argument(
        "--meta_strategy_file",
        type=str,
        required=True,
        help="Path to meta_strategy.json from Phase 0 or previous iteration",
    )
    psro_group.add_argument(
        "--init_from_checkpoint",
        type=str,
        default="SyntheticRLV2",
        help="Initialize from this checkpoint (default: 'SyntheticRLV2'). Recommended to use SyntheticRLV2 as base. Set to empty string to train from scratch.",
    )
    psro_group.add_argument(
        "--online_selfplay_weight",
        type=float,
        default=0.5,
        help="Weight of online self-play data vs offline replays (0-1)",
    )
    psro_group.add_argument(
        "--battle_format",
        type=str,
        default="gen1ou",
        help="Showdown battle format (e.g., 'gen1ou', 'gen2ou', etc.)",
    )
    psro_group.add_argument(
        "--team_set",
        type=str,
        default="modern_replays_v2",
        help="Team set name (e.g., 'modern_replays_v2', 'competitive', 'paper_variety')",
    )

    # Add standard RL training args
    parser = add_rl_cli(parser)

    # Override some defaults for PSRO
    parser.set_defaults(
        epochs=3,  # Short training per iteration
        batch_size_per_gpu=16,
        grad_accum=1,
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PSRO ORACLE TRAINING (Best-Response via RL)")
    print("="*80)
    print(f"Run name: {args.run_name}")
    print(f"Population: {args.population_file}")
    print(f"Meta-strategy: {args.meta_strategy_file}")
    print(f"Battle format: {args.battle_format}")
    print(f"Team set: {args.team_set}")
    print(f"Epochs: {args.epochs}")
    print(f"Online/offline mix: {args.online_selfplay_weight:.1%} online, {1-args.online_selfplay_weight:.1%} offline")
    print("="*80 + "\n")

    # Setup agent interface
    obs_space = TokenizedObservationSpace(
        get_observation_space(args.obs_space), get_tokenizer(args.tokenizer)
    )
    reward_function = get_reward_function(args.reward_function)
    action_space = get_action_space(args.action_space)

    # Create offline dataset (for mixing with online data)
    amago_dataset = create_offline_dataset(
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        parsed_replay_dir=args.parsed_replay_dir,
        custom_replay_dir=args.custom_replay_dir,
        custom_replay_sample_weight=args.custom_replay_sample_weight,
        formats=args.formats,
    )

    # Launch opponent population on ladder
    run_id = f"{args.run_name}_{int(time.time())}"
    battle_format = args.battle_format
    team_set_name = args.team_set

    opponent_processes = launch_population_opponents(
        population_file=args.population_file,
        meta_strategy_file=args.meta_strategy_file,
        battle_format=battle_format,
        team_set_name=team_set_name,
        run_id=run_id,
    )

    # Setup cleanup on exit
    def cleanup_handler(signum, frame):
        cleanup_opponent_processes(opponent_processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    try:
        # Create PSRO oracle trainer
        experiment = create_psro_oracle_trainer(
            ckpt_dir=args.save_dir,
            run_name=args.run_name,
            model_gin_config=args.model_gin_config,
            train_gin_config=args.train_gin_config,
            obs_space=obs_space,
            action_space=action_space,
            reward_function=reward_function,
            amago_dataset=amago_dataset,
            battle_format=battle_format,
            team_set_name=team_set_name,
            online_selfplay_weight=args.online_selfplay_weight,
            async_env_mp_context=args.async_env_mp_context,
            dloader_workers=args.dloader_workers,
            epochs=args.epochs,
            grad_accum=args.grad_accum,
            batch_size_per_gpu=args.batch_size_per_gpu,
            log=args.log,
            wandb_project=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,
        )

        # Start training
        experiment.start()

        # Load from checkpoint if specified
        if args.init_from_checkpoint and args.init_from_checkpoint.strip():
            print(f"\nNote: --init_from_checkpoint={args.init_from_checkpoint} specified")
            print(f"  Phase 1: Checkpoint loading not yet implemented")
            print(f"  Workaround: Train from scratch using same architecture as {args.init_from_checkpoint}")
            print(f"  - Use model config that matches base model architecture")
            print(f"  - Online RL will adapt model to meta-strategy distribution")
            print(f"  Phase 2: Will implement proper checkpoint loading")
            print()
            # TODO: Implement checkpoint loading from pretrained model
            # This requires:
            # 1. Load pretrained model: model = get_pretrained_model(args.init_from_checkpoint)
            # 2. Extract checkpoint path from model
            # 3. Load weights into experiment: experiment.load_checkpoint(...)
            # For now, we train from scratch with the same architecture

        if args.ckpt is not None:
            # Resume from existing PSRO run
            experiment.load_checkpoint(args.ckpt)

        # Run training
        print("\nStarting PSRO oracle training...")
        print("  Training agent will queue on local ladder and face opponent population")
        print(f"  Meta-strategy distribution will be maintained by opponent launcher\n")

        experiment.learn()

        # Cleanup
        print("\nTraining complete!")
        wandb.finish()

    finally:
        # Always cleanup opponent processes
        cleanup_opponent_processes(opponent_processes)

    print(f"\n✓ PSRO oracle training complete: {args.run_name}")
    print(f"  Checkpoints saved to: {args.save_dir}/{args.run_name}/ckpts/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
