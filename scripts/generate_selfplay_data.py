#!/usr/bin/env python3
"""
Generate high-quality self-play trajectory data for next training iteration.

Runs the best performing checkpoint(s) against themselves or each other to
generate diverse, high-level battle data for future finetuning.

Usage:
    # Generate 10k battles with best checkpoint playing itself
    python scripts/generate_selfplay_data.py \
        --model Gen1BinaryV0_Epoch2 \
        --num_battles 10000 \
        --output_dir ~/gen1_selfplay_data/v0 \
        --battle_format gen1ou \
        --team_set competitive

    # Generate 10k battles between two different checkpoints
    python scripts/generate_selfplay_data.py \
        --model Gen1BinaryV0_Epoch2 \
        --opponent_model Gen1BinaryV0_Epoch4 \
        --num_battles 10000 \
        --output_dir ~/gen1_selfplay_data/v0 \
        --battle_format gen1ou \
        --team_set competitive
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path is set
import metamon
from metamon.rl.gen1_binary_models import *  # Register custom models
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from metamon.env import get_metamon_teams


def generate_selfplay(
    model_name: str,
    opponent_model_name: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    output_dir: str,
    battle_backend: str = "poke-env",
    parallel_instances: int = 2,
):
    """
    Generate self-play battles by running two instances of models on local ladder.

    Args:
        model_name: Primary model for self-play
        opponent_model_name: Opponent model (can be same as model_name for pure self-play)
        battle_format: Showdown format (e.g., "gen1ou")
        team_set_name: Team set to use
        num_battles: Total battles to generate
        output_dir: Where to save trajectory data
        battle_backend: Battle state backend
        parallel_instances: Number of concurrent battle instances (higher = faster but more CPU)
    """

    print(f"\n{'='*80}")
    print(f"SELF-PLAY DATA GENERATION")
    print(f"Model: {model_name}")
    print(f"Opponent: {opponent_model_name}")
    print(f"Format: {battle_format}")
    print(f"Target battles: {num_battles}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Create output directories
    trajectories_dir = os.path.join(output_dir, battle_format)
    os.makedirs(trajectories_dir, exist_ok=True)

    # Calculate battles per instance
    battles_per_instance = num_battles // parallel_instances

    # Create timestamp for unique usernames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Launch multiple parallel instances
    processes = []

    for i in range(parallel_instances):
        username1 = f"SelfPlay_{model_name[:8]}_{timestamp}_A{i}"
        username2 = f"SelfPlay_{opponent_model_name[:8]}_{timestamp}_B{i}"

        print(f"Launching instance {i+1}/{parallel_instances}...")
        print(f"  Player 1: {username1}")
        print(f"  Player 2: {username2}")
        print(f"  Battles: {battles_per_instance}\n")

        # Launch player 1
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

model = get_pretrained_model('{model_name}')
team_set = get_metamon_teams('{battle_format}', '{team_set_name}')

pretrained_vs_local_ladder(
    pretrained_model=model,
    username='{username1}',
    battle_format='{battle_format}',
    team_set=team_set,
    total_battles={battles_per_instance},
    battle_backend='{battle_backend}',
    save_trajectories_to='{output_dir}',
    log_to_wandb=False,
)
""",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Stagger launches
        time.sleep(3)

        # Launch player 2
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

model = get_pretrained_model('{opponent_model_name}')
team_set = get_metamon_teams('{battle_format}', '{team_set_name}')

pretrained_vs_local_ladder(
    pretrained_model=model,
    username='{username2}',
    battle_format='{battle_format}',
    team_set=team_set,
    total_battles={battles_per_instance},
    battle_backend='{battle_backend}',
    save_trajectories_to='{output_dir}',
    log_to_wandb=False,
)
""",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        processes.append((proc1, proc2, username1, username2))

        # Stagger next instance
        time.sleep(5)

    print(f"All instances launched. Generating {num_battles} battles...")
    print(
        f"Estimated time: {num_battles / (500 * parallel_instances):.1f} hours (assuming ~500 battles/hour/instance)"
    )
    print("\nMonitoring progress (this script will wait for completion)...\n")

    # Monitor processes
    start_time = time.time()
    completed = 0

    while completed < len(processes):
        time.sleep(30)  # Check every 30 seconds

        # Count completed
        completed = sum(
            1 for proc1, proc2, _, _ in processes if proc1.poll() is not None
        )

        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600

        # Count trajectory files to estimate progress
        if os.path.exists(trajectories_dir):
            trajectory_count = len(list(Path(trajectories_dir).glob("*.json.lz4")))
        else:
            trajectory_count = 0

        print(
            f"[{elapsed_hours:.1f}h] Progress: {trajectory_count}/{num_battles} battles generated | {completed}/{len(processes)} instances complete"
        )

    # Wait for all processes
    for proc1, proc2, user1, user2 in processes:
        proc1.wait()
        proc2.wait()

        if proc1.returncode != 0:
            stderr = proc1.stderr.read().decode()
            print(f"WARNING: {user1} had errors: {stderr[:200]}")

        if proc2.returncode != 0:
            stderr = proc2.stderr.read().decode()
            print(f"WARNING: {user2} had errors: {stderr[:200]}")

    # Final count
    final_count = len(list(Path(trajectories_dir).glob("*.json.lz4")))
    total_time = (time.time() - start_time) / 3600

    print(f"\n{'='*80}")
    print(f"SELF-PLAY GENERATION COMPLETE")
    print(f"Total battles generated: {final_count}")
    print(f"Total time: {total_time:.1f} hours")
    print(f"Rate: {final_count / total_time:.0f} battles/hour")
    print(f"Output: {trajectories_dir}")
    print(f"{'='*80}\n")

    if final_count < num_battles * 0.9:
        print(
            f"WARNING: Only {final_count}/{num_battles} battles were generated ({final_count/num_battles*100:.1f}%)"
        )
        print("Consider re-running or investigating errors.")

    return final_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play trajectory data for training"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for self-play generation",
    )
    parser.add_argument(
        "--opponent_model",
        type=str,
        default=None,
        help="Opponent model (defaults to same as --model for pure self-play)",
    )
    parser.add_argument(
        "--num_battles",
        type=int,
        default=10000,
        help="Number of battles to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/gen1_selfplay_data/v0",
        help="Directory to save trajectory data (format subdirectory will be created)",
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
        help="Team set name",
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help="Battle state backend",
    )
    parser.add_argument(
        "--parallel_instances",
        type=int,
        default=2,
        help="Number of parallel battle instances (more = faster but more CPU)",
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
        return 0

    # Validate model exists
    available_models = get_pretrained_model_names()
    if args.model not in available_models:
        print(f"ERROR: Model '{args.model}' not found")
        print(f"Available models: {', '.join(sorted(available_models))}")
        return 1

    # Default opponent to same model
    opponent_model = args.opponent_model or args.model

    if opponent_model not in available_models:
        print(f"ERROR: Opponent model '{opponent_model}' not found")
        print(f"Available models: {', '.join(sorted(available_models))}")
        return 1

    # Expand home directory
    output_dir = os.path.expanduser(args.output_dir)

    # Generate self-play data
    generate_selfplay(
        model_name=args.model,
        opponent_model_name=opponent_model,
        battle_format=args.battle_format,
        team_set_name=args.team_set,
        num_battles=args.num_battles,
        output_dir=output_dir,
        battle_backend=args.battle_backend,
        parallel_instances=args.parallel_instances,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
