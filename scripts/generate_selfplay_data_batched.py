#!/usr/bin/env python3
"""
Generate self-play data using batched GPU inference via policy server.

Uses a single GPU policy server to serve multiple parallel battle workers,
enabling efficient batched inference and faster data generation.

Architecture:
1. One policy server runs on GPU (started separately)
2. Multiple battle workers connect as clients
3. Server batches inference requests from all workers
4. Workers run independently and save trajectories

Usage:
    # First, start the policy server in a separate terminal:
    python -m metamon.rl.policy_server --model DampedBinarySuperV1_Epoch4 \\
        --batch-size 32 --timeout-ms 50 --port 5555

    # Then, run this script to generate selfplay data:
    python scripts/generate_selfplay_data_batched.py \\
        --model DampedBinarySuperV1_Epoch4 \\
        --server_address tcp://localhost:5555 \\
        --num_battles 100000 \\
        --output_dir ~/gen1_selfplay_data/damped_v1 \\
        --battle_format gen1ou \\
        --team_set modern_replays_v2 \\
        --parallel_workers 16
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


def generate_selfplay_batched(
    model_name: str,
    server_address: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    output_dir: str,
    battle_backend: str = "poke-env",
    parallel_workers: int = 16,
    verbose: bool = False,
):
    """
    Generate self-play battles using batched GPU inference.

    Args:
        model_name: Model name (must match the server's loaded model)
        server_address: Policy server address (e.g., "tcp://localhost:5555")
        battle_format: Showdown format (e.g., "gen1ou")
        team_set_name: Team set to use
        num_battles: Total battles to generate
        output_dir: Where to save trajectory data
        battle_backend: Battle state backend
        parallel_workers: Number of parallel battle workers
        verbose: Enable verbose logging
    """

    print(f"\n{'='*80}")
    print(f"BATCHED GPU SELF-PLAY DATA GENERATION")
    print(f"Model: {model_name}")
    print(f"Server: {server_address}")
    print(f"Format: {battle_format}")
    print(f"Target battles: {num_battles}")
    print(f"Output: {output_dir}")
    print(f"Parallel workers: {parallel_workers}")
    print(f"Verbose: {verbose}")
    print(f"{'='*80}\n")

    # Create output directories
    trajectories_dir = os.path.join(output_dir, battle_format)
    logs_dir = os.path.join(output_dir, "worker_logs")
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Calculate battles per worker
    battles_per_worker_pair = num_battles // (parallel_workers // 2)

    # Create short timestamp (Showdown username limit is 18 chars)
    timestamp = datetime.now().strftime("%H%M%S")

    # Launch parallel workers in pairs (A vs B)
    processes = []
    num_pairs = parallel_workers // 2

    for i in range(num_pairs):
        # Short usernames: "GPU_A0_HHMMSS" = 14 chars max (under 18 limit)
        username1 = f"GPU_A{i}_{timestamp}"
        username2 = f"GPU_B{i}_{timestamp}"

        print(f"Launching worker pair {i+1}/{num_pairs}...")
        print(f"  Player 1: {username1}")
        print(f"  Player 2: {username2}")
        print(f"  Battles: {battles_per_worker_pair}")

        # Log file paths
        log1 = os.path.join(logs_dir, f"{username1}.log")
        log2 = os.path.join(logs_dir, f"{username2}.log")
        print(f"  Logs: {log1}, {log2}\n")

        # Worker code (shared between both players)
        worker_code = """
import sys
import traceback
sys.path.insert(0, '{repo_path}')

try:
    print("[Worker] Starting {username}...")
    from metamon.rl.gen1_binary_models import *
    from metamon.rl.evaluate_gpu import pretrained_vs_local_ladder_gpu
    from metamon.env import get_metamon_teams

    print("[Worker] Loading team set: {team_set_name}")
    team_set = get_metamon_teams('{battle_format}', '{team_set_name}')
    print("[Worker] Team set loaded successfully")

    print("[Worker] Connecting to server: {server_address}")
    print("[Worker] Starting {total_battles} battles as {username}...")

    results = pretrained_vs_local_ladder_gpu(
        pretrained_model_name='{model_name}',
        server_address='{server_address}',
        username='{username}',
        battle_format='{battle_format}',
        team_set=team_set,
        total_battles={total_battles},
        battle_backend='{battle_backend}',
        save_trajectories_to='{output_dir}',
    )

    print(f"[Worker] {username} completed successfully!")
    print(f"[Worker] Results: {{results}}")

except Exception as e:
    print(f"[Worker] ERROR in {username}: {{e}}")
    print(f"[Worker] Full traceback:")
    traceback.print_exc()
    sys.exit(1)
"""

        # Format worker code for player 1
        code1 = worker_code.format(
            repo_path=str(Path(__file__).parent.parent),
            username=username1,
            team_set_name=team_set_name,
            battle_format=battle_format,
            server_address=server_address,
            model_name=model_name,
            total_battles=battles_per_worker_pair,
            battle_backend=battle_backend,
            output_dir=output_dir,
        )

        # Launch player 1
        with open(log1, "w") as f1:
            proc1 = subprocess.Popen(
                [sys.executable, "-u", "-c", code1],
                stdout=f1,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )

        # Stagger launches to avoid overwhelming the server
        time.sleep(2)

        # Format worker code for player 2
        code2 = worker_code.format(
            repo_path=str(Path(__file__).parent.parent),
            username=username2,
            team_set_name=team_set_name,
            battle_format=battle_format,
            server_address=server_address,
            model_name=model_name,
            total_battles=battles_per_worker_pair,
            battle_backend=battle_backend,
            output_dir=output_dir,
        )

        # Launch player 2
        with open(log2, "w") as f2:
            proc2 = subprocess.Popen(
                [sys.executable, "-u", "-c", code2],
                stdout=f2,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )

        processes.append((proc1, proc2, username1, username2, log1, log2))

        # Stagger next pair
        time.sleep(3)

    print(f"All workers launched. Generating {num_battles} battles...")
    print(
        f"Estimated time: {num_battles / (600 * (parallel_workers // 2)):.1f} hours (assuming ~600 battles/hour/pair)"
    )
    print(f"\nWorker logs: {logs_dir}")
    print("\nMonitoring progress (this script will wait for completion)...\n")

    # Monitor processes
    start_time = time.time()
    completed = 0
    last_count = 0
    early_failures_reported = set()

    while completed < len(processes):
        time.sleep(30)  # Check every 30 seconds

        # Count completed and check for early failures
        new_completed = 0
        for idx, (proc1, proc2, user1, user2, log1, log2) in enumerate(processes):
            if proc1.poll() is not None:
                new_completed += 1

                # If process completed within first 2 minutes, likely failed
                elapsed = time.time() - start_time
                if elapsed < 120 and idx not in early_failures_reported:
                    early_failures_reported.add(idx)
                    print(f"\n[!] EARLY FAILURE DETECTED: Worker pair {idx+1}")
                    print(f"    Players: {user1}, {user2}")
                    print(f"    Return codes: {proc1.returncode}, {proc2.returncode}")

                    # Print last 30 lines of each log
                    if proc1.returncode != 0:
                        print(f"\n    === Last 30 lines of {user1} log ===")
                        try:
                            with open(log1, "r") as f:
                                lines = f.readlines()
                                for line in lines[-30:]:
                                    print(f"    {line.rstrip()}")
                        except Exception as e:
                            print(f"    Could not read log: {e}")

                    if proc2.returncode != 0:
                        print(f"\n    === Last 30 lines of {user2} log ===")
                        try:
                            with open(log2, "r") as f:
                                lines = f.readlines()
                                for line in lines[-30:]:
                                    print(f"    {line.rstrip()}")
                        except Exception as e:
                            print(f"    Could not read log: {e}")
                    print()

        completed = new_completed

        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600

        # Count trajectory files to estimate progress
        if os.path.exists(trajectories_dir):
            trajectory_count = len(list(Path(trajectories_dir).glob("*.json.lz4")))
        else:
            trajectory_count = 0

        # Calculate rate
        if trajectory_count > last_count:
            rate = (trajectory_count - last_count) / 30 * 3600  # battles/hour
            last_count = trajectory_count
            rate_str = f"(~{rate:.0f} battles/hour)"
        else:
            rate_str = ""

        print(
            f"[{elapsed_hours:.1f}h] Progress: {trajectory_count}/{num_battles} battles {rate_str} | {completed}/{len(processes)} worker pairs complete"
        )

    # Wait for all processes and collect final results
    print("\nWaiting for all workers to finish...")
    failed_workers = []

    for proc1, proc2, user1, user2, log1, log2 in processes:
        proc1.wait()
        proc2.wait()

        if proc1.returncode != 0:
            failed_workers.append((user1, log1, proc1.returncode))

        if proc2.returncode != 0:
            failed_workers.append((user2, log2, proc2.returncode))

    # Final count
    final_count = len(list(Path(trajectories_dir).glob("*.json.lz4")))
    total_time = (time.time() - start_time) / 3600

    print(f"\n{'='*80}")
    print(f"BATCHED GPU SELF-PLAY GENERATION COMPLETE")
    print(f"Total battles generated: {final_count}")
    print(f"Total time: {total_time:.1f} hours")
    if total_time > 0:
        print(f"Rate: {final_count / total_time:.0f} battles/hour")
    print(f"Output: {trajectories_dir}")
    print(f"{'='*80}\n")

    # Report failures
    if failed_workers:
        print(f"\n[!] {len(failed_workers)} workers failed:")
        for username, log_path, returncode in failed_workers:
            print(f"  - {username} (exit code: {returncode})")
            print(f"    Log: {log_path}")
        print()

    if final_count < num_battles * 0.9:
        print(
            f"WARNING: Only {final_count}/{num_battles} battles were generated ({final_count/num_battles*100:.1f}%)"
        )
        print("Check worker logs for errors:")
        print(f"  ls -lh {logs_dir}")
        print(f"  tail -50 {logs_dir}/*.log")

    return final_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play data using batched GPU inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (must match policy server's loaded model)",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="tcp://localhost:5555",
        help="Policy server address (default: tcp://localhost:5555)",
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
        default="~/gen1_selfplay_data/batched",
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
        default="modern_replays_v2",
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
        "--parallel_workers",
        type=int,
        default=16,
        help="Number of parallel battle workers (should be even, pairs will play each other)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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

    # Ensure parallel_workers is even
    if args.parallel_workers % 2 != 0:
        print(
            f"WARNING: parallel_workers should be even. Adjusting from {args.parallel_workers} to {args.parallel_workers + 1}"
        )
        args.parallel_workers += 1

    # Expand home directory
    output_dir = os.path.expanduser(args.output_dir)

    # Generate self-play data
    generate_selfplay_batched(
        model_name=args.model,
        server_address=args.server_address,
        battle_format=args.battle_format,
        team_set_name=args.team_set,
        num_battles=args.num_battles,
        output_dir=output_dir,
        battle_backend=args.battle_backend,
        parallel_workers=args.parallel_workers,
        verbose=args.verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
