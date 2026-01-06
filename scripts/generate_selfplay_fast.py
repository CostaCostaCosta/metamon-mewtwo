#!/usr/bin/env python3
"""
Fast and stable self-play data generation using GPU inference server.

This version includes:
- Subprocess isolation to handle PyKMN crashes
- Optimized batching strategy
- Real-time performance monitoring
- Automatic crash recovery
"""

import os
import sys
import argparse
import time
import subprocess
import json
import gc
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import multiprocessing as mp

# Set cache directory before imports
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")


def run_single_batch(
    batch_size: int,
    format_name: str,
    team_set: str,
    server_url: str,
    save_dir: Path,
    batch_id: int,
    verbose: bool = False,
) -> Dict:
    """
    Run a single batch of battles in an isolated subprocess.

    This isolation prevents memory accumulation and handles crashes gracefully.
    """
    from metamon.env.pykmn import PyKMNVectorEnv, SelfPlayRunner, load_random_teams, save_trajectories
    from metamon.inference.client import RemotePolicyRunner
    from metamon.rl.pretrained import get_pretrained_model
    from metamon.env.pykmn.features import precompute_mappings

    try:
        # Get observation space and reward function
        pretrained_cls = get_pretrained_model("SyntheticRLV2")
        obs_space = pretrained_cls.observation_space
        reward_fn = pretrained_cls.reward_function
        mappings = precompute_mappings()

        # Load teams
        cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
        team_dir = cache_dir / "teams" / team_set
        teams_p1 = load_random_teams(team_dir, format_name, batch_size)
        teams_p2 = load_random_teams(team_dir, format_name, batch_size)

        # Create environment
        env = PyKMNVectorEnv(
            num_envs=batch_size,
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            obs_space=obs_space,
            reward_fn=reward_fn,
            track_trajectories=True,
        )

        # Create remote policies
        policy_p1 = RemotePolicyRunner(server_url=server_url, model_name="SyntheticRLV2")
        policy_p2 = RemotePolicyRunner(server_url=server_url, model_name="SyntheticRLV2")

        # Create runner
        runner = SelfPlayRunner(
            vec_env=env,
            policy_p1=policy_p1,
            policy_p2=policy_p2,
        )

        # Collect trajectories
        start_time = time.time()
        trajectories = runner.collect_trajectories(
            num_battles=batch_size,
            max_steps_per_battle=500,
            verbose=False,
        )
        elapsed = time.time() - start_time

        # Save trajectories
        save_trajectories(
            trajectories=trajectories,
            output_dir=save_dir,
            mappings=mappings,
            battle_format=format_name,
            verbose=False,
        )

        # Clean up
        env.close()
        del env, runner, policy_p1, policy_p2
        gc.collect()

        return {
            "status": "success",
            "batch_id": batch_id,
            "battles": len(trajectories),
            "time": elapsed,
            "rate": len(trajectories) / elapsed,
        }

    except Exception as e:
        return {
            "status": "error",
            "batch_id": batch_id,
            "error": str(e),
            "battles": 0,
        }


def worker_process(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    server_url: str,
    save_dir: Path,
    verbose: bool,
):
    """Worker process that consumes tasks from queue."""
    while True:
        task = task_queue.get()
        if task is None:
            break

        result = run_single_batch(
            batch_size=task["batch_size"],
            format_name=task["format"],
            team_set=task["team_set"],
            server_url=server_url,
            save_dir=save_dir,
            batch_id=task["batch_id"],
            verbose=verbose,
        )
        result_queue.put(result)


def run_fast_selfplay(
    num_battles: int,
    batch_size: int,
    format_name: str,
    team_set: str,
    save_dir: Path,
    server_url: str = "http://localhost:8080",
    num_workers: int = 1,
    verbose: bool = True,
):
    """
    Run self-play with subprocess isolation and parallel workers.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    format_dir = save_dir / format_name
    format_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print("Fast Self-Play Data Generation")
        print(f"{'='*70}")
        print(f"Target battles: {num_battles}")
        print(f"Batch size: {batch_size}")
        print(f"Format: {format_name}")
        print(f"Workers: {num_workers}")
        print(f"Server: {server_url}")
        print(f"Output: {save_dir}")
        print(f"{'='*70}\n")

    # Calculate number of batches
    num_batches = (num_battles + batch_size - 1) // batch_size

    # Create task and result queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, server_url, save_dir, False),
        )
        p.start()
        workers.append(p)

    # Submit tasks
    for batch_id in range(num_batches):
        battles_in_batch = min(batch_size, num_battles - batch_id * batch_size)
        task = {
            "batch_id": batch_id,
            "batch_size": battles_in_batch,
            "format": format_name,
            "team_set": team_set,
        }
        task_queue.put(task)

    # Send stop signal to workers
    for _ in workers:
        task_queue.put(None)

    # Collect results
    start_time = time.time()
    battles_completed = 0
    batches_completed = 0
    errors = 0

    while batches_completed < num_batches:
        try:
            result = result_queue.get(timeout=60)
            batches_completed += 1

            if result["status"] == "success":
                battles_completed += result["battles"]

                if verbose:
                    elapsed = time.time() - start_time
                    overall_rate = battles_completed / elapsed if elapsed > 0 else 0
                    print(f"Batch {result['batch_id']+1}/{num_batches}: "
                          f"{result['battles']} battles in {result['time']:.1f}s "
                          f"({result['rate']:.1f} b/s) | "
                          f"Total: {battles_completed}/{num_battles} "
                          f"({overall_rate:.1f} b/s overall)")
            else:
                errors += 1
                print(f"ERROR in batch {result['batch_id']}: {result.get('error', 'Unknown error')}")

                # Retry the batch
                if errors < 3:
                    task = {
                        "batch_id": result["batch_id"],
                        "batch_size": batch_size,
                        "format": format_name,
                        "team_set": team_set,
                    }
                    task_queue.put(task)
                    num_batches += 1  # Account for retry

        except Exception as e:
            print(f"Error collecting results: {e}")
            break

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Final summary
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n{'='*70}")
        print("Self-Play Complete!")
        print(f"{'='*70}")
        print(f"Battles completed: {battles_completed}/{num_battles}")
        print(f"Batches: {batches_completed}")
        print(f"Errors: {errors}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"Average rate: {battles_completed/elapsed:.1f} battles/sec")
        print(f"Output directory: {save_dir}")
        print(f"{'='*70}")

    return battles_completed


def main():
    parser = argparse.ArgumentParser(
        description="Fast self-play data generation with crash recovery"
    )

    parser.add_argument(
        "--num_battles",
        type=int,
        required=True,
        help="Number of battles to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Battles per batch (default: 32)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Battle format (default: gen1ou)",
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="modern_replays_v2",
        help="Team set to use",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save trajectories",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8080",
        help="Inference server URL",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    run_fast_selfplay(
        num_battles=args.num_battles,
        batch_size=args.batch_size,
        format_name=args.format,
        team_set=args.team_set,
        save_dir=Path(args.save_dir),
        server_url=args.server_url,
        num_workers=args.workers,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()