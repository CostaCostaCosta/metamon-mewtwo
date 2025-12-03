#!/usr/bin/env python3
"""
Batched Trajectory Generation using Shared Policy Server

This script orchestrates high-throughput trajectory generation by:
1. Starting policy server(s) that load models once into GPU
2. Spawning worker processes that connect to policy servers
3. Workers run battles on Showdown ladder and save trajectories
4. Enables 30-50× speedup compared to redundant model loading

Usage:
    python scripts/generate_trajectories_batched.py \
        --model-a SyntheticRLV2 \
        --model-b SyntheticRLV2 \
        --battle-format gen1ou \
        --team-set competitive \
        --battles-per-worker 10 \
        --num-workers 10 \
        --save-dir ~/trajectories/gen1ou

Example (self-play with 20 workers):
    python scripts/generate_trajectories_batched.py \
        --model-a SyntheticRLV2 \
        --model-b SyntheticRLV2 \
        --battle-format gen1ou \
        --battles-per-worker 50 \
        --num-workers 20 \
        --save-dir ~/selfplay_data
"""

import argparse
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import os

import metamon


class PolicyServerManager:
    """Manages policy server processes."""

    def __init__(self, base_port: int = 5555):
        self.servers: Dict[str, Dict] = {}
        self.base_port = base_port
        self.next_port = base_port

    def start_server(
        self,
        model_name: str,
        checkpoint: Optional[int] = None,
        batch_size: int = 32,
        timeout_ms: int = 50,
    ) -> Tuple[str, int, subprocess.Popen]:
        """
        Start a policy server for a given model.

        Returns:
            Tuple of (model_name, port, process)
        """
        # Check if server already running for this model
        if model_name in self.servers:
            print(f"[ServerManager] Server for {model_name} already running on port {self.servers[model_name]['port']}")
            return (
                model_name,
                self.servers[model_name]["port"],
                self.servers[model_name]["process"],
            )

        # Assign port
        port = self.next_port
        self.next_port += 1

        # Build command (use -u for unbuffered output)
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            "-m",
            "metamon.rl.policy_server",
            "--model",
            model_name,
            "--batch-size",
            str(batch_size),
            "--timeout-ms",
            str(timeout_ms),
            "--port",
            str(port),
        ]

        if checkpoint is not None:
            cmd.extend(["--checkpoint", str(checkpoint)])

        print(f"[ServerManager] Starting server for {model_name} on port {port}")
        print(f"[ServerManager] Command: {' '.join(cmd)}")

        # Start server process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # Line buffered
        )

        # Store server info
        self.servers[model_name] = {
            "port": port,
            "process": process,
            "model": model_name,
            "checkpoint": checkpoint,
        }

        # Wait for server to initialize (check output for "Ready to serve")
        print(f"[ServerManager] Waiting for server to initialize...")
        ready = False
        for _ in range(60):  # Wait up to 60 seconds
            line = process.stdout.readline()
            if line:
                print(f"[Server-{model_name}] {line.rstrip()}")
                if "Ready to serve" in line:
                    ready = True
                    break

            # Check if process died
            if process.poll() is not None:
                print(f"[ServerManager] ERROR: Server process died during startup")
                return None

            time.sleep(1)

        if not ready:
            print(f"[ServerManager] WARNING: Server may not be ready yet")

        print(f"[ServerManager] Server for {model_name} ready on port {port}")

        return model_name, port, process

    def get_address(self, model_name: str) -> str:
        """Get ZMQ address for a model's server."""
        if model_name not in self.servers:
            raise ValueError(f"No server running for model {model_name}")

        port = self.servers[model_name]["port"]
        return f"tcp://localhost:{port}"

    def shutdown_all(self):
        """Shutdown all running servers."""
        print(f"[ServerManager] Shutting down {len(self.servers)} servers...")

        for model_name, info in self.servers.items():
            print(f"[ServerManager] Terminating server for {model_name}")
            process = info["process"]

            # Try graceful shutdown first
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                print(f"[ServerManager] Force killing server for {model_name}")
                process.kill()
                process.wait()

        self.servers.clear()
        print(f"[ServerManager] All servers shut down")


def run_worker(
    worker_id: int,
    model_name: str,
    server_address: str,
    username: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    save_dir: Path,
    avatar: str = "red-gen1main",
    battle_backend: str = "poke-env",
):
    """
    Run a single worker that battles on the ladder.

    This runs in a subprocess and uses evaluate_gpu.py to connect
    to the policy server.
    """
    # Import here (in subprocess)
    from metamon.rl.evaluate_gpu import pretrained_vs_local_ladder_gpu

    # Load team set
    team_set = metamon.env.get_metamon_teams(
        battle_format, team_set_name, set_type=metamon.env.TeamSet
    )

    # Create save directories
    traj_dir = save_dir / "trajectories" / battle_format
    team_dir = save_dir / "team_results" / battle_format
    traj_dir.mkdir(parents=True, exist_ok=True)
    team_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Worker-{worker_id}] Starting {num_battles} battles as {username}")
    print(f"[Worker-{worker_id}] Model: {model_name}, Server: {server_address}")

    try:
        results = pretrained_vs_local_ladder_gpu(
            pretrained_model_name=model_name,
            server_address=server_address,
            username=username,
            battle_format=battle_format,
            team_set=team_set,
            total_battles=num_battles,
            avatar=avatar,
            battle_backend=battle_backend,
            save_trajectories_to=str(traj_dir),
            save_team_results_to=str(team_dir),
        )

        print(f"[Worker-{worker_id}] Completed! Win rate: {results['win_rate']:.1%}")
        return results

    except Exception as e:
        print(f"[Worker-{worker_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectories using batched GPU inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model configuration
    parser.add_argument(
        "--model-a",
        required=True,
        help="First model name (e.g., SyntheticRLV2)",
    )
    parser.add_argument(
        "--model-b",
        required=True,
        help="Second model name (e.g., SyntheticRLV2). Use same as --model-a for self-play.",
    )
    parser.add_argument(
        "--checkpoint-a",
        type=int,
        default=None,
        help="Checkpoint for model A (default: use model's default)",
    )
    parser.add_argument(
        "--checkpoint-b",
        type=int,
        default=None,
        help="Checkpoint for model B (default: use model's default)",
    )

    # Battle configuration
    parser.add_argument(
        "--battle-format",
        default="gen1ou",
        help="Battle format (e.g., gen1ou, gen2ou)",
    )
    parser.add_argument(
        "--team-set",
        default="competitive",
        help="Team set name",
    )
    parser.add_argument(
        "--battle-backend",
        default="poke-env",
        choices=["poke-env", "metamon"],
        help="Battle backend",
    )

    # Worker configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--battles-per-worker",
        type=int,
        default=10,
        help="Battles per worker (default: 10). Total battles = num_workers * battles_per_worker",
    )

    # Server configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Policy server batch size (default: 32)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=50,
        help="Policy server batch timeout in ms (default: 50)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=5555,
        help="Base port for policy servers (default: 5555)",
    )

    # Output configuration
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory to save trajectories and results",
    )
    parser.add_argument(
        "--username-prefix",
        default="MetamonWorker",
        help="Prefix for worker usernames (default: MetamonWorker)",
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("BATCHED TRAJECTORY GENERATION")
    print("=" * 80)
    print(f"Model A: {args.model_a} (checkpoint: {args.checkpoint_a})")
    print(f"Model B: {args.model_b} (checkpoint: {args.checkpoint_b})")
    print(f"Battle format: {args.battle_format}")
    print(f"Team set: {args.team_set}")
    print(f"Workers: {args.num_workers}")
    print(f"Battles per worker: {args.battles_per_worker}")
    print(f"Total battles: {args.num_workers * args.battles_per_worker}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)

    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Start policy servers
    print("\n[Coordinator] Starting policy servers...")
    server_manager = PolicyServerManager(base_port=args.base_port)

    # Handle shutdown gracefully
    def shutdown_handler(signum, frame):
        print("\n[Coordinator] Received shutdown signal, cleaning up...")
        server_manager.shutdown_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # Start server for model A
        server_manager.start_server(
            model_name=args.model_a,
            checkpoint=args.checkpoint_a,
            batch_size=args.batch_size,
            timeout_ms=args.timeout_ms,
        )

        # Start server for model B (if different)
        if args.model_b != args.model_a:
            server_manager.start_server(
                model_name=args.model_b,
                checkpoint=args.checkpoint_b,
                batch_size=args.batch_size,
                timeout_ms=args.timeout_ms,
            )

        # Get server addresses
        server_a_address = server_manager.get_address(args.model_a)
        server_b_address = server_manager.get_address(args.model_b)

        print(f"\n[Coordinator] Servers ready!")
        print(f"  Model A: {server_a_address}")
        print(f"  Model B: {server_b_address}")

        # Spawn workers
        print(f"\n[Coordinator] Spawning {args.num_workers} workers...")

        worker_processes = []

        for i in range(args.num_workers):
            # Alternate between model A and model B
            if i % 2 == 0:
                model = args.model_a
                server = server_a_address
                username = f"{args.username_prefix}_A_{i}"
            else:
                model = args.model_b
                server = server_b_address
                username = f"{args.username_prefix}_B_{i}"

            # Spawn worker subprocess
            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, '{os.getcwd()}')

from scripts.generate_trajectories_batched import run_worker
from pathlib import Path

run_worker(
    worker_id={i},
    model_name='{model}',
    server_address='{server}',
    username='{username}',
    battle_format='{args.battle_format}',
    team_set_name='{args.team_set}',
    num_battles={args.battles_per_worker},
    save_dir=Path('{args.save_dir}'),
    avatar='red-gen1main',
    battle_backend='{args.battle_backend}',
)
""",
            ]

            proc = subprocess.Popen(cmd)
            worker_processes.append((i, username, proc))

            print(f"[Coordinator] Started worker {i} ({username}) using {model}")

            # Stagger worker starts to avoid overwhelming the ladder
            time.sleep(2)

        # Wait for all workers to complete
        print(f"\n[Coordinator] Waiting for {len(worker_processes)} workers to complete...")

        completed = 0
        while completed < len(worker_processes):
            for i, username, proc in worker_processes:
                if proc.poll() is not None:
                    # Worker finished
                    returncode = proc.returncode
                    if returncode == 0:
                        print(f"[Coordinator] Worker {i} ({username}) completed successfully")
                    else:
                        print(f"[Coordinator] Worker {i} ({username}) failed with code {returncode}")

                    worker_processes.remove((i, username, proc))
                    completed += 1
                    break

            time.sleep(5)

        print(f"\n[Coordinator] All workers completed!")

    finally:
        # Shutdown servers
        server_manager.shutdown_all()

    print("\n" + "=" * 80)
    print("TRAJECTORY GENERATION COMPLETE")
    print("=" * 80)
    print(f"Trajectories saved to: {args.save_dir / 'trajectories' / args.battle_format}")
    print(f"Team results saved to: {args.save_dir / 'team_results' / args.battle_format}")
    print("=" * 80)


if __name__ == "__main__":
    main()
