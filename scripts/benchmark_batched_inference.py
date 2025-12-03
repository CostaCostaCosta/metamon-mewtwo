#!/usr/bin/env python3
"""
Benchmark script for batched GPU inference system.

Tests and times the new batched inference architecture with 500 battles
to validate correctness and measure speedup.

Usage:
    python scripts/benchmark_batched_inference.py \
        --model SyntheticRLV2 \
        --battle-format gen1ou \
        --total-battles 500 \
        --num-workers 10
"""

import argparse
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Optional
import os
import json
import tempfile


class BenchmarkRunner:
    """Orchestrates benchmark execution."""

    def __init__(
        self,
        model_name: str,
        battle_format: str,
        total_battles: int,
        num_workers: int,
        checkpoint: Optional[int] = None,
        batch_size: int = 32,
        timeout_ms: int = 50,
        port: int = 5555,
        save_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.battle_format = battle_format
        self.total_battles = total_battles
        self.num_workers = num_workers
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.port = port
        self.save_dir = save_dir or Path(tempfile.mkdtemp(prefix="metamon_benchmark_"))

        self.server_process = None
        self.start_time = None
        self.end_time = None

    def check_dependencies(self):
        """Check that required dependencies are installed."""
        print("[Benchmark] Checking dependencies...")

        try:
            import zmq
            print(f"  ✓ pyzmq version: {zmq.zmq_version()}")
        except ImportError:
            print("  ✗ pyzmq not installed!")
            print("    Install with: pip install pyzmq")
            return False

        try:
            import metamon
            print(f"  ✓ metamon package found")
        except ImportError:
            print("  ✗ metamon not found!")
            return False

        try:
            import poke_env
            print(f"  ✓ poke-env installed")
        except ImportError:
            print("  ✗ poke-env not installed!")
            return False

        print("[Benchmark] All dependencies OK")
        return True

    def check_showdown_server(self):
        """Check if Showdown server is running."""
        print("[Benchmark] Checking Showdown server...")

        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", 8000))
            sock.close()

            if result == 0:
                print("  ✓ Showdown server is running on localhost:8000")
                return True
            else:
                print("  ✗ Showdown server not responding on localhost:8000")
                print("    Start with: cd server/pokemon-showdown && node pokemon-showdown start --no-security")
                return False

        except Exception as e:
            print(f"  ✗ Error checking server: {e}")
            return False

    def start_policy_server(self):
        """Start the policy server."""
        print(f"[Benchmark] Starting policy server...")
        print(f"  Model: {self.model_name}")
        print(f"  Checkpoint: {self.checkpoint}")
        print(f"  Port: {self.port}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Timeout: {self.timeout_ms}ms")

        cmd = [
            sys.executable,
            "-m",
            "metamon.rl.policy_server",
            "--model",
            self.model_name,
            "--port",
            str(self.port),
            "--batch-size",
            str(self.batch_size),
            "--timeout-ms",
            str(self.timeout_ms),
        ]

        if self.checkpoint is not None:
            cmd.extend(["--checkpoint", str(self.checkpoint)])

        # Start server
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Wait for server to be ready
        print("[Benchmark] Waiting for server to initialize...")
        ready = False
        timeout = 120  # 2 minutes for model loading

        for _ in range(timeout):
            line = self.server_process.stdout.readline()
            if line:
                print(f"  [Server] {line.rstrip()}")

                if "Ready to serve" in line:
                    ready = True
                    break

                if "Error" in line or "Failed" in line:
                    print("[Benchmark] ERROR: Server failed to start")
                    return False

            # Check if process died
            if self.server_process.poll() is not None:
                print("[Benchmark] ERROR: Server process terminated")
                return False

            time.sleep(1)

        if not ready:
            print("[Benchmark] WARNING: Server initialization timeout")
            print("[Benchmark] Proceeding anyway...")

        print("[Benchmark] Policy server ready!")
        return True

    def run_workers(self):
        """Run worker processes to generate battles."""
        print(f"\n[Benchmark] Starting {self.num_workers} workers...")
        print(f"  Total battles: {self.total_battles}")
        print(f"  Battles per worker: {self.total_battles // self.num_workers}")
        print(f"  Save directory: {self.save_dir}")

        battles_per_worker = self.total_battles // self.num_workers
        server_address = f"tcp://localhost:{self.port}"

        worker_processes = []

        # Create save directories
        traj_dir = self.save_dir / "trajectories" / self.battle_format
        team_dir = self.save_dir / "team_results" / self.battle_format
        traj_dir.mkdir(parents=True, exist_ok=True)
        team_dir.mkdir(parents=True, exist_ok=True)

        # Spawn workers
        for i in range(self.num_workers):
            username = f"BenchmarkWorker_{i}"

            cmd = [
                sys.executable,
                "-m",
                "metamon.rl.evaluate_gpu",
                "--model",
                self.model_name,
                "--server",
                server_address,
                "--eval-type",
                "ladder",
                "--username",
                username,
                "--battle-format",
                self.battle_format,
                "--team-set",
                "competitive",
                "--total-battles",
                str(battles_per_worker),
                "--save-trajectories-to",
                str(traj_dir),
                "--save-team-results-to",
                str(team_dir),
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            worker_processes.append((i, username, proc))
            print(f"  Started worker {i}: {username}")

            # Stagger starts
            time.sleep(1)

        # Monitor workers
        print(f"\n[Benchmark] Workers running... (this will take a while)")
        self.start_time = time.time()

        completed_workers = []
        worker_results = {}

        while len(completed_workers) < len(worker_processes):
            for i, username, proc in worker_processes:
                if i in completed_workers:
                    continue

                # Check if finished
                if proc.poll() is not None:
                    returncode = proc.returncode
                    completed_workers.append(i)

                    # Read final output
                    output = proc.stdout.read()

                    if returncode == 0:
                        print(f"  ✓ Worker {i} completed successfully")
                        worker_results[i] = {"status": "success", "output": output}
                    else:
                        print(f"  ✗ Worker {i} failed (exit code {returncode})")
                        worker_results[i] = {
                            "status": "failed",
                            "returncode": returncode,
                            "output": output,
                        }
                        print(f"    Output: {output[-500:]}")  # Last 500 chars

            # Progress update
            if len(completed_workers) > 0:
                elapsed = time.time() - self.start_time
                print(
                    f"[Benchmark] Progress: {len(completed_workers)}/{len(worker_processes)} workers done "
                    f"({elapsed:.1f}s elapsed)"
                )

            time.sleep(10)

        self.end_time = time.time()

        return worker_results

    def shutdown_server(self):
        """Shutdown the policy server."""
        if self.server_process:
            print("[Benchmark] Shutting down policy server...")
            self.server_process.terminate()

            try:
                self.server_process.wait(timeout=10)
                print("[Benchmark] Server shut down gracefully")
            except subprocess.TimeoutExpired:
                print("[Benchmark] Server not responding, force killing...")
                self.server_process.kill()
                self.server_process.wait()

    def analyze_results(self):
        """Analyze saved trajectories and compute statistics."""
        print("\n[Benchmark] Analyzing results...")

        traj_dir = self.save_dir / "trajectories" / self.battle_format

        if not traj_dir.exists():
            print(f"  ✗ Trajectory directory not found: {traj_dir}")
            return None

        # Count trajectory files
        traj_files = list(traj_dir.glob("*.json.lz4"))
        num_trajectories = len(traj_files)

        print(f"  Trajectory files: {num_trajectories}")

        # Calculate timing
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            battles_per_second = self.total_battles / total_time
            seconds_per_battle = total_time / self.total_battles

            print(f"\n[Benchmark] TIMING RESULTS")
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"  Battles completed: {num_trajectories}")
            print(f"  Throughput: {battles_per_second:.2f} battles/second")
            print(f"  Avg time per battle: {seconds_per_battle:.1f}s")

            results = {
                "model": self.model_name,
                "battle_format": self.battle_format,
                "num_workers": self.num_workers,
                "batch_size": self.batch_size,
                "total_battles": self.total_battles,
                "trajectories_saved": num_trajectories,
                "total_time_seconds": total_time,
                "battles_per_second": battles_per_second,
                "seconds_per_battle": seconds_per_battle,
            }

            # Save results
            results_file = self.save_dir / "benchmark_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n  Results saved to: {results_file}")

            return results

        return None

    def run(self):
        """Run the complete benchmark."""
        print("=" * 80)
        print("BATCHED GPU INFERENCE BENCHMARK")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Format: {self.battle_format}")
        print(f"Total battles: {self.total_battles}")
        print(f"Workers: {self.num_workers}")
        print("=" * 80)

        # Pre-flight checks
        if not self.check_dependencies():
            print("\n[Benchmark] FAILED: Missing dependencies")
            return False

        if not self.check_showdown_server():
            print("\n[Benchmark] FAILED: Showdown server not running")
            return False

        # Graceful shutdown handler
        def shutdown_handler(signum, frame):
            print("\n[Benchmark] Interrupted! Cleaning up...")
            self.shutdown_server()
            sys.exit(1)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        try:
            # Start policy server
            if not self.start_policy_server():
                print("\n[Benchmark] FAILED: Could not start policy server")
                return False

            # Run workers
            worker_results = self.run_workers()

            # Analyze results
            results = self.analyze_results()

            print("\n" + "=" * 80)
            print("BENCHMARK COMPLETE")
            print("=" * 80)

            if results:
                print(f"✓ Successfully completed {results['trajectories_saved']} battles")
                print(f"✓ Throughput: {results['battles_per_second']:.2f} battles/second")
                print(f"✓ Save directory: {self.save_dir}")
            else:
                print("✗ Benchmark completed but results could not be analyzed")

            return True

        except Exception as e:
            print(f"\n[Benchmark] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.shutdown_server()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batched GPU inference system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        default="SyntheticRLV2",
        help="Model to benchmark (default: SyntheticRLV2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Model checkpoint (default: use model's default)",
    )
    parser.add_argument(
        "--battle-format",
        default="gen1ou",
        help="Battle format (default: gen1ou)",
    )
    parser.add_argument(
        "--total-battles",
        type=int,
        default=500,
        help="Total battles to run (default: 500)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Server batch size (default: 32)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=50,
        help="Server batch timeout in ms (default: 50)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Server port (default: 5555)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Save directory (default: temp directory)",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = BenchmarkRunner(
        model_name=args.model,
        battle_format=args.battle_format,
        total_battles=args.total_battles,
        num_workers=args.num_workers,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        timeout_ms=args.timeout_ms,
        port=args.port,
        save_dir=args.save_dir,
    )

    success = benchmark.run()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
