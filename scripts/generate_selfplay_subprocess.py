#!/usr/bin/env python3
"""
Subprocess-isolated self-play data generation wrapper.

Wraps generate_selfplay_batched.py with subprocess isolation for crash resilience.
Each chunk of battles runs in a separate subprocess - if one crashes, others continue.

Usage:
    # Basic usage
    python scripts/generate_selfplay_subprocess.py \\
        --model SyntheticRLV2 \\
        --checkpoint 48 \\
        --num_battles 1000 \\
        --batch_size 16 \\
        --chunk_size 160 \\
        --format gen1ou \\
        --save_dir ~/selfplay_data/gen1ou

    # Head-to-head
    python scripts/generate_selfplay_subprocess.py \\
        --model_p1 SyntheticRLV2 \\
        --checkpoint_p1 48 \\
        --model_p2 SyntheticRLV1 \\
        --checkpoint_p2 40 \\
        --num_battles 10000 \\
        --batch_size 16 \\
        --chunk_size 320 \\
        --format gen1ou \\
        --save_dir ~/selfplay_data/gen1ou

Performance:
    - chunk_size=16 (1 batch): ~4% overhead, maximum crash protection
    - chunk_size=160 (10 batches): ~0.4% overhead, good crash protection
    - chunk_size=1000 (62 batches): ~0% overhead, minimal crash protection
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def run_chunk_subprocess(
    chunk_idx: int,
    chunk_battles: int,
    base_args: Dict,
    timeout: int = 600,
    verbose: bool = True,
) -> Dict:
    """Run a chunk of battles in a subprocess.

    Args:
        chunk_idx: Chunk index for tracking
        chunk_battles: Number of battles in this chunk
        base_args: Base arguments to pass to generate_selfplay_batched.py
        timeout: Subprocess timeout in seconds
        verbose: Print detailed progress

    Returns:
        Dict with 'success', 'battles_completed', 'time', 'error' fields
    """
    # Build command
    cmd = [sys.executable, "scripts/generate_selfplay_batched.py"]

    # Add arguments
    chunk_args = base_args.copy()
    chunk_args["num_battles"] = chunk_battles
    chunk_args["run_name"] = f"{base_args['run_name']}_chunk{chunk_idx:04d}"

    for key, value in chunk_args.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value is False:
            # For False boolean flags, just skip them (don't add --no_* which doesn't exist)
            continue
        elif value is not None:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    # Run subprocess
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,  # Run from repo root
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            return {
                "success": True,
                "battles_completed": chunk_battles,
                "time": elapsed,
                "rate": chunk_battles / elapsed if elapsed > 0 else 0,
            }
        else:
            return {
                "success": False,
                "battles_completed": 0,
                "time": elapsed,
                "error": f"Exit code {result.returncode}",
                "stderr": result.stderr[-1000:],  # Last 1000 chars
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "battles_completed": 0,
            "time": elapsed,
            "error": f"Timeout after {timeout}s",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "battles_completed": 0,
            "time": elapsed,
            "error": str(e),
        }


def _run_chunk_worker(args_tuple):
    """Worker function for parallel execution (must be picklable)."""
    chunk_idx, chunk_battles, base_args, max_retries, timeout = args_tuple
    return run_with_retry(
        chunk_idx=chunk_idx,
        chunk_battles=chunk_battles,
        base_args=base_args,
        max_retries=max_retries,
        timeout=timeout,
        verbose=False,
    )


def run_with_retry(
    chunk_idx: int,
    chunk_battles: int,
    base_args: Dict,
    max_retries: int = 3,
    timeout: int = 600,
    verbose: bool = True,
) -> Dict:
    """Run chunk with automatic retry on failure.

    Args:
        chunk_idx: Chunk index
        chunk_battles: Number of battles in this chunk
        base_args: Base arguments
        max_retries: Maximum retry attempts
        timeout: Subprocess timeout
        verbose: Print detailed progress

    Returns:
        Dict with result information
    """
    for attempt in range(max_retries):
        if attempt > 0 and verbose:
            print(f"  Retry attempt {attempt}/{max_retries-1}...")

        result = run_chunk_subprocess(
            chunk_idx=chunk_idx,
            chunk_battles=chunk_battles,
            base_args=base_args,
            timeout=timeout,
            verbose=verbose,
        )

        if result["success"]:
            if attempt > 0 and verbose:
                print(f"  ✓ Retry succeeded!")
            return result

        if verbose:
            print(f"  ❌ Attempt {attempt+1} failed: {result.get('error', 'unknown')}")

        # Brief pause before retry
        if attempt < max_retries - 1:
            time.sleep(2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Subprocess-isolated self-play data generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments (same as generate_selfplay_batched.py)
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, help="Model name for self-play")
    model_group.add_argument("--checkpoint", type=int, help="Checkpoint number")
    model_group.add_argument("--model_p1", type=str, help="Model for Player 1")
    model_group.add_argument("--checkpoint_p1", type=int, help="Checkpoint for P1")
    model_group.add_argument("--model_p2", type=str, help="Model for Player 2")
    model_group.add_argument("--checkpoint_p2", type=int, help="Checkpoint for P2")

    # Data arguments
    data_group = parser.add_argument_group("Data Generation")
    data_group.add_argument("--num_battles", type=int, required=True)
    data_group.add_argument("--batch_size", type=int, default=16)
    data_group.add_argument("--format", type=str, default="gen1ou")
    data_group.add_argument("--team_set", type=str, default="modern_replays_v2")
    data_group.add_argument("--team_dir", type=str, default=None)

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--save_dir", type=str, required=True)
    output_group.add_argument("--run_name", type=str, default=None)

    # Subprocess isolation arguments
    isolation_group = parser.add_argument_group("Subprocess Isolation")
    isolation_group.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Battles per subprocess (default: 10 × batch_size). "
        "Higher = less overhead, lower = more crash protection",
    )
    isolation_group.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Retry attempts for failed chunks (default: 3)",
    )
    isolation_group.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per chunk in seconds (default: 600)",
    )
    isolation_group.add_argument(
        "--save_failed_chunks",
        action="store_true",
        help="Save error logs for failed chunks",
    )
    isolation_group.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). "
        "Increase to 2-4 to utilize more GPU memory. Each worker loads the model (~2GB GPU memory).",
    )

    # Performance arguments
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--device", type=str, default="cuda")
    perf_group.add_argument("--use_amp", action="store_true", default=True)
    perf_group.add_argument("--temperature", type=float, default=1.0)

    # Logging
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    # Generate run name if not provided
    if args.run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model:
            args.run_name = f"{args.model}_subprocess_{timestamp}"
        else:
            args.run_name = f"{args.model_p1}_vs_{args.model_p2}_subprocess_{timestamp}"

    # Determine chunk size
    chunk_size = args.chunk_size or (args.batch_size * 10)
    num_chunks = (args.num_battles + chunk_size - 1) // chunk_size

    # Prepare base arguments for subprocess
    base_args = {
        "format": args.format,
        "batch_size": args.batch_size,
        "save_dir": args.save_dir,
        "run_name": args.run_name,
        "team_set": args.team_set,
        "team_dir": args.team_dir,
        "device": args.device,
        "temperature": args.temperature,
        "quiet": True,  # Use --quiet flag to disable verbose in subprocesses
    }

    # Add use_amp if True (it's the default, but be explicit)
    if args.use_amp:
        base_args["use_amp"] = True

    # Add model arguments
    if args.model:
        base_args["model"] = args.model
        base_args["checkpoint"] = args.checkpoint
    else:
        base_args["model_p1"] = args.model_p1
        base_args["checkpoint_p1"] = args.checkpoint_p1
        base_args["model_p2"] = args.model_p2
        base_args["checkpoint_p2"] = args.checkpoint_p2

    # Print configuration
    if verbose:
        print("=" * 70)
        print("SUBPROCESS-ISOLATED SELF-PLAY DATA GENERATION")
        print("=" * 70)
        print(f"Total battles: {args.num_battles}")
        print(f"Batch size: {args.batch_size}")
        print(f"Chunk size: {chunk_size} battles/subprocess")
        print(f"Number of chunks: {num_chunks}")
        print(f"Parallel workers: {args.num_workers}")
        print(f"Max retries: {args.max_retries}")
        print(f"Format: {args.format}")
        print(f"Output: {args.save_dir}")
        if args.num_workers > 1:
            print(f"\n⚡ Parallel mode: {args.num_workers} chunks will run simultaneously")
            print(f"   Expected speedup: ~{args.num_workers}x (if GPU memory allows)")
        print("=" * 70)
        print()

    # Run chunks (parallel or sequential)
    total_start_time = time.time()
    total_completed = 0
    total_failed = 0
    failed_chunks = []

    if args.num_workers > 1:
        # Parallel execution with multiprocessing
        from multiprocessing import Pool

        # Prepare chunk arguments (tuples for worker function)
        chunk_args = [
            (
                chunk_idx,
                min(chunk_size, args.num_battles - chunk_idx * chunk_size),
                base_args,
                args.max_retries,
                args.timeout,
            )
            for chunk_idx in range(num_chunks)
        ]

        # Run chunks in parallel using picklable worker function
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(_run_chunk_worker, chunk_args)

        # Process results
        for chunk_idx, result in enumerate(results):
            chunk_battles = chunk_args[chunk_idx][1]

            if result["success"]:
                total_completed += result["battles_completed"]
                if verbose:
                    print(f"✓ Chunk {chunk_idx+1}/{num_chunks}: {result['battles_completed']} battles in {result['time']:.1f}s ({result['rate']:.1f} battles/sec)")
            else:
                total_failed += chunk_battles
                failed_chunks.append((chunk_idx, result))
                if verbose:
                    print(f"❌ Chunk {chunk_idx+1}/{num_chunks}: Failed - {result.get('error', 'unknown')}")

                # Save failed chunk info
                if args.save_failed_chunks:
                    failed_dir = Path(args.save_dir).expanduser() / "failed_chunks"
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    with open(failed_dir / f"chunk_{chunk_idx:04d}_error.txt", "w") as f:
                        f.write(f"Chunk {chunk_idx}\n")
                        f.write(f"Battles: {chunk_battles}\n")
                        f.write(f"Error: {result.get('error', 'unknown')}\n")
                        f.write(f"\nStderr:\n{result.get('stderr', 'N/A')}\n")

    else:
        # Sequential execution (original behavior)
        for chunk_idx in range(num_chunks):
            chunk_battles = min(chunk_size, args.num_battles - chunk_idx * chunk_size)

            if verbose:
                print(f"Chunk {chunk_idx+1}/{num_chunks} ({chunk_battles} battles)...")

            result = run_with_retry(
                chunk_idx=chunk_idx,
                chunk_battles=chunk_battles,
                base_args=base_args,
                max_retries=args.max_retries,
                timeout=args.timeout,
                verbose=verbose,
            )

            if result["success"]:
                total_completed += result["battles_completed"]
                if verbose:
                    print(f"  ✓ Completed in {result['time']:.1f}s ({result['rate']:.1f} battles/sec)")
            else:
                total_failed += chunk_battles
                failed_chunks.append((chunk_idx, result))
                if verbose:
                    print(f"  ❌ Failed after {args.max_retries} attempts")
                    print(f"     Error: {result.get('error', 'unknown')}")

                # Save failed chunk info
                if args.save_failed_chunks:
                    failed_dir = Path(args.save_dir).expanduser() / "failed_chunks"
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    with open(failed_dir / f"chunk_{chunk_idx:04d}_error.txt", "w") as f:
                        f.write(f"Chunk {chunk_idx}\n")
                        f.write(f"Battles: {chunk_battles}\n")
                        f.write(f"Error: {result.get('error', 'unknown')}\n")
                        f.write(f"\nStderr:\n{result.get('stderr', 'N/A')}\n")

            if verbose:
                print()

    # Final statistics
    total_time = time.time() - total_start_time
    total_rate = total_completed / total_time if total_time > 0 else 0
    success_rate = total_completed / args.num_battles * 100

    if verbose:
        print("=" * 70)
        print("SELF-PLAY COMPLETE")
        print("=" * 70)
        print(f"Total battles: {args.num_battles}")
        print(f"Completed: {total_completed} ({success_rate:.1f}%)")
        print(f"Failed: {total_failed} ({100-success_rate:.1f}%)")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average rate: {total_rate:.1f} battles/sec")
        print("=" * 70)

        if failed_chunks:
            print(f"\n⚠️  {len(failed_chunks)} chunks failed:")
            for chunk_idx, result in failed_chunks:
                print(f"  - Chunk {chunk_idx}: {result.get('error', 'unknown')}")

            if args.save_failed_chunks:
                failed_dir = Path(args.save_dir).expanduser() / "failed_chunks"
                print(f"\nFailed chunk logs saved to: {failed_dir}")

    # Exit code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
