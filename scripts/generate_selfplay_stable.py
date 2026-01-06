#!/usr/bin/env python3
"""
Stable self-play data generation using subprocess isolation.

This script works around PyKMN memory corruption issues by running
batches in isolated subprocesses. It's slower than pure batched
inference but much more stable for long runs.

Usage:
    python scripts/generate_selfplay_stable.py \
        --model SyntheticRLV2 \
        --num_battles 1028 \
        --batch_size 64 \
        --chunk_size 256 \
        --format gen1ou \
        --team_set smogon_pass2 \
        --save_dir ~/metamon/trajectories/stable_run
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import shutil

def run_batch_in_subprocess(
    model: str,
    num_battles: int,
    batch_size: int,
    format: str,
    team_set: str,
    save_dir: str,
    checkpoint: int = None,
    temperature: float = 1.0,
) -> bool:
    """Run a batch of battles in an isolated subprocess."""
    cmd = [
        sys.executable,
        "scripts/generate_selfplay_batched.py",
        "--model", model,
        "--num_battles", str(num_battles),
        "--batch_size", str(batch_size),
        "--format", format,
        "--team_set", team_set,
        "--save_dir", save_dir,
        "--temperature", str(temperature),
    ]

    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env={**os.environ, "METAMON_CACHE_DIR": os.environ.get("METAMON_CACHE_DIR", str(Path.home() / "metamon_cache"))},
        )

        if result.returncode == 0:
            return True
        else:
            print(f"  ⚠️  Subprocess failed with code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[-500:]}")  # Last 500 chars of error
            return False

    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Subprocess timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  ⚠️  Subprocess error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Stable self-play data generation")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint epoch")
    parser.add_argument("--num_battles", type=int, required=True, help="Total battles to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for parallel battles")
    parser.add_argument("--chunk_size", type=int, default=256, help="Battles per subprocess chunk")
    parser.add_argument("--format", type=str, default="gen1ou", help="Battle format")
    parser.add_argument("--team_set", type=str, default="smogon_pass2", help="Team set to use")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()

    # Set cache directory
    if "METAMON_CACHE_DIR" not in os.environ:
        os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

    print("=" * 70)
    print("STABLE SELF-PLAY DATA GENERATION (SUBPROCESS ISOLATION)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Target battles: {args.num_battles}")
    print(f"Output: {args.save_dir}")
    print("=" * 70)

    # Create output directory
    output_path = Path(args.save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for chunks
    temp_dir = output_path / f"temp_{int(time.time())}"
    temp_dir.mkdir(exist_ok=True)

    battles_completed = 0
    chunk_num = 0
    start_time = time.time()

    while battles_completed < args.num_battles:
        battles_remaining = args.num_battles - battles_completed
        chunk_battles = min(args.chunk_size, battles_remaining)
        chunk_num += 1

        print(f"\nChunk {chunk_num}: Generating {chunk_battles} battles...")

        # Run chunk in subprocess
        chunk_dir = temp_dir / f"chunk_{chunk_num}"
        success = run_batch_in_subprocess(
            model=args.model,
            num_battles=chunk_battles,
            batch_size=args.batch_size,
            format=args.format,
            team_set=args.team_set,
            save_dir=str(chunk_dir),
            checkpoint=args.checkpoint,
            temperature=args.temperature,
        )

        if success:
            # Move generated files to final directory
            # Find the actual run directory (with timestamp)
            run_dirs = list(chunk_dir.glob(f"{args.model}_*"))
            if run_dirs:
                chunk_format_dir = run_dirs[0] / args.format
                if chunk_format_dir.exists():
                    final_format_dir = output_path / args.format
                    final_format_dir.mkdir(exist_ok=True)

                    # Move all .json.lz4 files
                    for file in chunk_format_dir.glob("*.json.lz4"):
                        dest = final_format_dir / f"{args.model}_chunk{chunk_num}_{file.name}"
                        shutil.move(str(file), str(dest))

                    num_files = len(list(final_format_dir.glob(f"*_chunk{chunk_num}_*.json.lz4")))
                    battles_completed += num_files

                    print(f"  ✓ Chunk {chunk_num} complete: {num_files} battles saved")
                    print(f"  Progress: {battles_completed}/{args.num_battles} battles")
                else:
                    print(f"  ⚠️  No trajectories found in chunk {chunk_num}")
            else:
                print(f"  ⚠️  No run directory found for chunk {chunk_num}")
        else:
            print(f"  ❌ Chunk {chunk_num} failed, retrying...")
            # Retry once
            success = run_batch_in_subprocess(
                model=args.model,
                num_battles=chunk_battles,
                batch_size=args.batch_size,
                format=args.format,
                team_set=args.team_set,
                save_dir=str(chunk_dir),
                checkpoint=args.checkpoint,
                temperature=args.temperature,
            )

            if not success:
                print(f"  ❌ Chunk {chunk_num} failed after retry, skipping")

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Final statistics
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("SELF-PLAY GENERATION COMPLETE")
    print("=" * 70)
    print(f"Battles completed: {battles_completed}/{args.num_battles}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Average rate: {battles_completed/elapsed:.1f} battles/sec")
    print(f"Output directory: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()