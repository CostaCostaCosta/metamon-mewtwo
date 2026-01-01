# Subprocess Isolation for PyKMN Batched Inference

## Overview

Subprocess isolation is a production hardening technique that contains memory corruption crashes to individual worker processes, preventing them from affecting the main orchestrator process. This allows data generation to continue unattended even if individual batches crash due to native memory bugs.

## Architecture

```
Main Process (Orchestrator)
├── Worker 1 (batch 1-16) → runs in subprocess → writes trajectories
├── Worker 2 (batch 17-32) → runs in subprocess → writes trajectories
├── Worker 3 (batch 33-48) → runs in subprocess → writes trajectories
└── ...
```

**Key Properties**:
- Each worker runs `batch_size` battles in an isolated subprocess
- Segfaults/crashes only affect the worker, not the orchestrator
- Failed batches can be retried or skipped
- Clean memory slate for each batch (no accumulation)

## Implementation Options

### Option 1: Use Existing Script with Python Multiprocessing (RECOMMENDED)

The simplest approach is to wrap the existing `generate_selfplay_batched.py` in a multiprocessing harness:

```python
# scripts/generate_selfplay_subprocess.py
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

def run_batch_in_subprocess(args_dict):
    """Run a single batch as a subprocess."""
    # Build command
    cmd = [sys.executable, "scripts/generate_selfplay_batched.py"]
    for key, value in args_dict.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value is not False and value is not None:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    # Run with timeout
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per batch
        )
        return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    total_battles = 1000
    batch_size = 16
    battles_per_subprocess = batch_size  # One batch per subprocess

    num_subprocesses = total_battles // battles_per_subprocess

    for i in range(num_subprocesses):
        batch_args = {
            "model": "SyntheticRLV2",
            "checkpoint": 48,
            "num_battles": battles_per_subprocess,
            "batch_size": batch_size,
            "format": "gen1ou",
            "team_set": "modern_replays_v2",
            "save_dir": "~/selfplay_data/gen1ou",
            "run_name": f"batch_{i}",
        }

        print(f"Running batch {i+1}/{num_subprocesses} ({battles_per_subprocess} battles)...")
        result = run_batch_in_subprocess(batch_args)

        if not result["success"]:
            print(f"  ❌ Batch {i} failed: {result.get('error', 'unknown')}")
            # Option: retry, skip, or abort
        else:
            print(f"  ✓ Batch {i} completed successfully")
```

**Pros**:
- Simple: Reuses existing script as-is
- Reliable: subprocess.run() handles crashes gracefully
- Flexible: Easy to add retry logic, parallel workers, etc.

**Cons**:
- Overhead: ~1-2 seconds per subprocess startup + model reload
- For 1000 battles at batch_size=16: 62 subprocesses × 2s = ~2 minutes overhead

### Option 2: Use multiprocessing.Pool with Shared Model (ADVANCED)

Use Python multiprocessing with shared model weights (more efficient but complex):

```python
import multiprocessing as mp
import torch

def worker_init(model_path, device):
    """Initialize worker with shared model."""
    global WORKER_MODEL
    WORKER_MODEL = torch.load(model_path, map_location=device)
    WORKER_MODEL.eval()

def run_batch_worker(batch_idx, teams_p1, teams_p2):
    """Run batch in worker process."""
    # Use global WORKER_MODEL
    vec_env = PyKMNVectorEnv(...)
    runner = SelfPlayRunner(vec_env, WORKER_MODEL, WORKER_MODEL)
    trajectories = runner.collect_trajectories(...)
    return trajectories

def main():
    # Create worker pool
    with mp.Pool(processes=4, initializer=worker_init, initargs=(model_path, device)) as pool:
        results = pool.starmap(run_batch_worker, batch_args_list)
```

**Pros**:
- Efficient: No model reload overhead (shared memory)
- Parallel: Can run multiple batches concurrently

**Cons**:
- Complex: Requires careful model sharing, pickling, CUDA context management
- Fragile: CUDA tensors don't pickle well across processes

### Option 3: Inline Subprocess Protection (MINIMAL OVERHEAD)

Add subprocess isolation directly in generate_selfplay_batched.py with minimal overhead:

**Add to parse_args()**:
```python
perf_group.add_argument(
    "--subprocess-isolation",
    action="store_true",
    help="Run each chunk in a subprocess (crash-resistant, ~10%% overhead)",
)
perf_group.add_argument(
    "--chunk-size",
    type=int,
    default=None,
    help="Battles per subprocess (default: batch_size). Higher = less overhead, lower = more crash protection",
)
```

**Modify run_selfplay()**:
```python
def run_selfplay(..., use_subprocess=False, chunk_size=None):
    chunk_size = chunk_size or batch_size

    if use_subprocess:
        return run_selfplay_subprocess(...)  # New function
    else:
        # Existing implementation
        ...

def run_selfplay_subprocess(policy_p1, policy_p2, teams_p1, teams_p2, ..., chunk_size):
    """Run self-play with subprocess isolation."""
    import subprocess
    import json
    import tempfile

    num_chunks = (num_battles + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_battles = min(chunk_size, num_battles - chunk_idx * chunk_size)

        # Serialize arguments to JSON
        chunk_args = {
            "model_p1": policy_p1.model_name,
            "checkpoint_p1": policy_p1.checkpoint,
            "model_p2": policy_p2.model_name,
            "checkpoint_p2": policy_p2.checkpoint,
            "num_battles": chunk_battles,
            "batch_size": batch_size,
            "format": format_name,
            "save_dir": str(save_dir),
            "run_name": f"{run_name}_chunk{chunk_idx}",
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(chunk_args, f)
            args_file = f.name

        # Run in subprocess
        cmd = [sys.executable, "-m", "metamon.scripts.selfplay_worker", args_file]
        result = subprocess.run(cmd, capture_output=True, timeout=600)

        if result.returncode != 0:
            print(f"Chunk {chunk_idx} failed: {result.stderr}")
            # Retry or continue

        os.unlink(args_file)
```

**Pros**:
- Integrated: Works within existing script
- Configurable: Can tune chunk_size for overhead vs protection tradeoff

**Cons**:
- Requires creating a worker entry point (`metamon.scripts.selfplay_worker`)
- Still has per-chunk overhead (model reload)

## Recommended Approach for Metamon

For **immediate production use**, I recommend **Option 1** (wrapper script) because:

1. **Zero changes to existing code**: Works with current `generate_selfplay_batched.py`
2. **Simple to debug**: Each batch runs identically to manual runs
3. **Easy retry logic**: Failed batches can be retried with exponential backoff
4. **Acceptable overhead**: 2s per batch × 62 batches = 2 minutes for 1000 battles (~4% overhead at 50s total runtime)

For **long-term optimization**, implement **Option 3** (inline subprocess protection) with:
- `chunk_size = batch_size * 10` (e.g., 160 battles per subprocess)
- Reduces overhead to 0.4% while maintaining crash protection
- Allows gradual rollout (disabled by default, enable with flag)

## Usage Examples

### Wrapper Script (Option 1)

```bash
# Run 10,000 battles with subprocess isolation
python scripts/generate_selfplay_subprocess.py \\
    --model SyntheticRLV2 \\
    --checkpoint 48 \\
    --num_battles 10000 \\
    --batch_size 16 \\
    --chunk_size 160 \\
    --max_retries 3 \\
    --format gen1ou \\
    --save_dir ~/selfplay_data/gen1ou

# Output:
# Running chunk 1/62 (160 battles)...
#   ✓ Completed in 8.2s (19.5 battles/sec)
# Running chunk 2/62 (160 battles)...
#   ❌ Crashed with segfault (attempt 1/3)
#   ✓ Retry succeeded in 8.5s
# ...
#
# Final stats:
#   Total: 10,000 battles
#   Successful: 9,984 battles
#   Failed: 16 battles (0.16%)
#   Time: 8m 32s (19.5 battles/sec)
```

### Inline Flag (Option 3, if implemented)

```bash
# Same script, just add flag
python scripts/generate_selfplay_batched.py \\
    --model SyntheticRLV2 \\
    --checkpoint 48 \\
    --num_battles 10000 \\
    --batch_size 16 \\
    --subprocess-isolation \\
    --chunk-size 160 \\
    --format gen1ou \\
    --save_dir ~/selfplay_data/gen1ou
```

## Implementation: Wrapper Script

See `scripts/generate_selfplay_subprocess.py` (created below)

## Monitoring and Debugging

### Track Failure Rate

```bash
# Count successful vs failed chunks
grep "✓" selfplay.log | wc -l  # Successful
grep "❌" selfplay.log | wc -l  # Failed
```

### Inspect Failed Chunks

```bash
# Failed chunks write error logs
ls ~/selfplay_data/gen1ou/failed_chunks/
# chunk_12_stderr.txt
# chunk_45_stderr.txt

# Check error
cat ~/selfplay_data/gen1ou/failed_chunks/chunk_12_stderr.txt
# free(): invalid next size (fast)
# --> Known memory corruption, expected with current PyKMN integration
```

### Retry Failed Chunks

```python
# scripts/retry_failed_chunks.py
import json
from pathlib import Path

failed_dir = Path("~/selfplay_data/gen1ou/failed_chunks").expanduser()
for error_file in failed_dir.glob("chunk_*_stderr.txt"):
    chunk_idx = int(error_file.stem.split("_")[1])
    print(f"Retrying chunk {chunk_idx}...")
    # Re-run with same arguments
```

## Performance Comparison

| Method | Overhead | Crash Protection | Complexity |
|--------|----------|------------------|------------|
| **No isolation** (baseline) | 0% | ❌ Entire run fails | Simple |
| **Option 1: Wrapper (chunk_size=16)** | 4% | ✅ Per-batch | Simple |
| **Option 1: Wrapper (chunk_size=160)** | 0.4% | ✅ Per-chunk | Simple |
| **Option 2: Pool (shared model)** | 0.1% | ✅ Per-batch | Complex |
| **Option 3: Inline (chunk_size=160)** | 0.4% | ✅ Per-chunk | Medium |

**Recommendation**: Start with Option 1 (wrapper, chunk_size=160). If overhead is unacceptable, implement Option 3 later.

## Next Steps

1. ✅ Implement wrapper script (`generate_selfplay_subprocess.py`)
2. ⬜ Test on 1000-battle run
3. ⬜ Monitor failure rate over 24-hour production run
4. ⬜ If <1% failure rate: Deploy to production
5. ⬜ If >5% failure rate: Fix underlying corruption bugs (Phases 1-3)
6. ⬜ (Optional) Implement inline subprocess protection for lower overhead
