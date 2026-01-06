#!/usr/bin/env python3
"""
Diagnostic wrapper for selfplay crashes.

Adds comprehensive logging to pinpoint exact crash location.
"""

import os
import sys
import signal
import traceback
from pathlib import Path

# Set cache directory before imports
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

# Enable Python fault handler for segfault tracebacks
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

# Track what we're doing when crash occurs
current_operation = "startup"
battles_completed = 0
current_step = 0

def log_operation(op: str):
    """Log current operation."""
    global current_operation
    current_operation = op
    print(f"[DEBUG {battles_completed}b/{current_step}s] {op}", flush=True)

def signal_handler(signum, frame):
    """Handle crash signals."""
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"CRASH DETECTED: Signal {signum}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"Battles completed: {battles_completed}", file=sys.stderr)
    print(f"Current step: {current_step}", file=sys.stderr)
    print(f"Current operation: {current_operation}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGABRT, signal_handler)
signal.signal(signal.SIGSEGV, signal_handler)

import numpy as np
import time
from metamon.env.pykmn import (
    load_random_teams,
    PyKMNVectorEnv,
    LocalPolicyRunner,
    SelfPlayRunner,
    save_trajectories,
    precompute_mappings,
)
from metamon.rl.pretrained import get_pretrained_model

# Monkey-patch critical functions to add logging
original_step = PyKMNVectorEnv.step
def logged_step(self, actions_p1, actions_p2):
    global current_step
    current_step += 1
    log_operation(f"step() start (step #{current_step})")

    try:
        log_operation("step: get legal masks")
        result = original_step(self, actions_p1, actions_p2)

        log_operation(f"step() complete: {result[5]['num_done']} done")
        return result
    except Exception as e:
        print(f"EXCEPTION in step(): {type(e).__name__}: {e}", file=sys.stderr)
        raise

PyKMNVectorEnv.step = logged_step

original_reset = PyKMNVectorEnv.reset
def logged_reset(self):
    log_operation("reset() start")
    try:
        result = original_reset(self)
        log_operation("reset() complete")
        return result
    except Exception as e:
        print(f"EXCEPTION in reset(): {type(e).__name__}: {e}", file=sys.stderr)
        raise

PyKMNVectorEnv.reset = logged_reset

original_get_completed = PyKMNVectorEnv.get_completed_trajectories
def logged_get_completed(self):
    log_operation("get_completed_trajectories() start")
    try:
        result = original_get_completed(self)
        log_operation(f"get_completed_trajectories() complete: {len(result)} trajectories")
        return result
    except Exception as e:
        print(f"EXCEPTION in get_completed_trajectories(): {type(e).__name__}: {e}", file=sys.stderr)
        raise

PyKMNVectorEnv.get_completed_trajectories = logged_get_completed

# Patch trajectory saving
from metamon.env.pykmn.trajectory_saver import save_trajectories as original_save

def logged_save_trajectories(trajectories, output_dir, *args, **kwargs):
    log_operation(f"save_trajectories() start: {len(trajectories)} trajectories")
    try:
        result = original_save(trajectories, output_dir, *args, **kwargs)
        log_operation("save_trajectories() complete")
        return result
    except Exception as e:
        print(f"EXCEPTION in save_trajectories(): {type(e).__name__}: {e}", file=sys.stderr)
        raise

import metamon.env.pykmn.trajectory_saver
metamon.env.pykmn.trajectory_saver.save_trajectories = logged_save_trajectories

def main():
    global battles_completed, current_step

    print("="*70)
    print("DIAGNOSTIC SELFPLAY CRASH TEST")
    print("="*70)
    print("This script adds extensive logging to pinpoint crash location")
    print("="*70)

    # Hardcoded test parameters
    model_name = "Kakuna"
    batch_size = 16
    num_battles = 500
    format_name = "gen1ou"
    team_set = "modern_replays_v2"
    save_dir = Path("/tmp/crash_test")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_operation("Loading teams")
    cache_dir = Path(os.environ["METAMON_CACHE_DIR"])
    team_dir = cache_dir / "teams" / team_set

    from metamon.env.pykmn import load_random_teams
    teams_p1 = load_random_teams(team_dir, format_name, batch_size)
    teams_p2 = load_random_teams(team_dir, format_name, batch_size)
    print(f"✓ Loaded {len(teams_p1) + len(teams_p2)} teams")

    log_operation("Loading model")
    policy = LocalPolicyRunner(
        model_name=model_name,
        checkpoint=None,
        device="cuda",
        use_amp=True,
        verbose=False,
    )
    print(f"✓ Model loaded")

    log_operation("Creating observation space and reward function")
    pretrained_cls = get_pretrained_model(model_name)
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    log_operation("Precomputing mappings")
    mappings = precompute_mappings()

    log_operation("Creating vectorized environment")
    vec_env = PyKMNVectorEnv(
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        num_envs=batch_size,
        obs_space=obs_space,
        reward_fn=reward_fn,
        battle_format=format_name,
        track_trajectories=True,
    )
    print(f"✓ Created vectorized environment")

    log_operation("Creating self-play runner")
    runner = SelfPlayRunner(vec_env, policy, policy)

    # Memory monitoring
    try:
        import psutil
        process = psutil.Process()
        has_psutil = True
    except ImportError:
        has_psutil = False

    all_trajectories = []
    start_time = time.time()

    log_operation("Starting main loop")
    print("\nStarting data collection...")
    print("Watch for the last logged operation before crash\n")

    while battles_completed < num_battles:
        battles_remaining = num_battles - battles_completed
        chunk_size = min(batch_size, battles_remaining)

        # Progress logging
        if battles_completed > 0 and battles_completed % 16 == 0:
            elapsed = time.time() - start_time
            rate = battles_completed / elapsed
            mem_str = ""
            if has_psutil:
                mem_mb = process.memory_info().rss / 1024**2
                mem_str = f" | Memory: {mem_mb:.1f} MB"
            print(f"Progress: {battles_completed}/{num_battles} | "
                  f"Rate: {rate:.1f} b/s{mem_str}")

        log_operation(f"Collecting batch (battles {battles_completed}-{battles_completed+chunk_size})")

        try:
            # This is where the crash happens
            log_operation("runner.collect_trajectories() call")
            trajectories = runner.collect_trajectories(
                num_battles=chunk_size,
                max_steps_per_battle=1000,
                verbose=False,
            )

            log_operation(f"Batch collected: {len(trajectories)} trajectories")
            all_trajectories.extend(trajectories)
            battles_completed += len(trajectories)

            # Save incrementally every 100 battles
            if len(all_trajectories) >= 100:
                log_operation(f"Saving {len(all_trajectories)} trajectories")
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"crash_test_{timestamp}"
                output_dir = save_dir / run_name / format_name
                output_dir.mkdir(parents=True, exist_ok=True)

                logged_save_trajectories(
                    all_trajectories,
                    output_dir,
                    mappings=mappings,
                    battle_format=format_name,
                    verbose=False,
                )
                print(f"✓ Saved {len(all_trajectories)} trajectories")
                all_trajectories = []

                log_operation("Save complete, garbage collection")
                import gc
                gc.collect()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break

        except Exception as e:
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"EXCEPTION in main loop", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)
            print(f"Battles completed: {battles_completed}", file=sys.stderr)
            print(f"Current operation: {current_operation}", file=sys.stderr)
            print(f"Exception: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise

    log_operation("Complete!")
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Test Complete!")
    print(f"{'='*70}")
    print(f"Battles: {battles_completed}/{num_battles}")
    print(f"Time: {total_time:.1f}s")
    print(f"Rate: {battles_completed/total_time:.1f} battles/sec")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
