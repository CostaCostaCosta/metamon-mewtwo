#!/usr/bin/env python3
"""
Comprehensive benchmark for batched AMAGO inference with PyKMN.

Measures:
- Throughput (steps/sec, battles/sec)
- Per-step latency breakdown (inference vs sim vs overhead)
- GPU memory usage
- Scaling behavior across batch sizes
"""

import os
import time
import psutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Set cache directory
os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

import torch
from metamon.env.pykmn import (
    load_random_teams,
    PyKMNVectorEnv,
    LocalPolicyRunner,
)
from metamon.interface import get_observation_space, get_reward_function
from metamon.rl.pretrained import get_pretrained_model


@dataclass
class BenchmarkResult:
    """Results from benchmarking a specific batch size."""
    batch_size: int
    num_steps: int
    total_time: float
    mean_step_time: float
    std_step_time: float
    mean_infer_time: float
    mean_sim_time: float
    steps_per_sec: float
    peak_vram_mb: float
    speedup: float = 1.0


def benchmark_batch_size(
    batch_size: int,
    num_steps: int = 100,
    warmup_steps: int = 10,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark batched inference with detailed metrics.

    Args:
        batch_size: Number of parallel environments
        num_steps: Number of steps to benchmark (after warmup)
        warmup_steps: Number of warmup steps (excluded from metrics)
        verbose: Print progress

    Returns:
        BenchmarkResult with detailed metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Benchmarking batch_size={batch_size}")
        print(f"{'='*70}")

    # Load teams
    team_dir = Path.home() / "metamon_cache" / "teams" / "modern_replays_v2"
    teams_p1 = load_random_teams(team_dir, "gen1ou", batch_size)
    teams_p2 = load_random_teams(team_dir, "gen1ou", batch_size)

    # Get observation space and reward function
    pretrained_cls = get_pretrained_model("SyntheticRLV2")
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    # Create vectorized environment
    vec_env = PyKMNVectorEnv(
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        num_envs=batch_size,
        obs_space=obs_space,
        reward_fn=reward_fn,
        battle_format="gen1ou",
        track_trajectories=False,
    )

    # Create policy runner
    policy = LocalPolicyRunner(
        model_name="SyntheticRLV2",
        checkpoint=48,
        device="cuda",
        use_amp=True,
        verbose=False,
    )

    # Reset environment
    obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()

    if verbose:
        print(f"✓ Environment and model loaded")

    # Measure VRAM before warmup
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup (compile, etc.)
    if verbose:
        print(f"Running {warmup_steps} warmup steps...")

    for _ in range(warmup_steps):
        actions_p1 = policy.infer(obs_p1, masks_p1)
        actions_p2 = policy.infer(obs_p2, masks_p2)
        obs_p1, obs_p2, rew_p1, rew_p2, dones, _ = vec_env.step(actions_p1, actions_p2)
        policy.update_rewards(rew_p1)
        if dones.any():
            policy.reset_hidden_state_for_dones(dones)
        masks_p1, masks_p2 = vec_env._extract_legal_masks()

    torch.cuda.synchronize()

    if verbose:
        print(f"✓ Warmup complete, starting benchmark...")

    # Benchmark proper
    step_times = []
    infer_times = []
    sim_times = []

    for step in range(num_steps):
        # Measure inference time
        torch.cuda.synchronize()
        t_infer_start = time.perf_counter()

        actions_p1 = policy.infer(obs_p1, masks_p1)
        actions_p2 = policy.infer(obs_p2, masks_p2)

        torch.cuda.synchronize()
        t_infer_end = time.perf_counter()
        infer_time = t_infer_end - t_infer_start

        # Measure simulation time
        t_sim_start = time.perf_counter()

        obs_p1, obs_p2, rew_p1, rew_p2, dones, _ = vec_env.step(actions_p1, actions_p2)
        policy.update_rewards(rew_p1)
        if dones.any():
            policy.reset_hidden_state_for_dones(dones)
        masks_p1, masks_p2 = vec_env._extract_legal_masks()

        t_sim_end = time.perf_counter()
        sim_time = t_sim_end - t_sim_start

        # Total step time
        step_time = t_infer_end - t_infer_start + sim_time

        step_times.append(step_time)
        infer_times.append(infer_time)
        sim_times.append(sim_time)

        if verbose and step % 20 == 0:
            print(f"  Step {step:3d}: {step_time*1000:6.2f}ms "
                  f"(infer: {infer_time*1000:5.2f}ms, sim: {sim_time*1000:5.2f}ms)")

    # Compute statistics
    total_time = np.sum(step_times)
    mean_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    mean_infer_time = np.mean(infer_times)
    mean_sim_time = np.mean(sim_times)
    steps_per_sec = (num_steps * batch_size) / total_time

    # Measure peak VRAM
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024 ** 2)

    if verbose:
        print(f"\n✓ Benchmark complete!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Mean step time: {mean_step_time*1000:.2f}ms ± {std_step_time*1000:.2f}ms")
        print(f"  Mean inference time: {mean_infer_time*1000:.2f}ms")
        print(f"  Mean simulation time: {mean_sim_time*1000:.2f}ms")
        print(f"  Steps/sec: {steps_per_sec:.1f}")
        print(f"  Peak VRAM: {peak_vram_mb:.1f} MB")

    return BenchmarkResult(
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
        mean_step_time=mean_step_time,
        std_step_time=std_step_time,
        mean_infer_time=mean_infer_time,
        mean_sim_time=mean_sim_time,
        steps_per_sec=steps_per_sec,
        peak_vram_mb=peak_vram_mb,
    )


def main():
    print("=" * 70)
    print("BATCHED AMAGO INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)

    # Benchmark different batch sizes
    batch_sizes = [1, 4, 16, 64]
    results: List[BenchmarkResult] = []

    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_size(
                batch_size=batch_size,
                num_steps=100,
                warmup_steps=10,
                verbose=True,
            )
            results.append(result)

            # Clear CUDA cache between runs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Benchmark failed for batch_size={batch_size}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Compute speedups relative to baseline
    if results:
        baseline_steps_per_sec = results[0].steps_per_sec
        for result in results:
            result.speedup = result.steps_per_sec / baseline_steps_per_sec

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Batch':<8} {'Steps/sec':<12} {'Step (ms)':<12} {'Infer (ms)':<12} "
          f"{'Sim (ms)':<12} {'VRAM (MB)':<12} {'Speedup':<10}")
    print("-" * 70)

    for result in results:
        print(
            f"{result.batch_size:<8} "
            f"{result.steps_per_sec:<12.1f} "
            f"{result.mean_step_time*1000:<12.2f} "
            f"{result.mean_infer_time*1000:<12.2f} "
            f"{result.mean_sim_time*1000:<12.2f} "
            f"{result.peak_vram_mb:<12.1f} "
            f"{result.speedup:<10.2f}x"
        )

    # Additional analysis
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("ANALYSIS")
        print(f"{'='*70}")

        # Inference scaling
        batch_1_infer = results[0].mean_infer_time
        batch_16_infer = results[2].mean_infer_time if len(results) > 2 else None

        if batch_16_infer:
            infer_speedup = (batch_1_infer * 16) / batch_16_infer
            print(f"Inference speedup (batch=16 vs batch=1): {infer_speedup:.1f}x")
            print(f"  Batch=1 inference: {batch_1_infer*1000:.2f}ms per step")
            print(f"  Batch=16 inference: {batch_16_infer*1000:.2f}ms per step")
            print(f"  Amortized per-env: {batch_16_infer*1000/16:.2f}ms")

        # Expected battles/sec (assuming 50 steps/battle)
        print(f"\nEstimated battles/sec (50 steps/battle):")
        for result in results:
            battles_per_sec = result.steps_per_sec / 50
            print(f"  Batch={result.batch_size}: {battles_per_sec:.1f} battles/sec")

    print(f"\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
