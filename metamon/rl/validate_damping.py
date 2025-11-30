"""
Validation script for Dynamic Damping v1.

This script analyzes training logs to verify that the adaptive controller is
working correctly according to the success criteria:

1. KL Divergence: hovering around 0.01-0.03 (not drifting upward)
2. Policy Entropy: in range 0.6-1.2 (not collapsing to 0.3-0.4)
3. KL Coefficient: oscillating up/down with KL (not monotone decay)
4. Learning Rate: visibly shrinking when KL high (not pegged at max)

Usage:
    python -m metamon.rl.validate_damping <log_dir> [--start_step START] [--end_step END]

The script will:
- Read train-update/* metrics from TensorBoard logs
- Analyze metrics over the specified step range (default: first 20-30% of training)
- Generate a report with pass/fail for each criterion
- Optionally generate plots showing the key metrics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_tensorboard_logs(log_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load TensorBoard event files and extract train-update metrics.

    Returns:
        Dictionary mapping metric names to list of (step, value) tuples
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("ERROR: tensorboard not installed. Install with: pip install tensorboard")
        sys.exit(1)

    # Find event files
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"ERROR: No TensorBoard event files found in {log_dir}")
        sys.exit(1)

    print(f"Found {len(event_files)} event file(s)")

    # Load all scalar data
    metrics: Dict[str, List[Tuple[int, float]]] = {}

    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()

        # Extract all scalars
        for tag in ea.Tags()['scalars']:
            if tag not in metrics:
                metrics[tag] = []

            events = ea.Scalars(tag)
            for event in events:
                metrics[tag].append((event.step, event.value))

    # Sort by step
    for tag in metrics:
        metrics[tag] = sorted(metrics[tag], key=lambda x: x[0])

    return metrics


def filter_by_step_range(
    metrics: Dict[str, List[Tuple[int, float]]],
    start_step: Optional[int] = None,
    end_step: Optional[int] = None
) -> Dict[str, List[Tuple[int, float]]]:
    """Filter metrics to only include values in the specified step range."""
    filtered = {}
    for tag, values in metrics.items():
        filtered_values = []
        for step, value in values:
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            filtered_values.append((step, value))

        if filtered_values:
            filtered[tag] = filtered_values

    return filtered


def get_metric_stats(values: List[Tuple[int, float]]) -> Dict[str, float]:
    """Compute statistics for a metric."""
    if not values:
        return {}

    vals = [v for _, v in values]
    return {
        "mean": np.mean(vals),
        "std": np.std(vals),
        "min": np.min(vals),
        "max": np.max(vals),
        "median": np.median(vals),
        "count": len(vals),
    }


def check_kl_divergence(metrics: Dict[str, List[Tuple[int, float]]]) -> Tuple[bool, str]:
    """Check KL divergence criterion."""
    kl_tag = "train-update/KL Divergence"

    if kl_tag not in metrics:
        return False, f"FAIL: Metric '{kl_tag}' not found in logs"

    stats = get_metric_stats(metrics[kl_tag])

    if not stats:
        return False, "FAIL: No KL divergence data"

    # Check 1: Mean should be in reasonable range (0.01 ± 0.02)
    target_range = (0.005, 0.05)
    mean_in_range = target_range[0] <= stats["mean"] <= target_range[1]

    # Check 2: No significant upward drift
    # Split into first and second half, compare means
    n = len(metrics[kl_tag])
    first_half = [v for _, v in metrics[kl_tag][:n//2]]
    second_half = [v for _, v in metrics[kl_tag][n//2:]]

    drift = np.mean(second_half) - np.mean(first_half)
    drift_ratio = drift / np.mean(first_half) if np.mean(first_half) > 0 else 0
    no_drift = drift_ratio < 0.5  # Allow up to 50% increase

    # Check 3: Values mostly in target band [0.007, 0.03]
    vals = [v for _, v in metrics[kl_tag]]
    in_band = sum(0.007 <= v <= 0.03 for v in vals) / len(vals)
    mostly_in_band = in_band >= 0.5  # At least 50% should be in band

    passed = mean_in_range and no_drift and mostly_in_band

    msg = f"""{'PASS' if passed else 'FAIL'}: KL Divergence
  Mean: {stats['mean']:.4f} (target: {target_range[0]}-{target_range[1]})
  Range: [{stats['min']:.4f}, {stats['max']:.4f}]
  Drift: {drift:+.4f} ({drift_ratio:+.1%})
  In band [0.007, 0.03]: {in_band:.1%}
  Checks: mean_ok={mean_in_range}, no_drift={no_drift}, in_band={mostly_in_band}"""

    return passed, msg


def check_policy_entropy(metrics: Dict[str, List[Tuple[int, float]]]) -> Tuple[bool, str]:
    """Check policy entropy criterion."""
    ent_tag = "train-update/Policy Entropy"

    if ent_tag not in metrics:
        return False, f"FAIL: Metric '{ent_tag}' not found in logs"

    stats = get_metric_stats(metrics[ent_tag])

    if not stats:
        return False, "FAIL: No policy entropy data"

    # Check: Mean should be in healthy range (0.6-1.5)
    target_range = (0.6, 1.5)
    mean_in_range = target_range[0] <= stats["mean"] <= target_range[1]

    # Check: Not collapsed (no sustained period below 0.4)
    vals = [v for _, v in metrics[ent_tag]]
    collapsed = sum(v < 0.4 for v in vals) / len(vals)
    not_collapsed = collapsed < 0.3  # Less than 30% of time below 0.4

    passed = mean_in_range and not_collapsed

    msg = f"""{'PASS' if passed else 'FAIL'}: Policy Entropy
  Mean: {stats['mean']:.4f} (target: {target_range[0]}-{target_range[1]})
  Range: [{stats['min']:.4f}, {stats['max']:.4f}]
  Fraction below 0.4: {collapsed:.1%}
  Checks: mean_ok={mean_in_range}, not_collapsed={not_collapsed}"""

    return passed, msg


def check_kl_coefficient(metrics: Dict[str, List[Tuple[int, float]]]) -> Tuple[bool, str]:
    """Check KL coefficient oscillation criterion."""
    kl_coef_tag = "train-update/Damping/KL Coefficient"

    if kl_coef_tag not in metrics:
        return False, f"FAIL: Metric '{kl_coef_tag}' not found in logs"

    stats = get_metric_stats(metrics[kl_coef_tag])

    if not stats:
        return False, "FAIL: No KL coefficient data"

    # Check: Should oscillate (not monotone)
    vals = [v for _, v in metrics[kl_coef_tag]]

    # Count direction changes (up/down transitions)
    changes = 0
    for i in range(1, len(vals)):
        if i+1 < len(vals):
            if (vals[i] > vals[i-1] and vals[i+1] < vals[i]) or \
               (vals[i] < vals[i-1] and vals[i+1] > vals[i]):
                changes += 1

    oscillating = changes >= len(vals) * 0.1  # At least 10% of points are local extrema

    # Check: Range should be significant (not stuck)
    range_ratio = (stats["max"] - stats["min"]) / stats["mean"] if stats["mean"] > 0 else 0
    has_range = range_ratio > 0.2  # Range > 20% of mean

    passed = oscillating and has_range

    msg = f"""{'PASS' if passed else 'FAIL'}: KL Coefficient Oscillation
  Mean: {stats['mean']:.4f}
  Range: [{stats['min']:.4f}, {stats['max']:.4f}]
  Range/mean: {range_ratio:.2f}
  Direction changes: {changes}/{len(vals)} ({changes/len(vals):.1%})
  Checks: oscillating={oscillating}, has_range={has_range}"""

    return passed, msg


def check_learning_rate(metrics: Dict[str, List[Tuple[int, float]]]) -> Tuple[bool, str]:
    """Check learning rate adaptation criterion."""
    lr_tag = "train-update/Damping/Learning Rate"

    if lr_tag not in metrics:
        return False, f"FAIL: Metric '{lr_tag}' not found in logs"

    stats = get_metric_stats(metrics[lr_tag])

    if not stats:
        return False, "FAIL: No learning rate data"

    # Check: Should not be pegged at max (should adapt)
    vals = [v for _, v in metrics[lr_tag]]
    max_lr = 1e-4  # From v1 config

    pegged = sum(abs(v - max_lr) < 1e-7 for v in vals) / len(vals)
    not_pegged = pegged < 0.8  # Less than 80% of time at max

    # Check: Should have visible variation
    range_ratio = (stats["max"] - stats["min"]) / stats["mean"] if stats["mean"] > 0 else 0
    has_variation = range_ratio > 0.2  # Range > 20% of mean

    passed = not_pegged and has_variation

    msg = f"""{'PASS' if passed else 'FAIL'}: Learning Rate Adaptation
  Mean: {stats['mean']:.6f}
  Range: [{stats['min']:.6f}, {stats['max']:.6f}]
  Range/mean: {range_ratio:.2f}
  Fraction at max: {pegged:.1%}
  Checks: not_pegged={not_pegged}, has_variation={has_variation}"""

    return passed, msg


def generate_report(
    metrics: Dict[str, List[Tuple[int, float]]],
    start_step: Optional[int],
    end_step: Optional[int]
) -> bool:
    """Generate validation report and return overall pass/fail."""
    print("\n" + "="*70)
    print("DYNAMIC DAMPING V1 VALIDATION REPORT")
    print("="*70)

    if start_step or end_step:
        print(f"\nStep range: {start_step or 'start'} to {end_step or 'end'}")

    print("\nAvailable metrics:")
    for tag in sorted(metrics.keys()):
        if metrics[tag]:
            steps = [s for s, _ in metrics[tag]]
            print(f"  - {tag}: {len(steps)} points, steps {min(steps)}-{max(steps)}")

    print("\n" + "-"*70)
    print("VALIDATION CHECKS")
    print("-"*70)

    checks = [
        ("KL Divergence", check_kl_divergence),
        ("Policy Entropy", check_policy_entropy),
        ("KL Coefficient", check_kl_coefficient),
        ("Learning Rate", check_learning_rate),
    ]

    results = []
    for name, check_func in checks:
        passed, msg = check_func(metrics)
        results.append(passed)
        print(f"\n{msg}")

    print("\n" + "="*70)
    passed_count = sum(results)
    total_count = len(results)
    print(f"OVERALL: {passed_count}/{total_count} checks passed")

    if all(results):
        print("✓ All checks passed! Controller is working correctly.")
        print("  You can proceed with full training run or move to v2 (with gentle annealing).")
    elif passed_count >= total_count * 0.75:
        print("⚠ Most checks passed, but some issues detected.")
        print("  Review failed checks and consider tuning parameters.")
    else:
        print("✗ Multiple checks failed. Controller may not be wired correctly.")
        print("  Debug the implementation before proceeding.")

    print("="*70 + "\n")

    return all(results)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Dynamic Damping v1 training logs"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=None,
        help="Start step for analysis (default: beginning)"
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=None,
        help="End step for analysis (default: end)"
    )
    parser.add_argument(
        "--auto_range",
        action="store_true",
        help="Automatically analyze first 20-30%% of training"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        sys.exit(1)

    print(f"Loading logs from: {log_dir}")
    metrics = load_tensorboard_logs(log_dir)

    if not metrics:
        print("ERROR: No metrics found in logs")
        sys.exit(1)

    # Auto-detect range if requested
    start_step = args.start_step
    end_step = args.end_step

    if args.auto_range and not (start_step or end_step):
        # Find max step across all metrics
        max_step = 0
        for values in metrics.values():
            if values:
                max_step = max(max_step, max(s for s, _ in values))

        # Use first 20-30% of training
        start_step = int(max_step * 0.05)  # Skip initial warmup
        end_step = int(max_step * 0.30)
        print(f"Auto-detected range: steps {start_step}-{end_step} (first 30% after warmup)")

    # Filter metrics to range
    if start_step or end_step:
        metrics = filter_by_step_range(metrics, start_step, end_step)

    # Generate report
    all_passed = generate_report(metrics, start_step, end_step)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
