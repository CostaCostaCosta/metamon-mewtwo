#!/usr/bin/env python3
"""
PyKMN Debug Ladder - Master Test Runner
Runs all modes sequentially to isolate the failure layer
"""

import subprocess
import sys
import os
import time
from datetime import datetime


def run_mode(mode_num: int, script_name: str, description: str) -> dict:
    """Run a single test mode and capture results."""
    print(f"\n{'='*70}")
    print(f"RUNNING MODE {mode_num}: {description}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*70)

    start_time = time.time()

    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        # Parse output to find failure point
        output_lines = result.stdout.split('\n')
        failure_batch = None
        passed = result.returncode == 0

        if not passed:
            # Look for batch size failure indicators
            for line in output_lines:
                if "First failure at batch_size=" in line:
                    parts = line.split("batch_size=")
                    if len(parts) > 1:
                        try:
                            failure_batch = int(parts[1].split()[0])
                        except:
                            pass

        return {
            'mode': mode_num,
            'description': description,
            'passed': passed,
            'failure_batch': failure_batch,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            'mode': mode_num,
            'description': description,
            'passed': False,
            'failure_batch': None,
            'elapsed_time': 600,
            'stdout': '',
            'stderr': 'TIMEOUT after 10 minutes',
            'returncode': -1
        }
    except Exception as e:
        return {
            'mode': mode_num,
            'description': description,
            'passed': False,
            'failure_batch': None,
            'elapsed_time': time.time() - start_time,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def main():
    """Run all test modes and produce summary."""
    print("=" * 70)
    print("PyKMN STABILITY DEBUG LADDER")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run 5 test modes to isolate the failure layer:")
    print("  Mode 0: Single battle baseline")
    print("  Mode 1: Raw PyKMN stepping")
    print("  Mode 2: NumPy conversion")
    print("  Mode 3: Torch tensors")
    print("  Mode 4: Full batched pipeline")
    print("\nEach mode adds one layer to isolate the issue.")

    # Define test modes
    modes = [
        (0, 'test_mode0_single_battle.py', 'Single Battle Baseline'),
        (1, 'test_mode1_raw_stepping.py', 'Raw PyKMN Stepping'),
        (2, 'test_mode2_numpy_conversion.py', 'NumPy Conversion'),
        (3, 'test_mode3_torch_tensors.py', 'Torch Tensors'),
        (4, 'test_mode4_batched_pipeline.py', 'Full Batched Pipeline')
    ]

    results = []
    first_failure_mode = None

    # Run each mode
    for mode_num, script, description in modes:
        # Check if script exists
        if not os.path.exists(script):
            print(f"\n✗ Skipping Mode {mode_num}: Script {script} not found")
            continue

        result = run_mode(mode_num, script, description)
        results.append(result)

        # Track first failure
        if not result['passed'] and first_failure_mode is None:
            first_failure_mode = mode_num

        # Print immediate result
        if result['passed']:
            print(f"\n✓ Mode {mode_num} PASSED")
        else:
            print(f"\n✗ Mode {mode_num} FAILED")
            if result['failure_batch']:
                print(f"  Failed at batch_size={result['failure_batch']}")

    # Generate summary report
    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n## Test Results:")
    print("-" * 40)

    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"Mode {result['mode']}: {status} - {result['description']}")
        if result['failure_batch']:
            print(f"         Failed at batch_size={result['failure_batch']}")
        print(f"         Time: {result['elapsed_time']:.1f}s")

    print("\n## Root Cause Analysis:")
    print("-" * 40)

    if all(r['passed'] for r in results):
        print("✓ All modes passed! PyKMN integration appears stable.")
        print("\nRecommendation: The issues may be in a different code path")
        print("or require longer runs to manifest.")

    elif first_failure_mode is not None:
        print(f"✗ First failure in Mode {first_failure_mode}")

        failure_batch = None
        for r in results:
            if r['mode'] == first_failure_mode:
                failure_batch = r['failure_batch']
                break

        # Provide specific diagnosis
        if first_failure_mode == 0:
            print("\nDIAGNOSIS: PyKMN is unstable even for single battles")
            print("ROOT CAUSE: Core PyKMN engine or binding issue")
            print("\nRECOMMENDED ACTIONS:")
            print("1. Check PyKMN version and rebuild with debug symbols")
            print("2. Run with AddressSanitizer to catch memory issues")
            print("3. Report issue to PyKMN maintainers")

        elif first_failure_mode == 1:
            print("\nDIAGNOSIS: PyKMN fails when running multiple battles")
            if failure_batch == 128:
                print("ROOT CAUSE: Hard-coded buffer limit at 128 in PyKMN")
            else:
                print(f"ROOT CAUSE: Resource exhaustion at batch_size={failure_batch}")
            print("\nRECOMMENDED ACTIONS:")
            print("1. Search PyKMN source for '128' constants")
            print("2. Check for static arrays or ring buffers")
            print("3. Consider process isolation for batches > limit")

        elif first_failure_mode == 2:
            print("\nDIAGNOSIS: Feature extraction introduces memory issues")
            print("ROOT CAUSE: pykmn_to_features_raw holds views into mutable buffers")
            print("\nRECOMMENDED ACTIONS:")
            print("1. Add explicit copying at Python/C++ boundary")
            print("2. Review pykmn_to_features_raw implementation")
            print("3. Check for buffer reuse in feature extraction")

        elif first_failure_mode == 3:
            print("\nDIAGNOSIS: Torch tensor creation/management issue")
            print("ROOT CAUSE: Tensor aliasing or lifetime management")
            print("\nRECOMMENDED ACTIONS:")
            print("1. Force tensor.clone() after creation")
            print("2. Avoid torch.from_numpy() (use torch.tensor())")
            print("3. Check for tensor reference cycles")

        elif first_failure_mode == 4:
            print("\nDIAGNOSIS: Batching/stacking operations fail")
            print("ROOT CAUSE: numpy.stack or batch assembly issue")
            print("\nRECOMMENDED ACTIONS:")
            print("1. Use chunked batching (multiple smaller batches)")
            print("2. Check array stride/alignment before stacking")
            print("3. Pre-allocate batch tensors")

        if failure_batch and failure_batch == 128:
            print("\n⚠️  CRITICAL: Failure at exactly 128 indicates hard limit!")
            print("This is likely a buffer size constant in native code.")

    # Save detailed report
    report_filename = f"pykmn_debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write("PyKMN Debug Ladder Report\n")
        f.write("=" * 70 + "\n\n")

        for result in results:
            f.write(f"\nMode {result['mode']}: {result['description']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Status: {'PASSED' if result['passed'] else 'FAILED'}\n")
            f.write(f"Time: {result['elapsed_time']:.1f}s\n")
            if result['failure_batch']:
                f.write(f"Failed at batch_size: {result['failure_batch']}\n")
            f.write("\nOutput:\n")
            f.write(result['stdout'][-5000:] if len(result['stdout']) > 5000 else result['stdout'])
            if result['stderr']:
                f.write("\nErrors:\n")
                f.write(result['stderr'])
            f.write("\n" + "=" * 70 + "\n")

    print(f"\nDetailed report saved to: {report_filename}")

    # Exit with appropriate code
    sys.exit(0 if all(r['passed'] for r in results) else 1)


if __name__ == "__main__":
    main()