#!/usr/bin/env python3
"""Simple test to verify the batched inference system works."""

import sys
import time
import subprocess
from pathlib import Path

print("=" * 80)
print("SIMPLE BATCHED INFERENCE TEST")
print("=" * 80)

# Step 1: Start policy server
print("\n[Step 1] Starting policy server...")
server_cmd = [
    sys.executable,
    "-m",
    "metamon.rl.policy_server",
    "--model", "SyntheticRLV2",
    "--port", "5555",
    "--batch-size", "8",
]

print(f"Command: {' '.join(server_cmd)}")
server_proc = subprocess.Popen(
    server_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1,
)

print("Waiting for server to initialize (this may take 30-60 seconds to load model)...")
ready = False
for i in range(120):
    line = server_proc.stdout.readline()
    if line:
        print(f"  [Server] {line.rstrip()}")
        if "Ready to serve" in line:
            ready = True
            break
    if server_proc.poll() is not None:
        print("ERROR: Server process died!")
        sys.exit(1)
    time.sleep(1)

if not ready:
    print("WARNING: Didn't see 'Ready to serve', but proceeding...")

print("✓ Server appears to be running")

# Step 2: Test with a single worker
print("\n[Step 2] Starting single worker for 2 battles...")

worker_cmd = [
    sys.executable,
    "-m",
    "metamon.rl.evaluate_gpu",
    "--model", "SyntheticRLV2",
    "--server", "tcp://localhost:5555",
    "--eval-type", "ladder",
    "--username", "TestWorker1",
    "--battle-format", "gen1ou",
    "--team-set", "competitive",
    "--total-battles", "2",
    "--save-trajectories-to", "/tmp/test_traj",
]

print(f"Command: {' '.join(worker_cmd)}")

try:
    result = subprocess.run(
        worker_cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    print("\n[Worker Output]")
    print(result.stdout)

    if result.returncode == 0:
        print("\n✓ Worker completed successfully!")
    else:
        print(f"\n✗ Worker failed with exit code {result.returncode}")
        print("[Worker Error Output]")
        print(result.stderr)

finally:
    # Cleanup
    print("\n[Cleanup] Shutting down server...")
    server_proc.terminate()
    server_proc.wait(timeout=10)
    print("✓ Test complete!")
