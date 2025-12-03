#!/usr/bin/env python3
"""
Test AMAGO integration with remote policy server.

This script tests that:
1. RemoteAMAGOAgent properly wraps a real AMAGO agent
2. Timestep preprocessing works correctly
3. evaluate_test() method is available and functional
"""

import sys
import subprocess
import time
from pathlib import Path

print("=" * 80)
print("AMAGO INTEGRATION TEST")
print("=" * 80)

# Step 1: Start policy server
print("\n[Step 1] Starting policy server...")
server_cmd = [
    sys.executable,
    "-u",
    "-m",
    "metamon.rl.policy_server",
    "--model", "SyntheticRLV2",
    "--port", "5555",
    "--batch-size", "8",
    "--timeout-ms", "50",
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

# Step 2: Test with a single battle using GPU evaluation
print("\n[Step 2] Testing single battle with AMAGO's evaluate_test()...")

worker_cmd = [
    sys.executable,
    "-c",
    """
import sys
sys.path.insert(0, '/home/eddie/repos/metamon')

from metamon.rl.evaluate_gpu import pretrained_vs_local_ladder_gpu
from metamon import env as metamon_env

# Load team set
team_set = metamon_env.get_metamon_teams(
    'gen1ou', 'competitive', set_type=metamon_env.TeamSet
)

# Run a single battle
try:
    results = pretrained_vs_local_ladder_gpu(
        pretrained_model_name='SyntheticRLV2',
        server_address='tcp://localhost:5555',
        username='TestWorker_AMAGO',
        battle_format='gen1ou',
        team_set=team_set,
        total_battles=1,
        avatar='red-gen1main',
        battle_backend='poke-env',
        save_trajectories_to=None,
        save_team_results_to=None,
    )
    print(f"\\n✓ Battle completed successfully!")
    print(f"  Results: {results}")
except Exception as e:
    print(f"\\n✗ Battle failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
""",
]

print(f"Running worker...")

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
        print("✓ AMAGO integration test PASSED!")
    else:
        print(f"\n✗ Worker failed with exit code {result.returncode}")
        print("[Worker Error Output]")
        print(result.stderr)
        sys.exit(1)

finally:
    # Cleanup
    print("\n[Cleanup] Shutting down server...")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait()
    print("✓ Test complete!")
