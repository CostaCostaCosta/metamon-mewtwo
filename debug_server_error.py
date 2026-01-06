#!/usr/bin/env python3
"""Debug script to capture server errors."""

import os
import subprocess
import time
import sys
from pathlib import Path

# Set environment
os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

# Kill any existing servers
subprocess.run(["pkill", "-f", "metamon.inference.server"], check=False)
time.sleep(2)

# Start server
print("Starting server...")
server_proc = subprocess.Popen(
    [
        ".venv/bin/python", "-u",  # -u for unbuffered output
        "-m", "metamon.inference.server",
        "--model", "SyntheticRLV2",
        "--batch_size", "4",
        "--port", "8080"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Wait for server to start
print("Waiting for server to start...")
time.sleep(20)

# Run test
print("\n=== RUNNING TEST ===\n")
test_proc = subprocess.run(
    [".venv/bin/python", "test_server_simple.py"],
    capture_output=True,
    text=True
)

print(test_proc.stdout)
if test_proc.stderr:
    print("STDERR:", test_proc.stderr)

# Give server time to print errors
time.sleep(2)

# Kill server and get its output
print("\n=== SERVER OUTPUT ===\n")
server_proc.terminate()
try:
    output, _ = server_proc.communicate(timeout=5)
    print(output)
except subprocess.TimeoutExpired:
    server_proc.kill()
    output, _ = server_proc.communicate()
    print(output)
