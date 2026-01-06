#!/usr/bin/env python3
import subprocess
import time
import os

os.environ["METAMON_CACHE_DIR"] = str(os.path.expanduser("~/metamon_cache"))

# Kill existing servers
subprocess.run(["pkill", "-f", "metamon.inference.server"], check=False)
time.sleep(2)

# Start server
print("Starting server...")
server = subprocess.Popen(
    [".venv/bin/python", "-u", "-m", "metamon.inference.server",
     "--model", "SyntheticRLV2", "--batch_size", "128", "--port", "8080"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)

time.sleep(20)

# Run test
print("\nRunning batch shape test...")
result = subprocess.run(
    [".venv/bin/python", "test_batch_shape.py"],
    capture_output=True, text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Kill server and get output
time.sleep(2)
server.terminate()
output, _ = server.communicate(timeout=5)

# Show relevant server debug output
for line in output.split('\n'):
    if 'DEBUG' in line or 'shape' in line.lower():
        print("SERVER:", line)
