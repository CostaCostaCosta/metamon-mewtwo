# Troubleshooting: Connection Refused Error

## The Problem

When you see this error:
```
RuntimeError: Cannot connect to inference server at http://localhost:8080
ConnectionRefusedError: [Errno 111] Connection refused
```

## Root Cause

The server is **not running**. There are several reasons this might happen:

### Reason 1: Virtual Environment Not Activated ⚠️ MOST COMMON

**Symptom**: Running `python -m metamon.inference.server` fails with import errors

**Fix**:
```bash
# You MUST activate the virtual environment first
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Then start the server
python -m metamon.inference.server --model SyntheticRLV2 --batch_size 128
```

**Why**: The system python (`/usr/bin/python3`) doesn't have the required packages installed. You need to use the virtual environment's python (`.venv/bin/python`).

---

### Reason 2: METAMON_CACHE_DIR Not Set

**Symptom**: Import errors or "No package metadata found"

**Fix**:
```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

**Why**: The server needs to know where to find pretrained models and teams.

---

### Reason 3: Server Crashed on Startup

**Symptom**: Server starts but immediately exits

**Debug**:
```bash
# Run server and look for errors
python -m metamon.inference.server --model SyntheticRLV2 --batch_size 128

# Common errors:
# - CUDA out of memory → reduce --batch_size
# - Model not found → check model name
# - Port already in use → change --port
```

---

### Reason 4: Port Already in Use

**Symptom**: "Address already in use"

**Fix**:
```bash
# Check what's using port 8080
lsof -i :8080

# Kill it or use a different port
python -m metamon.inference.server --port 8081
```

---

## The Correct Way to Start the Server

### Option 1: Use the Startup Script (Easiest)

```bash
# Terminal 1: Start server
./start_inference_server.sh

# Terminal 2: Run your client/tests
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
python test_inference_server.py
```

### Option 2: Manual Startup

```bash
# Terminal 1: Start server
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
python -m metamon.inference.server \
    --model SyntheticRLV2 \
    --batch_size 128 \
    --port 8080

# Wait for this message:
# "Inference server running on http://0.0.0.0:8080"

# Terminal 2: Run your client/tests
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
python test_inference_server.py
```

---

## Verification Checklist

Before running clients or tests, verify:

**1. Server is actually running**
```bash
# Check process
ps aux | grep "metamon.inference.server"

# Should show a python process running the server
```

**2. Server is listening on port 8080**
```bash
# Check port
lsof -i :8080

# Should show python listening on *:8080
```

**3. Server responds to health check**
```bash
curl http://localhost:8080/health

# Should return JSON:
# {"status": "healthy", "model": "SyntheticRLV2", ...}
```

**4. Virtual environment is activated in BOTH terminals**
```bash
# Your prompt should show (.venv)
# Example: (.venv) eddie@hostname:~/repos/metamon$

# If not, activate it:
source .venv/bin/activate
```

**5. METAMON_CACHE_DIR is set in BOTH terminals**
```bash
echo $METAMON_CACHE_DIR
# Should print: /home/eddie/metamon_cache

# If empty:
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

---

## Common Mistakes

### ❌ WRONG: Running without venv
```bash
python3 -m metamon.inference.server  # Will fail!
```

### ✅ CORRECT: Activate venv first
```bash
source .venv/bin/activate
python -m metamon.inference.server  # Will work!
```

---

### ❌ WRONG: Client runs before server is ready
```bash
# Terminal 1: Start server
python -m metamon.inference.server &  # Runs in background

# Terminal 2: Run client immediately
python test_inference_server.py  # FAILS - server not ready yet
```

### ✅ CORRECT: Wait for server to finish loading
```bash
# Terminal 1: Start server
python -m metamon.inference.server
# Wait for: "Inference server running on http://0.0.0.0:8080"

# Terminal 2: NOW run client
python test_inference_server.py  # Works!
```

---

### ❌ WRONG: Forgetting cache dir
```bash
python -m metamon.inference.server  # Might fail or behave incorrectly
```

### ✅ CORRECT: Always set cache dir
```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
python -m metamon.inference.server  # Will work!
```

---

## Quick Debug Script

Run this to diagnose issues:

```bash
#!/bin/bash
echo "=== Inference Server Diagnostics ==="

echo -e "\n1. Checking virtual environment..."
if [ -d ".venv" ]; then
    echo "✓ .venv exists"
    if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
        echo "✓ Virtual environment is activated"
    else
        echo "✗ Virtual environment NOT activated"
        echo "  Fix: source .venv/bin/activate"
    fi
else
    echo "✗ .venv not found"
fi

echo -e "\n2. Checking METAMON_CACHE_DIR..."
if [ -n "$METAMON_CACHE_DIR" ]; then
    echo "✓ METAMON_CACHE_DIR is set: $METAMON_CACHE_DIR"
else
    echo "✗ METAMON_CACHE_DIR not set"
    echo "  Fix: export METAMON_CACHE_DIR=/home/eddie/metamon_cache"
fi

echo -e "\n3. Checking if server is running..."
if pgrep -f "metamon.inference.server" > /dev/null; then
    echo "✓ Server process is running"
else
    echo "✗ Server is NOT running"
    echo "  Fix: ./start_inference_server.sh"
fi

echo -e "\n4. Checking port 8080..."
if lsof -i :8080 > /dev/null 2>&1; then
    echo "✓ Port 8080 is in use"
    lsof -i :8080
else
    echo "✗ Port 8080 is not in use"
fi

echo -e "\n5. Testing server health..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✓ Server responds to health check"
    curl -s http://localhost:8080/health | python3 -m json.tool
else
    echo "✗ Server does not respond"
fi

echo -e "\n=== End Diagnostics ==="
```

Save this as `diagnose_server.sh`, make it executable with `chmod +x diagnose_server.sh`, and run it.

---

## Still Having Issues?

If you've followed all steps and it still doesn't work:

1. **Check the server terminal for errors**
   - Look for CUDA errors
   - Look for model loading errors
   - Look for port binding errors

2. **Try a different port**
   ```bash
   python -m metamon.inference.server --port 8081
   # And update client to use http://localhost:8081
   ```

3. **Check GPU availability**
   ```bash
   nvidia-smi  # Should show your GPU
   ```

4. **Verify model is downloaded**
   ```bash
   ls ~/metamon_cache/pretrained_models/models--jakegrigsby--metamon/
   ```

5. **Try CPU mode (slow but works without GPU)**
   ```bash
   python -m metamon.inference.server --device cpu
   ```

---

## Summary

The error `Connection refused [Errno 111]` means the server isn't running. The most common reason is **forgetting to activate the virtual environment** before starting the server.

**Quick fix**:
```bash
# Terminal 1
./start_inference_server.sh

# Terminal 2 (wait for server to fully load first!)
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
python test_inference_server.py
```
