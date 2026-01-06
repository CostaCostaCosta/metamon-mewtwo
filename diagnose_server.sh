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
    if [ -d "$METAMON_CACHE_DIR" ]; then
        echo "✓ Cache directory exists"
    else
        echo "✗ Cache directory does not exist: $METAMON_CACHE_DIR"
    fi
else
    echo "✗ METAMON_CACHE_DIR not set"
    echo "  Fix: export METAMON_CACHE_DIR=/home/eddie/metamon_cache"
fi

echo -e "\n3. Checking if server is running..."
if pgrep -f "metamon.inference.server" > /dev/null; then
    echo "✓ Server process is running"
    pgrep -af "metamon.inference.server"
else
    echo "✗ Server is NOT running"
    echo "  Fix: ./start_inference_server.sh"
fi

echo -e "\n4. Checking port 8080..."
if command -v lsof &> /dev/null; then
    if lsof -i :8080 > /dev/null 2>&1; then
        echo "✓ Port 8080 is in use"
        lsof -i :8080
    else
        echo "✗ Port 8080 is not in use (server not listening)"
    fi
else
    echo "⚠ lsof not available, skipping port check"
fi

echo -e "\n5. Testing server health..."
if command -v curl &> /dev/null; then
    if curl -s -m 2 http://localhost:8080/health > /dev/null 2>&1; then
        echo "✓ Server responds to health check"
        curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8080/health
    else
        echo "✗ Server does not respond to health check"
        echo "  Make sure server is running and fully loaded"
    fi
else
    echo "⚠ curl not available, skipping health check"
fi

echo -e "\n6. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        echo "✓ GPU is available"
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    else
        echo "✗ GPU not accessible"
    fi
else
    echo "⚠ nvidia-smi not available (CPU-only mode?)"
fi

echo -e "\n7. Checking for pretrained models..."
if [ -n "$METAMON_CACHE_DIR" ]; then
    MODEL_DIR="$METAMON_CACHE_DIR/pretrained_models/models--jakegrigsby--metamon"
    if [ -d "$MODEL_DIR" ]; then
        echo "✓ Pretrained models found"
        ls -la "$MODEL_DIR/snapshots" 2>/dev/null | head -5
    else
        echo "✗ Pretrained models not found at: $MODEL_DIR"
        echo "  Models will be downloaded on first run"
    fi
fi

echo -e "\n=== End Diagnostics ==="
echo -e "\nSummary:"
echo "If server is NOT running, start it with:"
echo "  ./start_inference_server.sh"
echo ""
echo "If venv is NOT activated, activate it with:"
echo "  source .venv/bin/activate"
echo ""
echo "If METAMON_CACHE_DIR is NOT set, set it with:"
echo "  export METAMON_CACHE_DIR=/home/eddie/metamon_cache"
