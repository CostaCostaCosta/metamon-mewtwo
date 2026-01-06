#!/bin/bash
set -e

echo "Starting inference server..."
./start_inference_server.sh > /tmp/server_startup.log 2>&1 &
SERVER_PID=$!

echo "Waiting for server to start..."
sleep 25

echo "Checking server health..."
curl http://localhost:8080/health
echo ""

echo "Running throughput test..."
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
python test_throughput.py

echo "Killing server..."
kill $SERVER_PID 2>/dev/null || true
