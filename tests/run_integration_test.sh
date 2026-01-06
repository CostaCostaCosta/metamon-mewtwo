#!/bin/bash
#
# Run full pipeline integration tests.
#
# This script:
# 1. Starts the inference server in the background
# 2. Waits for it to become healthy
# 3. Runs the integration tests
# 4. Cleans up the server on exit
#

set -e

# Configuration
MODEL=${MODEL:-"Minikazam"}
BATCH_SIZE=${BATCH_SIZE:-128}
PORT=${PORT:-8080}
SERVER_URL="http://localhost:${PORT}"
TEST=${TEST:-"all"}

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "FULL PIPELINE INTEGRATION TEST RUNNER"
echo "========================================================================"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Port: $PORT"
echo "Test: $TEST"
echo ""

# Cleanup function
cleanup() {
    if [ ! -z "$SERVER_PID" ]; then
        echo ""
        echo "Cleaning up inference server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "✓ Server stopped"
    fi
}

trap cleanup EXIT INT TERM

# Start inference server
echo "Starting inference server..."
python -m metamon.inference.server \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --port $PORT \
    --device cuda \
    > /tmp/inference_server.log 2>&1 &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server to become healthy
echo "Waiting for server to become healthy..."
MAX_RETRIES=30
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is healthy${NC}"
        break
    fi

    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${RED}✗ Server process died during startup${NC}"
        echo ""
        echo "Server logs:"
        cat /tmp/inference_server.log
        exit 1
    fi

    sleep 1
    RETRY=$((RETRY + 1))
    echo "  Attempt $RETRY/$MAX_RETRIES..."
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo -e "${RED}✗ Server failed to become healthy after $MAX_RETRIES attempts${NC}"
    echo ""
    echo "Server logs:"
    cat /tmp/inference_server.log
    exit 1
fi

# Run tests
echo ""
echo "========================================================================"
echo "RUNNING INTEGRATION TESTS"
echo "========================================================================"
echo ""

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the test
python tests/test_full_pipeline.py --server_url "$SERVER_URL" --test "$TEST"
TEST_EXIT_CODE=$?

# Print server logs if tests failed
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================================================"
    echo "INFERENCE SERVER LOGS (last 50 lines)"
    echo "========================================================================"
    tail -n 50 /tmp/inference_server.log
fi

echo ""
echo "========================================================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
else
    echo -e "${RED}✗ TESTS FAILED${NC}"
fi
echo "========================================================================"

exit $TEST_EXIT_CODE
