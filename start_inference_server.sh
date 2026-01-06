#!/bin/bash
# Start the GPU inference server
# Usage: ./start_inference_server.sh [MODEL] [BATCH_SIZE] [PORT]

# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Parse arguments with defaults
MODEL=${1:-SyntheticRLV2}
BATCH_SIZE=${2:-128}
PORT=${3:-8080}

echo "Starting inference server..."
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Port: $PORT"
echo ""

# Start server
python -m metamon.inference.server \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --port "$PORT"
