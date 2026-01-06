#!/bin/bash
#
# Hardened Allocator Diagnostic Wrapper
#
# Runs self-play with hardened memory allocators and crash diagnostics
# to pinpoint exact location of heap corruption.
#
# Usage:
#   ./scripts/debug_memory.sh [batch_size] [num_battles]
#
# Example:
#   ./scripts/debug_memory.sh 80 160  # Test at threshold
#

set -euo pipefail

# Configuration
BATCH_SIZE="${1:-80}"
NUM_BATTLES="${2:-${BATCH_SIZE}}"  # Default: 1 batch
MODEL="${3:-Kakuna}"
FORMAT="${4:-gen1ou}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "HARDENED ALLOCATOR DIAGNOSTIC TEST"
echo "========================================================================"
echo "Batch size: ${BATCH_SIZE}"
echo "Num battles: ${NUM_BATTLES}"
echo "Model: ${MODEL}"
echo "Format: ${FORMAT}"
echo "========================================================================"
echo ""

# Activate environment
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Enable core dumps
echo -e "${YELLOW}Enabling core dumps...${NC}"
ulimit -c unlimited
echo "✓ Core dumps enabled (ulimit -c = $(ulimit -c))"
echo ""

# Set up hardened allocator environment
echo -e "${YELLOW}Configuring hardened allocators...${NC}"
export MALLOC_CHECK_=3              # Abort immediately on heap corruption
export MALLOC_PERTURB_=165          # Fill freed memory with 0xa5 pattern
export PYTHONFAULTHANDLER=1         # Python-level fault handler
export CUDA_LAUNCH_BLOCKING=1       # Synchronous CUDA (exact error location)
export TORCH_SHOW_CPP_STACKTRACES=1 # Show C++ stack traces

echo "✓ MALLOC_CHECK_=3 (abort on corruption)"
echo "✓ MALLOC_PERTURB_=165 (detect use-after-free)"
echo "✓ PYTHONFAULTHANDLER=1 (crash traces)"
echo "✓ CUDA_LAUNCH_BLOCKING=1 (sync CUDA)"
echo "✓ TORCH_SHOW_CPP_STACKTRACES=1 (C++ traces)"
echo ""

# Output directory
OUTPUT_DIR="${HOME}/metamon/trajectories/diagnostic_malloc_b${BATCH_SIZE}"
echo -e "${YELLOW}Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# Run test
echo "========================================================================"
echo "STARTING DIAGNOSTIC TEST"
echo "========================================================================"
echo ""

set +e  # Don't exit on error, we want to capture exit code

timeout 600 python scripts/generate_selfplay_batched.py \
    --model "${MODEL}" \
    --num_battles "${NUM_BATTLES}" \
    --batch_size "${BATCH_SIZE}" \
    --device cuda \
    --format "${FORMAT}" \
    --team_set smogon_pass2 \
    --save_dir "${OUTPUT_DIR}" \
    --enable_diagnostics

EXIT_CODE=$?
set -e

echo ""
echo "========================================================================"
echo "TEST COMPLETED"
echo "========================================================================"
echo "Exit code: ${EXIT_CODE}"

# Interpret exit code
if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✓ TEST PASSED - No crashes detected${NC}"
    echo ""
    echo "Battles completed successfully at batch_size=${BATCH_SIZE}"

elif [ ${EXIT_CODE} -eq 124 ]; then
    echo -e "${YELLOW}⏱  TEST TIMEOUT (600s limit)${NC}"
    echo ""
    echo "Test exceeded time limit. May indicate hang or very slow performance."

elif [ ${EXIT_CODE} -eq 134 ]; then
    echo -e "${RED}❌ TEST FAILED - SIGABRT (heap corruption detected by allocator)${NC}"
    echo ""
    echo "The hardened allocator caught heap corruption and aborted."
    echo "Check the stack trace above for the exact location."

elif [ ${EXIT_CODE} -eq 139 ]; then
    echo -e "${RED}❌ TEST FAILED - SIGSEGV (segmentation fault)${NC}"
    echo ""
    echo "Direct memory access violation (null pointer, out-of-bounds, etc.)"

else
    echo -e "${RED}❌ TEST FAILED - Exit code ${EXIT_CODE}${NC}"
fi

# Count trajectories saved
echo ""
echo "Checking saved trajectories..."
TRAJ_COUNT=$(find "${OUTPUT_DIR}" -name "*.json.lz4" 2>/dev/null | wc -l || echo "0")
echo "Trajectories saved: ${TRAJ_COUNT}"

# Check for core dump
if [ -f core ]; then
    echo ""
    echo -e "${YELLOW}Core dump generated: core${NC}"
    echo "Analyze with: gdb python core"
fi

echo ""
echo "========================================================================"

exit ${EXIT_CODE}
