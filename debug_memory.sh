#!/bin/bash
# Helper script for PyKMN memory corruption debugging
#
# Usage:
#   ./debug_memory.sh bisect              # Run bisect harness with hardening
#   ./debug_memory.sh bisect --test obs_space  # Run specific test
#   ./debug_memory.sh valgrind            # Run bisect under Valgrind
#   ./debug_memory.sh command <args>      # Run arbitrary command with hardening
#
# Examples:
#   ./debug_memory.sh bisect --batch-size 32
#   ./debug_memory.sh command python -m metamon.rl.finetune_from_hf ...

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Export allocator hardening flags
export PYTHONMALLOC=malloc
export MALLOC_CHECK_=3
export PYTHONFAULTHANDLER=1

case "$1" in
    bisect)
        print_header "Running Memory Corruption Bisect Harness"
        print_info "Allocator hardening flags:"
        print_info "  PYTHONMALLOC=malloc"
        print_info "  MALLOC_CHECK_=3"
        print_info "  PYTHONFAULTHANDLER=1"
        echo ""

        shift  # Remove 'bisect' argument
        exec python test_corruption_bisect.py "$@"
        ;;

    valgrind)
        print_header "Running Under Valgrind"
        print_warning "This will be VERY SLOW (10-50x slowdown)"
        echo ""

        VALGRIND_LOG="valgrind_$(date +%Y%m%d_%H%M%S).log"
        print_info "Output will be saved to: $VALGRIND_LOG"
        echo ""

        shift  # Remove 'valgrind' argument
        ARGS="${@:-test_corruption_bisect.py --test vectorized --num-batches 2}"

        valgrind \
            --leak-check=full \
            --track-origins=yes \
            --show-leak-kinds=all \
            --log-file="$VALGRIND_LOG" \
            --verbose \
            python $ARGS

        echo ""
        print_info "Valgrind output saved to: $VALGRIND_LOG"
        print_info "Checking for errors..."

        if grep -q "Invalid" "$VALGRIND_LOG"; then
            print_error "Memory errors detected! See $VALGRIND_LOG for details"
            echo ""
            echo "Summary of errors:"
            grep -A 5 "Invalid" "$VALGRIND_LOG" | head -50
            exit 1
        else
            print_info "No memory errors detected"
        fi
        ;;

    asan)
        print_header "Running with AddressSanitizer (ASAN)"
        print_warning "Requires Python/PyKMN built with ASAN support"
        print_info "See PYKMN_MEMORY_DEBUGGING_GUIDE.md for build instructions"
        echo ""

        export ASAN_OPTIONS="detect_leaks=1:symbolize=1:detect_stack_use_after_return=1:log_path=asan_$(date +%Y%m%d_%H%M%S).log"
        print_info "ASAN_OPTIONS=$ASAN_OPTIONS"
        echo ""

        shift  # Remove 'asan' argument
        exec python "$@"
        ;;

    command)
        print_header "Running Custom Command with Memory Hardening"
        print_info "Allocator hardening flags enabled"
        echo ""

        shift  # Remove 'command' argument
        exec "$@"
        ;;

    *)
        print_header "PyKMN Memory Corruption Debugging Helper"
        echo ""
        echo "Usage:"
        echo "  $0 bisect [args]         Run bisect harness with allocator hardening"
        echo "  $0 valgrind [args]       Run under Valgrind (slow but detailed)"
        echo "  $0 asan [args]           Run with ASAN (requires ASAN-enabled Python)"
        echo "  $0 command <cmd> [args]  Run arbitrary command with hardening"
        echo ""
        echo "Examples:"
        echo "  $0 bisect"
        echo "  $0 bisect --test vectorized --batch-size 32"
        echo "  $0 valgrind test_corruption_bisect.py --test obs_space"
        echo "  $0 command python -m metamon.rl.finetune_from_hf ..."
        echo ""
        echo "For more details, see: PYKMN_MEMORY_DEBUGGING_GUIDE.md"
        exit 1
        ;;
esac
