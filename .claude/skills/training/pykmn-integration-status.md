# PyKMN Integration Status

**Category**: Training Workflows
**Status**: ⚠️ **UNSTABLE** - Not ready for production use
**Last Updated**: 2026-01-05
**Owner**: Research Team

---

## Executive Summary

libpykmn has been integrated as a faster alternative to Pokémon Showdown for battle simulation. While the library itself is stable (validated: 4000+ battles at 5000+ battles/sec with zero crashes), **our integration is not production-ready**.

**Current Status:**
- ✅ libpykmn C library: Stable
- ✅ Basic integration: Works for small-scale testing
- ❌ Batched inference: Unstable at any batch size > 1
- ❌ Production reliability: Poor (~1-5 battles/sec with frequent instability)

**Bottom Line:** Do not use for production training runs. Showdown backend remains the stable option.

---

## What Works ✅

### 1. Small-Scale Testing (batch_size=1)

Single-environment simulation with pretrained models:

```bash
python scripts/generate_selfplay_batched.py \
    --model Kakuna \
    --num_battles 100 \
    --batch_size 1 \
    --device cuda \
    --format gen1ou \
    --save_dir ~/test_output
```

**Performance:** ~1-2 battles/sec
**Reliability:** Mostly stable for <100 battles

### 2. libpykmn Library Validation

Pure libpykmn benchmarks work flawlessly:

```bash
cd ~/repos/PyKMN
uv run python examples/pkmn_benchmark.py 1000 42
# Result: 4000 battles at ~5000 battles/sec, zero crashes
```

This proves the C++ library is stable. Issues are in our Python integration.

---

## What Doesn't Work ❌

### 1. Batched Inference (batch_size > 1)

**Problem:** Crashes with heap corruption at any meaningful batch size

**Evidence:**
```bash
# batch_size=16
python scripts/generate_selfplay_batched.py \
    --model Kakuna \
    --batch_size 16 \
    --num_battles 1000

# Result: "free(): invalid next size (fast)" after 80-256 battles
```

**Root Causes (Multiple):**
- Observation space state leaking between environments
- Tensor memory aliasing during CPU↔GPU transfers
- Unknown heap corruption patterns at ~128 battle mark
- Transformer KV cache incompatibility with micro-batching

### 2. High Throughput

**Problem:** Even when stable, throughput is poor

**Performance:**
- Expected (based on libpykmn benchmarks): 100-1000+ battles/sec
- Actual (our integration): 1-5 battles/sec
- Gap: 20-200x slower than theoretical maximum

**Likely Bottlenecks:**
- Excessive Python↔C++ boundary crossings
- Inefficient observation space construction
- CPU↔GPU transfer overhead
- Sequential processing instead of vectorization

### 3. Parallel Execution

**Problem:** Multiple workers crash more frequently

```bash
python scripts/generate_selfplay_subprocess.py \
    --num_workers 4 \
    --batch_size 16

# Result: 38% failure rate, CUDA context conflicts
```

---

## Technical Details for Senior Engineers

### Integration Architecture

```
PyKMN (C++) ←→ Python Wrapper ←→ metamon Observation Space ←→ AMAGO Model (GPU)
  [Stable]        [Unstable]           [Buggy]                    [Stable]
```

The instability is in the **Python integration layer**, not the underlying libraries.

### Known Issues

**Issue 1: Shared Mutable State**
- Single `ObservationSpace` instance shared across all vectorized environments
- `revealed_opponents` set grows unbounded across battles
- Observations leak between environments (correctness bug)

**Attempted Fix:** State-explicit observation protocol (pass per-env state dicts)
**Status:** Implemented but not fully validated

**Issue 2: Tensor Lifetime Problems**
- `torch.from_numpy()` creates views, not copies
- Numpy arrays can be freed/mutated during async GPU transfers
- Causes segfaults and heap corruption

**Attempted Fix:** Clone barriers (`.clone()` + synchronous transfers)
**Status:** Reduces crashes but doesn't eliminate them

**Issue 3: "128 Battle Barrier"**
- Consistent crash pattern at exactly 128 battles
- Affects all batch sizes (not memory pressure related)
- Error: "free(): invalid next size"

**Root Cause:** Unknown. Possibly:
- Hardcoded buffer limit in trajectory serialization
- File descriptor leak in libpykmn Python bindings
- CUDA event pool corruption

**Status:** Under investigation

### What We've Tried

✅ **State-explicit observation protocol** - Reduced contamination, didn't fix crashes
✅ **Clone barriers** - Reduced segfaults, didn't fix heap corruption
✅ **CUDA context locking** - Fixed parallel loading issues, didn't fix crashes
✅ **Defensive error handling** - Improved diagnostics, didn't fix root cause
❌ **Micro-batched inference** - Incompatible with transformer KV caches
❌ **Per-environment deepcopy** - Segfaults from C++ tokenizer duplication
❌ **Higher batch sizes** - Makes crashes worse, not better

---

## Current Best Configuration

If you must use PyKMN despite instability:

```bash
# Single-process, small batch, fail-fast
python scripts/generate_selfplay_batched.py \
    --model Kakuna \
    --num_battles 500 \
    --batch_size 16 \
    --device cuda \
    --format gen1ou \
    --team_set smogon_pass2 \
    --save_dir ~/trajectories/pykmn_test
```

**Expected:**
- ~3-5 battles/sec
- May crash after 80-200 battles
- Save incrementally (every 100 battles) to minimize data loss

**NOT RECOMMENDED FOR PRODUCTION**

---

## Recommended Path Forward

### For Production Training: Use Showdown

```bash
python scripts/generate_selfplay_data.py \
    --team_dir ~/metamon_cache/teams/modern_replays_v2 \
    --num_battles 10000 \
    --num_envs 16 \
    --format gen1ou \
    --save_dir ~/trajectories/production
```

**Performance:** 1-2 battles/sec (similar to PyKMN)
**Reliability:** ✅ Proven stable for millions of battles
**Bottom Line:** Until PyKMN integration is fixed, Showdown is the pragmatic choice

### For Research: Focus Investigation

If a senior engineer wants to fix this, priority order:

1. **Profile the "128 battle barrier"** - Use ASAN/Valgrind to find heap corruption source
2. **Optimize Python boundary crossings** - Current implementation is likely doing way too many small calls
3. **Validate observation space state isolation** - Verify per-env state protocol is working correctly
4. **Test alternative tensor transfer patterns** - Pinned memory, async with explicit barriers

**Estimated Effort:** 1-2 weeks of focused debugging

---

## Files & Documentation

**Integration Code:**
- `metamon/env/pykmn/` - Full pykmn integration module
- `scripts/generate_selfplay_batched.py` - Main data generation script
- `scripts/generate_selfplay_subprocess.py` - Crash-resistant wrapper

**Debugging Tools:**
- `metamon/env/pykmn/diagnostics.py` - Validation utilities
- `test_corruption_bisect.py` - Bisection harness
- `debug_memory.sh` - ASAN/Valgrind helper

**Documentation:**
- `PYKMN_BATCHED_INFERENCE_INVESTIGATION.md` - Detailed technical investigation
- `PYKMN_BATCHED_INFERENCE_FIX_SUMMARY.md` - Summary of attempted fixes
- Look directly at https://github.com/pkmn/engine for libpykmn reference

---

## Key Takeaways

1. **libpykmn itself is fast and stable** - The problem is our integration
2. **We're stuck at ~5 battles/sec** - 100-200x slower than theoretical maximum
3. **Batched inference is unstable** - Crashes are unpredictable and hard to debug
4. **Use Showdown for production** - Similar performance, proven reliability
5. **This needs focused debugging** - Not a quick fix, requires deep systems knowledge

---

**Status:** Not production-ready. Do not use for critical training runs.

**Recommendation:** Stick with Showdown backend until a senior engineer can dedicate time to proper investigation.
