# PyKMN Memory Corruption Debugging Guide

This guide provides step-by-step instructions for debugging memory corruption issues in the PyKMN batched inference pipeline using native memory debugging tools.

## Background

The PyKMN integration experiences intermittent crashes with signatures like:
- `free(): invalid next size (fast)`
- `malloc_consolidate(): invalid chunk size`
- Segmentation faults in malloc/free

These errors almost always indicate **native memory corruption** (buffer overrun, double-free, use-after-free) in C/C++ extension code, NOT Python-level memory issues.

## Quick Start: Run With Allocator Hardening

The fastest way to debug is to run with allocator hardening flags that surface corruption earlier:

```bash
# Set environment variables
export PYTHONMALLOC=malloc
export MALLOC_CHECK_=3
export PYTHONFAULTHANDLER=1

# Run bisect harness
python test_corruption_bisect.py

# Or run your actual workload
python -m metamon.rl.finetune_from_hf ...
```

### What These Flags Do

- **PYTHONMALLOC=malloc**: Forces Python to use system malloc instead of pymalloc arenas, making corruption more visible
- **MALLOC_CHECK_=3**: Enables glibc heap consistency checks (abort on corruption)
- **PYTHONFAULTHANDLER=1**: Dumps Python stack trace on segfault

## Tool Comparison

| Tool | Speed | Rebuild Required | Detection Quality | Recommendation |
|------|-------|------------------|-------------------|----------------|
| **Allocator Flags** | Fast (no overhead) | No | Basic (crash location) | **START HERE** |
| **Valgrind** | 10-50x slower | No | Good (catches most bugs) | If ASAN unavailable |
| **AddressSanitizer (ASAN)** | 2-3x slower | **Yes** | Excellent (exact corruption) | **BEST for root cause** |

## Step 1: Bisect the Corruption Source

Run the debugging harness to identify which layer causes crashes:

```bash
# Run all tests (stops at first failure)
python test_corruption_bisect.py

# Run specific test
python test_corruption_bisect.py --test vectorized --batch-size 32

# Run with allocator hardening
PYTHONMALLOC=malloc MALLOC_CHECK_=3 python test_corruption_bisect.py
```

The harness tests 4 layers:
1. **test_pure_pykmn**: Raw C++ PyKMN (no metamon)
2. **test_feature_extraction**: PyKMN + feature extraction (`pykmn_to_features_raw`)
3. **test_observation_space**: PyKMN + features + observation space conversion
4. **test_vectorized_env**: Full batched integration (16 parallel envs)

**Interpretation**:
- Crash in **test_pure_pykmn** → Report to PyKMN upstream
- Crash in **test_feature_extraction** → Bug in `metamon/env/pykmn/features.py`
- Crash in **test_observation_space** → Bug in `metamon/interface.py` observation spaces
- Crash in **test_vectorized_env** → Bug in `metamon/env/pykmn/vector_env.py` or batching logic

## Step 2: Get Detailed Stack Trace (Valgrind)

If you can't rebuild Python/extensions with ASAN, use Valgrind:

```bash
# Run bisect under Valgrind (SLOW but no rebuild needed)
valgrind \
    --leak-check=full \
    --track-origins=yes \
    --show-leak-kinds=all \
    --log-file=valgrind_output.txt \
    python test_corruption_bisect.py --test vectorized --num-batches 3

# Check output
cat valgrind_output.txt | grep -A 20 "Invalid"
```

**What to look for**:
- `Invalid read of size X` → Use-after-free or buffer overrun
- `Invalid write of size X` → Writing to freed memory
- `Source and destination overlap` → memcpy bug

Valgrind will show the **C/C++ stack trace** of the corruption, pointing to the exact function.

## Step 3: Build with AddressSanitizer (BEST)

ASAN provides the most precise detection with minimal overhead. Requires rebuilding Python and/or PyKMN.

### Option A: ASAN for PyKMN Extension Only

If PyKMN is a Python extension module (pybind11/CPython API):

```bash
# Rebuild PyKMN with ASAN
cd /path/to/pykmn
export CFLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
export CXXFLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
export LDFLAGS="-fsanitize=address"

# Rebuild
python setup.py build --force
python setup.py install --force

# Run tests with ASAN
export ASAN_OPTIONS=detect_leaks=1:symbolize=1:detect_stack_use_after_return=1
python test_corruption_bisect.py
```

### Option B: ASAN for Entire Python (More Robust)

Rebuild Python itself with ASAN (requires ~30 minutes):

```bash
# Download Python source
cd /tmp
wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
tar -xzf Python-3.11.9.tgz
cd Python-3.11.9

# Build with ASAN
./configure \
    --with-pydebug \
    --with-address-sanitizer \
    --prefix=$HOME/python-asan
make -j$(nproc)
make install

# Use ASAN Python
export PATH=$HOME/python-asan/bin:$PATH
export ASAN_OPTIONS=detect_leaks=1:symbolize=1

# Reinstall metamon dependencies
pip install -e /path/to/metamon

# Run tests
python test_corruption_bisect.py
```

### Interpreting ASAN Output

ASAN will print detailed reports like:

```
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x7f8b4c000000
READ of size 8 at 0x7f8b4c000000 thread T0
    #0 0x7f8b4c3d2e89 in pykmn::Battle::active_pokemon_species() /path/to/pykmn.cpp:123
    #1 0x7f8b4c5d1234 in pykmn_to_features_raw /path/to/features.py:145
    #2 0x7f8b4c6a5678 in PyKMNVectorEnv._extract_observations /path/to/vector_env.py:310

freed by thread T0 here:
    #0 0x7f8b4c1a2b3c in free (/usr/lib/x86_64-linux-gnu/libasan.so.6+0xb3c)
    #1 0x7f8b4c3d1234 in pykmn::Battle::~Battle() /path/to/pykmn.cpp:456
    #2 0x7f8b4c5e2345 in PyKMNVectorEnv.reset /path/to/vector_env.py:142
```

**This tells you**:
- What: `heap-use-after-free` (reading freed memory)
- Where: `pykmn::Battle::active_pokemon_species()` at line 123
- When freed: `PyKMNVectorEnv.reset()` at line 142
- **ROOT CAUSE**: `active_pokemon_species()` is being called on a Battle object after it was destroyed

## Step 4: Fix the Root Cause

Based on ASAN/Valgrind output, apply the appropriate fix:

### Case 1: Use-After-Free in Battle Object

**Symptom**: Freed in `vector_env.reset()`, accessed in `pykmn_to_features_raw()`

**Fix**: Ensure feature extraction completes before clearing battle references:

```python
# BAD: Clears battles before features are fully extracted
self.battles = [None] * num_envs
features = pykmn_to_features_raw(self.battles[i], ...)  # Use-after-free!

# GOOD: Extract features FIRST, then clear
features = pykmn_to_features_raw(self.battles[i], ...)
self.battles[i] = None
```

### Case 2: Double-Free in Tokenizer

**Symptom**: Freed twice, both in `__del__` methods

**Fix**: Don't deepcopy objects containing C++ state:

```python
# BAD: deepcopy duplicates C++ handle, freed twice
new_obs_space = copy.deepcopy(obs_space)

# GOOD: Share tokenizer, only copy Python state
class TokenizedObservationSpace:
    def __deepcopy__(self, memo):
        # Share tokenizer (don't copy C++ state)
        return TokenizedObservationSpace(self.base_obs_space, self.tokenizer)
```

### Case 3: Buffer Overrun in Array Access

**Symptom**: `Invalid write of size X` beyond allocation

**Fix**: Add bounds checks:

```python
# BAD: May write past end of array
revealed[self.revealed_count] = opponent_name
self.revealed_count += 1

# GOOD: Check bounds
if self.revealed_count < len(revealed):
    revealed[self.revealed_count] = opponent_name
    self.revealed_count += 1
```

## Common Patterns and Solutions

### Pattern 1: Shared C++ Objects Across Envs

**Problem**: Single tokenizer/obs_space shared across 16 envs, race condition in C++ state

**Solution**: Use per-env Python state, shared C++ objects (already implemented in Phase 1 fixes)

### Pattern 2: Lazy Deletion

**Problem**: `self.battles[i] = None` doesn't immediately free C++ object (GC timing)

**Solution**: Explicit `del` or `gc.collect()` at safe points:

```python
# Clear battles
for i in range(num_envs):
    self.battles[i] = None
gc.collect()  # Force cleanup NOW (safe here, not during churn)
```

### Pattern 3: Nested C++ References

**Problem**: UniversalState holds references to C++ strings, outlives Battle object

**Solution**: Copy C++ data to Python immediately:

```python
# BAD: Holds C++ reference
species_name = battle.active_pokemon_species(player)  # Returns C++ string view

# GOOD: Copy to Python string immediately
species_name = str(battle.active_pokemon_species(player))
```

## Production Workarounds (While Debugging)

If you need to keep data generation running while investigating:

### Workaround 1: Subprocess Isolation

Isolate each batch in a subprocess (see Phase 4 implementation):

```bash
# Crashes are contained, don't affect parent process
python scripts/generate_selfplay_batched.py --subprocess-isolation
```

### Workaround 2: Reduce Batch Size

Smaller batches reduce crash probability:

```bash
# From batch_size=64 → batch_size=16
python scripts/generate_selfplay_batched.py --batch-size 16
```

### Workaround 3: Use DefaultObservationSpace

If ExpandedObservationSpace is the issue, use stateless alternative:

```bash
python -m metamon.rl.finetune_from_hf --observation-space DefaultObservationSpace ...
```

## Summary Checklist

- [ ] Run `test_corruption_bisect.py` to identify layer
- [ ] Run with `PYTHONMALLOC=malloc MALLOC_CHECK_=3`
- [ ] If still unclear, run Valgrind for stack trace
- [ ] If possible, rebuild with ASAN for precise detection
- [ ] Apply fix based on ASAN/Valgrind output
- [ ] Re-run bisect harness to verify fix
- [ ] Run stress test (1000+ battles) to confirm stability

## References

- [AddressSanitizer Documentation](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [Valgrind Manual](https://valgrind.org/docs/manual/manual.html)
- [Python Memory Management](https://docs.python.org/3/c-api/memory.html)
- Engineer's feedback document (context above)

## Getting Help

If you're stuck:

1. Share **full ASAN/Valgrind output** (not just the crash message)
2. Share **bisect harness results** (which layer crashes)
3. Share **minimal reproduction** (smallest code that crashes)
4. Check PyKMN GitHub issues for similar reports
