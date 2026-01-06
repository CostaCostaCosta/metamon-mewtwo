# Full Pipeline Integration Test

## Overview

The **Full Pipeline Integration Test** validates the complete end-to-end inference pipeline:

```
PyKMN Battles → SafeBattleManager → FastFeatureExtractor →
InferenceWrapper → RemotePolicyRunner → GPU Inference Server → Actions
```

This test suite ensures that all components work together correctly at scale, with proper type safety, memory management, and performance.

## Test Scenarios

### Test 1: Basic Functionality (16 battles × 50 steps)
- **Purpose**: Validate basic pipeline operation
- **Checks**:
  - Observations are correctly formatted (numeric arrays)
  - Legal action masks are properly generated
  - Actions are type-safe (int32/int64)
  - No illegal actions selected
  - Hidden state management works
  - RL2 state tracking works

### Test 2: Scale Test (256 battles × 100 steps)
- **Purpose**: Test moderate-scale parallelism
- **Checks**:
  - Pipeline scales to 256 parallel battles
  - Performance > 50 battles/sec
  - No memory leaks
  - Type conversion stability

### Test 3: Stress Test (1024 battles × 100 steps)
- **Purpose**: Validate maximum scale operation
- **Checks**:
  - 1024 parallel battles without crashes
  - No type conversion errors
  - No memory corruption
  - Performance > 100 battles/sec
  - GPU inference working correctly

### Test 4: Long Episode Test (64 battles to completion)
- **Purpose**: Ensure battles can run to natural completion
- **Checks**:
  - Battles finish normally (not timeout)
  - Hidden state management over long episodes
  - Memory stability over extended runs
  - Auto-reset works correctly

## Success Criteria

✅ All 1024 battles run without crashes
✅ No type conversion errors (int32/int64 compatibility)
✅ No memory corruption (teams are unique)
✅ No illegal actions selected
✅ Performance > 100 battles/sec end-to-end
✅ GPU inference working correctly
✅ Hidden state management working
✅ RL2 state tracking working

## Prerequisites

1. **Inference server must be running**:
   ```bash
   python -m metamon.inference.server --model Minikazam --batch_size 128 --port 8080
   ```

2. **Virtual environment activated**:
   ```bash
   source .venv/bin/activate
   ```

3. **METAMON_CACHE_DIR set**:
   ```bash
   export METAMON_CACHE_DIR=/home/eddie/metamon_cache
   ```

## Running Tests

### Option 1: Automated (Recommended)

The `run_integration_test.sh` script automatically:
- Starts the inference server
- Waits for health check
- Runs tests
- Cleans up server on exit

```bash
# Run all tests
cd /home/eddie/repos/metamon
./tests/run_integration_test.sh

# Run specific test
TEST=basic ./tests/run_integration_test.sh
TEST=scale_256 ./tests/run_integration_test.sh
TEST=stress_1024 ./tests/run_integration_test.sh
TEST=long_episodes ./tests/run_integration_test.sh

# Use different model
MODEL=SmallRL BATCH_SIZE=64 ./tests/run_integration_test.sh
```

### Option 2: Manual

If you prefer to manage the server manually:

```bash
# Terminal 1: Start inference server
python -m metamon.inference.server \
    --model Minikazam \
    --batch_size 128 \
    --port 8080

# Terminal 2: Run tests
python tests/test_full_pipeline.py --test all

# Or run specific test
python tests/test_full_pipeline.py --test basic
python tests/test_full_pipeline.py --test scale_256
python tests/test_full_pipeline.py --test stress_1024
python tests/test_full_pipeline.py --test long_episodes
```

## Validation Checks

The test performs extensive validation at each step:

### Observation Validation
- Dictionary contains 'numbers' key
- Arrays are numpy.ndarray type
- Batch size matches expected
- Dtype is float32 or float64
- No NaN or Inf values

### Legal Mask Validation
- Array is numpy.ndarray
- Shape is (batch_size, 13)
- Dtype is bool
- Each environment has at least one legal action

### Action Validation
- Array is numpy.ndarray
- Shape is (batch_size,)
- Dtype is int32 or int64
- Actions in valid range [0, 13)
- Actions respect legal masks

### Performance Validation
- Steps/sec throughput
- Battles/sec completion rate
- Average step time
- Average inference time
- Memory growth

## What Gets Tested

### Type Safety
The test validates the critical fix for type conversion:
```python
# Server expects int32/int64, not Python int
actions = policy.infer(obs, legal_masks)
assert actions.dtype in [np.int32, np.int64]
```

### Memory Safety
Verifies teams are properly cloned (not shared):
```python
# Each environment gets unique team instance
teams_p1 = [create_test_team() for _ in range(N)]
# Not: teams_p1 = [team] * N  # This shares objects!
```

### Hidden State Management
Tests that hidden states are properly reset for done episodes:
```python
if dones.any():
    policy.reset_hidden_state_for_dones(dones)
```

### RL2 State Tracking
Validates that RL2 state (previous action + reward) is tracked:
```python
policy.update_rewards(rewards)  # Must be called after each step
```

### Illegal Action Prevention
Ensures no illegal actions are selected:
```python
for i in range(batch_size):
    action = actions[i]
    assert legal_masks[i, action], f"Illegal action {action}"
```

## Expected Output

### Successful Run
```
======================================================================
FULL PIPELINE INTEGRATION TEST SUITE
======================================================================

Checking inference server health...
✓ Inference server healthy: {'status': 'healthy', 'model': 'Minikazam', ...}

======================================================================
TEST 1: Basic Functionality (16 battles × 50 steps)
======================================================================
✓ Created InferenceWrapper with 16 environments
✓ Created RemotePolicyRunners (P1 & P2)
✓ Reset complete, observations validated
  obs_p1['numbers'] shape: (16, 48), dtype: float32
  legal_p1 shape: (16, 13), dtype: bool

Running 50 inference steps...
  Step 1/50: 0 done, 800 steps/sec, 15.2ms inference
  Step 21/50: 2 done, 950 steps/sec, 12.8ms inference
  Step 41/50: 5 done, 1050 steps/sec, 11.5ms inference

======================================================================
TEST RESULTS: Basic Functionality
======================================================================
  Time: 1.23s
  Battles completed: 8
  Steps/sec: 650
  Battles/sec: 6.5
  Avg step time: 24.6ms
  Avg inference time: 13.2ms
  Memory: 1250.5 MB → 1252.3 MB (+1.8 MB)
  Illegal actions: 0
  Validation failures: 0

✓ PASSED: Basic functionality test

[... Tests 2, 3, 4 ...]

======================================================================
TEST SUMMARY
======================================================================
✓ PASS: basic
✓ PASS: scale_256
✓ PASS: stress_1024
✓ PASS: long_episodes

======================================================================
✓ ALL TESTS PASSED
Full pipeline is working correctly!

Key validations:
  ✓ No type conversion crashes
  ✓ No memory corruption
  ✓ No illegal actions
  ✓ GPU inference working
  ✓ Hidden state management working
  ✓ Performance targets met
======================================================================
```

### Failed Run
```
======================================================================
TEST 1: Basic Functionality (16 battles × 50 steps)
======================================================================
...
✗ P1: Action 5 is illegal for environment 3 (legal: [0, 1, 2, 6, 7])

======================================================================
TEST RESULTS: Basic Functionality
======================================================================
...
  Illegal actions: 12
  Validation failures: 3

Validation failures:
  - Step 15: P1 actions invalid
  - Step 23: P2 actions invalid
  - Step 31: obs_p1 validation failed

✗ FAILED: 12 illegal actions detected
```

## Troubleshooting

### Server Connection Failed
```
✗ ERROR: Inference server not available at http://localhost:8080

Please start the server with:
  python -m metamon.inference.server --model Minikazam --batch_size 128 --port 8080
```

**Solution**: Start the inference server in a separate terminal.

### Type Conversion Errors
```
TypeError: Cannot convert numpy.int64 to torch.long
```

**Solution**: This indicates the type fix is not working. Check that actions are converted to int32/int64:
```python
actions = actions.astype(np.int32)
```

### Memory Corruption
```
AssertionError: Clone 5 shares ID with original!
```

**Solution**: Teams are not being cloned properly. Use `clone_pokemon_team()`:
```python
teams_p1 = [clone_pokemon_team(team) for _ in range(N)]
```

### Illegal Actions
```
⚠ P1: Illegal action 5 selected for env 12 (legal: [0, 1, 2, 6, 7])
```

**Solution**: Check that legal masks are properly passed to inference:
```python
actions = policy.infer(obs, legal_masks)  # Must pass legal_masks!
```

### Performance Issues
```
⚠ WARNING: Performance 45.2 < 50 battles/sec
```

**Solution**:
1. Check GPU utilization: `nvidia-smi`
2. Increase server batch size: `--batch_size 256`
3. Reduce test batch size if GPU memory limited

### GPU Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**:
1. Reduce batch size: `--batch_size 64`
2. Use smaller model: `MODEL=Minikazam`
3. Check GPU memory: `nvidia-smi`

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Steps/sec | > 500 | Full pipeline throughput |
| Battles/sec | > 100 | Battle completion rate |
| Avg step time | < 50ms | End-to-end latency |
| Avg inference time | < 20ms | GPU inference only |
| Memory growth | < 1GB | For 1024 battles test |

## Files

- `test_full_pipeline.py` - Main test suite
- `run_integration_test.sh` - Automated test runner
- `README_INTEGRATION_TEST.md` - This documentation
- `test_safe_wrapper.py` - Component-level tests (no GPU)

## Next Steps

After passing all tests:

1. **Run benchmarks**: Use `benchmark_gpu_server.py` for detailed profiling
2. **Test with larger models**: Try SmallRL, SyntheticRLV2
3. **Stress test**: Run overnight with 1024 battles × 10000 steps
4. **Production deployment**: Integrate into self-play data collection

## See Also

- `/home/eddie/repos/metamon/GPU_SERVER_PERFORMANCE_REPORT.md` - Performance analysis
- `/home/eddie/repos/metamon/PERFORMANCE_STATUS.md` - Current status
- `/home/eddie/repos/metamon/metamon/inference/README.md` - Inference system docs
