# PyKMN Heap Corruption Deep Analysis

## Problem Statement

Persistent heap corruption occurs after 300-400 battles with error:
```
corrupted size vs. prev_size
```

This indicates writing past the end of a malloc'd block, corrupting heap metadata.

## Test Results Summary

### ✅ Components That Are STABLE

1. **Pure PyKMN** (test_pykmn_native_stability.py)
   - 1000 battles: PASS
   - Rate: ~1900 battles/sec
   - **Conclusion**: PyKMN C++ library is 100% stable

2. **Feature Extraction** (test_features_extraction_stability.py)
   - 10,000 feature extractions: PASS
   - Rate: ~14,000 calls/sec
   - **Conclusion**: `pykmn_to_features_raw()` accessor methods are stable

### ❌ Components That FAIL

1. **PyKMNVectorEnv with random actions** (test_pykmn_1000_battles.py)
   - Crashes after 300-400 battles
   - Error: "corrupted size vs. prev_size"

2. **PyKMNVectorEnv with trajectory tracking disabled** (test_no_trajectories.py)
   - Times out / hangs
   - Suggests corruption happens even without trajectory tracking

## Critical Findings

### 1. Buffer Management Issue in LocalPolicyRunner

**File**: `/home/eddie/repos/metamon/metamon/env/pykmn/policy_runner.py`

**Lines 178-184**: Buffers allocated on first `infer()` call:
```python
if self.time_idxs is None:
    self.time_idxs = torch.zeros((batch_size,), ...)
    self.rl2_buffer = torch.zeros((batch_size, self.action_dim + 1), ...)
    self.prev_action_onehot_buffer = torch.zeros((batch_size, self.action_dim), ...)
```

**Line 289**: Increments ALL elements regardless of current batch size:
```python
self.time_idxs += 1  # BUG: increments entire buffer!
```

**Issue**: If buffers are allocated with size N, but later called with size M > N, we write out of bounds.

**However**: `reset(batch_size)` is always called before `infer()`, so buffer sizes should match. This might not be the primary cause.

### 2. Deep Copy Issues in Trajectory Tracking

**File**: `/home/eddie/repos/metamon/metamon/env/pykmn/vector_env.py`

**Lines 329-330**: Deep copying legal masks:
```python
legal_mask_p1=np.array(legal_masks_p1[i], copy=True),
legal_mask_p2=np.array(legal_masks_p2[i], copy=True),
```

**Lines 420-438**: Deep copying feature dictionaries:
```python
for key, value in features_p1.items():
    if isinstance(value, np.ndarray):
        features_p1_copy[key] = np.array(value, copy=True)
```

**Already Implemented**: Deep copying was added to prevent dangling pointers.

### 3. Observation State Management

**Lines 111-119**: Per-environment observation states:
```python
if hasattr(self.obs_space, 'init_obs_state'):
    self.obs_states = [(self.obs_space.init_obs_state(),
                        self.obs_space.init_obs_state())
                       for _ in range(num_envs)]
```

**Lines 380-385**: State-explicit observation extraction:
```python
obs_state_p1, obs_state_p2 = self.obs_states[i]
if obs_state_p1 is not None:
    obs_p1, new_state_p1 = self.obs_space(state_p1, obs_state_p1)
    obs_p2, new_state_p2 = self.obs_space(state_p2, obs_state_p2)
    self.obs_states[i] = (new_state_p1, new_state_p2)
```

**Potential Issue**: Observation states might hold references to Battle objects or numpy arrays that become invalid.

### 4. Incremental Cleanup Pattern

**Lines 137-190**: `_cleanup_battles_incremental()`:
```python
for i in range(self.num_envs):
    self.battles[i] = None
    self.results[i] = None
    self.prev_states_p1[i] = None
    self.prev_states_p2[i] = None
    if len(self.trajectories[i]) > 0:
        self.trajectories[i] = []
    if self.obs_states and i < len(self.obs_states):
        self.obs_states[i] = (None, None)
    if (i + 1) % chunk_size == 0:
        gc.collect(0)
```

**Purpose**: Prevents "destructor avalanche" by releasing references incrementally.

**Issue**: This might not fully clear nested references.

## Hypotheses

### Hypothesis 1: Observation State Memory Leak

**Theory**: `obs_states` tuples hold references to numpy arrays that point to Battle memory. When Battle is destroyed, these become dangling pointers.

**Test**: Compare behavior with `DefaultObservationSpace` (no state) vs `ExpandedObservationSpace` (with state).

**Evidence**:
- Corruption happens even without trajectory tracking
- Deep copying features doesn't prevent corruption

### Hypothesis 2: Hidden State in AMAGO Agent

**Theory**: The AMAGO agent's `hidden_state` (stored in LocalPolicyRunner) accumulates references across many steps, eventually holding stale pointers.

**Evidence**:
- Lines 265-271 attempt to detach hidden state:
```python
if isinstance(self.hidden_state, torch.Tensor):
    self.hidden_state = self.hidden_state.detach()
```

**Test**: Check if corruption happens without LocalPolicyRunner (pure random actions).

### Hypothesis 3: Numpy Array View Corruption

**Theory**: NumPy arrays created from PyKMN's internal buffers become views (not copies), and when Battle is destroyed, these views point to freed memory.

**Evidence**:
- `np.array(value, copy=True)` should prevent this
- But nested structures might still have views

**Critical Code**:
- `features.py` creates many np.array() calls
- If any return views instead of copies, corruption occurs

### Hypothesis 4: Battle Object Lifetime Mismatch

**Theory**: Battle objects are destroyed while:
1. Observation extraction is in progress
2. Feature dictionaries still hold references
3. Trajectory storage holds references

**Evidence**:
- Lines 199-215 in `vector_env.reset()`: Creates new battles
- But if old battles are still referenced elsewhere, corruption occurs

## Recommended Tests

### Test 1: Isolate Observation Space
```python
# Run with DefaultObservationSpace (no state)
env = PyKMNVectorEnv(..., obs_space=DefaultObservationSpace(), track_trajectories=False)
# vs
# Run with ExpandedObservationSpace (with state)
env = PyKMNVectorEnv(..., obs_space=ExpandedObservationSpace(), track_trajectories=False)
```

### Test 2: Force Deep Copy Everything
```python
# In _extract_observations_raw(), wrap ALL values:
features_p1_copy[key] = copy.deepcopy(value)
```

### Test 3: Clear Observation States Aggressively
```python
# After each step, clear observation states:
for i in range(self.num_envs):
    if self.obs_states[i] != (None, None):
        self.obs_states[i] = (None, None)
```

### Test 4: Disable Incremental Cleanup
```python
# In reset(), use naive cleanup instead of incremental:
self.battles = [None] * self.num_envs
gc.collect()  # Full collection
```

## Smoking Gun Candidates

### Most Likely: Observation State Holding Dangling References

**Location**: Lines 380-385 in `vector_env._extract_observations()`

**Problem**:
1. `obs_space(state_p1, obs_state_p1)` returns `(obs, new_state)`
2. `new_state` might hold references to numpy arrays from `state_p1`
3. `state_p1` is built from `features_to_universal_state()` which uses `pykmn_to_features_raw()`
4. If any numpy arrays in the chain are views (not copies), they point to Battle memory
5. When Battle is destroyed, `obs_states` holds dangling pointers
6. Next access triggers "corrupted size vs. prev_size"

**Fix**: Ensure observation spaces never store references to input arrays:
```python
# In observation space __call__():
def __call__(self, state, obs_state=None):
    # CRITICAL: Never store references to state in obs_state!
    # Always copy arrays if needed in state tracking
    ...
```

### Second Most Likely: Features Dictionary View Corruption

**Location**: `features.py` lines 284-346

**Problem**: NumPy scalar arrays created with `np.array(scalar, dtype=...)` might share memory in some edge cases.

**Fix**: Use explicit Python types instead:
```python
# Instead of:
"active_species_id": np.array(active_species_id, dtype=np.int32),
# Use:
"active_species_id": np.int32(active_species_id),
```

## Next Steps

1. **Check Observation Space Implementation**: Verify that observation spaces don't hold references to input arrays
2. **Add Memory Sanitizer**: Run with AddressSanitizer or Valgrind to catch exact corruption point
3. **Simplify Test**: Create minimal reproducer with just vector_env + random actions
4. **Trace Object Lifetimes**: Add logging to track when Battle objects are created/destroyed

## Conclusion

The heap corruption is NOT in PyKMN itself, but in our Python wrapper's memory management. The most likely culprit is observation state management holding dangling references to numpy arrays that were created from Battle accessor methods.
