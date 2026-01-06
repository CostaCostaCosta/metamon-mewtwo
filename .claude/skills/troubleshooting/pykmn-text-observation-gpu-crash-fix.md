# PyKMN Text Observation GPU Crash Fix

## Problem
GPU inference pipeline crashes when trying to convert text observations to PyTorch tensors.

### Error Fixed
```
TypeError: can't convert np.ndarray of type numpy.str_.
The only supported types are: float64, float32, float16, complex64, complex128,
int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
```

## Solution

### The Fix Applied
**File**: `metamon/env/pykmn/policy_runner.py`, lines 187-199

```python
# Fixed to skip text fields when converting to tensors
obs_torch = {}
for k, v in obs_dict.items():
    # Skip text fields that can't be converted to tensors
    if k == 'text' or (hasattr(v, 'dtype') and ('str' in str(v.dtype) or v.dtype == np.object_)):
        continue
    elif isinstance(v, np.ndarray):
        # Convert numeric/bool arrays to tensors
        obs_torch[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
    else:
        obs_torch[k] = v
```

### Required: Use TokenizedObservationSpace
Models expect tokenized integers, not raw text strings:

```python
from metamon.interface import TokenizedObservationSpace, DefaultObservationSpace
from metamon.tokenizer import PokemonTokenizer

# Create tokenized observation space
tokenizer = PokemonTokenizer()
vocab_path = os.path.join(os.environ["METAMON_CACHE_DIR"], "vocab.json")
if os.path.exists(vocab_path):
    tokenizer.load_tokens_from_disk(vocab_path)

base_obs_space = DefaultObservationSpace()
obs_space = TokenizedObservationSpace(base_obs_space, tokenizer)
```

## What This Fix Resolves
- ✅ Text observation type conversion error
- ✅ Allows GPU inference to start
- ✅ Models can now receive properly formatted observations

## What Still Crashes
- ❌ Segmentation faults still occur during selfplay generation
- ❌ Various heap corruption errors (`corrupted size vs. prev_size`, `free(): invalid next size`)
- ❌ Crashes appear non-deterministic but often after ~100-200 battles

## Unresolved Issues

### Current Crash Patterns
1. **Segmentation fault** - Memory access violation in native code
2. **Heap corruption** - Memory management issues in C++/Python boundary
3. **Non-deterministic timing** - Sometimes immediate, sometimes after many battles

### Possible Causes (Not Confirmed)
- PyKMN Battle object lifecycle management
- Memory accumulation without proper cleanup
- Threading/GIL issues with batch processing
- Tensor memory management across GPU/CPU boundary

## Current Workarounds

### Subprocess Isolation (Most Stable)
```bash
python scripts/generate_selfplay_subprocess.py \
    --model SyntheticRLV2 \
    --num_battles 1000 \
    --chunk_size 16 \
    --max_workers 4 \
    --format gen1ou \
    --team_set smogon_pass2 \
    --save_dir ~/selfplay_data
```

### Small Batches with Cleanup
```bash
# May still crash but less frequently
python scripts/generate_selfplay_safe.py \
    --model SyntheticRLV2 \
    --num_battles 1000 \
    --batch_size 8 \
    --battles_per_chunk 50 \
    --format gen1ou \
    --team_set smogon_pass2 \
    --save_dir ~/selfplay_data
```

## Testing Notes

### What We Verified
- PyKMN alone can handle 10,000+ battles without crashes
- Text observation conversion was definitely broken and is now fixed
- The "128 barrier" was a misdiagnosis - crashes happen at various batch sizes

### What Remains Unclear
- Root cause of segmentation faults
- Why subprocess isolation helps (different memory layout?)
- Relationship between batch size and crash frequency

## Files Modified
- `metamon/env/pykmn/policy_runner.py` - Fixed text observation conversion
- `scripts/generate_selfplay_batched.py` - Removed incorrect batch size warning (though crashes still occur)

## Status
⚠️ **PARTIALLY FIXED** - Text observation bug is resolved but underlying stability issues remain.

## Next Steps for Investigation
1. Run with AddressSanitizer/Valgrind to catch memory issues
2. Add comprehensive logging around crash points
3. Test with different PyKMN versions
4. Investigate Battle object cleanup in vector environment
5. Check for reference counting issues in Python/C++ boundary

## Important Note
The text observation fix was necessary and correct, but it only resolved one layer of issues. The underlying memory management problems in the PyKMN integration still need to be addressed for truly stable batch inference.