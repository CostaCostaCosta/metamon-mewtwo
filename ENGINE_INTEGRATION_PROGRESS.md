# Engine Integration Progress

## Goal
Use the optimized pkmn/engine (Zig-based) for Gen 1 battle simulation instead of Pokemon Showdown, achieving 10-100× speedup for self-play training.

## Phase 1: Minimal Engine Wrapper ✅ COMPLETE

### Completed (Day 1)
1. **Set up Zig compiler**
   - Downloaded and extracted Zig 0.13.0
   - Path: `~/repos/engine/zig-linux-x86_64-0.13.0/`

2. **Built engine library**
   - Static library: `~/repos/engine/zig-out/lib/libpkmn-showdown.a`
   - Shared library: `~/repos/engine/zig-out/lib/libpkmn-showdown.so`
   - Build command: `./zig-linux-x86_64-0.13.0/zig build -Dshowdown -Ddynamic -Doptimize=ReleaseFast`

3. **Created minimal Python ctypes bindings**
   - Location: `~/repos/metamon/metamon/engine/bindings.py`
   - Approach: Treat battle state as opaque (384 bytes), no manual struct definitions
   - Key functions wrapped:
     - `Gen1Battle` class with `update()` and `get_legal_choices()`
     - Battle state management
     - Choice encoding/decoding
   - Hardcoded example battle bytes from C example for testing

4. **Verified bindings work**
   - Test script: `~/repos/metamon/test_bindings_direct.py`
   - Successfully ran 77-turn battle to completion
   - Result: P1 LOSE (valid game outcome)

### Key Design Decisions
- **Opaque state**: Battle state kept as raw bytes, not decoded into structs (avoids fragility)
- **Minimal API**: Only wrapped essential functions (update, get_choices), not full C API
- **Hardcoded teams**: Using example bytes for initial testing (team parsing deferred)
- **No UniversalState in hot loop**: Following critique to keep simulation fast

## Phase 2: Self-Play Worker & Training Test (IN PROGRESS)

### Next Steps
1. **Create minimal self-play worker** (`selfplay_worker.py`)
   - Input: two policy callables, number of games
   - Output: raw trajectories (no UniversalState conversion yet)
   - For MVP: use random policies or simple heuristics

2. **Simple trajectory format**
   - Save as `.npz` files: `{'obs': array, 'actions': array, 'rewards': array, 'dones': array}`
   - OR: Save raw battle states + action sequences
   - Metadata: engine version, policies used, outcome

3. **Observation converter** (minimal for MVP)
   - Extract just HP, active Pokemon ID, available moves
   - Flat numpy array compatible with AMAGO
   - Full UniversalState conversion can be done offline later

4. **AMAGO adapter**
   - Minimal glue code to load engine trajectories
   - Test: Run AMAGO for a few gradient steps, verify loss decreases

5. **Validation experiment**
   - Generate 1000 engine games
   - Time it (should be <10 min vs hours with Showdown)
   - Run small training experiment
   - **Decision point**: If 10×+ faster AND training works → proceed to Phase 3

## Phase 3: Scale & Validate Quality (PENDING)

## Phase 4: Production Polish (PENDING)

## Current Blockers
- None! Ready to continue Phase 2

## Performance Baseline
- **Target**: 10+ games/sec single thread
- **Current**: Not yet benchmarked (need self-play worker first)
- **Showdown baseline**: TBD (need to measure)

## Testing
- Battle simulation: ✅ Works (77 turns, valid outcome)
- Random choices: ✅ Works
- Policy integration: ⏳ TODO
- Training loop: ⏳ TODO

## Files Created
- `~/repos/metamon/metamon/engine/__init__.py` - Package marker
- `~/repos/metamon/metamon/engine/bindings.py` - ctypes wrappers
- `~/repos/metamon/test_bindings_direct.py` - Test script

## Notes
- Zig 0.13 has ~10-20% performance regression (warning shown)
- Battle bytes are in native endianness
- Need actual team conversion for production use (currently using hardcoded bytes)
- UniversalState is complex (14+ fields) - avoid constructing in hot loop per critique
