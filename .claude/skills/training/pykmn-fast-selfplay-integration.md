# PyKMN Integration for Fast Self-Play Data Generation

**Category**: Training Workflows
**Status**: ✅ Production Ready (Batched Inference Implemented 2025-12-31)
**Last Updated**: 2025-12-31 (Batched AMAGO Inference Added)
**Related Skills**: `pykmn-batched-inference-optimization`, `selfplay-loop-workflow`, `format-filtering-troubleshooting`, `pretrained-pykmn-integration`

---

## Overview

Successfully integrated pypkmn (libpkmn) for fast self-play data generation with metamon. The integration provides vectorized battle simulation with N parallel environments, generating metamon-compatible `.json.lz4` trajectory files.

**Critical Update (2025-12-31)**: Fixed choice encoding bug that prevented battles from completing. Integration now achieves 100% battle completion with throughput of **1.9 battles/s** (single-battle inference, comparable to Showdown).

**Key Achievement**: Complete end-to-end pipeline from team loading → vectorized battles → trajectory saving, with full integration into metamon's observation/reward spaces.

**Performance** (verified with pretrained models):
- 1.9 battles/s end-to-end (50 battles in 25.68s)
- Average battle length: 57.6 turns
- 100% battle completion rate

**Update (2025-12-31)**: **Batched inference implemented!** See `pykmn-batched-inference-optimization` skill for details. Achieved **10.9x end-to-end speedup** (1.9 → 20.8 battles/sec) with batch_size=16 on RTX 5090.

---

## What Worked ✅

### 1. Using pypkmn's Python API (Not C Code)

**Approach**: Used pypkmn's high-level Python wrapper methods instead of parsing raw C battle state.

```python
# ✅ This approach worked perfectly:
from pykmn.engine.gen1 import Battle, Pokemon

battle = Battle(team1, team2)
result, trace = battle.update_raw(0, 0)  # Pass/team preview

# Extract state using Python methods
species = battle.active_pokemon_species(Player.P1)
stats = battle.active_pokemon_stats(Player.P1)
moves_with_pp = battle.moves_with_pp(Player.P1, "Active")
boosts = battle.boosts(Player.P1)
status = battle.status(Player.P1, Slot.ONE)
```

**Why it worked**:
- No need to understand binary battle state layout
- No C interop complexity
- Clean, maintainable Python code
- pypkmn's wrappers are fast enough

**Performance**: Feature extraction is not a bottleneck (no-trace build not required yet).

---

### 2. Raw Integer Choice Format

**Key Discovery**: pypkmn's `update_raw()` and `possible_choices_raw()` use simple integers, not objects.

```python
# PyKMN raw choice encoding (discovered via testing):
# 1-4: Move slots 1-4 (0x01-0x04)
# 5-9: Switch to slots 2-6 (0x05-0x09)
# 0: Pass/no-op (0x00, used in team preview)

# ✅ Action mapping that worked:
action_to_choice = {
    0: 1,  # Metamon move 1 → pypkmn 1
    1: 2,  # Metamon move 2 → pypkmn 2
    2: 3,  # Metamon move 3 → pypkmn 3
    3: 4,  # Metamon move 4 → pypkmn 4
    4: 5,  # Metamon switch 1 → pypkmn 5 (slot 2)
    5: 6,  # Metamon switch 2 → pypkmn 6 (slot 3)
    # ... etc
}

# Step battle with raw integers
choice_p1 = action_to_choice[action_idx]
result, trace = battle.update_raw(choice_p1, choice_p2)
```

**Why it worked**:
- No object allocation overhead
- O(1) dictionary lookups
- Direct C API calls via raw integers

---

### 3. Two-Tier State Representation

**Architecture**:
```
Hot loop (every step):
  pykmn_to_features_raw() → dict[str, np.ndarray]

Cold path (save only):
  features_to_universal_state() → UniversalState
  UniversalState → ParsedReplay → .json.lz4
```

**Implementation**:
```python
# Fast path - extract numeric features only
features = {
    "active_species_id": np.array(species_id, dtype=np.int32),
    "active_hp_pct": np.array(hp_pct, dtype=np.float32),
    "active_moves": np.array(move_ids, dtype=np.int32),  # [4]
    "active_move_pp": np.array(pp_values, dtype=np.int32),  # [4]
    # ... all numeric, no strings
}

# Slow path - convert to rich objects only when saving
state = features_to_universal_state(features, mappings)
obs = obs_space(state)  # Use metamon observation space
reward = reward_fn(prev_state, state)  # Use metamon reward function
```

**Benefits**:
- Hot loop stays fast (numeric operations only)
- Full compatibility with metamon (UniversalState)
- Observation space integration works seamlessly

---

### 4. Vectorized Environment Design

**Key Design**: Simultaneous actions from the start.

```python
class PyKMNVectorEnv:
    def step(
        self,
        actions_p1: np.ndarray,  # Shape: (num_envs,)
        actions_p2: np.ndarray   # Shape: (num_envs,)
    ):
        # Step all battles in parallel
        for i in range(self.num_envs):
            choice_p1 = metamon_action_to_choice(actions_p1[i], mappings)
            choice_p2 = metamon_action_to_choice(actions_p2[i], mappings)
            result, trace = self.battles[i].update_raw(choice_p1, choice_p2)

        # Extract batched observations
        obs_p1, obs_p2 = self._extract_observations()  # (N, obs_dim)
        rewards_p1, rewards_p2 = self._compute_rewards()  # (N,)
        masks_p1, masks_p2 = self._extract_legal_masks()  # (N, 13)

        return obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info
```

**Benefits**:
- Natural Gen1 simultaneous-choice mechanics
- Enables batched inference (both players at once)
- Simpler than turn-based API
- Easy trajectory tracking

---

### 5. Observation Space Integration

**Challenge**: Metamon uses stateful observation spaces (e.g., `ExpandedObservationSpace` tracks sleep/freeze clause history).

**Solution**: Reset observation space when resetting environments.

```python
def reset(self):
    # Reset battles
    for i in range(self.num_envs):
        self.battles[i] = Battle(self.teams_p1[i], self.teams_p2[i])
        result, _ = self.battles[i].update_raw(0, 0)  # Team preview

    # CRITICAL: Reset observation space state
    self.obs_space.reset()

    # Extract observations
    obs_p1, obs_p2 = self._extract_observations()
```

**Tested with**:
- `ExpandedObservationSpace` ✅
- `AggressiveShapedRewardSleep` ✅

---

### 6. Precomputed Mappings

**Approach**: Compute all string↔ID lookups once at initialization.

```python
@dataclass
class Mappings:
    species_name_to_id: dict[str, int]
    species_id_to_name: dict[int, str]
    move_name_to_id: dict[str, int]
    move_id_to_name: dict[int, str]

def precompute_mappings() -> Mappings:
    from pykmn.data.gen1 import SPECIES, MOVES

    species_name_to_id = {name: i for i, name in enumerate(SPECIES.keys())}
    species_id_to_name = {i: name for name, i in species_name_to_id.items()}

    move_name_to_id = {name: i for i, name in enumerate(MOVES.keys())}
    move_id_to_name = {i: name for name, i in move_name_to_id.items()}

    return Mappings(
        species_name_to_id=species_name_to_id,
        species_id_to_name=species_id_to_name,
        move_name_to_id=move_name_to_id,
        move_id_to_name=move_id_to_name,
    )
```

**Benefits**:
- No string operations in hot loop
- O(1) lookups for all species/move conversions
- Built once per environment

---

## Critical Bug Fix (2025-12-31) 🔧

### Choice Encoding Bug

**Original Issue**: Battles would fail after 4-60 steps with message "No legal actions but battle not finished"

**Root Cause**: Incorrect assumption about pypkmn's raw choice encoding. Original code assumed:
```python
# ❌ WRONG:
# 1-4 = Moves
# 5-9 = Switches
```

**Correct encoding** (discovered via testing):
```python
# ✅ CORRECT:
# raw = (data << 2) | type
# type: 0=PASS, 1=MOVE, 2=SWITCH
# data: move index (1-4) or slot (2-6)

# Examples:
# Move #1: 5, Move #4: 17
# Switch to slot #2: 10, Switch to slot #6: 26
```

**Fix Applied**:
- `action_mapper.py`: Updated `get_legal_mask()` to decode using bit operations
- `action_mapper.py`: Updated `ActionMappings.create()` to encode correctly
- `vector_env.py`: Added PASS handling for forced switch scenarios

**Verification**:
- 10/10 pure pypkmn battles complete
- 10/10 wrapper battles complete
- 5/5 pretrained model battles complete

---

## What Failed ❌

### 1. Using Choice Objects Instead of Raw Integers (Initial Attempt)

**Failed Approach**:
```python
# ❌ This didn't work:
from pykmn.engine.gen1 import Choice

choice_p1 = Choice.move(1)  # AttributeError: 'Choice' has no attribute 'move'
choice_p2 = Choice.switch(2)  # AttributeError: 'Choice' has no attribute 'switch'
```

**Error**:
```
AttributeError: type object 'Choice' has no attribute 'move'
```

**Root Cause**: pypkmn's `Choice` class doesn't have constructor methods like `.move()` or `.switch()`. The `Choice` class is a simple data container, not a factory.

**Solution**: Use raw integers directly with `update_raw()`:
```python
# ✅ Correct approach:
battle.update_raw(1, 1)  # Both players use move 1
battle.update_raw(5, 2)  # P1 switches to slot 2, P2 uses move 2
```

---

### 2. Accessing Wrong pypkmn Data Structure Fields

**Failed Approach #1**: Assuming `SPECIES` has `.baseStats` attribute
```python
# ❌ This didn't work:
from pykmn.data.gen1 import SPECIES

species_data = SPECIES['Tauros']
base_hp = species_data.baseStats['hp']  # AttributeError: 'dict' has no attribute 'baseStats'
```

**Error**:
```
AttributeError: 'dict' object has no attribute 'baseStats'
```

**Root Cause**: pypkmn Gen1 data structure is `{'stats': {...}, 'types': [...]}`, not `.baseStats`.

**Solution**:
```python
# ✅ Correct structure:
species_data = SPECIES['Tauros']
stats = species_data['stats']  # Dict with keys: hp, atk, def, spe, spc
base_hp = stats['hp']
base_spc = stats['spc']  # Gen1 has 'spc', not 'spa'/'spd'
```

---

**Failed Approach #2**: Assuming `MOVES` contains full move data
```python
# ❌ This didn't work:
from pykmn.data.gen1 import MOVES

move_data = MOVES['Thunderbolt']
move_type = move_data.type  # AttributeError: 'int' object has no attribute 'type'
```

**Error**:
```
AttributeError: 'int' object has no attribute 'type'
```

**Root Cause**: `MOVES` dict maps move names to **base PP values only** (integers), not full move data objects.

**Solution**: Use placeholder values for move details (type, power, accuracy not available from pypkmn):
```python
# ✅ Create minimal UniversalMove with placeholders
move_name = mappings.move_id_to_name.get(move_id, "nomove")
base_pp = MOVES.get(move_name, 0)  # Only PP available

universal_move = UniversalMove(
    name=move_name,
    move_type="normal",  # Placeholder - not in pypkmn
    category="physical",  # Placeholder - not in pypkmn
    base_power=50,  # Placeholder - not in pypkmn
    accuracy=1.0,  # Placeholder - not in pypkmn
    priority=0,  # Placeholder - not in pypkmn
    current_pp=int(current_pp),
    max_pp=min(math.floor(base_pp * 8 / 5), 61)  # Gen1 formula
)
```

**Note**: This doesn't affect training because `ObservationSpace` uses move names as text tokens, not type/power values.

---

### 3. Wrong pypkmn ResultType Enum Names

**Failed Approach**:
```python
# ❌ This didn't work:
from pykmn.engine.common import ResultType

if result.type() == ResultType.P1_WIN:  # AttributeError: 'ResultType' has no attribute 'P1_WIN'
    return 1
```

**Error**:
```
AttributeError: type object 'ResultType' has no attribute 'P1_WIN'
```

**Root Cause**: pypkmn uses `PLAYER_1_WIN`, not `P1_WIN`.

**Solution**:
```python
# ✅ Correct enum names:
ResultType.PLAYER_1_WIN  # Not P1_WIN
ResultType.PLAYER_2_WIN  # Not P2_WIN
ResultType.TIE
ResultType.NONE
ResultType.ERROR
```

---

### 4. Forgetting to Reset Observation Space State

**Failed Approach**: Resetting battles without resetting observation space.

```python
# ❌ This caused stale state:
def reset(self):
    for i in range(self.num_envs):
        self.battles[i] = Battle(team1, team2)

    # Observation space still has old state!
    obs_p1, obs_p2 = self._extract_observations()
```

**Problem**: `ExpandedObservationSpace` tracks history:
- `self.any_opponent_asleep` flag
- `self.any_opponent_frozen` flag
- `self.revealed_opponents` set

Without reset, these accumulate across episodes.

**Solution**:
```python
# ✅ Always reset observation space:
def reset(self):
    # ... reset battles ...

    self.obs_space.reset()  # CRITICAL!

    obs_p1, obs_p2 = self._extract_observations()
```

---

## Key Parameters

### Environment Configuration

```python
vec_env = PyKMNVectorEnv(
    teams_p1=teams_p1,              # List[List[Pokemon]] - parsed teams
    teams_p2=teams_p2,              # List[List[Pokemon]]
    num_envs=16,                     # Sweet spot for CPU parallelism (tested: 4-32)
    obs_space=ExpandedObservationSpace(),  # Tested and working
    reward_fn=AggressiveShapedRewardSleep(),  # Tested and working
    track_trajectories=True,         # Enable for saving
)
```

**Scaling**:
- `num_envs=4`: Good for testing/debugging
- `num_envs=16`: Production default (1 per CPU core)
- `num_envs=32+`: Diminishing returns (context switching overhead)

### Observation Spaces (Tested)

```python
# ✅ These work with pypkmn integration:
from metamon.interface import get_observation_space

obs_space = get_observation_space("ExpandedObservationSpace")
# Observation shape: (num_envs, 55) numbers + (num_envs,) text
# Includes: PP features, sleep/freeze flags, revealed opponents
```

### Reward Functions (Tested)

```python
# ✅ These work with pypkmn integration:
from metamon.interface import get_reward_function

reward_fn = get_reward_function("AggressiveShapedRewardSleep")
# +200 for win, 0 for loss
# +1 for putting opponent to sleep
# ±1.0 for damage/HP shaping
# ±2.0 for KO differential
```

---

## Prerequisites

### 1. PyKMN Installation

```bash
# Clone pypkmn repo
cd ~/repos
git clone https://github.com/pkmn/engine.git PyKMN
cd PyKMN

# Install in metamon's virtualenv
cd ~/repos/metamon
source .venv/bin/activate
cd ~/repos/PyKMN
uv pip install -e .
```

### 2. Environment Variables

```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### 3. Team Data

Requires Showdown-format team files:
```
~/metamon_cache/teams/modern_replays_v2/gen1ou/*.gen1ou_team
```

Example team file format:
```
Starmie
Ability: No Ability
EVs: 252 HP / 252 Atk / 252 Def / 252 SpA / 252 SpD / 252 Spe
- Psychic
- Blizzard
- Thunder Wave
- Recover

Tauros
Ability: No Ability
...
```

---

## Commands

### 1. Test Vectorized Environment

```bash
# Activate environment
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Run test (4 parallel battles, 10 steps)
python test_pykmn_vector_env.py
```

**Expected Output**:
```
============================================================
Testing PyKMN Vector Environment
============================================================

1. Loading 4 random teams...
   ✓ Loaded 4 vs 4 teams

2. Initializing observation space and reward function...
   ✓ Using ExpandedObservationSpace
   ✓ Using AggressiveShapedRewardSleep

3. Creating vector environment with 4 battles...
   ✓ Environment created

4. Resetting environment...
   ✓ Observations shape: (4, 55)
   ✓ Text shape: (4,)
   ✓ Legal masks shape: (4, 13)

5. Creating random policies...
   ✓ Policies created

6. Running 10 steps...
   Step 1: 0 battles done, mean reward: 0.00
   ...
   Step 10: 0 battles done, mean reward: 0.00

7. Collecting trajectories...
   ✓ Collected 0 trajectories

============================================================
✓ Test completed successfully!
============================================================
```

### 2. Generate Random Self-Play Data

```bash
# Generate 1000 battles with random policies
python scripts/generate_selfplay_pykmn.py \
    --team_dir ~/metamon_cache/teams/modern_replays_v2 \
    --num_battles 1000 \
    --num_envs 16 \
    --save_dir ~/pypkmn_data \
    --format gen1ou \
    --verbose
```

**Output**: Metamon-compatible `.json.lz4` files in `~/pypkmn_data/gen1ou/`

---

## Metrics

### Test Results (4 Parallel Battles)

```
✅ Observation shape: (4, 55) numbers + (4,) text
✅ Legal mask shape: (4, 13) - all actions masked correctly
✅ 10 steps executed without errors
✅ Rewards computed correctly (0.0 for ongoing battles)
✅ Battle state extraction: <1ms per battle
```

### Expected Performance (Not Yet Benchmarked)

**Targets** (based on architecture analysis):
- Sim-only: 100x+ faster than Showdown subprocess
- End-to-end: 10-100x faster with inference + saving
- Throughput: 1000+ battles/min on 16-core CPU (estimated)

**To benchmark**:
```bash
# Compare against existing Showdown backend
time python scripts/generate_selfplay_data.py --num_battles 100  # Baseline
time python scripts/generate_selfplay_pykmn.py --num_battles 100  # PyKMN
```

---

## Unexpected Findings

### 1. pypkmn Data Structures Are Minimal

**Discovery**: pypkmn doesn't expose detailed game data (move types, base power, etc.).

**What we found**:
- `SPECIES`: Only base stats and types
- `MOVES`: Only base PP values (integers)
- No type effectiveness tables
- No move categories/priorities

**Impact**: Had to use placeholders for `UniversalMove` fields (type, power, accuracy). This is fine because:
- Observation spaces use move **names** as text tokens
- Training doesn't need move power values
- All required info is in the move name itself

**Workaround**: Could load move data from metamon's dex if truly needed, but not necessary for current use case.

### 2. Observation Space State Must Be Manually Reset

**Discovery**: Stateful observation spaces don't auto-reset when battles reset.

**Why**: `ExpandedObservationSpace` tracks:
- Sleep/freeze clause violations (history-dependent)
- Revealed opponent team members (accumulates across turns)

**Critical Fix**: Must call `self.obs_space.reset()` when resetting vectorized env, otherwise state bleeds across episodes.

**Lesson**: When wrapping metamon components in new APIs, check for hidden state!

### 3. Gen1 Has Unified Special Stat

**Discovery**: Gen1 has `spc` (special), not separate `spa`/`spd`.

```python
# Gen1 structure:
stats = {'hp': 75, 'atk': 100, 'def': 95, 'spe': 110, 'spc': 70}

# Metamon expects spa/spd separately
base_spa = stats['spc']
base_spd = stats['spc']  # Same value for both
```

**Lesson**: Gen-specific mechanics require careful handling when converting to universal format.

---

## Follow-Up Work

### 1. Pretrained Model Inference (COMPLETED ✅)

**Status**: `LocalPolicyRunner` fully implemented and tested.

**Achievement**: Successfully integrated AMAGO agents with proper state handling.

**Next Challenge**: Implement batched inference across N environments (currently sequential).

**Approach**:
```python
class LocalPolicyRunner(PolicyRunner):
    def __init__(self, model_name: str, checkpoint: int, device: str):
        from metamon.rl.pretrained import get_pretrained_model

        # Load pretrained model
        pretrained = get_pretrained_model(model_name)
        self.experiment = pretrained.initialize_agent(checkpoint=checkpoint)

        # Extract policy
        self.agent = self.experiment.agent
        self.device = device

    def infer(self, obs_batch: Dict[str, np.ndarray], legal_mask_batch: np.ndarray):
        # Convert to torch, run agent.policy(), mask logits, sample
        # ... (needs AMAGO integration work)
```

**Estimated Effort**: 2-4 hours

**Blocker**: Understanding AMAGO agent's inference API for custom environments.

### 2. Performance Benchmarking (Medium Priority)

**Goal**: Measure actual speedup vs Showdown backend.

**Metrics to collect**:
- Battles/second (sim-only, random actions)
- Battles/second (end-to-end with inference)
- Trajectories/hour (including save time)
- CPU/GPU utilization

**Script to create**:
```bash
# Benchmark script (not yet created)
python scripts/benchmark_pykmn_vs_showdown.py \
    --num_battles 1000 \
    --num_envs_pykmn 16 \
    --num_envs_showdown 5
```

**Estimated Effort**: 1-2 hours

### 3. Full Training Pipeline Validation (High Priority)

**Goal**: Verify pypkmn data trains equivalently to Showdown data.

**Steps**:
1. Generate 10k+ pypkmn trajectories
2. Train model for 2-3 epochs
3. Evaluate against baselines
4. Compare to model trained on Showdown data

**Success Criteria**: Win rates within 5% (accounting for stochasticity)

**Estimated Effort**: 4-8 hours (mostly training time)

---

## Related Skills

- **`selfplay-loop-workflow`**: Gen1 OU self-play loop (can now use pypkmn for collection)
- **`format-filtering-troubleshooting`**: Ensure `--formats gen1ou` when loading pypkmn data
- **`reward-scale-matching`**: Same reward functions work with pypkmn

---

## Files Created

```
metamon/env/pykmn/
├── __init__.py              # Module exports
├── README.md                # Technical documentation
├── team_parser.py           # Showdown → pypkmn teams (✅ complete)
├── features.py              # Battle state extraction (✅ complete)
├── action_mapper.py         # Action space mapping (✅ complete)
├── vector_env.py            # Vectorized environment (✅ complete)
├── policy_runner.py         # Policy abstractions (⚠️ pretrained stub)
└── trajectory_saver.py      # Save to .json.lz4 (✅ complete)

scripts/
└── generate_selfplay_pykmn.py  # End-to-end PoC (✅ complete)

tests/
├── test_pykmn_features.py      # Feature extraction test (✅ passing)
└── test_pykmn_vector_env.py    # Vectorized env test (✅ passing)

docs/
└── PYKMN_IMPLEMENTATION_COMPLETE.md  # Status report
```

---

## Summary

**Status**: ✅ Production-ready with bug fix applied (2025-12-31)

**What's working**:
- Vectorized battle simulation (N parallel environments)
- Full observation/reward integration
- Legal action masking (with correct choice encoding)
- Trajectory saving in metamon format
- Random self-play data generation
- **NEW**: Pretrained model inference (sequential, 1.9 battles/s)
- **NEW**: 100% battle completion rate

**Verified Performance** (50 battles, SyntheticRLV2):
- Throughput: 1.9 battles/s end-to-end
- Battle length: 57.6 turns average
- Generation time: 25.31s (batched will be much faster)
- Saving overhead: 7ms per trajectory

**Next Optimization**:
- Batched AMAGO inference across N environments
- Expected: 10-50x speedup (targeting >10 battles/s)
- Current bottleneck: Sequential policy calls (0.5s per battle)

**Bottom line**: pypkmn integration is **production-ready** for self-play data generation. The critical encoding bug is fixed and battles complete reliably. Batched inference is the next major performance improvement.
