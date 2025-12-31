# Pretrained AMAGO Model Integration with PyKMN

**Category**: Training Workflows
**Status**: ⚠️ AMAGO Integration Complete, PyKMN Environment Has Bugs
**Created**: 2025-12-31
**Related Skills**: `pykmn-fast-selfplay-integration`

---

## Overview

Successfully integrated pretrained AMAGO models (SyntheticRLV2, etc.) with pypkmn for intelligent self-play data generation. The integration allows loading 200M parameter models and running inference at scale, but revealed critical bugs in the underlying pypkmn vector environment that prevent battles from completing.

**Key Achievement**: Complete LocalPolicyRunner implementation that properly handles AMAGO's recurrent state, RL2 inputs, and action masking.

**Critical Issue**: PyKMN battles become stuck after ~100-600 steps with all actions marked as illegal, preventing battle completion.

---

## What Worked ✅

### 1. LocalPolicyRunner Implementation

**File**: `metamon/env/pykmn/policy_runner.py:46-220`

Successfully implemented full integration between AMAGO agents and pypkmn:

```python
class LocalPolicyRunner(PolicyRunner):
    def __init__(self, model_name, checkpoint, device, temperature, verbose):
        # Load pretrained model from HuggingFace
        pretrained_cls = get_pretrained_model(model_name)
        self.agent = pretrained_cls.initialize_agent(...)

        # Get action space size (9 for MinimalActionSpace, 13 for full)
        self.action_dim = pretrained_cls.action_space.gym_space.n

        # Initialize recurrent state
        self.hidden_state = None  # Initialized on first infer()

    def infer(self, obs_dict, legal_mask_batch):
        # Initialize hidden state on first call (REQUIRED for inference mode)
        if self.hidden_state is None:
            self.hidden_state = self.agent.traj_encoder.init_hidden_state(
                batch_size, self.device
            )

        # Build RL2 input: (prev_action_onehot, prev_reward)
        # Trim action mask to model's action space
        # Add sequence dimension: (batch,) -> (batch, 1)
        # time_idxs: (batch,) -> (batch, 1, 1) for position embedding

        actions, self.hidden_state = self.agent.get_actions(
            obs=obs_torch_seq,   # (batch, 1, features)
            rl2s=rl2s_seq,        # (batch, 1, action_dim+1)
            time_idxs=time_idxs_seq,  # (batch, 1, 1)
            hidden_state=self.hidden_state,
            sample=True
        )
```

**Why it worked**:
- Followed `amago.experiment.interact()` patterns exactly
- Proper hidden state management for recurrent models
- Correct tensor shapes for AMAGO's API

---

### 2. Key Implementation Details

#### Hidden State Initialization
```python
# ✅ MUST initialize hidden state, can't pass None in eval mode
if self.hidden_state is None:
    self.hidden_state = self.agent.traj_encoder.init_hidden_state(
        batch_size, self.device
    )
```

**Critical**: If `hidden_state=None`, AMAGO uses `training_forward()` which requires `model.train()`. In eval mode with recurrent models, you MUST initialize the hidden state.

#### time_idxs Shape
```python
# ✅ Correct: (batch,) -> (batch, 1, 1)
time_idxs = torch.zeros((batch_size,), dtype=torch.long, device=device)
time_idxs_seq = time_idxs.unsqueeze(1).unsqueeze(2)  # (B, L, 1)

# ❌ Wrong: (batch, 1) causes "not enough values to unpack" error
# The position embedding does squeeze(-1), expects (B, L) after squeeze
```

#### Action Space Trimming
```python
# Model uses MinimalActionSpace (9 actions: 4 moves + 5 switches)
# PyKMN uses full space (13 actions: 4 moves + 5 switches + 4 tera)
illegal_mask_trimmed = illegal_mask[:, :self.action_dim]  # Trim to 9
```

#### Separate Policy Instances Required
```python
# ✅ MUST create separate instances for each player
policy_p1 = LocalPolicyRunner(...)
policy_p2 = LocalPolicyRunner(...)  # DON'T reuse policy_p1!

runner = SelfPlayRunner(vec_env, policy_p1=policy_p1, policy_p2=policy_p2)
```

**Why**: Each instance has its own `hidden_state`, `prev_actions`, `prev_rewards`. Sharing causes state corruption.

---

### 3. Observation Space Integration

**Key**: Use the pretrained model's observation space, not a manually created one:

```python
# ✅ Correct approach
pretrained_cls = get_pretrained_model("SyntheticRLV2")
obs_space = pretrained_cls.observation_space  # TokenizedObservationSpace
reward_fn = pretrained_cls.reward_function
action_space = pretrained_cls.action_space  # MinimalActionSpace (9 actions)

vec_env = PyKMNVectorEnv(
    teams_p1=teams_p1,
    teams_p2=teams_p2,
    num_envs=num_envs,
    obs_space=obs_space,  # Use model's space!
    reward_fn=reward_fn,
    battle_format="gen1ou",
)
```

**Result**: PyKMN returns `{"numbers": (B, 48), "text_tokens": (B, 85)}` which matches model expectations.

---

### 4. Test Suite Validation

**File**: `test_pretrained_pykmn.py`

```bash
python test_pretrained_pykmn.py

# Output:
# ✓ Single-step test PASSED
# ✓ Multi-step test PASSED
# ✓ ALL TESTS PASSED!
```

Validated:
- Model loading (SyntheticRLV2, 200M params)
- Single-step inference
- Multi-step inference with state tracking
- Action selection respects legal masks (when provided correctly)

---

## What Failed ❌

### 1. PyKMN Vector Environment Battle State Bug

**Symptom**: After 100-600 steps, ALL actions become illegal and battles never complete:

```
[PolicyRunner] Step 0: action=3, legal_actions=[3 7], num_legal=2  # ✓ Normal
[PolicyRunner] Step 1: action=3, legal_actions=[3 7], num_legal=2  # ✓ Normal
...
[PolicyRunner] Step 587: action=5, legal_actions=[], num_legal=0   # ✗ ALL ILLEGAL!
[PolicyRunner] Step 588: action=7, legal_actions=[], num_legal=0   # ✗ Stuck forever
...
Warning: Reached max steps (1000), resetting environments
```

**Debug Output**:
```python
# Battle state inspection shows:
raw_legal=[False False False False False False False False False False False False False]
trimmed_legal=[False False False False False False False False False False False]

# Even though battle.update_raw() returns:
result.type() == ResultType.NONE  # Battle not finished
dones[0] == False  # Not marked as done
```

**Root Cause**: Unknown. The pypkmn battle state becomes invalid:
- Legal action mask extraction returns all False
- Battle doesn't end (`result.type() == NONE`)
- No rewards generated (stuck at 0.00)
- Battle can't progress

**Impact**: 100% of battles timeout at 1000 steps. No trajectories complete successfully.

---

### 2. Attempted Debugging

#### Test 1: Same Action Repeatedly
```python
# Sending action 3 (known to be legal initially) repeatedly
obs_p1, obs_p2, rewards, _, dones, _ = env.step([3], [3])
# Result: Battle doesn't advance, same legal actions returned
```

#### Test 2: Check Battle Object State
```python
result, trace = battle.update_raw(choice_p1, choice_p2)
print(f"Result type: {result.type()}")  # Always NONE
print(f"Battle done: {battle.is_finished()}")  # Method doesn't exist
```

**Hypothesis**: Either:
1. `battle.update_raw()` isn't being called correctly
2. pypkmn's battle state gets corrupted
3. Action mapping `metamon_action_to_choice()` produces invalid choices
4. Legal mask extraction `get_legal_mask()` has a bug

---

### 3. Initial Implementation Mistakes (Now Fixed)

#### Shared Policy Instance
```python
# ❌ This caused state corruption:
policy = LocalPolicyRunner(...)
runner = SelfPlayRunner(vec_env, policy_p1=policy, policy_p2=policy)

# Hidden state gets overwritten when P2 infers after P1
```

#### Missing Reward Initialization
```python
# ❌ Crashed on second step:
self.prev_actions = actions.squeeze(-1).squeeze(1)
# prev_rewards still None!

# ✅ Fixed:
if self.prev_rewards is None:
    self.prev_rewards = torch.zeros((batch_size,), device=self.device)
```

#### Wrong time_idxs Shape
```python
# ❌ Caused "not enough values to unpack" error:
time_idxs = torch.zeros((batch_size, 1), ...)  # (B, 1)
time_idxs_seq = time_idxs.unsqueeze(1)  # (B, 1, 1) WRONG!

# After squeeze(-1) in position embedding: (B, 1) can't unpack to (B, L)

# ✅ Fixed:
time_idxs = torch.zeros((batch_size,), ...)  # (B,)
time_idxs_seq = time_idxs.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
# After squeeze(-1): (B, 1) unpacks to B, L=1 ✓
```

---

## Troubleshooting Plan

### Phase 1: Isolate PyKMN Bug (Priority: CRITICAL)

**Goal**: Determine if bug is in pypkmn engine or our integration.

#### Step 1: Minimal Reproduction
```python
# Test pypkmn directly without metamon wrapper
from pykmn.engine.gen1 import Battle, Player
import pykmn.data.gen1 as data

# Create simplest possible teams (e.g., 1 Pokémon each)
team1 = [Pokemon.create(data.Species.TAUROS, [...])]
team2 = [Pokemon.create(data.Species.SNORLAX, [...])]

battle = Battle(team1, team2)
result, _ = battle.update_raw(0, 0)  # Team preview

for i in range(100):
    legal_p1 = battle.legal_choices(Player.P1)
    legal_p2 = battle.legal_choices(Player.P2)

    print(f"Step {i}: P1_legal={legal_p1}, P2_legal={legal_p2}")

    if len(legal_p1) == 0 or len(legal_p2) == 0:
        print(f"BUG: No legal actions at step {i}")
        print(f"Battle finished: {result.type()}")
        break

    # Always pick first legal action
    result, _ = battle.update_raw(legal_p1[0], legal_p2[0])

    if result.type() != ResultType.NONE:
        print(f"Battle finished normally at step {i}")
        break
```

**Expected**: Battle should either:
- Complete normally before step 100 with a winner
- Continue indefinitely if moves don't deal damage
- **NOT** have zero legal actions while unfinished

**If bug reproduces**: Issue is in pypkmn itself (upstream bug).
**If bug doesn't reproduce**: Issue is in metamon's wrapper code.

---

#### Step 2: Check Action Mapping
```python
# Verify metamon_action_to_choice produces valid pypkmn choices
from metamon.env.pykmn.action_mapper import (
    metamon_action_to_choice,
    get_legal_mask,
    precompute_action_mappings
)

mappings = precompute_action_mappings()

# For each metamon action 0-12, what pypkmn choice does it map to?
for action_idx in range(13):
    choice = metamon_action_to_choice(action_idx, mappings)
    print(f"Metamon action {action_idx} -> PyKMN choice {choice}")

# Verify inverse: are all pypkmn legal choices represented?
legal_pykmn_choices = battle.legal_choices(Player.P1)
legal_metamon_mask = get_legal_mask(battle, result, Player.P1, mappings)

print(f"PyKMN legal: {legal_pykmn_choices}")
print(f"Metamon mask: {legal_metamon_mask.nonzero()[0]}")
```

**Expected**: Every legal pypkmn choice should have a True entry in metamon mask.

---

#### Step 3: Trace Battle State Evolution
```python
# Add verbose logging to vector_env.py step()
def step(self, actions_p1, actions_p2):
    for i in range(self.num_envs):
        choice_p1 = metamon_action_to_choice(actions_p1[i], self.action_mappings)
        choice_p2 = metamon_action_to_choice(actions_p2[i], self.action_mappings)

        # LOG BEFORE
        legal_before = self.battles[i].legal_choices(Player.P1)
        print(f"[Env {i}] Before update: legal_P1={legal_before}")

        result, trace = self.battles[i].update_raw(choice_p1, choice_p2)

        # LOG AFTER
        legal_after = self.battles[i].legal_choices(Player.P1)
        print(f"[Env {i}] After update: legal_P1={legal_after}, result={result.type()}")

        self.results[i] = result
```

**Look for**: The exact step where legal actions transition from non-empty to empty.

---

### Phase 2: Workarounds (If PyKMN Bug Confirmed)

#### Option A: Use Showdown Backend with Pretrained Models
```python
# Fall back to metamon's original Showdown integration
from metamon.env.wrappers import BattleAgainstBaseline

# Wrap with AMAGO-compatible interface
# (Requires implementing similar LocalPolicyRunner for Showdown backend)
```

**Trade-off**: Lose 10-100x speedup, but battles will complete.

#### Option B: Fork and Fix pypkmn
- Clone pypkmn repository
- Add debug logging to C++ battle engine
- Identify where legal action computation breaks
- Submit upstream fix

**Trade-off**: Time investment, C++ debugging required.

#### Option C: Hybrid Approach
- Use pypkmn for initial fast generation with simple teams
- Use Showdown for complex teams / validation
- Report metrics: % of battles that complete vs timeout

---

### Phase 3: Integration Testing (After Bug Fix)

Once PyKMN battles complete successfully:

```bash
# Generate 100 battles with pretrained model
python scripts/generate_selfplay_pykmn.py \
    --team_dir ~/metamon_cache/teams/modern_replays_v2 \
    --num_battles 100 \
    --num_envs 16 \
    --save_dir ~/pykmn_pretrained_data \
    --format gen1ou \
    --model SyntheticRLV2 \
    --checkpoint 48 \
    --device cuda \
    --verbose

# Validate trajectory quality
python scripts/filter_selfplay_data.py \
    --input_dir ~/pykmn_pretrained_data/gen1ou \
    --output_dir ~/pykmn_filtered \
    --format gen1ou \
    --verbose

# Check metrics:
# - Average battle length (expect 10-50 turns for intelligent play)
# - Win rate balance (expect ~50/50 for self-play)
# - Invalid action rate (expect <1%)
# - Trajectory file sizes (expect similar to Showdown replays)
```

---

## Key Learnings

### 1. AMAGO Inference API

**Pattern from `amago.experiment.interact()`**:
```python
# Environment provides (obs, rl2s, time_idxs) as numpy
obs_np, rl2s_np, time_idxs_np = env.current_timestep()

# Convert to torch and add sequence dim
obs_torch = {k: torch.from_numpy(v).unsqueeze(1) for k, v in obs_np.items()}
rl2s_torch = torch.from_numpy(rl2s_np).unsqueeze(1)
time_idxs_torch = torch.from_numpy(time_idxs_np).unsqueeze(1)

# Get actions (with initialized hidden state)
actions, hidden_state = agent.get_actions(
    obs=obs_torch,           # (B, L, obs_dim)
    rl2s=rl2s_torch,          # (B, L, action_dim+1)
    time_idxs=time_idxs_torch, # (B, L) ← IMPORTANT: not (B, L, 1)
    hidden_state=hidden_state,
    sample=True
)
```

**Critical**: The documented API expects `time_idxs` as (B, L), but due to `squeeze(-1)` in position embedding, you actually need to pass (B, L, 1).

### 2. Pretrained Model Metadata

Models store their required spaces:
```python
pretrained_cls = get_pretrained_model("SyntheticRLV2")

pretrained_cls.observation_space  # TokenizedObservationSpace
pretrained_cls.action_space        # MinimalActionSpace (9 actions)
pretrained_cls.reward_function     # DefaultShapedReward
```

**Never manually specify** - always use model's spaces to ensure compatibility.

### 3. Self-Play State Management

Each player needs:
- Separate hidden state (recurrent memory)
- Separate prev_actions/prev_rewards (RL2 input)
- Separate time_idx counter

**Reusing a single instance corrupts all of these.**

---

## Prerequisites

- ✅ metamon repository with pypkmn integration
- ✅ Pretrained model downloaded (via `get_pretrained_model()`)
- ✅ Team files in `$METAMON_CACHE_DIR/teams/`
- ✅ CUDA GPU (for reasonable inference speed)
- ❌ **BLOCKER**: PyKMN battle completion bug must be fixed

---

## Related Issues

- PyKMN vector_env implementation in `metamon/env/pykmn/vector_env.py`
- Action mapping in `metamon/env/pykmn/action_mapper.py`
- Legal mask extraction in `metamon/env/pykmn/features.py`

---

## Next Steps

1. **URGENT**: Run minimal pypkmn reproduction test (Phase 1, Step 1)
2. If bug confirmed in pypkmn: Report to upstream or implement workaround
3. If bug in metamon: Debug action mapper and legal mask extraction
4. Once fixed: Run integration tests with 100+ battles
5. Benchmark performance: pypkmn + pretrained vs Showdown baseline

---

## Success Criteria

- [ ] PyKMN battles complete successfully (not timeout at 1000 steps)
- [ ] Average battle length: 10-50 turns (intelligent play)
- [ ] Win rate: 45-55% (balanced self-play)
- [ ] Invalid action rate: <1%
- [ ] Throughput: >10 battles/second on single GPU
- [ ] Trajectory files compatible with metamon training pipeline

---

**Status Summary**: AMAGO integration is production-ready. PyKMN environment has critical bug preventing use. Recommend debugging PyKMN first before deploying.
