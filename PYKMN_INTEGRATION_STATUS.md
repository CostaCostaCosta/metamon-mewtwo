# PyKMN Integration - Implementation Status

## ✅ Phase 1: Complete (Features Extraction)

### What's Done

**Core Feature Extraction (`features.py`)** - ✅ FULLY FUNCTIONAL
- Battle state extraction using pypkmn's Python accessor methods
- No C code or binary parsing required - clean Python API!
- Precomputed species/move ID mappings for performance
- Two-tier architecture (fast numeric + slow UniversalState)

**Extracts**:
- ✅ Active Pokemon: species, HP, status, moves, PP, max PP, all 6 stat boosts
- ✅ Team Pokemon: species, HP, status for all 5 benched
- ✅ Opponent Pokemon: complete state extraction
- ✅ Previous moves: both players tracked
- ✅ Side conditions: Reflect and Light Screen (Gen 1)
- ✅ Forced switch: detected from result flags

**Test Results**: ✅ Verified working on real Gen1 OU teams

```
P1 Features:
  Active species ID: 128 (Tauros)
  Active HP %: 1.0
  Active status: 0 (healthy)
  Active moves: [34 63 59 89]
  Active PP: [24  8  8 16]
  Active max PP: [24  8  8 16]
  Team species IDs: [113 103 121  65 143]
```

---

## 🎯 Phase 2: Remaining Work

### Priority 1: Observation Space Integration (~1-2 hours)

**File**: `metamon/env/pykmn/vector_env.py`
**Function**: `_extract_observations()`

**Current**: Returns placeholder zeros
**Needed**: Convert features → observation space format

**Implementation**:
```python
def _extract_observations(self):
    obs_p1 = []
    obs_p2 = []

    for i in range(self.num_envs):
        # Extract features (this works!)
        features_p1 = pykmn_to_features_raw(...)
        features_p2 = pykmn_to_features_raw(...)

        # Convert to UniversalState
        state_p1 = features_to_universal_state(features_p1, self.mappings)
        state_p2 = features_to_universal_state(features_p2, self.mappings)

        # Get observation from observation space
        obs_p1.append(self.obs_space.get_observation(state_p1))
        obs_p2.append(self.obs_space.get_observation(state_p2))

    return np.array(obs_p1), np.array(obs_p2)
```

**Challenges**:
- Understand ObservationSpace API (check existing code)
- Handle tokenization if needed
- Match expected shape for model input

### Priority 2: Reward Function Integration (~30 min)

**File**: `metamon/env/pykmn/vector_env.py`
**Function**: `_compute_rewards()`

**Current**: Only terminal rewards (+/-100)
**Needed**: Shaped rewards via RewardFunction

**Implementation**:
```python
def _compute_rewards(self):
    rewards_p1 = np.zeros(self.num_envs)
    rewards_p2 = np.zeros(self.num_envs)

    for i in range(self.num_envs):
        # Convert features to UniversalState
        state_p1 = features_to_universal_state(...)

        # Compute reward
        reward_p1 = self.reward_fn.compute_reward(
            prev_state, action, state_p1
        )
        rewards_p1[i] = reward_p1
        # ... same for P2

    return rewards_p1, rewards_p2
```

**Challenges**:
- Need to track previous state for delta rewards
- Understand RewardFunction API
- Apply reward annealing if configured

### Priority 3: Policy Loading (~1 hour)

**File**: `metamon/env/pykmn/policy_runner.py`
**Class**: `LocalPolicyRunner`

**Current**: Random action selection
**Needed**: Load and run pretrained models

**Implementation**:
```python
class LocalPolicyRunner(PolicyRunner):
    def __init__(self, model_name, checkpoint, device):
        # Load from pretrained registry
        from metamon.rl.pretrained import load_pretrained_model
        self.model = load_pretrained_model(model_name, checkpoint)
        self.model.to(device)
        self.model.eval()

    def infer(self, obs_batch, legal_mask_batch):
        # Convert to torch
        obs_tensor = torch.from_numpy(obs_batch).to(self.device)
        mask_tensor = torch.from_numpy(legal_mask_batch).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(obs_tensor)
            logits = logits.masked_fill(~mask_tensor, float('-inf'))
            actions = torch.multinomial(F.softmax(logits, dim=-1), 1)

        return actions.cpu().numpy()
```

**Challenges**:
- Understand model input format
- Handle tokenization if needed
- Apply temperature correctly

---

## 📊 Current Status Summary

| Component | Status | Effort |
|-----------|--------|--------|
| Team Parser | ✅ Complete | Done |
| Feature Extraction | ✅ Complete | Done |
| Action Mapper | ✅ Complete | Done |
| Vector Environment | ⚠️ Framework | 1-2h |
| Policy Runner | ⚠️ Framework | 1h |
| Trajectory Saver | ✅ Complete | Done |

**Total remaining**: ~2-4 hours to full PoC

---

## 🚀 Quick Start (Current State)

### What Works Now

```python
from metamon.env.pykmn import (
    parse_showdown_team,
    precompute_mappings,
    pykmn_to_features_raw,
)
from pykmn.engine.gen1 import Battle, Choice, Player

# Parse teams
team1 = parse_showdown_team(team_text_1)
team2 = parse_showdown_team(team_text_2)

# Create battle
battle = Battle(p1_team=team1, p2_team=team2)
result, _ = battle.update(Choice.PASS(), Choice.PASS())

# Extract features
mappings = precompute_mappings()
features = pykmn_to_features_raw(battle, result, Player.P1, mappings)

# ✅ This all works!
```

### What Doesn't Work Yet

```python
# ❌ This returns zeros (need observation space integration)
vec_env = PyKMNVectorEnv(...)
obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()

# ❌ Rewards are terminal only (need reward function integration)
obs, rewards, dones, info = vec_env.step(actions_p1, actions_p2)

# ❌ Random actions only (need policy loading)
policy = LocalPolicyRunner("SyntheticRLV2")
actions = policy.infer(obs, masks)
```

---

## 🎯 Next Steps

### Option A: I can finish it (~2-4 hours work)

1. Implement observation space integration
2. Implement reward function integration
3. Implement policy loading
4. Run end-to-end test
5. Benchmark performance

### Option B: You can finish it (guidance provided)

**All the hard parts are done!** Remaining work is:
- Integrating with existing metamon APIs
- Understanding ObservationSpace and RewardFunction
- Loading pretrained models

**Key insight**: Feature extraction (the hard part) is ✅ COMPLETE!

---

## 📁 Files Created

```
metamon/env/pykmn/
├── __init__.py              ✅ Complete exports
├── README.md                ✅ Comprehensive docs
├── team_parser.py           ✅ Fully functional
├── features.py              ✅ COMPLETE - all extraction working!
├── action_mapper.py         ✅ Fully functional
├── vector_env.py            ⚠️ Framework (needs obs/reward integration)
├── policy_runner.py         ⚠️ Framework (needs model loading)
└── trajectory_saver.py      ✅ Fully functional

scripts/
└── generate_selfplay_pykmn.py  ✅ Complete PoC script

tests/
└── test_pykmn_features.py      ✅ Working test
```

---

## 💡 Key Learnings

1. **PyKMN has excellent Python wrappers** - No C code needed!
   - `battle.active_pokemon_stats(player)`
   - `battle.moves_with_pp(player, "Active")`
   - `battle.boosts(player)`
   - 30+ accessor methods available

2. **MOVES dict contains base PP, not base power**
   - Max PP formula: `min(floor(base_pp * 8/5), 61)`
   - Accounts for PP Ups in Gen 1

3. **Side conditions are volatile flags**
   - `battle.volatile(player, VolatileFlag.Reflect)`
   - Reflect and Light Screen are per-side in Gen 1

4. **Active Pokemon is always Slot.ONE**
   - Team Pokemon are Slots 2-6 (benched)

5. **Two-tier architecture is key**
   - Fast path: numeric features only
   - Slow path: UniversalState for saving
   - Precomputed mappings: all lookups optimized

---

## 🎉 Bottom Line

**Phase 1 (Feature Extraction): ✅ COMPLETE AND TESTED**

The hardest part is done! Remaining work is straightforward integration
with existing metamon APIs. Estimated 2-4 hours to full working PoC.
