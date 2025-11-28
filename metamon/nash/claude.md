# Nash Equilibrium Training: Implementation Details

**Last Updated**: November 28, 2025
**Status**: Phase 1 in progress

---

## Current State

### Phase 0: ✅ COMPLETE

**Location**: `/home/eddie/nash_phase0`

**Population** (5 policies):
- SyntheticRLV2 (200M multi-gen RL)
- SyntheticRLV1 (200M earlier version)
- SyntheticRLV1_SelfPlay (fine-tuned on self-play)
- PokeEnvHeuristic (simple heuristic)
- GymLeader (heuristic)

**Nash Equilibrium**:
- σ = [0%, 0%, 100%, 0%, 0%]
- 100% SyntheticRLV1_SelfPlay
- Exploitability = 0.44 (HIGH)

**Key Finding**: V1_SelfPlay beats V2 52%-48% despite V2 being newer. This reveals non-transitive dynamics and high exploitability.

### Phase 1: ⏳ IN PROGRESS

**Location**: `/home/eddie/nash_phase1`

**Status**:
- Infrastructure: ✅ Complete
- Training: Started (iteration_0 exists, but incomplete per summary.json)
- Goal: Complete 5 PSRO iterations
- Target: Reduce exploitability from 0.44 → ~0.1-0.2

---

## Running PSRO Loop

### Recommended: Full 5 Iterations

```bash
python -m metamon.nash.run_psro \
    --phase0_dir ~/nash_phase0 \
    --save_dir ~/nash_phase1 \
    --num_iterations 5 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --battles_per_matchup 200 \
    --collection_battles 500 \
    --oracle_epochs 3 \
    --oracle_batch_size 16 \
    --oracle_model_config synthetic_multitaskagent.gin \
    --oracle_train_config psro_oracle.gin \
    --init_from_checkpoint SyntheticRLV2 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou \
    --parallel_matchups 4 \
    --log
```

**Time**: 5-10 hours total
**Output**: `~/nash_phase1/iteration_0/` through `iteration_4/`
**Expected**: Exploitability drops from 0.44 → ~0.1-0.2

### Test: Single Iteration

For testing the pipeline:

```bash
python -m metamon.nash.run_psro \
    --phase0_dir ~/nash_phase0 \
    --save_dir ~/nash_phase1_test \
    --num_iterations 1 \
    --collection_battles 100 \
    --oracle_epochs 1 \
    --oracle_model_config synthetic_multitaskagent.gin \
    --oracle_train_config psro_oracle.gin \
    --init_from_checkpoint SyntheticRLV2 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou
```

**Time**: ~1 hour
**Purpose**: Verify pipeline works before committing to full run

---

## Implementation Approach

### Sequential Data Collection + Offline Training

**Why This Approach?**
- Simple and reliable
- No username collision issues
- Easy to debug and monitor
- Works well with checkpoint initialization

#### Step 1: Data Collection (`collect_psro_data.py`)

- Samples opponents from meta-strategy distribution
- Runs sequential 1v1 battles (500 battles per iteration)
- Saves trajectories to disk
- **Time**: ~30-60 minutes per iteration

#### Step 2: Offline BR Training (`metamon.rl.train`)

- Initializes from SyntheticRLV2 checkpoint
- Trains on 50% collected data + 50% human replays
- 3 epochs (fast PSRO iteration)
- **Time**: ~1-2 hours per iteration

### Key Configuration

**Model Architecture** (CRITICAL):
- Must use `synthetic_multitaskagent.gin`
- 5-layer TstepEncoder, 9-layer TrajEncoder
- 200M parameters
- Matches SyntheticRLV2 exactly (required for checkpoint loading)

**Training Config**:
- `psro_oracle.gin` (not `binary_rl.gin`)
- LR = 3e-5 (lower than standard for fine-tuning)
- Optimized for PSRO iteration speed

**Data Sources**:
- Collected trajectories: `~/nash_phase1/iteration_N/br_trajectories/`
- Human replays: `~/metamon_cache/parsed-replays/gen1ou` (175k battles)
- Mix ratio: 50/50 (`--custom_replay_sample_weight 0.5`)

**Base Model**:
- SyntheticRLV2 (strongest pretrained model)
- Epoch 48 checkpoint
- 200M parameters, trained on 9 generations
- **Do not fine-tune on Gen1** (see LESSONS_LEARNED.md)

---

## Module Reference

### Core Scripts

**PSRO Loop**:
- `run_psro.py` - Main PSRO driver (orchestrates full loop)
- `collect_psro_data.py` - Sequential data collection for BR training
- `compute_matrix.py` - Tournament runner and Nash solver

**Population & Nash**:
- `population.py` - PolicyPopulation class (manage Π)
- `interaction_matrix.py` - InteractionMatrix class (compute/store M)
- `solver.py` - Nash equilibrium solver (LP-based)

**Environment** (deprecated):
- `psro_env.py` - Environment with opponent sampling (not used - using sequential approach)
- `train_psro_oracle.py` - Online RL oracle training (not used - using offline approach)

### PSRO Loop Flow

```
Iteration t:
  1. Load σ_{t-1} (meta-strategy from previous iteration)
  2. Collect data (sequential 1v1 battles vs population, weighted by σ_{t-1})
  3. Train BR_t (offline RL: 50% collected + 50% replay, init from SyntheticRLV2)
  4. Add to population (Π_t ← Π_{t-1} ∪ {BR_t})
  5. Run tournament (compute M_t: K×K interaction matrix)
  6. Solve Nash (σ_t ← solve_nash_mixture(M_t))
  7. Log & repeat
```

### Output Structure

```
nash_phase1/
├── iteration_0/
│   ├── br_trajectories/gen1ou/           # Collected battle data
│   │   ├── battle_001.jsonl
│   │   └── ...
│   ├── br_training/
│   │   └── PSRO_BR0/
│   │       ├── ckpts/
│   │       │   ├── epoch_0.pt
│   │       │   ├── epoch_1.pt
│   │       │   ├── epoch_2.pt
│   │       │   └── epoch_3.pt            # Use this checkpoint
│   │       └── wandb_logs/
│   ├── population.json                    # Updated population (with BR_0)
│   ├── interaction_matrix.json            # M_0 (win-rates)
│   ├── meta_strategy.json                 # σ_0 (Nash mixture)
│   ├── meta_game_analysis.json            # Exploitability, dominant policies
│   └── iteration_summary.json
├── iteration_1/
│   └── ... (same structure)
├── ...
└── psro_summary.json                      # Overall summary
```

---

## Expected Progression

### Iteration 0

**Input**: Phase 0 results (σ = 100% V1_SelfPlay, Exploit = 0.44)
**Action**: Train BR_0 to beat V1_SelfPlay
**Expected**: BR_0 wins 60-70% vs V1_SelfPlay
**Output**: σ_0 ≈ [0%, 0%, 60%, 0%, 0%, 40%] (V1_SP + BR_0)

### Iteration 1

**Input**: σ_0 (mixed V1_SP and BR_0)
**Action**: Train BR_1 to beat mixture
**Expected**: BR_1 wins 60%+ vs σ_0
**Output**: σ_1 spreads across 3-4 policies

### Iteration 5

**Goal**: Exploitability < 0.15
**Nash**: Broadly distributed across multiple BRs
**Population**: 10 policies (5 original + 5 BRs)

---

## Troubleshooting

### Architecture Mismatch Error

**Symptom**: "Expected tensor shape [X] but got [Y]"

**Cause**: Wrong gin config (doesn't match SyntheticRLV2)

**Fix**: Must use `synthetic_multitaskagent.gin`

### Checkpoint Not Found

**Symptom**: "Cannot load checkpoint from SyntheticRLV2"

**Fix**:
```bash
# Download model
python -c "from metamon.rl.pretrained import get_pretrained_model; get_pretrained_model('SyntheticRLV2')"
```

### BR Not Beating Nash

**Symptom**: BR_0 only wins 50-55% vs V1_SelfPlay (expected: 60-70%)

**Possible causes**:
1. Not enough training epochs (increase to 5)
2. Not enough data collection (increase to 1000 battles)
3. Wrong opponent sampling (check meta-strategy file)

**Debug**:
```bash
# Check collected data distribution
ls ~/nash_phase1/iteration_0/br_trajectories/gen1ou/*.jsonl | wc -l
```

### Exploitability Not Decreasing

**Symptom**: After 3 iterations, still > 0.30

**Fixes**:
1. Increase tournament size (`--battles_per_matchup 400`)
2. Check BRs are being added to population
3. Check Nash is updating (σ should spread)
4. Run more iterations (need 5+ for convergence)

---

## Documentation

**User-Facing**:
- **`/NASH.md`**: Complete roadmap (Phases 0-4)
- **`/PSRO_GUIDE.md`**: Step-by-step training guide
- **`/LESSONS_LEARNED.md`**: Key findings from experiments

**Technical**:
- **`metamon/nash/README.md`**: Package documentation (this is for developers)
- **`metamon/nash/claude.md`**: Implementation details (this file)

**Historical** (for reference):
- `/PHASE0_IMPLEMENTATION_SUMMARY.md`: Phase 0 details
- `/PHASE1_IMPLEMENTATION_SUMMARY.md`: Phase 1 initial design
- `/P0_FINETUNING_EXPERIMENT_RESULTS.md`: Why fine-tuning failed
- `/Gen1_BinaryReward_Training_Summary.md`: Why sparse rewards failed

---

## Next Actions

### Immediate

1. ✅ Documentation consolidated
2. ⏳ Complete 5 PSRO iterations
3. ⏳ Validate exploitability trend

### Short-term (Phase 1 Completion)

1. ⏳ Analyze full population (after iteration 5)
2. ⏳ Deploy Nash mixture for evaluation
3. ⏳ Write Phase 1 completion report

### Medium-term (Phase 2)

1. 📋 Design BR/Average head split (NFSP)
2. 📋 Implement behavior cloning for average strategy
3. 📋 Set up human evaluation infrastructure

---

**Last Updated**: November 28, 2025
**Current Phase**: Phase 1 (PSRO) - In Progress
**Next Milestone**: Complete 5 iterations, achieve Exploitability < 0.15
