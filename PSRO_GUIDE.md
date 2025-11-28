# PSRO Training Guide: Step-by-Step Instructions

**Goal**: Train Nash equilibrium policies for Gen 1 OU using Policy Space Response Oracles (PSRO).

**Current Status**: Phase 1 in progress - building population through iterative best-response training.

---

## Prerequisites

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Verify parsed replays exist
ls $METAMON_CACHE_DIR/parsed-replays/gen1ou
```

### 2. Local Showdown Server

**Required for data collection**:
```bash
# In separate terminal
cd server/pokemon-showdown
node pokemon-showdown start --no-security
```

### 3. Phase 0 Complete

Verify Phase 0 results exist:
```bash
ls ~/nash_phase0/population.json
ls ~/nash_phase0/meta_strategy.json
ls ~/nash_phase0/interaction_matrix.json
```

If Phase 0 not complete, see "Setting Up Phase 0" section below.

---

## Quick Start: Run Full PSRO Loop

**Recommended for most users** - runs 5 PSRO iterations automatically:

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

**Time**: 5-10 hours (can run overnight)
**Output**: `~/nash_phase1/iteration_0/` through `iteration_4/`

---

## Step-by-Step: Manual PSRO Iterations

For more control, run iterations manually:

### Iteration 0: Train First Best-Response

#### Step 1: Collect Training Data

```bash
python -m metamon.nash.collect_psro_data \
    --run_name PSRO_BR0 \
    --population_file ~/nash_phase0/population.json \
    --meta_strategy_file ~/nash_phase0/meta_strategy.json \
    --save_dir ~/nash_phase1/iteration_0/br_trajectories \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --num_battles 500 \
    --init_from_checkpoint SyntheticRLV2 \
    --obs_space DefaultObservationSpace \
    --reward_function DefaultShapedReward \
    --action_space MinimalActionSpace \
    --tokenizer allreplays-v3
```

**What happens**:
- Samples 500 opponents from meta-strategy (100% V1_SelfPlay from Phase 0)
- Runs sequential 1v1 battles
- Saves trajectories to disk (~30-60 minutes)

#### Step 2: Train Best-Response Offline

```bash
python -m metamon.rl.train \
    --run_name PSRO_BR0 \
    --save_dir ~/nash_phase1/iteration_0/br_training \
    --epochs 3 \
    --batch_size_per_gpu 16 \
    --model_gin_config synthetic_multitaskagent.gin \
    --train_gin_config binary_rl.gin \
    --obs_space DefaultObservationSpace \
    --reward_function DefaultShapedReward \
    --action_space MinimalActionSpace \
    --tokenizer allreplays-v3 \
    --custom_replay_dir ~/nash_phase1/iteration_0/br_trajectories \
    --custom_replay_sample_weight 0.5 \
    --init_from_checkpoint SyntheticRLV2 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou \
    --log
```

**What happens**:
- Initializes from SyntheticRLV2 checkpoint
- Trains on 50% collected data + 50% human replays
- 3 epochs (~30-60 minutes per epoch)
- Saves checkpoint: `~/nash_phase1/iteration_0/br_training/PSRO_BR0/ckpts/epoch_3.pt`

#### Step 3: Evaluate BR_0

```bash
# Test vs Nash (V1_SelfPlay)
python -m metamon.rl.evaluate \
    --model_path ~/nash_phase1/iteration_0/br_training/PSRO_BR0/ckpts/epoch_3.pt \
    --opponent SyntheticRLV1_SelfPlay \
    --num_battles 100 \
    --battle_format gen1ou \
    --team_set competitive
```

**Expected**: BR_0 should beat V1_SelfPlay 60-70% (exploiting Nash)

#### Step 4: Update Population & Nash

```bash
# Manually add BR_0 to population.json, then:
python -m metamon.nash.compute_matrix \
    --population_file ~/nash_phase1/iteration_0/population.json \
    --battles_per_matchup 200 \
    --battle_format gen1ou \
    --team_set competitive \
    --output_dir ~/nash_phase1/iteration_0 \
    --parallel_matchups 4
```

**What happens**:
- Runs tournament with BR_0 included
- Computes new interaction matrix M_1
- Solves for new Nash σ_1
- Nash mixture should now split between V1_SelfPlay and BR_0

### Iteration 1-4: Repeat

Use updated `~/nash_phase1/iteration_0/` files as input for iteration 1, and so on.

---

## Monitoring Progress

### During Training

**Check data collection**:
```bash
# Count collected battles
ls ~/nash_phase1/iteration_0/br_trajectories/gen1ou/*.jsonl | wc -l
```

**Monitor training** (if using `--log`):
- Check wandb dashboard
- Watch for decreasing losses
- Monitor win rates vs validation opponents

**Check GPU usage**:
```bash
watch -n 1 nvidia-smi
```

### After Each Iteration

**Analyze results**:
```bash
# View Nash mixture
cat ~/nash_phase1/iteration_N/meta_strategy.json

# View exploitability
python3 << 'EOF'
import json
with open('~/nash_phase1/iteration_N/meta_game_analysis.json') as f:
    data = json.load(f)
print(f"Exploitability: {data['exploitability']:.3f}")
print(f"Nash mixture:")
for p in data['dominant_policies']:
    print(f"  {p['policy']}: {p['probability']:.1%}")
EOF
```

**Expected progression**:
```
Iter 0: Exploit = 0.44, Nash = 100% V1_SelfPlay
Iter 1: Exploit ≈ 0.35, Nash = ~60% V1_SP + ~40% BR_0
Iter 2: Exploit ≈ 0.25, Nash = mixed across 3-4 policies
Iter 3: Exploit ≈ 0.18, Nash = mixed across 4-5 policies
Iter 5: Exploit ≈ 0.10, Nash = broadly distributed
```

---

## Configuration Deep Dive

### Critical Parameters

#### Model Architecture (`--oracle_model_config`)

**Must use `synthetic_multitaskagent.gin`** to match SyntheticRLV2:
- 5-layer TstepEncoder
- 9-layer TrajEncoder
- 200M parameters
- Mismatch will prevent checkpoint loading!

#### Training Config (`--oracle_train_config`)

**Use `psro_oracle.gin`** (not `binary_rl.gin`):
- Lower learning rate (3e-5 vs 1.5e-4)
- Optimized for fine-tuning from strong checkpoint
- No warmup needed (already trained)

#### Data Mixing (`--custom_replay_sample_weight`)

**0.5 = 50/50 collected + replay**:
- Collected data: specific to current meta-strategy
- Replay data: provides stability and diversity
- Lower (0.3): more conservative, slower adaptation
- Higher (0.7): more aggressive, faster exploitation

#### Iteration Speed (`--oracle_epochs`)

**3 epochs recommended for Phase 1**:
- Fast PSRO cycles (iterate quickly)
- Each BR doesn't need to be perfect
- Quality improves through population growth

**5 epochs for Phase 2+**:
- Higher quality BRs
- Slower iteration but stronger policies

### Team Sets

**`modern_replays_v2`** (recommended):
- ~100s of diverse teams from replays
- Realistic meta distribution
- Good for training generalists

**`competitive`**:
- ~20 hand-crafted competitive teams
- Higher quality but less diverse
- Good for final evaluation

**`paper_variety`**:
- Paper's original evaluation set
- For comparison with published results

---

## Troubleshooting

### Issue: Training crashes with "checkpoint not found"

**Cause**: SyntheticRLV2 not downloaded or wrong path

**Fix**:
```bash
# Download from HuggingFace
python -c "from metamon.rl.pretrained import get_pretrained_model; get_pretrained_model('SyntheticRLV2')"

# Verify cache
ls ~/metamon_cache/models/SyntheticRLV2
```

### Issue: "Architecture mismatch" error

**Cause**: Using wrong gin config (doesn't match SyntheticRLV2)

**Fix**: Must use `synthetic_multitaskagent.gin`, not `small_agent.gin` or others

### Issue: BR_0 not beating Nash (only 50-55% win rate)

**Possible causes**:
1. Not enough training epochs → increase to 5
2. Not enough data collection → increase to 1000 battles
3. Data distribution mismatch → check opponent sampling

**Debug**:
```bash
# Verify opponent distribution in collected data
grep -r "opponent_name" ~/nash_phase1/iteration_0/br_trajectories/
# Should match meta-strategy (100% V1_SelfPlay for iter 0)
```

### Issue: Out of memory (CUDA OOM)

**Causes**: Batch size too large, model too big for GPU

**Fixes**:
```bash
# Reduce batch size
--batch_size_per_gpu 8  # (default: 16)

# Reduce data loading workers
--dloader_workers 4  # (default: 8)

# Check GPU memory
nvidia-smi
```

### Issue: Showdown server crashes during data collection

**Cause**: Too many concurrent battles, server overload

**Fix**:
```bash
# Restart server
cd server/pokemon-showdown
node pokemon-showdown start --no-security

# Reduce collection speed
# Edit collect_psro_data.py, add delays between battles
```

### Issue: Exploitability not decreasing

**After 3 iterations, still > 0.30**:

1. **Check BRs are being added**: Verify population size increases each iteration
2. **Check Nash is updating**: σ should spread across multiple policies
3. **Increase tournament size**: `--battles_per_matchup 400`
4. **Run more iterations**: Need 5+ for convergence

---

## Setting Up Phase 0 (If Not Complete)

### Create Initial Population

```bash
python -m metamon.nash.compute_matrix \
    --create_population \
    --population_file ~/nash_phase0/population.json \
    --pretrained_models SyntheticRLV2 SyntheticRLV1 SyntheticRLV1_SelfPlay \
    --heuristic_baselines PokeEnvHeuristic GymLeader \
    --battles_per_matchup 50 \
    --battle_format gen1ou \
    --team_set competitive \
    --output_dir ~/nash_phase0 \
    --parallel_matchups 4
```

**Time**: 2-3 hours
**Output**: Population + interaction matrix + Nash equilibrium

**Expected Nash**: 100% SyntheticRLV1_SelfPlay with exploitability ≈ 0.44

---

## Next Steps After Phase 1

### Analyze Full Population

After 5 iterations, analyze the meta-game:

```bash
# Plot exploitability over time
python3 << 'EOF'
import json
import matplotlib.pyplot as plt

exploits = []
for i in range(5):
    with open(f'~/nash_phase1/iteration_{i}/meta_game_analysis.json') as f:
        exploits.append(json.load(f)['exploitability'])

plt.plot(exploits, marker='o')
plt.xlabel('PSRO Iteration')
plt.ylabel('Exploitability')
plt.title('PSRO Convergence')
plt.savefig('psro_exploitability.png')
print("Saved plot to psro_exploitability.png")
EOF
```

### Deploy Nash Mixture

Create evaluator that samples from σ_5:

```bash
# Evaluate Nash mixture vs baselines
python scripts/evaluate_nash_mixture.py \
    --population_file ~/nash_phase1/iteration_4/population.json \
    --meta_strategy_file ~/nash_phase1/iteration_4/meta_strategy.json \
    --opponents PokeEnvHeuristic GymLeader Gen1BossAI \
    --num_battles 200
```

### Proceed to Phase 2: NFSP

See `NASH.md` for Phase 2 plan (Neural Fictitious Self-Play).

---

## Summary

### Quickest Path

1. **Verify Phase 0**: Check `~/nash_phase0` has results
2. **Run PSRO**: `python -m metamon.nash.run_psro [args]`
3. **Wait 5-10 hours**: Let it complete 5 iterations
4. **Analyze results**: Check exploitability trend
5. **Proceed to Phase 2**: If exploitability < 0.15

### Time Investment

| Task | Duration | Can Run Overnight? |
|------|----------|-------------------|
| Phase 0 setup | 2-3 hours | ✅ Yes |
| Single PSRO iteration | 1-2 hours | ✅ Yes |
| 5 PSRO iterations | 5-10 hours | ✅ Yes |
| Validation & analysis | 30 min | ❌ No (manual) |

### Success Criteria

Phase 1 is successful if after 5 iterations:
- ✅ Exploitability < 0.15 (down from 0.44)
- ✅ Nash mixture spreads across multiple policies
- ✅ Each BR beats previous Nash (60%+ win rate)
- ✅ Population size = 10 policies (5 original + 5 BRs)

---

**Questions?** See `metamon/nash/README.md` for technical details or `NASH.md` for the full roadmap.
