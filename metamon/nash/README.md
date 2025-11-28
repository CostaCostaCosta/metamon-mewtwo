# Nash Equilibrium Training (PSRO)

## Overview

This package implements **Policy Space Response Oracles (PSRO)** for finding Nash equilibrium policies in Gen 1 OU Pokémon battles. The system iteratively trains best-response policies against a population mixture to minimize exploitability and achieve superhuman performance.

**Paper Reference**: "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (Lanctot et al. 2017)

## Current Status

### Phase 0: Baseline & Infrastructure ✅ COMPLETE
- **Location**: `/home/eddie/nash_phase0`
- **Population**: 5 policies (SyntheticRLV2, V1, V1_SelfPlay, PokeEnvHeuristic, GymLeader)
- **Nash Equilibrium**: 100% SyntheticRLV1_SelfPlay
- **Exploitability**: 0.44 (high - strong best-response exists)
- **Key Finding**: V1_SelfPlay is current Nash but highly exploitable by best-response

### Phase 1: PSRO Loop ⏳ IN PROGRESS
- **Location**: `/home/eddie/nash_phase1`
- **Status**: Infrastructure complete, training in progress
- **Goal**: Reduce exploitability from 0.44 → ~0.1-0.2 through iterative best-response training
- **Next Step**: Complete 5 PSRO iterations to build stronger population

## Quick Start

### Run Full PSRO Loop (5 iterations)

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
    --oracle_model_config synthetic_multitaskagent.gin \
    --oracle_train_config psro_oracle.gin \
    --init_from_checkpoint SyntheticRLV2 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou \
    --log
```

**Time**: ~5-10 hours for 5 iterations
**Hardware**: GPU recommended (tested on single GPU)

## Package Structure

### Core Modules

- **`population.py`** - Manage policy populations (Π)
- **`interaction_matrix.py`** - Compute/store empirical win-rate matrices (M)
- **`solver.py`** - Solve for Nash equilibrium mixtures (σ) via linear programming
- **`run_psro.py`** - Orchestrate full PSRO loop (driver script)
- **`collect_psro_data.py`** - Collect training data via sequential matchups
- **`compute_matrix.py`** - Run tournaments and compute interaction matrices
- **`psro_env.py`** - Environment wrappers for opponent sampling (deprecated - using sequential approach)
- **`train_psro_oracle.py`** - RL oracle training (deprecated - using offline approach)

### PSRO Loop Overview

```
Iteration t:
  1. Load population Π_{t-1} and meta-strategy σ_{t-1}
  2. Collect data: Sequential 1v1 battles vs population weighted by σ_{t-1}
  3. Train BR_t: Offline RL on collected + replay data (initialized from SyntheticRLV2)
  4. Add to population: Π_t ← Π_{t-1} ∪ {BR_t}
  5. Run tournament: Compute M_t (interaction matrix)
  6. Solve Nash: σ_t ← solve_nash_mixture(M_t)
  7. Repeat
```

## Key Design Decisions

### Sequential Data Collection (Current Approach)
**Advantages**:
- Simple and reliable
- No username collision issues
- Easy to debug and monitor
- Works well with offline training

**Implementation**: `collect_psro_data.py`
- Samples opponents from meta-strategy
- Runs 1v1 matchups sequentially
- Saves trajectories for offline training

### Offline BR Training with Checkpoint Initialization
**Advantages**:
- Leverages strong pretrained models (SyntheticRLV2)
- Mixes collected data with human replays (50/50)
- No parallel training complexity
- Fast iteration (3 epochs per BR)

**Key Config**: `synthetic_multitaskagent.gin`
- Must match SyntheticRLV2 architecture exactly
- 5-layer TstepEncoder, 9-layer TrajEncoder
- Critical for proper checkpoint loading

## Phase 0 Results

### Interaction Matrix (Win Rates)
```
                        V2    V1   V1_SP  Heur  Leader
SyntheticRLV2         0.50  0.62  0.48   0.96  1.00
SyntheticRLV1         0.38  0.50  0.38   0.90  0.88
SyntheticRLV1_SP      0.52  0.62  0.50   0.86  0.94  ← Nash
PokeEnvHeuristic      0.04  0.10  0.14   0.50  0.50
GymLeader             0.00  0.12  0.06   0.50  0.50
```

### Key Insights
1. **V1_SelfPlay beats V2** (52% win rate) - unexpected!
2. **High exploitability** (0.44) - strong best-response exists
3. **Nash is unique** - 100% mass on single policy indicates dominant strategy
4. **Heuristics are weak** - all RL models dominate (90%+ win rates)

## Expected Progression

### After 5 PSRO Iterations

**Population Growth**:
```
Iter 0: {V2, V1, V1_SP, Heur, Leader}                → 5 policies
Iter 1: {V2, V1, V1_SP, Heur, Leader, BR_0}          → 6 policies
Iter 5: {V2, V1, V1_SP, Heur, Leader, BR_0-4}        → 10 policies
```

**Nash Mixture Evolution**:
```
Iter 0: σ = [0%, 0%, 100%, 0%, 0%]           Exploit = 0.44
Iter 1: σ = [0%, 0%, 60%, 0%, 0%, 40%]       Exploit = 0.35
Iter 3: σ = [15%, 0%, 15%, 0%, 0%, 20%, 25%, 25%]  Exploit = 0.15
Iter 5: σ = [mixed over multiple BRs]         Exploit = ~0.10
```

**Goal**: Exploitability < 0.15 (strong Nash approximation)

## Configuration Reference

### Model Configs (`metamon/rl/configs/models/`)
- **`synthetic_multitaskagent.gin`** ✅ Required for SyntheticRLV2
- **`small_agent.gin`** - Alternative smaller architecture
- **`medium_agent.gin`** - Alternative medium architecture

### Training Configs (`metamon/rl/configs/training/`)
- **`psro_oracle.gin`** ✅ Optimized for PSRO (LR=3e-5, offline-only)
- **`binary_rl.gin`** - Standard offline RL (not for PSRO)

### Key Parameters

**PSRO Loop**:
- `--num_iterations`: Number of PSRO iterations (5 recommended for Phase 1)
- `--collection_battles`: Battles to collect per iteration (500 recommended)
- `--battles_per_matchup`: Battles per tournament matchup (200 for Phase 1, 400 for Phase 2+)

**Oracle Training**:
- `--oracle_epochs`: Training epochs per BR (3 for fast iteration, 5 for quality)
- `--init_from_checkpoint`: Base model (SyntheticRLV2 required)
- `--oracle_model_config`: Must match base model architecture!
- `--custom_replay_sample_weight`: Mix ratio (0.5 = 50/50 collected+replay)

## Lessons Learned (From Phase 0 Experiments)

### ❌ What Didn't Work

1. **Offline Fine-tuning on Gen1 Replays** (`P0_FINETUNING_EXPERIMENT_RESULTS.md`)
   - Fine-tuned SyntheticRLV2 → worse performance (38% vs 62%)
   - Human replay data is lower quality than RL self-play
   - Multi-gen knowledge is valuable - don't discard it

2. **BinaryReward Training** (`Gen1_BinaryReward_Training_Summary.md`)
   - Sparse rewards caused flat loss curves (no learning)
   - Model couldn't adapt from shaped → sparse rewards
   - Stick with DefaultShapedReward for now

### ✅ What Works

1. **Use SyntheticRLV2 directly** - strongest general model
2. **Self-play > human replays** - for continued improvement
3. **Sequential data collection** - simpler than parallel online RL
4. **Mix collected + replay data** - 50/50 provides stability

## Next Steps

### Complete Phase 1 (Current)
- Run 5 PSRO iterations to build population
- Monitor exploitability convergence
- Validate BRs are beating previous Nash

### Phase 2: NFSP (Neural Fictitious Self-Play)
- Split policy into BR head (aggressive) and Average head (stable)
- Train average via behavior cloning
- Deploy average as main agent

### Phase 3: Exploitability Descent
- Add explicit exploiter policy
- Two-player min-max training loop
- Minimize exploitability directly

## Troubleshooting

### PSRO training fails
- **Check**: SyntheticRLV2 checkpoint accessible?
- **Check**: Using `synthetic_multitaskagent.gin`?
- **Check**: Parsed replays downloaded? (`~/metamon_cache/parsed-replays`)

### BR not beating Nash
- **Increase epochs**: 3 → 5
- **More data collection**: 500 → 1000 battles
- **Check opponent distribution**: Review sampled opponents

### Out of memory
- **Reduce batch size**: `--oracle_batch_size 8` (default: 16)
- **Fewer epochs**: Keep at 3
- **Check GPU memory**: `nvidia-smi`

## Documentation

- **This file**: Package overview and quick reference
- **`claude.md`**: Detailed implementation notes and status
- **Root `NASH.md`**: High-level Nash-first training plan
- **Root `PSRO_GUIDE.md`**: Step-by-step user guide

## References

- Lanctot et al. 2017, "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"
- Heinrich et al. 2015, "Fictitious Self-Play in Extensive-Form Games"
- Balduzzi et al. 2019, "Open-ended Learning in Symmetric Zero-sum Games"
