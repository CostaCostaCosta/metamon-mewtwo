# Nash Equilibrium Training: Phase-by-Phase Plan

**Goal**: Develop superhuman Gen 1 OU Pokémon agent using Nash-first approach via Policy Space Response Oracles (PSRO).

**Key Principle**: Never use search (no MCTS/tree search). Instead, train neural policies to approximate Nash equilibria through iterative best-response training and population-based learning.

---

## Overview

This document outlines the complete Nash-first training roadmap from Phase 0 (baseline) through Phase 4 (superhuman evaluation). The approach is based on the unified game-theoretic framework from "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (Lanctot et al. 2017).

**Core Components**:
1. **Policy Population (Π)**: Set of diverse strategies
2. **Interaction Matrix (M)**: Empirical win-rates between all policy pairs
3. **Meta-Strategy (σ)**: Nash equilibrium mixture over population
4. **RL ORACLE**: Train best-responses to σ via reinforcement learning (no search)

---

## Phase 0: Nash Framing + Gen 1 Baseline ✅ COMPLETE

**Duration**: 1-2 weeks
**Status**: ✅ Complete
**Location**: `/home/eddie/nash_phase0`

### Goals Achieved

1. ✅ Created policy population infrastructure
2. ✅ Implemented interaction matrix computation
3. ✅ Built Nash equilibrium solver (linear programming)
4. ✅ Established Gen 1 OU baseline population
5. ✅ Measured initial meta-game state

### Results

**Population**: 5 policies
- SyntheticRLV2 (200M multi-gen RL)
- SyntheticRLV1 (200M earlier version)
- SyntheticRLV1_SelfPlay (fine-tuned on self-play)
- PokeEnvHeuristic (simple max-damage heuristic)
- GymLeader (medium-strength heuristic)

**Nash Equilibrium**: 100% SyntheticRLV1_SelfPlay
**Exploitability**: 0.44 (HIGH - strong best-response exists)

**Key Finding**: V1_SelfPlay beats V2 52%-48% despite V2 being "stronger" on paper. This reveals non-transitive dynamics and the importance of Nash analysis.

### Infrastructure Built

**Code** (`metamon/nash/`):
- `population.py` - PolicyPopulation class
- `interaction_matrix.py` - InteractionMatrix class
- `solver.py` - Nash solver using scipy LP
- `compute_matrix.py` - Tournament runner
- `run_psro.py` - PSRO driver script
- `collect_psro_data.py` - Data collection for BR training

**Output Files**:
- `population.json` - Policy registry
- `interaction_matrix.json` - M (win-rates)
- `meta_strategy.json` - σ (Nash mixture)
- `meta_game_analysis.json` - Exploitability, dominance

### Lessons Learned

1. ❌ **Fine-tuning on Gen1 replays hurt performance**
   - P0_SYN_V2_GEN1 (Gen1 specialist) lost to base SyntheticRLV2 38-62%
   - Multi-gen knowledge is valuable
   - Human replays are lower quality than RL self-play

2. ❌ **BinaryReward too sparse**
   - Flat loss curves, no learning signal
   - Model couldn't adapt from shaped → sparse rewards
   - Stick with DefaultShapedReward

3. ✅ **Nash framework works with general models**
   - No need for domain-specific fine-tuning
   - Population-based approach is the path forward

---

## Phase 1: PSRO-Lite with RL ORACLE ⏳ IN PROGRESS

**Duration**: 4-6 weeks
**Status**: ⏳ Infrastructure complete, training in progress
**Location**: `/home/eddie/nash_phase1`

### Goals

1. Implement RL ORACLE for best-response training
2. Run iterative PSRO loop (5 iterations minimum)
3. Reduce exploitability from 0.44 → ~0.1-0.2
4. Build population of diverse strong policies

### PSRO Loop Structure

```
For iteration t = 0, 1, 2, ..., T:

  1. Load meta-strategy σ_{t-1} from previous iteration

  2. Collect Training Data:
     - Sample opponents from population weighted by σ_{t-1}
     - Run sequential 1v1 battles (500 battles)
     - Save trajectories to disk

  3. Train Best-Response BR_t:
     - Initialize from SyntheticRLV2 checkpoint
     - Offline RL on collected + replay data (50/50 mix)
     - 3 epochs (fast iteration)
     - Use synthetic_multitaskagent.gin (matches V2 architecture)

  4. Add to Population:
     - Π_t ← Π_{t-1} ∪ {BR_t}

  5. Run Tournament:
     - Round-robin: all pairs play 200 battles
     - Compute M_t (K×K interaction matrix)

  6. Solve Nash:
     - σ_t ← solve_nash_mixture(M_t)
     - Analyze exploitability

  7. Log Progress:
     - Nash mixture evolution
     - Exploitability trend
     - BR win-rates vs population
```

### Implementation Approach

**Sequential Data Collection** (Current):
- Simple and reliable
- `collect_psro_data.py` samples opponents and runs 1v1 matchups
- No username collision issues
- Works well with offline training

**Offline BR Training**:
- Leverages SyntheticRLV2 (strongest pretrained model)
- Mixes collected data (50%) + human replays (50%)
- 3 epochs per iteration for fast PSRO cycles
- Uses `psro_oracle.gin` (LR=3e-5, optimized for fine-tuning)

### Key Configuration

**Critical**: Must use `synthetic_multitaskagent.gin` to match SyntheticRLV2 architecture:
- 5-layer TstepEncoder
- 9-layer TrajEncoder
- 200M parameters

**Training Command**:
```bash
python -m metamon.nash.run_psro \
    --phase0_dir ~/nash_phase0 \
    --save_dir ~/nash_phase1 \
    --num_iterations 5 \
    --oracle_model_config synthetic_multitaskagent.gin \
    --oracle_train_config psro_oracle.gin \
    --init_from_checkpoint SyntheticRLV2 \
    --collection_battles 500 \
    --oracle_epochs 3 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou
```

### Expected Outcomes

**After 5 iterations**:

**Population Growth**:
```
t=0: {V2, V1, V1_SP, Heur1, Heur2}                 → 5 policies
t=5: {V2, V1, V1_SP, Heur1, Heur2, BR_0-4}        → 10 policies
```

**Nash Mixture Evolution**:
```
t=0: σ = [0%, 0%, 100%, 0%, 0%]              Exploit = 0.44
t=1: σ = [0%, 0%, 60%, 0%, 0%, 40%]          Exploit ≈ 0.35
t=3: σ = [15%, 0%, 15%, 0%, 0%, 20%, 30%, 20%]    Exploit ≈ 0.15
t=5: σ = [mixed across multiple BRs]         Exploit ≈ 0.10
```

**Success Criteria**:
- ✅ Exploitability < 0.15
- ✅ Nash mixture spreads across multiple policies
- ✅ Each BR beats previous Nash mixture (60%+ win rate)
- ✅ No single policy dominates

### Current Status

- Infrastructure: ✅ Complete
- Training: ⏳ In progress
- Next: Complete 5 PSRO iterations

---

## Phase 2: NFSP (Neural Fictitious Self-Play)

**Duration**: 6-8 weeks
**Status**: 📋 Planned
**Prerequisites**: Phase 1 complete

### Goals

1. Split policy into BR head and Average head
2. Train average strategy via behavior cloning
3. Deploy average as stable Nash approximation
4. Reduce exploitability further (0.10 → 0.05)

### Approach

**Dual-Head Architecture**:

1. **BR Head (π_BR)**: Aggressive best-response
   - Trained via RL (same as Phase 1)
   - Maximizes win-rate vs current meta-strategy
   - Exploits weaknesses in population

2. **Average Head (π_avg)**: Stable equilibrium strategy
   - Trained via supervised behavior cloning
   - Learns from BR's trajectory buffer
   - Approximates time-average strategy (Fictitious Play)

**Training Loop**:
```
For iteration t:
  1. Train π_BR vs meta-strategy σ_t (RL)
  2. Collect BR trajectories to behavior buffer
  3. Train π_avg via BC on buffer (supervised)
  4. Add π_avg to population
  5. Recompute σ_{t+1}
```

### Why NFSP?

**Theory**: Fictitious Play converges to Nash in many games. NFSP approximates this via:
- BR explores/exploits current meta
- Average accumulates stable mixed strategy
- Result: less exploitable than any single BR

**Practical Benefits**:
- π_avg is more robust (less cyclic behavior)
- π_BR adapts quickly to new opponents
- Separates exploration (BR) from deployment (avg)

### Expected Outcomes

**Metrics**:
- π_avg beats all Phase 1 policies consistently
- π_avg has fewer weaknesses (more balanced)
- Exploitability: 0.10 → 0.05

**Population**:
- 10-15 policies (BRs + averages)
- Nash mixture balances BR and avg policies

---

## Phase 3: Exploitability Descent (ApproxED-lite)

**Duration**: 6-8 weeks
**Status**: 📋 Planned
**Prerequisites**: Phase 2 complete

### Goals

1. Add explicit exploiter policy π_exp
2. Two-player min-max training loop
3. Minimize exploitability directly
4. Target: Exploitability < 0.02 (superhuman)

### Approach

**Exploit-Defend Loop**:

```
Alternate phases:

  Exploit Phase:
    - Train π_exp as BR to fixed π_avg
    - Maximize: winrate(π_exp → π_avg)
    - Finds weaknesses in current agent

  Defend Phase:
    - Add π_exp to PSRO population
    - Train new BR/avg against updated σ (includes π_exp)
    - π_avg patches weaknesses
```

**Exploitability Metric**:
```
Exp_t ≈ winrate(π_exp_t vs π_avg_t) - 0.5

Goal: Exp_t → 0 (no single policy can exploit avg)
```

### Expected Outcomes

**Metrics**:
- Exploitability < 0.02 (near-Nash)
- π_avg handles diverse strategies
- Hard to exploit with single best-response

**Population**:
- 15-20 policies (BRs, avgs, exploiters)
- Nash mixture heavily weighted on recent avgs
- Older policies fade to near-zero mass

---

## Phase 4: Superhuman Evaluation & Iteration

**Duration**: Ongoing
**Status**: 📋 Planned
**Prerequisites**: Phase 3 complete

### Goals

1. Formal evaluation harness
2. Human vs bot assessment
3. Gen1-specific theory probes
4. Continuous iteration and refinement

### Evaluation Methods

**Automated Tournaments**:
- vs internal population (track Elo/Glicko)
- vs heuristic baselines (should dominate)
- vs search-based engines (should compete or beat)
- vs previous Metamon models (should surpass)

**Human Evaluation**:
- Anonymous ladder testing (Gen 1 OU)
- Invite strong players for evaluation matches
- Track rating/GXE over hundreds of games
- Target: Top 1% (1900+ Elo equivalent)

**Theory Probes**:
- Curated test positions:
  - Known theory endgames
  - Sacrifice puzzles
  - Sleep race decisions
  - Freeze fish spots
- Measure action distribution vs optimal play
- Track improvement over phases

### Superhuman Checkpoints

**Claim "superhuman Gen 1" when**:

1. ✅ **vs Baselines**: Clear dominance over all heuristics/IL bots
2. ✅ **vs Search**: Beats or competes with best search engines
3. ✅ **vs Humans**: Rating comparable to top players (1900+ Elo)
4. ✅ **Theory Probes**: High accuracy on optimal lines
5. ✅ **Exploitability**: < 0.02 (near-Nash)

---

## Summary: What Makes This "Nash-First"?

### Traditional RL Approach
1. Train single policy to beat some opponents
2. Hope it generalizes
3. Maybe think about Nash later

### Nash-First Approach (This Plan)
1. **Always maintain population** (Π)
2. **Always maintain meta-game** (M, σ)
3. **Always train as best-response to σ** (not random opponents)
4. **Explicitly minimize exploitability** (Phase 3+)
5. **Never use search** (pure RL/learning)

### Why This Works

**Game-Theoretic Foundation**:
- Gen 1 OU is a symmetric zero-sum game
- Nash equilibrium exists
- PSRO provably finds approximate Nash

**Practical Advantages**:
- Avoids cycles (rock-paper-scissors)
- Reduces exploitability systematically
- Population diversity prevents overfitting
- Meta-strategy balances exploration/exploitation

**No Search Required**:
- Neural policies learn strategic patterns
- Self-play data higher quality than tree search
- Faster inference (no search overhead)
- Scales to imperfect information (future gens)

---

## Timeline

| Phase | Duration | Status | Deliverable |
|-------|----------|--------|-------------|
| **Phase 0** | 1-2 weeks | ✅ Complete | Baseline population + Nash infrastructure |
| **Phase 1** | 4-6 weeks | ⏳ In Progress | PSRO population, Exploit < 0.15 |
| **Phase 2** | 6-8 weeks | 📋 Planned | NFSP agent, Exploit < 0.05 |
| **Phase 3** | 6-8 weeks | 📋 Planned | Exploitability descent, Exploit < 0.02 |
| **Phase 4** | Ongoing | 📋 Planned | Superhuman validation |
| **Total** | 5-7 months | - | Superhuman Gen 1 agent |

---

## Next Actions

### Immediate (This Week)
1. ✅ Complete Phase 1 documentation consolidation
2. ⏳ Run 5 PSRO iterations to completion
3. ⏳ Validate BR_0 through BR_4 beat previous Nash

### Short-term (Next 2 Weeks)
1. Analyze Phase 1 results (exploitability trend)
2. Plan Phase 2 architecture (BR/avg split)
3. Set up human evaluation infrastructure

### Medium-term (Next Month)
1. Begin Phase 2 implementation
2. First human evaluation round
3. Theory probe design

---

## Documentation

**Core Docs**:
- This file: High-level plan and timeline
- `metamon/nash/README.md`: Package documentation
- `PSRO_GUIDE.md`: Step-by-step user guide
- `metamon/nash/claude.md`: Implementation details

**Experiment Reports**:
- `LESSONS_LEARNED.md`: Key findings from failed experiments
- Phase-specific reports (Phase 0, 1, 2, etc.)

**Reference**:
- `TEAM_SETS_REFERENCE.md`: Available team sets for training/eval

---

**Last Updated**: November 28, 2025
**Current Phase**: Phase 1 (PSRO) - In Progress
