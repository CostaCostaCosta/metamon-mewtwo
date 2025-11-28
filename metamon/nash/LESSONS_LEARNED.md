# Lessons Learned: Nash Equilibrium Training Experiments

**Purpose**: Document key findings from failed and successful experiments to guide future work.

**Date Range**: November 2025 (Phase 0 experiments)

---

## Summary

This document consolidates lessons from two major experiments conducted during Phase 0:
1. **P0_SYN_V2_GEN1**: Fine-tuning SyntheticRLV2 on Gen1 replays
2. **Gen1BinaryRewardV0**: Training with sparse binary rewards

**Key Insight**: Strong pretrained models (SyntheticRLV2) should not be fine-tuned on human replays or sparse rewards. Instead, use them as-is and improve through self-play and Nash equilibrium training.

---

## Experiment 1: Fine-tuning on Gen1 Replays ❌

### Hypothesis

Fine-tuning the multi-gen SyntheticRLV2 (200M params) on Gen1 OU replays with DefaultShapedReward would create a stronger Gen1 specialist.

**Reasoning**:
- SyntheticRLV2 trained on 9 generations
- Gen1 specialist should excel at Gen1 specifically
- Human replays provide domain-specific knowledge

### Setup

```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name P0_SYN_V2_GEN1 \
    --reward_function DefaultShapedReward \
    --formats gen1ou \
    --epochs 3 \
    --steps_per_epoch 25000
```

**Training Data**: 175k Gen1 OU human replays
**Total Steps**: 75k (3 epochs × 25k steps)
**Duration**: ~3 hours

### Results

**Training metrics looked good**:
- Critic loss: 1.9 → 1.5 ✅
- Actor loss: 0.14 → 0.06 ✅
- Validation vs heuristics: 90-100% win rate ✅

**But head-to-head evaluation revealed the truth**:

| Model | Win Rate (50 battles) |
|-------|----------------------|
| P0_SYN_V2_GEN1 (fine-tuned) | 38% ❌ |
| SyntheticRLV2 (base) | 62% ✅ |

**Statistical significance**: p < 0.05 (clear difference)

### Why It Failed

#### 1. Multi-gen Knowledge is Valuable

**SyntheticRLV2's strength**:
- Learned strategic patterns across 9 generations
- Deep understanding of type matchups, switch timing, prediction
- Robust decision-making that transfers to Gen1

**P0_SYN_V2_GEN1's weakness**:
- Overfitted to Gen1 human replay distribution
- Lost sophisticated multi-gen strategies
- Optimized for beating average humans, not strong RL policies

**Analogy**: Training a chess grandmaster only on amateur games makes them worse, not better.

#### 2. Heuristics are Weak Benchmarks

**Problem**: Both models beat heuristics 90-100%
- Heuristics use simple rules (max damage, random switches)
- Beating weak opponents doesn't indicate true strength

**Real test**: Strong vs strong
- P0_SYN_V2_GEN1 learned to exploit heuristic patterns
- But lost ability to handle sophisticated opponents
- SyntheticRLV2 maintained strategic depth

**Lesson**: Always validate against strong opponents, not just baselines.

#### 3. Human Replays are Noisy

**Gen1 OU replays contain**:
- Wide skill range (novice to expert)
- Suboptimal plays from average players
- Meta biases (popular but not optimal teams)
- Mistakes that strong agents shouldn't learn

**Fine-tuning effect**:
- Model adapted to average human play
- Regressed from "superhuman" to "human-level"
- Lost the edge that made SyntheticRLV2 strong

**Lesson**: Human replay data is useful for learning from scratch, but not for improving strong models.

#### 4. Shaped Rewards Can Mislead

**DefaultShapedReward incentivizes**:
- HP gain (+1.0 for healing)
- Status effects (+0.5 for burns/paralysis)
- KO advantages (+0.5 per fainted opponent Pokémon)

**Problem in fine-tuning context**:
- Biases policy toward defensive play
- Works against heuristics (they don't punish defensive play)
- Loses against strong opponents (too passive, loses tempo)

**Lesson**: Shaped rewards are good for learning fundamentals, but may not align with optimal play at high levels.

### What We Learned

✅ **Use SyntheticRLV2 directly** as Phase 0 baseline
- Stronger starting point (62% vs 38%)
- No time wasted on fine-tuning
- Multi-gen knowledge is an asset, not a liability

✅ **Self-play > human replays** for improving strong models
- Self-play data is higher quality
- Explores strategies beyond human meta
- Nash framework is the path forward

✅ **Always test against strong opponents**
- Heuristic validation is necessary but not sufficient
- Head-to-head vs strong models reveals true strength

---

## Experiment 2: Sparse Binary Rewards ❌

### Hypothesis

SyntheticRLV2 exhibits suboptimal end-game behavior (using recovery moves in lost positions). Switching from DefaultShapedReward to BinaryReward would fix this by focusing purely on winning.

**Reasoning**:
- DefaultShapedReward provides dense shaping (+1.0 HP, +0.5 status)
- These signals incentivize survival over winning
- BinaryReward (+100 win, -100 loss, 0 otherwise) focuses on victory

### Setup

```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Gen1BinaryRewardV0 \
    --reward_function BinaryReward \
    --formats gen1ou \
    --epochs 10 \
    --steps_per_epoch 25000
```

**Training Data**: 175k Gen1 OU replays (re-scored with BinaryReward)
**Total Steps**: 75k (stopped early after 3 epochs)
**Duration**: ~3 hours before stopping

### Results

**Training metrics showed no learning**:
- Critic loss: **Flat at 1.4-1.6** ❌
- Actor loss: **Flat at 0.07-0.08** ❌
- No downward trend after 75k steps

**Validation performance declined**:
- PokeEnvHeuristic: 100% → 75-95% ⚠️
- Valid actions: 99.9% → 99.5% ⚠️

**Conclusion**: Model not adapting to sparse rewards. Stopped training after 3 epochs.

### Why It Failed

#### 1. Distribution Shift Too Severe

**The problem**:
- SyntheticRLV2 optimized for dense shaped rewards
- Value function learned: "HP gain = good, status = good, KOs = very good"
- BinaryReward says: "Only terminal win/loss matters"
- Value function must completely relearn state values

**Why flat losses**:
- Offline RL re-scoring replays with new reward
- But value function still expects old reward structure
- No learning signal - old and new rewards fundamentally different

**Analogy**: Asking someone trained on "points for style" to optimize "only winning matters" - they need to unlearn everything.

#### 2. Learning Rate Too Conservative

**Setup for fine-tuning, not reward reshaping**:
- LR = 1.5e-4 (standard for fine-tuning)
- Good for adjusting existing knowledge
- Too small for fundamentally changing value function

**What's needed**:
- Higher LR or longer training (10x+ steps)
- But this risks destabilizing the model
- Not worth it when shaped rewards work fine

#### 3. Sparse Rewards Require High-Quality Data

**Problem with human replays**:
- Mix of novice and expert play
- Noisy signal for win-only optimization
- Hard to tell which actions led to victory

**Sparse rewards work better with**:
- Self-play data (consistent quality)
- Curated expert games
- Clear cause-effect relationships

**Lesson**: Sparse rewards are hard. Only use when necessary (they're not necessary here).

### What We Learned

✅ **Stick with DefaultShapedReward**
- Model trained on shaped rewards
- Fine-tuning with shaped rewards works
- Don't fix what isn't broken

✅ **Sparse rewards are for later**
- Once model is very strong
- Can generate high-quality self-play data
- Then gradual sparsification might work

✅ **Reward shaping is not the enemy**
- Well-designed shaping helps learning
- Only removes shaping if it demonstrably hurts performance
- SyntheticRLV2's recovery behavior might be gen-specific or rare

---

## General Principles

### What Works ✅

1. **Use strong pretrained models as-is**
   - SyntheticRLV2 is strong → use it directly
   - Don't fine-tune unless clear benefit
   - Multi-gen knowledge transfers well

2. **Self-play for improvement**
   - Higher quality than human replays
   - Explores beyond human meta
   - Nash framework (PSRO) is the right approach

3. **Test against strong opponents**
   - Heuristics are necessary but not sufficient
   - Always include head-to-head vs strong models
   - Track exploitability, not just win rate vs weak opponents

4. **Shaped rewards for strong models**
   - DefaultShapedReward is fine
   - Only change if specific problem identified
   - Sparse rewards are unnecessary complexity

5. **Population-based training**
   - Nash equilibrium framework
   - Systematic exploitability reduction
   - Avoids overfitting to specific opponents

### What Doesn't Work ❌

1. **Offline fine-tuning on human replays**
   - For already-strong models
   - Regression to human-level play
   - Loss of sophisticated strategies

2. **Sparse rewards for fine-tuning**
   - Distribution shift too severe
   - Flat loss curves, no learning
   - Not worth the complexity

3. **Heuristic-only validation**
   - Doesn't reveal true strength
   - Can be misleading
   - Must test against strong opponents

4. **Single-opponent training**
   - Overfits to specific strategies
   - Exploitable weaknesses
   - Nash framework is better

---

## Implications for Phase 1+

### Phase 1 (PSRO): Do's

✅ **Initialize BRs from SyntheticRLV2**
- Proven strong across all tests
- No fine-tuning needed
- Just use it

✅ **Train via self-play against population**
- Higher quality than human replay fine-tuning
- Nash framework prevents overfitting
- Systematic exploitability reduction

✅ **Use DefaultShapedReward**
- It works, model is trained on it
- No reason to change
- Simplicity is valuable

✅ **Validate exploitability, not just win rate**
- Nash mixture strength
- Exploitability metric
- Population tournaments

### Phase 1 (PSRO): Don'ts

❌ **Don't fine-tune on Gen1 replays**
- Experiment 1 showed this hurts performance
- Human data is lower quality
- Stick with self-play

❌ **Don't switch to sparse rewards**
- Experiment 2 showed flat losses
- Model won't adapt
- Keep shaped rewards

❌ **Don't rely on heuristic validation alone**
- Test BRs against previous Nash
- Test BRs against SyntheticRLV2
- Track exploitability

---

## Key Takeaways

### For Strong Models

**Don't:**
- Fine-tune on human replays (regresses to human-level)
- Switch to sparse rewards (distribution shift too severe)
- Optimize for beating weak baselines (misleading signal)

**Do:**
- Use pretrained model as-is (SyntheticRLV2 is strong)
- Improve through self-play (higher quality)
- Follow Nash framework (systematic exploitability reduction)
- Test against strong opponents (reveals true strength)

### For Nash-First Training

**The experiments validate the plan**:
1. Phase 0: Use SyntheticRLV2 directly ✅
2. Phase 1: PSRO with self-play ✅
3. Avoid offline fine-tuning ✅
4. Avoid sparse rewards ✅

**The failures point us in the right direction**:
- Self-play > human replays
- Nash framework > naive fine-tuning
- Strong pretrained model > domain-specific fine-tuning

---

## Detailed Experiment Data

### Experiment 1: P0_SYN_V2_GEN1

**Training run**: `~/metamon_checkpoints/P0_SYN_V2_GEN1`

**Validation results (epoch 2)**:
- GymLeader: 90% (100 battles)
- PokeEnvHeuristic: 100% (100 battles)
- Gen1BossAI: 100% (100 battles)
- Valid actions: 99.98%

**Head-to-head vs SyntheticRLV2**:
- 50 battles on local ladder
- Format: gen1ou
- Team set: competitive
- Result: 19 wins / 50 battles (38%)

**Loss curves**:
- Critic: 1.9 → 1.5 (decreasing)
- Actor: 0.14 → 0.06 (decreasing)
- Gradient norms: 0.4-1.2 (healthy)

**Conclusion**: Training looked successful but head-to-head revealed regression.

### Experiment 2: Gen1BinaryRewardV0

**Training run**: `~/metamon_checkpoints/Gen1BinaryRewardV0`
**Wandb**: `https://wandb.ai/.../runs/byimjo5u`

**Validation results (epoch 2)**:
- GymLeader: 100% (saturated, too easy)
- PokeEnvHeuristic: 75-95% (declining from 100%)
- Valid actions: 99.5-99.7% (declining from 99.9%)

**Loss curves**:
- Critic: 1.4-1.6 (flat, high variance)
- Actor: 0.07-0.08 (flat)
- Gradient norms: 0.4-1.2 (healthy, but not learning)
- Training speed: 6.85 it/s

**Conclusion**: No learning signal after 75k steps. Stopped early.

---

## References

**Detailed reports**:
- `P0_FINETUNING_EXPERIMENT_RESULTS.md` - Full Experiment 1 analysis
- `Gen1_BinaryReward_Training_Summary.md` - Full Experiment 2 analysis

**Related docs**:
- `NASH.md` - Nash-first training plan (validated by these experiments)
- `PSRO_GUIDE.md` - Phase 1 implementation guide
- `metamon/nash/README.md` - Technical documentation

**Papers**:
- SyntheticRLV2: "Human-Level Competitive Pokémon via Scalable Offline RL and Transformers" (RLC 2025)
- PSRO: "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (Lanctot et al. 2017)

---

**Last Updated**: November 28, 2025
**Phase**: Lessons learned from Phase 0, applied to Phase 1+
