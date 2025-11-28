# Gen 1 Specialist Training - BinaryReward Experiment Summary

**Date:** November 14, 2025
**Model:** SyntheticRLV2 (200M parameters)
**Objective:** Create superhuman Gen 1 RBY specialist by fixing learned pathological behaviors

---

## Motivation

The base SyntheticRLV2 model exhibits suboptimal end-game behavior:
- **Problem:** Uses recovery moves (Recover, Rest) in clearly lost positions instead of going for high-risk plays
- **Root cause:** `DefaultShapedReward` provides dense shaping signals (+1.0 for HP gained, +0.5 for status changes) that incentivize survival over winning
- **Solution hypothesis:** Now that the model understands game mechanics, switch to `BinaryReward` (sparse: +100 win, -100 loss, 0 otherwise) to focus purely on victory

---

## Experimental Setup

### Configuration
```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --reward_function BinaryReward \
    --formats gen1ou \
    --epochs 10 \
    --steps_per_epoch 25000 \
    --eval_gens 1
```

**Key changes:**
1. **Reward function:** `DefaultShapedReward` → `BinaryReward`
2. **Format specialization:** Train only on Gen 1 OU (175,570 battles) instead of all 9 generations
3. **Evaluation:** Gen 1 only (faster iteration)

### Technical Issues Resolved
- **PyTorch compile error:** Disabled `torch.compile` due to tensor stride assertion failures when changing reward functions
- **Showdown crashes:** Harmless service unavailable errors (trying to fetch online ladder data)

---

## Results After 3 Epochs (~3 hours)

### Training Metrics
| Metric | Observation | Assessment |
|--------|-------------|------------|
| **Critic Loss** | Flat at 1.4-1.6, no downward trend | ⚠️ **Concerning** |
| **Actor Loss** | Flat at 0.07-0.08, no downward trend | ⚠️ **Concerning** |
| **Gradient Norms** | Stable 0.4-1.2 | ✅ Healthy |
| **Training Speed** | 6.85 it/s, ~60 min/epoch | ✅ Good |

### Validation Performance
| Opponent | Epoch 0 | Epoch 2 | Trend |
|----------|---------|---------|-------|
| **GymLeader** | 100% | 100% | Saturated (too easy) |
| **PokeEnvHeuristic** | 100% | 75-95% | ⚠️ **Declining** |
| **Valid Actions** | 99.9%+ | 99.5-99.7% | ⚠️ Minor decline |

---

## Analysis

### The Problem: Model Not Adapting to Sparse Rewards

**Why flat loss curves matter:**
- Offline RL is re-computing rewards from replays using `BinaryReward`
- The value function must relearn what states are valuable (only terminal win/loss matters now)
- After 75,000 gradient steps, **no learning signal** - losses remain flat with high variance

**Hypothesis:**
1. **Distribution shift too severe:** Model optimized for dense rewards struggling with sparse signal
2. **Learning rate too conservative:** 1.5e-4 is tuned for fine-tuning, not reward function reshaping
3. **Dataset quality:** 175k Gen 1 battles include novice play that may confuse sparse reward learning

---

## Recommendations

### Option A: Improved Reward Function (Recommended)
**Use `AggressiveShapedReward` instead** - middle ground that keeps some shaping:
```python
reward = 1.0 * (damage_done + hp_gain)
       + 2.0 * (removed_pokemon - lost_pokemon)  # Emphasize KOs
       + 200.0 * victory  # Win only (not -200 for loss)
```
- Removes status shaping (addresses recovery move spam)
- Maintains HP/KO signals for faster learning
- Asymmetric victory bonus discourages clinging to lost positions

### Option B: Curated Gen 1 Dataset
**Filter training data to high-quality games only:**
- Self-play battles from SyntheticRLV2 vs itself
- High-ELO human replays (1500+ rating)
- Expert-curated competitive matches

**Rationale:** Sparse rewards require high-quality demonstrations. Current dataset mixes novice and expert play, creating noisy signal for win-only optimization.

### Option C: Two-Stage Approach
1. **Stage 1:** Specialize on Gen 1 with `DefaultShapedReward` (baseline)
2. **Stage 2:** Generate self-play data, then finetune with `BinaryReward`

---

## Next Steps

### Immediate (This Run)
- ⏳ Let current run complete (7 more epochs, ~7 hours) to confirm loss remains flat
- 📊 Document final metrics for comparison

### Short-term
- 🔄 **Restart with `AggressiveShapedReward`** - likely to succeed
- 🎯 Generate 50k high-quality Gen 1 self-play battles using SyntheticRLV2
- 📈 Compare eval against base model (not just heuristics)

### Medium-term (Superhuman Gen 1)
- 🔁 Iterative self-play loop: v0 → v1 → v2 generations
- 🎮 PokéAgent Challenge ladder evaluation
- 👥 Human expert evaluation

---

## Open Questions

1. **Is dataset quality the bottleneck?** Should we curate Gen 1 expert-only battles?
2. **What's the minimal shaping needed?** Can we ablate status/HP components independently?
3. **Does the model actually exhibit recovery spam in Gen 1?** May be gen-specific behavior

---

**Run ID:** `Gen1BinaryRewardV0`
**Wandb:** `https://wandb.ai/costacosta-personal-research/metamon/runs/byimjo5u`
**Checkpoints:** `~/metamon_checkpoints/Gen1BinaryRewardV0/ckpts/`
