# Quick Start: Gen1 OU Self-Play with Dynamic Damping

## TL;DR - Run These Commands

### Setup
```bash
cd /home/eddie/repos/metamon
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Ensure Pokémon Showdown server is running
cd server/pokemon-showdown && node pokemon-showdown start --no-security &
cd ../..
```

### Baseline Experiment (No Damping)
```bash
# Collect self-play data
python scripts/generate_selfplay_data.py \
  --model SyntheticRLV2 \
  --num_battles 50000 \
  --battle_format gen1ou \
  --team_set modern_replays_v2 \
  --output_dir ~/gen1_baseline/gen0 \
  --parallel_instances 4

# Filter data
python scripts/filter_selfplay_data.py \
  --input_dir ~/gen1_baseline/gen0/trajectories \
  --output_dir ~/gen1_baseline/gen0_filtered \
  --max_invalid_rate 0.05 \
  --balance_outcomes

# Train Generation 1 (no damping)
python -m metamon.rl.finetune_from_hf \
  --finetune_from_model SyntheticRLV2 \
  --run_name Baseline_Gen1_$(date +%Y%m%d) \
  --custom_replay_dir ~/gen1_baseline/gen0_filtered \
  --formats gen1ou \
  --train_gin_config vanilla_selfplay_baseline.gin \
  --epochs 20 \
  --save_dir ~/gen1_checkpoints/baseline_gen1 \
  --battle_format gen1ou \
  --team_set modern_replays_v2 \
  --log
```

### Dynamic Damping Experiment
```bash
# Same data collection
python scripts/generate_selfplay_data.py \
  --model SyntheticRLV2 \
  --num_battles 50000 \
  --battle_format gen1ou \
  --output_dir ~/gen1_damped/gen0 \
  --parallel_instances 4

python scripts/filter_selfplay_data.py \
  --input_dir ~/gen1_damped/gen0/trajectories \
  --output_dir ~/gen1_damped/gen0_filtered \
  --max_invalid_rate 0.05 \
  --balance_outcomes

# Train with dynamic damping
python -m metamon.rl.finetune_from_hf \
  --finetune_from_model SyntheticRLV2 \
  --run_name Damped_Gen1_$(date +%Y%m%d) \
  --custom_replay_dir ~/gen1_damped/gen0_filtered \
  --formats gen1ou \
  --train_gin_config vanilla_selfplay_damped.gin \  # ← Enables dynamic damping!
  --epochs 20 \
  --save_dir ~/gen1_checkpoints/damped_gen1 \
  --battle_format gen1ou \
  --team_set modern_replays_v2 \
  --log
```

---

## What Was Implemented?

### ✅ Dynamic Damping Core (`metamon/rl/dynamic_damping.py`)
- Reverse-KL regularization: `KL(π_new || π_ref)`
- Power-law coefficient schedules (smooth decay, no "entropy cliffs")
- Adaptive learning rate and KL coefficient control
- Action masking support for Pokémon's constrained action space

### ✅ AMAGO Integration (`metamon/rl/metamon_to_amago.py`)
- Extended `MetamonAMAGOExperiment` with damping hooks
- KL loss computation in `compute_loss()` override
- Schedule updates in `train_step()`
- Adaptive control in `train_epoch()`
- Comprehensive logging of damping metrics

### ✅ Configuration Files
- `vanilla_selfplay_baseline.gin` - No damping (control)
- `vanilla_selfplay_damped.gin` - With dynamic damping

### ✅ Documentation
- `claude.md` - Updated with self-play infrastructure overview
- `GEN1OU_SELFPLAY_GUIDE.md` - Complete workflow guide
- `DYNAMIC_DAMPING_IMPLEMENTATION.md` - Technical details

---

## Key Features

### Dynamic Damping Prevents Policy Collapse

**Without damping**: Policy can collapse, entropy drops suddenly, agent becomes deterministic too early

**With damping**:
- KL regularization keeps policy near reference
- Smooth schedule prevents entropy cliffs
- Adaptive control maintains stable update sizes

### Fully Configurable

All parameters controllable via gin config:
```gin
# KL regularization
MetamonAMAGOExperiment.kl_coef_init = 0.05      # Initial strength
MetamonAMAGOExperiment.kl_power_alpha = 0.5     # Decay rate

# Entropy regularization
MetamonAMAGOExperiment.ent_coef_init = 0.01     # Initial bonus
MetamonAMAGOExperiment.ent_power_alpha = 0.7    # Decay rate

# Adaptive control
MetamonAMAGOExperiment.target_kl_per_step = 0.01  # Target KL
MetamonAMAGOExperiment.kl_tolerance = 1.5         # ±50% band
```

### Production-Ready Infrastructure

**Uses existing battle-tested scripts**:
- `scripts/generate_selfplay_data.py` - Parallel data collection
- `scripts/filter_selfplay_data.py` - Quality filtering
- `scripts/self_play_tournament.py` - Evaluation
- `metamon.rl.finetune_from_hf` - Training

**No redundant implementations** - dynamic damping integrates seamlessly via gin configs.

---

## Monitoring Your Experiments

### Key Metrics (WandB/TensorBoard)

1. **KL Divergence** (`KL Divergence`)
   - Should hover near target (default: 0.01)
   - Too high → updates too conservative
   - Too low → updates too aggressive

2. **Policy Entropy** (`Policy Entropy`)
   - Should decay smoothly (not cliff-like)
   - Sudden drops → increase entropy coefficient

3. **Learning Rate** (`Learning Rate`)
   - Should stabilize after warmup
   - Wild oscillations → adjust tolerance

4. **Damping Coefficients**
   - `Damping/KL Coefficient` - Current KL weight
   - `Damping/Entropy Coefficient` - Current entropy bonus

### Success Indicators

✅ Damping is working if:
- Entropy decays smoothly (no sudden drops)
- KL stays near target (±tolerance)
- Learning rate stabilizes
- Training doesn't diverge or collapse

---

## Expected Timeline

**Per Generation** (50k battles, 20 epochs):
- Data collection: 1-2 hours (parallel)
- Filtering: 5-10 minutes
- Training: 2-4 hours (GPU)
- Evaluation: 30-60 minutes

**5 generations**: ~20-40 hours total

---

## Troubleshooting

### Environment Issues
```bash
# Activate venv
source .venv/bin/activate

# Set cache dir
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Start server
cd server/pokemon-showdown && node pokemon-showdown start --no-security
```

### Format Mixing
**CRITICAL**: Always use `--formats gen1ou` in training commands!

```bash
python -m metamon.rl.finetune_from_hf \
  --formats gen1ou \  # ← Prevents loading other formats
  ...
```

### CUDA OOM
Reduce batch size or increase gradient accumulation in gin config.

---

## Alternative: Automated Workflow

For automation (less control, more convenience):

```bash
python -m metamon.rl.train_vanilla_selfplay \
  --run_name Gen1_Automated \
  --init_checkpoint SyntheticRLV2 \
  --num_iterations 5 \
  --epochs_per_iteration 20 \
  --episodes_per_iteration 50000 \
  --save_dir ~/gen1_automated \
  --train_gin_config vanilla_selfplay_damped.gin \
  --battle_format gen1ou
```

**Note**: This orchestrator script is optional. For production experiments, the manual workflow (using existing scripts) is recommended for maximum control and transparency.

---

## Next Steps

1. **Run both experiments** (baseline + damped) for 5 generations
2. **Compare results**:
   - Elo progression
   - Win rates vs SynRL-V2
   - Entropy curves
   - Training stability
3. **If promising**: Extend to PSRO oracle training
4. **If not**: Tune hyperparameters (see `DYNAMIC_DAMPING_IMPLEMENTATION.md`)

---

## File Locations

```
metamon/
├── rl/
│   ├── dynamic_damping.py              # Core damping logic (NEW)
│   ├── metamon_to_amago.py             # AMAGO integration (MODIFIED)
│   ├── train_vanilla_selfplay.py       # Optional orchestrator
│   └── configs/training/
│       ├── vanilla_selfplay_baseline.gin   # No damping
│       └── vanilla_selfplay_damped.gin     # With damping
├── claude.md                           # Repository overview (UPDATED)
├── GEN1OU_SELFPLAY_GUIDE.md           # Complete workflow guide
└── DYNAMIC_DAMPING_IMPLEMENTATION.md  # Technical details
```

**Existing Production Scripts** (`scripts/`):
- `generate_selfplay_data.py` - Data collection
- `filter_selfplay_data.py` - Quality filtering
- `self_play_tournament.py` - Evaluation
- `calculate_elo.py` - ELO ratings

---

## Documentation

- **`GEN1OU_SELFPLAY_GUIDE.md`** - Complete step-by-step guide
- **`DYNAMIC_DAMPING_IMPLEMENTATION.md`** - Technical details and tuning
- **`claude.md`** - Self-play infrastructure overview
- **`scripts/README.md`** - Production scripts documentation

---

## Summary

Dynamic damping is fully integrated and ready to use! Key points:

✅ Uses existing production-ready scripts (don't reinvent the wheel)
✅ Enable via gin config (`vanilla_selfplay_damped.gin`)
✅ Always specify `--formats gen1ou` to prevent format mixing
✅ Monitor KL, entropy, and LR for tuning
✅ Compare baseline vs damped to measure impact

Time to run experiments! 🚀
