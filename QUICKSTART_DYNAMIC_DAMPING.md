# Quick Start: Dynamic Damping Experiments

## TL;DR

Dynamic damping is implemented and ready to use. Run these commands to start experiments:

### 1. Baseline Experiment (No Damping)
```bash
cd /home/eddie/repos/metamon
source .venv/bin/activate
export METAMON_CACHE_DIR=/tmp/metamon_cache

python -m metamon.rl.train_vanilla_selfplay \
  --run_name Baseline_$(date +%Y%m%d) \
  --init_checkpoint SyntheticRLV2 \
  --num_iterations 10 \
  --epochs_per_iteration 20 \
  --episodes_per_iteration 5000 \
  --save_dir ~/experiments/selfplay_baseline \
  --train_gin_config vanilla_selfplay_baseline.gin \
  --model_gin_config small_agent.gin
```

### 2. Dynamic Damping Experiment
```bash
python -m metamon.rl.train_vanilla_selfplay \
  --run_name Damped_$(date +%Y%m%d) \
  --init_checkpoint SyntheticRLV2 \
  --num_iterations 10 \
  --epochs_per_iteration 20 \
  --episodes_per_iteration 5000 \
  --save_dir ~/experiments/selfplay_damped \
  --train_gin_config vanilla_selfplay_damped.gin \
  --model_gin_config small_agent.gin
```

## What Was Implemented?

### Core Components
1. **Dynamic Damping Module** (`metamon/rl/dynamic_damping.py`)
   - Reverse-KL regularization: KL(π_new || π_ref)
   - Power-law schedules for entropy and KL coefficients
   - Adaptive learning rate control

2. **AMAGO Integration** (`metamon/rl/metamon_to_amago.py`)
   - Extended `MetamonAMAGOExperiment` with damping hooks
   - KL loss computation with action masking
   - Automatic logging of damping metrics

3. **Configuration Files**
   - `vanilla_selfplay_baseline.gin`: No damping (control)
   - `vanilla_selfplay_damped.gin`: With damping

4. **Training Scripts**
   - `collect_selfplay_data.py`: Collect self-play trajectories
   - `train_vanilla_selfplay.py`: Iterative self-play loop

## Key Features

✅ **Configurable via Gin**: Toggle damping on/off in config files
✅ **Action Masking Support**: Properly handles Metamon's illegal actions
✅ **Adaptive Control**: Automatically adjusts LR and KL coefficient
✅ **Comprehensive Logging**: KL, entropy, LR, coefficients to WandB/TB
✅ **Production Ready**: Fully tested and integrated

## Monitoring Your Experiments

### Key Metrics to Watch

1. **KL Divergence**: Should hover around target (0.01 by default)
   - Too high → updates too aggressive
   - Too low → updates too conservative

2. **Policy Entropy**: Should decay smoothly (not cliff-like)
   - Sudden drops → increase entropy coefficient
   - Too low too fast → agent becoming deterministic prematurely

3. **Learning Rate**: Should stabilize after warmup
   - Wild oscillations → adjust tolerance parameters

4. **Win Rate vs Baseline**: Track improvement over iterations
   - Baseline vs SynRL-V2
   - Damped vs SynRL-V2
   - Damped vs Baseline

### WandB Logging

Experiments automatically log to WandB if configured:
```bash
export METAMON_WANDB_PROJECT="metamon-selfplay"
export METAMON_WANDB_ENTITY="your-entity"
```

Look for these metric groups:
- `KL Divergence`, `Policy Entropy`
- `Damping/KL Coefficient`, `Damping/Entropy Coefficient`
- `Learning Rate`, `Actor Loss`, `Critic Loss`

## Expected Runtime

Per iteration (rough estimates):
- **Data Collection**: ~30-60 minutes (5000 battles)
- **Training**: ~1-2 hours (20 epochs)
- **Evaluation**: ~10-20 minutes (100 battles)

**Total for 10 iterations**: ~20-30 hours

## Troubleshooting

### "ModuleNotFoundError: No module named 'metamon'"
```bash
cd /home/eddie/repos/metamon
source .venv/bin/activate
```

### "Set METAMON_CACHE_DIR environment variable"
```bash
export METAMON_CACHE_DIR=/tmp/metamon_cache
```

### "Connection refused" (Ladder errors)
Make sure Pokemon Showdown server is running:
```bash
cd /home/eddie/repos/metamon/server/pokemon-showdown
node pokemon-showdown start --port 8000
```

### CUDA Out of Memory
Reduce batch size or use gradient accumulation in gin config.

## Next Steps After Experiments Complete

1. **Compare Results**:
   - Plot Elo curves (baseline vs damped)
   - Analyze entropy over time
   - Check KL divergence patterns

2. **If Damping Helps**:
   - Tune hyperparameters (see DYNAMIC_DAMPING_IMPLEMENTATION.md)
   - Extend to PSRO oracle training
   - Test with different reference policies

3. **If Damping Doesn't Help**:
   - Try different damping strengths
   - Adjust target KL values
   - Consider league mixture reference

## File Locations

```
metamon/
├── rl/
│   ├── dynamic_damping.py              # Core damping logic
│   ├── metamon_to_amago.py             # AMAGO integration (MODIFIED)
│   ├── collect_selfplay_data.py        # Data collection script
│   ├── train_vanilla_selfplay.py       # Main experiment script
│   └── configs/training/
│       ├── vanilla_selfplay_baseline.gin   # Config without damping
│       └── vanilla_selfplay_damped.gin     # Config with damping
└── DYNAMIC_DAMPING_IMPLEMENTATION.md   # Full documentation
```

## Questions or Issues?

Check the full documentation:
- `DYNAMIC_DAMPING_IMPLEMENTATION.md`: Complete technical details
- `metamon/rl/dynamic_damping.py`: Source code with docstrings
- Original implementation plan: See git history for planning discussion

## Success Criteria

Your experiments are successful if:
- ✅ Both baseline and damped experiments complete 10 iterations
- ✅ Damped agent maintains entropy better than baseline
- ✅ Damped agent achieves higher win rate vs SynRL-V2
- ✅ Damped agent shows more stable learning curves

Good luck with the experiments! 🚀
