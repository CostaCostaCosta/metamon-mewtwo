# Batched Self-Play Data Generation Guide

High-throughput self-play script using batched AMAGO inference with **10-20x speedup** over baseline.

---

## Quick Start

### Self-Play (One Model)

Generate 1000 battles with SyntheticRLV2 playing against itself:

```bash
python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --checkpoint 48 \
    --num_battles 1000 \
    --batch_size 16 \
    --format gen1ou \
    --team_set modern_replays_v2 \
    --save_dir ~/selfplay_data/gen1ou_syntheticrlv2
```

**Expected Performance:**
- Throughput: ~20 battles/sec
- Time for 1000 battles: ~50 seconds
- VRAM usage: ~930 MB

---

## Command-Line Options

### Model Configuration

**Self-Play (Same Model for Both Players):**
```bash
--model SyntheticRLV2        # Model name
--checkpoint 48              # Checkpoint number (optional, uses default if omitted)
```

**Head-to-Head (Different Models):**
```bash
--model_p1 SyntheticRLV2     # Player 1 model
--checkpoint_p1 48           # Player 1 checkpoint
--model_p2 SyntheticRLV1     # Player 2 model
--checkpoint_p2 40           # Player 2 checkpoint
```

**Available Models:**
- `SyntheticRLV2` (200M, best)
- `SyntheticRLV1` (200M)
- `LargeRL` / `LargeIL` (200M)
- `MediumRL` / `MediumIL` (50M)
- `SmallRL` / `SmallIL` (15M)
- `Minikazam` (4.7M)

### Data Generation

```bash
--num_battles 1000           # Number of battles to generate (required)
--batch_size 16              # Parallel environments (default: 16)
--format gen1ou              # Battle format (default: gen1ou)
```

**Supported Formats:**
- `gen1ou`, `gen2ou`, `gen3ou`, `gen4ou`

### Team Configuration

```bash
--team_set modern_replays_v2         # Team set name (default)
--team_dir ~/custom/teams            # Custom team directory (optional)
--num_teams 100                      # Max teams to sample (optional, default: all)
```

**Team Directory Structure:**
```
$METAMON_CACHE_DIR/teams/
└── modern_replays_v2/
    ├── team_001.gen1ou_team
    ├── team_002.gen1ou_team
    └── ...
```

### Output Configuration

```bash
--save_dir ~/selfplay_data/gen1ou    # Output directory (required)
--run_name experiment_001            # Run name (optional, auto-generated if omitted)
```

**Output Structure:**
```
~/selfplay_data/gen1ou/
└── experiment_001/
    └── gen1ou/
        ├── uuid1_pypkmn.json.lz4
        ├── uuid2_pypkmn.json.lz4
        └── ...
```

### Performance Tuning

```bash
--device cuda                # Device (default: cuda)
--use_amp                    # Enable mixed precision (default: True, ~1.5x speedup)
--no_amp                     # Disable mixed precision
--temperature 1.0            # Action sampling temperature (default: 1.0)
```

### Logging

```bash
--verbose                    # Detailed progress (default: True)
--quiet                      # Minimal output
--log_interval 10            # Log every N battles (default: 10)
```

---

## Usage Examples

### Example 1: Standard Self-Play

Generate 1000 Gen1 OU battles for training data:

```bash
python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --checkpoint 48 \
    --num_battles 1000 \
    --batch_size 16 \
    --format gen1ou \
    --team_set modern_replays_v2 \
    --save_dir ~/selfplay_data/gen1ou \
    --run_name syntheticrlv2_selfplay_001
```

**Output:**
```
======================================================================
BATCHED SELF-PLAY DATA GENERATION
======================================================================
Run name: syntheticrlv2_selfplay_001
Format: gen1ou
Batch size: 16
Target battles: 1000
======================================================================
Loading teams from: /home/eddie/metamon_cache/teams/modern_replays_v2
✓ Loaded 32 teams

Loading model: SyntheticRLV2
  Checkpoint: 48
  Device: cuda
  Mixed precision: True
  Temperature: 1.0
✓ Model loaded

======================================================================
Starting Self-Play Data Generation
======================================================================
Batch size: 16
Target battles: 1000
Format: gen1ou
Output: /home/eddie/selfplay_data/gen1ou
======================================================================

✓ Created vectorized environment with 16 parallel battles

Starting data collection...
Progress: 0/1000 battles (0.0%) | Rate: 20.3 battles/sec | ETA: 49.3s
Progress: 100/1000 battles (10.0%) | Rate: 20.1 battles/sec | ETA: 44.8s
...

======================================================================
Self-Play Complete!
======================================================================
Battles completed: 1000/1000
Total time: 48.2s (0.8 minutes)
Average rate: 20.8 battles/sec
Output directory: /home/eddie/selfplay_data/gen1ou
======================================================================
```

### Example 2: Head-to-Head Evaluation

Compare two models:

```bash
python scripts/generate_selfplay_batched.py \
    --model_p1 SyntheticRLV2 \
    --checkpoint_p1 48 \
    --model_p2 SyntheticRLV1 \
    --checkpoint_p2 40 \
    --num_battles 500 \
    --batch_size 16 \
    --format gen1ou \
    --team_set modern_replays_v2 \
    --save_dir ~/evaluation/syntheticrlv2_vs_syntheticrlv1
```

### Example 3: Large-Scale Data Generation

Generate 10,000 battles for training:

```bash
python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --num_battles 10000 \
    --batch_size 16 \
    --format gen1ou \
    --team_set modern_replays_v2 \
    --save_dir ~/selfplay_data/gen1ou_large \
    --log_interval 100
```

**Expected:** ~8 minutes total (vs 88 minutes baseline!)

### Example 4: Custom Teams

Use your own team directory:

```bash
python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --num_battles 1000 \
    --batch_size 16 \
    --format gen1ou \
    --team_dir ~/my_custom_teams/gen1ou \
    --save_dir ~/selfplay_data/custom
```

### Example 5: Higher Exploration (Temperature Sampling)

Increase action diversity with higher temperature:

```bash
python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --num_battles 1000 \
    --batch_size 16 \
    --temperature 1.5 \
    --format gen1ou \
    --team_set modern_replays_v2 \
    --save_dir ~/selfplay_data/gen1ou_temp1.5
```

---

## Performance Optimization

### Batch Size Selection

| Batch Size | Throughput | VRAM | Recommendation |
|------------|------------|------|----------------|
| 1 | 2.1 battles/s | 822 MB | Baseline (not recommended) |
| 4 | 7.4 battles/s | 850 MB | Low-memory GPUs |
| 16 | **20.8 battles/s** | 931 MB | **Optimal for RTX 5090** |
| 64 | ~60 battles/s | ~2 GB | High throughput (if env supports) |

**Recommendation:** Use `--batch_size 16` for RTX 5090 (optimal speedup with minimal memory)

### Mixed Precision

**Enabled (default):**
```bash
--use_amp  # bfloat16, ~1.5-2x speedup
```

**Disabled (for debugging):**
```bash
--no_amp   # fp32, slower but more stable
```

### Monitoring Performance

Watch GPU usage during generation:

```bash
watch -n 1 nvidia-smi
```

Expected:
- GPU utilization: 70-90%
- VRAM: ~930 MB (batch=16)
- Power: ~150-200W

---

## Troubleshooting

### Issue: "ImportError: cannot import name X"

**Solution:** Ensure metamon environment is activated:
```bash
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Issue: "Team directory not found"

**Solution:** Download teams or specify correct path:
```bash
# Download teams if needed
python metamon/data/download_teams.py --team_set modern_replays_v2

# Or specify custom directory
--team_dir ~/path/to/teams
```

### Issue: Slow performance (<5 battles/sec)

**Possible causes:**
1. Batch size too small → Try `--batch_size 16`
2. Mixed precision disabled → Remove `--no_amp`
3. CPU bottleneck → Check team loading, reduce `--batch_size` if needed
4. Trajectory saving overhead → Use larger `--num_battles` for better amortization

### Issue: CUDA out of memory

**Solution:** Reduce batch size:
```bash
--batch_size 8  # Instead of 16
```

---

## Integration with Training

### Using Generated Data for Training

Generated trajectories are compatible with metamon's training pipeline:

```bash
# Generate self-play data
python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --num_battles 5000 \
    --batch_size 16 \
    --format gen1ou \
    --save_dir ~/selfplay_data/gen1ou_syntheticrlv2

# Train on generated data
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --custom_replay_dir ~/selfplay_data/gen1ou_syntheticrlv2 \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config vanilla_selfplay_baseline.gin \
    --epochs 5 \
    --save_dir ~/models/finetuned_syntheticrlv2 \
    --log
```

### Iterative Self-Play Loop

For continuous improvement:

```bash
# Loop: generate data → train → generate more data with new model
for iteration in {1..10}; do
    echo "Iteration $iteration"

    # Generate data with current best model
    python scripts/generate_selfplay_batched.py \
        --model_p1 CurrentBest \
        --model_p2 CurrentBest \
        --num_battles 1000 \
        --batch_size 16 \
        --save_dir ~/selfplay_loop/iter_$iteration

    # Train on new data
    python -m metamon.rl.finetune_from_hf \
        --finetune_from_model CurrentBest \
        --custom_replay_dir ~/selfplay_loop/iter_$iteration \
        --epochs 3 \
        --save_dir ~/models/iter_$iteration
done
```

---

## Advanced Usage

### Parallel Data Generation (Multiple GPUs)

Run multiple instances on different GPUs:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --num_battles 5000 \
    --batch_size 16 \
    --save_dir ~/selfplay_data/gpu0 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/generate_selfplay_batched.py \
    --model SyntheticRLV2 \
    --num_battles 5000 \
    --batch_size 16 \
    --save_dir ~/selfplay_data/gpu1 &

wait
```

### Scripted Batch Generation

Generate data for multiple formats:

```bash
#!/bin/bash
for format in gen1ou gen2ou gen3ou gen4ou; do
    python scripts/generate_selfplay_batched.py \
        --model SyntheticRLV2 \
        --num_battles 1000 \
        --batch_size 16 \
        --format $format \
        --team_set modern_replays_v2 \
        --save_dir ~/selfplay_data/$format \
        --run_name syntheticrlv2_$format
done
```

---

## Performance Comparison

### Baseline (Sequential) vs Batched

| Metric | Baseline | Batched (N=16) | Improvement |
|--------|----------|----------------|-------------|
| Battles/sec | 1.9 | 20.8 | **10.9x faster** |
| 1000 battles | 8.8 min | 48 sec | **10.9x faster** |
| 10,000 battles | 88 min | 8 min | **10.9x faster** |

### Real-World Timings

| Task | Baseline | Batched | Time Saved |
|------|----------|---------|------------|
| Quick test (100 battles) | 53s | 5s | 48s |
| Medium run (1000 battles) | 8.8min | 48s | 7.9min |
| Large run (10,000 battles) | 88min | 8min | 80min |
| Full dataset (100,000 battles) | 14.7hrs | 1.4hrs | 13.3hrs |

---

## Best Practices

1. **Start small**: Test with `--num_battles 100` to verify setup
2. **Use batch_size=16**: Optimal for RTX 5090
3. **Enable mixed precision**: Keep `--use_amp` (default) for best performance
4. **Monitor progress**: Use `--log_interval 100` for large runs
5. **Save regularly**: Script auto-saves every 100 battles
6. **Organize output**: Use descriptive `--run_name` values
7. **Check quality**: Verify trajectories have expected win rates and lengths

---

## Support

For issues or questions:
- Check troubleshooting section above
- Review `BATCHED_INFERENCE_RESULTS.md` for implementation details
- File issue at https://github.com/anthropics/metamon/issues (if applicable)
