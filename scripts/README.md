# Self-Play Tournament & Data Generation Pipeline

Complete pipeline for evaluating Gen1 BinaryReward model checkpoints, computing ELO ratings, and generating high-quality self-play data for iterative training.

## Overview

This pipeline enables:
1. **Round-robin tournaments** between different training checkpoints
2. **ELO rating calculation** to quantify relative strength
3. **Self-play data generation** from the strongest models
4. **Data quality filtering** to ensure high-quality training data
5. **Visualization and analysis** of results

## Prerequisites

- Pokémon Showdown server running locally: `node pokemon-showdown start --no-security`
- Trained model checkpoints in `~/metamon_checkpoints/`
- Python environment with metamon installed

## Quick Start

### 1. Register Your Checkpoints

Checkpoints are automatically registered in `metamon/rl/gen1_binary_models.py`:

```python
# Available models (once training completes):
Gen1BinaryV0_Epoch0   # 75k steps
Gen1BinaryV0_Epoch2   # 125k steps
Gen1BinaryV0_Epoch4   # 175k steps
Gen1BinaryV0_Epoch6   # 225k steps
Gen1BinaryV0_Epoch8   # 275k steps
Gen1BinaryV0_Epoch10  # 325k steps
```

### 2. Run Tournament

```bash
# Example: Compare Epoch 0, Epoch 2, and base SyntheticRLV2
python scripts/self_play_tournament.py \
    --models Gen1BinaryV0_Epoch0 Gen1BinaryV0_Epoch2 SyntheticRLV2 \
    --battles_per_matchup 200 \
    --output_dir ~/gen1_tournament_results \
    --battle_format gen1ou \
    --team_set competitive

# Expected runtime: ~4-6 hours for 3 models (3 matchups × 200 battles)
```

### 3. Calculate ELO Ratings

```bash
python scripts/calculate_elo.py \
    --tournament_dir ~/gen1_tournament_results \
    --k_factor 32 \
    --initial_rating 1500
```

**Output:**
- `elo_results_ratings.json` - Final ELO for each model
- `elo_results_progression.csv` - ELO changes over time
- `elo_results_matchup_matrix.csv` - Win/loss records
- `elo_results_statistics.csv` - Battle statistics

### 4. Visualize Results

```bash
python scripts/visualize_results.py \
    --tournament_dir ~/gen1_tournament_results \
    --output_dir ~/gen1_tournament_results/visualizations
```

**Generates:**
- `elo_rankings.png` - Bar chart of final ELO ratings
- `elo_progression.png` - ELO changes over battles
- `winrate_heatmap.png` - Head-to-head matchup matrix
- `statistics_comparison.png` - Win rates, battle lengths, etc.
- `tournament_summary.csv` - Comprehensive results table

### 5. Generate Self-Play Data

Once you've identified the best checkpoint, generate training data:

```bash
# Pure self-play (model vs itself)
python scripts/generate_selfplay_data.py \
    --model Gen1BinaryV0_Epoch2 \
    --num_battles 50000 \
    --output_dir ~/gen1_selfplay_data/v0 \
    --battle_format gen1ou \
    --team_set competitive \
    --parallel_instances 4

# Mixed self-play (two different checkpoints)
python scripts/generate_selfplay_data.py \
    --model Gen1BinaryV0_Epoch2 \
    --opponent_model Gen1BinaryV0_Epoch4 \
    --num_battles 50000 \
    --output_dir ~/gen1_selfplay_data/v0 \
    --battle_format gen1ou \
    --team_set competitive \
    --parallel_instances 4

# Expected runtime: ~25 hours for 50k battles with 4 parallel instances
```

### 6. Filter Self-Play Data

```bash
python scripts/filter_selfplay_data.py \
    --input_dir ~/gen1_selfplay_data/v0/gen1ou \
    --output_dir ~/gen1_selfplay_data/v0_filtered/gen1ou \
    --max_invalid_rate 0.05 \
    --min_turns 10 \
    --balance_outcomes
```

**Filters:**
- Remove battles with >5% invalid actions
- Remove battles <10 turns (forfeit/disconnect)
- Balance win/loss outcomes to 50/50

### 7. Train Next Generation

Use filtered data for next iteration:

```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Gen1BinaryV1 \
    --save_dir ~/metamon_checkpoints/ \
    --custom_replay_dir ~/gen1_selfplay_data/v0_filtered \
    --custom_replay_sample_weight 0.5 \
    --reward_function BinaryReward \
    --formats gen1ou \
    --epochs 10 \
    --steps_per_epoch 25000 \
    --eval_gens 1 \
    --log
```

## Detailed Usage

### Tournament Script

```bash
python scripts/self_play_tournament.py --help

Options:
  --models MODELS [MODELS ...]    Models to compete (required)
  --battles_per_matchup INT       Battles per matchup (default: 200)
  --output_dir PATH               Results directory
  --battle_format FORMAT          Format (default: gen1ou)
  --team_set NAME                 Team set (default: competitive)
  --battle_backend BACKEND        poke-env or metamon (default: poke-env)
  --list_models                   List available models
```

**How it works:**
- Generates all unique pairings (round-robin)
- Launches both models on local ladder simultaneously
- Models battle each other automatically
- Saves trajectories and team results

### ELO Calculation Script

```bash
python scripts/calculate_elo.py --help

Options:
  --tournament_dir PATH     Tournament results directory (required)
  --k_factor FLOAT          ELO K-factor (default: 32.0)
  --initial_rating FLOAT    Starting ELO (default: 1500.0)
  --output_prefix STR       Output file prefix (default: elo_results)
```

**ELO System:**
- K-factor of 32 (standard for chess)
- Initial rating of 1500 for all models
- Ratings update after each battle
- Expected score: `E = 1 / (1 + 10^((R_b - R_a) / 400))`

### Self-Play Generation Script

```bash
python scripts/generate_selfplay_data.py --help

Options:
  --model MODEL                Model for self-play (required)
  --opponent_model MODEL       Opponent (default: same as --model)
  --num_battles INT            Target battles (default: 10000)
  --output_dir PATH            Output directory
  --battle_format FORMAT       Format (default: gen1ou)
  --team_set NAME              Team set (default: competitive)
  --parallel_instances INT     Concurrent instances (default: 2)
```

**Performance:**
- Single instance: ~250-500 battles/hour
- 4 parallel instances: ~1000-2000 battles/hour
- Trajectories saved in parsed replay format
- Compatible with existing training scripts

### Data Filtering Script

```bash
python scripts/filter_selfplay_data.py --help

Options:
  --input_dir PATH          Input trajectory directory (required)
  --output_dir PATH         Output directory (required)
  --max_invalid_rate FLOAT  Max invalid action rate (default: 0.05)
  --min_turns INT           Min battle length (default: 10)
  --max_turns INT           Max battle length (default: 1000)
  --balance_outcomes        Balance win/loss to 50/50
  --seed INT                Random seed (default: 42)
```

### Visualization Script

```bash
python scripts/visualize_results.py --help

Options:
  --tournament_dir PATH    Tournament directory (required)
  --output_dir PATH        Output directory (default: tournament_dir/visualizations)
```

## File Structure

```
metamon/
├── metamon/rl/
│   └── gen1_binary_models.py          # Checkpoint registration
├── scripts/
│   ├── self_play_tournament.py        # Round-robin tournaments
│   ├── calculate_elo.py               # ELO rating system
│   ├── generate_selfplay_data.py      # Self-play data generation
│   ├── filter_selfplay_data.py        # Data quality filtering
│   ├── visualize_results.py           # Results visualization
│   └── README.md                       # This file
└── ~/gen1_tournament_results/
    ├── trajectories/                   # Battle replays
    │   └── gen1ou/                     # Format-specific
    │       └── *.json.lz4              # Compressed trajectories
    ├── team_results/                   # CSV battle logs
    │   └── battle_log_*.csv
    ├── tournament_results.json         # Raw matchup data
    ├── elo_results_ratings.json        # Final ELO ratings
    ├── elo_results_progression.csv     # ELO over time
    ├── elo_results_matchup_matrix.csv  # Win/loss matrix
    ├── elo_results_statistics.csv      # Statistics
    └── visualizations/                 # Plots
        ├── elo_rankings.png
        ├── elo_progression.png
        ├── winrate_heatmap.png
        ├── statistics_comparison.png
        └── tournament_summary.csv
```

## Iterative Self-Play Training Loop

```bash
# Generation 0: Finetune from base model
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Gen1BinaryV0 \
    --reward_function BinaryReward \
    --formats gen1ou \
    --epochs 10 \
    [...]

# Evaluate Gen1BinaryV0 checkpoints
python scripts/self_play_tournament.py [...]
python scripts/calculate_elo.py [...]

# Generate self-play from best checkpoint
python scripts/generate_selfplay_data.py \
    --model Gen1BinaryV0_Epoch2 \
    --num_battles 50000 \
    [...]

# Filter data
python scripts/filter_selfplay_data.py [...]

# Generation 1: Train on self-play
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Gen1BinaryV1 \
    --custom_replay_dir ~/gen1_selfplay_data/v0_filtered \
    --custom_replay_sample_weight 0.5 \
    --reward_function BinaryReward \
    --formats gen1ou \
    [...]

# Repeat: Evaluate → Generate → Filter → Train
```

## Tips & Best Practices

### Tournament Setup
- Start with 100-200 battles per matchup for quick results
- Scale to 500+ battles for publication-quality statistics
- Monitor Showdown server CPU usage with many parallel battles

### Self-Play Generation
- Use `--parallel_instances 4-8` for fast generation (if you have CPU)
- Monitor disk space (50k battles ≈ 5-10 GB compressed)
- Check a few trajectories manually to verify quality

### Data Filtering
- `--max_invalid_rate 0.05` is reasonable for well-trained models
- `--balance_outcomes` prevents bias toward one playstyle
- Review `filtering_stats.json` to tune thresholds

### ELO Interpretation
- Differences of 100+ ELO are substantial (≈64% expected win rate)
- Differences of 200+ ELO are dominant (≈76% expected win rate)
- Differences of 400+ ELO are nearly unbeatable (≈92% expected win rate)

### Common Issues

**Issue: Tournament hangs or times out**
- Check Showdown server is running and responsive
- Reduce `--battles_per_matchup` to test
- Check for port conflicts (default: 8000)

**Issue: Low trajectory count after filtering**
- Relax filters: increase `--max_invalid_rate` to 0.10
- Remove `--balance_outcomes` if data is scarce
- Check original data quality

**Issue: Models performing worse than expected**
- Verify checkpoint paths in `gen1_binary_models.py`
- Check reward function matches training config
- Test against known baselines first

## Example: Full Pipeline Run

```bash
# 1. Tournament (6 hours)
python scripts/self_play_tournament.py \
    --models Gen1BinaryV0_Epoch0 Gen1BinaryV0_Epoch2 SyntheticRLV2 \
    --battles_per_matchup 300 \
    --output_dir ~/gen1_results

# 2. ELO Calculation (<1 minute)
python scripts/calculate_elo.py \
    --tournament_dir ~/gen1_results

# 3. Visualization (<1 minute)
python scripts/visualize_results.py \
    --tournament_dir ~/gen1_results

# 4. Self-Play Generation (20 hours)
python scripts/generate_selfplay_data.py \
    --model Gen1BinaryV0_Epoch2 \
    --num_battles 40000 \
    --output_dir ~/gen1_selfplay/v0 \
    --parallel_instances 4

# 5. Filter Data (5 minutes)
python scripts/filter_selfplay_data.py \
    --input_dir ~/gen1_selfplay/v0/gen1ou \
    --output_dir ~/gen1_selfplay/v0_filtered/gen1ou \
    --balance_outcomes

# 6. Next Training Iteration (10 hours)
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Gen1BinaryV1 \
    --custom_replay_dir ~/gen1_selfplay/v0_filtered \
    --custom_replay_sample_weight 0.5 \
    --reward_function BinaryReward \
    --formats gen1ou \
    --epochs 10 \
    --steps_per_epoch 25000 \
    --log
```

**Total time: ~36 hours** for one complete iteration

## Citation

If you use this pipeline in your research, please cite the Metamon paper:

```bibtex
@inproceedings{metamon2025,
  title={Human-Level Competitive Pokémon via Scalable Offline RL and Transformers},
  author={...},
  booktitle={Reinforcement Learning Conference (RLC)},
  year={2025}
}
```

## Support

For issues or questions:
- Check existing tournament results for similar issues
- Review Showdown server logs
- Open an issue on the Metamon GitHub repository
