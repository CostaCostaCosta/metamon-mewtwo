# Trajectory Curator Dashboard

A Gradio-based tool for analyzing and curating Pokémon Showdown battle trajectories to create high-quality expert datasets for reinforcement learning.

## Purpose

The Metamon dataset contains ~175,000 Gen 1 OU games. This tool helps you:

1. **Analyze** the dataset to understand player skill levels, battle characteristics, and Pokemon usage
2. **Filter** games based on quality criteria (rating, battle length, etc.)
3. **Export** a curated subset of expert games for training/finetuning

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch the dashboard
python scripts/trajectory_curator.py --data_dir ~/metamon_cache/parsed-replays/gen1ou
```

The dashboard will open at `http://localhost:7860`

### Command-line Options

```bash
python scripts/trajectory_curator.py \
    --data_dir ~/metamon_cache/parsed-replays/gen1ou \  # Data directory
    --port 7860 \                                        # Port (default: 7860)
    --share                                               # Create public share link
```

## Features

### 1. Load Data Tab
- Specify the directory containing `.json.lz4` trajectory files
- Click "Load Data" to scan and analyze all battles
- **Note:** Loading 175k battles takes ~5-10 minutes

### 2. Overview Tab
- View overall dataset statistics:
  - Total battles and unique players
  - Rating distribution (mean, median, range)
  - Battle length in turns
  - Struggle usage frequency
  - Date range
- **Pokemon usage table** showing the top 50 most-used Pokemon

### 3. Filter & Curate Tab
Apply filters to select high-quality expert games:

**Rating Filters:**
- `Min Rating`: Minimum ELO rating (e.g., 1500+ for expert games)
- `Max Rating`: Maximum ELO rating (set to 0 for no limit)

**Battle Length Filters:**
- `Min Turns`: Minimum battle length (filter out quick sweeps)
- `Max Turns`: Maximum battle length (filter out stall wars)

**Quality Filters:**
- `Exclude Struggle`: Remove battles where Struggle was used (indicates poor team composition)
- `Result Filter`: Include only wins, losses, or both

**After applying filters**, you'll see:
- How many battles match the criteria
- Statistics for the filtered subset
- Pokemon usage in the filtered set

### 4. Export Tab
- Specify an output directory
- Click "Export Dataset" to copy filtered battles
- The exported files maintain the same format and can be used directly with `ParsedReplayDataset`

## Recommended Curation Strategy

For creating a high-quality expert dataset:

### Phase 1: High-Skill Players Only
```
Min Rating: 1500
Max Rating: 0 (no limit)
Exclude Struggle: Yes
Result Filter: Both
```

**Rationale:**
- ELO 1500+ represents strong players in Gens 1-4
- Most games are "Unrated" (mapped to 1000) and include tournament games
- Excluding Struggle removes games where players ran out of PP

### Phase 2: Analyze Pokemon Usage
- Check the filtered Pokemon usage table
- Ensure the metagame looks reasonable (e.g., Tauros, Chansey, Snorlax common in Gen 1 OU)
- Identify any suspicious patterns

### Phase 3: Battle Length
Consider filtering by turns if needed:
- **Too short** (< 10 turns): May be early forfeits or sweeps
- **Too long** (> 100 turns): May be stall wars or suboptimal play

### Phase 4: Export
- Export to a dedicated directory like `~/metamon_expert_gen1ou/`
- Use this curated dataset for finetuning with `--parsed_replay_dir ~/metamon_expert_gen1ou`

## Dataset Statistics Display

### Rating Distribution
- **Mean/Median**: Central tendency of player skill
- **Range**: Spread of skill levels
- In Gen 1-4, most games are "Unrated" (1000), but 1500+ is expert level

### Battle Length
- **Mean/Median turns**: Typical game length
- Gen 1 OU averages 30-50 turns
- Very short games may indicate early forfeits
- Very long games may indicate stall strategies

### Struggle Usage
- Indicates a Pokemon ran out of PP for all moves
- High Struggle usage suggests poor team composition or knowledge
- Recommended to exclude for expert datasets

### Pokemon Usage
- Shows which Pokemon appear most frequently
- Useful for verifying metagame representation
- Gen 1 OU should show: Tauros, Chansey, Exeggutor, Snorlax, Alakazam, etc.

## Example Workflow

```bash
# 1. Launch dashboard
python scripts/trajectory_curator.py

# 2. In browser:
#    - Load Data: ~/metamon_cache/parsed-replays/gen1ou
#    - Wait for loading to complete (~5-10 min for 175k battles)

# 3. Check Overview:
#    - Note the rating distribution
#    - Check Pokemon usage looks reasonable

# 4. Apply Filters:
#    - Min Rating: 1500
#    - Exclude Struggle: Yes
#    - Click "Apply Filters"
#    - Review filtered statistics

# 5. Export:
#    - Output Directory: ~/metamon_expert_gen1ou
#    - Click "Export Dataset"

# 6. Use in training:
python -m metamon.rl.finetune_from_hf \
    --model_checkpoint SyntheticRLV2 \
    --parsed_replay_dir ~/metamon_expert_gen1ou \
    --formats gen1ou \
    ...
```

## Technical Details

### File Format
The tool reads trajectory files in the format:
```
{battle_id}_{rating}_{player1}_vs_{player2}_{date}_{result}.json.lz4
```

Where:
- `battle_id`: Showdown battle ID
- `rating`: ELO rating or "Unrated" (mapped to 1000)
- `player1`, `player2`: Player names
- `date`: MM-DD-YYYY format
- `result`: WIN or LOSS (perspective of player1)

### Data Structure
Each `.json.lz4` file contains:
- `states`: List of UniversalState objects (game state at each turn)
- `actions`: List of action indices taken

The analyzer extracts:
- Pokemon names from team rosters
- Battle length from action count
- Struggle usage from move lists in states

### Performance
- Loading 175k battles: ~5-10 minutes
- Memory usage: ~2-3 GB
- Filtering: Instantaneous (in-memory)
- Export: ~1-2 minutes per 10k battles

## Troubleshooting

### "No trajectory files found"
- Check the data directory path
- Ensure files end in `.json.lz4` or `.json`
- Verify the directory contains Gen 1 OU data

### Dashboard won't load
- Ensure Gradio is installed: `pip install gradio`
- Check port 7860 isn't in use: `lsof -i :7860`
- Try a different port: `--port 7861`

### Memory issues with large datasets
- The tool loads metadata only (not full trajectories)
- If still problematic, filter the directory first:
  ```bash
  # Example: only load battles from 2024+
  mkdir ~/gen1ou_2024
  cp ~/metamon_cache/parsed-replays/gen1ou/*2024*.json.lz4 ~/gen1ou_2024/
  ```

## Future Enhancements

Potential additions:
- Date range filtering in the UI
- Player name search
- More detailed battle analysis (win/loss patterns, opening moves)
- Automatic quality scoring
- Batch processing for multiple formats
