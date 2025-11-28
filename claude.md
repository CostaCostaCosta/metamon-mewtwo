# Metamon Repository Overview

## Project Description
Metamon is a research codebase for training reinforcement learning agents to play Pokémon Showdown at a human-competitive level. The project enables RL research on Pokémon Showdown by providing:

1. A standardized suite of teams and opponents for evaluation
2. A large dataset of RL trajectories reconstructed from real human battles (~4M trajectories)
3. Pretrained IL and RL policies available on HuggingFace
4. Training infrastructure using the AMAGO RL framework

The work is published as "Human-Level Competitive Pokémon via Scalable Offline RL and Transformers" (RLC, 2025).

## Environment Setup

Before running any commands, activate the virtual environment and set the cache directory:

```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

To start local Pokémon Showdown server:

```bash
cd server/pokemon-showdown && node pokemon-showdown start --no-security
```

## Repository Structure

### Core Components

- **`metamon/env/`** - Environment wrappers for Pokémon Showdown
  - `wrappers.py` - Gymnasium-compatible environment wrappers
  - `metamon_battle.py` - Battle logic integration
  - `metamon_player.py` - Player interface

- **`metamon/interface.py`** - Core abstractions
  - `ObservationSpace` - Defines how game state is represented to agents
  - `ActionSpace` - Converts agent outputs to game actions (13 discrete: 4 moves, 5 switches, 4 tera moves)
  - `RewardFunction` - Defines reward shaping
  - `UniversalState` / `UniversalAction` - Common representations

- **`metamon/rl/`** - Training and evaluation
  - `train.py` - Train agents from scratch on offline datasets
  - `finetune_from_hf.py` - Finetune pretrained models from HuggingFace
  - `evaluate.py` - Evaluate against baselines or ladder opponents
  - `pretrained.py` - Registry of all pretrained models
  - `metamon_to_amago.py` - Bridges Metamon environments to AMAGO framework

- **`metamon/data/`** - Dataset management
  - `parsed_replay_dset.py` - PyTorch dataset for reconstructed human battles
  - `download.py` - Downloads datasets from HuggingFace

- **`metamon/baselines/`** - Heuristic and learned baseline opponents
  - Includes: RandomBaseline, Grunt, GymLeader, EmeraldKaizo, etc.

- **`metamon/tokenizer/`** - Text tokenization for observations

- **`server/`** - Local Pokémon Showdown server (submodule)

## Available Pretrained Models

### Paper Models (Gens 1-4)
All available on HuggingFace at `jakegrigsby/metamon`:

| Model | Parameters | Description | Default Ckpt |
|-------|-----------|-------------|--------------|
| **SyntheticRLV2** | 200M | **Best model**: Actor-critic with value classification, 1M human + 4M synthetic self-play | 48 |
| SyntheticRLV1++ | 200M | SyntheticRLV1 finetuned on 2M battles vs diverse opponents | 40 |
| SyntheticRLV1_SelfPlay | 200M | SyntheticRLV1 finetuned on 2M self-play battles | 40 |
| SyntheticRLV1 | 200M | 1M human + 2M diverse self-play | 40 |
| SyntheticRLV0 | 200M | 1M human + 1M diverse self-play | 40 |
| LargeRL / LargeIL | 200M | Trained on 1M human battles only | 40 |
| MediumRL / MediumIL | 50M | Trained on 1M human battles only | 40 |
| SmallRL / SmallIL | 15M | Trained on 1M human battles only | 40 |
| Minikazam | 4.7M | Small RNN trained on parsed-replays v4 + self-play (best for finetuning on limited GPU) | varies |

### PokéAgent Challenge Models (Gens 1-4 + 9)
- **Abra** (57M), **Kadabra**, **Alakazam** - Gen 9-compatible models
- **SmallRLGen9Beta** (15M) - Prototype Gen 9 model

## Reward Functions

Located in `metamon/interface.py`:

1. **DefaultShapedReward** - Used by paper models
   - +/- 100 for win/loss
   - Light shaping for damage dealt, health recovered, status inflicted/received
   - **Contains annealing values** for these shaping terms

2. **AggressiveShapedReward** - Removes status shaping, makes rewards +200/+0

3. **BinaryReward** - Sparse: only +/- 100 for win/loss

## Observation Spaces

1. **DefaultObservationSpace** - Original paper space (text + numerical features)
2. **ExpandedObservationSpace** - Improved version with Gen 9 tera types
3. **TeamPreviewObservationSpace** - Adds opponent team preview
4. **OpponentMoveObservationSpace** - Includes revealed opponent moves

All can be tokenized using vocabs in `metamon/tokenizer/`.

---

## Current Work: Nash Equilibrium Training (PSRO)

**Goal**: Develop Gen 1 OU Nash equilibrium policy using Policy Space Response Oracles (PSRO).

**Status**: Phase 1 in progress
- Phase 0 complete: Baseline population established (exploitability = 0.44)
- Phase 1: Iterative best-response training to reduce exploitability to ~0.1-0.2

**Quick Start**:
```bash
python -m metamon.nash.run_psro \
    --phase0_dir ~/nash_phase0 \
    --save_dir ~/nash_phase1 \
    --num_iterations 5 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --collection_battles 500 \
    --oracle_epochs 3 \
    --oracle_model_config synthetic_multitaskagent.gin \
    --oracle_train_config psro_oracle.gin \
    --init_from_checkpoint SyntheticRLV2 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou \
    --log
```

**Documentation**:
- **`NASH.md`**: Complete Nash-first training roadmap (Phase 0-4)
- **`PSRO_GUIDE.md`**: Step-by-step PSRO training guide
- **`LESSONS_LEARNED.md`**: Key findings from Phase 0 experiments
- **`metamon/nash/README.md`**: Package technical documentation
- **`metamon/nash/claude.md`**: Implementation details and status

---

## Notes

- All pretrained models support Gens 1-4, but have bias toward training distribution
- Gen 9 support is experimental/beta
- Parsed replays location: `~/metamon_cache/parsed-replays` (set via `METAMON_CACHE_DIR` env var)
- Best practice: Offline pretraining → Online finetuning
