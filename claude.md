# Metamon Repository Overview

## Project Description
Metamon is a research codebase for training reinforcement learning agents to play Pokémon Showdown at a human-competitive level. The project enables RL research on Pokémon Showdown by providing:

1. A standardized suite of teams and opponents for evaluation
2. A large dataset of RL trajectories reconstructed from real human battles (~4M trajectories)
3. Pretrained IL and RL policies available on HuggingFace
4. Training infrastructure using the AMAGO RL framework

The work is published as "Human-Level Competitive Pokémon via Scalable Offline RL and Transformers" (RLC, 2025).

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

## Future Work: Gen 1 OU Specialist via Self-Play

### Goal
Create a Generation 1 OverUsed tier specialist by improving the best existing model through self-play, focusing only on Gen 1 OU battles.

### Approach

#### 1. Base Model Selection
**Use SyntheticRLV2** as the starting point:
- 200M parameters
- Best performing model according to paper
- Already trained on diverse self-play data
- Checkpoint 48 (default evaluation checkpoint)
- Model config: `synthetic_multitaskagent.gin`
- Training config: `binary_rl.gin`

#### 2. Reward Function Modification
Create a simplified reward function by **removing annealing values**:
- The default reward function includes time-dependent annealing for shaping terms
- For a Gen 1 specialist, these are less relevant since we already understand the game
- Options:
  - Start with `BinaryReward` (simplest: +100/-100 only)
  - Or create custom reward inheriting from `DefaultShapedReward` with annealing removed
  - Located at `metamon/interface.py` line ~846-964

#### 3. Self-Play Data Collection Pipeline

**Step 1: Set up local Showdown server**
```bash
cd server/pokemon-showdown
node pokemon-showdown start --no-security
```

**Step 2: Generate self-play battles**
Use `metamon.rl.evaluate.pretrained_vs_local_ladder` to battle SyntheticRLV2 against itself:
```bash
# Terminal 1: Player 1
python -m metamon.rl.evaluate \
  --eval_type ladder \
  --agent SyntheticRLV2 \
  --gens 1 \
  --formats ou \
  --total_battles 1000 \
  --username Gen1Specialist_P1 \
  --team_set competitive \
  --save_trajectories_to ~/gen1_selfplay_data/

# Terminal 2: Player 2
python -m metamon.rl.evaluate \
  --eval_type ladder \
  --agent SyntheticRLV2 \
  --gens 1 \
  --formats ou \
  --total_battles 1000 \
  --username Gen1Specialist_P2 \
  --team_set competitive
```

This creates parsed replay files in the same format as the human dataset.

#### 4. Finetuning on Self-Play Data

**Finetune SyntheticRLV2 on the collected self-play data:**
```bash
python -m metamon.rl.finetune_from_hf \
  --finetune_from_model SyntheticRLV2 \
  --run_name gen1ou_specialist_v1 \
  --save_dir ~/metamon_checkpoints/ \
  --custom_replay_dir ~/gen1_selfplay_data/ \
  --custom_replay_sample_weight 0.5 \
  --formats gen1ou \
  --reward_function BinaryReward \
  --epochs 10 \
  --steps_per_epoch 10000 \
  --eval_gens 1 \
  --log
```

Key parameters:
- `--custom_replay_sample_weight 0.5` - Mix 50% self-play with 50% original human data
- `--formats gen1ou` - Only train on Gen 1 OU data
- `--reward_function BinaryReward` - Use simplified reward (no annealing)
- `--eval_gens 1` - Only evaluate on Gen 1 between epochs

#### 5. Iterative Improvement
Repeat the cycle:
1. Collect more self-play data with the improved model
2. Finetune on the new data
3. Evaluate improvement
4. Repeat

Monitor win rates against:
- Heuristic baselines (GymLeader, EmeraldKaizo, etc.)
- Original SyntheticRLV2
- Human players on ladder

### Setup Instructions

**First-time setup:**

1. Initialize and install Pokemon Showdown server:
```bash
# From metamon root directory
git submodule init
git submodule update
cd server/pokemon-showdown
npm install
```

2. Create mock pg module (workaround for optional PostgreSQL dependency):
```bash
# Still in server/pokemon-showdown directory
mkdir -p node_modules/pg
cat > node_modules/pg/index.js << 'EOF'
// Mock pg module for local development without PostgreSQL
module.exports = {
  Pool: class Pool {
    constructor() {}
    query() { return Promise.resolve({ rows: [] }); }
    connect() { return Promise.resolve({ release: () => {} }); }
    end() { return Promise.resolve(); }
  },
  Client: class Client {
    constructor() {}
    connect() { return Promise.resolve(); }
    query() { return Promise.resolve({ rows: [] }); }
    end() { return Promise.resolve(); }
  }
};
EOF

cat > node_modules/pg/package.json << 'EOF'
{
  "name": "pg",
  "version": "8.11.3",
  "main": "index.js"
}
EOF
```

3. Start the server:
```bash
./pokemon-showdown start --no-security
# Server will run on http://localhost:8000
```

**Note:** The mock pg module is needed because Pokemon Showdown lists PostgreSQL as an optional dependency (for production user databases), but still tries to import it. For local RL battles, we don't need a real database.

### Minimal Example

**Quick test run (small scale):**

1. Start Showdown server (in background or separate terminal):
```bash
cd server/pokemon-showdown
./pokemon-showdown start --no-security
```

2. Test model loading and basic battle:
```python
from metamon.rl.pretrained import get_pretrained_model
from metamon.env import get_metamon_teams, QueueOnLocalLadder
from metamon.interface import get_reward_function

# Load best model
model = get_pretrained_model("SyntheticRLV2")
agent = model.initialize_agent(checkpoint=48)

# Get Gen 1 OU teams
teams = get_metamon_teams("gen1ou", "competitive")

# Create environment
env = QueueOnLocalLadder(
    battle_format="gen1ou",
    player_username="test_agent",
    num_battles=1,
    observation_space=model.observation_space,
    action_space=model.action_space,
    reward_function=get_reward_function("BinaryReward"),
    player_team_set=teams,
    save_trajectories_to="/tmp/test_replay"
)

# Run battle
obs, info = env.reset()
done = False
while not done:
    action = agent.policy.decide(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

3. Small-scale self-play collection (10 battles):
```bash
# Two terminals running evaluation with different usernames
python -m metamon.rl.evaluate --eval_type ladder --agent SyntheticRLV2 \
  --gens 1 --formats ou --total_battles 10 --username Player1 \
  --save_trajectories_to ~/test_selfplay/
```

4. Quick finetune test:
```bash
python -m metamon.rl.finetune_from_hf \
  --finetune_from_model SyntheticRLV2 \
  --run_name test_gen1_specialist \
  --save_dir ~/test_ckpts/ \
  --custom_replay_dir ~/test_selfplay/ \
  --epochs 2 \
  --steps_per_epoch 1000 \
  --formats gen1ou \
  --eval_gens 1
```

### Expected Improvements

- **Specialization**: Model focuses solely on Gen 1 OU metagame
- **Self-play exploitation**: Discovers strategies against its own playstyle
- **Reduced complexity**: Simplified reward removes unnecessary shaping
- **Faster convergence**: Starting from strong pretrained model

### Monitoring & Evaluation

Track these metrics during training:
- Win rate vs heuristic baselines
- Win rate vs original SyntheticRLV2
- Diversity of strategies (via team usage statistics)
- Average battle length
- Action entropy (ensure not overfitting to deterministic strategies)

Use `--log` flag with wandb to track all metrics automatically.

### Key Files to Modify

1. **`metamon/interface.py`** (~line 846-964) - Create new reward function if needed
2. **`metamon/rl/finetune_from_hf.py`** - May need to adjust hyperparameters
3. **`metamon/rl/configs/training/`** - Custom training configs if needed

### Datasets

- **Human battles**: Automatically downloaded from `jakegrigsby/metamon-parsed-replays` on HuggingFace
- **Self-play battles**: Generated locally and saved to `--save_trajectories_to` directory
- Both use identical format (parsed replay format)

### Resources Required

- **GPU**: Finetuning 200M model requires research GPU (A100/V100 recommended)
  - Alternative: Use Minikazam (4.7M) as base for smaller GPU
- **Storage**: ~50GB for model checkpoints + self-play data
- **Time**:
  - Self-play generation: ~1-2 hours per 1000 battles
  - Finetuning: ~4-8 hours per epoch (depends on GPU)

---

## Notes

- All pretrained models support Gens 1-4, but have bias toward training distribution
- Gen 9 support is experimental/beta
- The replay parser has a "sim2sim gap" - self-collected data from online env is more accurate than parsed human replays
- Best practice: Offline pretraining → Online finetuning
