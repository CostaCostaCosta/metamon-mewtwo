# Manual Agent Evaluation Workflow

**Category**: Evaluation
**Status**: Active
**Priority**: High

---

## Overview

This skill documents how to register trained checkpoints as named agents and run manual evaluations on the Pokemon Showdown ladder.

## Registering Checkpoints as Named Agents

### Location
Edit `metamon/rl/pretrained.py` to add new agent registrations.

### Pattern for Local Checkpoints

Use the `@pretrained_model()` decorator with `LocalPretrainedModel` base class:

```python
@pretrained_model()
class Loop6Epistemic_Epoch0(LocalPretrainedModel):
    """
    Brief description of the agent.

    Key features:
    - Feature 1
    - Feature 2
    - Training details
    """

    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/epistemic_test/loop6c-epistemic",
            model_name="loop6c-epistemic",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="epistemic_aware_rl.gin",
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            observation_space=get_observation_space("ExpandedObservationSpace"),
            action_space=get_action_space("DefaultActionSpace"),
            tokenizer=get_tokenizer("ExpandedObservationSpace-v1"),
            epoch=0,
        )
```

### Required Parameters

- **amago_ckpt_dir**: Full path to checkpoint directory (contains `ckpts/` folder)
  - Example: `/home/eddie/metamon/models/epistemic_test/loop6c-epistemic`
  - This is `--save_dir + --run_name` from training command

- **model_name**: Short identifier for logging
  - Example: `"loop6c-epistemic"`

- **model_gin_config**: Model architecture config
  - Common: `"synthetic_multitaskagent.gin"` (200M params)
  - Others: `"large_agent.gin"`, `"medium_agent.gin"`, `"small_agent.gin"`

- **train_gin_config**: Training config (determines agent behavior)
  - Common: `"epistemic_aware_rl.gin"`, `"selfplay_damped_aggressive.gin"`

- **reward_function**: Must match what was used in training
  - `get_reward_function("AggressiveShapedRewardSleep")` for sleep agents
  - `get_reward_function("DefaultShapedReward")` for standard agents
  - `get_reward_function("BinaryReward")` for binary reward agents

- **observation_space**: Must match training
  - `get_observation_space("ExpandedObservationSpace")` (most recent)
  - `get_observation_space("DefaultObservationSpace")` (legacy)

- **action_space**: Usually `get_action_space("DefaultActionSpace")`

- **tokenizer**: Must match observation space
  - `get_tokenizer("ExpandedObservationSpace-v1")` for ExpandedObservationSpace
  - `get_tokenizer("DefaultObservationSpace-v1")` for DefaultObservationSpace

- **epoch**: Which epoch checkpoint to load (0, 1, 2, etc.)

---

## Running Manual Evaluations

### Ladder Evaluation (Online Play)

**Format**:
```bash
python -m metamon.rl.evaluate \
    --agent Loop6Epistemic_Epoch0 \
    --eval_type ladder \
    --gens 1 \
    --formats ou \
    --total_battles 1000 \
    --username "YourUsername" \
    --team_set smogon_pass2_selected \
    --save_trajectories_to ~/metamon/trajectories/loop7/
```

**Parameters**:
- `--agent`: Registered agent name (class name without parentheses)
- `--eval_type`: `ladder` for online play, `baseline` for offline eval
- `--gens`: Generation(s) to battle (1 for Gen1, multiple possible)
- `--formats`: Format (e.g., `ou` for OU, `ubers` for Ubers)
- `--total_battles`: Number of battles to play
- `--username`: Showdown username for ladder
- `--team_set`: Team selection strategy
  - `smogon_pass2_selected`: Curated teams
  - `modern_replays_v2`: Sampled from replays
- `--save_trajectories_to`: Optional directory to save battle replays

### Baseline Evaluation (Offline)

**Format**:
```bash
python -m metamon.rl.evaluate \
    --agent Loop6Epistemic_Epoch0 \
    --eval_type baseline \
    --baseline RandomBaseline \
    --gens 1 \
    --formats ou \
    --num_battles 100
```

**Common Baselines**:
- `RandomBaseline`: Random action selection
- `Grunt`: Basic heuristic (switches/attacks randomly)
- `GymLeader`: Moderate heuristic
- `EmeraldKaizo`: Strong heuristic

---

## Typical Evaluation Workflow

### 1. Register Checkpoint After Training

After completing a training run:
```bash
# Training completed
# Checkpoint saved to: ~/metamon/models/epistemic_test/loop6c-epistemic/ckpts/policy_weights/epoch_0.pt

# Register in pretrained.py as shown above
```

### 2. Quick Sanity Check (Offline)

Test against RandomBaseline to verify agent loads correctly:
```bash
python -m metamon.rl.evaluate \
    --agent Loop6Epistemic_Epoch0 \
    --eval_type baseline \
    --baseline RandomBaseline \
    --gens 1 \
    --formats ou \
    --num_battles 20
```

**Expected**: Should win 80-100% of battles if agent is functioning.

### 3. Full Baseline Eval

Compare against standard baselines:
```bash
for baseline in RandomBaseline Grunt GymLeader; do
    python -m metamon.rl.evaluate \
        --agent Loop6Epistemic_Epoch0 \
        --eval_type baseline \
        --baseline $baseline \
        --gens 1 \
        --formats ou \
        --num_battles 100
done
```

### 4. Ladder Evaluation (Data Collection)

Play on ladder to collect self-play data:
```bash
python -m metamon.rl.evaluate \
    --agent Loop6Epistemic_Epoch0 \
    --eval_type ladder \
    --gens 1 \
    --formats ou \
    --total_battles 1000 \
    --username "Loop6EpistemicTest" \
    --team_set smogon_pass2_selected \
    --save_trajectories_to ~/metamon/trajectories/loop7/
```

**Usage for self-play data**:
- Saves battles to trajectory directory
- Can be filtered and used for next training loop
- Provides win rate estimate against ladder opponents

---

## Common Issues

### Agent Not Found
```
ValueError: Unknown pretrained model 'Loop6Epistemic_Epoch0'
```

**Solution**: Ensure you've added the `@pretrained_model()` decorator and restarted Python.

### Checkpoint Path Error
```
FileNotFoundError: .../ckpts/policy_weights/epoch_0.pt
```

**Solution**: Verify `amago_ckpt_dir` points to the directory **containing** the `ckpts/` folder, not the `ckpts/` folder itself.

### Reward Function Mismatch
```
RuntimeError: size mismatch for ...
```

**Solution**: Ensure `reward_function` in registration matches training. Check training logs for reward function used.

### Observation Space Mismatch
```
KeyError: 'some_observation_key'
```

**Solution**: Ensure `observation_space` and `tokenizer` match training. Most recent models use `ExpandedObservationSpace`.

---

## Evaluation Metrics to Track

### Win Rate
- Against RandomBaseline: Should be 80-100%
- Against Grunt: Should be 60-80%
- Against GymLeader: Should be 40-60%
- On Ladder: Varies (40-60% typical for good agents)

### Epistemic-Specific Metrics

For epistemic uncertainty checkpoints, compare epoch-0 performance to baseline (frozen actor):
- **Success**: Epoch-0 win rate > 40% (no catastrophic collapse)
- **Partial**: 30-40% (some collapse)
- **Failure**: < 30% (catastrophic collapse)

Track this in training logs:
- Epoch -1 (frozen actor): ~50% baseline
- Epoch 0 (with epistemic weighting): Should maintain ~40-50%
- Epoch 0 (without epistemic weighting): Typically drops to 0-10%

---

## Examples

### Epistemic Uncertainty Evaluation

```bash
# Register agent (already done in pretrained.py)
# @pretrained_model()
# class Loop6Epistemic_Epoch0(LocalPretrainedModel):
#     ...

# Quick test against random
python -m metamon.rl.evaluate \
    --agent Loop6Epistemic_Epoch0 \
    --eval_type baseline \
    --baseline RandomBaseline \
    --gens 1 \
    --formats ou \
    --num_battles 50

# Full baseline sweep
for baseline in RandomBaseline Grunt GymLeader EmeraldKaizo; do
    echo "Testing against $baseline"
    python -m metamon.rl.evaluate \
        --agent Loop6Epistemic_Epoch0 \
        --eval_type baseline \
        --baseline $baseline \
        --gens 1 \
        --formats ou \
        --num_battles 100 \
        --log
done

# Ladder evaluation with data collection
python -m metamon.rl.evaluate \
    --agent Loop6Epistemic_Epoch0 \
    --eval_type ladder \
    --gens 1 \
    --formats ou \
    --total_battles 1000 \
    --username "Loop6EpistemicTest" \
    --team_set smogon_pass2_selected \
    --save_trajectories_to ~/metamon/trajectories/loop7/ \
    --log
```

### Comparing Multiple Epochs

```bash
# Register all epochs
# @pretrained_model()
# class Loop6Epistemic_Epoch0(LocalPretrainedModel): ...
#
# @pretrained_model()
# class Loop6Epistemic_Epoch1(LocalPretrainedModel): ...
#
# @pretrained_model()
# class Loop6Epistemic_Epoch2(LocalPretrainedModel): ...

# Evaluate each
for epoch in 0 1 2; do
    python -m metamon.rl.evaluate \
        --agent Loop6Epistemic_Epoch${epoch} \
        --eval_type baseline \
        --baseline GymLeader \
        --gens 1 \
        --formats ou \
        --num_battles 100 \
        --log
done
```

---

## Success Criteria

### Checkpoint Registration
- ✅ Agent loads without errors
- ✅ Can complete battles
- ✅ Wins majority against RandomBaseline

### Epistemic Uncertainty Validation
- ✅ Epoch-0 win rate > 40% vs RandomBaseline
- ✅ Maintains performance compared to frozen baseline
- ✅ No catastrophic collapse in first epoch

### Data Collection for Self-Play
- ✅ Trajectories saved to specified directory
- ✅ Win rate on ladder > 40%
- ✅ Sufficient battles for next training loop (500-1000+)

---

**Status**: Active workflow for all checkpoint evaluations
**Last Updated**: 2025-12-21
**Related Skills**: `selfplay-loop-workflow`, `epistemic-uncertainty-actor-weighting`
