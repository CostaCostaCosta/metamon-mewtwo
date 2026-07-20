# PlasticTauros Belief-Control Notes

Date: 2026-06-13

This document summarizes the 15M PlasticTauros belief-control experiment,
how the belief system is implemented, and the recommended next scaled run.

## Summary

The full-budget 15M grouped belief-control run is the strongest 15M
PlasticTauros candidate tested so far.

- W&B run: `hr33xn01`
- Run name: `grouped_belief_control_150k`
- Model config: `smaller_multitaskagent_grouped_v2_belief_arch.gin`
- Train config: `plastic_tauros_15m_belief_control.gin`
- Dataset config: `gen1ou_small_tauros_core_belief.yaml`
- Training budget: `149,701` gradient steps
- Checkpoints: `policy_epoch_0.pt` through `policy_epoch_5.pt`

Compared with the previous non-belief 15M small-control run:

| Run | Final Heuristic Return | TaurosV0 H2H | Win Rate |
| --- | ---: | ---: | ---: |
| `grouped_belief_control_150k` | `144.43` | `12-38` | `24%` |
| `small-control` | `127.49` | `8-42` | `16%` |

The H2H gain is promising but not statistically decisive at 50 battles. The
belief-control Wilson interval is roughly `14%-37%`; the old small-control
interval is roughly `8%-29%`. The next verification step is a 200+ battle H2H
on checkpoint 5.

## Policy Health

The belief-control policy ends in a similar sharpness regime to the old
control rather than preserving substantially more action support:

- Policy entropy: `0.936`
- Entropy P10: `0.282`
- Effective support: `2.55`
- Invalid action mass: `0.00217`

The belief head learned the train target but should be checked on heldout data:

- Belief loss: `0.00784`
- Species top-6 recall: `0.690`
- Moves top-6 recall: `0.343`
- Belief embedding norm: `0.576`

The low final embedding norm may be benign compression, but it is worth
watching before scaling decisions depend on the auxiliary representation.

## Final Heuristic Breakdown

Final validation by opponent:

| Opponent | Return | Win Rate |
| --- | ---: | ---: |
| `PokeEnvHeuristic` | `164.33` | `0.80` |
| `Grunt` | `205.70` | `1.00` |
| `GymLeader` | `135.26` | `0.667` |
| `Gen1BossAI` | `178.82` | `0.857` |
| `EmeraldKaizo` | `38.03` | `0.20` |

The run is broad across heuristic opponents but remains weakest against
`EmeraldKaizo`.

## Belief Target Construction

The belief dataset config differs from the normal Tauros core dataset by one
field:

```yaml
belief_target_type: gen1_species_moves_set
```

This causes `ParsedReplayDataset` to build auxiliary `belief_*` tensors from
each replay. The targets are multi-hot tokenizer-vocabulary vectors:

- `belief_opp_species_set`
- `belief_opp_species_mask`
- `belief_opp_move_set`
- `belief_opp_move_mask`

Species targets aggregate opponent active species plus any parsed opponent
team-preview entries across the trajectory. Move targets aggregate opponent
active known moves and previous moves across the trajectory. The resulting set
labels are repeated at every timestep.

These labels are hindsight supervised targets for training. They are not
available at inference and are not passed directly to the actor.

## Belief Model Architecture

Normal 15M PlasticTauros uses `amago.agent.MultiTaskAgent`. The belief variant
uses `MetamonBeliefMultiTaskAgent`.

The underlying grouped timestep encoder and transformer trajectory encoder are
unchanged. The transformer state `s_rep` is fed to an auxiliary
`Gen1OpponentTeamBeliefHead`:

- Input: public trajectory state `s_rep`
- Trunk: 2-layer MLP, hidden dim `192`, LayerNorm, LeakyReLU, dropout `0.05`
- Species head: `2541` logits
- Move head: `2541` logits
- Actor projection: sigmoid species/move probabilities projected to a
  `64`-dim belief embedding

The actor receives:

```text
concat(s_rep, belief_embedding)
```

The critic still receives only `s_rep` and the action. This means the actor can
condition on inferred opponent-team information, while critic targets and value
learning remain on the public trajectory representation.

## Loss

The RL objective is the same filtered offline actor loss plus critic loss used
by the existing PlasticTauros control. The belief model adds:

```text
total_loss += 0.1 * belief_loss
```

The belief loss is BCE-with-logits against multi-hot species and move targets:

- Species loss weight: `1.0`
- Move loss weight: `0.5`
- Overall belief coefficient: `0.1`
- `belief_loss_backprop_to_encoder = True`

Because backprop to the encoder is enabled, the auxiliary objective shapes the
shared sequence representation, not just the belief head.

## Difference From Existing Tauros Setup

What changes:

- `MultiTaskAgent` is replaced by `MetamonBeliefMultiTaskAgent`.
- A `Gen1OpponentTeamBeliefHead` is added.
- The actor input grows by `belief_dim = 64`.
- The dataset emits auxiliary `belief_*` target tensors.
- Training includes the weighted auxiliary belief loss.

What stays the same in the 15M control comparison:

- Same `pac-tauros` data mix.
- Same grouped observation space.
- Same action space used for the 15M PlasticTauros sweep.
- Same aggressive shaped reward.
- Same ISAdvantageFilter filtered offline actor loss.
- Same PopArt/two-hot critic family.
- No dynamic damping.
- No EMA.

## FBC Metric Note

The W&B metric `Pct. of Actions Approved by Binary FBC Filter` is misleading
for these Tauros runs. The recipe uses `ISAdvantageFilter`, a positive
continuous weighting filter. AMAGO computes the logged approval as
`filter > 0`, so it reports `100%` whenever the positive weight floor is active.
The meaningful filter metrics are the mean and percentiles, not the binary
approval percentage.

## H2H Helper

The local H2H helper for the belief checkpoint is:

```text
scripts/eval_grouped_belief_tauros_h2h.py
```

The 50-battle result was written to:

```text
/home/eddie/metamon/evals/plastic_tauros_15m_belief_vs_taurosv0/summary.json
```

## Recommended Next Steps

1. Run a 200+ battle H2H for `grouped_belief_control_150k` checkpoint 5.
2. Add heldout `pac-tauros` loss and broad Gen1OU heldout loss.
3. Check heldout belief-head quality and embedding norm behavior.
4. If the H2H advantage holds, scale the belief-control recipe to TaurosV0
   architecture before revisiting damping.

## TaurosV0-Scale Belief Run

The TaurosV0-scale belief control configs are:

```text
metamon/rl/configs/models/grouped_v2_50m_belief.gin
metamon/rl/configs/training/plastic_tauros_50m_belief_control.gin
```

The model config mirrors `grouped_v2_50m.gin` and swaps the agent binding to
`MetamonBeliefMultiTaskAgent`. The train config mirrors
`grouped_v2_large_isfilter.gin` and binds the same large-run hparams to the
belief agent: `learning_rate = 1e-4`, `grad_clip = 1.0`, and
`lr_warmup_steps = 5000`.

Launch command:

```bash
METAMON_CACHE_DIR=/home/eddie/metamon_cache \
METAMON_WANDB_ENTITY=costacosta-personal-research \
METAMON_WANDB_PROJECT=metamon \
uv run python -m metamon.rl.train \
  --run_name grouped_belief_50m_control_tauros_scale \
  --save_dir /home/eddie/metamon/models/plastic-tauros-50m-belief \
  --model_gin_config grouped_v2_50m_belief.gin \
  --train_gin_config plastic_tauros_50m_belief_control.gin \
  --dataset_config gen1ou_small_tauros_core_belief.yaml \
  --obs_space GroupedObservationSpace \
  --action_space DefaultActionSpace \
  --reward_function AggressiveShapedReward \
  --epochs 63 \
  --steps_per_epoch 25000 \
  --batch_size_per_gpu 12 \
  --grad_accum 1 \
  --dloader_workers 10 \
  --ckpt_interval 1 \
  --eval_gens 1 \
  --async_env_mp_context forkserver \
  --log
```

This uses `DefaultActionSpace` to match TaurosV0. Use checkpoint 62 for the
primary comparison, matching TaurosV0's registered default checkpoint.
