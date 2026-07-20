# Lapras Finetuning Checkpoint

Date: 2026-05-28

This document is a handoff checkpoint for the Articuno -> Lapras finetuning
baseline and the next research steps. The short version is that Lapras
specialization is real, but the public baseline should be treated as a
positive control rather than the main continued-finetuning recipe.

## Objective

Build a Lapras-team Gen 1 OU specialist by finetuning the local `Articuno`
model on the Lapras-perspective dataset. The target objective should move
toward KL-anchored advantage-weighted BC, not aggressive offline RL.

The public `metamon/rl/configs/training/finetune.gin` run proves that useful
Lapras-specific improvements exist. It should not be used as the default
continuation regime because it lets a strong base policy move too far on a
narrow dataset.

## Data

Primary dataset config:

```text
metamon/rl/configs/datasets/lapras_only.yaml
```

It uses:

```yaml
formats:
  - gen1ou
replay_weight: 0.0
custom_replays:
  - dir: /home/eddie/metamon/trajectories/lapras/splits/v1/train/lapras
    weight: 1.0
```

The materialized training split is Lapras perspective only, but it is not
wins-only:

- Train split: 60,072 battles.
- Win/loss count from filenames: 33,069 wins / 27,003 losses.
- Approximate Lapras win rate in train split: 55.1%.

Full dataset report:

```text
/home/eddie/metamon/trajectories/lapras/dataset_report.json
```

## Baseline Training Command

The baseline run was launched with the public `finetune.gin` objective:

```bash
METAMON_CACHE_DIR=/home/eddie/metamon_cache \
uv run python -m metamon.rl.finetune \
  --run_name lapras_articuno_public_finetune_baseline \
  --save_dir /home/eddie/metamon/models/lapras-specialist \
  --base_model Articuno \
  --base_checkpoint 40 \
  --train_gin_config finetune.gin \
  --dataset_config lapras_only.yaml \
  --epochs 20 \
  --steps_per_epoch 1000 \
  --ckpt_interval 1 \
  --eval_gens 1 \
  --lapras_tauros_eval \
  --lapras_tauros_eval_battles 100 \
  --lapras_tauros_eval_agent_team_set lapras \
  --lapras_tauros_eval_opponent_team_set competitive \
  --log
```

Note: the observed progress bars showed `2000/2000` train batches per epoch,
so confirm the effective `steps_per_epoch` / `grad_accum` values from W&B or
the saved config if exact step accounting matters.

## Public `finetune.gin` Interpretation

Config:

```text
metamon/rl/configs/training/finetune.gin
```

Important settings:

- `MetamonAMAGOExperiment.agent_type = @custom_agent.MetamonFinetuneAgent`
  adds a frozen base snapshot, BC actor, and slow EMA tortoise copy.
- `online_coeff = 0.0`, `offline_coeff = 1.0` makes this a pure offline
  finetune.
- `reward_multiplier = 10.0` strengthens the reward/value scale.
- `ISAdvantageFilter.beta = 3.0` uses relatively sharp advantage weighting.
- Sequence filtering is enabled with `seq_p_low = 0.4`,
  `seq_p_full = 0.8`, and `seq_floor = 0.05`, which downweights lower
  advantage trajectories.
- `bc_coeff = 1.0` adds auxiliary BC loss for the behavior/data actor.
- `use_is_correction = False`, so the configured base-vs-data importance
  correction is skipped in this public baseline.
- `tortoise_tau = 0.005`, but `use_tortoise_for_inference = False`, so
  evaluation uses the live hare weights, not the EMA copy.
- `learning_rate = 8e-5`, `lr_warmup_steps = 5000`, `l2_coeff = 1e-4`,
  and `grad_clip = 1.5` make the optimization moderately conservative.
- No KL anchor/damping is active in this baseline.

Interpretation: this is a useful naive control run, but it is structurally
unsafe for continued finetuning. The objective is pure offline, uses a large
reward scale, sharp advantage weighting, sequence filtering that can over-focus
high-return routes, live hare inference instead of EMA inference, and no active
KL anchor/damping. It is expected to drift from Articuno and should be compared
against KL-anchored or dynamically damped variants.

## Built-In TaurosV0 Evaluation Hook

An opt-in training hook was added so finetuning can run a 100-game H2H eval
after checkpoint saves:

- Current local checkpoint plays with team set `lapras`.
- Opponent is `TaurosV0`, default checkpoint, with team set `competitive`.
- Results are logged into the same W&B run under:
  - `lapras-taurosv0-eval/lapras_win_rate`
  - `lapras-taurosv0-eval/lapras_wins`
  - `lapras-taurosv0-eval/taurosv0_wins`
  - `lapras-taurosv0-eval/total_battles`

Code touched:

- `metamon/rl/finetune.py`
- `metamon/rl/train.py`
- `metamon/rl/metamon_to_amago.py`
- `metamon/rl/evaluate/common.py`
- `metamon/rl/evaluate/serve_matchup.py`

The H2H worker path was extended so matchup subprocesses can load local
finetune checkpoints via `LocalFinetunedModel`, instead of requiring every
checkpoint to be pre-registered.

## Baseline Results

Base Articuno with the Lapras team was reported at about 56% win rate against
TaurosV0.

Public finetune baseline H2H results versus TaurosV0:

| Checkpoint | Lapras Wins | TaurosV0 Wins | Lapras Win Rate | Val Avg Return |
| --- | ---: | ---: | ---: | ---: |
| 0 | 69 | 31 | 69% | 195.755 |
| 1 | 63 | 37 | 63% | 197.324 |
| 2 | 64 | 36 | 64% | 199.393 |
| 3 | 68 | 32 | 68% | 199.207 |
| 4 | 69 | 31 | 69% | 203.237 |
| 5 | 64 | 36 | 64% | 176.527 |
| 6 | 56 | 44 | 56% | 194.790 |
| 7 | 67 | 33 | 67% | 207.018 |
| 8 | 64 | 36 | 64% | 173.267 |
| 9 | 56 | 44 | 56% | not recorded in pasted output |

Initial selection: checkpoint 4 was selected as the human-ladder baseline
candidate because it tied the best TaurosV0 result, had stronger validation
return than checkpoint 0, and appeared before later instability. Later ladder
testing invalidated that choice: checkpoint 4 posted a low ladder win rate.
The registered public baseline now defaults to checkpoint 0 instead.

The checkpoint table shows the expected failure mode: strong early gains, then
regressions back to the Articuno-on-Lapras smoke-test level at checkpoints 6
and 9. The correct interpretation is that the naive run found real
Lapras-specific improvements, but its objective is too unconstrained to be the
main training regime.

## Registered Ladder Alias

The selected baseline was registered in:

```text
metamon/rl/gen1_binary_models.py
```

Registry name:

```text
lapras_public_baseline
```

It points to:

```text
/home/eddie/metamon/models/lapras-specialist/lapras_articuno_public_finetune_baseline
```

Default checkpoint:

```text
0
```

Use in ladder/eval commands:

```bash
--agent lapras_public_baseline
```

Verification already performed:

- Registry lookup resolves `lapras_public_baseline`.
- `default_checkpoint == 0`.
- Checkpoint path exists:

```text
/home/eddie/metamon/models/lapras-specialist/lapras_articuno_public_finetune_baseline/ckpts/policy_weights/policy_epoch_0.pt
```

## Research Questions

Primary question:

Can a KL-anchored or dynamically damped Articuno -> Lapras finetune preserve
the early TaurosV0 gains while reducing instability and improving human-ladder
robustness?

Concerns with the public baseline:

- No KL anchor, so it can drift from Articuno's broader Gen 1 competence.
- Sequence filtering may overemphasize high-return/winning trajectories.
- Lapras-only team distribution can overfit to a narrow state/action support.
- TaurosV0 H2H is useful but not a complete proxy for human ladder.
- Validation return alone is not enough to promote a checkpoint because the
  public run already showed high validation return alongside unstable H2H.

## Target Training Objective

Use Ataraxos as a late-stage lesson, not as a license for aggressive updates:
when the policy is already strong, weak regularization is only appropriate when
update size is also small. In this project that means:

- Small learning rate.
- Small trainable surface.
- KL to Articuno.
- KL to previous checkpoint or epoch-start policy.
- Entropy floor or entropy-collapse guardrail.
- Conservative advantage filter.
- EMA/tortoise evaluation.
- Promotion by evaluation, not validation return alone.

Current implementation note: `MetamonFinetuneAgent` supports KL to the frozen
base snapshot via `kl_anchor_coeff`, optional KL to an epoch-start policy
snapshot via `epoch_start_kl_coeff`, high-percentile KL logging, and policy
entropy logging. An explicit entropy floor is still a target addition.

## First KL-Damped Actor-Only Run

The first KL-damped actor-only rerun from Articuno did not reproduce the
public baseline specialization effect. It was run as:

```text
lapras_articuno40_kl_anchor_actor_phase1
```

Configuration:

- `--base_model Articuno`
- `--base_checkpoint 40`
- `--train_gin_config lapras_bc_kl_anchor.gin`
- `--dataset_config lapras_only.yaml`
- `--trainable_patterns actor`
- 8 epochs, 1000 steps per epoch

TaurosV0 100-game H2H results:

| Checkpoint | Lapras Wins | TaurosV0 Wins | Lapras Win Rate | Val Avg Return |
| --- | ---: | ---: | ---: | ---: |
| 0 | 45 | 55 | 45% | 201.270 |
| 1 | 49 | 51 | 49% | 208.300 |
| 2 | 47 | 53 | 47% | 201.430 |
| 3 | 57 | 43 | 57% | 202.205 |
| 4 | 53 | 47 | 53% | 195.733 |
| 5 | 48 | 52 | 48% | 208.143 |
| 6 | 46 | 54 | 46% | 190.357 |
| 7 | 51 | 49 | 51% | not recorded in pasted output |

Interpretation: this run was stable in the sense that it avoided the public
baseline's large upward spike, but it was also worse than the public baseline
and roughly at or below the Articuno-on-Lapras smoke-test level. The exact
actor-only KL-damped BC recipe should not be treated as the next champion path.
The likely failure mode is under-adaptation: the actor-only trainable surface
plus fake-filtered BC and KL/EMA damping was too conservative to move toward
the useful Lapras-specific behavior found by `finetune.gin`.

## Next Run Direction

Do not continue `lapras_articuno40_kl_anchor_actor_phase1` unless diagnostic
KL/entropy traces show that it was still meaningfully moving and simply needed
more time. Prefer a controlled relaxation that keeps the safety instrumentation
but restores some adaptation pressure.

Candidate changes:

- Train actor plus one late trajectory layer, instead of actor-only.
- Keep KL-to-Articuno logging and high-percentile KL alerts, but lower the
  anchor coefficients if mean/high-percentile KL stayed below target.
- Reintroduce a conservative advantage filter instead of pure fake-filtered BC.
- Consider reward multiplier `2.0` before returning to `10.0`.
- Keep 100-game TaurosV0 H2H as a smoke test, but do not promote without
  broader retention and ladder checks.

Prior command shape for reference:

```bash
METAMON_CACHE_DIR=/home/eddie/metamon_cache \
uv run python -m metamon.rl.finetune \
  --run_name lapras_articuno40_kl_anchor_actor_phase1 \
  --save_dir /home/eddie/metamon/models/lapras-specialist \
  --base_model Articuno \
  --base_checkpoint 40 \
  --train_gin_config lapras_bc_kl_anchor.gin \
  --dataset_config lapras_only.yaml \
  --epochs 8 \
  --steps_per_epoch 1000 \
  --ckpt_interval 1 \
  --trainable_patterns actor \
  --eval_gens 1 \
  --lapras_tauros_eval \
  --lapras_tauros_eval_battles 100 \
  --lapras_tauros_eval_agent_team_set lapras \
  --lapras_tauros_eval_opponent_team_set competitive \
  --log
```

Before launching, verify that W&B is configured and keep `--log` enabled.

## Changes From Public `finetune.gin`

Most important changes:

- `learning_rate`: `8e-5` -> `1e-5` or `2e-5`.
- `reward_multiplier`: `10.0` -> `1.0` or `2.0`.
- `ISAdvantageFilter.beta`: `3.0` -> `0.5` to `1.0`, or use a binary filter.
- `seq_p_low` / `seq_p_full`: weaken or disable initially.
- KL: add KL to Articuno and, once implemented, KL to the epoch-start policy.
- EMA: evaluate the tortoise/EMA copy, not only the hare.
- Trainable parameters: actor-only proved too conservative in the first
  KL-damped run; prefer actor plus a small late-layer unfreeze next.

The tested `lapras_bc_kl_anchor.gin` Phase 1 config used LR `1e-5`,
fake-filtered offline BC-style training, KL-to-base coefficient `0.3`,
epoch-start KL coefficient `0.1`, grad clip `1.0`, critic loss disabled, and
`use_tortoise_for_inference = True` for EMA validation/H2H evaluation. That
combination under-adapted and should be relaxed before the next serious run.

## Promotion Metrics

Do not select the next champion only by 100-game TaurosV0 H2H. Use that as a
checkpoint smoke test, then promote only after broader checks:

1. TaurosV0 H2H: 500-1000 games for candidate checkpoints.
2. Articuno retention: candidate should not become worse than
   Articuno-on-Lapras against broad agents.
3. Entropy: no collapse relative to Articuno or public checkpoint 0; inspect
   action entropy by turn bucket and game phase.
4. KL traces: mean KL and high-percentile KL. High-percentile KL matters
   because rare tactical states can drift first.
5. Invalid action rate, especially late-game.
6. Value sanity: critic should remain calibrated enough to separate winning
   and losing trajectories when a critic is active.
7. Ladder probe: only for 2-4 selected checkpoints.

Candidate next runs:

1. KL anchor to base Articuno, late-layer + actor trainable.
2. Conservative binary-filtered BC if exponential advantage weights remain
   spiky.
3. Dynamic KL damping with target KL per update/epoch, logging KL coefficient
   and observed divergence.
4. Retention-mix run using `lapras_retention.yaml` or similar, if preserving
   general Gen 1 competence matters.
5. Repeat public baseline with a smaller LR or fewer epochs if epoch 0 remains
   the only public-run checkpoint that survives broader evaluation.

Suggested evaluation protocol:

- Keep 100-game Lapras-vs-TaurosV0 eval after each checkpoint.
- Run human ladder only for a small number of selected checkpoints, not every
  checkpoint.
- Compare at least:
  - base `Articuno` with Lapras team,
  - `lapras_public_baseline` checkpoint 0,
  - best KL-anchored variant,
  - best dynamic damping variant.
