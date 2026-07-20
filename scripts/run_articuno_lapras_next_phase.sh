#!/usr/bin/env bash
set -euo pipefail

export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
export METAMON_WANDB_ENTITY="${METAMON_WANDB_ENTITY:-costacosta-personal-research}"
export METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-metamon}"
export WANDB_MODE="${WANDB_MODE:-online}"

SAVE_DIR="${SAVE_DIR:-/home/eddie/metamon/models/articuno_lapras_next_phase_fixed_anchor_targetkl}"
BASE_MODEL="${BASE_MODEL:-Articuno}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-36}"
DATASET_CONFIG="${DATASET_CONFIG:-lapras_only.yaml}"
EPOCHS="${EPOCHS:-2}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-320}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DLOADER_WORKERS="${DLOADER_WORKERS:-10}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
EVAL_GENS="${EVAL_GENS-1}"
TRAINABLE_PATTERNS="${TRAINABLE_PATTERNS:-actor traj_encoder.tformer.layers.8 traj_encoder.tformer.layers.9}"
RUNS="${RUNS:-articuno-lapras-naive-control-no-kl articuno-lapras-v3-local24-targetkl004 articuno-lapras-v3-local16-targetkl004 articuno-lapras-v3-local32-targetkl005}"
LAPRAS_TAUROS_EVAL="${LAPRAS_TAUROS_EVAL:-0}"
LAPRAS_TAUROS_EVAL_BATTLES="${LAPRAS_TAUROS_EVAL_BATTLES:-50}"

run_one() {
  local run_name="$1"
  local train_gin

  case "$run_name" in
    articuno-lapras-naive-control-no-kl)
      train_gin="articuno_lapras_finetune_no_kl_control.gin"
      ;;
    articuno-lapras-v3-local24-targetkl004)
      train_gin="articuno_lapras_finetune_v3_local24_targetkl004.gin"
      ;;
    articuno-lapras-v3-local16-targetkl004)
      train_gin="articuno_lapras_finetune_v3_local16_targetkl004.gin"
      ;;
    articuno-lapras-v3-local32-targetkl005)
      train_gin="articuno_lapras_finetune_v3_local32_targetkl005.gin"
      ;;
    *)
      echo "Unknown run: $run_name" >&2
      exit 2
      ;;
  esac

  local -a eval_gens_args=(--eval_gens)
  if [[ -n "$EVAL_GENS" ]]; then
    read -r -a eval_gens_values <<< "$EVAL_GENS"
    eval_gens_args+=("${eval_gens_values[@]}")
  fi

  local -a trainable_args=()
  if [[ -n "$TRAINABLE_PATTERNS" ]]; then
    read -r -a trainable_values <<< "$TRAINABLE_PATTERNS"
    trainable_args=(--trainable_patterns "${trainable_values[@]}")
  fi

  local -a lapras_eval_args=()
  if [[ "$LAPRAS_TAUROS_EVAL" == "1" ]]; then
    lapras_eval_args=(
      --lapras_tauros_eval
      --lapras_tauros_eval_battles "$LAPRAS_TAUROS_EVAL_BATTLES"
      --lapras_tauros_eval_agent_team_set lapras
      --lapras_tauros_eval_opponent_team_set competitive
    )
  fi

  uv run python -m metamon.rl.finetune \
    --run_name "$run_name" \
    --save_dir "$SAVE_DIR" \
    --base_model "$BASE_MODEL" \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --train_gin_config "$train_gin" \
    --dataset_config "$DATASET_CONFIG" \
    --epochs "$EPOCHS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --ckpt_interval "$CKPT_INTERVAL" \
    --batch_size_per_gpu "$BATCH_SIZE_PER_GPU" \
    --grad_accum "$GRAD_ACCUM" \
    --dloader_workers "$DLOADER_WORKERS" \
    --async_env_mp_context forkserver \
    "${eval_gens_args[@]}" \
    "${trainable_args[@]}" \
    "${lapras_eval_args[@]}" \
    --log
}

for run_name in $RUNS; do
  run_one "$run_name"
done
