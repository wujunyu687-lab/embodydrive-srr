#!/usr/bin/env bash
set -euo pipefail

: "${OFFLINE_ROOT:?Set OFFLINE_ROOT to the processed wrist dataset}"
: "${MODEL_ROOT:?Set MODEL_ROOT to Wan2.1-T2V-1.3B}"
: "${VAE_PATH:?Set VAE_PATH to Wan2.1_VAE.pth}"
: "${CHECKPOINT:?Set CHECKPOINT to the candidate checkpoint directory}"
: "${EVAL_OUTPUT:?Set EVAL_OUTPUT to the candidate evaluation directory}"
: "${EPISODE_IDS:?Set EPISODE_IDS to comma-separated validation episode ids}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a episodes <<< "$EPISODE_IDS"

run_checkpoint() {
  local checkpoint="$1"
  local output="$2"
  mkdir -p "$output"
  for episode in "${episodes[@]}"; do
    python -m embodydrive.diagnose_rollout \
      --offline-root "$OFFLINE_ROOT" \
      --episode-id "$episode" \
      --checkpoint "$checkpoint" \
      --model-root "$MODEL_ROOT" \
      --vae-path "$VAE_PATH" \
      --output-dir "$output/$episode" \
      --history-latents "${HISTORY_LATENTS:-1}" \
      --chunk-latents "${CHUNK_LATENTS:-1}" \
      --rollout-steps "${ROLLOUT_STEPS:-4}" \
      --ensemble-size "${ENSEMBLE_SIZE:-4}" \
      --ensemble-mode "${ENSEMBLE_MODE:-mean}" \
      --seed "${SEED:-1234}"
  done

  local validation_args=()
  for episode in "${episodes[@]}"; do
    validation_args+=(--episode-id "$episode")
  done
  python -m embodydrive.validate_rollout_artifacts \
    --offline-root "$OFFLINE_ROOT" \
    --root "$output" \
    --output "$output/artifacts.json" \
    "${validation_args[@]}"
}

run_checkpoint "$CHECKPOINT" "$EVAL_OUTPUT"

if [[ -n "${BASELINE_CHECKPOINT:-}" ]]; then
  : "${BASELINE_OUTPUT:?Set BASELINE_OUTPUT when BASELINE_CHECKPOINT is set}"
  run_checkpoint "$BASELINE_CHECKPOINT" "$BASELINE_OUTPUT"
  gate_args=()
  for episode in "${episodes[@]}"; do
    gate_args+=(--episode-id "$episode")
  done
  python -m embodydrive.gate_rollouts \
    --baseline-root "$BASELINE_OUTPUT" \
    --candidate-root "$EVAL_OUTPUT" \
    --output "$EVAL_OUTPUT/gate.json" \
    "${gate_args[@]}"
fi
