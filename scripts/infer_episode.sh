#!/usr/bin/env bash
set -euo pipefail

: "${OFFLINE_ROOT:?Set OFFLINE_ROOT to the processed wrist dataset}"
: "${MODEL_ROOT:?Set MODEL_ROOT to Wan2.1-T2V-1.3B}"
: "${VAE_PATH:?Set VAE_PATH to Wan2.1_VAE.pth}"
: "${CHECKPOINT:?Set CHECKPOINT to a G0 or SRR checkpoint directory}"
: "${EPISODE_ID:?Set EPISODE_ID to a validation episode id}"
: "${INFER_OUTPUT:?Set INFER_OUTPUT to an output directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m embodydrive.infer_full_episode \
  --offline-root "$OFFLINE_ROOT" \
  --split "${SPLIT:-val}" \
  --episode-id "$EPISODE_ID" \
  --checkpoint "$CHECKPOINT" \
  --model-root "$MODEL_ROOT" \
  --vae-path "$VAE_PATH" \
  --output-dir "$INFER_OUTPUT" \
  --history-latents "${HISTORY_LATENTS:-1}" \
  --chunk-latents "${CHUNK_LATENTS:-1}" \
  --rollout-steps "${ROLLOUT_STEPS:-4}" \
  --condition-alignment causal \
  --ensemble-size "${ENSEMBLE_SIZE:-4}" \
  --ensemble-mode "${ENSEMBLE_MODE:-mean}" \
  --seed "${SEED:-1234}"
