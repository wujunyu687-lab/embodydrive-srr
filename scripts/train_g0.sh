#!/usr/bin/env bash
set -euo pipefail

: "${OFFLINE_ROOT:?Set OFFLINE_ROOT to the processed wrist dataset}"
: "${MODEL_ROOT:?Set MODEL_ROOT to Wan2.1-T2V-1.3B}"
: "${VAE_PATH:?Set VAE_PATH to Wan2.1_VAE.pth}"
: "${G0_OUTPUT:?Set G0_OUTPUT to the checkpoint output directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
G0_MAX_STEPS="${G0_MAX_STEPS:-3000}"

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  -m embodydrive.train_g0 \
  --offline-root "$OFFLINE_ROOT" \
  --stats-path "$OFFLINE_ROOT/stats.json" \
  --model-root "$MODEL_ROOT" \
  --vae-path "$VAE_PATH" \
  --output-dir "$G0_OUTPUT" \
  --frames 17 \
  --history-latents 1 \
  --batch-size 8 \
  --val-batch-size 1 \
  --learning-rate 1e-5 \
  --max-steps "$G0_MAX_STEPS" \
  --checkpointing-steps 500 \
  --validation-steps 500 \
  --visualization-steps 500 \
  --visualization-rollout-steps 8 \
  --gradient-checkpointing
