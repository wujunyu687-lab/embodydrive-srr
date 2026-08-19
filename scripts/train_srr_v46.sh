#!/usr/bin/env bash
set -euo pipefail

: "${OFFLINE_ROOT:?Set OFFLINE_ROOT to the processed wrist dataset}"
: "${MODEL_ROOT:?Set MODEL_ROOT to Wan2.1-T2V-1.3B}"
: "${VAE_PATH:?Set VAE_PATH to Wan2.1_VAE.pth}"
: "${G0_CHECKPOINT:?Set G0_CHECKPOINT to the G0 checkpoint directory}"
: "${SRR_OUTPUT:?Set SRR_OUTPUT to the checkpoint output directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
NUM_PROCESSES="${NUM_PROCESSES:-7}"

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  -m embodydrive.train_srr \
  --offline-root "$OFFLINE_ROOT" \
  --stats-path "$OFFLINE_ROOT/stats.json" \
  --model-root "$MODEL_ROOT" \
  --vae-path "$VAE_PATH" \
  --init-checkpoint "$G0_CHECKPOINT" \
  --output-dir "$SRR_OUTPUT" \
  --frames 161 \
  --history-latents 1 \
  --rollout-chunk-latents 1 \
  --rollout-depth 20 \
  --min-rollout-depth 1 \
  --rollout-steps 4 \
  --self-forcing-rollout-steps 4 \
  --self-forcing-random-position \
  --self-forcing-random-min-position 0 \
  --self-forcing-ensemble-size 1 \
  --sigma-mode euler \
  --boundary-blend-max 0.0 \
  --clean-loss-weight 0.5 \
  --endpoint-loss-weight 0.25 \
  --endpoint-loss-positions last \
  --endpoint-ensemble-size 1 \
  --condition-lr-multiplier 10.0 \
  --train-condition-only \
  --batch-size 56 \
  --learning-rate 2e-8 \
  --max-steps 3000 \
  --checkpointing-steps 200 \
  --seed 938 \
  --gradient-checkpointing

touch "$SRR_OUTPUT/TRAINING_DONE"
