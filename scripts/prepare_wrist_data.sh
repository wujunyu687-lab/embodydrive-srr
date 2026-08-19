#!/usr/bin/env bash
set -euo pipefail

: "${DROID_ROOT:?Set DROID_ROOT to the DROID TFRecord directory}"
: "${OFFLINE_ROOT:?Set OFFLINE_ROOT to the processed wrist dataset directory}"
: "${VAE_PATH:?Set VAE_PATH to Wan2.1_VAE.pth}"

NUM_SHARD_WORKERS="${NUM_SHARD_WORKERS:-1}"
SHARD_WORKER_INDEX="${SHARD_WORKER_INDEX:-0}"
LATENT_WORLD_SIZE="${LATENT_WORLD_SIZE:-1}"
LATENT_RANK="${LATENT_RANK:-0}"

python -m embodydrive.offline_droid_wrist \
  --data-root "$DROID_ROOT" \
  --output-root "$OFFLINE_ROOT" \
  --width 320 --height 192 \
  --source-fps 15 --target-fps 5 \
  --num-shard-workers "$NUM_SHARD_WORKERS" \
  --shard-worker-index "$SHARD_WORKER_INDEX"

python -m embodydrive.compute_offline_stats --data-root "$OFFLINE_ROOT"

python -m embodydrive.offline_wan_latents \
  --data-root "$OFFLINE_ROOT" \
  --vae-path "$VAE_PATH" \
  --world-size "$LATENT_WORLD_SIZE" \
  --rank "$LATENT_RANK"
