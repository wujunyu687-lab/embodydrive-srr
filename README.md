# EmbodyDrive SRR Training and Inference

This repository packages the latest complete training and inference path used
for the Wan2.1-based DROID wrist-camera experiment:

- DROID wrist video conversion at 320x192 and 5 FPS
- Wan VAE latent caching and action/proprio normalization
- G0 short-horizon flow-matching training
- v46 Scheduled Rollout Recovery (SRR) training
- full-episode autoregressive inference
- paired autoregressive/teacher-forced diagnostics
- artifact validation and baseline/candidate gating

The repository contains code only. It intentionally excludes DROID data, Wan
weights, checkpoints, logs, videos, and evaluation artifacts.

## Status

The included v46 configuration reproduces the current experimental code path.
It reduces closed-loop drift relative to the G0 baseline, but it does not yet
preserve small tabletop objects reliably. Treat it as research code, not a
production robot simulator.

## Repository Layout

```text
embodydrive/                 Training, rollout, data, inference, and evaluation code
videox_fun/                  Vendored Wan/HorizonDrive model implementation
scripts/                     Ready-to-run pipeline scripts
configs/v46.env.example      Environment variable template
tests/                       CPU unit tests for temporal alignment and rollout utilities
licenses/                    Third-party license texts
```

## Environment

The validated environment is Python 3.10, CUDA 12.8, and PyTorch 2.8.0.

```bash
conda create -n embodydrive python=3.10 -y
conda activate embodydrive
python -m pip install -r requirements-cu128.txt
python -m pip install -e .
```

`block-sparse-attn` is not required by this training path and is intentionally
not included because compatible Linux/Python 3.10 wheels are not generally
available.

## Model Files

Prepare the Wan2.1-T2V-1.3B transformer and VAE without committing them:

```text
/path/to/Wan2.1-T2V-1.3B/
  config.json
  diffusion_pytorch_model.safetensors
  Wan2.1_VAE.pth
```

The transformer implementation in this repository includes the robot action,
proprioception, and visual-history adapters required by the emitted
checkpoints. Replacing it with an unmodified HorizonDrive checkout will make
those checkpoints incompatible.

## Configure Paths

```bash
cp configs/v46.env.example .env
set -a
source .env
set +a
```

Edit `.env` before running commands. It is ignored by Git.

## 1. Prepare DROID Wrist Data

The raw directory must contain complete files named like
`droid_101-train.tfrecord-00000-of-02048`. Incomplete `.part` files are skipped.

```bash
bash scripts/prepare_wrist_data.sh
```

For parallel CPU conversion, launch one process per worker with matching
`NUM_SHARD_WORKERS` and distinct `SHARD_WORKER_INDEX`. For multi-GPU latent
extraction, do the same with `LATENT_WORLD_SIZE` and `LATENT_RANK` while assigning
one CUDA device to each process. All stages are resumable.

Expected processed layout:

```text
$OFFLINE_ROOT/
  annotation/{train,val}/*.json
  videos/{train,val}/<episode>/wrist.mp4
  latent_videos/{train,val}/<episode>/wrist.pt
  stats.json
```

## 2. Train G0

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
bash scripts/train_g0.sh
```

The default script runs 3,000 optimizer steps and writes a checkpoint every 500
steps. Inspect the short `visualizations/step-*/*-compare.mp4` files before
starting SRR. If short-horizon objects are already missing, SRR will not recover
their visual detail.

## 3. Train v46 SRR

Point `G0_CHECKPOINT` at the selected G0 checkpoint, then run:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NUM_PROCESSES=7
bash scripts/train_srr_v46.sh
```

The script matches the current v46 experiment: 161 frames, one history latent,
one-latent chunks, rollout depth 20, four Euler steps, random-position
self-forcing, condition-only updates, batch size 56, and checkpoints every 200
steps.

Do not select a model from training loss alone. In the current run,
`checkpoint-00000200` had better long-horizon autoregressive metrics than
`checkpoint-00003000`.

## 4. Run One Full-Episode Inference

Set `CHECKPOINT`, `EPISODE_ID`, and `INFER_OUTPUT`, then run:

```bash
bash scripts/infer_episode.sh
```

Outputs are written below `$INFER_OUTPUT/$EPISODE_ID/`:

```text
pred_full.mp4
gt_full.mp4
compare_full.mp4
metadata.json
```

The exact v46 evaluation used `ENSEMBLE_MODE=mean`. To avoid averaging objects
across stochastic candidates during qualitative inspection, also test:

```bash
export ENSEMBLE_MODE=continuous-select
bash scripts/infer_episode.sh
```

This is an inference ablation, not a validated replacement for the published
v46 metric path.

## 5. Run Paired Rollout Evaluation

Set comma-separated validation episode IDs and run the candidate:

```bash
export EPISODE_IDS=episode_a,episode_b,episode_c
bash scripts/evaluate_rollouts.sh
```

To compare against a G0 baseline in the same invocation:

```bash
export BASELINE_CHECKPOINT=/path/to/g0/checkpoint-00003000
export BASELINE_OUTPUT=/path/to/evals/baseline
bash scripts/evaluate_rollouts.sh
```

The evaluation writes AR/TF videos, `diagnostic.json`, `artifacts.json`, and an
optional `gate.json`. `artifacts.json: PASS` only confirms complete readable
outputs. Model acceptance should use `gate.json`, visual inspection, and object
preservation checks together.

## Tests

```bash
python -m compileall -q embodydrive videox_fun tests
python -m unittest discover -s tests -v
```

GPU training and full inference require the external data and weights, so the
repository tests intentionally cover CPU-side logic and CLI/import integrity.

## Acknowledgments and License

The vendored `videox_fun` model code comes from HorizonDrive/VideoX-Fun and is
distributed under the included MIT license. Wan VAE source files retain their
upstream attribution headers. See `THIRD_PARTY_NOTICES.md` and `licenses/`.
