"""Autoregressive full-episode inference for the trained G0 wrist model."""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embodydrive.train_g0 import build_transformer
from videox_fun.models.wan_vae import AutoencoderKLWan
from embodydrive.rollout import (
    adaptive_transition_stabilize,
    euler_sample_chunk,
    select_continuous_candidate,
    velocity_transition_blend,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--offline-root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--episode-id", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-root", required=True)
    p.add_argument("--vae-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--history-latents", type=int, default=1)
    p.add_argument("--chunk-latents", type=int, default=1)
    p.add_argument("--rollout-steps", type=int, default=4)
    p.add_argument(
        "--condition-alignment",
        choices=("global", "local", "causal"),
        default="causal",
        help="Temporal condition alignment; causal matches Wan's first-frame-plus-four-frame VAE grouping",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--noise-mode",
        choices=("independent", "shared", "zero"),
        default="independent",
        help="Latent initialization across autoregressive chunks",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Scale the initial latent noise; 1.0 is the trained flow-matching distribution",
    )
    p.add_argument("--ensemble-size", type=int, default=4)
    p.add_argument(
        "--ensemble-mode", choices=("mean", "continuous-select"), default="mean",
        help="Combine stochastic candidates by latent mean or continuous-path selection",
    )
    p.add_argument(
        "--latent-blend",
        type=float,
        default=0.0,
        help="Blend each generated latent toward the latest history latent",
    )
    p.add_argument("--adaptive-stabilize", action="store_true")
    p.add_argument(
        "--velocity-blend", type=float, default=0.0,
        help="Optional blend toward recent velocity continuation; default 0 is validated with ensemble=4",
    )
    p.add_argument("--velocity-reference-frames", type=int, default=4)
    p.add_argument("--stability-reference-frames", type=int, default=4)
    p.add_argument("--stability-motion-ratio", type=float, default=2.0)
    return p.parse_args()


def normalize(values, stats, kind):
    p01 = torch.tensor(stats[kind]["p01"], dtype=values.dtype)
    p99 = torch.tensor(stats[kind]["p99"], dtype=values.dtype)
    return ((2.0 * (values - p01) / (p99 - p01 + 1e-8)) - 1.0).clamp(-1.0, 1.0)


def aggregate_causal_raw(raw, kind):
    """Aggregate raw conditions on Wan's first-frame-plus-four clock."""
    latent_total = 1 + max(0, (raw.shape[0] - 1) // 4)
    required = 1 + max(0, latent_total - 1) * 4
    values = raw.float()
    if values.shape[0] < required:
        values = torch.cat(
            [values, values[-1:].repeat(required - values.shape[0], 1)], dim=0
        )
    values = values[:required]
    grouped = [values[:1]]
    if latent_total > 1:
        grouped_values = values[1:].reshape(latent_total - 1, 4, values.shape[-1])
        grouped_mean = grouped_values.mean(dim=1)
        circular_dims = (3, 4, 5) if kind == "action" else (10, 11, 12)
        dims = torch.as_tensor(circular_dims, device=values.device)
        angles = grouped_values.index_select(-1, dims)
        circular = torch.atan2(
            torch.sin(angles).mean(dim=1), torch.cos(angles).mean(dim=1)
        )
        grouped_mean = grouped_mean.clone()
        grouped_mean.index_copy_(-1, dims, circular)
        grouped.append(grouped_mean)
    return torch.cat(grouped, dim=0)


def local_conditions(raw, latent_start, total_latents, stats, kind):
    """Match training: downsample the complete raw sequence, then slice it.

    ``train_srr.encode_batch`` calls ``downsample_temporal`` before any
    rollout-window slicing. Re-padding each raw sub-window would change the
    temporal phase at every nonzero latent start and shift action/proprio
    relative to the generated latent history.
    """
    grouped = aggregate_causal_raw(raw, kind)
    window = grouped[latent_start : latent_start + total_latents]
    if window.shape[0] < total_latents:
        window = torch.cat(
            [window, window[-1:].repeat(total_latents - window.shape[0], 1)], dim=0
        )
    return normalize(window.unsqueeze(0), stats, kind)


def local_phase_conditions(raw, latent_start, total_latents, stats, kind):
    """Downsample a cropped raw window with a fresh local temporal phase."""
    raw_start = latent_start * 4
    required_raw = total_latents * 4 - 3
    values = raw[raw_start : raw_start + required_raw]
    if values.shape[0] == 0:
        values = raw[-1:].clone()
    if values.shape[0] < required_raw:
        values = torch.cat(
            [values, values[-1:].repeat(required_raw - values.shape[0], 1)], dim=0
        )
    grouped = aggregate_causal_raw(values, kind)
    grouped = grouped[:total_latents]
    if grouped.shape[0] < total_latents:
        grouped = torch.cat(
            [grouped, grouped[-1:].repeat(total_latents - grouped.shape[0], 1)], dim=0
        )
    return normalize(grouped.unsqueeze(0), stats, kind)


def causal_conditions(raw, latent_start, total_latents, stats, kind):
    """Align conditions to Wan's causal VAE temporal groups.

    Wan encodes the first frame separately, then consumes raw frames in groups
    of four.  The previous helper repeated the first frame four times before
    grouping, which shifted every group after the first one by one frame.
    """
    grouped = aggregate_causal_raw(raw, kind)
    window = grouped[latent_start : latent_start + total_latents]
    if window.shape[0] < total_latents:
        window = torch.cat(
            [window, window[-1:].repeat(total_latents - window.shape[0], 1)], dim=0
        )
    return normalize(window.unsqueeze(0), stats, kind)


def write_mp4(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264", macro_block_size=1)


@torch.no_grad()
def main():
    args = parse_args()
    if args.noise_scale < 0.0:
        raise ValueError("noise-scale must be non-negative")
    if args.ensemble_size <= 0:
        raise ValueError("ensemble-size must be positive")
    if not 0.0 <= args.latent_blend <= 1.0:
        raise ValueError("latent-blend must be in [0, 1]")
    if not 0.0 <= args.velocity_blend <= 1.0:
        raise ValueError("velocity-blend must be in [0, 1]")
    if args.stability_reference_frames <= 0 or args.stability_motion_ratio <= 0:
        raise ValueError("stability parameters must be positive")
    # HorizonDrive's model loader uses accelerate.logging even for single-GPU
    # inference, so initialize the local accelerate state before construction.
    Accelerator(mixed_precision="bf16")
    root = Path(args.offline_root)
    annotation_paths = sorted((root / "annotation" / args.split).glob("*.json"))
    if args.episode_id:
        annotation_paths = [p for p in annotation_paths if p.stem == args.episode_id]
    if not annotation_paths:
        raise FileNotFoundError("No matching episode annotation")
    annotation_path = annotation_paths[0]
    annotation = json.loads(annotation_path.read_text())
    episode_id = annotation["episode_id"]

    device = torch.device("cuda")
    dtype = torch.bfloat16
    stats = json.loads((root / "stats.json").read_text())

    vae = AutoencoderKLWan.from_pretrained(
        args.vae_path,
        additional_kwargs={"temporal_compression_ratio": 4, "spatial_compression_ratio": 8},
    ).to(device, dtype=dtype).eval()
    vae.requires_grad_(False)
    model = build_transformer(args.model_root, True, True, dtype).to(device).eval()
    state = load_file(str(Path(args.checkpoint) / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in missing
        if key
        not in {
            "action.token_projection.weight",
            "action.token_projection.bias",
            "proprio.modulation_projection.weight",
            "proprio.modulation_projection.bias",
            "action.circular_embedding.weight",
            "action.circular_embedding.bias",
            "proprio.circular_embedding.weight",
            "proprio.circular_embedding.bias",
        }
    ]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    latent_path = root / annotation["latent_path"]
    latents = torch.load(latent_path, map_location="cpu").contiguous()
    # Cache layout is [time, channels, height, width].
    latents = latents.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)
    total_latents = latents.shape[2]
    history = args.history_latents
    if total_latents <= history:
        raise RuntimeError(f"episode too short: latent frames={total_latents}")

    action_raw = torch.tensor(annotation["action"], dtype=torch.float32)
    proprio_raw = torch.tensor(annotation["proprio"], dtype=torch.float32)
    generated = [latents[:, :, i : i + 1] for i in range(history)]
    current = history
    chunk_index = 0
    shared_noise = None
    if args.noise_mode == "shared":
        shared_generator = torch.Generator(device=device).manual_seed(args.seed)
        shared_noise = torch.randn(
            1,
            latents.shape[1],
            1,
            latents.shape[3],
            latents.shape[4],
            device=device,
            dtype=dtype,
            generator=shared_generator,
        )
    while current < total_latents:
        future = min(args.chunk_latents, total_latents - current)
        history_latents = torch.cat(generated[-history:], dim=2)
        local_start = current - history
        total_window = history + future
        if args.condition_alignment == "global":
            condition_fn = local_conditions
        elif args.condition_alignment == "local":
            condition_fn = local_phase_conditions
        else:
            condition_fn = causal_conditions
        action = condition_fn(action_raw, local_start, total_window, stats, "action").to(device=device, dtype=dtype)
        proprio = condition_fn(proprio_raw, local_start, total_window, stats, "proprio").to(device=device, dtype=dtype)
        if args.noise_mode == "zero":
            noise = torch.zeros(
                1,
                latents.shape[1],
                future,
                latents.shape[3],
                latents.shape[4],
                device=device,
                dtype=dtype,
            )
        elif args.noise_mode == "shared":
            noise = shared_noise.expand(-1, -1, future, -1, -1).clone()
        else:
            generator = torch.Generator(device=device).manual_seed(args.seed + chunk_index)
            noise = torch.randn(
                1,
                latents.shape[1],
                future,
                latents.shape[3],
                latents.shape[4],
                device=device,
                dtype=dtype,
                generator=generator,
            )
        noise = noise * args.noise_scale
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            members = []
            for member in range(args.ensemble_size):
                if member == 0:
                    member_noise = noise
                else:
                    member_generator = torch.Generator(device=device).manual_seed(
                        args.seed + chunk_index * args.ensemble_size + member
                    )
                    member_noise = torch.randn(
                        1, latents.shape[1], future, latents.shape[3], latents.shape[4],
                        device=device, dtype=dtype, generator=member_generator
                    ) * args.noise_scale
                members.append(euler_sample_chunk(
                    model,
                    history_latents,
                    action,
                    proprio,
                    future,
                    args.rollout_steps,
                    noise=member_noise,
                ))
            candidates = torch.stack(members, dim=0)
            if args.ensemble_mode == "continuous-select":
                future_latents = select_continuous_candidate(candidates, history_latents)
            else:
                future_latents = candidates.mean(dim=0)
        if args.latent_blend > 0.0:
            future_latents = (
                (1.0 - args.latent_blend) * future_latents
                + args.latent_blend * history_latents[:, :, -1:]
            )
        if args.velocity_blend > 0.0:
            future_latents = velocity_transition_blend(
                future_latents,
                history_latents,
                args.velocity_blend,
                args.velocity_reference_frames,
            )
        if args.adaptive_stabilize:
            future_latents = adaptive_transition_stabilize(
                future_latents,
                history_latents,
                reference_frames=args.stability_reference_frames,
                max_motion_ratio=args.stability_motion_ratio,
            )
        generated.extend([future_latents[:, :, i : i + 1] for i in range(future)])
        current += future
        chunk_index += 1
        print(f"episode={episode_id} latent={current}/{total_latents} chunk={chunk_index}", flush=True)

    predicted_latents = torch.cat(generated, dim=2)
    decoded = vae.decode(predicted_latents).sample
    predicted = ((decoded.float().clamp(-1, 1) + 1.0) * 127.5).byte()[0]
    predicted = predicted.permute(1, 2, 3, 0).cpu().numpy()

    video_path = root / annotation["video_path"]
    gt = np.asarray(imageio.mimread(str(video_path)), dtype=np.uint8)
    frame_count = min(len(gt), len(predicted))
    gt = gt[:frame_count]
    predicted = predicted[:frame_count]
    compare = np.concatenate([predicted, gt], axis=2)
    output = Path(args.output_dir) / episode_id
    write_mp4(output / "pred_full.mp4", predicted, annotation.get("processed_fps", 5))
    write_mp4(output / "gt_full.mp4", gt, annotation.get("processed_fps", 5))
    write_mp4(output / "compare_full.mp4", compare, annotation.get("processed_fps", 5))
    metadata = {
        "episode_id": episode_id,
        "annotation": str(annotation_path),
        "checkpoint": args.checkpoint,
        "latent_frames": total_latents,
        "predicted_frames": len(predicted),
        "gt_frames": len(gt),
        "fps": annotation.get("processed_fps", 5),
        "history_latents": history,
        "chunk_latents": args.chunk_latents,
        "rollout_steps": args.rollout_steps,
        "condition_alignment": args.condition_alignment,
        "noise_mode": args.noise_mode,
        "noise_scale": args.noise_scale,
        "ensemble_size": args.ensemble_size,
        "ensemble_mode": args.ensemble_mode,
        "latent_blend": args.latent_blend,
        "velocity_blend": args.velocity_blend,
        "velocity_reference_frames": args.velocity_reference_frames,
        "adaptive_stabilize": args.adaptive_stabilize,
        "stability_reference_frames": args.stability_reference_frames,
        "stability_motion_ratio": args.stability_motion_ratio,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
