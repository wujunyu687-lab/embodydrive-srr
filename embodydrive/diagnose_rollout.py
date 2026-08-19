"""Compare autoregressive and teacher-forced long rollouts on one episode."""

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

from embodydrive.infer_full_episode import causal_conditions
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
    p.add_argument("--episode-id", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-root", required=True)
    p.add_argument("--vae-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--history-latents", type=int, default=20)
    p.add_argument(
        "--chunk-latents", type=int, default=1,
        help="Number of future latent frames generated and accepted per rollout",
    )
    p.add_argument("--rollout-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--noise-scale", type=float, default=1.0,
        help="Scale the initial noise of each generated latent",
    )
    p.add_argument(
        "--ensemble-size", type=int, default=4,
        help="Stable default: average four independent samples to reduce rollout variance",
    )
    p.add_argument(
        "--ensemble-mode", choices=("mean", "continuous-select"), default="mean",
        help="How to combine ensemble samples",
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
        help="Optional blend toward recent constant-velocity continuation; default 0 is validated with ensemble=4",
    )
    p.add_argument("--velocity-reference-frames", type=int, default=4)
    p.add_argument("--stability-reference-frames", type=int, default=4)
    p.add_argument("--stability-motion-ratio", type=float, default=2.0)
    return p.parse_args()


def load_model(args, device, dtype):
    model = build_transformer(args.model_root, True, True, dtype).to(device).eval()
    state = load_file(str(Path(args.checkpoint) / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {
        "action.token_projection.weight", "action.token_projection.bias",
        "proprio.modulation_projection.weight", "proprio.modulation_projection.bias",
        "action.circular_embedding.weight", "action.circular_embedding.bias",
        "proprio.circular_embedding.weight", "proprio.circular_embedding.bias",
    }
    missing = [k for k in missing if k not in allowed]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model


def decode(vae, latents):
    video = vae.decode(latents).sample
    video = ((video.float().clamp(-1, 1) + 1.0) * 127.5).byte()[0]
    return video.permute(1, 2, 3, 0).cpu().numpy()


@torch.no_grad()
def main():
    args = parse_args()
    if not 0.0 <= args.latent_blend <= 1.0:
        raise ValueError("latent-blend must be in [0, 1]")
    if args.noise_scale < 0.0:
        raise ValueError("noise-scale must be non-negative")
    if args.ensemble_size <= 0:
        raise ValueError("ensemble-size must be positive")
    if args.chunk_latents <= 0:
        raise ValueError("chunk-latents must be positive")
    if not 0.0 <= args.velocity_blend <= 1.0:
        raise ValueError("velocity-blend must be in [0, 1]")
    if args.stability_reference_frames <= 0 or args.stability_motion_ratio <= 0:
        raise ValueError("stability parameters must be positive")
    Accelerator(mixed_precision="bf16")
    root = Path(args.offline_root)
    ann_path = root / "annotation" / "val" / f"{args.episode_id}.json"
    ann = json.loads(ann_path.read_text())
    stats = json.loads((root / "stats.json").read_text())
    device = torch.device("cuda")
    dtype = torch.bfloat16

    vae = AutoencoderKLWan.from_pretrained(
        args.vae_path,
        additional_kwargs={"temporal_compression_ratio": 4, "spatial_compression_ratio": 8},
    ).to(device, dtype=dtype).eval()
    vae.requires_grad_(False)
    model = load_model(args, device, dtype)

    latents = torch.load(root / ann["latent_path"], map_location="cpu")
    latents = latents.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)
    total = latents.shape[2]
    history = args.history_latents
    actions = torch.tensor(ann["action"], dtype=torch.float32)
    proprio = torch.tensor(ann["proprio"], dtype=torch.float32)

    def cond(raw, start, count, kind):
        return causal_conditions(raw, start, count, stats, kind).to(device=device, dtype=dtype)

    # Use identical per-chunk noise in both modes so the comparison isolates
    # history feedback rather than stochastic initialization.
    noises = []
    chunk_starts = list(range(history, total, args.chunk_latents))
    for chunk_index, current in enumerate(chunk_starts):
        future = min(args.chunk_latents, total - current)
        chunk_noises = []
        for member in range(args.ensemble_size):
            g = torch.Generator(device=device).manual_seed(
                args.seed + chunk_index * args.ensemble_size + member
            )
            chunk_noises.append(torch.randn(
                1, latents.shape[1], future, latents.shape[3], latents.shape[4],
                device=device, dtype=dtype, generator=g
            ) * args.noise_scale)
        noises.append(chunk_noises)

    outputs = {}
    per_latent = {}
    for mode in ("autoregressive", "teacher_forced"):
        generated = [latents[:, :, i:i + 1] for i in range(history)]
        errors = []
        chunk_index = 0
        current = history
        while current < total:
            future_frames = min(args.chunk_latents, total - current)
            if mode == "teacher_forced":
                hist = latents[:, :, current - history:current]
            else:
                hist = torch.cat(generated[-history:], dim=2)
            start = current - history
            action = cond(actions, start, history + future_frames, "action")
            prop = cond(proprio, start, history + future_frames, "proprio")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                members = [euler_sample_chunk(
                    model, hist, action, prop, future_frames, args.rollout_steps,
                    noise=noise,
                ) for noise in noises[chunk_index]]
            candidates = torch.stack(members, dim=0)
            if args.ensemble_mode == "continuous-select":
                future = select_continuous_candidate(candidates, hist)
            else:
                future = candidates.mean(dim=0)
            if mode == "autoregressive" and args.latent_blend > 0.0:
                # A diagnostic/deployment-time stabilizer: suppress isolated
                # frame-to-frame jumps without changing the teacher-forced
                # comparison.  The effect is measured against the full GT
                # video before deciding whether to use it in inference.
                future = (
                    (1.0 - args.latent_blend) * future
                    + args.latent_blend * hist[:, :, -1:]
                )
            if mode == "autoregressive" and args.velocity_blend > 0.0:
                future = velocity_transition_blend(
                    future, hist, args.velocity_blend, args.velocity_reference_frames
                )
            if mode == "autoregressive" and args.adaptive_stabilize:
                future = adaptive_transition_stabilize(
                    future,
                    hist,
                    reference_frames=args.stability_reference_frames,
                    max_motion_ratio=args.stability_motion_ratio,
                )
            generated.extend(
                [future[:, :, index:index + 1] for index in range(future_frames)]
            )
            target_future = latents[:, :, current:current + future_frames]
            errors.extend(
                (future.float() - target_future.float()).abs()
                .mean(dim=(0, 1, 3, 4)).cpu().tolist()
            )
            current += future_frames
            chunk_index += 1
            if chunk_index == 1 or chunk_index % 10 == 0 or current >= total:
                print(json.dumps({
                    "mode": mode,
                    "chunk": chunk_index,
                    "current_latent": current,
                    "total_latent": total,
                }), flush=True)
        pred_latents = torch.cat(generated, dim=2)
        pred = decode(vae, pred_latents)
        gt = np.asarray(imageio.mimread(str(root / ann["video_path"])), dtype=np.uint8)
        n = min(len(pred), len(gt))
        frame_err = np.abs(pred[:n].astype(np.float32) - gt[:n].astype(np.float32)).mean(axis=(1, 2, 3))
        outputs[mode] = pred
        per_latent[mode] = {
            "latent_l1_generated": errors,
            "frame_mae": frame_err.tolist(),
            "overall_mae": float(frame_err.mean()),
            "after_16s_mae": float(frame_err[min(n, 80):].mean()),
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(imageio.mimread(str(root / ann["video_path"])), dtype=np.uint8)
    fps = ann.get("processed_fps", 5)
    for mode, pred in outputs.items():
        imageio.mimsave(str(out / f"{mode}.mp4"), pred, fps=fps, codec="libx264", macro_block_size=1)
        compare = np.concatenate([pred[:len(gt)], gt[:len(pred)]], axis=2)
        imageio.mimsave(str(out / f"{mode}_compare.mp4"), compare, fps=fps, codec="libx264", macro_block_size=1)
    report = {
        "episode_id": args.episode_id,
        "latent_frames": total,
        "history_latents": history,
        "chunk_latents": args.chunk_latents,
        "rollout_steps": args.rollout_steps,
        "seed": args.seed,
        "noise_scale": args.noise_scale,
        "ensemble_size": args.ensemble_size,
        "ensemble_mode": args.ensemble_mode,
        "latent_blend": args.latent_blend,
        "velocity_blend": args.velocity_blend,
        "velocity_reference_frames": args.velocity_reference_frames,
        "adaptive_stabilize": args.adaptive_stabilize,
        "stability_reference_frames": args.stability_reference_frames,
        "stability_motion_ratio": args.stability_motion_ratio,
        "metrics": per_latent,
    }
    (out / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
