"""Precompute Wan VAE latents for the offline wrist-camera MP4 cache."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from videox_fun.models.wan_vae import AutoencoderKLWan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--temporal-ratio", type=int, default=4)
    parser.add_argument("--spatial-ratio", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def write_json_atomic(path, value):
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan latent extraction")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid world-size/rank")

    root = Path(args.data_root)
    annotations = sorted(root.glob("annotation/*/*.json"))
    annotations = [path for index, path in enumerate(annotations) if index % args.world_size == args.rank]
    if args.max_episodes > 0:
        annotations = annotations[: args.max_episodes]
    if not annotations:
        print(f"rank={args.rank}: no annotations selected", flush=True)
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(
        args.vae_path,
        additional_kwargs={
            "temporal_compression_ratio": args.temporal_ratio,
            "spatial_compression_ratio": args.spatial_ratio,
        },
    ).to(device, dtype=dtype).eval()
    vae.requires_grad_(False)

    processed = 0
    skipped = 0
    for annotation_path in annotations:
        annotation = json.loads(annotation_path.read_text())
        latent_path = root / "latent_videos" / annotation["split"] / annotation["episode_id"] / "wrist.pt"
        if latent_path.exists():
            skipped += 1
            continue
        video_path = root / annotation["video_path"]
        frames = np.asarray(imageio.mimread(str(video_path)), dtype=np.uint8)
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        video = video.to(device=device, dtype=dtype) / 127.5 - 1.0
        with torch.no_grad():
            latents = (
                vae.encode(video)
                .latent_dist.sample()[0]
                .permute(1, 0, 2, 3)
                .contiguous()
                .to("cpu", dtype=torch.bfloat16)
            )
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents, latent_path)
        annotation["latent_path"] = str(latent_path.relative_to(root))
        annotation["latent_shape"] = list(latents.shape)
        annotation["latent_temporal_ratio"] = args.temporal_ratio
        annotation["latent_spatial_ratio"] = args.spatial_ratio
        write_json_atomic(annotation_path, annotation)
        processed += 1
        if processed % args.log_every == 0:
            print(
                f"rank={args.rank}/{args.world_size} processed={processed} skipped={skipped} "
                f"last={annotation['split']}/{annotation['episode_id']} "
                f"video={tuple(video.shape)} latent={tuple(latents.shape)}",
                flush=True,
            )

    print(f"done rank={args.rank} processed={processed} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
