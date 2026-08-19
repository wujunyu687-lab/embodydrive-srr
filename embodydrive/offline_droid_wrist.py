"""Offline DROID wrist-camera conversion for 5Hz, 192x320 training data.

The converter keeps one wrist MP4 and one aligned JSON annotation per episode.
It is resumable and can partition work by shard for multi-process execution.
Wan VAE encoding is intentionally a separate stage so the pixel conversion can
be inspected before spending GPU time on latent extraction.
"""

import argparse
import hashlib
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tfrecord.reader import tfrecord_loader


WRIST_KEY = "steps/observation/wrist_image_left"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--source-fps", type=int, default=15)
    parser.add_argument("--target-fps", type=int, default=5)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all complete shards")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all episodes")
    parser.add_argument("--shard-index", type=int, default=-1, help="Process only this sorted shard index")
    parser.add_argument("--num-shard-workers", type=int, default=1)
    parser.add_argument("--shard-worker-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def decode_and_resize(encoded, width, height):
    with Image.open(BytesIO(bytes(encoded).rstrip(b"\x00"))) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def aligned(record, key, raw_frames, dim):
    values = np.asarray(record[key], dtype=np.float32)
    return values.reshape(raw_frames, dim)


def episode_id(shard, record_index):
    source = f"{shard.name}:{record_index}".encode()
    return hashlib.sha1(source).hexdigest()[:16]


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if args.target_fps <= 0 or args.source_fps <= 0 or args.source_fps % args.target_fps != 0:
        raise ValueError("source-fps must be a positive multiple of target-fps")
    stride = args.source_fps // args.target_fps
    if args.num_shard_workers <= 0 or not 0 <= args.shard_worker_index < args.num_shard_workers:
        raise ValueError("invalid shard worker configuration")

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    shards = sorted(
        p for p in data_root.glob("droid_101-train.tfrecord-*-of-02048")
        if not p.name.endswith(".part")
    )
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    if args.shard_index >= 0:
        shards = [shards[args.shard_index]] if args.shard_index < len(shards) else []
    else:
        shards = [p for i, p in enumerate(shards) if i % args.num_shard_workers == args.shard_worker_index]
    if not shards:
        raise FileNotFoundError("No shards selected for this worker")

    processed = 0
    skipped = 0
    invalid = 0
    for shard in shards:
        for record_index, record in enumerate(tfrecord_loader(str(shard), None)):
            if args.max_episodes > 0 and processed >= args.max_episodes:
                break
            wrist = record.get(WRIST_KEY)
            if wrist is None or len(wrist) < stride:
                invalid += 1
                continue
            raw_frames = len(wrist)
            try:
                action = aligned(record, "steps/action", raw_frames, 7)
                joint = aligned(record, "steps/observation/joint_position", raw_frames, 7)
                cartesian = aligned(record, "steps/observation/cartesian_position", raw_frames, 6)
                gripper = aligned(record, "steps/observation/gripper_position", raw_frames, 1)
            except (KeyError, ValueError):
                invalid += 1
                continue

            eid = episode_id(shard, record_index)
            split = "val" if int(eid[-2:], 16) % 100 == 0 else "train"
            video_path = output_root / "videos" / split / eid / "wrist.mp4"
            annotation_path = output_root / "annotation" / split / f"{eid}.json"
            if video_path.exists() and annotation_path.exists():
                skipped += 1
                continue

            sample_indices = np.arange(0, raw_frames, stride, dtype=np.int64)
            frames = [
                decode_and_resize(wrist[int(index)], args.width, args.height)
                for index in sample_indices
            ]
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                str(video_path),
                frames,
                fps=args.target_fps,
                codec="libx264",
                macro_block_size=1,
            )

            proprio = np.concatenate(
                [joint, cartesian, np.nan_to_num(gripper, nan=0.0, posinf=0.0, neginf=0.0)],
                axis=-1,
            )
            instruction = record.get("steps/language_instruction")
            instruction = "" if instruction is None else bytes(instruction[0]).decode("utf-8", "ignore")
            annotation = {
                "episode_id": eid,
                "split": split,
                "source_shard": shard.name,
                "source_record_index": record_index,
                "raw_frame_count": raw_frames,
                "video_length": len(frames),
                "raw_size": [320, 180],
                "processed_size": [args.width, args.height],
                "source_fps": args.source_fps,
                "processed_fps": args.target_fps,
                "frame_stride": stride,
                "instruction": instruction,
                "video_path": f"videos/{split}/{eid}/wrist.mp4",
                "action": action[sample_indices].tolist(),
                "proprio": proprio[sample_indices].tolist(),
            }
            write_json_atomic(annotation_path, annotation)
            processed += 1
            if processed % args.log_every == 0:
                print(
                    f"worker={args.shard_worker_index}/{args.num_shard_workers} "
                    f"processed={processed} skipped={skipped} invalid={invalid} "
                    f"last={split}/{eid} frames={raw_frames}->{len(frames)}",
                    flush=True,
                )
        print(
            f"completed shard={shard.name} worker={args.shard_worker_index} "
            f"processed={processed} skipped={skipped} invalid={invalid}",
            flush=True,
        )

    print(
        f"done worker={args.shard_worker_index} processed={processed} "
        f"skipped={skipped} invalid={invalid} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
