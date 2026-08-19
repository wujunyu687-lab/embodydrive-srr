"""Validate that rollout videos and diagnostics are complete and readable."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def count_and_sample(path: Path, expected_width: int, expected_height: int, expected_frames: int):
    reader = imageio.get_reader(str(path))
    count = 0
    samples = []
    sample_indices = set(np.linspace(0, max(expected_frames - 1, 0), 9, dtype=int).tolist())
    try:
        for index, frame in enumerate(reader):
            if index in sample_indices:
                array = np.asarray(frame)
                if array.ndim != 3 or array.shape[0] != expected_height or array.shape[1] != expected_width:
                    raise RuntimeError(
                        f"{path}: bad frame shape {array.shape}, expected (*,{expected_height},{expected_width})"
                    )
                samples.append((float(array.mean()), float(array.std())))
            count = index + 1
    finally:
        reader.close()
    if count != expected_frames:
        raise RuntimeError(f"{path}: frame count {count} != expected {expected_frames}")
    if not samples:
        raise RuntimeError(f"{path}: no decodable frames")
    # Reject uniformly black/corrupt outputs while allowing dark scenes.
    if max(mean for mean, _ in samples) < 2.0 or max(std for _, std in samples) < 1.0:
        raise RuntimeError(f"{path}: sampled frames look uniformly black/constant")
    return count, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--episode-id",
        action="append",
        default=[],
        help="Episode to validate; repeat for multiple episodes. Defaults to output subdirectories.",
    )
    args = parser.parse_args()

    offline = Path(args.offline_root)
    root = Path(args.root)
    episodes = args.episode_id or sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and (path / "diagnostic.json").is_file()
    )
    if not episodes:
        raise FileNotFoundError(f"No rollout episodes found under {root}")
    result = {"root": str(root), "episodes": [], "status": "PASS", "errors": []}
    for episode in episodes:
        try:
            ann = json.loads((offline / "annotation" / "val" / f"{episode}.json").read_text())
            gt_path = offline / ann["video_path"]
            # diagnose_rollout.py uses mimread for the GT and computes metrics
            # against exactly that decoded sequence.  Keep the same decoder
            # and frame-count convention here; ffmpeg stream iteration can
            # expose a duplicated terminal frame for these files.
            gt_frames = np.asarray(imageio.mimread(str(gt_path)), dtype=np.uint8)
            gt_count = len(gt_frames)
            gt_shape = gt_frames[0].shape if gt_count else None
            if gt_shape is None or len(gt_shape) != 3:
                raise RuntimeError(f"{gt_path}: no decodable GT frames")
            height, width = gt_shape[:2]
            report_path = root / episode / "diagnostic.json"
            report = json.loads(report_path.read_text())
            evaluated_frames = len(report["metrics"]["autoregressive"]["frame_mae"])
            if len(report["metrics"]["teacher_forced"]["frame_mae"]) != evaluated_frames:
                raise RuntimeError(f"{report_path}: AR/TF frame_mae lengths differ")
            latent_frames = int(report["latent_frames"])
            expected_from_latent = 4 * (latent_frames - 1) + 1
            if evaluated_frames != expected_from_latent:
                raise RuntimeError(
                    f"{report_path}: evaluated frames {evaluated_frames} != VAE decoded length {expected_from_latent}"
                )
            # The source videos can contain up to three trailing frames that
            # are not representable after the temporal-compression crop. They
            # are excluded by diagnose_rollout.py via min(pred, gt); anything
            # larger indicates a real truncation or alignment error.
            trailing_gt = gt_count - evaluated_frames
            if trailing_gt < 0 or trailing_gt > 3:
                raise RuntimeError(
                    f"{report_path}: GT/evaluated frame mismatch is {trailing_gt}"
                )
            episode_result = {
                "episode": episode,
                "gt_frames": gt_count,
                "evaluated_frames": evaluated_frames,
                "trailing_gt_frames": trailing_gt,
                "gt_shape": [width, height],
            }
            for name, multiplier in (
                ("autoregressive.mp4", 1),
                ("teacher_forced.mp4", 1),
                ("autoregressive_compare.mp4", 2),
                ("teacher_forced_compare.mp4", 2),
            ):
                path = root / episode / name
                count, samples = count_and_sample(
                    path, width * multiplier, height, evaluated_frames
                )
                episode_result[name] = {
                    "frames": count,
                    "shape": [width * multiplier, height],
                    "sample_mean_range": [min(x[0] for x in samples), max(x[0] for x in samples)],
                    "sample_std_range": [min(x[1] for x in samples), max(x[1] for x in samples)],
                }
            result["episodes"].append(episode_result)
        except Exception as exc:
            result["status"] = "FAIL"
            result["errors"].append(f"{episode}: {exc}")

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
