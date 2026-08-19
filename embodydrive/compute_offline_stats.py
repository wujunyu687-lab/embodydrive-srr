"""Compute Ctrl-World-style robust action/proprio normalization statistics."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reservoir-size", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def update_reservoir(reservoir, count, values, rng, limit):
    for row in values:
        count += 1
        if len(reservoir) < limit:
            reservoir.append(row.astype(np.float32, copy=True))
        else:
            index = rng.integers(0, count)
            if index < limit:
                reservoir[index] = row.astype(np.float32, copy=True)
    return count


def stats(values):
    array = np.asarray(values, dtype=np.float32)
    return {
        "p01": np.percentile(array, 1, axis=0).tolist(),
        "p99": np.percentile(array, 99, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "count": int(array.shape[0]),
    }


def main():
    args = parse_args()
    root = Path(args.data_root)
    annotations = sorted(root.glob("annotation/*/*.json"))
    if not annotations:
        raise FileNotFoundError(f"No annotations under {root}/annotation")
    rng = np.random.default_rng(args.seed)
    action_reservoir = []
    proprio_reservoir = []
    action_count = 0
    proprio_count = 0
    for index, path in enumerate(annotations, start=1):
        annotation = json.loads(path.read_text())
        action_count = update_reservoir(
            action_reservoir,
            action_count,
            np.asarray(annotation["action"], dtype=np.float32),
            rng,
            args.reservoir_size,
        )
        proprio_count = update_reservoir(
            proprio_reservoir,
            proprio_count,
            np.asarray(annotation["proprio"], dtype=np.float32),
            rng,
            args.reservoir_size,
        )
        if index % 10000 == 0:
            print(f"scanned={index}/{len(annotations)}", flush=True)
    output = {
        "source_annotations": len(annotations),
        "action": stats(action_reservoir),
        "proprio": stats(proprio_reservoir),
    }
    path = root / "stats.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote={path}", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
