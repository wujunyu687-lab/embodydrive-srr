"""Compare candidate closed-loop rollout metrics against a baseline."""

import argparse
import json
from pathlib import Path


METRICS = ("ar_overall", "ar_after16", "tf_overall", "tf_after16")


def read_episode(root: Path, episode: str):
    report = json.loads((root / episode / "diagnostic.json").read_text())
    autoregressive = report["metrics"]["autoregressive"]
    teacher_forced = report["metrics"]["teacher_forced"]
    return {
        "episode": episode,
        "ar_overall": autoregressive["overall_mae"],
        "ar_after16": autoregressive["after_16s_mae"],
        "tf_overall": teacher_forced["overall_mae"],
        "tf_after16": teacher_forced["after_16s_mae"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-regression", type=float, default=0.05)
    parser.add_argument("--min-mean-ar-gain", type=float, default=0.01)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    candidate_root = Path(args.candidate_root)
    episodes = args.episode_id or sorted(
        path.name for path in candidate_root.iterdir()
        if path.is_dir() and (path / "diagnostic.json").is_file()
    )
    if not episodes:
        raise FileNotFoundError(f"No candidate diagnostics under {candidate_root}")

    baseline = {episode: read_episode(baseline_root, episode) for episode in episodes}
    candidate = {episode: read_episode(candidate_root, episode) for episode in episodes}
    checks = []
    for episode in episodes:
        if candidate[episode]["ar_after16"] >= baseline[episode]["ar_after16"]:
            checks.append(f"{episode} AR after16 did not improve")
        for metric in METRICS:
            limit = baseline[episode][metric] * (1.0 + args.max_regression)
            if candidate[episode][metric] > limit:
                checks.append(f"{episode} {metric} regressed beyond limit")

    baseline_mean = sum(baseline[x]["ar_after16"] for x in episodes) / len(episodes)
    candidate_mean = sum(candidate[x]["ar_after16"] for x in episodes) / len(episodes)
    gain = (baseline_mean - candidate_mean) / baseline_mean
    if gain < args.min_mean_ar_gain:
        checks.append("mean AR after16 gain is below threshold")

    summary = {
        "status": "PASS" if not checks else "FAIL",
        "episodes": episodes,
        "baseline_mean_ar_after16": baseline_mean,
        "candidate_mean_ar_after16": candidate_mean,
        "mean_change_percent": gain * 100.0,
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
