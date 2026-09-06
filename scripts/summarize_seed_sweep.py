from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched seed-sweep results with paired variant-minus-baseline deltas.")
    parser.add_argument("--input", default="results/seed_sweep_results.csv")
    parser.add_argument("--metric", default="coco5_i2t_R@1")
    parser.add_argument("--summary_output", default="results/seed_sweep_summary.csv")
    parser.add_argument("--paired_output", default="results/seed_sweep_paired_deltas.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(PROJECT_ROOT / args.input)
    if args.metric not in frame:
        raise ValueError(f"Metric {args.metric!r} is not present in {args.input}")
    if frame.duplicated(["comparison", "variant", "seed"]).any():
        raise ValueError("Duplicate comparison/variant/seed rows found; clean the sweep result before summarizing")

    summary = (
        frame.groupby(["comparison", "variant"])[args.metric]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    summary["sem"] = summary["std"] / summary["n"].pow(0.5)
    summary["ci95_half_width"] = [
        T_CRITICAL_95.get(int(n) - 1, 1.96) * sem if int(n) > 1 else float("nan")
        for n, sem in zip(summary["n"], summary["sem"], strict=True)
    ]

    paired_rows: list[dict[str, object]] = []
    for comparison, group in frame.groupby("comparison"):
        baseline = group[group["variant"] == "baseline"][["seed", args.metric]].rename(columns={args.metric: "baseline"})
        for variant in sorted(set(group["variant"]) - {"baseline"}):
            candidate = group[group["variant"] == variant][["seed", args.metric]].rename(columns={args.metric: "candidate"})
            paired = baseline.merge(candidate, on="seed", validate="one_to_one")
            paired["delta"] = paired["candidate"] - paired["baseline"]
            n = len(paired)
            std = float(paired["delta"].std()) if n > 1 else float("nan")
            sem = std / n**0.5 if n > 1 else float("nan")
            paired_rows.append(
                {
                    "comparison": comparison,
                    "variant": variant,
                    "metric": args.metric,
                    "n": n,
                    "baseline_mean": float(paired["baseline"].mean()),
                    "candidate_mean": float(paired["candidate"].mean()),
                    "mean_delta": float(paired["delta"].mean()),
                    "delta_std": std,
                    "delta_ci95_half_width": T_CRITICAL_95.get(n - 1, 1.96) * sem if n > 1 else float("nan"),
                    "wins": int((paired["delta"] > 0).sum()),
                    "ties": int((paired["delta"] == 0).sum()),
                    "losses": int((paired["delta"] < 0).sum()),
                }
            )
    paired_frame = pd.DataFrame(paired_rows)
    summary_path = PROJECT_ROOT / args.summary_output
    paired_path = PROJECT_ROOT / args.paired_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    paired_frame.to_csv(paired_path, index=False)
    print(summary.to_string(index=False))
    print()
    print(paired_frame.to_string(index=False))
    print(f"\nWrote {summary_path} and {paired_path}")


if __name__ == "__main__":
    main()
