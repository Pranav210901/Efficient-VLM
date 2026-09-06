from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from .statistics import matched_seed_differences, pareto_front, seed_statistics


def _csvs(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.glob(pattern))
    frames = [pd.read_csv(path) for path in paths if path.stat().st_size]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_phase2_report(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    output = root / "results/phase2"
    reranking = _csvs(output, "reranking/runs/*/results.csv")
    reranking_samples_paths = sorted(output.glob("reranking/runs/*/per_sample_results.parquet"))
    reranking_samples = pd.concat([pd.read_parquet(path) for path in reranking_samples_paths], ignore_index=True) if reranking_samples_paths else pd.DataFrame()
    reranking_coverage = _csvs(output, "reranking/runs/*/candidate_coverage.csv")
    reranking_efficiency = _csvs(output, "reranking/runs/*/efficiency.csv")
    ablations = _csvs(output, "ablations/runs/*/results.csv")
    dense = _csvs(output, "dense_teacher/runs/*/results.csv")
    seeds = _csvs(output, "seed_sweep/runs/*/results.csv")
    oracle_rows = []
    retrieval_oracle = root / "results/phase15/oracle/retrieval_oracle.csv"
    if retrieval_oracle.exists():
        frame = pd.read_csv(retrieval_oracle)
        for column, metric in (("oracle_R@1", "recall_at_1"), ("oracle_R@5", "recall_at_5"), ("oracle_R@10", "recall_at_10")):
            oracle_rows.append({"method": "oracle_upper_bound", "task": "coco_retrieval", "metric": metric, "value": float(frame[column].max())})
    classification_oracle = root / "results/phase15/oracle/classification_oracle.csv"
    if classification_oracle.exists():
        frame = pd.read_csv(classification_oracle)
        for task, task_frame in frame.groupby("dataset"):
            oracle_rows.append({"method": "oracle_upper_bound", "task": task, "metric": "top1", "value": float(task_frame["oracle_top1_accuracy"].max())})
    oracle = pd.DataFrame(oracle_rows)
    parts = []
    for method, frame in (("cross_attention_bridge", reranking), ("ablation", ablations), ("dense_teacher", dense), ("oracle", oracle)):
        if not frame.empty:
            item = frame.copy(); item["method_family"] = method; parts.append(item)
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["method_family", "metric", "value"])
    atomic_csv(long, output / "phase2_results_long.csv")
    reranking_root = output / "reranking"; reranking_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(reranking, reranking_root / "results.csv")
    if not reranking_samples.empty: reranking_samples.to_parquet(reranking_root / "per_sample_results.parquet", index=False)
    atomic_csv(reranking_coverage, reranking_root / "candidate_coverage.csv")
    atomic_csv(reranking_efficiency, reranking_root / "efficiency.csv")
    bridge_rows = []
    for path in sorted(output.glob("bridges/runs/*/summary.json")):
        payload = json.loads(path.read_text()); bridge_rows.append(payload)
    atomic_csv(pd.DataFrame(bridge_rows), output / "bridges/results_best.csv")
    atomic_csv(ablations, output / "ablations/results.csv")
    atomic_text("# Focused Phase 2 Ablations\n\n" + (ablations.to_markdown(index=False) if not ablations.empty else "No completed ablations yet.") + "\n", output / "ablations/summary.md")
    if not long.empty and {"method", "task", "metric", "value"}.issubset(long):
        wide = long.pivot_table(index="method", columns=["task", "metric"], values="value", aggfunc="mean").reset_index()
        wide.columns = ["__".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col for col in wide.columns]
    else:
        wide = pd.DataFrame()
    atomic_csv(wide, output / "phase2_results_wide.csv")
    atomic_csv(seeds, output / "phase2_seed_results.csv")
    statistics = seed_statistics(seeds, ["method", "task", "metric"]) if not seeds.empty and {"method", "task", "metric", "value"}.issubset(seeds) else pd.DataFrame()
    if not seeds.empty and {"best_bridge", "dense_teacher"}.issubset(set(seeds["method"])):
        matched_source = seeds[seeds["metric"].eq("mean_sample_utility")]
        matched = matched_seed_differences(matched_source, "dense_teacher", "best_bridge")
        matched_row = pd.DataFrame([{ "method": "dense_teacher_minus_best_bridge", "task": "multitask", "metric": "mean_sample_utility", **matched}])
        statistics = pd.concat([statistics, matched_row], ignore_index=True)
    atomic_csv(statistics, output / "phase2_statistics.csv")
    efficiency = _csvs(output, "**/efficiency.csv")
    if not efficiency.empty and {"performance", "latency_ms"}.issubset(efficiency):
        frontier = pareto_front(efficiency)
    else:
        frontier = pd.DataFrame(columns=["method", "performance", "latency_ms", "pareto_optimal"])
    atomic_csv(frontier, output / "phase2_pareto_frontier.csv")
    missing = [name for name, frame in (("reranking", reranking), ("ablations", ablations), ("dense_teacher", dense), ("seeds", seeds)) if frame.empty]
    status = "COMPLETE" if not missing else "INCOMPLETE"
    report = {"status": status, "missing_sections": missing, "rows": {"long": len(long), "seeds": len(seeds), "statistics": len(statistics), "pareto": len(frontier)}}
    markdown = "# Phase 2 Report\n\n" + f"Status: **{status}**\n\n" + ("Missing result sections: " + ", ".join(missing) + "\n\n" if missing else "All required result sections are present.\n\n")
    if not statistics.empty: markdown += "## Multi-seed statistics\n\n" + statistics.to_markdown(index=False) + "\n\n"
    if not frontier.empty: markdown += "## Pareto frontier\n\n" + frontier.to_markdown(index=False) + "\n"
    atomic_text(markdown, output / "phase2_report.md")
    atomic_json(report, output / "phase2_report.json")
    return report
