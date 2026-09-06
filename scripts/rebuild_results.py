from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision_encoders import VISION_MODEL_REGISTRY
from src.utils.config import load_config


VARIANTS = ("local_global", "baseline", "global", "local")
TEXT_ALIASES = {"minilm_l6": "all_minilm_l6_v2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a consistent best-epoch result table from per-run metrics.csv files.")
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--source_results", default="results/alignment_matrix_results.csv")
    parser.add_argument("--output", default="results/alignment_matrix_best_clean.csv")
    parser.add_argument("--selection_metric", default="i2t_R@1")
    parser.add_argument("--include_text_aliases", action="store_true", help="Keep both MiniLM labels instead of dropping minilm_l6.")
    return parser.parse_args()


def parse_run_name(name: str) -> tuple[str, str, str] | None:
    variant = next((candidate for candidate in VARIANTS if name.endswith(f"_{candidate}")), None)
    if variant is None:
        return None
    stem = name[: -(len(variant) + 1)]
    vision_encoder = next(
        (candidate for candidate in sorted(VISION_MODEL_REGISTRY, key=len, reverse=True) if stem.startswith(f"{candidate}_")),
        None,
    )
    if vision_encoder is None:
        return None
    return vision_encoder, stem[len(vision_encoder) + 1 :], variant


def variant_flags(variant: str) -> tuple[bool, bool]:
    return variant in {"local", "local_global"}, variant in {"global", "local_global"}


def source_metadata(path: Path) -> dict[tuple[str, str, bool, bool], dict[str, str]]:
    if not path.exists():
        return {}
    metadata: dict[tuple[str, str, bool, bool], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("vision_encoder")),
                str(row.get("text_encoder")),
                str(row.get("use_local_blf")).lower() == "true",
                str(row.get("use_global_blf")).lower() == "true",
            )
            metadata.setdefault(key, row)
    return metadata


def optional_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    return load_config(config_path) if config_path.exists() else {}


def rebuild(args: argparse.Namespace) -> pd.DataFrame:
    checkpoint_root = PROJECT_ROOT / args.checkpoint_root
    metadata = source_metadata(PROJECT_ROOT / args.source_results)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(checkpoint_root.glob("*/metrics.csv")):
        parsed = parse_run_name(metrics_path.parent.name)
        if parsed is None:
            continue
        vision_encoder, text_encoder, variant = parsed
        if not args.include_text_aliases and text_encoder in TEXT_ALIASES:
            continue
        metrics = pd.read_csv(metrics_path)
        if metrics.empty or args.selection_metric not in metrics:
            continue
        best = metrics.loc[metrics[args.selection_metric].astype(float).idxmax()].to_dict()
        use_local, use_global = variant_flags(variant)
        old = metadata.get((vision_encoder, text_encoder, use_local, use_global), {})
        config = optional_config(metrics_path.parent)
        training_cfg = config.get("training", {})
        row: dict[str, Any] = {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "variant": variant,
            "use_local_blf": use_local,
            "use_global_blf": use_global,
            "fusion_type": old.get("fusion_type", config.get("model", {}).get("fusion_type", "concat_mlp")),
            "selection_metric": args.selection_metric,
            "best_epoch": int(float(best["epoch"])),
            "seed": config.get("seed", "unknown"),
            "batch_size": training_cfg.get("batch_size", "unknown"),
            "run_dir": metrics_path.parent.relative_to(PROJECT_ROOT).as_posix(),
        }
        for key, value in best.items():
            if key != "epoch":
                row[key] = value
        for key in ("params_total", "params_trainable"):
            if key not in row or pd.isna(row[key]):
                value = old.get(key)
                if value not in {None, ""}:
                    row[key] = float(value)
        if "i2t_R@1" in row and "t2i_R@1" in row:
            row["mean_R@1"] = (float(row["i2t_R@1"]) + float(row["t2i_R@1"])) / 2
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No usable metrics files found under {checkpoint_root}")
    key_columns = ["vision_encoder", "text_encoder", "use_local_blf", "use_global_blf"]
    if frame.duplicated(key_columns).any():
        duplicates = frame.loc[frame.duplicated(key_columns, keep=False), key_columns]
        raise RuntimeError(f"Duplicate configurations remain after rebuilding:\n{duplicates.to_string(index=False)}")
    return frame.sort_values(["vision_encoder", "text_encoder", "variant"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    frame = rebuild(args)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    mismatches = 0
    source_path = PROJECT_ROOT / args.source_results
    if source_path.exists():
        source = pd.read_csv(source_path)
        source_keys = ["vision_encoder", "text_encoder", "use_local_blf", "use_global_blf"]
        source = source.drop_duplicates(source_keys, keep="first")
        merged = frame.merge(source[source_keys + [args.selection_metric]], on=source_keys, how="left", suffixes=("_best", "_source"))
        mismatches = int(
            (
                merged[f"{args.selection_metric}_source"].notna()
                & ((merged[f"{args.selection_metric}_best"] - merged[f"{args.selection_metric}_source"]).abs() > 1e-8)
            ).sum()
        )
    print(f"Wrote {len(frame)} clean best-epoch rows to {output}")
    print(f"Rows that differed from the old aggregate: {mismatches}")


if __name__ == "__main__":
    main()
