"""Retrospective paired-bootstrap uncertainty for frozen final outputs.

The intervals quantify dataset/query sampling variation.  Student predictions
are averaged over the three frozen seeds before resampling; seed-level effect
SD is reported separately and is not absorbed into the bootstrap interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.alignment_v3.final_zero_shot import ROOT
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text

INPUT_ROOT = ROOT / "artifacts/07_final_evaluation/zero_shot/per_run"
OUTPUT_ROOT = ROOT / "artifacts/07_final_evaluation/uncertainty"
STUDENT_IDS = ("mt1_strict_frozen", "mt1_dual_lora")
REFERENCE_IDS = (
    "openclip_vit_b32_quickgelu_openai",
    "mobileclip2_s0_dfndr2b",
    "siglip2_vit_b32_256_webli",
)
TASKS = ("cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot")
SEEDS = (42, 43, 44)


def _paired_bootstrap(
    differences: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
    chunk_size: int = 64,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired bootstrap requires a non-empty vector")
    draws = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    return {
        "n": len(values),
        "effect": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_iterations": int(iterations),
    }


def _paired_retrieval_bootstrap(
    i2t: np.ndarray,
    t2i: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
    chunk_size: int = 64,
) -> dict[str, float | int]:
    i2t = np.asarray(i2t, dtype=np.float64)
    t2i = np.asarray(t2i, dtype=np.float64)
    draws = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        count = stop - start
        image_indices = rng.integers(0, len(i2t), size=(count, len(i2t)))
        text_indices = rng.integers(0, len(t2i), size=(count, len(t2i)))
        draws[start:stop] = 0.5 * (
            i2t[image_indices].mean(axis=1) + t2i[text_indices].mean(axis=1)
        )
    return {
        "n_image_queries": len(i2t),
        "n_text_queries": len(t2i),
        "effect": float(0.5 * (i2t.mean() + t2i.mean())),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_iterations": int(iterations),
    }


def _run_dir(model_id: str, seed: int | None = None) -> Path:
    return INPUT_ROOT / model_id / (f"seed_{seed}" if seed is not None else "reference")


def _classification_vector(model_id: str, task: str, seed: int | None = None) -> pd.Series:
    path = _run_dir(model_id, seed) / f"{task}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample IDs in {path}")
    return frame.set_index("sample_id")["correct"].astype(float).sort_index()


def _retrieval_vectors(model_id: str, seed: int | None = None) -> dict[str, pd.Series]:
    path = _run_dir(model_id, seed) / "flickr30k_test_embeddings.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    images = torch.nn.functional.normalize(value["image_embeds"].float(), dim=-1)
    texts = torch.nn.functional.normalize(value["text_embeds"].float(), dim=-1)
    image_paths = [str(path) for path in value["image_paths"]]
    text_paths = [str(path) for path in value["text_image_paths"]]
    image_lookup = {path: index for index, path in enumerate(image_paths)}
    if len(image_lookup) != len(image_paths):
        raise ValueError(f"duplicate Flickr image identities in {path}")
    with torch.inference_mode():
        scores = images @ texts.t()
        best_text = scores.argmax(dim=1).tolist()
        best_image = scores.argmax(dim=0).tolist()
    i2t = np.asarray(
        [text_paths[text_index] == image_path for image_path, text_index in zip(image_paths, best_text)],
        dtype=np.float64,
    )
    t2i = np.asarray(
        [image_paths[image_index] == text_path for text_path, image_index in zip(text_paths, best_image)],
        dtype=np.float64,
    )
    return {
        "i2t": pd.Series(i2t, index=[f"image:{index:04d}:{path}" for index, path in enumerate(image_paths)]),
        "t2i": pd.Series(t2i, index=[f"caption:{index:05d}:{path}" for index, path in enumerate(text_paths)]),
    }


def _mean_student_classification(model_id: str, task: str) -> tuple[pd.Series, list[float]]:
    vectors = [_classification_vector(model_id, task, seed) for seed in SEEDS]
    frame = pd.concat(vectors, axis=1, join="inner")
    if len(frame) != len(vectors[0]) or frame.isna().any().any():
        raise ValueError(f"classification sample mismatch for {model_id}/{task}")
    return frame.mean(axis=1), [float(vector.mean()) for vector in vectors]


def _mean_student_retrieval(model_id: str) -> tuple[dict[str, pd.Series], list[float]]:
    values = [_retrieval_vectors(model_id, seed) for seed in SEEDS]
    result = {}
    for direction in ("i2t", "t2i"):
        frame = pd.concat([value[direction] for value in values], axis=1, join="inner")
        if len(frame) != len(values[0][direction]) or frame.isna().any().any():
            raise ValueError(f"retrieval query mismatch for {model_id}/{direction}")
        result[direction] = frame.mean(axis=1)
    per_seed = [
        0.5 * (float(value["i2t"].mean()) + float(value["t2i"].mean()))
        for value in values
    ]
    return result, per_seed


def _align(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner")
    if len(joined) != len(left) or len(joined) != len(right) or joined.isna().any().any():
        raise ValueError("paired outputs do not have identical sample identities")
    return joined["left"].to_numpy(), joined["right"].to_numpy()


def analyze(*, iterations: int = 10_000, seed: int = 20260828) -> dict[str, Any]:
    if iterations < 1000:
        raise ValueError("use at least 1,000 bootstrap iterations")
    rng = np.random.default_rng(seed)
    comparisons = [
        ("dual_vs_strict", "mt1_dual_lora", "mt1_strict_frozen"),
        ("strict_vs_openclip", "mt1_strict_frozen", "openclip_vit_b32_quickgelu_openai"),
        ("dual_vs_openclip", "mt1_dual_lora", "openclip_vit_b32_quickgelu_openai"),
        ("strict_vs_mobileclip", "mt1_strict_frozen", "mobileclip2_s0_dfndr2b"),
        ("dual_vs_mobileclip", "mt1_dual_lora", "mobileclip2_s0_dfndr2b"),
        ("strict_vs_siglip2", "mt1_strict_frozen", "siglip2_vit_b32_256_webli"),
        ("dual_vs_siglip2", "mt1_dual_lora", "siglip2_vit_b32_256_webli"),
    ]
    class_cache: dict[tuple[str, str], tuple[pd.Series, list[float] | None]] = {}
    retrieval_cache: dict[str, tuple[dict[str, pd.Series], list[float] | None]] = {}

    def classification(model: str, task: str):
        key = (model, task)
        if key not in class_cache:
            if model in STUDENT_IDS:
                class_cache[key] = _mean_student_classification(model, task)
            else:
                class_cache[key] = (_classification_vector(model, task), None)
        return class_cache[key]

    def retrieval(model: str):
        if model not in retrieval_cache:
            if model in STUDENT_IDS:
                retrieval_cache[model] = _mean_student_retrieval(model)
            else:
                retrieval_cache[model] = (_retrieval_vectors(model), None)
        return retrieval_cache[model]

    rows: list[dict[str, Any]] = []
    for comparison, left_id, right_id in comparisons:
        left, left_seeds = retrieval(left_id)
        right, right_seeds = retrieval(right_id)
        left_i2t, right_i2t = _align(left["i2t"], right["i2t"])
        left_t2i, right_t2i = _align(left["t2i"], right["t2i"])
        result = _paired_retrieval_bootstrap(
            left_i2t - right_i2t,
            left_t2i - right_t2i,
            iterations=iterations,
            rng=rng,
        )
        seed_effects = None
        if left_seeds is not None and right_seeds is not None:
            seed_effects = np.asarray(left_seeds) - np.asarray(right_seeds)
        rows.append(
            {
                "comparison": comparison,
                "left_model": left_id,
                "right_model": right_id,
                "dataset": "flickr30k_test",
                "metric": "mean_R@1",
                **{key: (100.0 * value if key in {"effect", "ci_low", "ci_high"} else value) for key, value in result.items()},
                "paired_seed_effect_sd_pp": (
                    float(np.std(seed_effects, ddof=1) * 100.0) if seed_effects is not None else None
                ),
            }
        )
        for task in TASKS:
            left, left_seeds = classification(left_id, task)
            right, right_seeds = classification(right_id, task)
            left_values, right_values = _align(left, right)
            result = _paired_bootstrap(
                left_values - right_values, iterations=iterations, rng=rng
            )
            seed_effects = None
            if left_seeds is not None and right_seeds is not None:
                seed_effects = np.asarray(left_seeds) - np.asarray(right_seeds)
            rows.append(
                {
                    "comparison": comparison,
                    "left_model": left_id,
                    "right_model": right_id,
                    "dataset": task,
                    "metric": "top1_accuracy",
                    **{key: (100.0 * value if key in {"effect", "ci_low", "ci_high"} else value) for key, value in result.items()},
                    "paired_seed_effect_sd_pp": (
                        float(np.std(seed_effects, ddof=1) * 100.0) if seed_effects is not None else None
                    ),
                }
            )

    queue_path = ROOT / "results/queue_identification/report/report.json"
    queue_result = None
    if queue_path.is_file():
        values = json.loads(queue_path.read_text())
        fractions = np.asarray(
            [row["fraction_of_damage_from_staleness"] for row in values["rows"]], dtype=np.float64
        )
        queue_result = _paired_bootstrap(fractions, iterations=iterations, rng=rng)
        queue_result = {
            **queue_result,
            "effect_percent": 100.0 * float(queue_result.pop("effect")),
            "ci_low_percent": 100.0 * float(queue_result.pop("ci_low")),
            "ci_high_percent": 100.0 * float(queue_result.pop("ci_high")),
            "unit": "six age-by-seed cells; descriptive cell bootstrap",
        }

    frame = pd.DataFrame(rows)
    atomic_csv(frame, OUTPUT_ROOT / "paired_bootstrap.csv")
    payload = {
        "status": "COMPLETE",
        "iterations": iterations,
        "seed": seed,
        "interval_scope": "dataset/query sampling conditional on frozen checkpoints",
        "comparisons": rows,
        "queue_staleness_share": queue_result,
    }
    atomic_json(payload, OUTPUT_ROOT / "report.json")
    lines = [
        "# Retrospective paired-bootstrap uncertainty",
        "",
        "Effects are left minus right in percentage points. Intervals quantify dataset/query sampling conditional on the frozen checkpoints; they are not training-population confidence intervals. Student predictions are averaged over seeds before resampling, with paired seed-effect SD shown separately where available.",
        "",
        frame.to_markdown(index=False),
        "",
        "## Queue staleness share",
        "",
        "```json",
        json.dumps(queue_result, indent=2),
        "```",
    ]
    atomic_text("\n".join(lines) + "\n", OUTPUT_ROOT / "report.md")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    value = analyze(iterations=args.iterations, seed=args.seed)
    print(json.dumps({"status": value["status"], "comparisons": len(value["comparisons"]), "output": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
