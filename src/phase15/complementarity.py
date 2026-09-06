from __future__ import annotations

import numpy as np


def _binary(a: list[bool] | np.ndarray, b: list[bool] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if left.shape != right.shape: raise ValueError("prediction arrays must have equal shape")
    return left, right


def binary_complementarity(a, b) -> dict[str, float]:
    a, b = _binary(a, b); union = a | b; intersection = a & b
    return {"both_successful": int(intersection.sum()), "only_a_successful": int((a & ~b).sum()), "only_b_successful": int((~a & b).sum()), "both_unsuccessful": int((~a & ~b).sum()), "jaccard": float(intersection.sum() / max(1, union.sum())), "oracle_accuracy": float(union.mean())}


def retrieval_complementarity(a, b, k: int = 1) -> dict[str, float]:
    result = binary_complementarity(np.asarray(a) <= k, np.asarray(b) <= k); result["oracle_recall"] = result.pop("oracle_accuracy"); return result


def classification_complementarity(a, b) -> dict[str, float]:
    result = binary_complementarity(a, b); result["agreement_rate"] = float(np.asarray(a).astype(bool).eq(np.asarray(b).astype(bool)).mean()) if hasattr(np.asarray(a), "eq") else float((np.asarray(a) == np.asarray(b)).mean()); return result


def compositional_complementarity(a, b) -> dict[str, float]:
    return binary_complementarity(a, b)
