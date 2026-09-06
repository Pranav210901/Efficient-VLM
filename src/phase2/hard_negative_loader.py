from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


RETRIEVAL_REQUIRED = {
    "allowed_for_training", "source_split", "configuration_id", "direction", "query_id",
    "positive_id", "positive_text", "negative_id", "negative_text", "candidate_rank",
}
CLASSIFICATION_REQUIRED = {
    "allowed_for_training", "source_split", "configuration_id", "source_task", "sample_id",
    "true_class", "negative_class", "hard_negative_rank", "true_prompts", "negative_prompts",
}


def validate_training_frame(frame: pd.DataFrame, kind: str, protected_ids: Iterable[str] = ()) -> dict[str, int | str]:
    required = RETRIEVAL_REQUIRED if kind == "retrieval" else CLASSIFICATION_REQUIRED
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{kind} hard-negative schema is missing {missing}")
    if not frame["allowed_for_training"].fillna(False).astype(bool).all():
        raise ValueError(f"{kind} contains rows not authorised for training")
    forbidden = {str(value) for value in protected_ids}
    id_columns = [column for column in ("query_id", "positive_id", "negative_id", "sample_id") if column in frame]
    overlap = sum(frame[column].astype(str).isin(forbidden).sum() for column in id_columns)
    if overlap:
        raise ValueError(f"{kind} contains {overlap} protected-test ID references")
    if kind == "retrieval":
        same_image_i2t = frame["direction"].eq("i2t") & frame["negative_owner_id"].astype(str).eq(frame["positive_id"].astype(str))
        if same_image_i2t.any():
            raise ValueError("Retrieval negatives contain same-image captions")
    return {"kind": kind, "rows": int(len(frame)), "protected_overlap": int(overlap), "status": "valid"}


def read_hard_negatives(path: str | Path, *, kind: str, config_id: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    if max_rows is not None:
        import pyarrow.dataset as ds
        source = ds.dataset(str(path), format="parquet")
        expression = ds.field("configuration_id") == config_id if config_id else None
        if kind == "retrieval" and "direction" in source.schema.names:
            per_direction = max(1, int(max_rows) // 2)
            parts = [source.scanner(filter=(expression & (ds.field("direction") == direction)) if expression is not None else ds.field("direction") == direction).head(per_direction).to_pandas() for direction in ("i2t", "t2i")]
            frame = pd.concat(parts, ignore_index=True)
        else:
            frame = source.scanner(filter=expression).head(int(max_rows)).to_pandas()
    else:
        filters = [("configuration_id", "==", config_id)] if config_id else None
        frame = pd.read_parquet(path, filters=filters)
    validate_training_frame(frame, kind)
    return frame


def dataset_fingerprint(root: str | Path) -> str:
    schema = Path(root) / "schema.json"
    if not schema.exists():
        raise FileNotFoundError(schema)
    payload = json.loads(schema.read_text())
    value = payload.get("dataset_fingerprint") or payload.get("fingerprint")
    if value:
        return str(value)
    import hashlib
    return hashlib.sha256(schema.read_bytes()).hexdigest()[:20]
