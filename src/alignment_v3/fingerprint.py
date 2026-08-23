from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def hash_payload(value: Any) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def hash_config(config: dict[str, Any]) -> str:
    return hash_payload(config)


def hash_dataset_split(
    image_ids: Iterable[str],
    *,
    annotation_path: str | Path | None = None,
    reject_duplicates: bool = True,
) -> str:
    values = [str(value) for value in image_ids]
    if reject_duplicates and len(values) != len(set(values)):
        raise ValueError("dataset split contains duplicate image IDs")
    payload: dict[str, Any] = {"image_ids": sorted(set(values))}
    if annotation_path is not None:
        annotation = Path(annotation_path)
        payload["annotation_path"] = str(annotation.resolve())
        payload["annotation_sha256"] = sha256_file(annotation)
    return hash_payload(payload)


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in ("torch", "timm", "transformers", "open_clip"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "unavailable"
    return versions


@dataclass(frozen=True)
class Fingerprint:
    config_hash: str
    dataset_split_hash: str
    model_checkpoint_id: str
    cache_version: str
    preprocessing_hash: str
    code_version: str
    seed: int | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return hash_payload(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "digest": self.digest}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Fingerprint":
        fields = {
            key: payload[key]
            for key in (
                "config_hash",
                "dataset_split_hash",
                "model_checkpoint_id",
                "cache_version",
                "preprocessing_hash",
                "code_version",
                "seed",
                "schema_version",
            )
        }
        value = cls(**fields)
        if payload.get("digest") != value.digest:
            raise ValueError("fingerprint digest does not match its fields")
        return value


def code_fingerprint(root: str | Path, paths: Iterable[str | Path]) -> str:
    base = Path(root).resolve()
    rows = []
    for value in sorted((Path(path) for path in paths), key=str):
        path = value if value.is_absolute() else base / value
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": str(path.relative_to(base)), "sha256": sha256_file(path)})
    return hash_payload(rows)


def write_fingerprint(path: str | Path, fingerprint: Fingerprint) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(fingerprint.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
    return target


def read_fingerprint(path: str | Path) -> Fingerprint | None:
    value = Path(path)
    if not value.is_file():
        return None
    return Fingerprint.from_dict(json.loads(value.read_text()))


def is_fresh(path: str | Path, expected: Fingerprint) -> bool:
    try:
        actual = read_fingerprint(path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return actual is not None and actual.digest == expected.digest

