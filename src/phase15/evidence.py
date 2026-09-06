"""Resolve the canonical Phase 1 evidence root for derived Phase 1.5 work."""
from __future__ import annotations

import json
from pathlib import Path


def resolve_evaluation_root(
    project_root: str | Path,
    evaluation_root: str | Path | None = None,
    *,
    prefer_development: bool = True,
) -> Path:
    root = Path(project_root).resolve()
    if evaluation_root is not None:
        selected = Path(evaluation_root)
        return selected.resolve() if selected.is_absolute() else (root / selected).resolve()
    development = root / "results/phase1_multitask_development"
    manifest = development / "development_evidence_manifest.json"
    if prefer_development and manifest.exists():
        try:
            if json.loads(manifest.read_text()).get("status") == "complete":
                return development
        except Exception:
            pass
    return root / "results/phase1_multitask"
