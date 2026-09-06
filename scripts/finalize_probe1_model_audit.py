from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/probe1_error_decomposition"
OUT = BASE / "annotation_scaffolding"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_band(rank: int) -> str:
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51+"


def main() -> None:
    labels = json.loads((OUT / "model_audit_labels.json").read_text())
    audit = read_csv(BASE / "audit_sample_blinded.csv")
    failure = read_csv(BASE / "failure_sample.csv")
    assert len(audit) == len(labels["ambiguity"]) == 100
    assert len(failure) == len(labels["failure"]) == 50

    audit_out = []
    for row in audit:
        item = labels["ambiguity"][row["audit_id"]]
        audit_out.append({
            **row,
            "audit_label": item["label"],
            "audit_rationale": item["rationale"],
            "audit_confidence": item["confidence"],
        })

    failure_out = []
    for index, row in enumerate(failure):
        item = labels["failure"][f"failure_{index:03d}"]
        failure_out.append({
            **row,
            "rank_band": rank_band(int(row["positive_rank_seed42"])),
            "audit_label": item["label"],
            "audit_rationale": item["rationale"],
            "audit_confidence": item["confidence"],
        })

    audit_path = OUT / "audit_sample_model_assisted_audit.csv"
    failure_path = OUT / "failure_sample_model_assisted_audit.csv"
    write_csv(audit_path, audit_out)
    write_csv(failure_path, failure_out)

    summary = {
        "status": "COMPLETE",
        "description": "Model-assisted audit using a rubric frozen before visual inspection; not human ground truth.",
        "blinding": "The ambiguity audit was completed without reading or joining audit_key.csv.",
        "counts": {"ambiguity": len(audit_out), "failure": len(failure_out)},
        "ambiguity_labels": dict(Counter(row["audit_label"] for row in audit_out)),
        "failure_labels": dict(Counter(row["audit_label"] for row in failure_out)),
        "failure_rank_bands": dict(Counter(row["rank_band"] for row in failure_out)),
        "confidence": {
            "ambiguity": dict(Counter(row["audit_confidence"] for row in audit_out)),
            "failure": dict(Counter(row["audit_confidence"] for row in failure_out)),
        },
        "source_sha256": {
            "audit_sample_blinded.csv": sha256(BASE / "audit_sample_blinded.csv"),
            "failure_sample.csv": sha256(BASE / "failure_sample.csv"),
        },
        "outputs": {"ambiguity": str(audit_path.resolve()), "failure": str(failure_path.resolve())},
    }
    (OUT / "model_assisted_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Probe 1 model-assisted audit",
        "",
        "This is a rubric-constrained model-assisted audit, not a human audit and not ground truth.",
        "The ambiguity sample remained blinded: `audit_key.csv` was neither read nor joined during labelling.",
        "",
        "## Ambiguity labels",
        "",
    ]
    lines += [f"- `{key}`: {value}" for key, value in sorted(summary["ambiguity_labels"].items())]
    lines += ["", "## Failure labels", ""]
    lines += [f"- `{key}`: {value}" for key, value in sorted(summary["failure_labels"].items())]
    lines += ["", "## Interpretation boundary", "", "These labels are structured judgements under the frozen rubric. They support descriptive sensitivity analysis, but must not be presented as independently verified human ground truth.", ""]
    (OUT / "model_assisted_audit_summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
