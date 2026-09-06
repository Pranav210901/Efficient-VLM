from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/probe1_error_decomposition"
OUT = SOURCE / "annotation_scaffolding"
AUDIT = SOURCE / "audit_sample_blinded.csv"
FAILURE = SOURCE / "failure_sample.csv"
KEY = SOURCE / "audit_key.csv"
REPORT_JSON = SOURCE / "report.json"
DISPLAY_SEED = 20260801


SCHEMA = {
    "audit_sample_blinded.csv": {
        "audit_id": ("string", "Opaque audit-row identifier used to join to the separately protected blinding key."),
        "query_caption": ("string", "Caption query shown to the human annotator."),
        "candidate_A": ("string/path", "Blinded candidate image path labelled A."),
        "candidate_B": ("string/path", "Blinded candidate image path labelled B."),
        "candidate_C": ("string/path", "Blinded candidate image path labelled C."),
        "candidate_D": ("string/path", "Blinded candidate image path labelled D."),
        "candidate_E": ("string/path", "Blinded candidate image path labelled E."),
        "rubric_categories": ("JSON string", "Pre-existing serialized rubric suggestions carried by the export; not human annotations."),
        "candidate_F": ("nullable string/path", "Optional sixth blinded candidate, present when the positive item was not already among the five retrieved candidates."),
    },
    "failure_sample.csv": {
        "direction": ("categorical string", "Retrieval direction; the exported rows are t2i because i2t did not meet the 50-item floor."),
        "query_id": ("string", "Canonical caption identifier for the t2i query."),
        "query": ("string", "Human-readable query caption."),
        "positive_id": ("string", "Canonical identifier of the annotated positive image."),
        "positive_rank_seed42": ("integer", "Positive image rank under display seed 42."),
        "positive_item": ("JSON string", "Annotated positive image identifier and path."),
        "ranks_all_seeds": ("JSON string", "Positive rank for each of C4 seeds 42, 43 and 44."),
        "top10_seed42": ("JSON string", "Ordered top-10 retrieved image identifiers for display seed 42."),
        "retrieved_items_seed42": ("JSON string", "Ordered top-10 retrieved image identifiers and paths for display seed 42."),
        "predeclared_categories": ("JSON string", "Pre-existing serialized failure-category suggestions; no category has been assigned."),
    },
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_block(header: list[str], rows: list[dict[str, str]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().rstrip()


def counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(row[column] or "<empty>" for row in rows).items()))


def write_template(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    output_fields = [*fields, "rubric_label_placeholder", "human_notes"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{field: row.get(field, "") for field in fields}, "rubric_label_placeholder": "", "human_notes": ""})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit_header, audit_rows = read_rows(AUDIT)
    failure_header, failure_rows = read_rows(FAILURE)
    key_header, key_rows = read_rows(KEY)
    source_report = json.loads(REPORT_JSON.read_text())

    # The key is used only for aggregate counts. It is never joined to, or
    # written beside, the blinded display rows.
    audit_aggregate = {
        "partition": counts(key_rows, "partition"),
        "tfidf_cross_bin": counts(key_rows, "tfidf_cross_bin"),
        "bge_cross_bin": counts(key_rows, "bge_cross_bin"),
        "partition_x_tfidf": dict(sorted(Counter(f"{row['partition']}|{row['tfidf_cross_bin']}" for row in key_rows).items())),
    }
    failure_rank_outcomes = Counter()
    for row in failure_rows:
        ranks = json.loads(row["ranks_all_seeds"])
        for seed, rank in ranks.items():
            failure_rank_outcomes[f"seed_{seed}|outside_top10={int(rank) > 10}"] += 1

    rng = random.Random(DISPLAY_SEED)
    audit_random_indices = sorted(rng.sample(range(len(audit_rows)), 5))
    failure_random_indices = sorted(rng.sample(range(len(failure_rows)), 5))
    audit_random = [audit_rows[index] for index in audit_random_indices]
    failure_random = [failure_rows[index] for index in failure_random_indices]

    audit_template = OUT / "audit_sample_annotation_template.csv"
    failure_template = OUT / "failure_sample_annotation_template.csv"
    audit_display_fields = [
        "audit_id", "query_caption", "candidate_A", "candidate_B",
        "candidate_C", "candidate_D", "candidate_E", "candidate_F",
    ]
    failure_display_fields = [
        "direction", "query_id", "query", "positive_id", "positive_rank_seed42",
        "positive_item", "ranks_all_seeds", "top10_seed42", "retrieved_items_seed42",
    ]
    write_template(audit_template, audit_display_fields, audit_rows)
    write_template(failure_template, failure_display_fields, failure_rows)

    lines = [
        "# Probe 1 human-annotation preparation",
        "",
        "Preparation only: no labels were assigned, no rows were resampled, no source export was changed, and the blinded audit was not unblinded.",
        "",
        "## Files and integrity",
        "",
        "| Role | Absolute path | Rows | SHA-256 |",
        "|---|---|---:|---|",
        f"| Blinded audit source | `{AUDIT.resolve()}` | {len(audit_rows)} | `{sha(AUDIT)}` |",
        f"| Failure source | `{FAILURE.resolve()}` | {len(failure_rows)} | `{sha(FAILURE)}` |",
        f"| Protected blinding key | `{KEY.resolve()}` | {len(key_rows)} | `{sha(KEY)}` |",
        f"| Audit annotation template | `{audit_template.resolve()}` | {len(audit_rows)} | `{sha(audit_template)}` |",
        f"| Failure annotation template | `{failure_template.resolve()}` | {len(failure_rows)} | `{sha(failure_template)}` |",
        "",
        "## 1. Schemas",
        "",
    ]
    for filename, header in ((AUDIT.name, audit_header), (FAILURE.name, failure_header)):
        lines.extend([f"### `{filename}`", "", "| Column | dtype | Description |", "|---|---|---|"])
        for column in header:
            dtype, description = SCHEMA[filename][column]
            lines.append(f"| `{column}` | {dtype} | {description} |")
        lines.append("")

    lines.extend([
        "### Blinding contract",
        "",
        "The blinded CSV exposes only `audit_id`, the query caption, randomly lettered candidate image paths A–F, and the serialized pre-existing rubric suggestions. It withholds `query_id`, `positive_id`, `positive_rank`, retrieval `partition`, TF-IDF bin, BGE bin, and the mapping from candidate letters to canonical image IDs.",
        "",
        f"The reproducibility key is stored separately at `{KEY.resolve()}` with columns `{', '.join(key_header)}`. This preparation read it only to produce aggregate bin counts; it never joined key rows to blinded rows and never copied key fields into an annotation template.",
        "",
        "## 2. Row counts and sampling",
        "",
        f"- Blinded ambiguity audit: **{len(audit_rows)} rows**. Source population: seed-42 C4 t2i queries. It initially sampled up to 11 rows from each of 9 `partition × TF-IDF-cross-bin` cells using seed `20260731`, redistributed deficits by largest remaining capacity, and assigned the final slot to the largest remaining stratum.",
        f"- Failure export: **{len(failure_rows)} rows**, all t2i. Direction was the only sampling stratum: i2t had 47 all-three-seed outside-top-10 queries and exported 0 because it missed the 50-row floor; t2i had 533 and sampled 50 uniformly without replacement using seed `20260732`. There was no further ambiguity-bin or failure-category stratification.",
        "",
        "Audit sample population before redistribution:",
        "",
        "```json", json.dumps(source_report["audit_export"]["population_counts_before_redistribution"], indent=2), "```",
        "",
        "## 3. Row examples",
        "",
        "All fields are printed. Random displays use fixed seed `20260801`; selected zero-based source-row indices are reported.",
        "",
        "### Blinded audit — first five",
        "", "```csv", csv_block(audit_header, audit_rows[:5]), "```", "",
        f"### Blinded audit — random five (indices {audit_random_indices})", "", "```csv", csv_block(audit_header, audit_random), "```", "",
        "### Failure sample — first five", "", "```csv", csv_block(failure_header, failure_rows[:5]), "```", "",
        f"### Failure sample — random five (indices {failure_random_indices})", "", "```csv", csv_block(failure_header, failure_random), "```", "",
        "## 4. Existing distributions",
        "",
        "TF-IDF is the primary ambiguity metric. BGE is a robustness check only. Their source-level cross-axis Spearman correlation is 0.5211 and weighted bin kappa is 0.3538; both are reported, but their agreement is not treated as validation.",
        "",
        "Aggregate distributions from the protected key (not joined to displayed rows):",
        "", "```json", json.dumps(audit_aggregate, indent=2), "```", "",
        "Failure sample direction and per-seed outcome distributions:",
        "", "```json", json.dumps({"direction": counts(failure_rows, "direction"), "per_seed_outside_top10": dict(sorted(failure_rank_outcomes.items()))}, indent=2), "```", "",
        "The serialized `rubric_categories` and `predeclared_categories` columns contain suggestion lists, not assigned categorical outcomes, so no label-frequency distribution is reported.",
        "",
        "## 5. Annotation templates",
        "",
        "Each template carries the existing item identifier and display fields required for human judgement, plus empty `rubric_label_placeholder` and `human_notes` columns. No rubric categories were invented or assigned.",
    ])
    report_path = OUT / "preparation_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    manifest = {
        "status": "COMPLETE",
        "display_seed": DISPLAY_SEED,
        "source_files_unchanged": True,
        "audit_unblinded": False,
        "files_read": [str(path.resolve()) for path in (AUDIT, FAILURE, KEY, REPORT_JSON)],
        "files_written": [str(path.resolve()) for path in (audit_template, failure_template, report_path)],
        "source_sha256": {path.name: sha(path) for path in (AUDIT, FAILURE, KEY, REPORT_JSON)},
        "row_counts": {"audit": len(audit_rows), "failure": len(failure_rows)},
    }
    (OUT / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
