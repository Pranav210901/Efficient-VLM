#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "results/alignment_v4"
QUESTION = (
    "Can modern frozen SSL vision encoders, combined with compatibility-based "
    "pair selection and sub-5M-parameter adaptation, approach compact jointly "
    "pretrained vision-language models under a fixed inference budget?"
)
SOURCES = {
    "MobileCLIP2-S0": DESTINATION / "mobileclip2/probe_report.json",
    "SigLIP2-B/32": DESTINATION / "siglip2/probe_report.json",
}


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Cannot combine incomplete v4 probes; missing: {missing}")
    rows = []
    for label, path in SOURCES.items():
        payload = json.loads(path.read_text())
        if payload.get("status") != "COMPLETE":
            raise SystemExit(f"Probe is not complete: {path}")
        rows.append({"teacher_label": label, **payload})
    frame = pd.DataFrame(rows)
    decisions = set(frame["decision"])
    if "VIABLE_ABSOLUTE_PATH" in decisions:
        overall = "PURSUE_ABSOLUTE_PATH"
    elif decisions == {"PIVOT_EFFICIENCY_NORMALIZED"}:
        overall = "PIVOT_EFFICIENCY_NORMALIZED"
    else:
        overall = "INCONCLUSIVE"
    result = {
        "status": "COMPLETE",
        "version": "alignment-v4-capture-probe-1.0",
        "main_dissertation_question": QUESTION,
        "overall_probe_decision": overall,
        "claim_status": (
            "DIAGNOSTIC_ONLY: this capture probe decides the next methodology; "
            "it does not by itself answer the dissertation question."
        ),
        "teachers": rows,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "combined_probe_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    frame.to_csv(DESTINATION / "combined_probe_report.csv", index=False)
    columns = [
        "teacher_label", "baseline_R@1", "distilled_R@1", "teacher_R@1",
        "delta_R@1", "capture_fraction", "decision",
    ]
    markdown = (
        "# Alignment v4 — distillation capture probe\n\n"
        f"**Main dissertation question:** {QUESTION}\n\n"
        f"**Overall probe decision:** `{overall}`\n\n"
        + frame[columns].to_markdown(index=False)
        + "\n\n"
        "This is a diagnostic decision, not a positive answer to the dissertation "
        "question. Teacher anchors are sealed COCO-val measurements while student "
        "probe results use the disjoint COCO-dev split, so capture fractions are "
        "directional and the absolute student thresholds are the primary gate.\n"
    )
    (DESTINATION / "combined_probe_report.md").write_text(markdown)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

