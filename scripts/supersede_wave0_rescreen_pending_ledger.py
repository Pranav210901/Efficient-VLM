#!/usr/bin/env python3
"""Append supersession records for historical Wave 0 re-screen PENDING rows."""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "logs/run_ledger.jsonl"
MANIFEST = (
    ROOT
    / "results/alignment_v4_wave0/manifests/wave0_rescreen_ledger_supersession.json"
)


def main() -> None:
    with LEDGER.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = [json.loads(line) for line in handle if line.strip()]
        pending = [
            row
            for row in rows
            if row.get("wave") == "wave0_rescreen"
            and row.get("status") == "PENDING"
        ]
        already = {
            str(row.get("supersedes_timestamp"))
            for row in rows
            if row.get("status") == "SUPERSEDED"
        }
        complete_keys = {
            (str(row.get("config_fingerprint")), str(row.get("question")))
            for row in rows
            if row.get("status") == "COMPLETE"
        }
        additions = []
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for row in pending:
            original_timestamp = str(row["timestamp"])
            if original_timestamp in already:
                continue
            complete_exists = (
                str(row.get("config_fingerprint")),
                str(row.get("question")),
            ) in complete_keys
            additions.append(
                {
                    **row,
                    "timestamp": now,
                    "status": "SUPERSEDED",
                    "supersedes_timestamp": original_timestamp,
                    "superseded_reason": (
                        "Historical PENDING bookkeeping row; the matching run "
                        "completed or the attempt was replaced by the accepted "
                        "teaching/native-BF16 re-screen."
                    ),
                    "superseded_by_matching_complete": complete_exists,
                }
            )
        handle.seek(0, 2)
        for row in additions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    result = {
        "status": "COMPLETE",
        "historical_pending_found": len(pending),
        "historical_pending_marked_superseded": len(pending),
        "effective_unresolved_historical_pending": 0,
        "supersession_records_appended_this_invocation": len(additions),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "append_only": True,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(MANIFEST)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
