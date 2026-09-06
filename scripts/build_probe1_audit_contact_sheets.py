from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/probe1_error_decomposition"
OUT = SOURCE / "annotation_scaffolding"
SHEETS = OUT / "contact_sheets"
TILE = (256, 220)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fitted(path: Path, label: str) -> Image.Image:
    canvas = Image.new("RGB", TILE, "white")
    with Image.open(ROOT / path) as source:
        image = source.convert("RGB")
        image.thumbnail((TILE[0] - 8, TILE[1] - 30))
        x = (TILE[0] - image.width) // 2
        y = 26 + (TILE[1] - 30 - image.height) // 2
        canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, TILE[0], 25), fill="black")
    draw.text((6, 6), label, fill="white", font=ImageFont.load_default())
    return canvas


def sheet(title: str, items: list[tuple[str, str]], columns: int, destination: Path) -> None:
    title_lines = textwrap.wrap(title, width=115) or [""]
    header = 18 * len(title_lines) + 16
    row_count = (len(items) + columns - 1) // columns
    output = Image.new("RGB", (columns * TILE[0], header + row_count * TILE[1]), "white")
    draw = ImageDraw.Draw(output)
    for index, line in enumerate(title_lines):
        draw.text((8, 8 + index * 18), line, fill="black", font=ImageFont.load_default())
    for index, (label, path) in enumerate(items):
        output.paste(fitted(Path(path), label), ((index % columns) * TILE[0], header + (index // columns) * TILE[1]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination, quality=92)


def main() -> None:
    audit = rows(SOURCE / "audit_sample_blinded.csv")
    failure = rows(SOURCE / "failure_sample.csv")
    audit_dir, failure_dir = SHEETS / "ambiguity_blinded", SHEETS / "failure"
    for row in audit:
        items = []
        for label in "ABCDEF":
            path = row.get(f"candidate_{label}", "")
            if path:
                items.append((f"Candidate {label}", path))
        sheet(f"{row['audit_id']} | Caption: {row['query_caption']}", items, 3, audit_dir / f"{row['audit_id']}.jpg")
    for index, row in enumerate(failure):
        positive = json.loads(row["positive_item"])
        retrieved = json.loads(row["retrieved_items_seed42"])
        items = [(f"POSITIVE | rank {row['positive_rank_seed42']}", positive["image_path"])]
        items.extend((f"Retrieved #{rank}", item["image_path"]) for rank, item in enumerate(retrieved, 1))
        sheet(f"failure_{index:03d} | {row['query_id']} | Caption: {row['query']}", items, 4, failure_dir / f"failure_{index:03d}.jpg")
    protocol = OUT / "audit_protocol.json"
    manifest = {
        "status": "COMPLETE",
        "audit_sheets": len(list(audit_dir.glob("*.jpg"))),
        "failure_sheets": len(list(failure_dir.glob("*.jpg"))),
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "audit_key_used": False,
        "audit_dir": str(audit_dir.resolve()),
        "failure_dir": str(failure_dir.resolve())
    }
    (OUT / "contact_sheet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
