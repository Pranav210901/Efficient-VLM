from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None
        if self.path.exists():
            with self.path.open(newline="") as handle:
                reader = csv.reader(handle)
                self.fieldnames = next(reader, None)

    def write(self, row: dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key) for key in self.fieldnames})


def count_parameters(model: object) -> dict[str, float]:
    params = list(model.parameters())  # type: ignore[attr-defined]
    total = sum(param.numel() for param in params)
    trainable = sum(param.numel() for param in params if param.requires_grad)
    return {
        "params_total": float(total),
        "params_trainable": float(trainable),
    }
