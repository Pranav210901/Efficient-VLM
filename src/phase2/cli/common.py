from __future__ import annotations

import argparse
from pathlib import Path

from src.phase2.config import load_phase2_config, project_root


def parser(description: str, default_config: str) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--config", default=default_config)
    value.add_argument("--project-root", default=None)
    value.add_argument("--array-index", type=int, default=None)
    value.add_argument("--resume", action="store_true")
    return value


def context(args) -> tuple[Path, dict]:
    root = project_root(args.project_root)
    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = root / config_path
    return root, load_phase2_config(config_path)
