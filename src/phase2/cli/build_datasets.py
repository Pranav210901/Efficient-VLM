from __future__ import annotations

from .common import context, parser
from src.phase2.data_build import build_phase2_data


def main() -> None:
    args = parser("Materialise leakage-safe Phase 2 data", "configs/phase2/data_build.yaml").parse_args()
    root, _ = context(args); print(build_phase2_data(root))


if __name__ == "__main__": main()
