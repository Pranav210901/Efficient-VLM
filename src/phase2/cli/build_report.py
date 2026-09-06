from __future__ import annotations

from .common import context, parser
from src.phase2.reporting import build_phase2_report


def main() -> None:
    args = parser("Build final Phase 2 report", "configs/phase2/reporting.yaml").parse_args()
    root, _ = context(args); print(build_phase2_report(root))


if __name__ == "__main__": main()
