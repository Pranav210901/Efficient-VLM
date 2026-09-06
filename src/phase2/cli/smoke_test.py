from __future__ import annotations

from .common import context, parser
import os
from src.phase2.smoke import run_ddp_smoke, run_synthetic_smoke


def main() -> None:
    args = parser("Phase 2 cross-attention smoke test", "configs/phase2/smoke.yaml").parse_args()
    root, _ = context(args)
    print(run_ddp_smoke(root) if int(os.environ.get("WORLD_SIZE", "1")) > 1 else run_synthetic_smoke(root))


if __name__ == "__main__": main()
