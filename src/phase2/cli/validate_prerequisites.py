from __future__ import annotations

from .common import context, parser
from src.phase2.ablations import build_phase2_manifests
from src.phase2.prerequisites import validate_phase2_prerequisites


def main() -> None:
    args = parser("Validate locked Phase 1.5 inputs", "configs/phase2/prerequisites.yaml").parse_args()
    root, _ = context(args)
    report = validate_phase2_prerequisites(root, write=True)
    manifests = build_phase2_manifests(root)
    print({**report, "manifests": {key: str(value) for key, value in manifests.items()}})


if __name__ == "__main__": main()
