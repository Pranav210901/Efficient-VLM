from __future__ import annotations

from .common import context, parser
from src.phase2.dense_training import evaluate_dense_teacher


def main() -> None:
    args = parser("Evaluate dense teacher", "configs/phase2/dense_teacher.yaml").parse_args()
    root, config = context(args); print(evaluate_dense_teacher(root, config))


if __name__ == "__main__": main()
