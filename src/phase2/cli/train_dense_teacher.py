from __future__ import annotations

from .common import context, parser
from src.phase2.dense_training import train_dense_teacher
from src.phase2.distributed import cleanup_distributed


def main() -> None:
    args = parser("Train dense task-conditioned teacher", "configs/phase2/dense_teacher.yaml").parse_args()
    root, config = context(args)
    summaries = []
    try:
        for strategy in config.get("strategies", [config.get("strategy", "task_and_input")]):
            active = dict(config); active["strategy"] = strategy
            summaries.append(
                train_dense_teacher(
                    root,
                    active,
                    resume=args.resume,
                    cleanup_process_group=False,
                )
            )
    finally:
        # One torchrun invocation trains every strategy. Keep its NCCL process
        # group alive between strategies and close it exactly once here.
        cleanup_distributed()
    print(summaries)


if __name__ == "__main__": main()
