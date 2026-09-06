#!/usr/bin/env python3
"""Execute only the Alignment v2 notebook cells associated with a pipeline stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks/02_alignment_v2_results.ipynb"
STAGES = ("validate", "prefetch", "reference", "train", "evaluate", "report", "status", "all")
COMMON_TAG = "alignment-v2-common"


def cell_tags(cell: dict[str, Any]) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def selected_cell_indices(notebook: dict[str, Any], stage: str) -> list[int]:
    """Return common and stage cell indices in their original notebook order."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    requested = None if stage == "all" else f"alignment-v2-{stage}"
    selected: list[int] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        tags = cell_tags(cell)
        if COMMON_TAG in tags or (requested and requested in tags):
            selected.append(index)
        elif stage == "all" and any(tag.startswith("alignment-v2-") for tag in tags):
            selected.append(index)
    return selected


def default_output(stage: str, index: str | None) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", f"manual_{int(time.time())}")
    task = index if index is not None else os.environ.get("SLURM_ARRAY_TASK_ID", "main")
    return PROJECT_ROOT / "results/alignment_v2/notebooks" / stage / f"{job_id}_{task}.ipynb"


def execute_selected_cells(
    notebook: dict[str, Any],
    indices: list[int],
    *,
    stage: str,
    index: str | None,
    timeout: int,
    allow_errors: bool,
) -> dict[str, Any]:
    executed = deepcopy(notebook)
    execution_count = 0
    environment = os.environ.copy()
    environment["ALIGNMENT_V2_NOTEBOOK_STAGE"] = stage
    if index is not None:
        environment["ALIGNMENT_V2_NOTEBOOK_INDEX"] = index
    environment["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{environment.get('PATH', '')}"

    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(PROJECT_ROOT), env=environment)
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=timeout)
        for cell_index in indices:
            cell = executed["cells"][cell_index]
            execution_count += 1
            cell["execution_count"] = execution_count
            cell["outputs"] = []
            message_id = client.execute(
                cell.get("source", ""),
                allow_stdin=False,
                stop_on_error=not allow_errors,
            )
            deadline = time.monotonic() + timeout
            failed: dict[str, Any] | None = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"cell {cell_index} exceeded {timeout} seconds")
                message = client.get_iopub_msg(timeout=remaining)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                message_type = message["header"]["msg_type"]
                content = message["content"]
                if message_type == "status" and content.get("execution_state") == "idle":
                    break
                if message_type == "stream":
                    cell["outputs"].append(
                        {
                            "output_type": "stream",
                            "name": content.get("name", "stdout"),
                            "text": content.get("text", ""),
                        }
                    )
                elif message_type in {"display_data", "execute_result"}:
                    output = {
                        "output_type": message_type,
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                    }
                    if message_type == "execute_result":
                        output["execution_count"] = execution_count
                    cell["outputs"].append(output)
                elif message_type == "error":
                    failed = {
                        "output_type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    }
                    cell["outputs"].append(failed)
            if failed and not allow_errors:
                raise RuntimeError(
                    f"notebook cell {cell_index} failed: {failed['ename']}: {failed['evalue']}"
                )
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

    executed.setdefault("metadata", {})["alignment_v2_execution"] = {
        "stage": stage,
        "array_index": index,
        "selected_cell_indices": indices,
        "executed_at_unix": int(time.time()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    return executed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute tagged cells from the Alignment v2 results notebook"
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument(
        "--index",
        help="Slurm array index to visualise; defaults to SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="Print selected cell indices without starting a kernel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = args.notebook
    if not notebook_path.is_absolute():
        notebook_path = PROJECT_ROOT / notebook_path
    notebook = json.loads(notebook_path.read_text())
    indices = selected_cell_indices(notebook, args.stage)
    if not indices:
        raise RuntimeError(f"no cells are tagged for stage {args.stage!r}")
    if args.list_cells:
        print(json.dumps({"stage": args.stage, "cell_indices": indices}))
        return

    array_index = args.index or os.environ.get("SLURM_ARRAY_TASK_ID")
    output = args.output or default_output(args.stage, array_index)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        executed = execute_selected_cells(
            notebook,
            indices,
            stage=args.stage,
            index=array_index,
            timeout=args.timeout,
            allow_errors=args.allow_errors,
        )
    except Exception:
        # Preserve any pre-existing output and surface the error to Slurm.
        raise
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(executed, indent=1))
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()

