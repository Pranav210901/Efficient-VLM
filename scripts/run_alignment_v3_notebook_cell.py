#!/usr/bin/env python3
"""Execute tagged cells from the Alignment v3 result notebook."""

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


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/03_alignment_v3_results.ipynb"
COMMON_TAG = "alignment-v3-common"


def cell_tags(cell: dict[str, Any]) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def selected_cell_indices(notebook: dict[str, Any], stage: str) -> list[int]:
    selected = []
    requested = None if stage == "all" else f"alignment-v3-{stage}"
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        tags = cell_tags(cell)
        if COMMON_TAG in tags:
            selected.append(index)
        elif requested and requested in tags:
            selected.append(index)
        elif stage == "all" and any(tag.startswith("alignment-v3-") for tag in tags):
            selected.append(index)
    return selected


def execute(
    notebook: dict[str, Any],
    indices: list[int],
    *,
    stage: str,
    index: str | None,
    timeout: int,
) -> dict[str, Any]:
    result = deepcopy(notebook)
    environment = os.environ.copy()
    environment["ALIGNMENT_V3_NOTEBOOK_STAGE"] = stage
    if index is not None:
        environment["ALIGNMENT_V3_NOTEBOOK_INDEX"] = index
    environment["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{environment.get('PATH', '')}"
    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(ROOT), env=environment)
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=timeout)
        for execution_count, cell_index in enumerate(indices, start=1):
            cell = result["cells"][cell_index]
            cell["outputs"] = []
            cell["execution_count"] = execution_count
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(str(line) for line in source)
            message_id = client.execute(source, allow_stdin=False, stop_on_error=True)
            deadline = time.monotonic() + timeout
            failure = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"cell {cell_index} exceeded {timeout}s")
                message = client.get_iopub_msg(timeout=remaining)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                kind = message["header"]["msg_type"]
                content = message["content"]
                if kind == "status" and content.get("execution_state") == "idle":
                    break
                if kind == "stream":
                    cell["outputs"].append({"output_type": "stream", "name": content["name"], "text": content["text"]})
                elif kind in {"display_data", "execute_result"}:
                    output = {"output_type": kind, "data": content.get("data", {}), "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        output["execution_count"] = execution_count
                    cell["outputs"].append(output)
                elif kind == "error":
                    failure = {
                        "output_type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    }
                    cell["outputs"].append(failure)
            if failure:
                raise RuntimeError(f"cell {cell_index} failed: {failure['ename']}: {failure['evalue']}")
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    result.setdefault("metadata", {})["alignment_v3_execution"] = {
        "stage": stage,
        "array_index": index,
        "selected_cell_indices": indices,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "executed_at_unix": int(time.time()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--index")
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--list-cells", action="store_true")
    args = parser.parse_args()
    notebook_path = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    notebook = json.loads(notebook_path.read_text())
    indices = selected_cell_indices(notebook, args.stage)
    if not indices:
        raise RuntimeError(f"no notebook cells tagged for {args.stage!r}")
    if args.list_cells:
        print(json.dumps({"stage": args.stage, "cell_indices": indices}))
        return
    index = args.index or os.environ.get("SLURM_ARRAY_TASK_ID")
    output = args.output or ROOT / "results/alignment_v3/notebooks" / args.stage / f"{os.environ.get('SLURM_JOB_ID', 'manual')}_{index or 'main'}.ipynb"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = execute(notebook, indices, stage=args.stage, index=index, timeout=args.timeout)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=1))
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
