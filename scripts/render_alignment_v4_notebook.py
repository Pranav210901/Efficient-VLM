#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook", default="notebooks/04_alignment_v4_results.ipynb"
    )
    parser.add_argument(
        "--output",
        default="results/alignment_v4/notebooks/04_alignment_v4_results.ipynb",
    )
    args = parser.parse_args()
    source = Path(args.notebook).resolve()
    output = Path(args.output).resolve()
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(source.parents[1])}},
    )
    client.execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()

