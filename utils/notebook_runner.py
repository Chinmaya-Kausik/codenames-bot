"""
Lightweight notebook runner that executes code cells without a Jupyter kernel.

Many CI sandboxes prohibit launching kernels/listening on sockets, which breaks
`jupyter nbconvert --execute`. This runner parses notebooks with nbformat and
executes each code cell directly in a shared Python namespace, skipping IPython
magics or shell commands that cannot run in restricted environments.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
from typing import Iterable

import nbformat


def _clean_cell_source(source: str) -> str:
    """
    Remove IPython magics and shell escapes from a code cell.

    Lines beginning with %, !, or ? are dropped entirely. Cells that start
    with a cell magic (%%) are skipped by returning an empty string.
    """
    stripped = source.lstrip()
    if stripped.startswith("%%"):
        return ""

    cleaned_lines = []
    for line in source.splitlines():
        lstripped = line.lstrip()
        if not lstripped:
            cleaned_lines.append(line)
            continue
        if lstripped.startswith("%"):
            continue
        if lstripped.startswith("!"):
            continue
        if lstripped.startswith("?"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


class _DummyIPython:
    """Minimal stub so notebooks that call get_ipython() do not crash."""

    def run_line_magic(self, *args, **kwargs):  # pragma: no cover - debug aid
        print(f"[notebook_runner] Skipping line magic: args={args}, kwargs={kwargs}")

    def run_cell_magic(self, *args, **kwargs):  # pragma: no cover - debug aid
        print(f"[notebook_runner] Skipping cell magic: args={args}, kwargs={kwargs}")


def run_notebook(path: pathlib.Path) -> None:
    """Execute all code cells from a notebook file."""
    path = path.resolve()
    nb = nbformat.read(path, as_version=4)
    env = {
        "__name__": "__main__",
        "__file__": str(path),
        "get_ipython": lambda: _DummyIPython(),
    }

    _ensure_plotting_libraries()

    original_cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for idx, cell in enumerate(nb.cells, start=1):
            if cell.cell_type != "code":
                continue
            source = _clean_cell_source(cell.source)
            if not source.strip():
                continue
            try:
                exec(compile(source, f"{path}::cell{idx}", "exec"), env, env)
            except Exception as exc:  # pragma: no cover - used as a CLI
                raise RuntimeError(f"Notebook {path} failed at cell {idx}") from exc
    finally:
        os.chdir(original_cwd)


def run_many(paths: Iterable[pathlib.Path]) -> None:
    for path in paths:
        print(f"[notebook_runner] Executing {path} ...")
        run_notebook(path)


def _ensure_plotting_libraries() -> None:
    """Ensure required plotting libraries are available."""
    for module in ("matplotlib.pyplot", "seaborn"):
        importlib.import_module(module)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Execute notebooks without Jupyter kernels.")
    parser.add_argument("paths", nargs="+", type=pathlib.Path, help="Notebook files to execute")
    args = parser.parse_args(argv)

    run_many(args.paths)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
