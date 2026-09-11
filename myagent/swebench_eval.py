"""Windows-safe launcher for the official SWE-bench evaluation harness.

The upstream harness materializes Linux ``eval.sh`` and patch files with
``Path.write_text``. On Windows that translates LF to CRLF, and the mounted
script then fails inside the Linux container. This launcher preserves LF for
those container-bound artifacts and otherwise delegates unchanged to the
official CLI.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path


_ORIGINAL_WRITE_TEXT = Path.write_text


def _container_artifact_write_text(
    self: Path,
    data: str,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> int:
    if self.name == "eval.sh" or self.suffix == ".diff":
        newline = "\n"
    return _ORIGINAL_WRITE_TEXT(
        self, data, encoding=encoding, errors=errors, newline=newline
    )


def main() -> int:
    if os.name != "nt":
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
        return 0

    Path.write_text = _container_artifact_write_text
    try:
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
    finally:
        Path.write_text = _ORIGINAL_WRITE_TEXT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
