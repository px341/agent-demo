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
import sys
from pathlib import Path


_ORIGINAL_WRITE_TEXT = Path.write_text


def _pull_with_credential_fallback(original, client, repository, *args, **kwargs):
    try:
        return original(client, repository, *args, **kwargs)
    except Exception as exc:
        if (
            not repository.startswith("swebench/")
            or kwargs.get("auth_config") is not None
            or "Credentials store error" not in str(exc)
        ):
            raise
        # Official SWE-bench images are public. A failed Windows credential
        # helper inside WSL should not prevent their anonymous download.
        print("Docker credential helper failed; retrying the public SWE-bench image anonymously.", file=sys.stderr)
        return original(client, repository, *args, **{**kwargs, "auth_config": {}})


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
    from docker.api.image import ImageApiMixin
    original_pull = ImageApiMixin.pull

    def public_pull(client, repository, *args, **kwargs):
        return _pull_with_credential_fallback(original_pull, client, repository, *args, **kwargs)

    ImageApiMixin.pull = public_pull
    if os.name == "nt":
        Path.write_text = _container_artifact_write_text
    try:
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
    finally:
        Path.write_text = _ORIGINAL_WRITE_TEXT
        ImageApiMixin.pull = original_pull
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
