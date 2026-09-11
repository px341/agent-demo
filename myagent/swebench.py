"""Generate SWE-bench predictions with myagent.

This module intentionally separates inference from the official SWE-bench
Docker evaluator.  It consumes an exported dataset JSON/JSONL file, checks out
each instance at ``base_commit``, runs the agent in an isolated directory and
writes the standard predictions JSONL format accepted by SWE-bench.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .agent_config import AgentParams
from .agent_loop import AgentLoop
from .context import ContextComposer
from .contracts import AgentRequest, AgentResponse, LLMClient, StopReason
from .errors import ToolPermissionError, ValidationError
from .provider import OpenAICompatibleModelClient
from .tools import ToolExecutor


class BenchmarkToolExecutor:
    """Keep the sandbox boundary but let the model recover from denied calls."""

    def __init__(self, cwd: Path, timeout: float | None):
        self.inner = ToolExecutor(cwd, timeout=timeout)

    def execute(self, name: str, args: dict[str, Any]) -> str:
        try:
            return self.inner.execute(name, args)
        except ToolPermissionError as exc:
            raise ValidationError(
                f"该操作被评测沙箱拒绝，请只使用当前 checkout 内的安全路径/命令：{exc}"
            ) from exc


TASK_TEMPLATE = """You are solving one SWE-bench issue in an isolated checkout.

Issue:
{problem_statement}

Inspect the repository, implement the smallest correct fix, and run relevant
tests when possible. Work directly in the checkout using the provided tools.
Do not merely describe a patch and do not commit changes. End with a concise
summary after the implementation is complete. Do not download dependencies,
reference wheels, or create scratch/reproduction files inside the repository;
use existing tests and source files. SWE-bench Lite solutions modify tracked
files, so finish by editing the actual implementation rather than adding notes.
"""


def load_instances(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL dataset export and validate required fields."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("dataset JSON must contain an array")
        records = value
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    required = {"instance_id", "repo", "base_commit", "problem_statement"}
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"instance {index} is not an object")
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"instance {index} is missing: {', '.join(missing)}")
    return records


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "git failed").strip()
        raise RuntimeError(detail[:2000])
    return proc.stdout


def _safe_instance_name(instance_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id)
    if not name or name in {".", ".."}:
        raise ValueError(f"invalid instance_id: {instance_id!r}")
    return name


def prepare_checkout(instance: dict[str, Any], work_root: Path) -> Path:
    """Clone and detach-checkout one instance in a fresh, bounded directory."""
    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    checkout = (work_root / _safe_instance_name(str(instance["instance_id"]))).resolve()
    if checkout.parent != work_root:
        raise ValueError("instance checkout escaped work root")
    if checkout.exists():
        raise FileExistsError(
            f"checkout already exists: {checkout}; remove it or use --resume"
        )
    repo = str(instance["repo"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError(f"invalid GitHub repo: {repo!r}")
    url = f"https://github.com/{repo}.git"
    try:
        _run_git(["clone", "--quiet", "--no-checkout", url, str(checkout)])
        _run_git(["checkout", "--quiet", "--detach", str(instance["base_commit"])], cwd=checkout)
    except Exception:
        # Only remove the exact fresh checkout created by this function.
        if checkout.exists() and checkout.parent == work_root:
            shutil.rmtree(checkout)
        raise
    return checkout


def capture_patch(checkout: Path) -> str:
    """Return the tracked binary-safe diff expected by SWE-bench Lite.

    Lite instances do not require creating files. Excluding untracked files also
    prevents temporary downloads and reproduction artifacts from contaminating
    a prediction.
    """
    return _run_git(["diff", "--binary", "--no-ext-diff", "HEAD"], cwd=checkout)


def run_instance(
    instance: dict[str, Any],
    checkout: Path,
    llm: LLMClient,
    *,
    max_turns: int = 30,
    tool_timeout: float | None = 600,
) -> tuple[dict[str, str], AgentResponse]:
    """Run one isolated instance and return an official prediction record."""
    params = AgentParams(
        cwd=str(checkout),
        max_turns=max_turns,
        tool_timeout=tool_timeout,
    )
    loop = AgentLoop(
        params,
        llm=llm,
        tools=BenchmarkToolExecutor(checkout, timeout=tool_timeout),
        # No approval gate: the checkout is created solely for this instance.
        approval_gate=None,
        composer=ContextComposer(
            max_input_tokens=params.max_input_tokens,
            max_output_tokens=params.max_output_tokens,
            max_tool_tokens=params.max_tool_tokens,
            max_total_tool_tokens=params.max_total_tool_tokens,
        ),
    )
    task = TASK_TEMPLATE.format(problem_statement=instance["problem_statement"])
    response = loop.run(AgentRequest(user_input=task))
    patch = capture_patch(checkout)
    model_name = getattr(llm, "model", llm.__class__.__name__)
    prediction = {
        "instance_id": str(instance["instance_id"]),
        "model_name_or_path": str(model_name),
        "model_patch": patch,
    }
    return prediction, response


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(str(json.loads(line)["instance_id"]))
    return ids


def _select_instances(
    records: Iterable[dict[str, Any]], ids: set[str], limit: int | None
) -> list[dict[str, Any]]:
    selected = [r for r in records if not ids or str(r["instance_id"]) in ids]
    return selected[:limit] if limit is not None else selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m myagent.swebench",
        description="Generate SWE-bench-compatible prediction patches",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="JSON/JSONL dataset export")
    parser.add_argument("--predictions", required=True, type=Path, help="output predictions JSONL")
    parser.add_argument("--work-root", required=True, type=Path, help="fresh isolated checkouts directory")
    parser.add_argument("--instance-id", action="append", default=[], help="only run this instance (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of instances")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--tool-timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true", help="skip IDs already present in predictions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    records = load_instances(args.dataset)
    wanted = set(args.instance_id)
    records = _select_instances(records, wanted, args.limit)
    if wanted:
        found = {str(r["instance_id"]) for r in records}
        missing = wanted - found
        if missing:
            raise SystemExit(f"instance IDs not found: {', '.join(sorted(missing))}")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(args.predictions) if args.resume else set()
    if args.predictions.exists() and not args.resume:
        raise SystemExit("predictions file already exists; use a new path or --resume")

    client = OpenAICompatibleModelClient(AgentParams())
    failures = 0
    with args.predictions.open("a", encoding="utf-8", newline="\n") as output:
        for index, instance in enumerate(records, 1):
            instance_id = str(instance["instance_id"])
            if instance_id in completed:
                print(f"[{index}/{len(records)}] skip {instance_id}")
                continue
            print(f"[{index}/{len(records)}] run {instance_id}", flush=True)
            try:
                checkout = prepare_checkout(instance, args.work_root)
                prediction, response = run_instance(
                    instance,
                    checkout,
                    client,
                    max_turns=args.max_turns,
                    tool_timeout=args.tool_timeout,
                )
                if response.stop_reason is not StopReason.FINAL_ANSWER:
                    failures += 1
                    print(
                        f"  warning: agent stopped with {response.stop_reason.value} "
                        f"(turns={response.turns_used}, tools={response.tool_calls}): "
                        f"{response.error or ''}",
                        file=sys.stderr,
                    )
                print(
                    f"  result: stop={response.stop_reason.value} "
                    f"turns={response.turns_used} tools={response.tool_calls} "
                    f"patch_chars={len(prediction['model_patch'])}"
                )
            except Exception as exc:
                failures += 1
                print(f"  failed: {exc}", file=sys.stderr)
                prediction = {
                    "instance_id": instance_id,
                    "model_name_or_path": str(client.model),
                    "model_patch": "",
                }
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
    print(f"wrote {args.predictions}; generation_failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
