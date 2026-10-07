"""Generate SWE-bench predictions with myagent.

This module intentionally separates inference from the official SWE-bench
Docker evaluator.  It consumes an exported dataset JSON/JSONL file, checks out
each instance at ``base_commit``, runs the agent in an isolated directory and
writes the standard predictions JSONL format accepted by SWE-bench.
"""
from __future__ import annotations

import argparse
import hashlib
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
from .provider import OpenAICompatibleModelClient
from .benchmark_container import DockerSandbox, benchmark_schema, load_environments


TASK_TEMPLATE = """You are solving one SWE-bench issue in an isolated checkout.

Issue:
{problem_statement}

Inspect the repository, implement the smallest correct fix, and run relevant
tests. All tools run inside an offline Docker container at /testbed. Bash pipes
and redirections work. Python and task dependencies are already prepared.
Search within /testbed (use rg or find .); do not search the entire filesystem.
Work directly in the checkout using the provided tools.
Do not merely describe a patch and do not commit changes. End with a concise
summary after checking git diff and relevant test results. Add new implementation
or test files when needed; remove temporary reproduction artifacts before finishing.
Do not download dependencies or reference fixes. An empty patch is undelivered.
"""


GENERATION_FIELDS = {"instance_id", "repo", "base_commit", "problem_statement"}


def generation_record(record: dict) -> dict:
    return {key: record[key] for key in sorted(GENERATION_FIELDS)}


def load_instances(path: Path, *, generation_only: bool = False) -> list[dict[str, Any]]:
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
    return [generation_record(row) for row in records] if generation_only else records


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
    base_commit = str(instance["base_commit"])
    if not re.fullmatch(r"[0-9a-fA-F]{40}", base_commit):
        raise ValueError("base_commit must be a full commit SHA")
    try:
        # Fetch only the task snapshot. A full clone exposes later repair commits,
        # which an agent can recover with git log --all / git show.
        checkout.mkdir()
        _run_git(["init", "--quiet"], cwd=checkout)
        _run_git(["config", "core.autocrlf", "false"], cwd=checkout)
        _run_git(["fetch", "--quiet", "--no-tags", "--depth=1", url, base_commit], cwd=checkout)
        _run_git(["checkout", "--quiet", "--detach", base_commit], cwd=checkout)
        (checkout / ".git/FETCH_HEAD").unlink(missing_ok=True)
    except Exception:
        # Only remove the exact fresh checkout created by this function.
        if checkout.exists() and checkout.parent == work_root:
            shutil.rmtree(checkout)
        raise
    return checkout


def capture_patch(checkout: Path) -> str:
    """Collect tracked changes and non-ignored new files via a temporary index."""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        for args in (["read-tree", "HEAD"], ["add", "-A", "--", "."]):
            subprocess.run(["git", *args], cwd=checkout, env=env, check=True, capture_output=True, timeout=60)
        return subprocess.run(["git", "diff", "--cached", "--binary", "--no-ext-diff", "HEAD"],
                              cwd=checkout, env=env, check=True, capture_output=True, text=True, timeout=60).stdout


def run_instance(
    instance: dict[str, Any],
    checkout: Path,
    llm: LLMClient,
    *,
    max_turns: int = 80,
    max_output_tokens: int = 16384,
    tool_timeout: float = 60,
    environment: dict | None = None,
    sandbox=None,
    instance_seconds: float = 1800,
) -> tuple[dict[str, str], AgentResponse]:
    """Run one isolated instance and return an official prediction record."""
    if sandbox is None:
        if environment is None:
            raise ValueError("An audited Docker environment is required; host tools are disabled")
        sandbox = DockerSandbox(checkout, environment, timeout=tool_timeout, instance_seconds=instance_seconds)
    params = AgentParams(
        cwd="/testbed",
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
        tool_timeout=tool_timeout,
    )
    loop = AgentLoop(
        params,
        llm=llm,
        tools=sandbox,
        tools_schema=benchmark_schema,
        environment_prompt=f"Offline task container. Working directory: /testbed. Environment check: {sandbox.preflight}",
        # No approval gate: the checkout is created solely for this instance.
        approval_gate=None,
        composer=ContextComposer(
            max_input_tokens=params.max_input_tokens,
            max_output_tokens=params.max_output_tokens,
            max_tool_tokens=params.max_tool_tokens,
            max_total_tool_tokens=params.max_total_tool_tokens,
        ),
    )
    loop.system_prompt = "You are a coding agent. Use the supplied container tools to implement and verify the issue. All tool actions are authorized inside the disposable task container."
    task = TASK_TEMPLATE.format(problem_statement=instance["problem_statement"])
    try:
        from .benchmark_checks import isolation_check
        isolation_check(sandbox)
        response = loop.run(AgentRequest(user_input=task))
        from .errors import ToolError
        try:
            response.verification["git_status"] = sandbox.execute("git_status", {})
            response.verification["git_diff"] = sandbox.execute("git_diff", {})
            test_observations = []
            for step in response.steps:
                for index, call in enumerate(step.tool_calls):
                    function = call["function"]
                    arguments = json.loads(function["arguments"] or "{}")
                    if function["name"] == "run_shell" and re.search(r"\bpytest\b|\bunittest\b|runtests\.py", arguments.get("command", "")):
                        test_observations.append({"command": arguments["command"],
                                                  "result": step.observations[index]})
            response.verification["agent_test_runs"] = test_observations
            smoke_command = getattr(sandbox, "spec", {}).get("smoke_command")
            if not test_observations and smoke_command:
                response.verification["fallback_test_command"] = smoke_command
                response.verification["fallback_test_result"] = sandbox.execute("run_shell", {"command": smoke_command})
        except ToolError as exc:
            response.verification["error"] = str(exc)
        patch = sandbox.capture_patch()
    finally:
        sandbox.close()
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
    parser.add_argument("--environments", required=True, type=Path)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--tool-timeout", type=float, default=60)
    parser.add_argument("--instance-seconds", type=float, default=1800)
    parser.add_argument("--resume", action="store_true", help="skip IDs already present in predictions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    records = load_instances(args.dataset, generation_only=True)
    environments = load_environments(args.environments)
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

    client = OpenAICompatibleModelClient(AgentParams(max_output_tokens=args.max_output_tokens))
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
                    max_output_tokens=args.max_output_tokens,
                    instance_seconds=args.instance_seconds,
                    environment=environments[instance_id],
                )
                if response.stop_reason is not StopReason.FINAL_ANSWER or not prediction["model_patch"]:
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
            receipt = {"instance_id": instance_id,
                       "patch_sha256": hashlib.sha256(prediction["model_patch"].encode()).hexdigest(),
                       "delivery_status": "delivered" if prediction["model_patch"] else "undelivered_empty_patch"}
            with args.predictions.with_suffix(".receipts.jsonl").open("a", encoding="utf-8") as receipts:
                receipts.write(json.dumps(receipt) + "\n")
    print(f"wrote {args.predictions}; generation_failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
