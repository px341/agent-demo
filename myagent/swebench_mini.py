"""Prepare and run small, budgeted SWE-bench Verified Mini batches.

The default command only checks local prerequisites. Dataset preparation never
constructs a model client or starts an evaluation container.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from .agent_config import AgentParams
from .contracts import StopReason
from .provider import OpenAICompatibleModelClient
from .swebench import GENERATION_FIELDS, generation_record, load_instances, prepare_checkout, run_instance
from .benchmark_container import DockerSandbox, load_environments, prepare_runtime
from .benchmark_checks import isolation_check

MINI_NAME = "MariusHobbhahn/swe-bench-verified-mini"
MINI_REVISION = "b316c349947c29963fce3f4a65967c9807a4b673"
VERIFIED_NAME = "SWE-bench/SWE-bench_Verified"
DEFAULT_DATASET = Path(".swebench-work/data/swebench-verified-mini-test.jsonl")
DEFAULT_EVALUATION_DATASET = Path(".swebench-work/scoring/swebench-verified-mini-test.jsonl")
DEFAULT_ENVIRONMENTS = Path(".swebench-work/data/generation-environments.json")
EVAL_FIELDS = {"image", "eval_script", "log_parser", "eval_type"}
# Conservative Flash peak rates, CNY per million tokens, checked 2026-10-07.
# https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
INPUT_RATE = Decimal("2")
OUTPUT_RATE = Decimal("8")
MILLION = Decimal("1000000")


def adapt_mini(mini: list[dict], verified: list[dict]) -> list[dict]:
    """Preserve Mini order/content; add official harness fields by exact ID."""
    ids = [row["instance_id"] for row in mini]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate Mini instance IDs")
    official = {row["instance_id"]: row for row in verified}
    result = []
    for row in mini:
        instance_id = row["instance_id"]
        if instance_id not in official:
            raise ValueError(f"Mini instance not in official Verified: {instance_id}")
        source = official[instance_id]
        for key in ("repo", "base_commit", "problem_statement"):
            if row[key] != source[key]:
                raise ValueError(f"Mini/Verified mismatch: {instance_id}, {key}")
        missing = EVAL_FIELDS - source.keys()
        if missing:
            raise ValueError(f"Official evaluation fields missing: {sorted(missing)}")
        result.append({**row, **{key: source[key] for key in EVAL_FIELDS}})
    return result


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(args) -> None:
    if bool(args.mini_source) != bool(args.verified_source):
        raise ValueError("Pass both --mini-source and --verified-source, or neither")
    if args.mini_source:
        mini = load_instances(args.mini_source)
        verified = load_instances(args.verified_source)
    else:
        from datasets import load_dataset
        mini = list(load_dataset(MINI_NAME, revision=MINI_REVISION, split="test"))
        verified = list(load_dataset(VERIFIED_NAME, split="test"))
    if len(mini) != 50:
        raise ValueError(f"Expected the fixed 50-task Mini dataset, got {len(mini)}")
    records = adapt_mini(mini, verified)
    from swebench.harness.utils import make_test_spec
    for row in records:
        make_test_spec(row)
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    # The generation export is a strict allowlist, not a blacklist of known answers.
    args.evaluation_dataset.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                                       encoding="utf-8", newline="\n")
    write_json(args.evaluation_dataset.with_suffix(".manifest.json"), {
        "schema": 1, "sha256": hashlib.sha256(args.evaluation_dataset.read_bytes()).hexdigest(),
        "source": VERIFIED_NAME, "instance_ids": [row["instance_id"] for row in records]})
    payload = "".join(json.dumps(generation_record(row), ensure_ascii=False) + "\n" for row in records)
    args.dataset.write_text(payload, encoding="utf-8", newline="\n")
    manifest = {
        "dataset": MINI_NAME, "mini_revision": MINI_REVISION, "split": "test",
        "evaluation_metadata_source": VERIFIED_NAME,
        "instance_count": len(records), "instance_ids": [r["instance_id"] for r in records],
        "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    }
    write_json(args.dataset.with_suffix(".manifest.json"), manifest)
    print(f"Prepared Verified Mini: {len(records)} tasks -> {args.dataset}")
    print("No model requests, checkouts, image pulls or evaluations were started.")


def checked_records(dataset: Path) -> list[dict]:
    records = load_instances(dataset)
    manifest = json.loads(dataset.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != MINI_NAME or manifest["mini_revision"] != MINI_REVISION:
        raise ValueError("Dataset manifest does not match the pinned Mini source")
    if len(records) != 50 or [r["instance_id"] for r in records] != manifest["instance_ids"]:
        raise ValueError("Mini dataset IDs/count do not match its manifest")
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Mini dataset checksum mismatch; run prepare again")
    for row in records:
        if set(row) != GENERATION_FIELDS:
            raise ValueError("Generation dataset contains non-generation fields; run prepare to split scoring data")
    return records


def checked_evaluation_records(dataset: Path) -> list[dict]:
    manifest = json.loads(dataset.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source") != VERIFIED_NAME or hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("Private scoring dataset checksum/source mismatch")
    rows = load_instances(dataset)
    if [row["instance_id"] for row in rows] != manifest["instance_ids"]:
        raise ValueError("Private scoring membership mismatch")
    return rows


def prepare_environments(args, records):
    import docker
    import shlex
    selected = select_records(records, args.instance_id, args.limit)
    scoring = {row["instance_id"]: row for row in checked_evaluation_records(args.evaluation_dataset)}
    existing = load_environments(args.environments) if args.environments.exists() else {}
    client = docker.from_env(timeout=300)
    try:
        for row in selected:
            key = row["instance_id"]
            source = scoring[key]
            if generation_record(source) != row:
                raise ValueError("Scoring/base metadata mismatch")
            cached = existing.get(key, {})
            if (cached.get("audit") == "runtime-only-v1" and cached.get("base_commit") == row["base_commit"]
                    and cached.get("smoke_command") and cached.get("source_module")):
                try:
                    image = client.images.get(cached["image_id"])
                except docker.errors.ImageNotFound:
                    pass
                else:
                    config = image.attrs["Config"]
                    if image.id == cached["image_id"] and not any(config.get(k) for k in ("Env", "Entrypoint", "Volumes")):
                        print(f"Reusing audited runtime for {key}", flush=True)
                        continue
            image_name = source["image"]
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                print(f"Pulling environment source for {key}", flush=True)
                from .swebench_eval import _pull_with_credential_fallback
                def pull(c, repository, **kw):
                    last_progress = time.monotonic()
                    for event in c.api.pull(repository, stream=True, decode=True, **kw):
                        if event.get("error"):
                            raise docker.errors.APIError(event["error"])
                        if time.monotonic() - last_progress >= 20:
                            print(f"Image {key}: {event.get('status', '')} {event.get('progress', '')}", flush=True)
                            last_progress = time.monotonic()
                    return c.images.get(repository)
                _pull_with_credential_fallback(pull, client, image_name)
            module = {"django/django": "django", "marshmallow-code/marshmallow": "marshmallow",
                      "scikit-learn/scikit-learn": "sklearn", "matplotlib/matplotlib": "matplotlib",
                      "psf/requests": "requests", "pytest-dev/pytest": "pytest", "pallets/flask": "flask"}.get(row["repo"], row["repo"].split("/")[-1].replace("-", "_"))
            print(f"Flattening runtime-only image for {key}", flush=True)
            image_id = prepare_runtime(image_name, client, "myagent-runtime-" + key.lower(), source_module=module)
            check_script = (f"import sys, {module}; print(sys.version); print({module}.__file__); "
                            f"assert {module}.__file__.startswith('/testbed/'), 'Task module must come from base source'")
            existing[key] = {"image_id": image_id, "audit": "runtime-only-v1",
                             "python": "/opt/miniconda3/envs/testbed/bin/python",
                             "check_command": "python -c " + shlex.quote(check_script) + " && python -m unittest -h >/dev/null",
                             "base_commit": row["base_commit"]}
            existing[key]["source_module"] = module
            existing[key]["smoke_command"] = {"django/django": "python tests/runtests.py basic --parallel 1",
                                              "sphinx-doc/sphinx": "python -m pytest -q tests/test_util.py"}.get(row["repo"])
            args.environments.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.environments, {"schema": 1, "environments": existing})
    finally:
        client.close()


def check_isolation(args, records, *, root=None):
    import tempfile
    environments = load_environments(args.environments)
    selected = select_records(records, args.instance_id, args.limit)
    with tempfile.TemporaryDirectory(prefix="myagent-isolation-", dir=root) as temporary:
        results = {}
        for row in selected:
            spec = environments[row["instance_id"]]
            if spec["base_commit"] != row["base_commit"]:
                raise ValueError("Environment/base commit mismatch")
            checkout = prepare_checkout(row, Path(temporary))
            sandbox = DockerSandbox(checkout, spec, timeout=args.tool_timeout, instance_seconds=args.instance_seconds)
            try:
                results[row["instance_id"]] = {"image_id": spec["image_id"], "checks": isolation_check(sandbox),
                                                "preflight": sandbox.preflight, "image_audit": sandbox.image_audit,
                                                "boundary": sandbox.boundary}
                if not spec.get("smoke_command"):
                    raise ValueError("A real existing-test smoke command is required")
                results[row["instance_id"]]["test_output"] = sandbox.execute("run_shell", {"command": spec["smoke_command"]})
            finally:
                sandbox.close()
            print(f"Isolation passed: {row['instance_id']}", flush=True)
        args.environments.with_suffix(".isolation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return results


def select_records(records: list[dict], ids: list[str], limit: int) -> list[dict]:
    wanted = set(ids)
    missing = wanted - {r["instance_id"] for r in records}
    if missing:
        raise ValueError(f"Unknown Mini instance IDs: {sorted(missing)}")
    if len(wanted) > limit:
        raise ValueError("--limit is smaller than the requested instance count")
    return [r for r in records if not wanted or r["instance_id"] in wanted][:limit]


class BudgetedClient:
    """Share one conservative budget across the batch; never retry SDK calls."""

    def __init__(self, inner, budget: Decimal, max_output: int, usage_path: Path, batch_seconds=7200, request_timeout=180):
        if not budget.is_finite() or budget <= 0 or max_output < 1:
            raise ValueError("Budget and output limit must be positive and finite")
        self.model = inner.model
        if self.model not in {"deepseek-flash", "deepseek-v4-flash"}:
            raise ValueError("Budgeted Mini runner supports only DeepSeek Flash pricing")
        if urlsplit(str(inner.client.base_url)).hostname != "api.deepseek.com":
            raise ValueError("Budget pricing requires the official DeepSeek API endpoint")
        self.sdk = inner.client.with_options(max_retries=0, timeout=120)
        self.inner = inner
        self.inner.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create)))
        self.budget = budget
        self.max_output = max_output
        self.usage_path = usage_path
        self.spent = Decimal("0")
        self.stopped = False
        self.instance_id = ""
        self.calls = 0
        self.deadline = time.monotonic() + batch_seconds
        self.instance_deadline = self.deadline
        self.request_timeout = request_timeout

    def complete(self, messages, *, tools=None, max_new_tokens=None):
        return self.inner.complete(messages, tools=tools, max_new_tokens=max_new_tokens)

    def _bounded_request(self, kwargs, seconds):
        # SDK timeouts bound socket inactivity, not total wall time. A daemon
        # worker cannot keep the controller alive after an abandoned request.
        result = queue.Queue(maxsize=1)
        def request():
            try:
                result.put((True, self.sdk.chat.completions.create(**kwargs)))
            except Exception as exc:
                result.put((False, exc))
        threading.Thread(target=request, daemon=True).start()
        try:
            success, value = result.get(timeout=seconds)
        except queue.Empty:
            raise TimeoutError(f"Model request exceeded {seconds:.1f}s wall-time limit; batch stopped") from None
        if not success:
            raise value
        return value

    def create(self, **kwargs):
        if self.stopped:
            raise RuntimeError("Batch stopped; further model requests are disabled")
        remaining = min(self.deadline, self.instance_deadline) - time.monotonic()
        if remaining <= 0:
            self.stopped = time.monotonic() >= self.deadline
            raise RuntimeError("Model request time limit reached")
        kwargs["timeout"] = min(self.request_timeout, remaining)
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", self.max_output), self.max_output)
        kwargs["reasoning_effort"] = "low"
        size = len(json.dumps({"messages": kwargs["messages"], "tools": kwargs.get("tools", [])},
                              ensure_ascii=False).encode("utf-8"))
        token_bound = size + 256 * (len(kwargs["messages"]) + len(kwargs.get("tools", [])))
        reserve = (token_bound * INPUT_RATE + kwargs["max_tokens"] * OUTPUT_RATE) / MILLION
        if self.spent + reserve > self.budget:
            self.stopped = True
            raise RuntimeError(f"Budget guard: spent={self.spent}, next reserve={reserve}, "
                               f"batch budget={self.budget} CNY")
        self.calls += 1
        record = {"instance_id": self.instance_id, "call": self.calls,
                  "max_output_tokens": kwargs["max_tokens"], "reserved_cny": str(reserve)}
        started = time.perf_counter()
        try:
            response = self._bounded_request(kwargs, kwargs["timeout"])
            record["finish_reason"] = getattr(response.choices[0], "finish_reason", None) if getattr(response, "choices", None) else None
            usage = getattr(response, "usage", None)
            if usage is None:
                self.spent += reserve
                record["usage_missing"] = True
            else:
                cost = (usage.prompt_tokens * INPUT_RATE + usage.completion_tokens * OUTPUT_RATE) / MILLION
                self.spent += cost
                record.update(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                              conservative_peak_cny=str(cost))
            return response
        except Exception:
            self.spent += reserve
            self.stopped = True  # An ambiguous failure must not trigger more paid calls.
            record["request_failed"] = True
            raise
        finally:
            record.update(elapsed_seconds=round(time.perf_counter() - started, 3),
                          cumulative_peak_cny=str(self.spent))
            with self.usage_path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(json.dumps(record) + "\n")


def generate(args, records: list[dict], run_dir: Path) -> int:
    selected = select_records(records, args.instance_id, args.limit)
    # Actual Docker checks must pass before any paid client is constructed.
    check_isolation(args, records)
    environments = load_environments(args.environments)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "dataset.jsonl").write_text(
        "".join(json.dumps(generation_record(r), ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    client = BudgetedClient(OpenAICompatibleModelClient(AgentParams(max_output_tokens=args.max_output_tokens)),
                            args.budget_cny, args.max_output_tokens, run_dir / "usage.jsonl", args.batch_seconds, args.request_timeout)
    summary = {"benchmark": "SWE-bench Verified Mini", "split": "test", "model": client.model,
               "reasoning_effort": "low", "budget_cny": str(args.budget_cny),
               "max_turns_per_instance": args.max_turns, "max_output_tokens": args.max_output_tokens,
               "sdk_retries": 0, "instance_seconds": args.instance_seconds, "batch_seconds": args.batch_seconds,
               "tool_timeout": args.tool_timeout, "selected_ids": [r["instance_id"] for r in selected], "instances": []}
    summary["request_timeout"] = args.request_timeout
    summary["generation_dataset_sha256"] = hashlib.sha256((run_dir / "dataset.jsonl").read_bytes()).hexdigest()
    started = time.perf_counter()
    try:
        with (run_dir / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as output:
            for row in selected:
                client.instance_id = row["instance_id"]
                instance_started = time.perf_counter()
                client.instance_deadline = time.monotonic() + args.instance_seconds
                response = None
                try:
                    checkout = prepare_checkout(row, run_dir / "checkouts")
                    prediction, response = run_instance(row, checkout, client,
                                                       max_turns=args.max_turns, tool_timeout=args.tool_timeout,
                                                       max_output_tokens=args.max_output_tokens,
                                                       instance_seconds=args.instance_seconds,
                                                       environment=environments[row["instance_id"]])
                    detail = {"instance_id": row["instance_id"], "stop_reason": response.stop_reason.value,
                              "turns": response.turns_used, "tool_calls": response.tool_calls,
                              "error": response.error, "patch_chars": len(prediction["model_patch"])}
                except Exception as exc:
                    prediction = {"instance_id": row["instance_id"], "model_name_or_path": client.model,
                                  "model_patch": ""}
                    detail = {"instance_id": row["instance_id"], "stop_reason": "error", "error": str(exc)}
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
                detail["patch_sha256"] = hashlib.sha256(prediction["model_patch"].encode()).hexdigest()
                detail["patch_chars"] = len(prediction["model_patch"])
                detail["delivery_status"] = "delivered" if prediction["model_patch"] else "undelivered_empty_patch"
                detail["truncated_responses"] = sum(s.assistant_metadata.get("finish_reason") == "length" for s in response.steps) if response else 0
                detail["empty_responses"] = sum(s.assistant_metadata.get("rejected_response", "").startswith("Empty response") for s in response.steps) if response else 0
                detail["tool_timeouts"] = sum(o.error_type == "TimeoutError" for s in response.steps for o in s.outcomes) if response else 0
                detail["verification"] = response.verification if response else {}
                if response:
                    from dataclasses import asdict
                    (run_dir / (row["instance_id"] + ".trajectory.json")).write_text(
                        json.dumps([asdict(s) for s in response.steps], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                detail["elapsed_seconds"] = round(time.perf_counter() - instance_started, 3)
                summary["instances"].append(detail)
                summary.update(model_calls=client.calls, conservative_peak_cny=str(client.spent),
                               generation_elapsed_seconds=round(time.perf_counter() - started, 3))
                summary["unattempted_ids"] = [r["instance_id"] for r in selected[len(summary["instances"]):]]
                write_json(run_dir / "generation-summary.json", summary)
                print(json.dumps(detail, ensure_ascii=False), flush=True)
                if client.stopped:
                    break
    finally:
        summary["predictions_sha256"] = hashlib.sha256((run_dir / "predictions.jsonl").read_bytes()).hexdigest()
        summary["generation_finished"] = True
        write_json(run_dir / "generation-summary.json", summary)
        client.sdk.close()
    print(f"Generation summary: {run_dir / 'generation-summary.json'}")
    return 1 if client.stopped or any(r["stop_reason"] != StopReason.FINAL_ANSWER.value or r["delivery_status"] != "delivered"
                                     for r in summary["instances"]) else 0


def verify_seal(run_dir: Path) -> dict:
    seal = json.loads((run_dir / "generation-summary.json").read_text(encoding="utf-8"))
    predictions = run_dir / "predictions.jsonl"
    if not seal.get("generation_finished") or hashlib.sha256(predictions.read_bytes()).hexdigest() != seal.get("predictions_sha256"):
        raise ValueError("Generation is unfinished or predictions changed after sealing")
    expected = {row["instance_id"]: row["patch_sha256"] for row in seal["instances"]}
    seen = set()
    for line in predictions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["instance_id"]
        if key in seen or key not in expected or hashlib.sha256(row["model_patch"].encode()).hexdigest() != expected[key]:
            raise ValueError("Prediction membership/patch hash mismatch")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Predictions do not match completed generation records")
    dataset_hash = seal.get("generation_dataset_sha256")
    if not dataset_hash or hashlib.sha256((run_dir / "dataset.jsonl").read_bytes()).hexdigest() != dataset_hash:
        raise ValueError("Generation dataset changed or has no sealed hash")
    return seal


def check(records: list[dict], args) -> None:
    import docker
    from importlib.metadata import version
    from .context.tokens import count_tokens
    client = docker.from_env(timeout=15)
    try:
        if not client.ping():
            raise RuntimeError("Docker ping failed")
        print(f"Docker ready: {client.version()['Version']}")
    finally:
        client.close()
    for name in ("swebench", "datasets", "tiktoken"):
        print(f"{name}={version(name)}")
    selected = select_records(records, args.instance_id, args.limit)
    print(f"Verified Mini test: {len(records)} tasks; default batch: {len(selected)}")
    print(f"Selected IDs: {[r['instance_id'] for r in selected]}")
    print(f"Batch budget: {args.budget_cny} CNY; output limit: {args.max_output_tokens}; "
          f"max turns per instance: {args.max_turns}")
    print(f"Tokenizer ready: {count_tokens('Verified Mini setup check')} tokens")
    print("Prerequisites checked. No model requests, checkouts, image pulls or evaluations started.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("prepare", "prepare-environments", "isolation-check", "check", "generate", "eval", "all"), default="check")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--evaluation-dataset", type=Path, default=DEFAULT_EVALUATION_DATASET)
    parser.add_argument("--environments", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument("--mini-source", type=Path)
    parser.add_argument("--verified-source", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--budget-cny", type=Decimal, default=Decimal("2"))
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--tool-timeout", type=float, default=60)
    parser.add_argument("--instance-seconds", type=float, default=1800)
    parser.add_argument("--batch-seconds", type=float, default=7200)
    parser.add_argument("--request-timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 50 or args.max_turns < 1 or args.max_output_tokens < 1:
        parser.error("limit must be 1..50; turn/output limits must be positive")
    if not args.budget_cny.is_finite() or args.budget_cny <= 0:
        parser.error("budget must be positive and finite")
    if any(not 0 < value < float('inf') for value in (args.tool_timeout, args.instance_seconds, args.batch_seconds, args.request_timeout)):
        parser.error("Time limits must be positive and finite")
    if args.stage == "prepare":
        prepare_dataset(args)
        return 0
    records = checked_records(args.dataset)
    if args.stage == "prepare-environments":
        prepare_environments(args, records)
        return 0
    if args.stage == "isolation-check":
        check_isolation(args, records)
        return 0
    if args.stage == "check":
        check(records, args)
        return 0
    if args.stage == "eval" and not args.run_id:
        parser.error("eval requires --run-id from an existing generation")
    run_id = args.run_id or "verified-mini-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in run_id) or run_id in {".", ".."}:
        parser.error("Invalid run ID")
    run_dir = Path(".swebench-work/runs") / run_id
    print(f"Run ID: {run_id}")
    status = 0
    if args.stage in {"generate", "all"}:
        status = generate(args, records, run_dir)
    if args.stage in {"eval", "all"}:
        # Evaluate the selected snapshot only, including unfinished tasks in its denominator.
        predictions = run_dir / "predictions.jsonl"
        dataset = run_dir / "dataset.jsonl"
        if not predictions.exists() or not dataset.exists():
            parser.error("Run is missing predictions or its selected dataset snapshot")
        allowed = {r["instance_id"] for r in records}
        snapshot = load_instances(dataset)
        if not snapshot or len(snapshot) > 50 or any(r["instance_id"] not in allowed for r in snapshot):
            parser.error("Run snapshot contains tasks outside Verified Mini")
        seal = verify_seal(run_dir)
        if [row["instance_id"] for row in snapshot] != seal["selected_ids"]:
            parser.error("Evaluation selection differs from generation selection")
        # Only now load private scoring records, in the evaluator controller.
        scoring = {r["instance_id"]: r for r in checked_evaluation_records(args.evaluation_dataset)}
        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        rows = []
        for row in snapshot:
            source = scoring[row["instance_id"]]
            if generation_record(source) != row:
                parser.error("Scoring snapshot differs from generation metadata")
            rows.append(source)
        dataset = evaluation_dir / "dataset.jsonl"
        dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        evaluated = time.perf_counter()
        result = subprocess.run([
            sys.executable, "-m", "myagent.swebench_eval", "--dataset_name", str(dataset),
            "--split", "test", "--predictions_path", str(predictions), "--max_workers", "1",
            "--run_id", run_id, "--report_dir", str(run_dir),
        ], check=False)
        write_json(run_dir / "evaluation-timing.json", {
            "elapsed_seconds": round(time.perf_counter() - evaluated, 3), "exit_code": result.returncode})
        status = status or result.returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
