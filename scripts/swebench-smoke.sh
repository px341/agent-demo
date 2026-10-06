#!/usr/bin/env bash
# Run from WSL/Linux. The default stage only checks prerequisites.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

stage="${1:-check}"
case "$stage" in
  check|generate|eval|all) ;;
  *) echo "Usage: bash scripts/swebench-smoke.sh [check|generate|eval|all] [run-id]" >&2; exit 2 ;;
esac

python="${MYAGENT_PYTHON:-$PWD/agentvenv/bin/python}"
dataset="$PWD/.swebench-work/data/swebench-lite-dev.jsonl"
run_id="${2:-myagent-dev-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9_.-]+$ || "$run_id" == . || "$run_id" == .. ]]; then
  echo "run-id must contain only letters, numbers, dots, underscores and hyphens" >&2
  exit 2
fi
if [[ "$stage" == eval && $# -lt 2 ]]; then
  echo "Pass the run-id printed during generation to evaluate its predictions." >&2
  exit 2
fi
run_dir="$PWD/.swebench-work/runs/$run_id"

"$python" -m pip check
"$python" - "$dataset" "$stage" <<'PY'
import os
import sys
from pathlib import Path
from importlib.metadata import version
from myagent.provider import ENV_PATH, DEFAULT_MODEL
from myagent.swebench import load_instances
from swebench.harness.utils import load_swebench_dataset, make_test_spec
from myagent.context.tokens import count_tokens

for name in ("swebench", "datasets", "tiktoken"):
    print(f"{name}={version(name)}")
records = load_instances(Path(sys.argv[1]))
assert records, "dataset is empty"
official = load_swebench_dataset(sys.argv[1], split="dev")
assert len(records) == len(official)
specs = [make_test_spec(instance) for instance in official]
print(f"Lite dev dataset: {len(records)} instances; first={records[0]['instance_id']}")
print(f"Official evaluation specs ready: {len(specs)}; first image={specs[0].image}")
print(f"Tokenizer ready: {count_tokens('SWE-bench smoke test')} tokens")
key = os.getenv("DEEPSEEK_API_KEY", "").strip()
ready = bool(key) and key not in {"your-api-key", "sk-xxx", "unit-test-placeholder"}
print(f"Model config: {ENV_PATH}; key={'configured' if ready else 'pending'}")
print(f"Model: {os.getenv('DEEPSEEK_MODEL') or DEFAULT_MODEL}")
if sys.argv[2] in {"generate", "all"} and not ready:
    raise SystemExit("Fill in .env.llm before generating predictions.")
PY

if [[ "$stage" == check || "$stage" == eval || "$stage" == all ]]; then
  "$python" - <<'PY'
import docker
try:
    client = docker.from_env(timeout=15)
    try:
        assert client.ping()
        print(f"Docker ready: {client.version()['Version']}")
    finally:
        client.close()
except Exception as exc:
    raise SystemExit(
        "Docker connection failed. Start Docker Desktop and enable Ubuntu-24.04 "
        "in Settings > Resources > WSL Integration.\n" + str(exc)
    )
PY
fi

if [[ "$stage" == check ]]; then
  echo "Prerequisites checked; no model requests or evaluations were made."
  exit 0
fi
echo "Run ID: $run_id"
echo "Predictions: $run_dir/predictions.jsonl"
mkdir -p "$run_dir"

if [[ "$stage" == generate || "$stage" == all ]]; then
  "$python" -m myagent.swebench \
    --dataset "$dataset" \
    --predictions "$run_dir/predictions.jsonl" \
    --work-root "$run_dir/checkouts" \
    --limit 1 --max-turns 30 --tool-timeout 600
fi

if [[ "$stage" == eval || "$stage" == all ]]; then
  if [[ ! -s "$run_dir/predictions.jsonl" ]]; then
    echo "No predictions found for run $run_id" >&2
    exit 1
  fi
  "$python" -m myagent.swebench_eval \
    --dataset_name "$dataset" --split dev \
    --predictions_path "$run_dir/predictions.jsonl" \
    --max_workers 1 --run_id "$run_id" --report_dir "$run_dir"
fi
