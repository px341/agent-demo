"""SWE-bench prediction runner tests; all use local temporary repositories."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from myagent.contracts import LLMResponse, StopReason
from myagent.errors import ValidationError
from myagent.swebench import (
    BenchmarkToolExecutor,
    capture_patch,
    load_instances,
    run_instance,
)
from myagent.swebench_eval import _container_artifact_write_text


class ScriptedLLM:
    model = "fake-swe-model"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, tools=None, max_new_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    {
                        "id": "write-1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {"path": "answer.txt", "content": "fixed\n"}
                            ),
                        },
                    }
                ],
            )
        return LLMResponse(text="Implemented the fix.")


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "answer.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "answer.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)


class DatasetTest(unittest.TestCase):
    def test_load_jsonl(self):
        record = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc",
            "problem_statement": "fix it",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(load_instances(path), [record])

    def test_missing_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text('[{"instance_id": "x"}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                load_instances(path)


class PredictionTest(unittest.TestCase):
    def test_windows_eval_artifacts_are_written_with_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.sh"
            _container_artifact_write_text(path, "a\nb\n", encoding="utf-8")
            self.assertEqual(path.read_bytes(), b"a\nb\n")

    def test_benchmark_executor_keeps_boundary_but_error_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = BenchmarkToolExecutor(Path(tmp), timeout=10)
            with self.assertRaises(ValidationError) as ctx:
                executor.execute("read_file", {"path": "../outside.txt"})
            self.assertTrue(ctx.exception.recoverable)

    def test_run_instance_writes_standard_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            init_repo(repo)
            instance = {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": "unused",
                "problem_statement": "Change old to fixed.",
            }
            prediction, response = run_instance(instance, repo, ScriptedLLM())
            self.assertEqual(response.stop_reason, StopReason.FINAL_ANSWER)
            self.assertEqual(
                set(prediction),
                {"instance_id", "model_name_or_path", "model_patch"},
            )
            self.assertEqual(prediction["model_name_or_path"], "fake-swe-model")
            self.assertIn("-old", prediction["model_patch"])
            self.assertIn("+fixed", prediction["model_patch"])

    def test_capture_patch_excludes_untracked_scratch_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            init_repo(repo)
            (repo / "new.txt").write_text("new\n", encoding="utf-8")
            patch = capture_patch(repo)
            self.assertEqual(patch, "")


if __name__ == "__main__":
    unittest.main()
