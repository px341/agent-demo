"""SWE-bench prediction runner tests; all use local temporary repositories."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from myagent.contracts import LLMResponse, StopReason
from myagent.errors import ValidationError
from myagent.swebench import (
    capture_patch,
    load_instances,
    prepare_checkout,
    run_instance,
)
from myagent.swebench_eval import _container_artifact_write_text, _pull_with_credential_fallback


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
    def test_public_image_pull_recovers_from_failed_credential_helper(self):
        original = Mock(side_effect=[RuntimeError("Credentials store error: helper failed"), "image"])
        client = object()
        self.assertEqual(_pull_with_credential_fallback(original, client, "swebench/image", tag="latest"), "image")
        self.assertEqual(original.call_count, 2)
        self.assertEqual(original.call_args.kwargs["auth_config"], {})

    def test_pull_does_not_retry_other_repositories_or_registry_failures(self):
        for repo, message in (("private/image", "Credentials store error"), ("swebench/image", "registry unavailable")):
            original = Mock(side_effect=RuntimeError(message))
            with self.subTest(repository=repo), self.assertRaises(RuntimeError):
                _pull_with_credential_fallback(original, object(), repo)
            self.assertEqual(original.call_count, 1)

    def test_checkout_cannot_read_future_repair_commit(self):
        from myagent.swebench import _run_git
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            init_repo(source)
            base = _run_git(["rev-parse", "HEAD"], cwd=source).strip()
            (source / "answer.txt").write_text("future solution\n", encoding="utf-8")
            _run_git(["add", "answer.txt"], cwd=source)
            _run_git(["commit", "-qm", "future fix"], cwd=source)
            future = _run_git(["rev-parse", "HEAD"], cwd=source).strip()
            _run_git(["tag", "future-fix"], cwd=source)

            def local_transport(args, *, cwd=None):
                args = [source.as_uri() if arg == "https://github.com/owner/repo.git" else arg for arg in args]
                return _run_git(args, cwd=cwd)

            instance = {"instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": base}
            with patch("myagent.swebench._run_git", side_effect=local_transport):
                checkout = prepare_checkout(instance, root / "checkouts")
            self.assertEqual(_run_git(["rev-parse", "HEAD"], cwd=checkout).strip(), base)
            self.assertEqual((checkout / "answer.txt").read_text(), "old\n")
            self.assertEqual(_run_git(["rev-list", "--all", "--count"], cwd=checkout).strip(), "1")
            self.assertEqual(_run_git(["remote"], cwd=checkout).strip(), "")
            self.assertEqual(_run_git(["tag"], cwd=checkout).strip(), "")
            self.assertFalse((checkout / ".git/FETCH_HEAD").exists())
            self.assertFalse((checkout / ".git/objects/info/alternates").exists())
            with self.assertRaises(RuntimeError):
                _run_git(["show", future], cwd=checkout)

    def test_generation_requires_docker_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Docker environment"):
                run_instance({}, Path(tmp), ScriptedLLM())

    def test_windows_eval_artifacts_are_written_with_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.sh"
            _container_artifact_write_text(path, "a\nb\n", encoding="utf-8")
            self.assertEqual(path.read_bytes(), b"a\nb\n")

    def test_generation_export_uses_strict_allowlist(self):
        from myagent.swebench import generation_record
        record = {"instance_id": "a", "repo": "b/c", "base_commit": "d", "problem_statement": "e",
                  "patch": "secret", "test_patch": "secret", "FAIL_TO_PASS": "secret", "future_answer": "secret"}
        self.assertEqual(set(generation_record(record)), {"instance_id", "repo", "base_commit", "problem_statement"})

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
            from myagent.tools import ToolExecutor
            class FakeSandbox:
                preflight = "test runtime"
                def execute(self, name, args):
                    return ToolExecutor(repo).execute(name, args)
                def capture_patch(self):
                    return capture_patch(repo)
                def close(self):
                    pass
            with patch("myagent.benchmark_checks.isolation_check", return_value={}):
                prediction, response = run_instance(instance, repo, ScriptedLLM(), sandbox=FakeSandbox())
            self.assertEqual(response.stop_reason, StopReason.FINAL_ANSWER)
            self.assertEqual(
                set(prediction),
                {"instance_id", "model_name_or_path", "model_patch"},
            )
            self.assertEqual(prediction["model_name_or_path"], "fake-swe-model")
            self.assertIn("-old", prediction["model_patch"])
            self.assertIn("+fixed", prediction["model_patch"])

    def test_capture_patch_includes_new_files_and_preserves_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            init_repo(repo)
            (repo / "new.txt").write_text("new\n", encoding="utf-8")
            (repo / ".gitignore").write_text("cache/\n")
            (repo / "cache").mkdir()
            (repo / "cache/temp").write_text("ignored")
            before = (repo / ".git/index").read_bytes()
            result = capture_patch(repo)
            self.assertIn("new file mode", result)
            self.assertIn("+new", result)
            self.assertNotIn("cache/temp", result)
            self.assertEqual(before, (repo / ".git/index").read_bytes())


if __name__ == "__main__":
    unittest.main()
