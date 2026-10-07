"""Verified Mini membership, paid-call guards and safe default dispatch."""
from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from myagent.swebench_mini import BudgetedClient, adapt_mini, generate, main, select_records, verify_seal


def row(instance_id="owner__repo-1"):
    return {"instance_id": instance_id, "repo": "owner/repo", "base_commit": "abc",
            "problem_statement": "fix", "patch": "reference"}


def official(record):
    return {**record, "image": "test/image", "eval_script": "echo test",
            "eval_type": "unit", "log_parser": "pytest"}


class MiniDatasetTest(unittest.TestCase):
    def test_isolation_failure_prevents_paid_client_construction(self):
        args = SimpleNamespace(instance_id=[], limit=1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch("myagent.swebench_mini.check_isolation", side_effect=RuntimeError("isolation failed")), \
             patch("myagent.swebench_mini.OpenAICompatibleModelClient") as model:
            with self.assertRaisesRegex(RuntimeError, "isolation failed"):
                generate(args, [row()], Path(tmp) / "run")
            model.assert_not_called()

    def test_seal_rejects_tampering_and_unfinished_generation(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"instance_id": "a", "model_patch": "patch"}) + "\n")
            dataset = root / "dataset.jsonl"
            dataset.write_text("base snapshot")
            seal = {"generation_finished": True,
                    "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                    "generation_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    "instances": [{"instance_id": "a", "patch_sha256": hashlib.sha256(b"patch").hexdigest()}]}
            summary = root / "generation-summary.json"
            summary.write_text(json.dumps(seal))
            verify_seal(root)
            predictions.write_text("changed")
            with self.assertRaisesRegex(ValueError, "predictions changed"):
                verify_seal(root)
            seal["generation_finished"] = False
            summary.write_text(json.dumps(seal))
            with self.assertRaisesRegex(ValueError, "unfinished"):
                verify_seal(root)
    def test_adaptation_keeps_mini_order_and_ignores_other_verified_tasks(self):
        mini = [row("owner__repo-2"), row("owner__repo-1")]
        full = [official(row("owner__repo-1")), official(row("owner__repo-3")),
                official(row("owner__repo-2"))]
        result = adapt_mini(mini, full)
        self.assertEqual([r["instance_id"] for r in result], [r["instance_id"] for r in mini])
        self.assertEqual(result[0]["patch"], "reference")
        self.assertIn("image", result[0])

    def test_base_commit_mismatch_is_rejected(self):
        full = official(row())
        full["base_commit"] = "different"
        with self.assertRaisesRegex(ValueError, "base_commit"):
            adapt_mini([row()], [full])

    def test_missing_duplicate_ids_and_missing_eval_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in official"):
            adapt_mini([row()], [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapt_mini([row(), row()], [official(row())])
        with self.assertRaisesRegex(ValueError, "fields missing"):
            adapt_mini([row()], [row()])

    def test_requested_ids_are_not_silently_truncated(self):
        records = [row("owner__repo-1"), row("owner__repo-2")]
        with self.assertRaisesRegex(ValueError, "smaller"):
            select_records(records, [r["instance_id"] for r in records], 1)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            select_records(records, ["unknown"], 3)

    def test_default_stage_never_constructs_model_or_runs_evaluator(self):
        with patch("myagent.swebench_mini.checked_records", return_value=[row()]), \
             patch("myagent.swebench_mini.check") as check, \
             patch("myagent.swebench_mini.OpenAICompatibleModelClient") as model, \
             patch("myagent.swebench_mini.subprocess.run") as process:
            self.assertEqual(main([]), 0)
        check.assert_called_once()
        model.assert_not_called()
        process.assert_not_called()


class FakeSDK:
    base_url = "https://api.deepseek.com"

    def __init__(self, *, error=False, missing_usage=False):
        self.error = error
        self.missing_usage = missing_usage
        self.requests = []
        self.options = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise RuntimeError("ambiguous API failure")
        usage = None if self.missing_usage else SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        return SimpleNamespace(usage=usage)


class BudgetTest(unittest.TestCase):
    def test_wall_timeout_stops_batch_even_when_sdk_ignores_socket_timeout(self):
        import threading
        sdk = FakeSDK()
        release = threading.Event()
        sdk.chat.completions.create = lambda **kw: release.wait(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client, _ = self.make_client(tmp, sdk, budget="2")
                client.request_timeout = 0.02
                with self.assertRaisesRegex(TimeoutError, "wall-time"):
                    self.request(client)
                self.assertTrue(client.stopped)
                self.assertGreater(client.spent, 0)
                with self.assertRaisesRegex(RuntimeError, "Batch stopped"):
                    self.request(client)
        finally:
            release.set()

    def make_client(self, tmp, sdk=None, budget="0.00090"):
        sdk = sdk or FakeSDK()
        inner = SimpleNamespace(model="deepseek-flash", client=sdk)
        return BudgetedClient(inner, Decimal(budget), 10, Path(tmp) / "usage.jsonl"), sdk

    def request(self, client):
        return client.create(model="deepseek-flash", messages=[{"role": "user", "content": "hi"}],
                             max_tokens=10000)

    def test_budget_blocks_next_paid_call_and_caps_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, sdk = self.make_client(tmp)
            self.request(client)
            with self.assertRaisesRegex(RuntimeError, "Budget guard"):
                self.request(client)
            self.assertEqual(len(sdk.requests), 1)
            self.assertEqual(sdk.requests[0]["max_tokens"], 10)
            self.assertEqual(sdk.options["max_retries"], 0)
            self.assertEqual(client.spent, Decimal("0.00028"))
            record = json.loads((Path(tmp) / "usage.jsonl").read_text())
            self.assertEqual(record["prompt_tokens"], 100)

    def test_ambiguous_failure_blocks_remaining_batch_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, sdk = self.make_client(tmp, FakeSDK(error=True), budget="2")
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                self.request(client)
            with self.assertRaisesRegex(RuntimeError, "Batch stopped"):
                self.request(client)
            self.assertEqual(len(sdk.requests), 1)
            self.assertGreater(client.spent, 0)

    def test_missing_usage_charges_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = self.make_client(tmp, FakeSDK(missing_usage=True), budget="2")
            self.request(client)
            record = json.loads((Path(tmp) / "usage.jsonl").read_text())
            self.assertEqual(client.spent, Decimal(record["reserved_cny"]))

    def test_other_providers_and_nonfinite_budgets_are_rejected_before_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            sdk.base_url = "https://example.invalid"
            with self.assertRaisesRegex(ValueError, "official DeepSeek"):
                self.make_client(tmp, sdk)
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                self.make_client(tmp, budget="NaN")
            self.assertEqual(sdk.requests, [])


if __name__ == "__main__":
    unittest.main()
