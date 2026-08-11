"""ConsoleApprovalGate 交互测试：y/n/非法输入/EOF。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.approval import ConsoleApprovalGate


class ConsoleApprovalGateTest(unittest.TestCase):
    def test_yes_allows(self):
        with mock.patch("builtins.input", return_value="y"):
            self.assertTrue(ConsoleApprovalGate().request("read_file", {"path": "a"}))

    def test_yes_variants(self):
        for answer in ("Y", "yes", "YES", " y "):
            with mock.patch("builtins.input", return_value=answer):
                self.assertTrue(ConsoleApprovalGate().request("t", {}))

    def test_no_denies(self):
        for answer in ("n", "N", "no"):
            with mock.patch("builtins.input", return_value=answer):
                self.assertFalse(ConsoleApprovalGate().request("t", {}))

    def test_invalid_input_retries(self):
        with mock.patch("builtins.input", side_effect=["x", "y"]):
            self.assertTrue(ConsoleApprovalGate().request("t", {}))

    def test_eof_denies_by_default(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(ConsoleApprovalGate().request("t", {}))

    def test_prompt_shows_tool_and_args(self):
        with mock.patch("builtins.input", return_value="y") as mocked:
            ConsoleApprovalGate().request("delete_file", {"path": "a.txt"})
        prompt = mocked.call_args[0][0]
        self.assertIn("delete_file", prompt)
        self.assertIn("a.txt", prompt)

    def test_read_risk_auto_approved(self):
        """read 类工具默认免询问。"""
        with mock.patch("builtins.input") as mocked:
            self.assertTrue(
                ConsoleApprovalGate().request("read_file", {"path": "a"})
            )
        mocked.assert_not_called()

    def test_unknown_tool_still_asks(self):
        """未注册工具（spec 未知）不享受免审，走询问（安全默认）。"""
        with mock.patch("builtins.input", return_value="n") as mocked:
            self.assertFalse(ConsoleApprovalGate().request("no_such_tool", {}))
        mocked.assert_called_once()

    def test_write_risk_asks(self):
        with mock.patch("builtins.input", return_value="y") as mocked:
            self.assertTrue(
                ConsoleApprovalGate().request("write_file", {"path": "a"})
            )
        mocked.assert_called_once()

    def test_delete_risk_asks(self):
        with mock.patch("builtins.input", return_value="n"):
            self.assertFalse(
                ConsoleApprovalGate().request("delete_dir", {"path": "d"})
            )

    def test_all_ask_when_auto_approve_empty(self):
        """auto_approve_risks=set() 恢复全询问（read 也问）。"""
        gate = ConsoleApprovalGate(auto_approve_risks=set())
        with mock.patch("builtins.input", return_value="y"):
            self.assertTrue(gate.request("read_file", {}))


if __name__ == "__main__":
    unittest.main()
