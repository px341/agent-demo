"""工具错误体系测试：分类、分级退出、会话记录、composer 兼容。

覆盖：
- errors.py 层级（recoverable / error_type）；
- 主循环：可恢复错误（Validation/NotFound）回灌继续；
- 不可恢复错误（Permission/Timeout/Execution）→ TOOL_ERROR 终止；
- 会话记录：TOOL_ERROR step 重建 assistant + tool(错误) 配对，
  经 step_to_messages 进 REPL history / memory 存档；
- composer：错误 tool 消息正常配对、不被误删。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams, Message
from myagent.agent_loop import AgentLoop, step_to_messages
from myagent.contracts import AgentRequest, LLMResponse, StopReason
from myagent.errors import (
    ApprovalDenied,
    ExecutionError,
    NotFoundError,
    ToolError,
    ToolPermissionError,
    ToolTimeoutError,
    ValidationError,
)
from myagent.memory.archive import append_archive, read_archive
from myagent.tools import ToolExecutor


def tc(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": __import__("json").dumps(args)},
    }


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    @staticmethod
    def _wrap(output):
        return output if isinstance(output, LLMResponse) else LLMResponse(text=output)

    def complete(self, messages, *, tools=None, max_new_tokens=None):
        self.calls.append(list(messages))
        return self._wrap(self.outputs.pop(0))


class ErrorsHierarchyTest(unittest.TestCase):
    def test_recoverable_flags(self):
        self.assertTrue(ValidationError("").recoverable)
        self.assertTrue(NotFoundError("").recoverable)
        self.assertFalse(ToolPermissionError("").recoverable)
        self.assertFalse(ApprovalDenied("").recoverable)
        self.assertFalse(ToolTimeoutError("").recoverable)
        self.assertFalse(ExecutionError("").recoverable)

    def test_error_types(self):
        self.assertEqual(ValidationError("x").error_type, "ValidationError")
        self.assertEqual(NotFoundError("x").error_type, "NotFoundError")
        # ToolPermissionError 的 error_type 对齐用户分类名 PermissionError。
        self.assertEqual(ToolPermissionError("x").error_type, "PermissionError")
        self.assertEqual(ApprovalDenied("x").error_type, "ApprovalDenied")
        self.assertEqual(ToolTimeoutError("x").error_type, "TimeoutError")
        self.assertEqual(ExecutionError("x").error_type, "ExecutionError")

    def test_str_format(self):
        self.assertEqual(
            str(NotFoundError("文件不存在：a.py")),
            "ToolError[NotFoundError]: 文件不存在：a.py",
        )

    def test_all_are_tool_error_subclass(self):
        for cls in (ValidationError, NotFoundError, ToolPermissionError,
                    ApprovalDenied, ToolTimeoutError, ExecutionError):
            self.assertTrue(issubclass(cls, ToolError))


class MainLoopErrorSemanticsTest(unittest.TestCase):
    """主循环分级退出：可恢复回灌继续、不可恢复 TOOL_ERROR 终止。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_recoverable_validation_error_continues(self):
        """ValidationError（未知工具）→ 回灌继续，模型修正后 final。"""
        tools = ToolExecutor(self.root)
        llm = FakeLLM(
            [
                LLMResponse(text="", tool_calls=[tc("no_such_tool", {}, "call_1")]),
                "修正后完成",
            ]
        )
        loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "修正后完成")
        # 可恢复错误以观察文本回灌给模型（tool 消息）。
        second_turn = llm.calls[1]
        tool_msg = [m for m in second_turn if m.role == "tool"][-1]
        self.assertIn("ToolError[ValidationError]", tool_msg.content)

    def test_recoverable_not_found_continues(self):
        """NotFoundError（读不存在文件）→ 回灌继续。"""
        tools = ToolExecutor(self.root)
        llm = FakeLLM(
            [
                LLMResponse(text="", tool_calls=[tc("read_file", {"path": "nope.txt"}, "call_1")]),
                "换了个文件完成了",
            ]
        )
        loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="读文件"))
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        second_turn = llm.calls[1]
        tool_msg = [m for m in second_turn if m.role == "tool"][-1]
        self.assertIn("ToolError[NotFoundError]", tool_msg.content)

    def test_fatal_permission_error_terminates(self):
        """ToolPermissionError（路径越界）→ 终止 TOOL_ERROR。"""
        tools = ToolExecutor(self.root)
        llm = FakeLLM(
            [
                LLMResponse(text="", tool_calls=[tc("read_file", {"path": "../outside.txt"}, "call_1")]),
                "不应被用到",
            ]
        )
        loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="读文件"))

        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
        self.assertIsNone(resp.final_answer)
        self.assertIn("ToolError[PermissionError]", resp.error)
        self.assertEqual(len(llm.calls), 1)  # 未进入下一轮
        # 错误 step 已记录。
        self.assertEqual(len(resp.steps), 1)
        self.assertIn("ToolError[PermissionError]", resp.steps[0].observations[0])

    def test_fatal_execution_error_terminates(self):
        """ExecutionError（工具抛未知异常）→ 终止 TOOL_ERROR。"""
        from myagent.tools.registry import register_tool

        @register_tool("_boom_tool")
        def _boom(args, cwd):
            raise RuntimeError("爆炸")

        try:
            tools = ToolExecutor(self.root)
            llm = FakeLLM(
                [
                    LLMResponse(text="", tool_calls=[tc("_boom_tool", {}, "call_1")]),
                    "不应被用到",
                ]
            )
            loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
            resp = loop.run(AgentRequest(user_input="任务"))
            self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
            self.assertIn("ToolError[ExecutionError]", resp.error)
        finally:
            from myagent.tools import TOOLS

            del TOOLS["_boom_tool"]

    def test_timeout_error_terminates(self):
        """TimeoutError → 终止 TOOL_ERROR。"""
        from myagent.tools.registry import register_tool

        @register_tool("_slow_tool")
        def _slow(args, cwd):
            import time

            time.sleep(5)
            return "太慢"

        try:
            tools = ToolExecutor(self.root, timeout=0.1)
            llm = FakeLLM(
                [
                    LLMResponse(text="", tool_calls=[tc("_slow_tool", {}, "call_1")]),
                    "不应被用到",
                ]
            )
            loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
            resp = loop.run(AgentRequest(user_input="任务"))
            self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
            self.assertIn("ToolError[TimeoutError]", resp.error)
        finally:
            from myagent.tools import TOOLS

            del TOOLS["_slow_tool"]


class SessionRecordingTest(unittest.TestCase):
    """会话记录：TOOL_ERROR step 正常进 REPL history / memory 存档。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_fatal_step_rebuilds_assistant_tool_pair(self):
        """TOOL_ERROR step → step_to_messages 输出 assistant + tool(错误) 配对。"""
        tools = ToolExecutor(self.root)
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[tc("read_file", {"path": "../x"}, "call_9")],
                ),
                "x",
            ]
        )
        loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="任务"))
        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)

        # 与 REPL 历史重建同一来源：assistant(tool_calls) + tool(错误)。
        rebuilt = step_to_messages(resp.steps[0])
        self.assertEqual(rebuilt[0].role, "assistant")
        self.assertEqual(rebuilt[0].tool_calls[0]["id"], "call_9")
        self.assertEqual(rebuilt[1].role, "tool")
        self.assertEqual(rebuilt[1].tool_call_id, "call_9")
        self.assertIn("ToolError[PermissionError]", rebuilt[1].content)

    def test_fatal_step_archived_to_memory_jsonl(self):
        """TOOL_ERROR 的 assistant/tool 消息正常写入 memory jsonl 存档。"""
        memory_dir = self.root / "memories"
        tools = ToolExecutor(self.root)
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[tc("read_file", {"path": "../x"}, "call_5")],
                ),
                "x",
            ]
        )
        loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="任务"))
        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)

        from myagent.memory import MemoryManager

        mgr = MemoryManager(memory_dir=memory_dir, llm=None, session_id="session_err")
        for message in step_to_messages(resp.steps[0]):
            mgr.append_message(message)
        records = read_archive(memory_dir, "session_err")
        roles = [r["role"] for r in records]
        self.assertEqual(roles, ["assistant", "tool"])
        self.assertIn("ToolError[PermissionError]", records[1]["content"])

    def test_recoverable_error_archived(self):
        """可恢复错误消息同样进 jsonl（assistant + tool 配对）。"""
        memory_dir = self.root / "memories"
        tools = ToolExecutor(self.root)
        llm = FakeLLM(
            [
                LLMResponse(text="", tool_calls=[tc("no_such_tool", {}, "call_7")]),
                "完成",
            ]
        )
        loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="任务"))
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)

        from myagent.memory import MemoryManager

        mgr = MemoryManager(memory_dir=memory_dir, llm=None, session_id="session_rec")
        for step in resp.steps:
            for message in step_to_messages(step):
                mgr.append_message(message)
        records = read_archive(memory_dir, "session_rec")
        self.assertEqual(records[1]["role"], "tool")
        self.assertIn("ToolError[ValidationError]", records[1]["content"])


class ComposerErrorCompatTest(unittest.TestCase):
    """composer 与错误 tool 消息：正常配对、不被误删。"""

    def test_error_tool_message_pairs_in_composer(self):
        from myagent.context import ContextComposer

        assistant = Message(
            role="assistant",
            content="",
            tool_calls=[tc("read_file", {"path": "../x"}, "call_err")],
        )
        error_tool = Message(
            role="tool",
            content="ToolError[PermissionError]: 路径越界",
            tool_call_id="call_err",
        )
        c = ContextComposer(max_input_tokens=10000, max_output_tokens=500)
        out = c.recompress([assistant, error_tool, Message(role="user", content="u")])
        groups = c._collect_tool_pairs(out)
        # 错误 tool 消息作为一组被正确收集，包含声明与结果。
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        # 错误消息未被单独丢弃。
        tool_msgs = [m for m in out if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("PermissionError", tool_msgs[0].content)


class MixedBatchPartialTest(unittest.TestCase):
    """混合批 [成功, 致命错误, 未执行]：partial 标记 + 无孤儿声明。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_mixed_batch_marks_partial_and_skipped(self):
        """[write_file(成功), delete_file(权限错), 未执行] → 3 outcomes。"""
        from myagent.tools.registry import register_tool

        # 注册一个会抛 PermissionError 的工具，插入批中间。
        @register_tool("_fatal_tool")
        def _fatal(args, cwd):
            from myagent.errors import ToolPermissionError

            raise ToolPermissionError("越界")

        try:
            tools = ToolExecutor(self.root)
            llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[
                            tc("write_file", {"path": "a.txt", "content": "x"}, "call_1"),
                            tc("_fatal_tool", {}, "call_2"),
                            tc("write_file", {"path": "b.txt", "content": "y"}, "call_3"),
                        ],
                    ),
                    "不应被用到",
                ]
            )
            loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
            resp = loop.run(AgentRequest(user_input="任务"))

            self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
            step = resp.steps[0]
            outcomes = step.outcomes
            self.assertEqual(len(outcomes), 3)
            # 1) 成功 → partial（批被后续致命错误中断）
            self.assertEqual(outcomes[0].status, "success")
            self.assertTrue(outcomes[0].partial)
            # 2) 致命错误 → error，带类型
            self.assertEqual(outcomes[1].status, "error")
            self.assertEqual(outcomes[1].error_type, "PermissionError")
            # 3) 未执行 → skipped
            self.assertEqual(outcomes[2].status, "skipped")
            self.assertIn("未执行", outcomes[2].observation)
            # 文件 a.txt 已成功写入（副作用保留）。
            self.assertTrue((self.root / "a.txt").is_file())
            self.assertFalse((self.root / "b.txt").exists())
        finally:
            from myagent.tools import TOOLS

            del TOOLS["_fatal_tool"]

    def test_no_orphan_tool_calls_in_rebuilt(self):
        """混合批重建消息：每个声明都有配对 tool 消息，无孤儿。"""
        from myagent.tools.registry import register_tool

        @register_tool("_fatal_tool")
        def _fatal(args, cwd):
            from myagent.errors import ToolPermissionError

            raise ToolPermissionError("越界")

        try:
            tools = ToolExecutor(self.root)
            llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[
                            tc("write_file", {"path": "a.txt", "content": "x"}, "call_1"),
                            tc("_fatal_tool", {}, "call_2"),
                            tc("write_file", {"path": "b.txt", "content": "y"}, "call_3"),
                        ],
                    ),
                    "x",
                ]
            )
            loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
            resp = loop.run(AgentRequest(user_input="任务"))
            self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)

            rebuilt = step_to_messages(resp.steps[0])
            # assistant 声明 3 个，tool 消息必须也是 3 条（配对无孤儿）。
            assistant = rebuilt[0]
            self.assertEqual(len(assistant.tool_calls), 3)
            tool_msgs = [m for m in rebuilt if m.role == "tool"]
            self.assertEqual(len(tool_msgs), 3)
            self.assertEqual(
                [m.tool_call_id for m in tool_msgs],
                ["call_1", "call_2", "call_3"],
            )
            # 成功/部分执行消息带 partial 元数据；错误带 error_type。
            self.assertTrue(tool_msgs[0].metadata.get("partial"))
            self.assertEqual(tool_msgs[1].metadata.get("error_type"), "PermissionError")
            self.assertIn("未执行", tool_msgs[2].content)
        finally:
            from myagent.tools import TOOLS

            del TOOLS["_fatal_tool"]

    def test_partial_flag_in_memory_archive(self):
        """partial / error_type 结构化落盘 jsonl。"""
        from myagent.tools.registry import register_tool

        @register_tool("_fatal_tool")
        def _fatal(args, cwd):
            from myagent.errors import ToolPermissionError

            raise ToolPermissionError("越界")

        try:
            memory_dir = self.root / "memories"
            tools = ToolExecutor(self.root)
            llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[
                            tc("write_file", {"path": "a.txt", "content": "x"}, "call_1"),
                            tc("_fatal_tool", {}, "call_2"),
                        ],
                    ),
                    "x",
                ]
            )
            loop = AgentLoop(AgentParams(cwd=str(self.root)), llm=llm, tools=tools)
            resp = loop.run(AgentRequest(user_input="任务"))
            self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)

            from myagent.memory import MemoryManager

            mgr = MemoryManager(memory_dir=memory_dir, llm=None, session_id="session_mix")
            for message in step_to_messages(resp.steps[0]):
                mgr.append_message(message)
            records = read_archive(memory_dir, "session_mix")
            tool_records = [r for r in records if r["role"] == "tool"]
            # 第一条（partial 成功）带 partial；第二条（致命）带 error_type。
            self.assertTrue(tool_records[0].get("partial"))
            self.assertEqual(tool_records[1].get("error_type"), "PermissionError")
        finally:
            from myagent.tools import TOOLS

            del TOOLS["_fatal_tool"]


class ErrorRenderingTest(unittest.TestCase):
    """summary 渲染与 prompt：错误重点记录。"""

    def test_render_transcript_error_and_partial(self):
        from myagent.memory.summarizer import render_transcript

        records = [
            {"role": "tool", "content": "ToolError[NotFoundError]: 文件不存在：a.py", "error_type": "NotFoundError"},
            {"role": "tool", "content": "已写入 a.txt", "partial": True},
            {"role": "tool", "content": "ToolError[ApprovalDenied]: 用户拒绝", "error_type": "ApprovalDenied"},
            {"role": "tool", "content": "普通结果"},
        ]
        text = render_transcript(records)
        self.assertIn("→ 工具错误[NotFoundError]：", text)
        self.assertIn("→ 工具结果（部分执行，任务中断）：已写入 a.txt", text)
        self.assertIn("→ 工具错误[ApprovalDenied]：", text)
        self.assertIn("→ 工具结果：普通结果", text)

    def test_extract_prompt_emphasizes_errors(self):
        """summary_extract_prompt.md 必须包含错误重点规则。"""
        from myagent.agent_config import PROMPT_DIR

        prompt = (PROMPT_DIR / "summary_extract_prompt.md").read_text(encoding="utf-8")
        self.assertIn("工具错误必须重点记录", prompt)
        self.assertIn("部分执行 / 任务中断必须记录", prompt)

    def test_aggregate_prompt_prioritizes_errors(self):
        from myagent.agent_config import PROMPT_DIR

        prompt = (PROMPT_DIR / "summary_aggregate_prompt.md").read_text(encoding="utf-8")
        self.assertIn("错误经验优先保留", prompt)


if __name__ == "__main__":
    unittest.main()
