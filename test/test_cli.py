"""cli 接线测试：REPL 的流程与跨轮历史重建。

用 FakeLoop 替代真实 AgentLoop（不触网），验证：
- REPL 每轮调用 loop.run(AgentRequest)，并维护跨轮历史；
- /exit /quit 退出；空输入提示后继续；
- _trim_history 配对安全裁剪；
- _print_response 按结束原因返回 0/1。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import Message
from myagent.agent_loop import step_to_messages
from myagent.contracts import (
    AgentRequest,
    AgentResponse,
    StepRecord,
    StopReason,
)
from myagent.cli import _one_shot, _print_response, _repl, _trim_history


class FakeLoop:
    """记录每次 run 的请求快照，按序返回预设响应。

    与主循环契约一致：记录时复制 messages 快照，
    不持有调用方后续会继续修改的列表引用。
    """

    def __init__(self, responses: list[AgentResponse]):
        self.responses = list(responses)
        self.requests: list[AgentRequest] = []

    def run(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(
            AgentRequest(
                user_input=request.user_input,
                messages=list(request.messages) if request.messages else None,
                max_turns=request.max_turns,
            )
        )
        return self.responses.pop(0)


def final_response(
    answer: str = "好",
    steps: list[StepRecord] | None = None,
) -> AgentResponse:
    return AgentResponse(
        final_answer=answer,
        stop_reason=StopReason.FINAL_ANSWER,
        turns_used=1,
        tool_calls=0,
        steps=steps or [],
    )


class CliReplTest(unittest.TestCase):
    def test_one_shot_runs_once_without_input(self):
        loop = FakeLoop([final_response("完成")])
        with mock.patch("builtins.input") as prompt:
            code = _one_shot(loop, "修复问题")
        self.assertEqual(code, 0)
        prompt.assert_not_called()
        self.assertEqual(len(loop.requests), 1)
        self.assertEqual(loop.requests[0].user_input, "修复问题")

    def test_one_shot_rejects_empty_task(self):
        loop = FakeLoop([])
        self.assertEqual(_one_shot(loop, "  "), 2)
        self.assertEqual(loop.requests, [])

    def test_exit_command(self):
        loop = FakeLoop([])
        with mock.patch("builtins.input", side_effect=["/exit"]):
            code = _repl(loop)
        self.assertEqual(code, 0)
        self.assertEqual(loop.requests, [])

    def test_eof_exits_cleanly(self):
        """Ctrl-D（EOFError）应正常退出而非 traceback。"""
        loop = FakeLoop([])
        with mock.patch("builtins.input", side_effect=EOFError):
            code = _repl(loop)
        self.assertEqual(code, 0)
        self.assertEqual(loop.requests, [])

    def test_quit_command(self):
        loop = FakeLoop([])
        with mock.patch("builtins.input", side_effect=["/quit"]):
            code = _repl(loop)
        self.assertEqual(code, 0)

    def test_empty_input_continues(self):
        loop = FakeLoop([final_response()])
        with mock.patch("builtins.input", side_effect=["", "你好", "/exit"]):
            code = _repl(loop)
        self.assertEqual(code, 0)
        # 空输入不触发 run，只有"你好"一次。
        self.assertEqual(len(loop.requests), 1)
        self.assertEqual(loop.requests[0].user_input, "你好")
        self.assertIsNone(loop.requests[0].messages)

    def test_history_carried_to_next_turn(self):
        step = StepRecord(
            turn=1,
            raw_output="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.py"}',
                    },
                }
            ],
            observations=["观察结果"],
            assistant_metadata={"reasoning_content": "思考过程"},
        )
        loop = FakeLoop([final_response("第一轮完成", [step]), final_response("第二轮完成")])
        with mock.patch("builtins.input", side_effect=["读 a.py", "然后呢", "/exit"]):
            code = _repl(loop)
        self.assertEqual(code, 0)
        self.assertEqual(len(loop.requests), 2)

        # 第二轮请求应携带第一轮重建的历史：user + assistant + tool。
        history = loop.requests[1].messages
        self.assertIsNotNone(history)
        roles = [m.role for m in history]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(history[0].content, "读 a.py")
        self.assertEqual(history[1].content, step.raw_output)
        self.assertEqual(history[2].content, "观察结果")
        self.assertEqual(history[2].tool_call_id, "call_1")
        # assistant 必须携带与 tool 消息配对的 tool_calls 声明（API 硬性要求）。
        self.assertEqual(history[1].tool_calls[0]["id"], "call_1")
        self.assertEqual(
            history[1].tool_calls[0]["function"]["name"],
            "read_file",
        )
        # reasoning_content 元数据随历史重建保留。
        self.assertEqual(history[1].metadata, {"reasoning_content": "思考过程"})

    def test_retry_step_rebuilt_as_user_feedback(self):
        # 纯文本 assistant 轮（非工具轮）在历史里只重建一条 assistant 消息。
        plain_step = StepRecord(
            turn=1,
            raw_output="中间说明",
            final_answer=None,
        )
        loop = FakeLoop(
            [final_response("第一轮完成", [plain_step]), final_response("第二轮完成")]
        )
        with mock.patch("builtins.input", side_effect=["任务", "继续", "/exit"]):
            code = _repl(loop)
        self.assertEqual(code, 0)

        history = loop.requests[1].messages
        roles = [m.role for m in history]
        # 纯文本 assistant 轮只有一条 assistant 消息，无 tool 配对。
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(history[1].content, "中间说明")
        self.assertIsNone(history[1].tool_call_id)
        self.assertEqual(history[1].tool_calls, [])

    def test_print_response_exit_codes(self):
        self.assertEqual(_print_response(final_response("好")), 0)
        self.assertEqual(
            _print_response(
                AgentResponse(
                    final_answer=None,
                    stop_reason=StopReason.MAX_TURNS,
                    turns_used=5,
                    tool_calls=0,
                )
            ),
            1,
        )
        self.assertEqual(
            _print_response(
                AgentResponse(
                    final_answer=None,
                    stop_reason=StopReason.ERROR,
                    turns_used=1,
                    tool_calls=0,
                    error="boom",
                )
            ),
            1,
        )


class TrimHistoryTest(unittest.TestCase):
    """_trim_history 配对安全裁剪。"""

    def test_within_limit_unchanged(self):
        history = [Message(role="user", content="a"), Message(role="assistant", content="b")]
        self.assertEqual(_trim_history(history, limit=10), history)

    def test_trims_oldest_when_over_limit(self):
        history = [Message(role="user", content=f"m{i}") for i in range(5)]
        kept = _trim_history(history, limit=3)
        self.assertEqual([m.content for m in kept], ["m2", "m3", "m4"])

    def test_trims_leading_orphan_tool(self):
        # 开头残留孤儿 tool 结果（无配对声明）→ 被清理。
        history = [
            Message(role="tool", content="孤儿结果"),
            Message(role="user", content="m0"),
            Message(role="user", content="m1"),
            Message(role="user", content="m2"),
        ]
        kept = _trim_history(history, limit=3)
        self.assertNotEqual(kept[0].role, "tool")
        self.assertEqual([m.content for m in kept], ["m0", "m1", "m2"])

    def test_trims_trailing_orphan_tool_calls(self):
        # 结尾残留孤儿 tool_calls 声明（配对结果被截掉）→ 删除。
        # 保留最近 limit 条 = [m1, assistant]，孤儿声明删除后只剩 m1
        # （正确性优先于数量，允许裁剪后少于 limit）。
        history = [
            Message(role="user", content="m0"),
            Message(role="user", content="m1"),
            Message(
                role="assistant",
                content='{"action": "tool_call"}',
                tool_calls=[{"id": "call_9", "type": "function", "function": {}}],
            ),
        ]
        kept = _trim_history(history, limit=2)
        self.assertEqual([m.role for m in kept], ["user"])
        self.assertEqual(kept[0].content, "m1")

    def test_paired_tool_kept_intact(self):
        # 完整配对的 assistant 声明 + tool 结果保留。
        history = [
            Message(role="user", content="m0"),
            Message(
                role="assistant",
                content='{"action": "tool_call"}',
                tool_calls=[{"id": "call_1", "type": "function", "function": {}}],
            ),
            Message(role="tool", content="结果", tool_call_id="call_1"),
        ]
        kept = _trim_history(history, limit=3)
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[2].tool_call_id, "call_1")


if __name__ == "__main__":
    unittest.main()
