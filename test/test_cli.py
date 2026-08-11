"""cli 接线测试：REPL / one_shot 的流程与跨轮历史重建。

用 FakeLoop 替代真实 AgentLoop（不触网），验证：
- REPL 每轮调用 loop.run(AgentRequest)，并维护跨轮历史；
- /exit /quit 退出；空输入提示后继续；
- one_shot 只跑一次并正确返回退出码；
- _print_response 按结束原因返回 0/1。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.actions import FinalAnswer, Retry, ToolCall
from myagent.contracts import AgentRequest, AgentResponse, StepRecord, StopReason
from myagent.cli import _one_shot, _print_response, _repl


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
            raw_output='{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
            action=ToolCall(
                name="read_file",
                args={"path": "a.py"},
                raw='{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
            ),
            observation="观察结果",
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
        retry_step = StepRecord(
            turn=1,
            raw_output="不是 JSON",
            action=Retry(reason="不是合法 JSON"),
            observation="输出不符合契约，请重新输出合法 JSON。原因：不是合法 JSON",
        )
        loop = FakeLoop(
            [final_response("第一轮完成", [retry_step]), final_response("第二轮完成")]
        )
        with mock.patch("builtins.input", side_effect=["任务", "继续", "/exit"]):
            code = _repl(loop)
        self.assertEqual(code, 0)

        history = loop.requests[1].messages
        roles = [m.role for m in history]
        # Retry 反馈不伪造工具调用，以 user 角色回灌。
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(history[2].content, retry_step.observation)
        self.assertIsNone(history[2].tool_call_id)

    def test_one_shot(self):
        loop = FakeLoop([final_response("答案")])
        with mock.patch("builtins.input", return_value="任务描述"):
            code = _one_shot(loop)
        self.assertEqual(code, 0)
        self.assertEqual(loop.requests[0].user_input, "任务描述")
        self.assertIsNone(loop.requests[0].messages)

    def test_one_shot_empty_input(self):
        loop = FakeLoop([])
        with mock.patch("builtins.input", return_value="  "):
            code = _one_shot(loop)
        self.assertEqual(code, 1)
        self.assertEqual(loop.requests, [])

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


if __name__ == "__main__":
    unittest.main()
