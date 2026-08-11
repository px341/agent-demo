"""AgentLoop 主循环的契约测试。

不依赖真实 LLM：注入 FakeLLM / FakeTools 验证主循环各分支
（final / tool_call / retry / max_turns / 无工具 / LLM 异常 / 历史保留）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.actions import FinalAnswer, Retry, ToolCall
from myagent.agent_config import AgentParams, Message
from myagent.agent_loop import AgentLoop, step_to_messages
from myagent.contracts import AgentRequest, LLMResponse, StopReason


class FakeLLM:
    """按脚本序列应答，并记录每次收到的消息快照。

    接受 str（自动包装为 LLMResponse）或显式 LLMResponse。
    """

    def __init__(self, outputs: list[str | LLMResponse], fallback: str | LLMResponse | None = None):
        self.outputs = list(outputs)
        self.fallback = fallback
        self.calls: list[tuple[list[Message], int | None]] = []

    @staticmethod
    def _wrap(output: str | LLMResponse) -> LLMResponse:
        return output if isinstance(output, LLMResponse) else LLMResponse(text=output)

    def complete(self, messages, *, max_new_tokens=None):
        self.calls.append((list(messages), max_new_tokens))
        if self.outputs:
            return self._wrap(self.outputs.pop(0))
        if self.fallback is not None:
            return self._wrap(self.fallback)
        raise AssertionError("FakeLLM 输出脚本耗尽")


class FakeTools:
    """记录被调用的工具与参数。"""

    def __init__(self, result: str = "观察结果"):
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return self.result


def make_loop(
    llm,
    tools=None,
    approval_gate=None,
    **params_overrides,
) -> AgentLoop:
    params = AgentParams(**params_overrides)
    return AgentLoop(
        params,
        llm=llm,
        tools=tools,
        approval_gate=approval_gate,
    )


class AgentLoopTest(unittest.TestCase):
    def test_final_answer_direct(self):
        llm = FakeLLM(['{"action": "final", "answer": "42"}'])
        loop = make_loop(llm)
        resp = loop.run(AgentRequest(user_input="1+1=?"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "42")
        self.assertEqual(resp.turns_used, 1)
        self.assertEqual(resp.tool_calls, 0)
        self.assertEqual(len(resp.steps), 1)
        self.assertIsInstance(resp.steps[0].action, FinalAnswer)

        # 首轮消息 = system + user，且 max_new_tokens 来自 AgentParams。
        msgs, max_new_tokens = llm.calls[0]
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[-1].role, "user")
        self.assertEqual(msgs[-1].content, "1+1=?")
        self.assertEqual(max_new_tokens, AgentParams().max_output_tokens)

    def test_tool_call_then_final(self):
        tools = FakeTools("文件内容")
        llm = FakeLLM(
            [
                '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                '{"action": "final", "answer": "读完了"}',
            ]
        )
        loop = make_loop(llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="读 a.py"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "读完了")
        self.assertEqual(resp.turns_used, 2)
        self.assertEqual(resp.tool_calls, 1)
        self.assertEqual(tools.calls, [("read_file", {"path": "a.py"})])

        # 第二步模型应看到 assistant（携带 tool_calls 声明）+ tool 观察结果，
        # 二者通过 tool_call_id=call_1 配对（OpenAI 兼容 API 的硬性要求）。
        msgs = llm.calls[1][0]
        self.assertEqual(msgs[-2].role, "assistant")
        self.assertEqual(msgs[-1].role, "tool")
        self.assertEqual(msgs[-1].content, "文件内容")
        self.assertEqual(msgs[-1].tool_call_id, "call_1")
        self.assertEqual(
            msgs[-2].tool_calls,
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.py"}',
                    },
                }
            ],
        )

        # 轨迹里记录了解析出的 ToolCall 与观察。
        step = resp.steps[0]
        self.assertIsInstance(step.action, ToolCall)
        self.assertEqual(step.observation, "文件内容")

    def test_retry_recovers(self):
        llm = FakeLLM(
            [
                "这不是合法 JSON",
                '{"action": "final", "answer": "重试成功"}',
            ]
        )
        loop = make_loop(llm)
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "重试成功")
        self.assertEqual(resp.turns_used, 2)
        self.assertEqual(resp.tool_calls, 0)
        self.assertIsInstance(resp.steps[0].action, Retry)

        # 重试原因以 role=user 回灌（不伪造工具调用声明）。
        msgs = llm.calls[1][0]
        self.assertEqual(msgs[-1].role, "user")
        self.assertIn("输出不符合契约", msgs[-1].content)

    def test_max_turns_stop(self):
        tool_call = (
            '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}'
        )
        llm = FakeLLM([tool_call], fallback=tool_call)
        loop = make_loop(llm, tools=FakeTools())
        resp = loop.run(AgentRequest(user_input="任务", max_turns=3))

        self.assertEqual(resp.stop_reason, StopReason.MAX_TURNS)
        self.assertIsNone(resp.final_answer)
        self.assertEqual(resp.turns_used, 3)
        self.assertEqual(resp.tool_calls, 3)
        self.assertEqual(len(resp.steps), 3)

    def test_no_tools_tool_call_observation(self):
        llm = FakeLLM(
            [
                '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                '{"action": "final", "answer": "没有工具也能答"}',
            ]
        )
        loop = make_loop(llm, tools=None)
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.tool_calls, 0)
        msgs = llm.calls[1][0]
        self.assertEqual(msgs[-1].role, "tool")
        self.assertIn("未启用工具", msgs[-1].content)

    def test_llm_error(self):
        class BoomLLM:
            def complete(self, messages, *, max_new_tokens=None):
                raise RuntimeError("boom")

        loop = make_loop(BoomLLM())
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.ERROR)
        self.assertIsNone(resp.final_answer)
        self.assertEqual(resp.error, "boom")
        self.assertEqual(resp.turns_used, 1)

    def test_max_turns_clamped_to_one(self):
        """max_turns <= 0 时按 1 轮收口，保证 turns_used >= 1 的返回值契约。"""
        llm = FakeLLM(['{"action": "final", "answer": "一轮完成"}'])
        loop = make_loop(llm)
        resp = loop.run(AgentRequest(user_input="任务", max_turns=0))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.turns_used, 1)

    def test_reasoning_content_carried_to_next_turn(self):
        """DeepSeek thinking mode：assistant 的 reasoning_content 必须随消息回传。"""
        llm = FakeLLM(
            [
                LLMResponse(
                    text='{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                    metadata={"reasoning_content": "思考过程"},
                ),
                '{"action": "final", "answer": "完成"}',
            ]
        )
        loop = make_loop(llm, tools=FakeTools("观察"))
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        # 第二步消息里 assistant 消息应携带 reasoning_content 元数据。
        msgs = llm.calls[1][0]
        self.assertEqual(msgs[-2].metadata, {"reasoning_content": "思考过程"})
        # 轨迹重建路径（cli 历史）同样保留元数据。
        rebuilt = step_to_messages(resp.steps[0])
        self.assertEqual(rebuilt[0].metadata, {"reasoning_content": "思考过程"})

    def test_thought_field_ignored_by_parser(self):
        """prompt 契约含 thought 字段：parse_action 忽略多余字段，主循环正常。"""
        llm = FakeLLM(
            [
                '{"thought": "先看目录结构", "action": "tool_call", "tool": "list_files", "args": {"path": "."}}',
                '{"thought": "信息充足", "action": "final", "answer": "完成"}',
            ]
        )
        loop = make_loop(llm, tools=FakeTools("目录内容"))
        resp = loop.run(AgentRequest(user_input="看看目录"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "完成")
        self.assertEqual(resp.tool_calls, 1)
        # 解析出的 Action 只含契约字段；thought 保留在原始输出中。
        self.assertEqual(resp.steps[0].action.args, {"path": "."})
        self.assertIn("thought", resp.steps[0].raw_output)

    def test_approval_gate_denies_tool_call(self):
        """审批拒绝：不执行工具、不计 tool_calls、观察提示用户拒绝、模型继续。"""
        tools = FakeTools("文件内容")

        class Gate:
            def __init__(self, calls):
                self.calls = calls

            def request(self, name, args):
                self.calls.append((name, args))
                return False

        gate = Gate([])
        llm = FakeLLM(
            [
                '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                '{"action": "final", "answer": "被拒后直接回答"}',
            ]
        )
        loop = make_loop(llm, tools=tools, approval_gate=gate)
        resp = loop.run(AgentRequest(user_input="读文件"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "被拒后直接回答")
        self.assertEqual(resp.tool_calls, 0)
        # 工具未执行，闸门被询问且收到拒绝。
        self.assertEqual(tools.calls, [])
        self.assertEqual(gate.calls, [("read_file", {"path": "a.py"})])
        # 观察文本提示用户拒绝，模型能看到。
        self.assertIn("用户拒绝", resp.steps[0].observation)

    def test_approval_gate_allows_tool_call(self):
        """审批允许：正常执行、计入 tool_calls。"""

        class Gate:
            def request(self, name, args):
                return True

        tools = FakeTools("文件内容")
        llm = FakeLLM(
            [
                '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                '{"action": "final", "answer": "读完"}',
            ]
        )
        loop = make_loop(llm, tools=tools, approval_gate=Gate())
        resp = loop.run(AgentRequest(user_input="读文件"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.tool_calls, 1)
        self.assertEqual(tools.calls, [("read_file", {"path": "a.py"})])

    def test_no_gate_executes_without_asking(self):
        """approval_gate=None（默认）：不询问直接执行（既有行为）。"""
        tools = FakeTools("文件内容")
        llm = FakeLLM(
            [
                '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                '{"action": "final", "answer": "读完"}',
            ]
        )
        loop = make_loop(llm, tools=tools)  # 不传 approval_gate
        resp = loop.run(AgentRequest(user_input="读文件"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.tool_calls, 1)
        self.assertEqual(tools.calls, [("read_file", {"path": "a.py"})])

    def test_history_preserved_and_untouched(self):
        history = [Message(role="user", content="旧消息")]
        llm = FakeLLM(['{"action": "final", "answer": "完成"}'])
        loop = make_loop(llm)
        resp = loop.run(AgentRequest(user_input="新消息", messages=history))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        # 调用方传入的历史未被修改。
        self.assertEqual(history, [Message(role="user", content="旧消息")])
        # 模型看到：system + 历史 + 当前请求。
        msgs = llm.calls[0][0]
        self.assertEqual([m.role for m in msgs], ["system", "user", "user"])
        self.assertEqual(msgs[1].content, "旧消息")
        self.assertEqual(msgs[2].content, "新消息")

    def test_non_object_json_retry(self):
        """模型输出合法 JSON 但不是对象（字符串/数组）时，应走 Retry 而非崩溃。"""
        llm = FakeLLM(
            [
                '"我不是对象"',
                '{"action": "final", "answer": "恢复了"}',
            ]
        )
        loop = make_loop(llm)
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "恢复了")
        self.assertEqual(resp.turns_used, 2)
        self.assertIsInstance(resp.steps[0].action, Retry)

    def test_request_max_turns_overrides_params(self):
        params = AgentParams(max_turns=100)
        tool_call = (
            '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}'
        )
        llm = FakeLLM([tool_call], fallback=tool_call)
        loop = AgentLoop(params, llm=llm, tools=FakeTools())
        resp = loop.run(AgentRequest(user_input="任务", max_turns=2))

        self.assertEqual(resp.stop_reason, StopReason.MAX_TURNS)
        self.assertEqual(resp.turns_used, 2)


if __name__ == "__main__":
    unittest.main()
