"""AgentLoop 主循环的契约测试（OpenAI 原生 tool_calls 协议）。

不依赖真实 LLM：注入 FakeLLM / FakeTools 验证主循环各分支
（final / 单工具 / 并行多工具 / 无工具 / 审批 / max_turns / LLM 异常 /
历史保留 / reasoning_content 透传）。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams, Message
from myagent.agent_loop import AgentLoop, step_to_messages
from myagent.contracts import AgentRequest, AgentResponse, LLMResponse, StopReason


def tc(name: str, args: dict, call_id: str = "call_1") -> dict:
    """构造 OpenAI 原生 tool_calls 单条。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


class FakeLLM:
    """按脚本序列应答，并记录每次收到的消息快照。

    接受 str（自动包装为 LLMResponse）或显式 LLMResponse。
    """

    def __init__(self, outputs: list[str | LLMResponse], fallback: str | LLMResponse | None = None):
        self.outputs = list(outputs)
        self.fallback = fallback
        self.calls: list[tuple[list[Message], list[dict], int | None]] = []

    @staticmethod
    def _wrap(output: str | LLMResponse) -> LLMResponse:
        return output if isinstance(output, LLMResponse) else LLMResponse(text=output)

    def complete(self, messages, *, tools=None, max_new_tokens=None):
        self.calls.append((list(messages), list(tools or []), max_new_tokens))
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
        llm = FakeLLM(["42"])
        loop = make_loop(llm)
        resp = loop.run(AgentRequest(user_input="1+1=?"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "42")
        self.assertEqual(resp.turns_used, 1)
        self.assertEqual(resp.tool_calls, 0)
        self.assertEqual(len(resp.steps), 1)
        self.assertEqual(resp.steps[0].final_answer, "42")
        self.assertEqual(resp.steps[0].tool_calls, [])

        # 首轮消息 = system + user，且 max_new_tokens 来自 AgentParams。
        msgs, _, max_new_tokens = llm.calls[0]
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[-1].role, "user")
        self.assertEqual(msgs[-1].content, "1+1=?")
        self.assertEqual(max_new_tokens, AgentParams().max_output_tokens)

    def test_tool_call_then_final(self):
        tools = FakeTools("文件内容")
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[tc("read_file", {"path": "a.py"}, "call_x1")],
                ),
                "读完了",
            ]
        )
        loop = make_loop(llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="读 a.py"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "读完了")
        self.assertEqual(resp.turns_used, 2)
        self.assertEqual(resp.tool_calls, 1)
        self.assertEqual(tools.calls, [("read_file", {"path": "a.py"})])

        # 第二步模型应看到 assistant（携带原生 tool_calls 声明）+ tool 观察结果，
        # 二者通过模型返回的原生 id 配对。
        msgs = llm.calls[1][0]
        self.assertEqual(msgs[-2].role, "assistant")
        self.assertEqual(msgs[-1].role, "tool")
        self.assertEqual(msgs[-1].content, "文件内容")
        self.assertEqual(msgs[-1].tool_call_id, "call_x1")
        self.assertEqual(msgs[-2].tool_calls[0]["id"], "call_x1")
        self.assertEqual(msgs[-2].tool_calls[0]["function"]["name"], "read_file")

        # 轨迹记录工具调用与观察。
        step = resp.steps[0]
        self.assertEqual(step.tool_calls[0]["id"], "call_x1")
        self.assertEqual(step.observations, ["文件内容"])

    def test_parallel_tool_calls_executed(self):
        """原生 FC：一次响应多个 tool_calls → 全部执行（并行语义）。"""
        tools = FakeTools("结果")
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[
                        tc("read_file", {"path": "a.py"}, "call_a"),
                        tc("list_files", {"path": "."}, "call_b"),
                    ],
                ),
                "都看完了",
            ]
        )
        loop = make_loop(llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="看两个"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.tool_calls, 2)
        self.assertEqual(
            tools.calls,
            [("read_file", {"path": "a.py"}), ("list_files", {"path": "."})],
        )

        # 回灌消息：assistant 声明 + 两条 tool 结果，各按原生 id 配对。
        msgs = llm.calls[1][0]
        tool_msgs = [m for m in msgs if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        self.assertEqual(tool_msgs[0].tool_call_id, "call_a")
        self.assertEqual(tool_msgs[1].tool_call_id, "call_b")
        assistant = [m for m in msgs if m.role == "assistant" and m.tool_calls][-1]
        self.assertEqual(assistant.tool_calls[0]["id"], "call_a")

    def test_parallel_gate_batch_deny(self):
        """审批批量拒绝：两个调用都不执行、终止 TOOL_ERROR。"""
        tools = FakeTools("结果")

        class Gate:
            def __init__(self):
                self.calls = None

            def request_batch(self, calls):
                self.calls = list(calls)
                return False

        gate = Gate()
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[
                        tc("read_file", {"path": "a.py"}, "call_a"),
                        tc("list_files", {"path": "."}, "call_b"),
                    ],
                ),
                "被拒后直接回答",
            ]
        )
        loop = make_loop(llm, tools=tools, approval_gate=gate)
        resp = loop.run(AgentRequest(user_input="读"))

        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
        self.assertEqual(resp.tool_calls, 0)
        self.assertEqual(tools.calls, [])
        self.assertEqual(gate.calls, [("read_file", {"path": "a.py"}), ("list_files", {"path": "."})])
        # 两条观察都含 ApprovalDenied。
        self.assertTrue(all("ApprovalDenied" in obs for obs in resp.steps[0].observations))

    def test_parallel_gate_batch_allow(self):
        """审批批量允许：两个调用都执行。"""

        class Gate:
            def request_batch(self, calls):
                return True

        tools = FakeTools("结果")
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[
                        tc("read_file", {"path": "a.py"}, "call_a"),
                        tc("list_files", {"path": "."}, "call_b"),
                    ],
                ),
                "完成",
            ]
        )
        loop = make_loop(llm, tools=tools, approval_gate=Gate())
        resp = loop.run(AgentRequest(user_input="读"))
        self.assertEqual(resp.tool_calls, 2)
        self.assertEqual(len(tools.calls), 2)

    def test_max_turns_stop(self):
        tool_resp = LLMResponse(text="", tool_calls=[tc("read_file", {"path": "a.py"})])
        llm = FakeLLM([tool_resp], fallback=tool_resp)
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
                LLMResponse(text="", tool_calls=[tc("read_file", {"path": "a.py"})]),
                "没有工具也能答",
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
            def complete(self, messages, *, tools=None, max_new_tokens=None):
                raise RuntimeError("boom")

        loop = make_loop(BoomLLM())
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.ERROR)
        self.assertIsNone(resp.final_answer)
        self.assertEqual(resp.error, "boom")
        self.assertEqual(resp.turns_used, 1)

    def test_max_turns_clamped_to_one(self):
        """max_turns <= 0 时按 1 轮收口，保证 turns_used >= 1 的返回值契约。"""
        llm = FakeLLM(["一轮完成"])
        loop = make_loop(llm)
        resp = loop.run(AgentRequest(user_input="任务", max_turns=0))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.turns_used, 1)

    def test_reasoning_content_carried_to_next_turn(self):
        """DeepSeek thinking mode：assistant 的 reasoning_content 必须随消息回传。"""
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[tc("read_file", {"path": "a.py"})],
                    metadata={"reasoning_content": "思考过程"},
                ),
                "完成",
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

    def test_approval_gate_denies_tool_call(self):
        """审批拒绝（ApprovalDenied）：不执行工具、终止本轮 TOOL_ERROR。"""
        tools = FakeTools("文件内容")

        class Gate:
            def __init__(self, calls):
                self.calls = calls

            def request_batch(self, calls):
                self.calls.extend(calls)
                return False

        gate = Gate([])
        llm = FakeLLM(
            [
                LLMResponse(text="", tool_calls=[tc("read_file", {"path": "a.py"})]),
                "被拒后直接回答",  # 不应被用到（审批拒绝即终止）
            ]
        )
        loop = make_loop(llm, tools=tools, approval_gate=gate)
        resp = loop.run(AgentRequest(user_input="读文件"))

        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
        self.assertIsNone(resp.final_answer)
        self.assertIn("ApprovalDenied", resp.error)
        self.assertEqual(resp.tool_calls, 0)
        self.assertEqual(tools.calls, [])
        self.assertEqual(gate.calls, [("read_file", {"path": "a.py"})])
        # 错误 step 已记录（会话轨迹完整）。
        self.assertEqual(len(resp.steps), 1)
        self.assertIn("ApprovalDenied", resp.steps[0].observations[0])
        # 只用了一次 LLM 调用（未进入下一轮）。
        self.assertEqual(len(llm.calls), 1)

    def test_approval_gate_allows_tool_call(self):
        """审批允许：正常执行、计入 tool_calls。"""

        class Gate:
            def request_batch(self, calls):
                return True

        tools = FakeTools("文件内容")
        llm = FakeLLM(
            [
                LLMResponse(text="", tool_calls=[tc("read_file", {"path": "a.py"})]),
                "读完",
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
                LLMResponse(text="", tool_calls=[tc("read_file", {"path": "a.py"})]),
                "读完",
            ]
        )
        loop = make_loop(llm, tools=tools)  # 不传 approval_gate
        resp = loop.run(AgentRequest(user_input="读文件"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.tool_calls, 1)
        self.assertEqual(tools.calls, [("read_file", {"path": "a.py"})])

    def test_tools_passed_to_llm(self):
        """主循环每次调用都把注册表工具传给 LLM（tools 参数非空）。"""
        llm = FakeLLM(["42"])
        loop = make_loop(llm)
        loop.run(AgentRequest(user_input="任务"))
        _, tools, _ = llm.calls[0]
        names = [t["function"]["name"] for t in tools]
        self.assertIn("read_file", names)
        self.assertIn("list_files", names)
        for tool in tools:
            self.assertEqual(tool["type"], "function")

    def test_invalid_arguments_json_handled(self):
        """arguments 不是合法 JSON 时按空参数执行，不崩溃。"""
        tools = FakeTools("结果")
        llm = FakeLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=[
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "不是JSON"},
                        }
                    ],
                ),
                "完成",
            ]
        )
        loop = make_loop(llm, tools=tools)
        resp = loop.run(AgentRequest(user_input="任务"))
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(tools.calls, [("read_file", {})])

    def test_history_preserved_and_untouched(self):
        history = [Message(role="user", content="旧消息")]
        llm = FakeLLM(["完成"])
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

    def test_request_max_turns_overrides_params(self):
        params = AgentParams(max_turns=100)
        tool_resp = LLMResponse(text="", tool_calls=[tc("read_file", {"path": "a.py"})])
        llm = FakeLLM([tool_resp], fallback=tool_resp)
        loop = AgentLoop(params, llm=llm, tools=FakeTools())
        resp = loop.run(AgentRequest(user_input="任务", max_turns=2))

        self.assertEqual(resp.stop_reason, StopReason.MAX_TURNS)
        self.assertEqual(resp.turns_used, 2)


class StepToMessagesTest(unittest.TestCase):
    """step_to_messages：轨迹 → 消息（工具轮 / 答案轮）。"""

    def test_tool_step_pairs_with_native_ids(self):
        from myagent.contracts import StepRecord

        step = StepRecord(
            turn=1,
            raw_output="",
            tool_calls=[tc("read_file", {"path": "a.py"}, "call_9")],
            observations=["内容"],
        )
        msgs = step_to_messages(step)
        self.assertEqual(msgs[0].role, "assistant")
        self.assertEqual(msgs[0].tool_calls[0]["id"], "call_9")
        self.assertEqual(msgs[1].role, "tool")
        self.assertEqual(msgs[1].content, "内容")
        self.assertEqual(msgs[1].tool_call_id, "call_9")

    def test_final_step_single_assistant(self):
        from myagent.contracts import StepRecord

        step = StepRecord(turn=1, raw_output="答案", final_answer="答案")
        msgs = step_to_messages(step)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, "assistant")
        self.assertEqual(msgs[0].content, "答案")
        self.assertEqual(msgs[0].tool_calls, [])


if __name__ == "__main__":
    unittest.main()
