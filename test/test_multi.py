"""Multi-agent（编排者-工人）契约测试。

不依赖真实 LLM：注入 FakeLLM 验证委派链路
（delegate_agent 工具注册 / 线程局部 host / 工人多步 ReAct /
工人失败隔离 / 上下文边界：工人轨迹不进入编排者历史）。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams
from myagent.agent_loop import AgentLoop
from myagent.contracts import (
    AgentRequest,
    AgentResponse,
    LLMResponse,
    StopReason,
)
from myagent.errors import ValidationError
from myagent.multi import (
    OrchestratorLoop,
    build_orchestrator_loop,
    register_worker_tool,
    unregister_worker_tool,
)
from myagent.multi.context import render_worker_prompt
from myagent.multi.worker_tool import worker_tools_schema
from myagent.tools import ToolExecutor
from myagent.tools.registry import TOOLS, to_openai_tools


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
    """按脚本序列应答；耗尽后抛 AssertionError（防止静默越界）。"""

    def __init__(self, outputs: list[str | LLMResponse]):
        self.outputs = list(outputs)
        self.calls: list[list] = []

    @staticmethod
    def _wrap(output: str | LLMResponse) -> LLMResponse:
        return output if isinstance(output, LLMResponse) else LLMResponse(text=output)

    def complete(self, messages, *, tools=None, max_new_tokens=None):
        self.calls.append([list(messages), list(tools or []), max_new_tokens])
        if not self.outputs:
            raise AssertionError("FakeLLM 输出脚本耗尽")
        return self._wrap(self.outputs.pop(0))


class RecordingExecutor(ToolExecutor):
    """记录每次工具调用的执行器（复用真实分发：delegate_agent 走线程局部 host）。"""

    def __init__(self, cwd: str | Path):
        super().__init__(cwd)
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict) -> str:
        self.calls.append((name, dict(args or {})))
        return super().execute(name, args)


class DelegateToolTest(unittest.TestCase):
    """delegate_agent 工具注册与基本行为。"""

    def setUp(self):
        register_worker_tool()  # 幂等

    def tearDown(self):
        unregister_worker_tool()  # 测试隔离：不污染全局注册表（test_tools 断言依赖）

    def test_tool_registered_in_schema(self):
        spec = TOOLS.get("delegate_agent")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.risk, "read")  # 委派本身免询问
        names = [t["function"]["name"] for t in to_openai_tools()]
        self.assertIn("delegate_agent", names)

    def test_worker_schema_excludes_delegate(self):
        """工人 schema 不含 delegate_agent：从源头防止嵌套委派。"""
        names = [t["function"]["name"] for t in worker_tools_schema()]
        self.assertNotIn("delegate_agent", names)
        self.assertIn("read_file", names)  # 其余工具照常可见

    def test_missing_task_raises_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = RecordingExecutor(tmp)
            with self.assertRaises(ValidationError):
                executor.execute("delegate_agent", {})

    def test_without_host_returns_error_text(self):
        """非编排者上下文（无线程局部 host）→ 错误观察，不崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = RecordingExecutor(tmp)
            obs = executor.execute("delegate_agent", {"task": "调查"})
            self.assertIn("只能在编排者模式下使用", obs)


class WorkerResultTest(unittest.TestCase):
    """WorkerResult 文本化（DONE / FAILED + 统计行）。"""

    def _result(self, resp: AgentResponse):
        from myagent.multi.orchestrator import WorkerResult

        return WorkerResult.from_response(resp)

    def test_done_format(self):
        resp = AgentResponse(
            final_answer="结论 X", stop_reason=StopReason.FINAL_ANSWER,
            turns_used=3, tool_calls=4,
        )
        obs = self._result(resp).to_observation()
        self.assertTrue(obs.startswith("DONE: 结论 X"))
        self.assertIn("turns=3", obs)
        self.assertIn("tools=4", obs)

    def test_failed_format(self):
        resp = AgentResponse(
            final_answer=None, stop_reason=StopReason.MAX_TURNS,
            turns_used=12, tool_calls=12,
        )
        obs = self._result(resp).to_observation()
        self.assertTrue(obs.startswith("FAILED: max_turns"))

    def test_failed_includes_step_trace(self):
        from myagent.contracts import StepRecord
        from myagent.multi.orchestrator import WorkerResult

        step = StepRecord(
            turn=1, raw_output="", tool_calls=[tc("read_file", {"path": "a.py"}, "w1")],
        )
        resp = AgentResponse(
            final_answer=None, stop_reason=StopReason.ERROR,
            turns_used=1, tool_calls=1, error="boom", steps=[step],
        )
        obs = WorkerResult.from_response(resp).to_observation()
        self.assertIn("boom", obs)
        self.assertIn("read_file", obs)


class OrchestratorTest(unittest.TestCase):
    """编排者委派端到端链路。"""

    def tearDown(self):
        unregister_worker_tool()  # _make_orchestrator 会注册 delegate_agent

    def _make_orchestrator(
        self, tmpdir: str, orchestrator_llm: FakeLLM, worker_llm: FakeLLM
    ) -> OrchestratorLoop:
        executor = RecordingExecutor(tmpdir)
        params = AgentParams(cwd=tmpdir)
        inner = AgentLoop(params, llm=orchestrator_llm, tools=executor)

        def worker_factory(task, role, worker_turns):
            wparams = AgentParams(cwd=tmpdir, max_turns=worker_turns or 12)
            return AgentLoop(wparams, llm=worker_llm, tools=executor)

        register_worker_tool()  # 与 cli 装配一致：注册委派工具
        return OrchestratorLoop(inner, worker_factory=worker_factory)

    def test_delegate_end_to_end(self):
        """编排者委派 → 工人多步 ReAct → DONE 观察 → 编排者收口。"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.py").write_text("print(1)", encoding="utf-8")
            orch_llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[tc("delegate_agent", {"task": "调查 a.py"}, "call_d")],
                    ),
                    "汇总完成",
                ]
            )
            worker_llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[tc("read_file", {"path": "a.py"}, "w1")],
                    ),
                    "a.py 内容：print(1)",
                ]
            )
            orch = self._make_orchestrator(tmp, orch_llm, worker_llm)
            resp = orch.run(AgentRequest(user_input="帮我调查"))

            self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
            self.assertEqual(resp.final_answer, "汇总完成")

            # 委派观察文本：DONE 前缀 + 工人统计（工人跑了两轮）。
            msgs = orch_llm.calls[1][0]
            tool_msgs = [m for m in msgs if m.role == "tool"]
            self.assertEqual(len(tool_msgs), 1)
            self.assertTrue(tool_msgs[0].content.startswith("DONE: a.py 内容"))
            self.assertIn("turns=2", tool_msgs[0].content)
            # 工人子轨迹未进入编排者历史：没有 read_file 的独立 tool 消息。
            self.assertNotIn("w1", [m.tool_call_id for m in msgs if m.role == "tool"])

    def test_worker_max_turns_isolated(self):
        """工人达到轮数上限 → FAILED 观察，编排者继续收口。"""
        with tempfile.TemporaryDirectory() as tmp:
            orch_llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[tc("delegate_agent", {"task": "任务", "worker_turns": 3}, "call_d")],
                    ),
                    "已了解",
                ]
            )
            tool_resp = LLMResponse(text="", tool_calls=[tc("read_file", {"path": "a.py"}, "w1")])
            worker_llm = FakeLLM([tool_resp] * 3)  # 3 轮全调工具 → 到轮数上限
            orch = self._make_orchestrator(tmp, orch_llm, worker_llm)
            resp = orch.run(AgentRequest(user_input="委派"))

            self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
            msgs = orch_llm.calls[1][0]
            tool_msg = [m for m in msgs if m.role == "tool"][0]
            self.assertTrue(tool_msg.content.startswith("FAILED: max_turns"))
            self.assertIn("turns=3", tool_msg.content)

    def test_worker_exception_isolated(self):
        """工人 loop 抛异常 → 转换为 FAILED 观察，编排者不崩。"""

        class BoomWorkerLLM:
            def complete(self, messages, *, tools=None, max_new_tokens=None):
                raise RuntimeError("worker boom")

        with tempfile.TemporaryDirectory() as tmp:
            orch_llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[tc("delegate_agent", {"task": "任务"}, "call_d")],
                    ),
                    "兜底完成",
                ]
            )
            executor = RecordingExecutor(tmp)
            params = AgentParams(cwd=tmp)
            inner = AgentLoop(params, llm=orch_llm, tools=executor)

            def worker_factory(task, role, worker_turns):
                return AgentLoop(
                    AgentParams(cwd=tmp), llm=BoomWorkerLLM(), tools=executor
                )

            register_worker_tool()  # 与 cli 装配一致
            orch = OrchestratorLoop(inner, worker_factory=worker_factory)
            resp = orch.run(AgentRequest(user_input="委派"))

            self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
            msgs = orch_llm.calls[1][0]
            tool_msg = [m for m in msgs if m.role == "tool"][0]
            self.assertTrue(tool_msg.content.startswith("FAILED:"))
            self.assertIn("worker boom", tool_msg.content)


class BuildOrchestratorTest(unittest.TestCase):
    """cli 装配路径：build_orchestrator_loop。"""

    def tearDown(self):
        unregister_worker_tool()  # build_orchestrator_loop 会注册 delegate_agent

    def test_build_registers_tool_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch_llm = FakeLLM(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[tc("delegate_agent", {"task": "统计文件数"}, "call_d")],
                    ),
                    "完成",
                ]
            )
            worker_llm = FakeLLM(
                [
                    LLMResponse(text="", tool_calls=[tc("list_files", {"path": "."}, "w1")]),
                    "共 1 个文件",
                ]
            )
            params = AgentParams(cwd=tmp)
            tools = RecordingExecutor(tmp)

            # worker 也用同一 FakeLLM 无法直接注入，这里用 build 工厂 + 手动替换。
            orch = build_orchestrator_loop(
                agent_params=params, llm=orch_llm, tools=tools, max_workers=2,
            )
            # 覆盖工人 llm：build 工厂内部用同一个 llm，此处换掉内层即可。
            orch._worker_factory = self._worker_factory(tmp, worker_llm, tools)

            resp = orch.run(AgentRequest(user_input="委派"))
            self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
            self.assertEqual(resp.final_answer, "完成")
            # 装配即注册工具。
            names = [t["function"]["name"] for t in to_openai_tools()]
            self.assertIn("delegate_agent", names)

    @staticmethod
    def _worker_factory(tmp, worker_llm, tools):
        def factory(task, role, worker_turns):
            return AgentLoop(
                AgentParams(cwd=tmp, max_turns=worker_turns or 12),
                llm=worker_llm, tools=tools,
            )

        return factory


class RenderWorkerPromptTest(unittest.TestCase):
    """工人提示词渲染：{task} 替换 / 契约兜底 / 角色模板 / 回退。"""

    def test_substitutes_task_and_appends_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "worker.md").write_text(
                "你是工人。\n\n## 你的任务\n\n{task}\n", encoding="utf-8"
            )
            rendered = render_worker_prompt(tmp, "读 a.py")
            self.assertIn("读 a.py", rendered)
            self.assertIn("## 输出契约", rendered)
            self.assertIn("DONE: <结论>", rendered)

    def test_role_template_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "worker.md").write_text("通用\n{task}", encoding="utf-8")
            (Path(tmp) / "worker_researcher.md").write_text(
                "你是调查员\n{task}\n## 输出契约\n已有", encoding="utf-8"
            )
            rendered = render_worker_prompt(tmp, "查 a.py", role="researcher")
            self.assertIn("调查员", rendered)
            # 模板已自带契约 → 不重复追加。
            self.assertEqual(rendered.count("## 输出契约"), 1)

    def test_fallback_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendered = render_worker_prompt(tmp, "任务", role="ghost_role")
            self.assertIn("任务", rendered)
            self.assertIn("## 输出契约", rendered)


class ArgparseTest(unittest.TestCase):
    """--multi_agent 系列参数解析。"""

    def test_multi_flags_parsed(self):
        from myagent.argparse import parse_args

        args = parse_args(["--multi_agent", "--max_workers", "2"])
        self.assertTrue(args.multi_agent)
        self.assertEqual(args.max_workers, 2)
        self.assertIsNone(args.worker_prompt_dir)

    def test_defaults_off(self):
        from myagent.argparse import parse_args

        args = parse_args([])
        self.assertFalse(args.multi_agent)
        self.assertEqual(args.max_workers, 4)


if __name__ == "__main__":
    unittest.main()
