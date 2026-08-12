"""上下文压缩测试：三部分预算（system+memory / session history / 用户输入）。

覆盖：
- 预算充足零改动；用户输入永不压缩；
- system 超限时优先压缩 memory 段；
- 单条 tool 输出超限 → 开头+结尾各半 + 裁剪标记，配对保持；
- tool 总量超限 → 最早 tool 对整对丢弃、最近保留；
- 非 tool 强压缩 → 丢最早 user/纯文本 assistant，保护 tool 链；
- 兜底阶段才丢 tool 对；无孤儿消息；相对顺序保持；
- AgentLoop 注入 composer 后 run() 生效；composer=None 原逻辑不变。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams, Message
from myagent.agent_loop import AgentLoop
from myagent.context import ContextComposer
from myagent.context.tokens import (
    clip_head_tail,
    count_message_tokens,
    count_tokens,
    truncate_to_tokens,
)
from myagent.contracts import (
    AgentRequest,
    AgentResponse,
    LLMResponse,
    StepRecord,
    StopReason,
)


def make_composer(
    max_input=4000,
    max_output=500,
    max_tool_tokens=400,
    max_total_tool_tokens=1000,
) -> ContextComposer:
    return ContextComposer(
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        max_tool_tokens=max_tool_tokens,
        max_total_tool_tokens=max_total_tool_tokens,
    )


def tool_pair(prefix: str = "t1", obs: str = "结果") -> tuple[Message, Message]:
    """构造一对 (assistant tool_calls 声明, tool 结果) 消息。"""
    assistant = Message(
        role="assistant",
        content=f'{{"action": "tool_call", "tool": "read_file"}}',
        tool_calls=[
            {
                "id": f"call_{prefix}",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "a.py"}',
                },
            }
        ],
    )
    result = Message(role="tool", content=obs, tool_call_id=f"call_{prefix}")
    return assistant, result


class TokensTest(unittest.TestCase):
    """token 计数与裁剪工具。"""

    def test_count_tokens_empty_and_text(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens(None), 0)
        self.assertGreater(count_tokens("hello world"), 0)
        # 中文也走 cl100k 分词，token 数 > 0。
        self.assertGreater(count_tokens("你好世界"), 0)

    def test_truncate_to_tokens_keeps_head(self):
        text = "A" * 1000
        clipped = truncate_to_tokens(text, 50)
        self.assertIn("已裁剪", clipped)
        self.assertLess(count_tokens(clipped), 100)
        self.assertTrue(clipped.startswith("AAA"))

    def test_truncate_within_budget_unchanged(self):
        text = "short"
        self.assertEqual(truncate_to_tokens(text, 1000), text)

    def test_clip_head_tail_keeps_both_ends(self):
        text = "头" + "中" * 2000 + "尾"
        clipped = clip_head_tail(text, 100)
        self.assertIn("已裁剪", clipped)
        self.assertLess(count_tokens(clipped), 200)
        self.assertTrue(clipped.startswith("头"))
        self.assertTrue(clipped.endswith("尾"))

    def test_clip_head_tail_zero_budget(self):
        self.assertEqual(clip_head_tail("abc", 0), "")


class ComposeThreePartTest(unittest.TestCase):
    """三部分结构：system + history + user。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_budget_sufficient_returns_unchanged(self):
        c = make_composer()
        hist = [Message(role="user", content="旧问题"), Message(role="assistant", content="旧回答")]
        out = c.compose("system prompt", hist, "新问题")
        self.assertEqual([m.role for m in out], ["system", "user", "assistant", "user"])
        self.assertEqual(out[1].content, "旧问题")
        self.assertEqual(out[2].content, "旧回答")
        self.assertEqual(out[3].content, "新问题")

    def test_user_input_never_compressed(self):
        c = make_composer(max_input=200, max_output=50)
        user_input = "非常重要的用户输入" * 100  # 远超预算
        out = c.compose("x", [], user_input)
        self.assertEqual(out[-1].content, user_input)  # 完整保留

    def test_user_input_not_modified_when_present_with_history(self):
        c = make_composer(max_input=200, max_output=50)
        hist = [Message(role="user", content="H" * 100)]
        user_input = "最终问题"
        out = c.compose("x", list(hist), user_input)
        self.assertEqual(out[-1].content, "最终问题")

    def test_inputs_not_mutated(self):
        c = make_composer(max_input=200, max_output=50)
        hist = [Message(role="user", content="H" * 1000)]
        hist_snapshot = [(m.role, m.content) for m in hist]
        out = c.compose("system", hist, "u")
        self.assertEqual([(m.role, m.content) for m in hist], hist_snapshot)
        # 返回列表是全新对象。
        self.assertIsNot(out[1], hist[0])


class SystemCompressionTest(unittest.TestCase):
    """Part 1：memory 段优先压缩。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _prompt_with_memory(self, memory_body: str) -> str:
        return "## 跨会话记忆\n\n" + memory_body + "\n\n## 工作环境与角色\n\n环境内容"

    def test_system_within_budget_unchanged(self):
        c = make_composer()
        prompt = self._prompt_with_memory("短记忆")
        out = c.compose(prompt, [], "u")
        self.assertEqual(out[0].content, prompt)

    def test_memory_section_compressed_first(self):
        c = make_composer(max_input=350, max_output=50)
        prompt = self._prompt_with_memory("记忆" * 500)  # 大记忆段 + 小环境段
        out = c.compose(prompt, [], "u")
        system = out[0].content
        # 环境段（"工作环境与角色" 标题）保留，memory 段被压缩。
        self.assertIn("工作环境与角色", system)
        self.assertIn("已裁剪", system)
        # 压缩后 system 落在预算内（预算 = 350 - 50 = 300）。
        self.assertLessEqual(count_tokens(system), 300)

    def test_system_without_memory_truncated_tail(self):
        c = make_composer(max_input=300, max_output=50)
        prompt = "开头\n\n" + "填充" * 1000 + "\n\n结尾"
        out = c.compose(prompt, [], "u")
        self.assertIn("已裁剪", out[0].content)
        self.assertTrue(out[0].content.startswith("开头"))


class ToolClippingTest(unittest.TestCase):
    """阶段 1：单条 tool 输出裁剪。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_oversized_tool_output_clipped_head_tail(self):
        c = make_composer(max_tool_tokens=100, max_total_tool_tokens=1000)
        assistant, result = tool_pair("a", "头" + "中" * 2000 + "尾")
        out = c.compose("s", [assistant, result], "u")
        tool = [m for m in out if m.role == "tool"][0]
        self.assertIn("已裁剪", tool.content)
        self.assertTrue(tool.content.startswith("头"))
        self.assertTrue(tool.content.endswith("尾"))
        # 配对保持：tool_call_id 与 assistant 的 tool_calls 声明一致。
        assistant_msg = [m for m in out if m.role == "assistant"][0]
        self.assertEqual(tool.tool_call_id, assistant_msg.tool_calls[0]["id"])

    def test_small_tool_output_unchanged(self):
        c = make_composer(max_tool_tokens=1000)
        assistant, result = tool_pair("a", "小结果")
        out = c.compose("s", [assistant, result], "u")
        self.assertEqual([m for m in out if m.role == "tool"][0].content, "小结果")

    def test_clip_does_not_touch_assistant_calls_declaration(self):
        c = make_composer(max_tool_tokens=50, max_total_tool_tokens=1000)
        assistant, result = tool_pair("a", "X" * 2000)
        out = c.compose("s", [assistant, result], "u")
        assistant_msg = [m for m in out if m.role == "assistant"][0]
        self.assertEqual(assistant_msg.tool_calls[0]["id"], "call_a")
        self.assertEqual(
            assistant_msg.tool_calls[0]["function"]["name"], "read_file"
        )


class ToolBudgetDropTest(unittest.TestCase):
    """阶段 2：tool 总量超限 → 丢弃最早 tool 对。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_oldest_tool_pair_dropped_recent_kept(self):
        c = make_composer(
            max_input=4000, max_output=100,
            max_tool_tokens=10000, max_total_tool_tokens=200,
        )
        a1, r1 = tool_pair("a", "A" * 500)  # 早期大结果
        a2, r2 = tool_pair("b", "B" * 500)  # 最近大结果
        out = c.compose("s", [a1, r1, a2, r2], "u")
        # 最早的 a1/r1 对被丢，最近的 b 保留。
        self.assertNotIn("call_a", {m.tool_call_id for m in out if m.role == "tool"})
        self.assertIn("call_b", {m.tool_call_id for m in out if m.role == "tool"})
        self.assertTrue(all(m.role != "assistant" or not m.tool_calls or m.tool_calls[0]["id"] != "call_a" for m in out))

    def test_no_orphan_tool_messages(self):
        c = make_composer(max_input=4000, max_output=100, max_total_tool_tokens=100)
        a1, r1 = tool_pair("a", "A" * 500)
        a2, r2 = tool_pair("b", "B" * 500)
        out = c.compose("s", [a1, r1, a2, r2], "u")
        roles = [m.role for m in out]
        # 不允许出现孤儿 tool（前面没有配对的 assistant tool_calls）。
        self.assertFalse(
            any(
                m.role == "tool"
                and not any(
                    mm.role == "assistant" and mm.tool_calls
                    for mm in out[: i]
                )
                for i, m in enumerate(out)
            )
        )


class NonToolCompressionTest(unittest.TestCase):
    """阶段 3：非 tool 强压缩（丢最早 user / 纯文本 assistant）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_oldest_non_tool_dropped_tool_chain_protected(self):
        # 构造：早期 user → 早期纯文本 assistant → tool 对 → 最近 user。
        c = make_composer(
            max_input=2000, max_output=100,
            max_tool_tokens=1000, max_total_tool_tokens=2000,
        )
        msgs = [
            Message(role="user", content="早期问题" + "E" * 400),
            Message(role="assistant", content="早期回答" + "F" * 400),
            *tool_pair("a", "G" * 100),
            Message(role="user", content="最近问题"),
        ]
        out = c.compose("s", msgs, "u")
        # 最早 user 和纯文本 assistant 被丢；tool 对与最近 user 保留。
        self.assertNotIn("早期问题", [m.content for m in out])
        self.assertNotIn("早期回答", [m.content for m in out])
        self.assertIn("call_a", {m.tool_call_id for m in out if m.role == "tool"})
        self.assertIn("最近问题", [m.content for m in out])

    def test_assistant_with_tool_calls_not_dropped_in_phase3(self):
        c = make_composer(
            max_input=1500, max_output=100,
            max_tool_tokens=10000, max_total_tool_tokens=10000,
        )
        assistant, result = tool_pair("a", "结果" * 300)
        old_user = Message(role="user", content="问题" * 200)
        out = c.compose("s", [old_user, assistant, result], "u")
        # 带 tool_calls 的 assistant 声明保留（tool 链完整性）。
        self.assertTrue(
            any(m.role == "assistant" and m.tool_calls for m in out)
        )


class ComposeIntegrationTest(unittest.TestCase):
    """AgentLoop 注入 composer 后生效；composer=None 原逻辑不变。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _fake_llm(self, outputs):
        class Fake:
            calls = []

            def complete(self, messages, *, tools=None, max_new_tokens=None):
                self.calls.append(list(messages))
                return LLMResponse(text=outputs.pop(0))

        return Fake()

    def test_loop_with_composer_compresses(self):
        llm = self._fake_llm(["ok"])
        params = AgentParams(cwd=str(self.tmp.name), max_input_tokens=200, max_output_tokens=50)
        composer = ContextComposer(
            max_input_tokens=params.max_input_tokens,
            max_output_tokens=params.max_output_tokens,
            max_tool_tokens=20,
            max_total_tool_tokens=50,
        )
        loop = AgentLoop(params, llm=llm, composer=composer)
        loop.run(AgentRequest(user_input="u"))
        msgs = llm.calls[0]
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[-1].content, "u")

    def test_loop_without_composer_unchanged(self):
        llm = self._fake_llm(["ok"])
        params = AgentParams(cwd=str(self.tmp.name))
        loop = AgentLoop(params, llm=llm)
        loop.run(AgentRequest(user_input="u"))
        msgs = llm.calls[0]
        self.assertEqual([m.role for m in msgs], ["system", "user"])

    def test_loop_with_composer_default_params_zero_change(self):
        # 默认参数 + 小历史：composer 预算充足 → 消息与不注入时一致。
        llm = self._fake_llm(["ok"])
        params = AgentParams(cwd=str(self.tmp.name))
        composer = ContextComposer(
            max_input_tokens=params.max_input_tokens,
            max_output_tokens=params.max_output_tokens,
            max_tool_tokens=params.max_tool_tokens,
            max_total_tool_tokens=params.max_total_tool_tokens,
        )
        loop = AgentLoop(params, llm=llm, composer=composer)
        loop.run(AgentRequest(user_input="u"))
        msgs = llm.calls[0]
        self.assertEqual([m.role for m in msgs], ["system", "user"])


class BudgetRespectTest(unittest.TestCase):
    """压缩结果必须落在整体预算内（输入 - 输出预留）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_compose_result_within_budget(self):
        c = make_composer(max_input=1000, max_output=100)
        # 巨大的 history 必须被压进 900 token 预算。
        hist = [
            Message(role="user", content=f"问题{i}" + "中" * 50)
            for i in range(50)
        ]
        out = c.compose("s", hist, "u")
        total = sum(count_message_tokens(m) for m in out)
        self.assertLessEqual(total, 900)

    def test_recompress_result_within_budget(self):
        c = make_composer(max_input=1000, max_output=100)
        msgs = [
            Message(role="system", content="s"),
            *tool_pair("a", "结果" * 200),
            Message(role="user", content="问题" * 100),
        ]
        out = c.recompress(msgs)
        total = sum(count_message_tokens(m) for m in out)
        self.assertLessEqual(total, 900)

    def test_budget_zero_still_returns_system_and_user(self):
        # 极端预算下 system 可能被压到只剩开头，但 user 必须完整保留。
        c = ContextComposer(max_input_tokens=100, max_output_tokens=100)
        out = c.compose("x", [], "问题" * 100)
        self.assertEqual(out[0].role, "system")
        self.assertEqual(out[-1].content, "问题" * 100)


class OrderPreservationTest(unittest.TestCase):
    """压缩/丢弃必须保持消息相对顺序。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_relative_order_preserved_after_drop(self):
        c = make_composer(
            max_input=2000, max_output=100,
            max_tool_tokens=1000, max_total_tool_tokens=2000,
        )
        msgs = [
            Message(role="user", content="u1" + "中" * 300),
            Message(role="assistant", content="a1" + "中" * 300),
            *tool_pair("a", "t1"),
            Message(role="user", content="u2" + "中" * 300),
            Message(role="assistant", content="a2" + "中" * 300),
            *tool_pair("b", "t2"),
            Message(role="user", content="最近问题"),
        ]
        out = c.compose("s", msgs, "最近输入")
        body = out[1:-1]
        idx = 0
        for m in msgs:
            if idx >= len(body):
                break
            if body[idx].role == m.role and body[idx].content == m.content:
                idx += 1
        # 全部 body 消息都能按原序匹配到（子序列）。
        self.assertEqual(idx, len(body))

    def test_order_preserved_in_tool_pair_drop(self):
        c = make_composer(
            max_input=4000, max_output=100,
            max_tool_tokens=10000, max_total_tool_tokens=100,
        )
        a1, r1 = tool_pair("a", "A" * 500)
        a2, r2 = tool_pair("b", "B" * 500)
        a3, r3 = tool_pair("c", "C" * 500)
        out = c.compose("s", [a1, r1, a2, r2, a3, r3], "u")
        tool_ids = [m.tool_call_id for m in out if m.role == "tool"]
        # 保留的最近对，顺序仍正确。
        self.assertEqual(tool_ids, ["call_c"])


class EdgeCaseTest(unittest.TestCase):
    """极端参数防御。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_zero_tool_budget_clips_everything(self):
        c = make_composer(max_tool_tokens=0, max_total_tool_tokens=0)
        assistant, result = tool_pair("a", "内容")
        out = c.compose("s", [assistant, result], "u")
        # tool 结果被裁空但消息与配对仍在。
        tool = [m for m in out if m.role == "tool"]
        self.assertEqual(len(tool), 1)
        self.assertEqual(tool[0].content, "")

    def test_tiny_budget_still_returns_system_and_user(self):
        c = make_composer(max_input=30, max_output=5)
        out = c.compose("s" * 100, [], "u" * 100)
        # system + user 始终在（user 不可压缩，system 尽力保留）。
        self.assertEqual(out[0].role, "system")
        self.assertEqual(out[-1].role, "user")
        self.assertEqual(out[-1].content, "u" * 100)

    def test_no_history(self):
        c = make_composer()
        out = c.compose("s", [], "u")
        self.assertEqual([m.role for m in out], ["system", "user"])


class RecompressTest(unittest.TestCase):
    """recompress（每轮发给 LLM 前的压缩）：tool 约束必须无条件生效。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_oversized_tool_clipped_even_when_overall_within_budget(self):
        # 单条 tool 超限 → 必须裁剪，即使整体预算充足（独立约束）。
        c = make_composer(max_tool_tokens=100, max_total_tool_tokens=1000)
        msgs = [
            Message(role="system", content="s"),
            *tool_pair("a", "头" + "中" * 2000 + "尾"),
            Message(role="user", content="u"),
        ]
        out = c.recompress(msgs)
        tool = [m for m in out if m.role == "tool"][0]
        self.assertIn("已裁剪", tool.content)
        self.assertTrue(tool.content.startswith("头"))
        self.assertTrue(tool.content.endswith("尾"))

    def test_oldest_tool_pair_dropped_when_total_over_budget(self):
        # tool 总量超限 → 无条件丢最早对，即使整体预算充足。
        c = make_composer(
            max_input=4000, max_output=100,
            max_tool_tokens=10000, max_total_tool_tokens=200,
        )
        msgs = [
            Message(role="system", content="s"),
            *tool_pair("a", "A" * 500),
            *tool_pair("b", "B" * 500),
            Message(role="user", content="u"),
        ]
        out = c.recompress(msgs)
        tool_ids = {m.tool_call_id for m in out if m.role == "tool"}
        self.assertIn("call_b", tool_ids)
        self.assertNotIn("call_a", tool_ids)
        # 配对的 assistant 声明同样被移除。
        self.assertFalse(any(m.tool_calls for m in out if m.role == "assistant" and m.tool_calls[0]["id"] == "call_a"))

    def test_last_user_message_never_dropped(self):
        # 最后一条 user 是用户输入 → 即使极端预算也不丢（Part 3 铁律）。
        c = make_composer(max_input=200, max_output=50)
        msgs = [
            Message(role="system", content="s" * 200),
            Message(role="user", content="重要问题"),
        ]
        out = c.recompress(msgs)
        self.assertEqual(out[-1].role, "user")
        self.assertEqual(out[-1].content, "重要问题")
        self.assertEqual(out[0].role, "system")

    def test_first_message_system_kept(self):
        # 第一条 system 永不删除（保留但可能被截断）。
        c = make_composer(max_input=100, max_output=20)
        msgs = [
            Message(role="system", content="sys" * 100),
            *tool_pair("a", "结果"),
            Message(role="user", content="问题"),
        ]
        out = c.recompress(msgs)
        self.assertEqual(out[0].role, "system")

    def test_unchanged_when_within_budget(self):
        c = make_composer(max_tool_tokens=1000, max_total_tool_tokens=1000)
        msgs = [
            Message(role="system", content="s"),
            Message(role="user", content="旧"),
            Message(role="assistant", content="旧回答"),
            *tool_pair("a", "小结果"),
            Message(role="user", content="u"),
        ]
        out = c.recompress(msgs)
        self.assertEqual(len(out), len(msgs))
        self.assertEqual(
            [(m.role, m.content) for m in out],
            [(m.role, m.content) for m in msgs],
        )

    def test_system_untouched_by_recompress(self):
        # recompress 不重复压缩 system（其压缩在 compose 初始完成）。
        c = make_composer()
        system_text = "原样 system"
        msgs = [
            Message(role="system", content=system_text),
            Message(role="user", content="问题"),
        ]
        out = c.recompress(msgs)
        self.assertEqual(out[0].content, system_text)

    def test_no_orphan_after_recompress(self):
        c = make_composer(
            max_input=4000, max_output=100,
            max_tool_tokens=10000, max_total_tool_tokens=100,
        )
        msgs = [
            Message(role="system", content="s"),
            *tool_pair("a", "A" * 300),
            *tool_pair("b", "B" * 300),
            Message(role="user", content="u"),
        ]
        out = c.recompress(msgs)
        # 无孤儿 tool 结果（每个 tool 前面都有配对的 assistant 声明）。
        for i, m in enumerate(out):
            if m.role == "tool":
                self.assertTrue(
                    any(
                        mm.role == "assistant" and mm.tool_calls
                        for mm in out[:i]
                    ),
                    f"发现孤儿 tool 消息 {m.tool_call_id}",
                )

    def test_recompress_mutates_only_clipped_tool_content(self):
        # 裁剪只替换超限 tool 的 content 字符串；其余消息对象引用不变。
        c = make_composer(max_tool_tokens=50, max_total_tool_tokens=1000)
        small = Message(role="user", content="小消息")
        assistant, result = tool_pair("a", "X" * 2000)
        msgs = [
            Message(role="system", content="s"),
            small,
            assistant,
            result,
            Message(role="user", content="u"),
        ]
        out = c.recompress(msgs)
        # 未裁剪的消息是同一对象。
        self.assertIs(out[1], small)


class InTurnComposeTest(unittest.TestCase):
    """轮内集成：主循环每轮调 composer.recompress，tool 约束对累积消息生效。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _make(self, max_input=3000, max_output=100, max_tool=100, max_total=1000):
        return ContextComposer(
            max_input_tokens=max_input,
            max_output_tokens=max_output,
            max_tool_tokens=max_tool,
            max_total_tool_tokens=max_total,
        )

    class _FakeTools:
        def __init__(self, content):
            self.content = content

        def execute(self, name, args):
            return self.content

    class _FakeLLM:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = []  # 每次 complete 收到的 messages 快照

        def complete(self, messages, *, tools=None, max_new_tokens=None):
            self.calls.append(list(messages))
            return self.outputs.pop(0)

    def _tool_resp(self, name: str, call_id: str) -> LLMResponse:
        return LLMResponse(
            text="",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": '{"path": "x"}'},
                }
            ],
        )

    def test_later_turn_tool_result_is_clipped(self):
        # 回归：第二轮发给 LLM 的 tool 结果必须被裁剪（此前被绕过）。
        tools = self._FakeTools("X" * 5000)
        llm = self._FakeLLM(
            [self._tool_resp("read_file", "call_1"), LLMResponse(text="ok")]
        )
        params = AgentParams(cwd=str(self.tmp.name), max_turns=3)
        loop = AgentLoop(
            params, llm=llm, tools=tools, composer=self._make(max_input=3000)
        )
        resp = loop.run(AgentRequest(user_input="读文件"))
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        second_turn = llm.calls[1]
        tool_msgs = [m for m in second_turn if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("已裁剪", tool_msgs[0].content)
        self.assertLess(len(tool_msgs[0].content), 1000)
        # 配对保持。
        self.assertEqual(tool_msgs[0].tool_call_id, "call_1")

    def test_later_turn_user_message_never_dropped(self):
        # 多轮会话中，当前用户输入（被 tool 对顶到中间）始终保留。
        tools = self._FakeTools("小结果")
        llm = self._FakeLLM(
            [self._tool_resp("read_file", "call_1"), LLMResponse(text="ok")]
        )
        params = AgentParams(cwd=str(self.tmp.name), max_turns=3)
        loop = AgentLoop(params, llm=llm, tools=tools, composer=self._make(max_input=3000))
        loop.run(AgentRequest(user_input="读文件"))
        second_turn = llm.calls[1]
        user_msgs = [m for m in second_turn if m.role == "user"]
        self.assertEqual(user_msgs[-1].content, "读文件")

    def test_multi_tool_turns_keep_only_recent(self):
        # 三轮 tool 调用后，总量超限只保留最近一对，且系统/用户输入仍在。
        tools = self._FakeTools("数据" * 200)  # 每条约 200 token
        llm = self._FakeLLM(
            [
                self._tool_resp("read_file", "call_1"),
                self._tool_resp("read_file", "call_2"),
                self._tool_resp("read_file", "call_3"),
                LLMResponse(text="done"),
            ]
        )
        params = AgentParams(cwd=str(self.tmp.name), max_turns=5)
        loop = AgentLoop(
            params, llm=llm, tools=tools,
            composer=self._make(max_input=3000, max_tool=10000, max_total=250),
        )
        loop.run(AgentRequest(user_input="读三个文件"))
        last_turn = llm.calls[-1]
        self.assertEqual(last_turn[0].role, "system")
        # 当前用户输入保留（被 tool 对顶到中间）。
        user_msgs = [m for m in last_turn if m.role == "user"]
        self.assertEqual(user_msgs[-1].content, "读三个文件")
        tool_ids = [m.tool_call_id for m in last_turn if m.role == "tool"]
        self.assertEqual(tool_ids, ["call_3"])  # 只保留最新一对
        self.assertFalse(
            any(
                m.role == "assistant" and m.tool_calls
                and m.tool_calls[0]["id"] in ("call_1", "call_2")
                for m in last_turn
            )
        )

    def test_recompress_disabled_when_composer_none(self):
        tools = self._FakeTools("X" * 5000)
        llm = self._FakeLLM(
            [self._tool_resp("read_file", "call_1"), LLMResponse(text="ok")]
        )
        params = AgentParams(cwd=str(self.tmp.name), max_turns=3)
        loop = AgentLoop(params, llm=llm, tools=tools, composer=None)
        loop.run(AgentRequest(user_input="读文件"))
        tool_msgs = [m for m in llm.calls[1] if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertNotIn("已裁剪", tool_msgs[0].content)
        self.assertEqual(len(tool_msgs[0].content), 5000)


if __name__ == "__main__":
    unittest.main()
