"""OpenAICompatibleModelClient.complete() 的契约测试。

mock 掉 OpenAI SDK 客户端，验证：
- 消息列表 → chat.completions payload 的映射（system 保留、tool 带 tool_call_id）；
- max_new_tokens / model / max_output_tokens 的参数传递；
- 返回值取 choices[0].message.content。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams, Message
from myagent.agent_loop import AgentLoop
from myagent.contracts import AgentRequest, StopReason
from myagent.provider import DEFAULT_MODEL, OpenAICompatibleModelClient


class FakeCompletions:
    """记录 create() 参数，返回固定内容（可选带 reasoning_content）。"""

    def __init__(
        self,
        content: str = "模型回复",
        reasoning: str | None = None,
        via_model_extra: bool = False,
    ):
        self.content = content
        self.reasoning = reasoning
        self.via_model_extra = via_model_extra
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.via_model_extra and self.reasoning is not None:
            # 模拟 SDK 未暴露属性、仅通过 model_extra 携带扩展字段的情况。
            msg = SimpleNamespace(
                content=self.content,
                model_extra={"reasoning_content": self.reasoning},
            )
        else:
            msg = SimpleNamespace(content=self.content)
            if self.reasoning is not None:
                msg.reasoning_content = self.reasoning
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_client(
    content: str = "模型回复",
    reasoning: str | None = None,
    via_model_extra: bool = False,
) -> tuple[OpenAICompatibleModelClient, FakeCompletions]:
    client = OpenAICompatibleModelClient(AgentParams())
    fake = FakeCompletions(content, reasoning, via_model_extra)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


class ProviderCompleteTest(unittest.TestCase):
    def test_message_mapping(self):
        client, fake = make_client()
        messages = [
            Message(role="system", content="系统提示"),
            Message(role="user", content="你好"),
        ]
        client.complete(messages, max_new_tokens=100)

        self.assertEqual(
            fake.kwargs["messages"],
            [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "你好"},
            ],
        )
        self.assertEqual(fake.kwargs["max_tokens"], 100)
        self.assertEqual(fake.kwargs["stream"], False)

    def test_tool_call_pair_passed_through(self):
        """assistant 的 tool_calls 声明与 tool 消息的 tool_call_id 必须原样透传。"""
        client, fake = make_client()
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "a.py"}',
                },
            }
        ]
        messages = [
            Message(
                role="assistant",
                content='{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                tool_calls=tool_calls,
            ),
            Message(role="tool", content="观察结果", tool_call_id="call_1"),
        ]
        client.complete(messages)

        self.assertEqual(
            fake.kwargs["messages"],
            [
                {
                    "role": "assistant",
                    "content": '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}',
                    "tool_calls": tool_calls,
                },
                {"role": "tool", "content": "观察结果", "tool_call_id": "call_1"},
            ],
        )

    def test_max_new_tokens_falls_back_to_params(self):
        client, fake = make_client()
        client.complete([Message(role="user", content="hi")])

        self.assertEqual(fake.kwargs["max_tokens"], AgentParams().max_output_tokens)

    def test_returns_choices_content(self):
        client, _ = make_client("最终输出")
        resp = client.complete([Message(role="user", content="hi")])
        self.assertEqual(resp.text, "最终输出")

    def test_reasoning_content_extracted(self):
        """thinking mode 的 reasoning_content 应提取进 LLMResponse.metadata。"""
        client, _ = make_client("回答", reasoning="思考过程")
        resp = client.complete([Message(role="user", content="hi")])
        self.assertEqual(resp.text, "回答")
        self.assertEqual(resp.metadata, {"reasoning_content": "思考过程"})

    def test_reasoning_content_via_model_extra(self):
        """SDK 未暴露属性时，应从 model_extra 兜底提取。"""
        client, _ = make_client("回答", reasoning="思考过程", via_model_extra=True)
        resp = client.complete([Message(role="user", content="hi")])
        self.assertEqual(resp.metadata, {"reasoning_content": "思考过程"})

    def test_reasoning_content_passed_back(self):
        """assistant 消息的 reasoning_content 元数据应恢复进请求 payload。"""
        client, fake = make_client()
        messages = [
            Message(
                role="assistant",
                content="输出",
                metadata={"reasoning_content": "上一轮思考"},
            )
        ]
        client.complete(messages)
        self.assertEqual(
            fake.kwargs["messages"][0]["reasoning_content"],
            "上一轮思考",
        )

    def test_no_reasoning_content_omitted(self):
        """无 reasoning_content 时请求 payload 不含该字段。"""
        client, fake = make_client()
        client.complete([Message(role="assistant", content="输出")])
        self.assertNotIn("reasoning_content", fake.kwargs["messages"][0])

    def test_complete_drives_agent_loop(self):
        """complete() 满足 LLMClient Protocol：返回文本可被主循环解析为 final。"""
        client, fake = make_client('{"action": "final", "answer": "集成OK"}')
        loop = AgentLoop(AgentParams(), llm=client)
        resp = loop.run(AgentRequest(user_input="任务"))

        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(resp.final_answer, "集成OK")
        self.assertEqual(fake.kwargs["max_tokens"], AgentParams().max_output_tokens)

    def test_model_from_env_with_default(self):
        with mock.patch.dict("os.environ", {"DEEPSEEK_MODEL": "my-model"}):
            client = OpenAICompatibleModelClient(AgentParams())
        self.assertEqual(client.model, "my-model")

        with mock.patch.dict("os.environ", {"DEEPSEEK_MODEL": ""}):
            client = OpenAICompatibleModelClient(AgentParams())
        self.assertEqual(client.model, DEFAULT_MODEL)


if __name__ == "__main__":
    unittest.main()
