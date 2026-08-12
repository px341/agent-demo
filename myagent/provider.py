"""从 .env.llm 加载 LLM provider 配置，并提供统一模型推理接口。

- OpenAI SDK 客户端**懒加载**：import 本模块无副作用（不打印、
  不构造客户端），首次实例化 OpenAICompatibleModelClient 时才构造；
- ``OpenAICompatibleModelClient.complete`` 满足 contracts.LLMClient 协议，
  是主循环唯一的模型调用入口。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .agent_config import AgentParams, Message
from .contracts import LLMResponse

# 读取本地的 .env.llm（真实配置/密钥），把变量注入进程环境。
ENV_PATH = Path(__file__).resolve().parent.parent / ".env.llm"
DEFAULT_PROVIDER = "DEEPSEEK"  # 默认使用 DeepSeek provider
DEFAULT_MODEL = "deepseek-v4-flash"

load_dotenv(ENV_PATH)

#: 延迟创建的 OpenAI 兼容客户端（首次调用时构造）。
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """按环境变量构造并缓存 OpenAI 兼容客户端（懒加载，避免 import 副作用）。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_API_BASE"),
        )
    return _client


class OpenAICompatibleModelClient:
    def __init__(self, agent_params: AgentParams):
        self.client = _get_client()
        self.agent_params = agent_params
        self.model = os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        max_new_tokens: int | None = None,
    ) -> LLMResponse:
        """主循环的模型推理接口：完整消息列表 → 结构化返回（文本 + 工具调用）。

        使用 OpenAI SDK 的 chat.completions（OpenAI 兼容端点通用格式），
        消息结构与主循环的 Message 契约一一对应：
        system / user / assistant 消息原样透传，tool 消息附带 tool_call_id。

        ``tools`` 为 OpenAI 原生 tools 数组（由注册表 to_openai_tools 生成）；
        DeepSeek 原生支持 tool_calls 输出（含一次多个并行调用）。

        DeepSeek thinking mode 下 assistant 响应带 ``reasoning_content``，
        后续请求必须原样回传，否则端点 400；因此这里提取进 metadata，
        由主循环随 assistant 消息携带回传。
        """
        payload = []
        for message in messages:
            item = {"role": message.role, "content": message.content or ""}
            if message.role == "tool" and message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if message.role == "assistant":
                reasoning = (message.metadata or {}).get("reasoning_content")
                if reasoning:
                    item["reasoning_content"] = reasoning
            payload.append(item)

        kwargs: dict = {
            "model": self.model,
            "messages": payload,
            "stream": False,
            "max_tokens": max_new_tokens or self.agent_params.max_output_tokens,
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        text = message.content or ""
        # DeepSeek 在 assistant 消息上扩展了 reasoning_content；
        # SDK 可能通过属性或 model_extra 暴露，两处都取一下。
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None and hasattr(message, "model_extra"):
            reasoning = (message.model_extra or {}).get("reasoning_content")
        metadata = {"reasoning_content": reasoning} if reasoning else {}

        tool_calls = []
        if getattr(message, "tool_calls", None):
            tool_calls = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return LLMResponse(text=text, tool_calls=tool_calls, metadata=metadata)
