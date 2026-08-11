"""从 .env.llm（或 .env.example 模板）加载 LLM provider，并提供统一调用函数。

"""

from __future__ import annotations
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

import os
import json
from pydantic import BaseModel

from .agent_config import AgentParams, Message
from .contracts import LLMResponse

# 优先读取本地的 .env.llm（真实配置/密钥），缺失时回退到 .env.example 模板。
ENV_PATH = (
    Path(__file__).resolve().parent.parent / ".env.llm"
)
DEFAULT_PROVIDER = "DEEPSEEK"  # 默认使用 DeepSeek provider

print(f"加载 LLM provider 配置：{ENV_PATH}")

load_dotenv(ENV_PATH)


DeepseekClient = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
)

DEFAULT_MODEL = "deepseek-v4-flash"


class PlannerResponse(BaseModel):
    plan: str
    steps: list[str]


class OpenAICompatibleModelClient:
    def __init__(self, agent_params: AgentParams):
        self.client = DeepseekClient
        self.agent_params = agent_params
        self.model = os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL

    def complete(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
    ) -> LLMResponse:
        """主循环的模型推理接口：完整消息列表 → 结构化返回（文本 + 元数据）。

        使用 OpenAI SDK 的 chat.completions（OpenAI 兼容端点通用格式），
        消息结构与主循环的 Message 契约一一对应：
        system / user / assistant 消息原样透传，tool 消息附带 tool_call_id。

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            stream=False,
            max_tokens=max_new_tokens or self.agent_params.max_output_tokens,
            temperature=0.2,
        )

        message = response.choices[0].message
        text = message.content or ""
        # DeepSeek 在 assistant 消息上扩展了 reasoning_content；
        # SDK 可能通过属性或 model_extra 暴露，两处都取一下。
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None and hasattr(message, "model_extra"):
            reasoning = (message.model_extra or {}).get("reasoning_content")
        metadata = {"reasoning_content": reasoning} if reasoning else {}
        return LLMResponse(text=text, metadata=metadata)

    def responses_planner(self, input: str = "在目录下创建一个文件夹保存一首诗"):
        response = self.client.responses.parse(
            model=self.model,
            instructions="你是一个planner",
            input=input,
            stream=False,
            max_output_tokens=self.agent_params.max_output_tokens,
            temperature=0.2,
            text_format=PlannerResponse,
        )
        
        # 直接获取解析后的 Pydantic 对象
        return response.output_parsed 
