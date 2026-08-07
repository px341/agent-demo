"""LLM 生成参数：集中存储传给模型的默认配置。

这里的默认值对所有协议统一管理；各客户端在发请求时按自己支持的范围取用。
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """传给 LLM 的生成参数。

    协议说明（本项目的 DeepSeek 走 Anthropic / messages 协议）：
    - OpenAI  (responses)：OpenAI Responses API
    - Anthropic (messages)：Anthropic Messages API
    - Ollama：本地 Ollama

    每个字段末尾标注适用协议；客户端按需取用，不支持的字段直接忽略。
    """

    # 采样温度，越高越随机。OpenAI / Anthropic / Ollama 均支持。
    temperature: float = 0.7

    # 核采样概率截断。OpenAI / Anthropic / Ollama 均支持。
    top_p: float | None = None

    # 取概率最高的前 K 个词。Anthropic / Ollama 支持（OpenAI 不支持）。
    top_k: int | None = None

    # 最大生成 token 数。OpenAI 里对应 max_output_tokens；三家均支持。
    max_tokens: int = 8192

    # 触发停止生成的字符串序列。OpenAI=stop / Anthropic=stop_sequences / Ollama=stop。
    stop: tuple[str, ...] = ()

    # 对已出现 token 的重复惩罚。仅 Ollama 支持（本项目 OpenAI/Anthropic 端点不支持）。
    presence_penalty: float | None = None

    # 对高出现频率 token 的惩罚。仅 Ollama 支持。
    frequency_penalty: float | None = None

    # 随机种子，用于结果可复现。仅 Ollama 支持。
    seed: int | None = None

    # 是否流式返回。三家均支持（当前实现均为非流式）。
    stream: bool = False
    
    # 系统提示词。OpenAI=instructions / Anthropic=system / Ollama=system。
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


# 全局默认参数：调用方不传 params 时使用。
MODELS_PARAMS = GenerationParams()
