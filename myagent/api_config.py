"""LLM 默认配置：目前仅保留系统提示词回退。

DEFAULT_SYSTEM_PROMPT 是主循环加载 prompts/tools_system_prompt.md 失败时
的兜底提示词（agent_loop._load_system_prompt）。
"""
from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
