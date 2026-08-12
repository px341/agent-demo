"""上下文压缩：三部分预算管理（system+memory / 本轮 session / 用户输入）。

- tokens.py  ：tiktoken 计数（惰性加载，离线可用）；
- composer.py：ContextComposer（三部分压缩主逻辑）。

满足 contracts.ContextComposer 协议，直接注入 AgentLoop：:

    from myagent.context import ContextComposer
    composer = ContextComposer(
        max_input_tokens=params.max_input_tokens,
        max_output_tokens=params.max_output_tokens,
        max_tool_tokens=params.max_tool_tokens,
        max_total_tool_tokens=params.max_total_tool_tokens,
    )
    loop = AgentLoop(params, llm=client, composer=composer)
"""
from __future__ import annotations

from .composer import ContextComposer

__all__ = ["ContextComposer"]
