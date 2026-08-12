"""记忆系统：会话存档 + 摘要聚合 + 常驻注入。

- manager.py    ：MemoryManager（会话生命周期 / 水位线状态机 / 注入文本）；
- archive.py    ：jsonl 原文存档、脱敏、files_changed 机械提取；
- summarizer.py ：单会话提取 + 跨会话聚合（走 LLMClient）。

满足 contracts.MemoryStore 协议，直接注入 AgentLoop：:

    from myagent.memory import MemoryManager
    memory = MemoryManager(memory_dir=params.memory_dir, llm=client)
    loop = AgentLoop(params, llm=client, tools=..., memory=memory)
"""
from __future__ import annotations

from .manager import MemoryManager, new_session_id

__all__ = ["MemoryManager", "new_session_id"]
