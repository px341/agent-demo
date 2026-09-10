"""multi-agent 包：编排者-工人（orchestrator-worker）模式。

- orchestrator.py ：OrchestratorLoop（包装 AgentLoop + 委派能力）与
  build_orchestrator_loop 装配工厂；
- worker_tool.py   ：``delegate_agent`` 工具注册（显式传播的调用上下文 host）；
- context.py       ：工人角色提示词渲染（worker_<role>.md / worker.md）。

用法（cli --multi_agent）：:

    loop = build_orchestrator_loop(
        agent_params=params, llm=client, tools=tools,
        worker_prompt_dir=...,
    )
    try:
        response = loop.run(AgentRequest(user_input=...))
    finally:
        loop.close()
"""
from __future__ import annotations

from .context import render_worker_prompt
from .orchestrator import (
    OrchestratorLoop,
    WorkerResult,
    build_orchestrator_loop,
)
from .worker_tool import register_worker_tool, unregister_worker_tool

__all__ = [
    "OrchestratorLoop",
    "WorkerResult",
    "build_orchestrator_loop",
    "register_worker_tool",
    "unregister_worker_tool",
    "render_worker_prompt",
]
