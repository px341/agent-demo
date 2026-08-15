"""``delegate_agent`` 工具：把子任务委派给工人 agent。

多 agent 的核心接入点：编排者（orchestrator）只是一个普通 AgentLoop，
通过本工具获得委派能力——工具执行时从**线程局部**取当前编排者 host，
调用其 ``delegate_worker(task, role, worker_turns)``，把工人结果
文本化为观察结果返回。主循环（AgentLoop）对多 agent 完全无感知。

- 工具标 ``risk="read"``（默认免询问）：委派本身不直接修改工作区，
  工人内部写操作仍受底层 shell 黑名单 / 工具错误体系兜底；
- 线程局部 host：编排者 ``run()`` 期间在**当前线程**设置 host，
  结束后清除。工人跑在后台线程，其线程内没有 host —— 因此
  **禁止嵌套委派**（工人调用 delegate_agent 会得到错误观察），
  防止委派失控递归。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..tools.registry import TOOLS, register_tool, to_openai_tools

#: 线程局部：当前线程的编排者 host（提供 delegate_worker 方法）。
_host = threading.local()


def _set_host(host: Any) -> None:
    """设置当前线程的编排者 host（orchestrator.run 调用，结束后清除）。"""
    _host.orchestrator = host


def _clear_host() -> None:
    try:
        del _host.orchestrator
    except AttributeError:
        pass


def _current_host() -> Any:
    return getattr(_host, "orchestrator", None)


#: 工具参数 schema（供 OpenAI 原生 tool_calls 渲染与参数校验）。
DELEGATE_PARAMETERS: dict[str, Any] = {
    "task": {
        "type": "string",
        "description": "要委派给工人 agent 的子任务描述（自包含、可独立完成）",
        "required": True,
    },
    "role": {
        "type": "string",
        "description": "工人角色名（可选）：加载 prompts/worker_<role>.md 模板",
        "required": False,
    },
    "worker_turns": {
        "type": "integer",
        "description": "工人 agent 的推理轮数上限（可选，默认 12）",
        "required": False,
    },
}


def register_worker_tool() -> None:
    """把 ``delegate_agent`` 注册进全局工具表（幂等，重复调用覆盖同名）。

    仅编排者模式调用；单 agent 进程不 import 本模块，注册表保持原样。
    """

    @register_tool(
        "delegate_agent",
        description=(
            "把子任务委派给一个独立的工人 agent 并行处理；工人与你共享工作目录，"
            "会自己调用工具完成任务，完成后返回结论。适合：可独立完成的子任务、"
            "需要并行推进的多条线索、独立调查/编码/审查。"
        ),
        parameters=DELEGATE_PARAMETERS,
        risk="read",
    )
    def _delegate_agent(args: dict[str, Any], cwd: Path) -> str:
        host = _current_host()
        if host is None:
            return (
                "ToolError[ExecutionError]: delegate_agent 只能在编排者模式下使用，"
                "且工人不允许嵌套委派；请直接完成当前任务。"
            )
        task = (args or {}).get("task")
        if not task or not isinstance(task, str):
            from ..errors import ValidationError

            raise ValidationError("delegate_agent 缺少必填参数 'task'（子任务描述）")
        role = (args or {}).get("role")
        worker_turns = (args or {}).get("worker_turns")
        result = host.delegate_worker(task, role=role, worker_turns=worker_turns)
        return result.to_observation()


def worker_tools_schema() -> list[dict[str, Any]]:
    """工人看到的工具 schema：全量工具去掉 ``delegate_agent``。

    防止工人嵌套委派（其线程内没有编排者 host，调用会得到错误观察，
    不如在 schema 层直接不可见，从源头避免模型发起递归委派）。
    """
    return [t for t in to_openai_tools() if t.get("function", {}).get("name") != "delegate_agent"]


def unregister_worker_tool() -> None:
    """把 ``delegate_agent`` 从全局工具表移除（幂等，不存在则无操作）。

    供测试隔离与动态关闭编排者能力使用；单 agent 进程不 import 本模块，
    正常路径无需调用。
    """
    TOOLS.pop("delegate_agent", None)
