"""``delegate_agent`` 工具：把子任务委派给工人 agent。

多 agent 的核心接入点：编排者（orchestrator）只是一个普通 AgentLoop，
通过本工具获得委派能力——工具执行时从调用上下文取当前编排者 host，
调用其 ``delegate_worker(task, role, worker_turns)``，把工人结果
文本化为观察结果返回。主循环（AgentLoop）对多 agent 完全无感知。

- 工具标 ``risk="read"``（默认免询问）：委派本身不直接修改工作区，
  工人的写操作已由**只读白名单**（schema + executor 双保险）杜绝——
  写者归一，写统一由编排者执行；
- ContextVar host：工具调度和超时线程显式复制调用上下文，
  工人执行线程使用空上下文，不继承 host —— 因此
  **禁止嵌套委派**（工人调用 delegate_agent 会得到错误观察），
  防止委派失控递归。
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from ..tools.registry import TOOLS, register_tool, to_openai_tools

#: 只传播到工具调度线程，不传播到工人 agent。
_host: ContextVar[Any] = ContextVar("orchestrator_host", default=None)


def _set_host(host: Any) -> Token:
    """设置编排者并返回恢复原上下文所需的 token。"""
    return _host.set(host)


def _clear_host(token: Token) -> None:
    _host.reset(token)


def _current_host() -> Any:
    return _host.get()


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
            "将独立子任务放在同一轮连续的 delegate_agent 调用中以并行执行。"
            "工人只读，修改建议以 patch 返回；所有写入由你等待工人结束后串行执行。"
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


def read_only_tool_names() -> set[str]:
    """从注册表推导只读工具名集合（risk=="read"），显式排除 delegate_agent。

    写者归一的安全核心：工人只有只读工具可用。schema（工人看不到写工具）
    与 executor 白名单（调了也拒绝）共用此集合；未来新增只读工具自动纳入。
    """
    return {
        name
        for name, spec in TOOLS.items()
        if spec.risk == "read" and name != "delegate_agent"
    }


def worker_tools_schema() -> list[dict[str, Any]]:
    """工人看到的工具 schema：只渲染只读工具（与 executor 白名单一致）。

    写者归一：工人只做只读调研、产出建议（patch 文本），不直接落盘；
    写操作由编排者统一执行与裁决。schema 层与 executor 层双保险——
    即使模型幻想调用写工具，也会被白名单拒绝（ValidationError 回灌）。
    """
    allowed = read_only_tool_names()
    return [
        t
        for t in to_openai_tools()
        if t.get("function", {}).get("name") in allowed
    ]


def unregister_worker_tool() -> None:
    """把 ``delegate_agent`` 从全局工具表移除（幂等，不存在则无操作）。

    供测试隔离与动态关闭编排者能力使用；单 agent 进程不 import 本模块，
    正常路径无需调用。
    """
    TOOLS.pop("delegate_agent", None)
