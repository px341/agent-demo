"""工具注册表：工具以 ToolSpec 元数据注册，执行层按名称分发。

注册与执行解耦：

- registry.py：注册表本体（ToolSpec / TOOLS / register_tool）；
- executor.py：执行分发（ToolExecutor，满足 contracts.ToolExecutor 协议）；
- file_tools.py / dir_tools.py：具体工具实现（模块导入即完成注册）。

工具实现签名统一为 ``func(args: dict, cwd: Path) -> str``：
``args`` 为模型传入的参数 dict，``cwd`` 为执行器注入的工作目录（用于路径安全解析）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: 工具实现签名：参数 dict + 工作目录 → 观察文本。
ToolFunc = Callable[[dict[str, Any], Path], str]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """一个工具的元数据与实现。"""

    #: 工具名称（模型在 tool_call 中使用）。
    name: str

    #: 实现函数。
    func: ToolFunc

    #: 工具说明（用于向模型描述工具用途）。
    description: str = ""

    #: 风险等级：read（只读）/ write（写入、修改）/ delete（删除、破坏性）。
    #: 审批闸门据此决定是否免询问（默认 read 免审）。
    risk: str = "write"

    #: 参数 JSON Schema 风格描述（供参数校验 / 动态渲染 prompt 使用）。
    parameters: dict[str, Any] = field(default_factory=dict)


#: 注册表：名称 → ToolSpec。
TOOLS: dict[str, ToolSpec] = {}


RISK_LEVELS = ("read", "write", "delete")


def to_openai_tools() -> list[dict[str, Any]]:
    """把注册表渲染为 OpenAI 原生 tools 数组（供 LLMClient.complete 传入）。

    ToolSpec.parameters 是参数 schema 字典（键为参数名，值为各参数的
    JSON Schema 片段）。OpenAI 端点要求 function.parameters 是合法的
    JSON Schema 对象（顶层 ``type: object``），因此这里包装成
    ``{"type": "object", "properties": {...}}``，并把参数级
    ``required: true`` 汇总到顶层 ``required``（OpenAI 不允许在属性内
    出现 required 键）。
    """
    tools: list[dict[str, Any]] = []
    for spec in TOOLS.values():
        raw = spec.parameters or {}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, schema in raw.items():
            prop = dict(schema)
            # OpenAI 属性级 schema 不支持 required / default 键，剥离。
            prop.pop("required", None)
            prop.pop("default", None)
            if schema.get("required"):
                required.append(name)
            properties[name] = prop
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": parameters,
                },
            }
        )
    return tools


def register_tool(
    name: str,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    risk: str = "write",
):
    """装饰器：把工具实现注册进 TOOLS。

    ``risk`` 取值 read / write / delete：read 只读，write 写入修改，
    delete 删除等破坏性操作（审批闸门默认对 read 免询问）。
    """
    if risk not in RISK_LEVELS:
        raise ValueError(f"未知风险等级 {risk!r}，可选：{'/'.join(RISK_LEVELS)}")

    def decorator(func: ToolFunc) -> ToolFunc:
        TOOLS[name] = ToolSpec(
            name=name,
            func=func,
            description=description,
            risk=risk,
            parameters=parameters or {},
        )
        return func

    return decorator
