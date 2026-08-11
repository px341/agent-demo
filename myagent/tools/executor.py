"""工具执行器：按名称分发执行，把结果/异常统一转成观察文本。

``ToolExecutor`` 满足 ``contracts.ToolExecutor`` 协议
（``execute(name, args) -> str``），可直接注入 AgentLoop。

错误约定：任何失败都返回以「错误：」开头的文本，由主循环作为
observation 回灌给模型继续推理，不向主循环抛异常。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import TOOLS, ToolSpec


class ToolExecutor:
    """带工作目录上下文的工具执行器。"""

    def __init__(self, cwd: str | Path):
        self.cwd = Path(cwd).resolve()

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """按名称执行工具并返回观察文本；未知工具或任何失败都转成错误文本。"""
        spec = TOOLS.get(name)
        if spec is None:
            available = ", ".join(list_tools())
            return f"错误：未知工具 {name!r}，可用工具：{available}"
        invalid = _validate_params(spec, args)
        if invalid is not None:
            return invalid
        try:
            return str(spec.func(dict(args or {}), self.cwd))
        except KeyError as exc:
            return f"错误：工具 {name} 缺少参数 {exc.args[0]!r}"
        except (TypeError, ValueError) as exc:
            return f"错误：工具 {name} 参数不正确：{exc}"
        except Exception as exc:
            return f"错误：工具 {name} 执行失败：{exc}"


def _validate_params(spec: "ToolSpec", args: dict[str, Any]) -> str | None:
    """按 ToolSpec.parameters（JSON Schema 风格）校验必填与类型；返回错误文本或 None。"""
    supplied = args or {}
    for name, schema in spec.parameters.items():
        if schema.get("required") and name not in supplied:
            return f"错误：工具 {spec.name} 缺少参数 {name!r}"
        if name not in supplied:
            continue
        expected = schema.get("type")
        value = supplied[name]
        if expected == "string" and not isinstance(value, str):
            return f"错误：工具 {spec.name} 参数 {name!r} 必须是字符串"
        if expected == "boolean" and not isinstance(value, bool):
            return f"错误：工具 {spec.name} 参数 {name!r} 必须是布尔值"
        # bool 是 int 子类，需显式排除。
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return f"错误：工具 {spec.name} 参数 {name!r} 必须是整数"
    return None


def list_tools() -> list[str]:
    """返回所有已注册工具名称（排序）。"""
    return sorted(TOOLS)
