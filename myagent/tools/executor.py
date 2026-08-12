"""工具执行器：按名称分发执行，异常分类上抛给主循环。

``ToolExecutor`` 满足 ``contracts.ToolExecutor`` 协议
（``execute(name, args) -> str``），可直接注入 AgentLoop。

错误约定：工具失败抛 :class:`myagent.errors.ToolError` 子类（分类见
errors.py），由主循环按 recoverable 决定「回灌继续」还是「终止循环」；
``execute`` 本身不吞掉 ToolError，但把非 ToolError 的未知异常包装为
``ExecutionError``。超时通过线程池实现（``timeout`` 秒，0/None 表示不超时）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from ..errors import (
    ExecutionError,
    ToolError,
    ToolTimeoutError,
    ValidationError,
)
from .registry import TOOLS, ToolSpec


class ToolExecutor:
    """带工作目录上下文的工具执行器。"""

    def __init__(self, cwd: str | Path, timeout: float | None = None):
        self.cwd = Path(cwd).resolve()
        #: 单次工具执行的超时秒数；None / <=0 表示不超时。
        self.timeout = timeout if timeout and timeout > 0 else None

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """按名称执行工具；失败抛 ToolError 子类（未知异常包装为 ExecutionError）。"""
        spec = TOOLS.get(name)
        if spec is None:
            available = ", ".join(list_tools())
            raise ValidationError(f"未知工具 {name!r}，可用工具：{available}")
        _validate_params(spec, args)
        func = spec.func
        kwargs = dict(args or {})
        try:
            if self.timeout is None:
                return str(func(kwargs, self.cwd))
            return self._run_with_timeout(func, kwargs)
        except ToolError:
            raise  # 分类语义交给主循环
        except Exception as exc:
            raise ExecutionError(f"工具 {name} 执行失败：{exc}") from exc

    def _run_with_timeout(self, func, kwargs: dict) -> str:
        """在线程池中执行并限制耗时；超时抛 ToolTimeoutError。"""
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(func, kwargs, self.cwd)
            try:
                return str(future.result(timeout=self.timeout))
            except FutureTimeout:
                raise ToolTimeoutError(f"工具执行超过 {self.timeout} 秒")
            except ToolError:
                raise
            except Exception as exc:
                raise ExecutionError(f"工具执行失败：{exc}") from exc


def _validate_params(spec: "ToolSpec", args: dict[str, Any]) -> None:
    """按 ToolSpec.parameters（JSON Schema 风格）校验必填与类型；失败抛 ValidationError。"""
    supplied = args or {}
    for name, schema in spec.parameters.items():
        if schema.get("required") and name not in supplied:
            raise ValidationError(f"工具 {spec.name} 缺少参数 {name!r}")
        if name not in supplied:
            continue
        expected = schema.get("type")
        value = supplied[name]
        if expected == "string" and not isinstance(value, str):
            raise ValidationError(f"工具 {spec.name} 参数 {name!r} 必须是字符串")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValidationError(f"工具 {spec.name} 参数 {name!r} 必须是布尔值")
        # bool 是 int 子类，需显式排除。
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValidationError(f"工具 {spec.name} 参数 {name!r} 必须是整数")


def list_tools() -> list[str]:
    """返回所有已注册工具名称（排序）。"""
    return sorted(TOOLS)
