"""工具注册表与执行器。

工具以「名称 -> 函数」的形式注册到 TOOLS，execute_tool(name, args) 按名称分发执行，
返回字符串结果（供模型作为 role="tool" 的消息继续推理）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

TOOLS: dict[str, Callable[..., str]] = {}


def register_tool(name: str):
    """装饰器：把函数注册为可用工具。"""

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        TOOLS[name] = func
        return func

    return decorator


@register_tool("read_file")
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """读取文本文件内容（可指定行区间）。"""
    file_path = Path(path)
    if not file_path.is_file():
        return f"错误：文件不存在：{path}"
    lines = file_path.read_text(encoding="utf-8").splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line) if end_line is not None else len(lines)
    if start > end:
        return f"错误：行区间无效（{start} > {end}）"
    content = "\n".join(lines[start - 1 : end])
    return f"{path}（第 {start}-{end} 行，共 {len(lines)} 行）：\n{content}"


@register_tool("list_files")
def list_files(path: str = ".") -> str:
    """列出目录下的内容。"""
    dir_path = Path(path)
    if not dir_path.is_dir():
        return f"错误：目录不存在：{path}"
    items = sorted(
        str(item.name) + ("/" if item.is_dir() else "")
        for item in dir_path.iterdir()
    )
    if not items:
        return f"{path}/：空目录"
    return f"{path}/：\n" + "\n".join(items)


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """按名称执行工具并返回字符串结果；未知工具或参数错误时返回错误信息。"""
    func = TOOLS.get(name)
    if func is None:
        return f"错误：未知工具 {name!r}，可用工具：{', '.join(sorted(TOOLS))}"
    try:
        return str(func(**args))
    except TypeError as exc:
        return f"错误：工具 {name} 参数不正确：{exc}"
    except Exception as exc:
        return f"错误：工具 {name} 执行失败：{exc}"


def list_tools() -> list[str]:
    """返回所有已注册工具名称。"""
    return sorted(TOOLS)
