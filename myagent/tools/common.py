"""工具共享设施：路径安全解析。

所有文件/目录工具都通过 :func:`safe_resolve` 把模型传入的路径解析为
工作目录内的绝对路径，越界（``../`` 逃逸或指向 cwd 之外的绝对路径）一律
抛 :class:`myagent.errors.ToolPermissionError`（不可恢复，终止本轮任务），
保证工具无法读写工作区之外的内容。
"""
from __future__ import annotations

from pathlib import Path

from ..errors import ToolPermissionError


def safe_resolve(cwd: Path, path: str) -> Path:
    """把工具参数中的路径解析为 cwd 内的绝对路径；越界抛 ToolPermissionError。"""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(cwd.resolve())
    except ValueError:
        raise ToolPermissionError(f"路径越界（禁止访问工作目录之外）：{path}")
    return resolved
