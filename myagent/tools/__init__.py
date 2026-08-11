"""工具系统：注册 + 执行，文件/目录工具分层组织。

- registry.py   ：ToolSpec 注册表与 register_tool 装饰器；
- executor.py   ：ToolExecutor（满足 contracts.ToolExecutor 协议）与 list_tools；
- file_tools.py ：文件工具（read/create/write/edit/delete_file）；
- dir_tools.py  ：目录工具（list/create/write/rename/delete_dir）。

使用示例：:

    from myagent.tools import ToolExecutor
    executor = ToolExecutor(cwd=".")
    executor.execute("read_file", {"path": "a.py"})
"""
from __future__ import annotations

from . import dir_tools, file_tools  # noqa: F401  # 导入即完成工具注册
from .executor import ToolExecutor, list_tools
from .registry import TOOLS, ToolSpec, register_tool

__all__ = ["TOOLS", "ToolSpec", "ToolExecutor", "register_tool", "list_tools"]
