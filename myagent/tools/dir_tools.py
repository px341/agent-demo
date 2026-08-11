"""目录类工具：读取 / 创建 / 写入 / 重命名 / 删除。

语义对应：
- list_files  ：读取目录内容（对应「读取目录」）；
- create_dir  ：创建单层目录（对应「创建目录」）；
- write_dir   ：递归创建目录结构（mkdir -p，对应「写入目录」）；
- rename_dir  ：重命名/移动目录（对应「修改目录」）；
- delete_dir  ：删除目录，默认仅空目录（对应「删除目录」）。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .common import safe_resolve
from .registry import register_tool

_PARAM_PATH = {"type": "string", "required": True, "description": "目录路径（相对工作目录或绝对路径）"}


@register_tool(
    "list_files",
    description="列出目录内容（文件和子目录，目录带 / 后缀）。",
    risk="read",
    parameters={
        "path": {"type": "string", "default": ".", "description": "目录路径"},
    },
)
def list_files(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args.get("path", "."))
    if not path.is_dir():
        return f"错误：目录不存在：{args.get('path', '.')}"
    items = sorted(
        str(item.name) + ("/" if item.is_dir() else "")
        for item in path.iterdir()
    )
    if not items:
        return f"{args.get('path', '.')}/：空目录"
    return f"{args.get('path', '.')}/：\n" + "\n".join(items)


@register_tool(
    "create_dir",
    description="创建单层目录（父目录必须已存在）。",
    parameters={"path": _PARAM_PATH},
)
def create_dir(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    try:
        path.mkdir()
    except FileExistsError:
        return f"错误：目录已存在：{args['path']}"
    except FileNotFoundError:
        return f"错误：父目录不存在：{args['path']}；如需递归创建请用 write_dir"
    return f"已创建目录：{args['path']}"


@register_tool(
    "write_dir",
    description="递归创建目录结构（mkdir -p，多级目录可一次建出）。",
    parameters={"path": _PARAM_PATH},
)
def write_dir(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    path.mkdir(parents=True, exist_ok=True)
    return f"已创建目录（递归）：{args['path']}"


@register_tool(
    "rename_dir",
    description="重命名或移动目录（可跨目录移动，目标仍在工作目录内）。",
    parameters={
        "src": _PARAM_PATH,
        "dst": {"type": "string", "required": True, "description": "新路径（目标名）"},
    },
)
def rename_dir(args: dict, cwd: Path) -> str:
    src = safe_resolve(cwd, args["src"])
    dst = safe_resolve(cwd, args["dst"])
    if not src.is_dir():
        return f"错误：目录不存在：{args['src']}"
    if src == cwd.resolve():
        return "错误：禁止重命名工作目录本身"
    if dst.exists():
        return f"错误：目标已存在：{args['dst']}"
    shutil.move(str(src), str(dst))
    return f"已重命名目录：{args['src']} → {args['dst']}"


@register_tool(
    "delete_dir",
    description="删除目录；默认仅删除空目录，recursive=true 时递归删除。",
    risk="delete",
    parameters={
        "path": _PARAM_PATH,
        "recursive": {"type": "boolean", "default": False, "description": "是否递归删除非空目录"},
    },
)
def delete_dir(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    if not path.is_dir():
        return f"错误：目录不存在：{args['path']}"
    if path == cwd.resolve():
        return "错误：禁止删除工作目录本身"
    if args.get("recursive"):
        shutil.rmtree(path)
        return f"已删除目录（递归）：{args['path']}"
    try:
        path.rmdir()
    except OSError:
        return f"错误：目录非空，删除失败：{args['path']}；如需递归删除请设 recursive=true"
    return f"已删除目录：{args['path']}"
