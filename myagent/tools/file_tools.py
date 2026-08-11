"""文件类工具：读取 / 创建 / 写入 / 编辑 / 删除。

每个工具都通过 safe_resolve 限制在工作目录内；任何失败都返回
以「错误：」开头的文本，由执行器回传给模型。
"""
from __future__ import annotations

from pathlib import Path

from .common import safe_resolve
from .registry import register_tool

_PARAM_PATH = {"type": "string", "required": True, "description": "文件路径（相对工作目录或绝对路径）"}


@register_tool(
    "read_file",
    description="读取 UTF-8 文本文件内容（可指定行区间）。",
    risk="read",
    parameters={
        "path": _PARAM_PATH,
        "start_line": {"type": "integer", "default": 1, "description": "起始行（从 1 开始）"},
        "end_line": {"type": "integer", "description": "结束行（含），缺省读到最后一行"},
    },
)
def read_file(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    if not path.is_file():
        return f"错误：文件不存在：{args['path']}"
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(1, int(args.get("start_line") or 1))
    end = min(len(lines), int(args["end_line"])) if args.get("end_line") is not None else len(lines)
    if start > end:
        return f"错误：行区间无效（{start} > {end}）"
    content = "\n".join(lines[start - 1 : end])
    return f"{args['path']}（第 {start}-{end} 行，共 {len(lines)} 行）：\n{content}"


@register_tool(
    "create_file",
    description="创建新文件（UTF-8，文件已存在则失败）。",
    parameters={
        "path": _PARAM_PATH,
        "content": {"type": "string", "default": "", "description": "文件内容"},
    },
)
def create_file(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    if path.exists():
        return f"错误：文件已存在：{args['path']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args.get("content", "")), encoding="utf-8")
    return f"已创建文件：{args['path']}"


@register_tool(
    "write_file",
    description="写入文本文件（新建或覆盖已有内容）。",
    parameters={
        "path": _PARAM_PATH,
        "content": {"type": "string", "required": True, "description": "要写入的完整内容"},
    },
)
def write_file(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args["content"]), encoding="utf-8")
    return f"已写入文件：{args['path']}"


@register_tool(
    "edit_file",
    description="编辑文件：精确文本替换（old_text 必须只出现一次）。",
    parameters={
        "path": _PARAM_PATH,
        "old_text": {"type": "string", "required": True, "description": "被替换的原文"},
        "new_text": {"type": "string", "default": "", "description": "替换后的文本"},
    },
)
def edit_file(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    if not path.is_file():
        return f"错误：文件不存在：{args['path']}"
    old_text = str(args["old_text"])
    new_text = str(args.get("new_text", ""))
    content = path.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count == 0:
        return "错误：文件中未找到要替换的文本"
    if count > 1:
        return f"错误：要替换的文本出现 {count} 次，请提供更精确的匹配"
    path.write_text(content.replace(old_text, new_text), encoding="utf-8")
    return f"已修改 {args['path']}：替换 1 处"


@register_tool(
    "delete_file",
    description="删除文件。",
    risk="delete",
    parameters={"path": _PARAM_PATH},
)
def delete_file(args: dict, cwd: Path) -> str:
    path = safe_resolve(cwd, args["path"])
    if not path.is_file():
        return f"错误：文件不存在：{args['path']}"
    path.unlink()
    return f"已删除文件：{args['path']}"
