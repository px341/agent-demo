"""环境上下文 prompt 构建：角色定义 + 环境信息 + workspace tree。

主循环每次运行时调用 :func:`build_environment_prompt` 生成
「环境与角色」系统提示（与静态的 tools_system_prompt.md 拼接），
让模型无需调用工具即可感知工作区结构与运行环境。

- workspace tree 有深度限制并忽略内部/临时目录（.git/.venv 等），
  避免大目录撑爆上下文；
- 环境信息包括 cwd、操作系统、Python 版本、Git 仓库状态、当前时间。
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

#: 文件树中忽略的目录/文件名（内部、临时或包含敏感数据）。
IGNORED_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".storage",
    ".idea",
    ".reasonix",
    ".DS_Store",
}

#: 默认目录树深度。max_depth=N 表示展示到第 N 层目录的名字、
#: 不展开其内容（max_depth=1 只列根目录一层，至少为 1）。
DEFAULT_MAX_DEPTH = 3

#: 单个目录最多列出的条目数，超出后折叠显示。
DEFAULT_MAX_ENTRIES = 100

TEMPLATE_NAME = "environment_system_prompt.md"


def _git_info(cwd: Path) -> str:
    """检测 cwd 是否为 Git 仓库；是则返回分支信息，否则返回"否"。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "否"
    if proc.returncode != 0:
        return "否"
    branch = proc.stdout.strip()
    return f"是（当前分支 {branch}）" if branch else "是"


def build_workspace_tree(
    root: Path,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> str:
    """生成工作区目录树文本；root 不存在时返回空提示。"""
    if not root.is_dir():
        return "（工作目录不存在或不可读）"
    max_depth = max(1, max_depth)

    lines: list[str] = ["."]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                (e for e in directory.iterdir() if e.name not in IGNORED_NAMES),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except OSError:
            return

        if len(entries) > max_entries:
            shown, rest = entries[:max_entries], entries[max_entries:]
        else:
            shown, rest = entries, []

        for index, entry in enumerate(shown):
            is_last = index == len(shown) - 1 and not rest
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            # symlink 一律按叶子显示，避免跟随符号链接造成递归循环。
            if entry.is_dir() and not entry.is_symlink():
                lines.append(prefix + connector + entry.name + "/")
                walk(entry, child_prefix, depth + 1)
            else:
                lines.append(prefix + connector + entry.name)
        if rest:
            lines.append(prefix + f"└── … 还有 {len(rest)} 项（已省略）")

    walk(root, "", 1)
    return "\n".join(lines)


def collect_environment_info(cwd: Path) -> dict[str, Any]:
    """收集环境信息，键与模板占位符一一对应。"""
    return {
        "cwd": str(cwd),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "git_repo": _git_info(cwd),
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_depth": DEFAULT_MAX_DEPTH,
    }


def build_environment_prompt(prompt_dir: Path | str, cwd: Path | str) -> str:
    """读取环境 prompt 模板并填入运行时信息，返回完整的环境系统提示。"""
    template = Path(prompt_dir) / TEMPLATE_NAME
    try:
        text = template.read_text(encoding="utf-8")
    except OSError:
        return ""

    resolved_cwd = Path(cwd).resolve()
    info = collect_environment_info(resolved_cwd)
    info["workspace_tree"] = build_workspace_tree(resolved_cwd)
    for key, value in info.items():
        text = text.replace("{" + key + "}", str(value))
    return text
