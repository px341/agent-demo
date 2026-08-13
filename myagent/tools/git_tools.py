"""git 工具：状态查看 / diff（hunk 对齐裁剪）/ patch 应用。

- 全部 ``git -C <cwd>`` 执行（cwd 由 executor 注入），timeout=30 硬超时；
- 非零退出 / 非 git 仓库 → CommandError（可恢复，stderr 回灌）；
- ``git_diff`` 对 context 有直接指导意义：默认先给 ``--stat`` 摘要，
  完整 diff 用 hunk 对齐裁剪（clip_diff_hunks）——保留完整的 hunk，
  不把 patch 从中间切碎，模型看到的变更结构完整。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..context.tokens import clip_diff_hunks
from ..errors import CommandError, ToolTimeoutError
from .registry import register_tool

#: git 命令硬超时（真正杀掉子进程）。
GIT_TIMEOUT = 30

#: diff hunk 裁剪预算（token）。
DIFF_BUDGET_TOKENS = 4000


def _run_git(cwd: Path, args: list[str], input_text: str | None = None) -> str:
    """执行 git 命令，成功返回 stdout；失败抛 CommandError。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            input=input_text,
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(f"git 命令超过 {GIT_TIMEOUT} 秒：{' '.join(args[:3])}") from exc
    except OSError as exc:
        raise CommandError(f"git 命令执行失败：{exc}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise CommandError(f"git {' '.join(args[:3])} 退出码 {proc.returncode}：\n{stderr[:2000]}")
    return proc.stdout or ""


@register_tool(
    "git_status",
    description="查看 git 仓库状态：当前分支、暂存区、未跟踪文件（git status --short）。",
    risk="read",
)
def git_status(args: dict, cwd: Path) -> str:
    branch = _run_git(cwd, ["branch", "--show-current"]).strip()
    status = _run_git(cwd, ["status", "--short"])
    lines = [f"分支：{branch or '(detached)'}", "变更："]
    lines.append(status if status.strip() else "（工作区干净）")
    return "\n".join(lines)


@register_tool(
    "git_diff",
    description="查看工作区变更（git diff）：默认返回 stat 摘要 + 完整 diff"
    "（hunk 对齐裁剪，保留完整 hunk）；可指定单文件 path。",
    risk="read",
    parameters={
        "path": {"type": "string", "description": "只看指定文件的 diff（相对工作目录）"},
        "cached": {"type": "boolean", "default": False, "description": "查看暂存区（git diff --cached）"},
        "max_tokens": {"type": "integer", "default": 4000, "description": "diff 裁剪预算（token）"},
    },
)
def git_diff(args: dict, cwd: Path) -> str:
    cached = bool(args.get("cached"))
    path = args.get("path")
    max_tokens = int(args.get("max_tokens") or DIFF_BUDGET_TOKENS)
    base = ["diff"] + (["--cached"] if cached else [])

    # 1) stat 摘要（小，总能完整返回）。
    stat = _run_git(cwd, [*base, "--stat"])
    # 2) 完整 diff（单文件或全部），hunk 对齐裁剪。
    diff_args = [*base]
    if path:
        diff_args.extend(["--", str(path)])
    raw_diff = _run_git(cwd, diff_args)

    if not raw_diff.strip():
        return f"工作区无变更（暂存区={cached}）"
    clipped = clip_diff_hunks(raw_diff, max_tokens)
    return f"{stat}\n\n{clipped}"


@register_tool(
    "git_apply_patch",
    description="应用 patch 到工作区（git apply）。patch 由模型生成或从别处获得。",
    risk="write",
    parameters={
        "patch": {"type": "string", "required": True, "description": "unified diff patch 文本"},
        "cached": {"type": "boolean", "default": False, "description": "应用到暂存区（--cached）"},
    },
)
def git_apply_patch(args: dict, cwd: Path) -> str:
    patch = str(args.get("patch") or "")
    if not patch.strip():
        raise CommandError("patch 为空")
    apply_args = ["apply", "--whitespace=nowarn"] + (["--cached"] if args.get("cached") else [])
    _run_git(cwd, apply_args, input_text=patch)
    return "patch 已成功应用"
