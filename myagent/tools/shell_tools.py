"""shell 工具：执行任意命令（高风险，审批必问）。

- 参数列表执行（shlex.split，**禁止 shell=True**），cwd 由 executor 注入，
  天然限制在工作目录内；
- 硬超时：subprocess.run(timeout=...)，真正杀掉子进程
  （executor 的线程池超时只能放弃等待，杀不死进程）；
- 破坏性命令黑名单：命中即抛 ToolPermissionError（不可恢复）；
- 非零退出码抛 CommandError（可恢复），命令输出回灌给模型。

安全模型：审批闸门（risk=write 必询问）+ 命令黑名单 + cwd 限制 + 硬超时。
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from ..errors import CommandError, ToolPermissionError, ToolTimeoutError
from .registry import register_tool

#: 输出截断上限（工具层自限，防单条 tool 输出爆 max_tool_tokens）。
MAX_OUTPUT_CHARS = 8000

#: 错误输出截断上限（回灌给模型时各截断）。
MAX_ERROR_CHARS = 2000

#: 危险命令集合（命中即拦截）。
_DANGEROUS_COMMANDS = {
    "dd", "mkfs", "mke2fs", "shutdown", "reboot", "halt", "poweroff",
    "sudo", "su", "chmod", "chown", "fdisk", "parted", "kill", "pkill",
    "killall", "youtube-dl", "curl", "wget", "nc", "ncat", "telnet",
    "scp", "rsync", "fuser", "lsof",
}

#: git 破坏性操作（reset --hard / clean -fdx 等）。
_DANGEROUS_GIT = {
    "reset", "clean", "checkout",
}


def _check_blacklist(tokens: list[str]) -> None:
    """扫描命令参数，命中破坏性/危险模式即抛 ToolPermissionError。"""
    if not tokens:
        raise ToolPermissionError("命令为空")
    first = tokens[0]
    if first in _DANGEROUS_COMMANDS:
        raise ToolPermissionError(f"命令 {first!r} 被列入黑名单，禁止执行")
    if first == "rm":
        # rm -rf / rm -fr / rm -r -f 等强制递归删除。
        if any(t.startswith("-") and "r" in t and "f" in t for t in tokens[1:]):
            raise ToolPermissionError("rm -rf 递归强制删除被禁止")
    if first == "git":
        for i, tok in enumerate(tokens[1:], 1):
            if tok in _DANGEROUS_GIT:
                # reset --hard / clean -fdx / checkout -f 破坏工作区。
                rest = tokens[i + 1 :]
                if any(t in ("-f", "--force", "-fd", "-fdx", "--hard", "-ff") for t in rest):
                    raise ToolPermissionError(f"git {tok} 破坏性操作被禁止")
    # 含管道/分号的组合命令同样黑名单拦截（避免绕过首个命令检查）。
    for token in tokens:
        if token in ("&&", "||", ";", "|", ">", ">>", "<"):
            raise ToolPermissionError("组合命令（&& / | / ; 等）被禁止，请使用单个命令")


@register_tool(
    "run_shell",
    description="在工作目录内执行 shell 命令（参数列表方式，禁止管道/重定向；"
    "非零退出码返回错误）。用于运行测试、编译、查看命令输出。",
    risk="write",
    parameters={
        "command": {"type": "string", "required": True, "description": "要执行的命令（单个命令，无管道/分号）"},
        "timeout": {"type": "integer", "default": 30, "description": "超时秒数（默认 30）"},
    },
)
def run_shell(args: dict, cwd: Path) -> str:
    command = str(args["command"]).strip()
    timeout = int(args.get("timeout") or 30)
    if not command:
        raise CommandError("命令为空")
    tokens = shlex.split(command)
    _check_blacklist(tokens)

    try:
        proc = subprocess.run(
            tokens,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(
            f"命令执行超过 {timeout} 秒：{command[:80]}"
        ) from exc
    except OSError as exc:
        raise CommandError(f"命令执行失败：{exc}") from exc

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        # 非零退出：输出回灌给模型（可恢复，模型决定重试或基于输出继续）。
        raise CommandError(
            f"命令退出码 {proc.returncode}：\n"
            f"{stdout[:MAX_ERROR_CHARS]}\n{stderr[:MAX_ERROR_CHARS]}"
        )
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = stdout[:MAX_OUTPUT_CHARS] + "\n…[输出已截断]…"
    if stderr:
        return stdout + "\n(stderr)\n" + stderr[:MAX_ERROR_CHARS]
    return stdout or "（命令执行成功，无输出）"
