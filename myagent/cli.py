from __future__ import annotations

import sys

import readline

from .provider import complete


BANNER = r"""
 ███╗   ███╗██╗   ██╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗
 ████╗ ████║╚██╗ ██╔╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
 ██╔████╔██║ ╚████╔╝ ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
 ██║╚██╔╝██║  ╚██╔╝  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
 ██║ ╚═╝ ██║   ██║   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
 ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   

                 AI Agent Terminal
"""


def main() -> int:
    print(BANNER)

    # 一次性调用：支持 python -m myagent "问题"，无参数时进入交互输入。
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        try:
            prompt = input("你：").strip()
        except EOFError:
            prompt = ""

    if not prompt:
        print("没有输入内容，已退出。", file=sys.stderr)
        return 1

    try:
        result = complete(prompt)
    except RuntimeError as exc:
        print(f"连接失败：{exc}", file=sys.stderr)
        return 1

    print(f"{result.settings.name}（{result.settings.model}）：{result.answer}")
    return 0


