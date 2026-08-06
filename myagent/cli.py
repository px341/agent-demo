from __future__ import annotations

import os
import sys

import readline

from .argparse import parse_args
from .params import DEFAULT_PARAMS
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

    args = parse_args()

    # 工作目录设定：--cwd 指定后先切换过去。
    try:
        os.chdir(args.cwd)
    except OSError as exc:
        print(f"无法进入工作目录 {args.cwd!r}：{exc}", file=sys.stderr)
        return 1

    # 两种模式：--one_shot 调用一次；否则进入交互式多轮。
    if args.one_shot:
        return _one_shot()
    return _repl()


def _one_shot() -> int:
    """one_shot 模式：只调用一次 LLM，提示词由用户直接输入。"""
    try:
        prompt = input("你：").strip()
    except EOFError:
        prompt = ""

    if not prompt:
        print("没有输入内容，已退出。", file=sys.stderr)
        return 1

    try:
        result = complete(prompt, max_new_tokens=DEFAULT_PARAMS.max_tokens)
    except RuntimeError as exc:
        print(f"连接失败：{exc}", file=sys.stderr)
        return 1

    print(f"{result.settings.name}（{result.settings.model}）：{result.answer}")
    return 0


def _repl() -> int:
    """非 one_shot 模式：交互式多轮对话，输入 /exit、exit 或 quit 退出。"""
    print("输入 /exit 退出。")
    while True:
        try:
            prompt = input("你：").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt in ("/exit", "/quit", "exit", "quit"):
            return 0

        try:
            result = complete(prompt, max_new_tokens=DEFAULT_PARAMS.max_tokens)
        except RuntimeError as exc:
            print(f"连接失败：{exc}", file=sys.stderr)
            continue

        print(f"{result.settings.name}（{result.settings.model}）：{result.answer}")


