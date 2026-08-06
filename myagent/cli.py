from __future__ import annotations

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

    # 仅 --one_shot 模式：调用一次 LLM，提示词由用户直接输入（不依赖 --prompt）。
    if not args.one_shot:
        print("用法：python -m myagent --one_shot", file=sys.stderr)
        return 0

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


