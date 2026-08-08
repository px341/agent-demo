from __future__ import annotations

import os
import sys

import readline

from pathlib import Path
from .argparse import parse_args
from .api_config import MODELS_PARAMS
from .provider import complete
from .agent_config import AgentParams
from .agent_loop import AgentLoop



BANNER = r"""
 ███╗   ███╗██╗   ██╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗
 ████╗ ████║╚██╗ ██╔╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
 ██╔████╔██║ ╚████╔╝ ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
 ██║╚██╔╝██║  ╚██╔╝  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
 ██║ ╚═╝ ██║   ██║   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
 ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   

                 AI Agent Terminal

转义命令：
    /exit   退出
    /quit   退出
"""


def main() -> int:
    print(BANNER)

    args = parse_args()
    agent_params = AgentParams(
        cwd=args.cwd,
    )

    # 工作目录设定：--cwd 指定后先切换过去。
    try:
        os.chdir(args.cwd)
    except OSError as exc:
        print(f"无法进入工作目录 {args.cwd!r}：{exc}", file=sys.stderr)
        return 1

    # 两种模式：--one_shot 调用一次；否则进入交互式多轮。
    if args.one_shot:
        return _one_shot(agent_params)
    else:
        agent_loop = AgentLoop(agent_params)
        agent_loop.run()


def _one_shot(agent_params: AgentParams) -> int:
    """one_shot 模式：只调用一次 LLM，提示词由用户直接输入。"""
    try:
        prompt = input("你：").strip()
    except EOFError:
        prompt = ""

    if not prompt:
        print("没有输入内容，已退出。", file=sys.stderr)
        return 1

    # 通过修改 agent_params 加载 one_shot 系统提示词文件内容：
    # 赋值为文件名，setter 会自动读取 prompts/one_shot_system_prompt.md。
    agent_params.system_prompt = "one_shot_system_prompt"
    system_prompt = (agent_params.system_prompt or "").strip()
    if system_prompt:
        prompt = f"{prompt}\n\n{system_prompt}"

    try:
        result = complete(max_new_tokens=MODELS_PARAMS.max_tokens, prompt=prompt)
    except RuntimeError as exc:
        print(f"连接失败：{exc}", file=sys.stderr)
        return 1

    print(f"{result.settings.name}（{result.settings.model}）：{result.answer}")
    return 0


