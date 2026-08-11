from __future__ import annotations

import os
import sys

import readline

from pathlib import Path


from .agent_config import AgentParams, Message
from .agent_loop import AgentLoop, step_to_messages
from .approval import ConsoleApprovalGate
from .argparse import parse_args
from .contracts import AgentRequest, AgentResponse, StopReason
from .provider import OpenAICompatibleModelClient
from .tools import ToolExecutor



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

    # 装配主循环：LLM 用 OpenAI SDK 客户端；工具用注册表执行器（工作目录内）；
    # 每次工具调用前经命令行审批闸门询问用户 y/n。
    client = OpenAICompatibleModelClient(agent_params)
    loop = AgentLoop(
        agent_params,
        llm=client,
        tools=ToolExecutor(agent_params.cwd),
        approval_gate=ConsoleApprovalGate(),
    )

    # 两种模式：--one_shot 调用一次；否则进入交互式多轮。
    if args.one_shot:
        return _one_shot(loop)
    return _repl(loop)


def _one_shot(loop: AgentLoop) -> int:
    """one_shot 模式：只跑一次 ReAct 任务，提示词由用户直接输入。"""
    try:
        prompt = input("你：").strip()
    except EOFError:
        prompt = ""

    if not prompt:
        print("没有输入内容，已退出。", file=sys.stderr)
        return 1

    response = loop.run(AgentRequest(user_input=prompt))
    return _print_response(response)


def _repl(loop: AgentLoop) -> int:
    """交互式多轮对话：每轮驱动一次主循环，跨轮历史由本层持有。"""
    history: list[Message] = []

    while True:
        try:
            user_input = input("你：").strip()
        except EOFError:
            print("已退出。")
            return 0

        if user_input in ("/exit", "/quit"):
            print("已退出。")
            return 0

        if not user_input:
            print("没有输入内容，请重新输入。")
            continue

        response = loop.run(
            AgentRequest(user_input=user_input, messages=history or None)
        )
        _print_response(response)

        # 从响应轨迹重建消息（与主循环同一来源 step_to_messages），
        # 供下一轮作为会话历史。
        history.append(Message(role="user", content=user_input))
        for step in response.steps:
            history.extend(step_to_messages(step))


def _print_response(response: AgentResponse) -> int:
    """按结束原因打印主循环结果，返回进程退出码。"""
    if response.stop_reason is StopReason.FINAL_ANSWER:
        print(f"🤖 {response.final_answer}")
        return 0
    if response.stop_reason is StopReason.MAX_TURNS:
        print(f"⚠️ 达到轮数上限（{response.turns_used} 轮），已停止。")
        return 1
    print(f"❌ 出错：{response.error}")
    return 1
