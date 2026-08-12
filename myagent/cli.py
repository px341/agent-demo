from __future__ import annotations

import os
import sys

import readline

from pathlib import Path


from .agent_config import AgentParams, Message
from .agent_loop import AgentLoop, step_to_messages
from .approval import ConsoleApprovalGate
from .argparse import parse_args
from .context import ContextComposer
from .contracts import AgentRequest, AgentResponse, StopReason
from .memory import MemoryManager
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
        memory_dir=args.memory_dir or AgentParams().memory_dir,
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

    # 记忆：--no_memory 禁用；否则创建记忆管理器（会话存档 + 聚合摘要注入）。
    memory = None
    if not args.no_memory:
        memory = MemoryManager(memory_dir=agent_params.memory_dir, llm=client)
        # 启动扫描：汇总上次启动前结束的历史会话（mtime 水位线幂等，
        # 失败自动重试），产出跨会话 summary.md 供本轮注入。
        _sweep_memory(memory)

    # 上下文压缩：--no_compose 禁用；否则按预算压缩输入上下文
    # （system+memory / 本轮 session / 用户输入）。
    composer = None
    if not args.no_compose:
        composer = ContextComposer(
            max_input_tokens=agent_params.max_input_tokens,
            max_output_tokens=agent_params.max_output_tokens,
            max_tool_tokens=agent_params.max_tool_tokens,
            max_total_tool_tokens=agent_params.max_total_tool_tokens,
        )

    loop = AgentLoop(
        agent_params,
        llm=client,
        tools=ToolExecutor(agent_params.cwd),
        approval_gate=ConsoleApprovalGate(),
        memory=memory,
        composer=composer,
    )

    return _repl(loop, memory)


def _sweep_memory(memory: MemoryManager) -> None:
    """启动时汇总历史会话记忆；失败只警告，不阻断启动。

    sweep 内部已做幂等（mtime 水位线）与失败重试（.state retry≤3），
    这里只负责兜底：任何意外异常都不应阻止用户进入会话。
    """
    print("正在汇总历史会话记忆…")
    try:
        memory.sweep()
    except Exception as exc:
        print(f"⚠️ 历史会话汇总失败（不影响本次会话）：{exc}", file=sys.stderr)


def _repl(loop: AgentLoop, memory: MemoryManager | None = None) -> int:
    """交互式多轮对话：每轮驱动一次主循环，跨轮历史由本层持有。

    启用记忆时，每轮把 user 输入与轨迹消息（与 history 同一来源
    step_to_messages）逐条追加进会话存档；退出路径（/exit、Ctrl-D）
    标记会话可汇总，聚合摘要由下次启动的 sweep 统一完成。
    """
    history: list[Message] = []

    while True:
        try:
            user_input = input("你：").strip()
        except EOFError:
            print("已退出。")
            if memory is not None:
                memory.close_session()
            return 0

        if user_input in ("/exit", "/quit"):
            print("已退出。")
            if memory is not None:
                memory.close_session()
            return 0

        if not user_input:
            print("没有输入内容，请重新输入。")
            continue

        if memory is not None:
            memory.append_message(Message(role="user", content=user_input))
        response = loop.run(
            AgentRequest(user_input=user_input, messages=history or None)
        )
        if memory is not None:
            _archive_steps(memory, response)
        _print_response(response)

        # 从响应轨迹重建消息（与主循环同一来源 step_to_messages），
        # 供下一轮作为会话历史；超出上限时配对安全裁剪。
        history.append(Message(role="user", content=user_input))
        for step in response.steps:
            history.extend(step_to_messages(step))
        history = _trim_history(history)


def _archive_steps(memory: MemoryManager, response: AgentResponse) -> None:
    """把一次响应的轨迹消息追加进会话存档（与历史重建同一来源）。"""
    for step in response.steps:
        for message in step_to_messages(step):
            memory.append_message(message)


#: REPL 持有的会话历史消息数上限（超出后保留最近，配对被安全裁剪）。
MAX_HISTORY_MESSAGES = 300


def _trim_history(history: list[Message], limit: int = MAX_HISTORY_MESSAGES) -> list[Message]:
    """按消息数上限裁剪 REPL 会话历史，保证不产生孤儿消息。

    规则：
    1. 超限时保留最近 ``limit`` 条；
    2. 开头若残留孤儿 tool 结果（role=tool 但前面没有配对的
       assistant tool_calls 声明）→ 继续删；
    3. 结尾若残留孤儿 tool_calls 声明（role=assistant 带 tool_calls 但
       配对结果被截掉）→ 删除该声明。
    以上保证裁剪后的历史对 OpenAI 兼容端点合法（tool 消息必须配对
    前置的 assistant tool_calls 声明）。
    """
    if len(history) <= limit:
        return history

    kept = history[-limit:]

    # 2. 修复开头孤儿 tool 结果：从头删到第一个非 tool 消息。
    start = 0
    while start < len(kept) and kept[start].role == "tool":
        start += 1
    kept = kept[start:]

    # 3. 修复结尾孤儿 tool_calls 声明：末尾带 tool_calls 的 assistant
    #    若无配对 tool 结果则删除（它后面的工具结果已被截掉）。
    while kept and kept[-1].role == "assistant" and kept[-1].tool_calls:
        kept = kept[:-1]

    return kept


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
