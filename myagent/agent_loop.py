"""ReAct 主循环：一次任务请求驱动「思考→行动→观察」循环，返回最终结果。

输入 / 输出契约见 contracts.py。模型输出走 OpenAI 原生 tool_calls 协议：
每轮 ``llm.complete(messages, tools=...)`` 返回结构化 ``LLMResponse``，
主循环按 ``tool_calls`` 执行（并行多调用一次执行），结果以 role=tool
消息回灌；无 tool_calls 的轮即最终答案。

依赖通过 Protocol 注入：

- ``llm``：LLMClient（必需）—— 唯一的模型调用入口；
- ``tools``：ToolExecutor（可选，None 表示未启用工具）；
- ``memory``：MemoryStore（可选，None 表示未启用记忆）；
- ``composer``：ContextComposer（可选，None 表示不压缩上下文）。

工具执行、记忆、上下文压缩的具体实现各自接入，主循环只感知注入接口：
memory 的聚合摘要经 ``context_block()`` 前置进 system prompt。
"""
from __future__ import annotations

import json
from pathlib import Path

from .agent_config import AgentParams, Message
from .api_config import DEFAULT_SYSTEM_PROMPT
from .contracts import (
    AgentRequest,
    AgentResponse,
    ApprovalGate,
    ContextComposer,
    LLMClient,
    LLMResponse,
    MemoryStore,
    StepRecord,
    StopReason,
    ToolExecutor,
)
from .environment import build_environment_prompt
from .tools.registry import to_openai_tools


def step_to_messages(step: StepRecord) -> list[Message]:
    """把一步轨迹重建为 LLM 消息（assistant + 工具结果回灌）。

    与主循环发给 LLM 的轮内消息同构：
    - 工具轮：assistant 携带原生 tool_calls 声明（id 由模型返回），
      每个工具结果以 role=tool + 对应 tool_call_id 回灌（OpenAI 兼容 API
      要求 tool 消息必须配对前置 assistant 消息的 tool_calls 声明）；
    - 最终答案轮：仅 assistant 消息（text 即答案）。
    """

    if not step.tool_calls:
        return [
            Message(
                role="assistant",
                content=step.raw_output,
                metadata=dict(step.assistant_metadata),
            )
        ]
    return [
        Message(
            role="assistant",
            content=step.raw_output,
            metadata=dict(step.assistant_metadata),
            tool_calls=list(step.tool_calls),
        ),
        *[
            Message(
                role="tool",
                content=observation or "",
                tool_call_id=call.get("id"),
            )
            for call, observation in zip(step.tool_calls, step.observations)
        ],
    ]


class AgentLoop:
    """ReAct 主循环。

    典型用法：:

        from myagent.tools import ToolExecutor
        loop = AgentLoop(
            agent_params,
            llm=client,
            tools=ToolExecutor(cwd=agent_params.cwd),
        )
        response = loop.run(AgentRequest(user_input="帮我读 a.py"))
    """

    def __init__(
        self,
        agent_params: AgentParams,
        llm: LLMClient,
        tools: ToolExecutor | None = None,
        memory: MemoryStore | None = None,
        composer: ContextComposer | None = None,
        approval_gate: ApprovalGate | None = None,
    ):
        self.agent_params = agent_params
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.composer = composer
        self.approval_gate = approval_gate
        self.cwd = Path(agent_params.cwd).resolve()
        self.system_prompt = self._load_system_prompt()

    def _build_context_prompt(self) -> str:
        """拼接本次请求的系统提示：跨会话记忆 + 动态环境 prompt + 静态工具 prompt。

        每次 run 重新生成，保证 workspace tree 反映最新工作区状态。
        记忆段仅在注入 MemoryStore 且聚合摘要非空时前置。
        """
        memory_block = ""
        if self.memory is not None:
            memory_block = self.memory.context_block()

        environment = build_environment_prompt(
            self.agent_params.prompt_dir,
            self.cwd,
        )
        parts = []
        if memory_block:
            parts.append(memory_block)
        if environment:
            parts.append(environment)
        if not parts:
            return self.system_prompt
        return "\n\n".join(parts) + "\n\n" + self.system_prompt

    def _load_system_prompt(self) -> str:
        """从 prompt_dir 加载系统提示词；文件缺失时回退默认提示词。"""
        prompt_file = Path(self.agent_params.prompt_dir) / "tools_system_prompt.md"
        try:
            return prompt_file.read_text(encoding="utf-8")
        except OSError:
            return DEFAULT_SYSTEM_PROMPT

    def run(self, request: AgentRequest) -> AgentResponse:
        """执行一次任务请求，返回最终结果与逐步轨迹。

        每轮流程：

        1. ``llm.complete(messages, tools=...)`` 得到结构化返回；
        2. 有 ``tool_calls`` → 循环执行全部（并行，审批用 request_batch），
           观察结果以 role=tool 回灌，进入下一轮；
        3. 无 ``tool_calls`` → ``text`` 即最终答案，正常返回；
        4. 超过 max_turns → 以 MAX_TURNS 收口；
        5. LLM 异常 → 以 ERROR 收口。
        """
        max_turns = (
            request.max_turns
            if request.max_turns is not None
            else self.agent_params.max_turns
        )
        # 轮数上限至少为 1，保证返回值契约 turns_used >= 1。
        max_turns = max(1, max_turns)

        # 本次请求的消息上下文：系统提示（环境 + 工具） + 会话历史 + 当前请求。
        # 不修改调用方传入的历史；注入 composer 时由它按预算压缩三部分，
        # 未注入则原样拼接。
        system = self._build_context_prompt()
        if self.composer is not None:
            messages = self.composer.compose(
                system, list(request.messages or []), request.user_input
            )
        else:
            messages = [Message(role="system", content=system)]
            if request.messages:
                messages.extend(request.messages)
            messages.append(Message(role="user", content=request.user_input))

        steps: list[StepRecord] = []
        tool_calls = 0

        for turn in range(1, max_turns + 1):
            # 每轮发给 LLM 前压缩当前消息列表：tool 结果是轮内 append 进来的，
            # 若不每轮压缩，单条 tool 输出与总量限制会被绕过。
            if self.composer is not None:
                messages = self.composer.recompress(
                    messages, request.user_input
                )
            try:
                result: LLMResponse = self.llm.complete(
                    messages,
                    tools=to_openai_tools(),
                    max_new_tokens=self.agent_params.max_output_tokens,
                )
            except Exception as exc:
                return AgentResponse(
                    final_answer=None,
                    stop_reason=StopReason.ERROR,
                    turns_used=turn,
                    tool_calls=tool_calls,
                    steps=steps,
                    error=str(exc),
                )

            if not result.tool_calls:
                # 无工具调用 = 本轮即最终答案。
                steps.append(
                    StepRecord(
                        turn=turn,
                        raw_output=result.text,
                        final_answer=result.text,
                        assistant_metadata=result.metadata,
                    )
                )
                return AgentResponse(
                    final_answer=result.text,
                    stop_reason=StopReason.FINAL_ANSWER,
                    turns_used=turn,
                    tool_calls=tool_calls,
                    steps=steps,
                )

            # 工具轮：并行执行全部调用（审批用批量闸门）。
            calls: list[tuple[str, dict]] = []
            for call in result.tool_calls:
                function = (call or {}).get("function") or {}
                name = function.get("name")
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append((name, args))

            observations, executed = self._execute_tool_batch(calls)
            tool_calls += executed

            step = StepRecord(
                turn=turn,
                raw_output=result.text,
                tool_calls=list(result.tool_calls),
                observations=observations,
                assistant_metadata=result.metadata,
            )
            steps.append(step)
            messages.extend(step_to_messages(step))

        # 循环自然结束 = 达到轮数上限。
        return AgentResponse(
            final_answer=None,
            stop_reason=StopReason.MAX_TURNS,
            turns_used=max_turns,
            tool_calls=tool_calls,
            steps=steps,
        )

    def _execute_tool_batch(self, calls: list[tuple[str, dict]]) -> tuple[list[str], int]:
        """并行执行一批工具调用，返回 (观察结果列表, 实际执行数)。

        审批闸门存在时整批询问一次，拒绝则整批不执行、不计执行数；
        任何失败都转成错误文本观察。
        """
        if self.tools is None:
            prefix = "错误：当前未启用工具，无法调用"
            return (
                [f"{prefix} {name!r}；请直接给出最终答案。" for name, _ in calls],
                0,
            )

        if self.approval_gate is not None and not self.approval_gate.request_batch(
            calls
        ):
            return (
                [
                    "错误：用户拒绝了这批工具调用；请改用其他方式或直接给出最终答案。"
                    for _ in calls
                ],
                0,
            )

        observations = []
        for name, args in calls:
            try:
                observations.append(str(self.tools.execute(name, args)))
            except Exception as exc:
                observations.append(f"错误：工具 {name} 执行失败：{exc}")
        return observations, len(calls)
