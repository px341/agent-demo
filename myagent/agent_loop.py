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
memory 的聚合摘要经 ``context_block(query=...)`` 前置进 system prompt，
query 为当前请求文本（记忆实现按相关性检索注入相关会话摘要）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .agent_config import AgentParams, Message
from .api_config import DEFAULT_SYSTEM_PROMPT
from .contracts import (
    AgentRequest,
    AgentResponse,
    ApprovalGate,
    CallOutcome,
    ContextComposer,
    LLMClient,
    LLMResponse,
    MemoryStore,
    StepRecord,
    StopReason,
    ToolExecutor,
)
from .environment import build_environment_prompt
from .errors import ToolError
from .tools.registry import to_openai_tools


def step_to_messages(step: StepRecord) -> list[Message]:
    """把一步轨迹重建为 LLM 消息（assistant + 工具结果回灌）。

    与主循环发给 LLM 的轮内消息同构：
    - 工具轮：assistant 携带原生 tool_calls 声明（id 由模型返回），
      每个声明都配一条 role=tool 消息（OpenAI 兼容 API 要求 tool 消息
      必须配对前置 assistant 消息的 tool_calls 声明，缺配对会 400）。
      未执行（skipped）的调用生成占位错误文本，保证无孤儿声明；
    - 最终答案轮：仅 assistant 消息（text 即答案）。

    tool 消息把 error_type / partial 放进 metadata，供 memory 存档
    结构化记录（archive.py 从 metadata 提取 error_type）。
    """

    if not step.tool_calls:
        return [
            Message(
                role="assistant",
                content=step.raw_output,
                metadata=dict(step.assistant_metadata),
            )
        ]

    outcomes = step.outcomes or []
    tool_messages: list[Message] = []
    for index, call in enumerate(step.tool_calls):
        call_id = (call or {}).get("id")
        outcome = outcomes[index] if index < len(outcomes) else None
        if outcome is not None:
            observation = outcome.observation
            metadata: dict = {}
            if outcome.error_type:
                metadata["error_type"] = outcome.error_type
            if outcome.partial:
                metadata["partial"] = True
        else:
            # 无 outcome（防御）：fallback 到 observations 或占位。
            observations = step.observations
            observation = (
                observations[index] if index < len(observations) else ""
            )
            metadata = {}
        tool_messages.append(
            Message(
                role="tool",
                content=observation or "",
                tool_call_id=call_id,
                metadata=metadata,
            )
        )
    return [
        Message(
            role="assistant",
            content=step.raw_output,
            metadata=dict(step.assistant_metadata),
            tool_calls=list(step.tool_calls),
        ),
        *tool_messages,
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
        tools_schema: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.agent_params = agent_params
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.composer = composer
        self.approval_gate = approval_gate
        self.cwd = Path(agent_params.cwd).resolve()
        self.system_prompt = self._load_system_prompt()
        #: 每轮发给 LLM 的工具 schema 提供者；None 时用全局注册表全量渲染。
        #: 多 agent 场景按角色裁剪（如工人不暴露 delegate_agent）时覆盖。
        self._tools_schema = tools_schema or to_openai_tools

    def _build_context_prompt(self, user_input: str) -> str:
        """拼接本次请求的系统提示：跨会话记忆 + 动态环境 prompt + 静态工具 prompt。

        每次 run 重新生成，保证 workspace tree 反映最新工作区状态。
        记忆段仅在注入 MemoryStore 且聚合摘要非空时前置；传入当前请求
        作为 query，由记忆实现按相关性检索注入相关会话摘要。
        """
        memory_block = ""
        if self.memory is not None:
            memory_block = self.memory.context_block(query=user_input)

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
        system = self._build_context_prompt(request.user_input)
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
                    tools=self._tools_schema(),
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
            calls: list[tuple[str, str, dict]] = []
            for call in result.tool_calls:
                call_id = (call or {}).get("id")
                function = (call or {}).get("function") or {}
                name = function.get("name")
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append((call_id, name, args))

            observations, outcomes, fatal = self._execute_tool_batch(calls)
            # 计数「实际分发到执行器的调用」：无工具场景不计入。
            if self.tools is not None:
                tool_calls += sum(1 for o in outcomes if o.status != "skipped")

            step = StepRecord(
                turn=turn,
                raw_output=result.text,
                tool_calls=list(result.tool_calls),
                observations=observations,
                outcomes=outcomes,
                assistant_metadata=result.metadata,
            )
            steps.append(step)
            messages.extend(step_to_messages(step))

            if fatal is not None:
                # 不可恢复错误：终止本轮 ReAct 循环。
                # step 已 append 进 steps 并重建进 messages，
                # 保证 REPL history / memory 存档记录了完整
                # assistant(tool_calls) + tool(错误) 配对。
                return AgentResponse(
                    final_answer=None,
                    stop_reason=StopReason.TOOL_ERROR,
                    turns_used=turn,
                    tool_calls=tool_calls,
                    steps=steps,
                    error=fatal,
                )

        # 循环自然结束 = 达到轮数上限。
        return AgentResponse(
            final_answer=None,
            stop_reason=StopReason.MAX_TURNS,
            turns_used=max_turns,
            tool_calls=tool_calls,
            steps=steps,
        )

    def _execute_tool_batch(
        self, calls: list[tuple[str, dict]]
    ) -> tuple[list[str], list[CallOutcome], str | None]:
        """执行一批工具调用，返回 (观察文本, 结构化结果, 致命错误文本)。

        审批闸门存在时整批询问一次，拒绝 → 全部 skipped + ApprovalDenied（致命）。
        工具抛 ToolError：
        - recoverable（Validation/NotFound）→ 记 error outcome，继续循环；
        - 不可恢复（Permission/Timeout/Execution/Approval）→ 记 error outcome，
          整批终止：之前已成功的调用打 partial 标记，剩余调用记 skipped。

        结构化结果 CallOutcome 供 memory 存档与 summary 消费：
        错误带 error_type，部分执行带 partial=True。
        """
        if self.tools is None:
            prefix = "ToolError[ExecutionError]: 当前未启用工具，无法调用"
            outcomes = [
                CallOutcome(
                    tool_call_id=call_id,
                    status="error",
                    observation=f"{prefix} {name!r}；请直接给出最终答案。",
                    error_type="ExecutionError",
                )
                for call_id, name, _ in calls
            ]
            return (
                [o.observation for o in outcomes],
                outcomes,
                None,
            )

        if self.approval_gate is not None and not self.approval_gate.request_batch(
            [(name, args) for _, name, args in calls]
        ):
            from .errors import ApprovalDenied

            message = str(ApprovalDenied("用户拒绝了这批工具调用"))
            outcomes = [
                CallOutcome(
                    tool_call_id=call_id,
                    status="skipped",
                    observation=message,
                    error_type="ApprovalDenied",
                )
                for call_id, _, _ in calls
            ]
            return [o.observation for o in outcomes], outcomes, message

        outcomes: list[CallOutcome] = []
        for index, (call_id, name, args) in enumerate(calls):
            try:
                observation = str(self.tools.execute(name, args))
                outcomes.append(
                    CallOutcome(
                        tool_call_id=call_id,
                        status="success",
                        observation=observation,
                    )
                )
            except ToolError as exc:
                message = str(exc)
                outcomes.append(
                    CallOutcome(
                        tool_call_id=call_id,
                        status="error",
                        observation=message,
                        error_type=exc.error_type,
                    )
                )
                if not exc.recoverable:
                    # 不可恢复：之前已成功的调用打 partial，剩余 skipped。
                    for prior in outcomes:
                        if prior.status == "success":
                            prior.partial = True
                    for _ in range(index + 1, len(calls)):
                        outcomes.append(
                            CallOutcome(
                                tool_call_id=None,
                                status="skipped",
                                observation=(
                                    "ToolError[ExecutionError]: 未执行"
                                    "（批被后续致命错误中断）"
                                ),
                                error_type="ExecutionError",
                            )
                        )
                    return (
                        [o.observation for o in outcomes],
                        outcomes,
                        message,
                    )
            except Exception as exc:
                message = f"ToolError[ExecutionError]: 工具 {name} 执行失败：{exc}"
                outcomes.append(
                    CallOutcome(
                        tool_call_id=None,
                        status="error",
                        observation=message,
                        error_type="ExecutionError",
                    )
                )
                for prior in outcomes:
                    if prior.status == "success":
                        prior.partial = True
                for _ in range(index + 1, len(calls)):
                    outcomes.append(
                        CallOutcome(
                            tool_call_id=None,
                            status="skipped",
                            observation=(
                                "ToolError[ExecutionError]: 未执行"
                                "（批被后续致命错误中断）"
                            ),
                            error_type="ExecutionError",
                        )
                    )
                return [o.observation for o in outcomes], outcomes, message
        return [o.observation for o in outcomes], outcomes, None
