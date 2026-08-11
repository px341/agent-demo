"""ReAct 主循环：一次任务请求驱动「思考→行动→观察」循环，返回最终结果。

输入 / 输出契约见 contracts.py。依赖通过 Protocol 注入：

- ``llm``：LLMClient（必需）—— 唯一的模型调用入口；
- ``tools``：ToolExecutor（可选，None 表示未启用工具）；
- ``memory``：MemoryStore（可选，None 表示未启用记忆）。

工具执行、记忆、上下文压缩的具体实现后续各自接入，主循环不感知。
"""
from __future__ import annotations

import json
from pathlib import Path

from .actions import FinalAnswer, Retry, ToolCall, parse_action
from .agent_config import AgentParams, Message
from .api_config import DEFAULT_SYSTEM_PROMPT
from .contracts import (
    AgentRequest,
    AgentResponse,
    ApprovalGate,
    LLMClient,
    LLMResponse,
    MemoryStore,
    StepRecord,
    StopReason,
    ToolExecutor,
)
from .environment import build_environment_prompt
from .tools.render import render_tool_section


def step_to_messages(step: StepRecord) -> list[Message]:
    """把一步轨迹重建为 LLM 消息（assistant + 可选反馈）。

    - ToolCall：assistant 携带原生 tool_calls 声明（id=call_{turn}），
      工具结果以 role=tool + tool_call_id 回灌；OpenAI 兼容 API 要求
      tool 消息必须配对前置 assistant 消息的 tool_calls 声明；
    - Retry：输出不合法的反馈以 role=user 回灌（不伪造工具调用）；
    - FinalAnswer：仅 assistant 消息。
    """

    if isinstance(step.action, ToolCall):
        return [
            Message(
                role="assistant",
                content=step.raw_output,
                metadata=dict(step.assistant_metadata),
                tool_calls=[
                    {
                        "id": f"call_{step.turn}",
                        "type": "function",
                        "function": {
                            "name": step.action.name,
                            "arguments": json.dumps(
                                step.action.args, ensure_ascii=False
                            ),
                        },
                    }
                ],
            ),
            Message(
                role="tool",
                content=step.observation or "",
                tool_call_id=f"call_{step.turn}",
            ),
        ]
    if isinstance(step.action, Retry):
        return [
            Message(
                role="assistant",
                content=step.raw_output,
                metadata=dict(step.assistant_metadata),
            ),
            Message(role="user", content=step.observation or ""),
        ]
    return [
        Message(
            role="assistant",
            content=step.raw_output,
            metadata=dict(step.assistant_metadata),
        )
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
        approval_gate: ApprovalGate | None = None,
    ):
        self.agent_params = agent_params
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.approval_gate = approval_gate
        self.cwd = Path(agent_params.cwd).resolve()
        self.system_prompt = self._load_system_prompt()

    def _build_context_prompt(self) -> str:
        """拼接本次请求的系统提示：动态环境 prompt + 静态工具 prompt。

        每次 run 重新生成，保证 workspace tree 反映最新工作区状态。
        """
        environment = build_environment_prompt(
            self.agent_params.prompt_dir,
            self.cwd,
        )
        if not environment:
            return self.system_prompt
        return environment + "\n\n" + self.system_prompt

    def _load_system_prompt(self) -> str:
        """从 prompt_dir 加载系统提示词；文件缺失时回退默认提示词。

        ``{tool_list}`` 占位符替换为注册表动态生成的工具列表，
        保证 prompt 工具清单与 TOOLS 注册表始终一致。
        """
        prompt_file = Path(self.agent_params.prompt_dir) / "tools_system_prompt.md"
        try:
            text = prompt_file.read_text(encoding="utf-8")
        except OSError:
            return DEFAULT_SYSTEM_PROMPT
        return text.replace("{tool_list}", render_tool_section())

    def run(self, request: AgentRequest) -> AgentResponse:
        """执行一次任务请求，返回最终结果与逐步轨迹。

        每轮流程：

        1. ``llm.complete(messages)`` 得到原始输出；
        2. ``parse_action`` 解析为 Action：
           - FinalAnswer → 正常返回；
           - ToolCall → 通过注入的 tools 执行（未注入则回灌错误观察）；
           - Retry → 把原因作为观察回灌给模型继续；
        3. 超过 max_turns → 以 MAX_TURNS 收口；
        4. LLM 异常 → 以 ERROR 收口。
        """
        max_turns = (
            request.max_turns
            if request.max_turns is not None
            else self.agent_params.max_turns
        )
        # 轮数上限至少为 1，保证返回值契约 turns_used >= 1。
        max_turns = max(1, max_turns)

        # 本次请求的消息上下文：系统提示（环境 + 工具） + 会话历史 + 当前请求。
        # 不修改调用方传入的历史（extend 的是新列表）。
        messages: list[Message] = [
            Message(role="system", content=self._build_context_prompt())
        ]
        if request.messages:
            messages.extend(request.messages)
        messages.append(Message(role="user", content=request.user_input))

        steps: list[StepRecord] = []
        tool_calls = 0

        for turn in range(1, max_turns + 1):
            try:
                result: LLMResponse = self.llm.complete(
                    messages,
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

            raw_output = result.text
            action = parse_action(raw_output)

            if isinstance(action, FinalAnswer):
                steps.append(
                    StepRecord(
                        turn=turn,
                        raw_output=raw_output,
                        action=action,
                        assistant_metadata=result.metadata,
                    )
                )
                return AgentResponse(
                    final_answer=action.text,
                    stop_reason=StopReason.FINAL_ANSWER,
                    turns_used=turn,
                    tool_calls=tool_calls,
                    steps=steps,
                )

            if isinstance(action, ToolCall):
                # 审批闸门：工具调用前询问用户，拒绝则不执行、不计调用次数。
                if self.approval_gate is not None and not self.approval_gate.request(
                    action.name, action.args
                ):
                    observation = (
                        f"错误：用户拒绝了工具调用 {action.name!r}；"
                        "请改用其他方式或直接给出最终答案。"
                    )
                else:
                    observation = self._execute_tool(action)
                    if self.tools is not None:
                        tool_calls += 1
                step = StepRecord(
                    turn=turn,
                    raw_output=raw_output,
                    action=action,
                    observation=observation,
                    assistant_metadata=result.metadata,
                )
                steps.append(step)
                messages.extend(step_to_messages(step))
                continue

            # Retry：把原因作为 user 反馈回灌，模型重试，不消耗工具计数。
            reason = action.reason or "输出不符合契约"
            observation = (
                f"输出不符合契约，请重新输出合法 JSON。原因：{reason}"
            )
            step = StepRecord(
                turn=turn,
                raw_output=raw_output,
                action=action,
                observation=observation,
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

    def _execute_tool(self, call: ToolCall) -> str:
        """执行一次工具调用并返回观察结果字符串；任何失败都转成错误文本。"""
        if self.tools is None:
            return (
                f"错误：当前未启用工具，无法调用 {call.name!r}；"
                "请直接给出最终答案。"
            )
        try:
            return str(self.tools.execute(call.name, call.args))
        except Exception as exc:
            return f"错误：工具 {call.name} 执行失败：{exc}"
