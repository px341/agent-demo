"""主循环的输入/输出契约与依赖注入接口。

主循环（AgentLoop）只依赖本文件中的类型，不感知任何具体实现：

- 输入契约：AgentRequest —— 一次任务请求；
- 返回值契约：AgentResponse / StepRecord / StopReason —— 一次请求的最终结果；
- 依赖接口：LLMClient（必需）、ToolExecutor / MemoryStore / ContextComposer（可选注入）。

模型输出走 OpenAI 原生 tool_calls 协议（tool_calls 数组 + 文本内容），
不再使用自定义 JSON 契约（已移除）。工具执行、记忆、上下文压缩的
具体实现分别放入 tools 包 / memory 包 / context 包，只要满足对应
Protocol 即可接入，主循环代码无需改动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .agent_config import Message


class LLMClient(Protocol):
    """模型推理接口（唯一必需依赖）。

    ``complete()`` 接收完整消息列表与工具描述（OpenAI 格式），返回结构化
    ``LLMResponse``：
    - 有 ``tool_calls`` → 模型请求调用工具（主循环执行后回灌）；
    - 无 ``tool_calls`` → ``text`` 即最终答案。
    metadata（如 DeepSeek thinking mode 的 reasoning_content）由主循环
    随 assistant 消息回传，保证多轮调用合法。
    """

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
    ) -> LLMResponse: ...


@dataclass(slots=True)
class LLMResponse:
    """模型一次调用的结构化返回。

    - ``text``：模型输出的文本内容（无 tool_calls 时为最终答案）；
    - ``tool_calls``：OpenAI 原生工具调用数组
      ``[{"id", "type", "function": {"name", "arguments"}}]``；空表示未调用工具；
    - ``metadata``：需要随 assistant 消息原样回传的附加字段
      （如 DeepSeek thinking mode 的 ``reasoning_content``），
      缺失会导致真实端点 400。
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolExecutor(Protocol):
    """工具执行器（可选依赖；None 表示未启用工具）。

    注入示例：tools 包提供满足该协议的执行器：:

        from myagent.tools import ToolExecutor
        tools = ToolExecutor(cwd=".")
    """

    def execute(self, name: str, args: dict[str, Any]) -> str: ...


class MemoryStore(Protocol):
    """会话记忆（可选依赖；None 表示不启用记忆）。

    实现见 myagent.memory 包（MemoryManager）。主循环只读取注入用的
    ``context_block()``；其余方法是供调用方（REPL 层）在会话生命周期
    内调用：每轮消息追加存档、会话结束标记、下次启动统一扫描汇总。

    实现示例：:

        from myagent.memory import MemoryManager
        memory = MemoryManager(memory_dir=params.memory_dir)
        loop = AgentLoop(params, llm=client, memory=memory)
    """

    #: 会话上下文唯一标识（JSONL 存档 / 状态文件名）。
    session_id: str

    def context_block(self, max_chars: int | None = None) -> str:
        """返回注入 system prompt 的跨会话记忆文本；无记忆或空内容时返回 ""。"""

    def append_message(self, message: Message) -> None:
        """把一条会话消息（脱敏后）追加写入本次会话 JSONL 存档。"""

    def close_session(self) -> None:
        """标记本次会话为「可汇总」；实现应刷新存档并记录状态，不做 LLM 调用。"""

    def sweep(self) -> None:
        """扫描全部会话存档，按水位线汇总单会话摘要并重写跨会话 summary.md。"""


class ContextComposer(Protocol):
    """上下文压缩器（可选依赖；None 表示不压缩）。

    实现见 myagent.context 包（ContextComposer）。主循环在组装完
    system prompt 后调用 ``compose``，由实现按三部分预算压缩：
    - Part 1：system（prompt + 跨会话 memory）；
    - Part 2：本轮会话历史（tool 输出单条裁剪 + 总量丢弃，非 tool 强压缩）；
    - Part 3：用户输入不可压缩。

    实现不修改入参；返回全新消息列表。实现示例：:

        from myagent.context import ContextComposer
        composer = ContextComposer(
            max_input_tokens=params.max_input_tokens,
            max_output_tokens=params.max_output_tokens,
        )
        loop = AgentLoop(params, llm=client, composer=composer)
    """

    def compose(
        self,
        system_prompt: str,
        history: list[Message],
        user_input: str,
    ) -> list[Message]: ...

    def recompress(
        self, messages: list[Message], user_input: str | None = None
    ) -> list[Message]: ...


class ApprovalGate(Protocol):
    """工具调用审批闸门（可选依赖；None 表示不审批、直接执行）。

    主循环在**每轮**执行工具前调用 ``request`` / ``request_batch``：
    - 返回 True  → 允许执行；
    - 返回 False → 拒绝执行，观察文本提示用户拒绝，模型继续推理。
    """

    def request(self, name: str, args: dict[str, Any]) -> bool: ...

    def request_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> bool:
        """一次性审批多个工具调用（并行）；返回是否全部允许。"""


@dataclass(slots=True)
class AgentRequest:
    """一次任务请求：主循环的输入。

    主循环每次只处理一个请求；会话级 REPL、多轮历史由调用方持有，
    通过 ``messages`` 传入。主循环不会修改调用方传入的历史。
    """

    #: 当前用户输入 / 任务描述。
    user_input: str

    #: 已有会话历史；None 表示新会话。
    messages: list[Message] | None = None

    #: 本次请求的轮数上限；None 时使用 AgentParams.max_turns。
    max_turns: int | None = None


class StopReason(str, Enum):
    """主循环结束原因。"""

    #: 模型给出最终答案（无 tool_calls 轮），正常收口。
    FINAL_ANSWER = "final_answer"

    #: 达到轮数上限（模型始终未给出最终答案）。
    MAX_TURNS = "max_turns"

    #: 工具调用触发不可恢复错误（Permission / Timeout / Execution / 审批拒绝）。
    TOOL_ERROR = "tool_error"

    #: LLM 调用抛出异常。
    ERROR = "error"


@dataclass(slots=True)
class CallOutcome:
    """一次工具调用的执行结果（结构化，供 memory / summary 消费）。"""

    #: 本次调用在 assistant tool_calls 中的 id（与 tool 消息配对）。
    tool_call_id: str | None = None

    #: 执行状态：success（成功）/ error（抛 ToolError）/ skipped（未执行）。
    status: str = "success"

    #: 成功输出 / 错误文本 / 未执行占位文本。
    observation: str = ""

    #: ToolError 分类名（error_type），如 PermissionError / TimeoutError；
    #: 成功或未执行时为 None。
    error_type: str | None = None

    #: 部分执行标记：本调用已成功生效，但整批因后续致命错误中断
    #: （副作用留在工作区，跨会话需知悉）。
    partial: bool = False


@dataclass(slots=True)
class StepRecord:
    """ReAct 一步的完整轨迹：模型原始输出、工具调用、观察结果。

    供日志 / 调试 / 审计 / 记忆使用，不参与模型推理。
    """

    #: 从 1 开始的推理轮数。
    turn: int

    #: 模型原始输出文本（无 tool_calls 时为最终答案，即 raw_text）。
    raw_output: str

    #: 本轮模型请求的工具调用（OpenAI 原生结构）；空表示本轮即最终答案。
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    #: 无 tool_calls 时的最终答案文本；None 表示本轮调用了工具。
    final_answer: str | None = None

    #: 工具执行结果文本列表（与 tool_calls 一一对应）；无工具轮为空。
    observations: list[str] = field(default_factory=list)

    #: 每次调用的结构化结果（与 tool_calls 一一对应，含错误类型 / 部分执行标记）。
    outcomes: list[CallOutcome] = field(default_factory=list)

    #: 本轮 assistant 消息的附加元数据（如 reasoning_content），
    #: 历史重建时随 assistant 消息回传。
    assistant_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResponse:
    """一次任务请求的最终结果：主循环的返回值。"""

    #: 最终答案；stop_reason 非 FINAL_ANSWER 时为 None。
    final_answer: str | None

    #: 结束原因。
    stop_reason: StopReason

    #: 实际消耗的推理轮数（>= 1）。
    turns_used: int

    #: 实际分发给工具执行器的调用次数。
    tool_calls: int

    #: 逐步轨迹。
    steps: list[StepRecord] = field(default_factory=list)

    #: stop_reason=ERROR 时的异常信息。
    error: str | None = None
