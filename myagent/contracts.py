"""主循环的输入/输出契约与依赖注入接口。

主循环（AgentLoop）只依赖本文件中的类型，不感知任何具体实现：

- 输入契约：AgentRequest —— 一次任务请求；
- 返回值契约：AgentResponse / StepRecord / StopReason —— 一次请求的最终结果；
- 依赖接口：LLMClient（必需）、ToolExecutor / MemoryStore（可选注入）。

工具执行、记忆、上下文压缩的具体实现后续分别放入 tools.py / memory 模块 /
context_manager.py，只要满足对应 Protocol 即可接入，主循环代码无需改动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .actions import Action
from .agent_config import Message


class LLMClient(Protocol):
    """模型推理接口（唯一必需依赖）。

    ``complete()`` 接收完整消息列表，返回结构化 ``LLMResponse``：
    文本交给主循环用 ``actions.parse_action`` 解析；
    metadata（如 DeepSeek thinking mode 的 reasoning_content）
    由主循环随 assistant 消息回传，保证多轮调用合法。
    """

    def complete(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
    ) -> LLMResponse: ...


@dataclass(slots=True)
class LLMResponse:
    """模型一次调用的结构化返回。

    - ``text``：模型输出文本（主循环用于解析 Action）；
    - ``metadata``：需要随 assistant 消息原样回传的附加字段
      （如 DeepSeek thinking mode 的 ``reasoning_content``），
      缺失会导致真实端点 400。
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolExecutor(Protocol):
    """工具执行器（可选依赖；None 表示未启用工具）。

    注入示例：tools.py 的 execute_tool 是模块级函数，需要适配成
    带 ``execute`` 方法的对象：

        from types import SimpleNamespace
        tools = SimpleNamespace(execute=execute_tool)
    """

    def execute(self, name: str, args: dict[str, Any]) -> str: ...


class MemoryStore(Protocol):
    """记忆读写（可选依赖；解耦接缝，具体实现后续加入）。"""

    def retrieve(self, query: str) -> list[str]: ...

    def store(self, entry: str) -> None: ...


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

    #: 模型给出最终答案，正常收口。
    FINAL_ANSWER = "final_answer"

    #: 达到轮数上限（模型始终未给出 final）。
    MAX_TURNS = "max_turns"

    #: LLM 调用抛出异常。
    ERROR = "error"


@dataclass(slots=True)
class StepRecord:
    """ReAct 一步的完整轨迹：模型原始输出、解析后的 Action、观察结果。

    供日志 / 调试 / 审计使用，不参与模型推理。
    """

    #: 从 1 开始的推理轮数。
    turn: int

    #: 模型原始输出文本。
    raw_output: str

    #: 解析后的 Action（ToolCall / FinalAnswer / Retry）。
    action: Action

    #: 工具执行结果或错误提示；tool_call / retry 轮次为非 None。
    observation: str | None = None

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
