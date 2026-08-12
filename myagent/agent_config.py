"""agent 的参数。

`AgentParams.system_prompt` 可写提示词文件名：赋值时会自动读取
`prompts/` 目录下对应文件（自动补 .md 后缀）的完整内容。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MEMORY_DIR = BASE_DIR / "memories"


@dataclass(frozen=False, slots=True)
class AgentParams:
    """agent 的参数。"""

    # 工作目录：myagent 运行时的当前工作目录，默认为 "."。
    cwd: str = "."

    # token 限制：用于限制消息的最大长度，默认为 128000。
    max_input_tokens: int = 128000

    max_output_tokens: int = 16392

    # 最大轮数：用于限制消息的最大轮数，默认为 15。
    max_turns: int = 15

    # 上下文压缩：单条 tool 输出的 token 上限（超出裁剪为开头+结尾各半）。
    max_tool_tokens: int = 4000

    # 上下文压缩：历史中所有 tool 输出的总 token 上限（超出丢弃最早结果）。
    max_total_tool_tokens: int = 30000

    # 记忆存档目录：会话原文 jsonl + 摘要 + .state 水位线；空串表示不启用记忆。
    memory_dir: str = MEMORY_DIR

    # system_prompt 的路径
    prompt_dir: str = PROMPT_DIR


Role = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]


@dataclass(slots=True)
class Message:
    """
    Agent内部统一消息格式。
    """

    role: Role

    content: str | None = None

    # 工具调用信息
    tool_calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    # 工具返回结果
    tool_call_id: str | None = None

    # 扩展字段
    metadata: dict[str, Any] = field(
        default_factory=dict
    )