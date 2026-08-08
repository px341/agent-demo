"""agent 的参数。

`AgentParams.system_prompt` 可写提示词文件名：赋值时会自动读取
`prompts/` 目录下对应文件（自动补 .md 后缀）的完整内容。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / ".storage"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str | None) -> str | None:
    """按名称读取提示词文件；相对路径基于 prompts/ 目录，自动补全 .md 后缀。

    传入 None 或空字符串时原样返回，便于直接赋给 system_prompt。
    """
    if not name:
        return name
    path = Path(name)
    if not path.is_absolute():
        path = PROMPT_DIR / path
    if path.suffix == "":
        path = path.with_suffix(".md")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=False, slots=True)
class AgentParams:
    """agent 的参数。"""

    # 工作目录：myagent 运行时的当前工作目录，默认为 "."。
    cwd: str = "."

    # 工具列表：用于存放可用工具的列表，默认为 None。
    tools: list[str] = None

    # 消息历史记录：用于存放消息历史记录的列表，默认为 None。
    messages: list[Message] = None

    # 是否可以使用工具：用于指示是否可以使用工具，默认为 False。
    can_use_tools: bool = False

    # token 限制：用于限制消息的最大长度，默认为 128000。
    token_limit: int = 128000

    token_out_limit: int = 16392

    # 最大轮数：用于限制消息的最大轮数，默认为 10。
    max_turns: int = 10

    # 对话历史存储目录：本地永久化保存对话的文件夹路径，默认为 STORAGE_DIR。
    storage_dir: str = STORAGE_DIR

    # system_prompt 的存储字段（不作为 __init__ 参数）。
    _system_prompt: str = field(default=None, init=False, repr=False)

    @property
    def system_prompt(self) -> str | None:
        """系统提示词：赋文件名时自动读取 prompts/ 下对应文件内容。"""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None) -> None:
        self._system_prompt = load_prompt(value)


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