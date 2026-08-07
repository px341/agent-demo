"""agent 的参数。

支持 `#sym:<名称>` 符号引用：把 `#sym:system_prompt` 之类的标记展开为
`prompts/` 目录下对应提示词文件的完整内容，方便在配置里直接引用已有 prompt 文件。
"""
from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "prompts"

# 匹配 #sym:xxx 形式的符号引用，名称允许字母、数字、下划线、斜杠、点与短横线。
_SYM_PATTERN = re.compile(r"#sym:([A-Za-z0-9_./-]+)")


def _load_prompt(name: str) -> str:
    """按名称读取提示词文件；相对路径基于 prompts/ 目录，自动补全 .md 后缀。"""
    path = Path(name)
    if not path.is_absolute():
        path = BASE_DIR / path
    if path.suffix == "":
        path = path.with_suffix(".md")
    return path.read_text(encoding="utf-8")


def expand_syms(text: str) -> str:
    """把字符串中的 `#sym:<名称>` 展开为对应提示词文件的内容。"""
    if not text:
        return text
    return _SYM_PATTERN.sub(lambda m: _load_prompt(m.group(1)), text)


@dataclass(frozen=False, slots=True)
class AgentParams:
    """agent 的参数。"""

    # 工作目录：myagent 运行时的当前工作目录，默认为 "."。
    cwd: str = "."

    # 系统 prompt：可直接写文本，或用 "#sym:system_prompt" 引用 prompts/ 下的文件内容。
    system_prompt: str = None

    # 工具列表：用于存放可用工具的列表，默认为 None。
    tools: list[str] = None

    # 消息历史记录：用于存放消息历史记录的列表，默认为 None。
    messages: list[str] = None

    # 是否可以使用工具：用于指示是否可以使用工具，默认为 False。
    can_use_tools: bool = False

    # token 限制：用于限制消息的最大长度，默认为 128000。
    token_limit: int = 128000

    # 最大轮数：用于限制消息的最大轮数，默认为 10。
    max_turns: int = 10

    def __post_init__(self) -> None:
        # 初始化后展开 #sym:... 符号引用（例如 "#sym:system_prompt"）。
        if self.system_prompt:
            self.system_prompt = expand_syms(self.system_prompt)


