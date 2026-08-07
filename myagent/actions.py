"""模型输出解析：把模型输出转为结构化 Action（ToolCall / FinalAnswer / Retry）。

输出契约（JSON，由系统提示词约束模型按此输出，一次一个 JSON 对象）：

    {"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}
    {"action": "final", "answer": "这里是最终答案"}

- action=tool_call → ToolCall（必须带 tool 与 args 对象）；
- action=final      → FinalAnswer（必须带 answer）；
- action=retry      → Retry（可选 reason）；
- 无法解析为合法 JSON、字段缺失或 action 未知 → Retry。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型请求调用某个工具。"""

    name: str
    args: dict[str, Any]
    raw: str


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """模型的最终答案。"""

    text: str


@dataclass(frozen=True, slots=True)
class Retry:
    """输出不符合契约，需要让模型重试。"""

    reason: str = ""


Action = ToolCall | FinalAnswer | Retry


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从文本中提取第一个完整 JSON 对象（容忍前后杂文/代码块）。"""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_action(text: str) -> Action:
    """从模型输出中提取结构化 Action。"""
    text = (text or "").strip()
    if not text:
        return Retry("模型输出为空")

    obj = _extract_json_object(text)
    if obj is None:
        return Retry(f"输出不是合法 JSON：{text[:80]!r}")

    action = obj.get("action")
    if action == "tool_call":
        tool = obj.get("tool")
        args = obj.get("args")
        if not tool:
            return Retry(f"tool_call 缺少 tool 字段：{obj!r}")
        if not isinstance(args, dict):
            return Retry(f"args 必须是 JSON 对象：{args!r}")
        return ToolCall(name=str(tool), args=args, raw=text)

    if action == "final":
        answer = obj.get("answer")
        if answer is None:
            return Retry(f"final 缺少 answer 字段：{obj!r}")
        return FinalAnswer(str(answer).strip())

    if action == "retry":
        return Retry(str(obj.get("reason", "")))

    return Retry(f"未知 action 类型：{action!r}")
