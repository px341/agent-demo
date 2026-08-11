"""工具列表 → prompt 文本渲染。

主循环加载 tools_system_prompt.md 时把 ``{tool_list}`` 占位符替换为
本模块生成的「可用工具」段，工具列表单一来源为注册表，
彻底消除「prompt 手写工具列表与注册表漂移」的问题。
"""
from __future__ import annotations

import json

from .registry import TOOLS


def render_tool_section() -> str:
    """按注册表生成「可用工具」markdown 段（含 JSON 调用示例）。"""
    blocks: list[str] = []
    for name in sorted(TOOLS):
        spec = TOOLS[name]
        example = {
            "action": "tool_call",
            "tool": name,
            "args": _example_args(spec.parameters),
        }
        blocks.extend(
            [
                f"### {name}",
                spec.description or "",
                "```",
                json.dumps(example, ensure_ascii=False),
                "```",
                "",
            ]
        )
    return "\n".join(blocks).rstrip()


def _example_args(parameters: dict) -> dict:
    """按参数 schema 生成示例参数值（只含必填参数）。"""
    example: dict = {}
    for name, schema in parameters.items():
        if schema.get("required"):
            example[name] = _sample_value(schema)
    return example


def _sample_value(schema: dict):
    """按 schema 生成一个示例值：优先 default，其次按类型取占位。"""
    if "default" in schema:
        return schema["default"]
    kind = schema.get("type")
    if kind == "boolean":
        return False
    if kind == "integer":
        return 1
    if kind == "array":
        return []
    if kind == "object":
        return {}
    # 字符串：用 <参数名> 作占位，提示这是待填值。
    return "<value>"
