"""token 计数：tiktoken cl100k_base 惰性加载（离线可用）。

复用被删 context_manager.py（9748ddc / fa65185）的 tiktoken 先例：
编码器首次使用懒加载并缓存，未调用时零开销。

计数方式：按消息的 role / content / tool_calls / tool_call_id 分项计数，
比旧版「整段 JSON 序列化」更接近真实请求 token 数。
"""
from __future__ import annotations

import json

import tiktoken

#: 编码器缓存（懒加载）。
_encoding = None


def _get_encoding():
    """获取 cl100k_base 编码器（首次调用时加载并缓存）。"""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str | None) -> int:
    """统计一段文本的 token 数；None / 空串为 0。"""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def count_message_tokens(message) -> int:
    """统计一条消息的 token 数（role + content + tool_calls + tool_call_id）。"""
    total = count_tokens(message.role)
    total += count_tokens(message.content)
    if message.tool_calls:
        total += count_tokens(json.dumps(message.tool_calls, ensure_ascii=False))
    total += count_tokens(message.tool_call_id)
    return total


def truncate_to_tokens(text: str, budget: int) -> str:
    """按 token 预算截断文本尾部；超出时保留开头 + 裁剪标记。"""
    if budget <= 0:
        return ""
    tokens = _get_encoding().encode(text)
    if len(tokens) <= budget:
        return text
    return _get_encoding().decode(tokens[:budget]) + "\n…[已裁剪]"


def clip_head_tail(text: str, budget: int) -> str:
    """按 token 预算裁剪文本为开头 + 结尾各半，中间省略并加标记。

    保留的都是原文字符串（按 token 边界切割），不改写内容；
    工具输出裁剪用（单条超限时），头尾信息对模型都关键。
    """
    if budget <= 0:
        return ""
    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= budget:
        return text
    half = max(1, budget // 2)
    head = encoding.decode(tokens[:half])
    tail = encoding.decode(tokens[-half:])
    return head + f"\n…[已裁剪：原 {len(tokens)} token]…\n" + tail
