"""token 计数：tiktoken cl100k_base 惰性加载（离线可用）。

复用被删 context_manager.py（9748ddc / fa65185）的 tiktoken 先例：
编码器首次使用懒加载并缓存，未调用时零开销。

计数方式：按消息的 role / content / tool_calls / tool_call_id 分项计数，
比旧版「整段 JSON 序列化」更接近真实请求 token 数。
"""
from __future__ import annotations

import json
import re

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


#: unified diff hunk 起始行的正则（``@@ -12,7 +12,8 @@ ...``）。
_HUNK_RE = re.compile(r"^@@[ \t]")


def clip_diff_hunks(text: str, budget: int) -> str:
    """按 token 预算裁剪 unified diff，保留**完整的 hunk**（不切 hunk 中间）。

    git diff 的输出若用头尾 token 裁剪，会从 hunk 中间切开、把 ``@@`` 头
    变成孤儿，模型看到的是一块拼不起来的碎片。本函数按 ``@@`` hunk 边界
    裁剪：文件头全部保留，hunk 从前往后尽量多保留，再从后往前补保留，
    中间被裁剪的 hunk 用 ``…[已裁剪 N 个 hunk]…`` 标记。

    保留的都是原文字符串，不改写内容。
    """
    if budget <= 0:
        return ""
    encoding = _get_encoding()
    if len(encoding.encode(text)) <= budget:
        return text

    lines = text.split("\n")
    # 切分为块：header（文件头/无 hunk 内容）与 hunk（@@ 起始）。
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] = []
    kind = "header"

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append((kind, current))
            current = []

    for line in lines:
        if line.startswith("diff --git"):
            flush()
            current = [line]
            kind = "header"
        elif _HUNK_RE.match(line):
            flush()
            current = [line]
            kind = "hunk"
        else:
            current.append(line)
    flush()

    header_lines: list[str] = []
    hunks: list[list[str]] = []
    for block_kind, block_lines in blocks:
        if block_kind == "header":
            header_lines.extend(block_lines)
        else:
            hunks.append(block_lines)

    if not hunks:
        # 纯 stat 或非 hunk 内容：退化为头尾裁剪。
        return clip_head_tail(text, budget)

    header_tokens = len(encoding.encode("\n".join(header_lines)))
    budget_left = budget - header_tokens
    if budget_left <= 0:
        return clip_head_tail(text, budget)

    hunk_tokens = [len(encoding.encode("\n".join(h))) for h in hunks]
    kept = [False] * len(hunks)
    used = 0
    for i, t in enumerate(hunk_tokens):
        if used + t <= budget_left:
            kept[i] = True
            used += t
    for i in range(len(hunks) - 1, -1, -1):
        if kept[i]:
            continue
        if used + hunk_tokens[i] <= budget_left:
            kept[i] = True
            used += hunk_tokens[i]

    kept_count = sum(kept)
    skipped_total = len(hunks) - kept_count
    if skipped_total == 0:
        return text  # 预算实际够（token 计数误差回退）

    out = list(header_lines)
    prev_kept_idx = -1
    marker_emitted = False
    for i, (h, is_kept) in enumerate(zip(hunks, kept)):
        if not is_kept:
            continue
        if prev_kept_idx == -1 and i > 0:
            out.append(f"\n…[已裁剪 {i} 个 hunk，保留 {kept_count}/{len(hunks)}]…\n")
            marker_emitted = True
        elif prev_kept_idx >= 0 and i - prev_kept_idx > 1:
            out.append(f"\n…[已裁剪 {i - prev_kept_idx - 1} 个 hunk]…\n")
            marker_emitted = True
        out.extend(h)
        prev_kept_idx = i
    if not marker_emitted and skipped_total:
        out.append(f"\n…[已裁剪 {skipped_total} 个 hunk]…")
    return "\n".join(out)
