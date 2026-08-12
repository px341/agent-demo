"""上下文压缩：三部分预算管理（system+memory / 本轮 session / 用户输入）。

对齐设计：
- **Part 3（用户输入）不可压缩**：token 全额预留，content 永不改动；
- **Part 1（prompt + memory）**：正常体量不触发；仅当 system 本身超预算时
  先压缩 memory 段（识别 ``## 跨会话记忆`` 标记），还不够再截 system 尾部；
- **Part 2（本轮 session history）** 压缩分两档：
  - **tool 独立预算（无条件执行，与整体预算正交）**：
    1. 单条 tool 输出超 ``max_tool_tokens`` → 裁剪为开头+结尾各半（保留原文，
       只动 content 字符串，不改 assistant 的 tool_calls 声明与配对）；
    2. tool 总 token 超 ``max_total_tool_tokens`` → 从最早工具组整组丢弃，
       保留最新一组；
  - **整体预算不足时**：
    3. 从最早非 tool 消息丢弃（user 输入 / assistant 纯文本 / 反馈），
       保护带 tool_calls 的 assistant；
    4. 兜底：非 tool 删光仍超 → 才丢最早工具组。

约束：丢弃单元是工具组（assistant 声明 + 其全部配对 tool 结果，原生 FC 下
一个 assistant 可带多个 tool_calls），绝不留下孤儿 tool 消息或孤儿
tool_calls 声明；消息相对顺序保持不变；不修改入参。
"""
from __future__ import annotations

from ..agent_config import Message
from .tokens import clip_head_tail, count_message_tokens, count_tokens, truncate_to_tokens

#: system prompt 中跨会话记忆段的固定标题（与 memory/manager.py 一致）。
_MEMORY_MARKER = "## 跨会话记忆"

#: 压缩后的标记文案。
_CLIPPED_MARK = "…[已裁剪]"


class ContextComposer:
    """按三部分预算压缩输入上下文；返回全新消息列表，不修改入参。"""

    def __init__(
        self,
        max_input_tokens: int,
        max_output_tokens: int,
        max_tool_tokens: int = 4000,
        max_total_tool_tokens: int = 30000,
    ):
        # 输出预留：防止输入塞满窗口导致真实端点 400（max_output_tokens 或 20% 取小）。
        reserve = min(max_output_tokens, int(0.2 * max_input_tokens))
        self.input_budget = max(1, max_input_tokens - reserve)
        self.max_tool_tokens = max(0, max_tool_tokens)
        self.max_total_tool_tokens = max(0, max_total_tool_tokens)

    def compose(
        self,
        system_prompt: str,
        history: list[Message],
        user_input: str,
    ) -> list[Message]:
        """组装 system + history + user，超预算时逐部分压缩。

        返回值：``[system] + 压缩后的 history + [user]``。
        user_input 永不压缩（Part 3）。
        """
        user_tokens = count_tokens(user_input)
        budget = self.input_budget - user_tokens

        # Part 1：system（含 memory），正常不触发，超限先压 memory 段。
        system = self._compress_system(system_prompt, budget)
        budget -= count_tokens(system)

        # Part 2：本轮 history。history 里与 user_input 同内容的消息一并保护。
        msgs = self._compress_history(
            list(history), budget, protected_contents={user_input}
        )

        return [
            Message(role="system", content=system),
            *msgs,
            Message(role="user", content=user_input),
        ]

    def recompress(
        self, messages: list[Message], user_input: str | None = None
    ) -> list[Message]:
        """对已组装好的消息列表压缩；主循环每轮发给 LLM 前调用。

        ``compose`` 只在 run 初始组装时生效，而 tool 结果是轮内 append 后
        才进入消息列表的——若不在每轮压缩，单条 tool 输出限制与总量限制
        都会被绕过。规则：

        - 第一条 system 保留（其压缩已在 run 初始由 compose 完成）；
        - ``user_input`` 指定的当前用户输入消息**永不丢弃**（Part 3 铁律）。
          多轮 tool 场景下用户输入会被 append 的 tool 对顶到中间，不能只按
          「最后一条 user」识别，需按 content 匹配；
        - 其余消息按与 compose 相同的预算规则压缩（tool 单条裁剪 /
          tool 总量丢弃 / 非 tool 丢弃 / 兜底丢 tool 对）。
        """
        if not messages:
            return messages
        system = messages[0]
        body = messages[1:]
        protected = {user_input} if user_input is not None else None
        # 预算 = 输入预算 - system - 受保护用户输入（永不丢，token 全额预留）。
        reserved = count_tokens(system.content)
        if user_input is not None:
            reserved += count_tokens(user_input)
        budget = self.input_budget - reserved
        compressed = self._compress_history(
            body, budget, protected_contents=protected
        )
        return [system, *compressed]

    # ---- Part 1：system prompt 压缩 ----

    def _compress_system(self, prompt: str, budget: int) -> str:
        if budget <= 0:
            return truncate_to_tokens(prompt, 1) if prompt else ""
        if count_tokens(prompt) <= budget:
            return prompt
        return self._clip_memory_section(prompt, budget)

    def _clip_memory_section(self, prompt: str, budget: int) -> str:
        """优先压缩 memory 段（``## 跨会话记忆`` → 下一个二级标题）；不够再截尾部。"""
        idx = prompt.find(_MEMORY_MARKER)
        if idx != -1:
            body_start = prompt.find("\n\n", idx)
            if body_start != -1:
                body_start += 2
                next_heading = prompt.find("\n## ", body_start)
                seg_end = next_heading if next_heading != -1 else len(prompt)
                prefix = prompt[:body_start]
                suffix = prompt[seg_end:]
                body = prompt[body_start:seg_end]
                # 预算预留裁剪标记的 token 数，保证候选结果不超预算。
                mark_tokens = count_tokens(f"\n{_CLIPPED_MARK}")
                body_budget = (
                    budget - count_tokens(prefix) - count_tokens(suffix) - mark_tokens
                )
                if body_budget > 0:
                    clipped = truncate_to_tokens(body, body_budget)
                else:
                    clipped = f"（{_MEMORY_MARKER} 内容已裁剪）\n"
                candidate = prefix + clipped + suffix
                if count_tokens(candidate) <= budget:
                    return candidate
        # memory 段不存在或压缩后仍超：整体截尾。
        return truncate_to_tokens(prompt, budget)

    # ---- Part 2：history 压缩 ----

    def _compress_history(
        self,
        msgs: list[Message],
        budget: int,
        protected_contents: set[str] | None = None,
    ) -> list[Message]:
        """按预算分层压缩。

        ``protected_contents`` 中的 user 消息永不丢弃（Part 3 铁律，
        用于多轮场景下按 content 识别当前用户输入）。

        阶段 1/2 是 tool 的**独立预算约束**（单条上限、总量上限），与整体
        输入预算正交、无条件执行；阶段 3/4 才在整体预算不足时执行。
        """
        # 阶段 1：单条 tool 输出超限 → 裁剪为开头+结尾各半（只换 content，
        # 不动 assistant 的 tool_calls 声明与配对，否则真实端点 400）。
        for i, m in enumerate(msgs):
            if m.role == "tool" and count_tokens(m.content) > self.max_tool_tokens:
                msgs[i] = Message(
                    role="tool",
                    content=clip_head_tail(m.content, self.max_tool_tokens),
                    tool_call_id=m.tool_call_id,
                    metadata=dict(m.metadata),
                )

        # 阶段 2：tool 总量超限 → 从最早工具组整组丢弃，保留最新一组。
        while True:
            groups = self._collect_tool_pairs(msgs)
            if len(groups) <= 1:
                break  # 最新一组（或没有）永不因总量被丢
            tool_total = sum(
                count_message_tokens(msgs[i]) for group in groups for i in group
            )
            if tool_total <= self.max_total_tool_tokens:
                break
            # 删除最早一组（索引降序，避免位移）。
            for i in sorted(groups[0], reverse=True):
                del msgs[i]

        total = sum(count_message_tokens(m) for m in msgs)
        if total <= budget:
            return msgs

        # 阶段 3：非 tool 强压缩 —— 从最早的非 tool 消息丢弃，
        # 保护带 tool_calls 的 assistant（tool 链完整性）与受保护 user。
        while total > budget:
            idx = self._oldest_droppable(msgs, protected_contents)
            if idx is None:
                break
            del msgs[idx]
            total = sum(count_message_tokens(m) for m in msgs)
        if total <= budget:
            return msgs

        # 阶段 4：兜底 —— 非 tool 删光仍超，才丢最早工具组（同样保最新一组）。
        while total > budget:
            groups = self._collect_tool_pairs(msgs)
            if len(groups) <= 1:
                break
            for i in sorted(groups[0], reverse=True):
                del msgs[i]
            total = sum(count_message_tokens(m) for m in msgs)
        return msgs

    def _oldest_droppable(
        self, msgs: list[Message], protected_contents: set[str] | None = None
    ) -> int | None:
        """最早的、可安全丢弃的非 tool 消息索引；无则返回 None。

        可丢：role=user（含 retry 反馈，但受保护内容除外）、role=assistant
        但无 tool_calls（纯文本/最终答案）；不可丢：system、tool、
        带 tool_calls 的 assistant、受保护的用户输入。
        """
        for i, m in enumerate(msgs):
            if m.role == "system" or m.role == "tool":
                continue
            if m.role == "assistant" and m.tool_calls:
                continue
            if protected_contents and m.role == "user" and m.content in protected_contents:
                continue
            return i
        return None

    def _collect_tool_pairs(self, msgs: list[Message]) -> list[list[int]]:
        """按出现顺序收集工具调用组：每个组 = [assistant 声明, *配对 tool 结果]。

        一个 assistant 可带多个 tool_calls（原生 FC 并行），其全部结果
        同属一组；配对按 tool_call_id 匹配，只收连续跟随的 role=tool 消息。
        """
        groups: list[list[int]] = []
        i = 0
        while i < len(msgs):
            m = msgs[i]
            if m.role == "assistant" and m.tool_calls:
                call_ids = {(call or {}).get("id") for call in m.tool_calls}
                group = [i]
                j = i + 1
                while j < len(msgs) and msgs[j].role == "tool":
                    if msgs[j].tool_call_id in call_ids:
                        group.append(j)
                    j += 1
                groups.append(group)
                i = j
                continue
            i += 1
        return groups
