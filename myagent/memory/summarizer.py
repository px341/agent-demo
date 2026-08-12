"""摘要生成：单会话提取 + 跨会话聚合，全部走 LLMClient（复用 provider 客户端）。

- :func:`extract`：把会话原文渲染成紧凑 transcript，交 LLM 提取
  ``rollout_summary`` + ``raw_memory`` 两段 prose；``files_changed`` 由
  :func:`myagent.memory.archive.extract_files_changed` 机械提取，本地组装成
  完整 JSON 契约写盘。LLM 输出无法解析时**降级**保存（保留原始输出，
  不抛异常、不重试——同一输出重提无意义）；
- :func:`aggregate`：把全部单会话摘要交 LLM 全量重写 ``summary.md``；
  输出本身即 markdown，不二次解析。LLM 调用异常向上抛，由 manager
  走 failed/retry 状态机。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..contracts import LLMClient, Message

#: transcript 传给 LLM 前的字符上限（防长会话撑爆输入）。
DEFAULT_MAX_TRANSCRIPT_CHARS = 30000

#: 无法解析时降级保存的原始输出截断长度。
FALLBACK_RAW_LIMIT = 2000


class SummarizeError(Exception):
    """LLM 调用失败（网络/API 异常）等可重试错误。"""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """宽松提取第一个 JSON 对象（容忍代码块包裹）。"""
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


def _strip_code_fence(text: str) -> str:
    """去掉 markdown 代码块包裹（``` 首尾），返回净文本。"""
    if "```" not in text:
        return text
    pattern = re.compile(r"^```(?:[A-Za-z0-9_+-]*)?\s*$", re.MULTILINE)
    cleaned = pattern.sub("", text).strip()
    return cleaned.strip("\n")


def render_transcript(records: list[dict[str, Any]]) -> str:
    """把存档记录渲染为紧凑对话文本（供 LLM 提取摘要）。

    工具错误与部分执行（任务中断）有专门标记，便于模型重点提取：
    - content 含 ``ToolError[`` 的 tool 记录 → ``→ 工具错误[类型]：...``
      （错误信息是记忆重点，截断放宽到 800 字符）；
    - metadata.partial 的记录 → ``→ 工具结果（部分执行，任务中断）：...``。
    """
    lines: list[str] = []
    for record in records:
        role = record.get("role", "?")
        if role == "tool":
            content = record.get("content") or ""
            error_type = record.get("error_type")
            if error_type or content.startswith("ToolError["):
                prefix = f"→ 工具错误[{error_type or '?'}]："
                body = content[:800]
            elif record.get("partial"):
                prefix = "→ 工具结果（部分执行，任务中断）："
                body = content[:800]
            else:
                prefix = "→ 工具结果："
                body = content[:500]
            lines.append(prefix + body)
            continue
        if role == "assistant" and record.get("tool_calls"):
            for call in record.get("tool_calls") or []:
                function = (call or {}).get("function") or {}
                lines.append(
                    f"[{function.get('name', '?')}] {function.get('arguments', '{}')}"
                )
            continue
        content = (record.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _render_summary_markdown(session_id: str, payload: dict[str, Any]) -> str:
    """单会话摘要 markdown：元信息 + JSON 契约（代码块包裹）。"""
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"# 会话摘要 {session_id}\n\n"
        f"```json\n{body}\n```\n"
    )


class Summarizer:
    """用注入的 LLMClient 生成单会话摘要与跨会话聚合摘要。"""

    def __init__(
        self,
        llm: LLMClient,
        prompts_dir: Path | str,
        max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
    ):
        self.llm = llm
        self.prompts_dir = Path(prompts_dir)
        self.max_transcript_chars = max_transcript_chars

    def _load_prompt(self, name: str) -> str:
        path = self.prompts_dir / name
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return name

    def extract(
        self, session_id: str, records: list[dict[str, Any]]
    ) -> str:
        """单会话提取：返回写盘的摘要 markdown（含 files_changed 机械组装）。"""
        from .archive import extract_files_changed

        files_changed = extract_files_changed(records)
        transcript = render_transcript(records)
        if len(transcript) > self.max_transcript_chars:
            transcript = (
                transcript[: self.max_transcript_chars] + "\n…（已截断）"
            )
        system_prompt = self._load_prompt("summary_extract_prompt.md")
        try:
            result = self.llm.complete(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=transcript),
                ]
            )
        except Exception as exc:
            raise SummarizeError(f"单会话摘要 LLM 调用失败：{exc}") from exc

        output = (result.text or "").strip()
        payload = _parse_json_object(output)
        if payload is None or "rollout_summary" not in payload:
            # 降级：保留原始输出，不抛异常不重试。
            raw = output[:FALLBACK_RAW_LIMIT] or "（模型未返回可解析内容）"
            payload = {
                "rollout_summary": raw,
                "files_changed": files_changed,
                "raw_memory": "",
            }
        else:
            payload = {
                "rollout_summary": str(payload.get("rollout_summary", "")).strip(),
                "files_changed": files_changed,
                "raw_memory": str(payload.get("raw_memory", "")).strip(),
            }
        return _render_summary_markdown(session_id, payload)

    def aggregate(self, summaries: list[str]) -> str:
        """跨会话聚合：所有单会话摘要 → summary.md 完整内容。"""
        if not summaries:
            return ""
        body = "\n\n---\n\n".join(summaries)
        if len(body) > self.max_transcript_chars:
            body = body[: self.max_transcript_chars] + "\n…（已截断）"
        system_prompt = self._load_prompt("summary_aggregate_prompt.md")
        try:
            result = self.llm.complete(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=body),
                ]
            )
        except Exception as exc:
            raise SummarizeError(f"跨会话聚合 LLM 调用失败：{exc}") from exc
        text = _strip_code_fence(result.text or "").strip()
        return text if text else "（聚合摘要为空）"
