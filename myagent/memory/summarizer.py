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

#: 词法兜底关键词提取的通用停用词（英文 token 与高频中文虚词）。
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "what",
    "was", "were", "have", "has", "had", "not", "but", "are", "its",
    "will", "been", "they", "them", "then", "than", "just", "also",
    "when", "where", "which", "while", "because", "about", "after",
    "before", "more", "most", "some", "such", "only", "over", "under",
    "would", "could", "should", "does", "done", "did", "doing", "ok",
    "can", "may", "get", "got", "let", "use", "used", "via",
    "可以", "进行", "完成", "使用", "需要", "一个", "一次", "相关",
    "输出", "结果", "内容", "文件", "工作", "代码", "修改", "实现",
    "没有", "已经", "然后", "通过", "对于", "我们", "自己", "当前",
    # 摘要 JSON 契约的字段名/结构词（词法兜底时不应成为关键词）。
    "rollout_summary", "raw_memory", "files_changed", "keywords",
    "json", "session", "summary", "会话摘要", "改动文件",
}

#: 已知工具名（词法关键词提取时优先收录；与 tools/ 注册表保持一致）。
TOOL_NAMES = {
    "read_file", "create_file", "write_file", "edit_file", "delete_file",
    "list_files", "create_dir", "write_dir", "rename_dir", "delete_dir",
    "run_shell", "git_status", "git_diff", "git_apply_patch",
    "delegate_agent",
}

#: 文件路径/文件名模式（词法关键词提取）。
_FILE_RE = re.compile(
    r"[\w./-]+\.(?:py|md|json|jsonl|toml|txt|sh|js|ts|go|rs|java|c|cpp|"
    r"h|yml|yaml|ini|cfg|log|html|css|env|lock)\b",
    re.IGNORECASE,
)

#: 英文 token 模式（词法关键词提取）。
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _normalize_keywords(value: Any) -> list[str]:
    """规范化模型输出的 keywords：只保留非空字符串，去重保序。"""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def build_keywords(text: str, limit: int = 20) -> list[str]:
    """词法兜底关键词：从文本机械提取文件路径/文件名、工具名、英文 token。

    用于旧摘要（LLM 未产出 keywords 字段）在读取/检索时补索引，
    零 LLM 成本。去重保序，最多返回 ``limit`` 个。
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            found.append(token)

    for match in _FILE_RE.finditer(text):
        add(match.group(0))
        if len(found) >= limit:
            return found
    for name in TOOL_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text):
            add(name)
            if len(found) >= limit:
                return found
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.lower() in _STOPWORDS:
            continue
        add(token)
        if len(found) >= limit:
            break
    return found


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


def parse_summary_markdown(text: str) -> dict[str, Any]:
    """把 :func:`_render_summary_markdown` 产物解析回 payload dict。

    兼容旧契约（LLM 未产出 keywords 的摘要文件）：
    ``keywords`` 缺失或为空时用 :func:`build_keywords` 词法兜底补索引，
    零 LLM 成本。解析失败返回 {}。
    """
    payload = _parse_json_object(text)
    if payload is None:
        return {}
    payload.setdefault("files_changed", [])
    payload.setdefault("raw_memory", "")
    keywords = _normalize_keywords(payload.get("keywords"))
    if not keywords:
        keywords = build_keywords(text)
    payload["keywords"] = keywords
    return payload


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
                "keywords": [],
            }
        else:
            payload = {
                "rollout_summary": str(payload.get("rollout_summary", "")).strip(),
                "files_changed": files_changed,
                "raw_memory": str(payload.get("raw_memory", "")).strip(),
                "keywords": _normalize_keywords(payload.get("keywords")),
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
