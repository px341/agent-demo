"""记忆检索索引：扫描单会话摘要、按查询相关性打分排序。

- :func:`index_summaries`：扫描 ``*.summary.md``，解析 JSON 契约
  （旧摘要无 ``keywords`` 时词法兜底补索引），组装可注入的摘要文本；
- :func:`rank`：查询与摘要关键词的重叠打分——query token 与 keyword
  小写全等 +2（文件路径/工具名/英文名词），互为子串（含中文 bigram
  落在 keyword 中）+1；按分降序返回命中条目。

纯本地词法匹配，零 LLM 成本；与 :mod:`myagent.memory.summarizer`
的关键词契约（``keywords`` 字段）配合。query 与 keywords 两侧共用
同一套 token 化（:func:`tokenize`），保证匹配口径一致。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .summarizer import (
    TOOL_NAMES,
    _FILE_RE,
    _STOPWORDS,
    _TOKEN_RE,
    parse_summary_markdown,
)

#: 中文 bigram 模式（query 中文与 LLM 中文关键词标签的匹配通道）。
_HANZI_RE = re.compile(r"[\u4e00-\u9fff]")

#: 精确命中权重（query token 与 keyword 小写全等）。
EXACT_WEIGHT = 2
#: 子串/包含命中权重（含中文 bigram 落在 keyword 中）。
SUBSTRING_WEIGHT = 1


def tokenize(text: str) -> set[str]:
    """把文本拆成检索 token 集合（小写归一）。

    覆盖：文件路径/文件名、工具名、英文 token（过滤停用词）、中文
    bigram。query 与 keywords 两侧使用同一套 token 化。
    """
    tokens: set[str] = set()
    for match in _FILE_RE.finditer(text):
        tokens.add(match.group(0).lower())
    for name in TOOL_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text):
            tokens.add(name)
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        if token not in _STOPWORDS:
            tokens.add(token)
    hanzi = _HANZI_RE.findall(text)
    for i in range(len(hanzi) - 1):
        tokens.add(hanzi[i] + hanzi[i + 1])
    return tokens


def index_summaries(memory_dir: str | Path) -> list[dict[str, Any]]:
    """扫描全部单会话摘要，解析为带 keywords 与注入文本的条目。

    返回按 session_id 排序的列表；每个条目：:

        {"session_id": str, "keywords": list[str],
         "text": str   # 可注入的摘要文本（rollout_summary + raw_memory + files_changed）
        }

    解析失败或文件不可读的摘要跳过。
    """
    base = Path(memory_dir)
    entries: list[dict[str, Any]] = []
    for summary_file in sorted(base.glob("*.summary.md")):
        session_id = summary_file.stem.removesuffix(".summary")
        try:
            raw = summary_file.read_text(encoding="utf-8")
        except OSError:
            continue
        payload = parse_summary_markdown(raw)
        if not payload:
            continue
        files = payload.get("files_changed") or []
        files_text = ""
        if files:
            files_text = "\n- 改动文件：" + "、".join(str(f) for f in files)
        text = (
            f"## 会话摘要 {session_id}\n\n"
            f"- 做了什么：{payload.get('rollout_summary', '')}\n"
            f"- 记住：{payload.get('raw_memory', '')}"
            f"{files_text}"
        ).strip()
        entries.append(
            {
                "session_id": session_id,
                "keywords": payload.get("keywords") or [],
                "text": text,
            }
        )
    return entries


def rank(query: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 query 与摘要关键词的重叠打分，返回分数 > 0 且降序的条目副本。

    打分规则：
    - query token 与 keyword 小写全等 → +``EXACT_WEIGHT``
      （文件路径/工具名/英文名词精确命中）；
    - query token 与 keyword 互为子串 → +``SUBSTRING_WEIGHT``
      （含中文 bigram 落在 keyword 中的情况）。
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scored: list[dict[str, Any]] = []
    for entry in entries:
        score = 0
        for token in query_tokens:
            for keyword in entry.get("keywords") or []:
                kw = keyword.lower()
                if token == kw:
                    score += EXACT_WEIGHT
                elif token in kw or kw in token:
                    score += SUBSTRING_WEIGHT
        if score > 0:
            item = dict(entry)
            item["score"] = score
            scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored
