"""会话原文存档：jsonl 追加/读取、敏感信息脱敏、files_changed 机械提取。

- 每条存档是独立 JSON 行：``{"ts": ISO, "role": ..., "content": ..., ...}``，
  字段与 Message 契约对应（含 tool_calls / tool_call_id）；
- 写入前对全部字符串做 :func:`redact` 脱敏，满足「写入前去掉 API key 等
  敏感信息」的铁律；
- :func:`extract_files_changed` 从 assistant 消息的 tool_calls 声明中机械
  提取写类工具的 path 参数，零 LLM 成本、确定性去重。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from ..agent_config import Message

#: 单行存档的时间戳格式（ISO 8601 本地时间）。
TS_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: 写入类工具（改变工作区内容的路径才计入 files_changed）。
#: read_file / list_files 等只读工具不在此列。
WRITE_TOOLS = {
    "create_file",
    "write_file",
    "edit_file",
    "delete_file",
    "rename_dir",
    "create_dir",
    "write_dir",
    "delete_dir",
}

#: 脱敏规则：正则 → 替换模板。key 分组用于保留字段名前缀与分隔符。
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # sk- 开头的 API key（OpenAI/DeepSeek 等惯例）。
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    # GitHub 个人访问令牌（ghp_ 前缀）。
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "ghp_***"),
    # Authorization: Bearer <凭据>（先于 key-value 规则，避免 key-value 吞掉 Bearer）。
    (re.compile(r"(?i)\bbearer\s+\S+"), "Bearer ***"),
    # 以 key/token/secret/password 等命名的赋值（大小写不敏感、容忍下划线/连字符）。
    # 负向前瞻 (?!bearer\b) 跳过已被 Bearer 规则处理的「Bearer <token>」形态。
    (
        re.compile(
            r"(?i)(api[_-]?key|token|secret|password|passwd|authorization)"
            r"\s*([:=])\s*(?!bearer\b)\S+"
        ),
        r"\1\2 ***",
    ),
]


def redact(text: str) -> str:
    """清洗文本中的 API key / token / 密码等敏感信息；逐条应用规则。"""
    for pattern, repl in _REDACT_PATTERNS:
        text = pattern.sub(repl, text)
    return text


#: 敏感字段名（dict 键命中时其 value 视为敏感信息）。
_KEY_NAME_RE = re.compile(
    r"(?i)^(api[_-]?key|token|secret|password|passwd|authorization)$"
)


def _redact_value(value: Any) -> Any:
    """对 dict/list 深度脱敏；非字符串原样返回。

    dict 中 key 命中敏感字段名时，对应 value 直接标 ``***``
    （覆盖 ``{"password": "..."}`` 这类 JSON 结构形态，保持结构合法）。
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(key, str) and _KEY_NAME_RE.match(key):
                result[key] = "***"
            else:
                result[key] = _redact_value(item)
        return result
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    return value


def _message_to_record(message: Message) -> dict[str, Any]:
    """Message → 存档 dict（脱敏后）；tool_calls 里嵌的 JSON 字符串一并清洗。"""
    record: dict[str, Any] = {
        "ts": datetime.now().strftime(TS_FORMAT),
        "role": message.role,
    }
    if message.content is not None:
        record["content"] = redact(message.content)
    if message.tool_calls:
        record["tool_calls"] = _redact_value(message.tool_calls)
    if message.tool_call_id:
        record["tool_call_id"] = message.tool_call_id
    if message.metadata:
        record["metadata"] = _redact_value(message.metadata)
    return record


def append_archive(memory_dir, session_id: str, message: Message) -> None:
    """把一条消息追加写入会话 JSONL 存档（追加模式，原子性靠单行写）。

    写入前先按字段脱敏（含 dict 结构的敏感 key），保证落盘内容无明文密钥。
    """
    path = archive_path(memory_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_message_to_record(message), ensure_ascii=False))
        fh.write("\n")


def archive_path(memory_dir, session_id: str) -> "Path":
    """会话 JSONL 存档路径。"""
    from pathlib import Path

    return Path(memory_dir) / f"{session_id}.jsonl"


def read_archive(memory_dir, session_id: str) -> list[dict[str, Any]]:
    """读取会话全部存档记录；空/损坏行跳过（损坏行不影响后续读取）。"""
    path = archive_path(memory_dir, session_id)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_files_changed(records: list[dict[str, Any]]) -> list[str]:
    """从存档记录机械提取改动路径（写类工具），去重保序。

    依据 assistant 消息的 tool_calls 声明：``name`` 在 WRITE_TOOLS 内，
    从 ``function.arguments`` JSON 中取 ``path``；rename_dir 取 src→dst。
    """
    paths: list[str] = []
    for record in records:
        if record.get("role") != "assistant":
            continue
        for call in record.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            name = function.get("name")
            if name not in WRITE_TOOLS:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if name == "rename_dir":
                if args.get("src"):
                    paths.append(str(args["src"]))
                if args.get("dst"):
                    paths.append(str(args["dst"]))
            elif args.get("path"):
                paths.append(str(args["path"]))
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique
