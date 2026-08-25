"""MemoryManager：会话存档、水位线状态机、摘要聚合与注入文本。

对齐 oh-my-pi 精简版记忆设计：

- 会话原文逐条追加 ``memories/<sessionId>.jsonl``（脱敏后）；
- 退出只打 ``status="closed"``（零 LLM 成本），下次启动 :meth:`sweep`
  统一扫描：closed 会话按内容水位线（``archive_mtime`` vs mtime）判断是否
  需要重新汇总，failed 会话在 retry 上限内重试，open 会话按 mtime 静止阈值
  兜底进程被杀的场景；
- 幂等：``.state/<sessionId>.json`` 记录 ``archive_mtime`` 水位线，
  内容没变（mtime 相同且 status=done）不重提，不重复花 LLM 钱；
- 可重试：LLM 失败记 ``status=failed, retry+1``，下次启动重试，超过
  ``max_retry``（默认 3 次）放弃并标记 ``failed_done``；
- 聚合：本次扫描有单会话摘要内容变化（sha256 比对 manifest）才全量重写
  ``summary.md``，无变化零调用；聚合失败记 ``aggregate_error`` 下次重试；
- 注入：:meth:`context_block` 带 ``query``（当前请求）时按关键词相关性
  检索单会话摘要（``index.rank``，纯词法零 LLM 成本），命中则优先注入
  top-K 相关摘要、剩余预算给全局 ``summary.md`` 兜底；无命中回落全局。

``MemoryManager`` 可无 LLM 构造（``llm=None``），此时仅作为存档器：
``append_message`` / ``close_session`` 正常，``sweep`` 无副作用。
"""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agent_config import Message
from ..contracts import LLMClient
from .archive import append_archive, archive_path, read_archive
from .index import index_summaries, rank
from .summarizer import SummarizeError, Summarizer

#: 状态值。
STATUS_OPEN = "open"          # 会话进行中（追加消息时的初始状态）
STATUS_CLOSED = "closed"      # 会话已正常退出，待汇总
STATUS_DONE = "done"          # 已成功汇总（幂等跳过）
STATUS_FAILED = "failed"      # 汇总失败，待重试
STATUS_FAILED_DONE = "failed_done"  # 重试次数耗尽，放弃

#: open 会话被判定为「已结束」的 mtime 静止阈值（秒）。
DEFAULT_IDLE_THRESHOLD = 60

#: 单会话汇总失败允许的重试次数。
DEFAULT_MAX_RETRY = 3

#: 注入 system prompt 的跨会话记忆截断字符数。
DEFAULT_MAX_CONTEXT_CHARS = 4000

#: 相关检索注入时优先注入的单会话摘要条数。
DEFAULT_TOP_K = 2

#: 聚合 manifest 文件名。
AGGREGATE_MANIFEST = "aggregate.json"


def new_session_id(now: datetime | None = None) -> str:
    """生成会话 ID：session_YYYYMMDD_HHMMSS_<4 位随机>。"""
    now = now or datetime.now()
    return f"session_{now:%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MemoryManager:
    """会话记忆管理器：实现 contracts.MemoryStore 协议。"""

    def __init__(
        self,
        memory_dir: str | Path,
        llm: LLMClient | None = None,
        prompts_dir: str | Path | None = None,
        session_id: str | None = None,
        idle_threshold: float = DEFAULT_IDLE_THRESHOLD,
        max_retry: int = DEFAULT_MAX_RETRY,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ):
        self.memory_dir = Path(memory_dir).resolve()
        self.state_dir = self.memory_dir / ".state"
        self.llm = llm
        self.session_id = session_id or new_session_id()
        self.idle_threshold = idle_threshold
        self.max_retry = max_retry
        self.max_context_chars = max_context_chars

        if llm is not None:
            from ..agent_config import PROMPT_DIR

            self.summarizer = Summarizer(
                llm, prompts_dir or PROMPT_DIR
            )
        else:
            self.summarizer = None

        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ---- 存档（会话生命周期，由 REPL 层调用） ----

    def append_message(self, message: Message) -> None:
        """把一条会话消息脱敏后追加进本次会话 JSONL 存档。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        append_archive(self.memory_dir, self.session_id, message)
        self._save_state(self.session_id, {"status": STATUS_OPEN})

    def close_session(self) -> None:
        """标记本次会话「可汇总」；不触发 LLM 调用。"""
        if not archive_path(self.memory_dir, self.session_id).is_file():
            return
        state = self._load_state(self.session_id)
        state["status"] = STATUS_CLOSED
        self._save_state(self.session_id, state)

    # ---- 注入（主循环读取） ----

    def context_block(
        self, query: str | None = None, max_chars: int | None = None
    ) -> str:
        """返回注入 system prompt 的跨会话记忆文本；无记忆或空内容时返回 ""。

        - ``query`` 非空 → 按关键词相关性检索单会话摘要，命中则优先注入
          top-K 相关摘要，剩余预算再给全局 summary.md 兜底；无命中回落全局；
        - ``query`` 为空/None → 维持现状：全量 summary.md 截断注入。
        """
        limit = max_chars or self.max_context_chars
        if query and query.strip():
            block = self._relevant_block(query.strip(), limit)
            if block:
                return block
        return self._global_block(limit)

    def _global_block(self, limit: int) -> str:
        """全量聚合摘要 summary.md，截断后包成 markdown 段返回。"""
        summary_file = self.memory_dir / "summary.md"
        try:
            content = summary_file.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not content:
            return ""
        if len(content) > limit:
            content = content[:limit] + "\n…（已截断）"
        return f"## 跨会话记忆\n\n{content}"

    def _relevant_block(self, query: str, limit: int) -> str:
        """按相关性检索单会话摘要：相关摘要（top-K）+ 全局 summary.md 兜底。

        无摘要文件或全部无命中时返回 ""（调用方回落 :meth:`_global_block`）。
        """
        hits = rank(query, index_summaries(self.memory_dir))
        if not hits:
            return ""
        top = hits[:DEFAULT_TOP_K]
        parts = [f"## 相关会话记忆（命中 {len(top)} 条）"]
        budget = limit
        for item in top:
            text = item["text"]
            if budget <= 0:
                break
            if len(text) > budget:
                text = text[:budget] + "\n…（已截断）"
            parts.append(text)
            budget -= len(text)
        # 剩余预算给全局聚合摘要兜底，保证整体概览仍在。
        summary_file = self.memory_dir / "summary.md"
        try:
            content = summary_file.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        if content and budget > 0:
            if len(content) > budget:
                content = content[:budget] + "\n…（已截断）"
            parts.append(f"## 跨会话记忆（全局兜底）\n\n{content}")
        return "\n\n".join(parts)

    # ---- 汇总（下次启动，由调用方触发） ----

    def sweep(self) -> None:
        """扫描全部会话存档，按状态机汇总并重写跨会话 summary.md。

        无 LLM 注入时直接返回（仅存档用途）；出错均不抛出，状态记入
        .state 供下次重试。
        """
        if self.llm is None or self.summarizer is None:
            return
        if not self.memory_dir.is_dir():
            return

        changed = False
        for session_file in sorted(self.memory_dir.glob("session_*.jsonl")):
            session_id = session_file.stem
            if not self._should_process(session_id, session_file):
                continue
            if self._process_session(session_id):
                changed = True

        self._maybe_aggregate()

    def _should_process(self, session_id: str, session_file: Path) -> bool:
        """按状态与 mtime 水位线决定本次是否需要处理该会话。"""
        state = self._load_state(session_id)
        status = state.get("status")
        mtime = session_file.stat().st_mtime

        if status == STATUS_DONE:
            # 幂等：内容没变（mtime 与上次处理时一致）→ 跳过。
            return state.get("archive_mtime") != mtime
        if status == STATUS_FAILED_DONE:
            return False  # 已放弃
        if status == STATUS_FAILED:
            if state.get("retry", 0) < self.max_retry:
                return True
            # 重试次数耗尽：标记放弃，避免下次再次进入判定。
            self._save_state(
                session_id,
                {"status": STATUS_FAILED_DONE, "retry": state.get("retry", 0)},
            )
            return False
        if status == STATUS_CLOSED:
            # 内容驱动：仅当从未汇总过或存档有新增（mtime 变化）才重新处理。
            # 不能「closed 无条件处理」——否则重新进入一个已结束的会话，
            # 即使一句话没说（JSONL 内容未变），每次退出后 sweep 都会
            # 重复调用 LLM 做 extract（会话可恢复场景的重复计费）。
            return state.get("archive_mtime") != mtime
        # open 或无状态：mtime 静止超阈值才兜底处理（进程被杀场景）。
        if state.get("archive_mtime") is not None:
            return False
        return (datetime.now().timestamp() - mtime) >= self.idle_threshold

    def _process_session(self, session_id: str) -> bool:
        """汇总单个会话；成功返回 True（触发聚合），失败记状态返回 False。"""
        state = self._load_state(session_id)
        session_file = archive_path(self.memory_dir, session_id)
        mtime = session_file.stat().st_mtime
        records = read_archive(self.memory_dir, session_id)
        if not records:
            # 空存档：没有可汇总内容，直接标记 done 避免反复扫描。
            self._save_state(
                session_id, {"status": STATUS_DONE, "archive_mtime": mtime}
            )
            return False

        try:
            assert self.summarizer is not None
            summary_md = self.summarizer.extract(session_id, records)
        except SummarizeError as exc:
            retry = state.get("retry", 0) + 1
            self._save_state(
                session_id,
                {
                    "status": STATUS_FAILED,
                    "retry": retry,
                    "archive_mtime": mtime,
                    "last_error": str(exc),
                },
            )
            return False

        (self.memory_dir / f"{session_id}.summary.md").write_text(
            summary_md, encoding="utf-8"
        )
        self._save_state(
            session_id,
            {"status": STATUS_DONE, "archive_mtime": mtime, "retry": 0},
        )
        return True

    def _maybe_aggregate(self) -> None:
        """所有单会话摘要 → 全量重写 summary.md（内容变化才调 LLM）。"""
        summaries = sorted(self.memory_dir.glob("*.summary.md"))
        input_text = "\n".join(
            f.read_text(encoding="utf-8") for f in summaries
        )
        input_sha = _sha256(input_text)
        manifest = self._load_manifest()
        if input_sha == manifest.get("input_sha") and not manifest.get(
            "aggregate_error"
        ):
            return  # 幂等：无变化且上次聚合成功

        if not summaries:
            # 没有任何单会话摘要：清空聚合产物与错误，避免空转。
            summary_file = self.memory_dir / "summary.md"
            if summary_file.exists():
                summary_file.unlink()
            self._save_manifest({"input_sha": input_sha})
            return

        try:
            assert self.summarizer is not None
            content = self.summarizer.aggregate(
                [f.read_text(encoding="utf-8") for f in summaries]
            )
        except SummarizeError as exc:
            manifest["aggregate_error"] = str(exc)
            self._save_manifest(manifest)
            return
        (self.memory_dir / "summary.md").write_text(content, encoding="utf-8")
        self._save_manifest({"input_sha": input_sha})

    # ---- .state 读写 ----

    def _state_path(self, session_id: str) -> Path:
        return self.state_dir / f"{session_id}.json"

    def _load_state(self, session_id: str) -> dict[str, Any]:
        path = self._state_path(session_id)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self, session_id: str, state: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self._state_path(session_id)
        path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _manifest_path(self) -> Path:
        return self.state_dir / AGGREGATE_MANIFEST

    def _load_manifest(self) -> dict[str, Any]:
        path = self._manifest_path()
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
