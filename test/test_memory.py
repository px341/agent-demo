"""记忆系统测试：jsonl 存档、脱敏、files_changed 机械提取、水位线状态机、
聚合幂等、注入文本与 AgentLoop 接线。

覆盖三条铁律：幂等（mtime 水位线跳过）、可重试（retry<=3）、脱敏（sk- 等）。
全部使用临时目录隔离，不触碰真实 memories/ 与 .storage/。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.actions import FinalAnswer, ToolCall
from myagent.agent_config import AgentParams, Message
from myagent.agent_loop import AgentLoop
from myagent.cli import _archive_steps, _one_shot, _repl
from myagent.contracts import (
    AgentRequest,
    AgentResponse,
    StepRecord,
    StopReason,
    LLMResponse,
)
from myagent.memory.archive import (
    append_archive,
    archive_path,
    extract_files_changed,
    read_archive,
    redact,
)
from myagent.memory.manager import (
    STATUS_CLOSED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_OPEN,
    MemoryManager,
    _sha256,
)
from myagent.memory.summarizer import (
    SummarizeError,
    Summarizer,
    _parse_json_object,
    render_transcript,
)


class FakeLLM:
    """脚本化输出序列；记录每次调用的消息快照。"""

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = list(outputs or [])
        self.calls: list[list[Message]] = []

    def complete(self, messages, *, max_new_tokens=None):
        self.calls.append(list(messages))
        if self.outputs:
            return LLMResponse(text=self.outputs.pop(0))
        raise AssertionError("FakeLLM 输出脚本耗尽")


def make_manager(tmp: Path, llm=None, **kwargs) -> MemoryManager:
    kwargs.setdefault("session_id", "session_20260812_000000_ab12")
    return MemoryManager(memory_dir=tmp / "memories", llm=llm, **kwargs)


def tool_call_message(name: str, args: dict) -> Message:
    """构造一条 assistant 工具调用声明消息（与 step_to_messages 输出同构）。"""
    return Message(
        role="assistant",
        content=f'{{"action": "tool_call", "tool": "{name}"}}',
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ],
    )


def final_step_message(answer: str) -> Message:
    return Message(role="assistant", content=answer)


def summary_payload_path(tmp: Path, session_id: str) -> Path:
    return tmp / "memories" / f"{session_id}.summary.md"


class ArchiveTest(unittest.TestCase):
    """jsonl 往返 / 脱敏 / files_changed 机械提取。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_append_and_read_roundtrip(self):
        mgr = make_manager(self.tmp)
        mgr.append_message(Message(role="user", content="你好"))
        mgr.append_message(tool_call_message("write_file", {"path": "a.py", "content": "x"}))
        records = read_archive(self.tmp / "memories", mgr.session_id)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["role"], "user")
        self.assertEqual(records[0]["content"], "你好")
        self.assertEqual(records[1]["role"], "assistant")
        self.assertEqual(records[1]["tool_calls"][0]["function"]["name"], "write_file")

    def test_redact_api_key_and_password(self):
        self.assertEqual(redact("key=sk-abcdef1234567890"), "key=sk-***")
        self.assertEqual(redact("api_key: sk-abcdef1234567890"), "api_key: ***")
        self.assertEqual(redact("password= hunter2 secret"), "password= *** secret")
        self.assertEqual(redact("Authorization: Bearer abc.def.ghi12345"), "Authorization: Bearer ***")
        self.assertEqual(redact("ghp_abcdefghijklmnopqrstuvwxyz123"), "ghp_***")
        self.assertEqual(redact("sk-abc"), "sk-abc")  # 太短不脱敏

    def test_archive_redacts_secrets_before_write(self):
        mgr = make_manager(self.tmp)
        mgr.append_message(Message(role="user", content="我的 key 是 sk-abcdef1234567890"))
        records = read_archive(self.tmp / "memories", mgr.session_id)
        self.assertIn("sk-***", records[0]["content"])
        self.assertNotIn("sk-abcdef1234567890", records[0]["content"])

    def test_extract_files_changed_mechanical(self):
        records = [
            json.loads(json.dumps({
                "role": "assistant",
                "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
            })),
            json.loads(json.dumps({
                "role": "assistant",
                "tool_calls": [{"function": {"name": "write_file", "arguments": '{"path": "a.py"}'}}],
            })),
            json.loads(json.dumps({
                "role": "assistant",
                "tool_calls": [{"function": {"name": "edit_file", "arguments": '{"path": "b.py"}'}}],
            })),
            json.loads(json.dumps({
                "role": "assistant",
                "tool_calls": [{"function": {"name": "rename_dir", "arguments": '{"src": "old", "dst": "new"}'}}],
            })),
            json.loads(json.dumps({
                "role": "assistant",
                "tool_calls": [{"function": {"name": "list_files", "arguments": '{"path": "c.py"}'}}],
            })),
        ]
        self.assertEqual(
            extract_files_changed(records),
            ["a.py", "b.py", "old", "new"],
        )

    def test_append_archive_direct_creates_dirs(self):
        append_archive(self.tmp / "memories", "session_direct", Message(role="user", content="直接调用"))
        records = read_archive(self.tmp / "memories", "session_direct")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["content"], "直接调用")

    def test_read_archive_skips_broken_lines(self):
        path = archive_path(self.tmp / "memories", "session_broken")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"role": "user", "content": "正常1"}\n'
            "这不是合法 JSON 行\n"
            '{"role": "user", "content": "正常2"}\n'
            "\n",
            encoding="utf-8",
        )
        records = read_archive(self.tmp / "memories", "session_broken")
        self.assertEqual([r["content"] for r in records], ["正常1", "正常2"])

    def test_redact_separator_and_case_variants(self):
        self.assertEqual(redact("API_KEY=sk-abcdef1234567890"), "API_KEY= ***")
        self.assertEqual(redact("token=abc123"), "token= ***")
        self.assertEqual(redact("apiKey: abc123"), "apiKey: ***")
        self.assertEqual(redact("secret = xyz789"), "secret= ***")
        self.assertEqual(redact("DEEPSEEK_API_KEY=sk-abcdef1234567890"), "DEEPSEEK_API_KEY= ***")
        # 无分隔符的值不脱敏（v1 覆盖范围：仅 key-value 与 Bearer 形态）。
        self.assertEqual(redact("password hunter2"), "password hunter2")
        # 无敏感信息原样。
        self.assertEqual(redact("这里没有敏感信息"), "这里没有敏感信息")

    def test_redact_nested_values(self):
        from myagent.memory.archive import _redact_value

        value = {
            "api_key": "sk-abcdef1234567890",
            "nested": {"password": "hunter2", "ok": "保留"},
            "items": ["sk-abcdef1234567890", "普通"],
        }
        cleaned = _redact_value(value)
        dumped = json.dumps(cleaned, ensure_ascii=False)
        self.assertNotIn("sk-abcdef1234567890", dumped)
        self.assertNotIn("hunter2", dumped)
        self.assertIn("保留", dumped)
        # 敏感键值被整体标红且保持 JSON 结构合法。
        self.assertEqual(cleaned["api_key"], "***")
        self.assertEqual(cleaned["nested"]["password"], "***")
        json.loads(dumped)

    def test_extract_files_changed_dedup_preserves_order(self):
        records = [
            _tool_record("write_file", {"path": "a.py"}),
            _tool_record("write_file", {"path": "a.py"}),  # 重复
            _tool_record("edit_file", {"path": "b.py"}),
        ]
        self.assertEqual(extract_files_changed(records), ["a.py", "b.py"])

    def test_extract_files_changed_skips_invalid(self):
        records = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": "不是JSON"}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": "{}"}}]},  # 缺 path
            {"role": "assistant", "tool_calls": [{}]},  # 缺 function
            {"role": "assistant", "tool_calls": None},  # None 防御
        ]
        self.assertEqual(extract_files_changed(records), [])

    def test_extract_files_changed_ignores_non_assistant(self):
        records = [
            {"role": "tool", "content": "结果"},
            {"role": "user", "content": "指令"},
            {"role": "assistant", "content": "纯文本"},
        ]
        self.assertEqual(extract_files_changed(records), [])


def _tool_record(name: str, args: dict) -> dict:
    """构造存档用的 assistant tool_calls 记录（简化版）。"""
    import json as _json

    return {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": name, "arguments": _json.dumps(args)}}
        ],
    }


class SummaryWriteTest(unittest.TestCase):
    """摘要写盘内容（files_changed 机械组装 + 降级）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_summary_writes_full_contract(self):
        llm = FakeLLM([
            '{"rollout_summary": "创建了 a.py", "raw_memory": "项目约定：目录勿放根路径"}',
            "# 跨会话记忆\n\n聚合内容",
        ])
        mgr = make_manager(self.tmp, llm=llm)
        mgr.append_message(Message(role="user", content="帮我建个 a.py"))
        mgr.append_message(tool_call_message("write_file", {"path": "a.py", "content": "x"}))
        mgr.append_message(final_step_message("完成"))
        mgr.close_session()
        mgr.sweep()

        payload_path = summary_payload_path(self.tmp, mgr.session_id)
        self.assertTrue(payload_path.is_file())
        text = payload_path.read_text(encoding="utf-8")
        self.assertIn("创建了 a.py", text)
        # files_changed 由机械提取组装：解析 JSON 契约验证。
        json_block = text.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
        payload = json.loads(json_block)
        self.assertEqual(payload["files_changed"], ["a.py"])

    def test_summary_falls_back_on_bad_json(self):
        llm = FakeLLM(["这段输出完全不是 JSON"])
        mgr = make_manager(self.tmp, llm=llm)
        mgr.append_message(Message(role="user", content="x"))
        mgr.close_session()
        mgr.sweep()
        text = summary_payload_path(self.tmp, mgr.session_id).read_text(encoding="utf-8")
        self.assertIn("这段输出完全不是 JSON", text)
        self.assertIn('"files_changed": []', text)


class SweepStateMachineTest(unittest.TestCase):
    """水位线状态机：closed 无条件、done 幂等、open 静止阈值、retry 上限。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _ready_manager(self, llm=None) -> MemoryManager:
        """已 close 且含一条消息的会话。"""
        mgr = make_manager(self.tmp, llm=llm, idle_threshold=0)
        mgr.append_message(Message(role="user", content="x"))
        mgr.close_session()
        return mgr

    def test_closed_session_is_processed(self):
        llm = FakeLLM(['{"rollout_summary": "s1", "raw_memory": "m1"}', "# 聚合"])
        mgr = self._ready_manager(llm=llm)
        mgr.sweep()
        # 1 次单会话提取 + 1 次跨会话聚合。
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(summary_payload_path(self.tmp, mgr.session_id).is_file())
        state = mgr._load_state(mgr.session_id)
        self.assertEqual(state["status"], STATUS_DONE)

    def test_done_session_skipped_when_unchanged(self):
        llm = FakeLLM(['{"rollout_summary": "s1", "raw_memory": "m1"}', "# 聚合"])
        mgr = self._ready_manager(llm=llm)
        mgr.sweep()
        self.assertEqual(len(llm.calls), 2)
        mgr.sweep()  # 幂等：mtime 未变、摘要未变 → 跳过，零调用
        self.assertEqual(len(llm.calls), 2)

    def test_open_session_skipped_when_fresh(self):
        llm = FakeLLM(['{"rollout_summary": "s1", "raw_memory": "m1"}'])
        mgr = make_manager(self.tmp, llm=llm, idle_threshold=100000)
        mgr.append_message(Message(role="user", content="x"))
        mgr.sweep()
        self.assertEqual(len(llm.calls), 0)

    def test_failed_then_retried_and_retry_cap(self):
        failing = FakeLLM()
        mgr = self._ready_manager(llm=failing)
        # 直接制造失败状态：手写 failed 状态。
        mgr._save_state(
            mgr.session_id,
            {"status": STATUS_FAILED, "retry": 1, "archive_mtime": 1.0},
        )
        # retry=1 < 3 → 再次尝试，这次成功。
        good = FakeLLM(['{"rollout_summary": "ok", "raw_memory": "m"}', "# 聚合"])
        mgr2 = make_manager(self.tmp, llm=good, idle_threshold=0)
        mgr2.session_id = mgr.session_id
        mgr2.sweep()
        self.assertEqual(len(good.calls), 2)  # extract + aggregate
        state = mgr2._load_state(mgr.session_id)
        self.assertEqual(state["status"], STATUS_DONE)
        self.assertEqual(state["retry"], 0)

    def test_retry_cap_gives_up(self):
        mgr = self._ready_manager()
        mgr._save_state(
            mgr.session_id,
            {"status": STATUS_FAILED, "retry": 3, "archive_mtime": 1.0},
        )
        # 重试机会耗尽 → failed_done，不再调用 LLM。
        llm = FakeLLM()
        mgr2 = make_manager(self.tmp, llm=llm, idle_threshold=0)
        mgr2.session_id = mgr.session_id
        mgr2.sweep()
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(mgr2._load_state(mgr.session_id)["status"], "failed_done")

    def test_closed_empty_archive_marks_done_without_llm(self):
        llm = FakeLLM()
        mgr = make_manager(self.tmp, llm=llm)
        archive_path(self.tmp / "memories", mgr.session_id).write_text("", encoding="utf-8")
        mgr._save_state(mgr.session_id, {"status": STATUS_CLOSED})
        mgr.sweep()
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(mgr._load_state(mgr.session_id)["status"], STATUS_DONE)

    def test_done_session_reprocessed_when_archive_modified(self):
        llm = FakeLLM([
            '{"rollout_summary": "s1", "raw_memory": "m1"}', "# 聚合",
            '{"rollout_summary": "s1+", "raw_memory": "m2"}', "# 聚合2",
        ])
        mgr = self._ready_manager(llm=llm)
        mgr.sweep()
        self.assertEqual(len(llm.calls), 2)
        # 追加新消息 → mtime 变化 → 重新汇总。
        mgr.append_message(Message(role="user", content="补充"))
        mgr.close_session()
        mgr.sweep()
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(mgr._load_state(mgr.session_id)["status"], STATUS_DONE)

    def test_open_idle_session_processed(self):
        llm = FakeLLM(['{"rollout_summary": "s", "raw_memory": "m"}', "# 聚合"])
        mgr = make_manager(self.tmp, llm=llm, idle_threshold=100000)
        mgr.append_message(Message(role="user", content="x"))
        # 把 jsonl mtime 拨到 100001 秒前，模拟进程被杀后的陈旧 open 会话。
        path = archive_path(self.tmp / "memories", mgr.session_id)
        old = datetime.now().timestamp() - 100001
        os.utime(path, (old, old))
        mgr.sweep()
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(mgr._load_state(mgr.session_id)["status"], STATUS_DONE)

    def test_failed_retry_increments_on_real_exception(self):
        class Boom:
            def complete(self, messages, *, max_new_tokens=None):
                raise RuntimeError("API boom")

        mgr = self._ready_manager(llm=Boom())
        mgr.sweep()
        state = mgr._load_state(mgr.session_id)
        self.assertEqual(state["status"], STATUS_FAILED)
        self.assertEqual(state["retry"], 1)
        self.assertIn("API boom", state["last_error"])

    def test_retry_chain_gives_up_after_three(self):
        class Boom:
            def complete(self, messages, *, max_new_tokens=None):
                raise RuntimeError("API boom")

        mgr = make_manager(self.tmp, llm=Boom(), idle_threshold=0)
        mgr.append_message(Message(role="user", content="x"))
        mgr.close_session()
        for expected in (1, 2, 3):
            mgr.sweep()
            state = mgr._load_state(mgr.session_id)
            self.assertEqual(state["retry"], expected)
        # 第 4 次：retry=3 == max_retry → 放弃，不再调用。
        mgr.sweep()
        state = mgr._load_state(mgr.session_id)
        self.assertEqual(state["status"], "failed_done")
        self.assertEqual(state["retry"], 3)

    def test_close_session_marks_closed(self):
        mgr = make_manager(self.tmp)
        mgr.append_message(Message(role="user", content="x"))
        mgr.close_session()
        self.assertEqual(mgr._load_state(mgr.session_id)["status"], STATUS_CLOSED)

    def test_close_session_noop_without_archive(self):
        mgr = make_manager(self.tmp)
        mgr.close_session()
        self.assertFalse(
            (self.tmp / "memories" / ".state" / f"{mgr.session_id}.json").exists()
        )

    def test_sweep_safe_when_memory_dir_missing(self):
        mgr = make_manager(self.tmp, llm=FakeLLM())
        shutil.rmtree(self.tmp / "memories")
        mgr.sweep()  # 不抛异常

    def test_multi_session_sweep_aggregates_all(self):
        llm = FakeLLM([
            '{"rollout_summary": "s1", "raw_memory": "m1"}',
            '{"rollout_summary": "s2", "raw_memory": "m2"}',
            "# 跨会话记忆（两个会话）",
        ])
        for sid in ("session_a", "session_b"):
            m = make_manager(self.tmp, llm=llm, session_id=sid)
            m.append_message(Message(role="user", content=f"内容 {sid}"))
            m.close_session()
        # 新 manager 扫描两个会话。
        MemoryManager(memory_dir=self.tmp / "memories", llm=llm).sweep()
        self.assertEqual(len(llm.calls), 3)
        summary = (self.tmp / "memories" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("跨会话记忆", summary)

    def test_corrupt_state_file_treated_as_empty(self):
        state_dir = self.tmp / "memories" / ".state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "session_x.json").write_text("{bad json", encoding="utf-8")
        mgr = make_manager(self.tmp, llm=FakeLLM())
        self.assertEqual(mgr._load_state("session_x"), {})


class AggregateTest(unittest.TestCase):
    """跨会话聚合：内容变化才重写，无变化零调用。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _complete_session(self, mgr: MemoryManager, llm: FakeLLM) -> None:
        mgr.append_message(Message(role="user", content="x"))
        mgr.close_session()
        mgr.sweep()

    def test_aggregate_writes_summary_and_is_idempotent(self):
        llm = FakeLLM([
            '{"rollout_summary": "s1", "raw_memory": "m1"}',  # extract
            "# 跨会话记忆\n\n做过 X。",                          # aggregate
        ])
        mgr = make_manager(self.tmp, llm=llm)
        self._complete_session(mgr, llm)
        summary_file = self.tmp / "memories" / "summary.md"
        self.assertTrue(summary_file.is_file())
        self.assertIn("做过 X", summary_file.read_text(encoding="utf-8"))
        # 第二次 sweep：摘要没变 → 聚合幂等，不再调 LLM。
        mgr.sweep()
        self.assertEqual(len(llm.calls), 2)

    def test_aggregate_error_recorded_for_retry(self):
        failing = FakeLLM()
        mgr = make_manager(self.tmp, llm=failing)
        mgr._save_state(mgr.session_id, {"status": STATUS_DONE, "archive_mtime": 1.0})
        # 直接放一个单会话摘要文件，触发聚合。
        (self.tmp / "memories" / f"{mgr.session_id}.summary.md").write_text(
            "# 会话摘要", encoding="utf-8"
        )
        mgr.sweep()
        manifest = mgr._load_manifest()
        self.assertIsNotNone(manifest.get("aggregate_error"))
        self.assertFalse((self.tmp / "memories" / "summary.md").exists())


class ContextBlockTest(unittest.TestCase):
    """注入文本：截断、空内容返回空串。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_context_block_truncates(self):
        mgr = make_manager(self.tmp, max_context_chars=20)
        (self.tmp / "memories").mkdir(parents=True, exist_ok=True)
        (self.tmp / "memories" / "summary.md").write_text("A" * 100, encoding="utf-8")
        block = mgr.context_block()
        # 截断到 20 字符 + 截断后缀 + 标题段（约 10 字符），总长应远小于原文。
        self.assertLess(len(block), 50)
        self.assertIn("已截断", block)
        self.assertIn("跨会话记忆", block)

    def test_context_block_empty_when_no_summary(self):
        mgr = make_manager(self.tmp)
        self.assertEqual(mgr.context_block(), "")

    def test_context_block_full_content_without_truncation(self):
        mgr = make_manager(self.tmp, max_context_chars=1000)
        (self.tmp / "memories").mkdir(parents=True, exist_ok=True)
        (self.tmp / "memories" / "summary.md").write_text("短内容", encoding="utf-8")
        self.assertEqual(mgr.context_block(), "## 跨会话记忆\n\n短内容")

    def test_context_block_whitespace_only_empty(self):
        (self.tmp / "memories").mkdir(parents=True, exist_ok=True)
        (self.tmp / "memories" / "summary.md").write_text("   \n\n  ", encoding="utf-8")
        self.assertEqual(make_manager(self.tmp).context_block(), "")


class AgentLoopMemoryTest(unittest.TestCase):
    """AgentLoop 注入记忆段：有记忆前置，None 时行为不变。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _loop_with_memory(self, llm, memory) -> AgentLoop:
        params = AgentParams(cwd=str(self.tmp))
        return AgentLoop(params, llm=llm, memory=memory)

    def test_system_prompt_prepends_memory_block(self):
        (self.tmp / "memories").mkdir(parents=True, exist_ok=True)
        (self.tmp / "memories" / "summary.md").write_text("记忆内容", encoding="utf-8")
        mgr = make_manager(self.tmp)
        llm = FakeLLM(['{"action": "final", "answer": "ok"}'])
        loop = self._loop_with_memory(llm, mgr)
        loop.run(AgentRequest(user_input="hi"))
        system = llm.calls[0][0].content
        self.assertIn("## 跨会话记忆", system)
        self.assertIn("记忆内容", system)

    def test_loop_without_memory_unchanged(self):
        llm = FakeLLM(['{"action": "final", "answer": "ok"}'])
        loop = AgentLoop(AgentParams(cwd=str(self.tmp)), llm=llm)
        loop.run(AgentRequest(user_input="hi"))
        system = llm.calls[0][0].content
        self.assertNotIn("跨会话记忆", system)

    def test_memory_block_prepended_before_environment(self):
        (self.tmp / "memories").mkdir(parents=True, exist_ok=True)
        (self.tmp / "memories" / "summary.md").write_text("记忆内容", encoding="utf-8")
        mgr = make_manager(self.tmp)
        llm = FakeLLM(['{"action": "final", "answer": "ok"}'])
        loop = self._loop_with_memory(llm, mgr)
        loop.run(AgentRequest(user_input="hi"))
        system = llm.calls[0][0].content
        # 记忆段置顶，环境段紧随其后。
        self.assertTrue(system.startswith("## 跨会话记忆"))
        self.assertIn("工作环境与角色", system)

    def test_empty_memory_block_not_injected(self):
        # 有 memory 但 summary.md 为空 → 不注入记忆段（环境段正常）。
        mgr = make_manager(self.tmp)
        llm = FakeLLM(['{"action": "final", "answer": "ok"}'])
        loop = self._loop_with_memory(llm, mgr)
        loop.run(AgentRequest(user_input="hi"))
        system = llm.calls[0][0].content
        self.assertNotIn("跨会话记忆", system)
        self.assertIn("工作环境与角色", system)


class SweepWithoutLLMTest(unittest.TestCase):
    """无 LLM 构造：仅存档器，sweep 无副作用。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_sweep_noop_without_llm(self):
        mgr = make_manager(self.tmp, llm=None)
        mgr.append_message(Message(role="user", content="x"))
        mgr.close_session()
        mgr.sweep()
        self.assertFalse(summary_payload_path(self.tmp, mgr.session_id).exists())


class SummarizerDirectTest(unittest.TestCase):
    """Summarizer 直接行为（不经状态机）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        from myagent.agent_config import PROMPT_DIR

        self.prompts = PROMPT_DIR

    def test_aggregate_strips_code_fence(self):
        llm = FakeLLM(["```markdown\n# 聚合\n内容\n```"])
        summarizer = Summarizer(llm, self.prompts)
        self.assertEqual(
            summarizer.aggregate(["# 会话摘要 s1"]),
            "# 聚合\n内容",
        )

    def test_aggregate_empty_input(self):
        summarizer = Summarizer(FakeLLM(), self.prompts)
        self.assertEqual(summarizer.aggregate([]), "")

    def test_aggregate_llm_exception_raises(self):
        class Boom:
            def complete(self, messages, *, max_new_tokens=None):
                raise RuntimeError("网络挂了")

        summarizer = Summarizer(Boom(), self.prompts)
        with self.assertRaises(SummarizeError):
            summarizer.aggregate(["# 会话摘要 s1"])

    def test_extract_llm_exception_raises(self):
        class Boom:
            def complete(self, messages, *, max_new_tokens=None):
                raise RuntimeError("网络挂了")

        summarizer = Summarizer(Boom(), self.prompts)
        with self.assertRaises(SummarizeError):
            summarizer.extract("s1", [{"role": "user", "content": "x"}])

    def test_render_transcript_various_roles(self):
        records = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "content": "文件内容"},
            {"role": "assistant", "content": "最终答案"},
            {"role": "assistant", "content": "   "},  # 空白跳过
            {"role": "user"},  # 无 content 跳过
        ]
        text = render_transcript(records)
        self.assertIn("user: 你好", text)
        self.assertIn('[read_file] {"path": "a.py"}', text)
        self.assertIn("→ 工具结果：文件内容", text)
        self.assertIn("assistant: 最终答案", text)

    def test_render_transcript_truncates_long_tool_result(self):
        records = [{"role": "tool", "content": "X" * 1000}]
        text = render_transcript(records)
        self.assertLess(len(text), 600)
        self.assertIn("X" * 500, text)

    def test_extract_truncates_oversized_transcript(self):
        llm = FakeLLM(['{"rollout_summary": "ok", "raw_memory": "m"}'])
        summarizer = Summarizer(llm, self.prompts, max_transcript_chars=100)
        summarizer.extract("s1", [{"role": "user", "content": "A" * 500}])
        user_msg = llm.calls[0][1]
        self.assertIn("已截断", user_msg.content)
        self.assertLess(len(user_msg.content), 200)

    def test_parse_json_object_variants(self):
        self.assertEqual(_parse_json_object('前文 {"a": 1} 后文'), {"a": 1})
        self.assertEqual(_parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_parse_json_object('{"a": 1} 尾随'), {"a": 1})
        self.assertIsNone(_parse_json_object("完全没有对象"))
        self.assertIsNone(_parse_json_object("{broken"))
        self.assertIsNone(_parse_json_object(""))


class CliMemoryTest(unittest.TestCase):
    """REPL / one_shot 的记忆接线：每轮存档消息、退出标记会话。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_repl_archives_messages_and_closes_on_exit(self):
        step = StepRecord(
            turn=1,
            raw_output='{"action": "final", "answer": "好"}',
            action=FinalAnswer(text="好"),
        )
        loop = FakeLoop([final_response("好", [step])])
        memory = FakeMemory()
        with mock.patch("builtins.input", side_effect=["hi", "/exit"]):
            code = _repl(loop, memory)
        self.assertEqual(code, 0)
        self.assertTrue(memory.closed)
        self.assertEqual([m.role for m in memory.appended], ["user", "assistant"])
        self.assertEqual(memory.appended[0].content, "hi")

    def test_repl_archives_and_closes_on_eof(self):
        loop = FakeLoop([final_response("好")])
        memory = FakeMemory()
        with mock.patch("builtins.input", side_effect=EOFError):
            code = _repl(loop, memory)
        self.assertEqual(code, 0)
        self.assertTrue(memory.closed)
        self.assertEqual(memory.appended, [])

    def test_one_shot_archives_and_closes(self):
        step = StepRecord(
            turn=1,
            raw_output='{"action": "final", "answer": "完成"}',
            action=FinalAnswer(text="完成"),
        )
        loop = FakeLoop([final_response("完成", [step])])
        memory = FakeMemory()
        with mock.patch("builtins.input", return_value="任务描述"):
            code = _one_shot(loop, memory)
        self.assertEqual(code, 0)
        self.assertTrue(memory.closed)
        # assistant 消息存档内容与 step_to_messages 一致（携带模型原始输出）。
        self.assertEqual(
            [m.content for m in memory.appended],
            ["任务描述", '{"action": "final", "answer": "完成"}'],
        )

    def test_archive_steps_appends_tool_pair(self):
        memory = FakeMemory()
        raw = '{"action": "tool_call", "tool": "read_file", "args": {"path": "a.py"}}'
        step = StepRecord(
            turn=1,
            raw_output=raw,
            action=ToolCall(name="read_file", args={"path": "a.py"}, raw=raw),
            observation="内容",
        )
        _archive_steps(memory, final_response(steps=[step]))
        self.assertEqual([m.role for m in memory.appended], ["assistant", "tool"])
        self.assertEqual(memory.appended[1].tool_call_id, "call_1")

    def test_repl_without_memory_unchanged(self):
        # memory=None 时 REPL 仍正常，不碰任何记忆接口。
        loop = FakeLoop([final_response("好")])
        with mock.patch("builtins.input", side_effect=["hi", "/exit"]):
            code = _repl(loop)
        self.assertEqual(code, 0)
        self.assertEqual(len(loop.requests), 1)


class FakeLoop:
    """记录每次 run 请求的最小循环桩。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class FakeMemory:
    """记录 append_message / close_session 调用的最小记忆桩。"""

    def __init__(self):
        self.session_id = "session_test"
        self.appended = []
        self.closed = False

    def context_block(self, max_chars=None):
        return ""

    def append_message(self, message):
        self.appended.append(message)

    def close_session(self):
        self.closed = True

    def sweep(self):
        pass


def final_response(answer="好", steps=None):
    return AgentResponse(
        final_answer=answer,
        stop_reason=StopReason.FINAL_ANSWER,
        turns_used=1,
        tool_calls=0,
        steps=steps or [],
    )


if __name__ == "__main__":
    unittest.main()
