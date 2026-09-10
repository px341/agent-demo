"""并行委派、单写者屏障及跨线程上下文的回归测试（不调用真实 LLM）。"""
from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

from myagent.agent_config import AgentParams
from myagent.agent_loop import step_to_messages
from myagent.contracts import AgentRequest, LLMResponse, StopReason
from myagent.errors import ExecutionError
from myagent.multi import build_orchestrator_loop, unregister_worker_tool
from myagent.multi.worker_tool import _current_host, _set_host, _clear_host
from myagent.tools import ToolExecutor
from test.test_multi import tc


class RoutedLLM:
    """按任务路由应答，工人之间不共享可变输出队列。"""

    def __init__(self, calls, worker):
        self.calls = calls
        self.worker = worker
        self.parent_messages = []

    def complete(self, messages, *, tools=None, max_new_tokens=None):
        task = next(m.content for m in messages if m.role == "user")
        if task == "parent":
            self.parent_messages.append(list(messages))
            if not any(m.role == "tool" for m in messages):
                return LLMResponse(text="", tool_calls=self.calls)
            return LLMResponse(text="汇总完成")
        return self.worker(task, messages, tools)


class ParallelDelegationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(unregister_worker_tool)

    def build(self, calls, worker, *, max_workers=2, timeout=None, tools=None, gate=None):
        llm = RoutedLLM(calls, worker)
        orch = build_orchestrator_loop(
            agent_params=AgentParams(cwd=self.root, tool_timeout=timeout),
            llm=llm,
            tools=tools or ToolExecutor(self.root, timeout=timeout),
            max_workers=max_workers,
            approval_gate=gate,
        )
        self.addCleanup(orch.close)
        return orch, llm

    def test_parallel_workers_are_bounded_and_results_keep_call_order(self):
        barrier = threading.Barrier(2, timeout=5)
        fast_finished = [threading.Event(), threading.Event()]
        lock = threading.Lock()
        active = peak = 0
        finished = []

        def worker(task, messages, tools):
            nonlocal active, peak
            index = int(task)
            self.assertIsNone(_current_host())
            self.assertEqual([m.content for m in messages if m.role == "user"], [task])
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if index < 4:
                    barrier.wait()
                    if index % 2 == 0:
                        self.assertTrue(fast_finished[index // 2].wait(5))
                with lock:
                    finished.append(index)
                if index < 4 and index % 2:
                    fast_finished[index // 2].set()
                return LLMResponse(text=f"result {index}")
            finally:
                with lock:
                    active -= 1

        calls = [tc("delegate_agent", {"task": str(i)}, f"d{i}") for i in range(5)]
        orch, llm = self.build(calls, worker)
        resp = orch.run(AgentRequest(user_input="parent"))
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(peak, 2)
        self.assertEqual(finished, [1, 0, 3, 2, 4])
        messages = [m for m in llm.parent_messages[-1] if m.role == "tool"]
        self.assertEqual([m.tool_call_id for m in messages], [f"d{i}" for i in range(5)])
        for i, message in enumerate(messages):
            self.assertTrue(message.content.startswith(f"DONE: result {i}"))
        self.assertIsNone(_current_host())

    def test_writes_wait_for_workers_and_run_on_single_owner(self):
        target = self.root / "shared.txt"
        target.write_text("0")
        owner_thread = threading.get_ident()
        barrier = threading.Barrier(2, timeout=5)
        lock = threading.Lock()
        events = []
        active = 0
        case = self

        class WriterExecutor(ToolExecutor):
            def execute(self, name, args):
                if name == "write_file":
                    case.assertEqual(threading.get_ident(), owner_thread)
                    with lock:
                        case.assertEqual(active, 0)
                        events.append(f"write:{args['content']}")
                return super().execute(name, args)

        def worker(task, messages, tools):
            nonlocal active
            with lock:
                active += 1
                events.append(f"start:{task}")
            try:
                expected = "0" if task in ("a", "b") else "2"
                self.assertEqual(target.read_text(), expected)
                barrier.wait()
                self.assertEqual(target.read_text(), expected)
                return LLMResponse(text=task)
            finally:
                with lock:
                    events.append(f"end:{task}")
                    active -= 1

        calls = [tc("delegate_agent", {"task": t}, t) for t in ("a", "b")]
        calls += [tc("write_file", {"path": "shared.txt", "content": v}, f"w{v}") for v in ("1", "2")]
        calls += [tc("delegate_agent", {"task": t}, t) for t in ("c", "d")]
        orch, _ = self.build(calls, worker, tools=WriterExecutor(self.root))
        resp = orch.run(AgentRequest(user_input="parent"))
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(target.read_text(), "2")
        self.assertTrue(all(o.status == "success" for o in resp.steps[0].outcomes))
        self.assertTrue(all("DONE:" in resp.steps[0].observations[i] for i in (0, 1, 4, 5)))
        self.assertLess(max(events.index("end:a"), events.index("end:b")), events.index("write:1"))
        self.assertLess(events.index("write:1"), events.index("write:2"))
        self.assertLess(events.index("write:2"), min(events.index("start:c"), events.index("start:d")))

    def test_timeout_threads_preserve_host_but_workers_cannot_delegate_or_write(self):
        barrier = threading.Barrier(2, timeout=5)

        def worker(task, messages, tools):
            self.assertIsNone(_current_host())
            names = {t["function"]["name"] for t in tools}
            self.assertNotIn("delegate_agent", names)
            self.assertNotIn("write_file", names)
            self.assertNotIn("run_shell", names)
            observations = [m.content for m in messages if m.role == "tool"]
            if not observations:
                barrier.wait()
                return LLMResponse(text="", tool_calls=[
                    tc("delegate_agent", {"task": "nested"}, "nested"),
                    tc("write_file", {"path": "forbidden", "content": task}, "write"),
                    tc("list_files", {"path": "."}, "read"),
                ])
            self.assertIn("ValidationError", observations[0])
            self.assertIn("ValidationError", observations[1])
            self.assertNotIn("ToolError", observations[2])
            return LLMResponse(text=f"safe {task}")

        orch, _ = self.build(
            [tc("delegate_agent", {"task": t}, t) for t in ("a", "b")], worker,
            timeout=10,
        )
        sentinel = object()
        token = _set_host(sentinel)
        try:
            resp = orch.run(AgentRequest(user_input="parent"))
            self.assertIs(_current_host(), sentinel)
        finally:
            _clear_host(token)
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertTrue(all(o.startswith("DONE: safe") for o in resp.steps[0].observations))
        self.assertFalse((self.root / "forbidden").exists())

    def test_max_workers_one_remains_serial(self):
        seen = []

        def worker(task, messages, tools):
            seen.append(task)
            return LLMResponse(text=task)

        orch, _ = self.build(
            [tc("delegate_agent", {"task": t}, t) for t in ("a", "b", "c")], worker,
            max_workers=1,
        )
        resp = orch.run(AgentRequest(user_input="parent"))
        self.assertEqual(seen, ["a", "b", "c"])
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)

    def test_expired_delegate_waits_for_worker_exit_before_returning(self):
        started = threading.Event()
        release = threading.Event()
        ended = threading.Event()

        def worker(task, messages, tools):
            started.set()
            try:
                self.assertTrue(release.wait(5))
                return LLMResponse(text="finished")
            finally:
                ended.set()

        orch, _ = self.build([
            tc("delegate_agent", {"task": "slow"}, "slow"),
            tc("write_file", {"path": "must-not-exist", "content": "x"}, "write"),
        ], worker, timeout=0.05)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(orch.run, AgentRequest(user_input="parent"))
            try:
                self.assertTrue(started.wait(5))
                # 超时不会放任工人继续在后台读取、同时允许下一轮写入。
                with self.assertRaises(FutureTimeout):
                    future.result(timeout=0.15)
            finally:
                release.set()
            resp = future.result(timeout=5)
        self.assertTrue(ended.is_set())
        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
        self.assertEqual(resp.steps[0].outcomes[0].error_type, "TimeoutError")
        self.assertEqual(resp.steps[0].outcomes[1].status, "skipped")
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_worker_failure_does_not_abort_other_workers(self):
        barrier = threading.Barrier(2, timeout=5)

        def worker(task, messages, tools):
            barrier.wait()
            if task == "bad":
                raise RuntimeError("worker failed")
            return LLMResponse(text="worker succeeded")

        orch, _ = self.build(
            [tc("delegate_agent", {"task": t}, t) for t in ("bad", "good")], worker,
        )
        resp = orch.run(AgentRequest(user_input="parent"))
        self.assertEqual(resp.stop_reason, StopReason.FINAL_ANSWER)
        self.assertIn("FAILED: worker failed", resp.steps[0].observations[0])
        self.assertIn("DONE: worker succeeded", resp.steps[0].observations[1])

    def test_fatal_parallel_call_keeps_started_results_and_skips_future_writes(self):
        barrier = threading.Barrier(2, timeout=5)

        class FailingExecutor(ToolExecutor):
            def execute(self, name, args):
                if name == "delegate_agent":
                    barrier.wait()
                    if args["task"] == "bad":
                        raise ExecutionError("dispatch failed")
                    return "DONE: completed"
                raise AssertionError("write must not run")

        orch, _ = self.build([
            tc("delegate_agent", {"task": "bad"}, "bad"),
            tc("delegate_agent", {"task": "good"}, "good"),
            tc("write_file", {"path": "x", "content": "x"}, "write"),
        ], None, tools=FailingExecutor(self.root))
        resp = orch.run(AgentRequest(user_input="parent"))
        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
        self.assertEqual(resp.tool_calls, 2)
        outcomes = resp.steps[0].outcomes
        self.assertEqual([o.status for o in outcomes], ["error", "success", "skipped"])
        self.assertTrue(outcomes[1].partial)
        self.assertEqual([o.tool_call_id for o in outcomes], ["bad", "good", "write"])
        self.assertEqual(len(step_to_messages(resp.steps[0])), 4)

    def test_approval_denial_starts_no_workers(self):
        case = self
        owner = threading.get_ident()

        class DenyGate:
            def request_batch(self, calls):
                case.assertEqual(threading.get_ident(), owner)
                return False

        def worker(*args):
            self.fail("worker must not start")

        orch, _ = self.build(
            [tc("delegate_agent", {"task": t}, t) for t in ("a", "b")], worker,
            gate=DenyGate(),
        )
        resp = orch.run(AgentRequest(user_input="parent"))
        self.assertEqual(resp.stop_reason, StopReason.TOOL_ERROR)
        self.assertEqual(resp.tool_calls, 0)
        self.assertTrue(all(o.status == "skipped" for o in resp.steps[0].outcomes))


if __name__ == "__main__":
    unittest.main()
