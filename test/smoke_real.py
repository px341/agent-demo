"""真实链路冒烟：DeepSeek 端点 + 主循环 + 工具链路（一次性验证脚本，会真实调用 API 消耗额度）。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams, Message
from myagent.agent_loop import AgentLoop, step_to_messages
from myagent.contracts import AgentRequest, StopReason
from myagent.memory import MemoryManager
from myagent.provider import OpenAICompatibleModelClient
from myagent.tools import ToolExecutor


class DenyGate:
    """冒烟用：拒绝所有工具调用。"""

    def request(self, name, args):
        return False


def run_case(
    title: str,
    user_input: str,
    use_tools: bool,
    approval_gate=None,
) -> None:
    print(f"\n===== {title} =====")
    params = AgentParams(cwd=".", max_turns=5)
    client = OpenAICompatibleModelClient(params)
    loop = AgentLoop(
        params,
        llm=client,
        tools=ToolExecutor(params.cwd) if use_tools else None,
        approval_gate=approval_gate,
    )
    resp = loop.run(AgentRequest(user_input=user_input))
    print("stop_reason:", resp.stop_reason.value)
    print("turns_used:", resp.turns_used, "tool_calls:", resp.tool_calls)
    if resp.error:
        print("error:", resp.error)
    if resp.final_answer:
        print("final:", resp.final_answer)
    for s in resp.steps:
        obs = " | ".join(s.observations)[:80]
        raw = (s.raw_output or "").replace("\n", " ")[:150]
        print(f"  turn{s.turn}: {raw}")
        action = "tool_calls" if s.tool_calls else "final_answer"
        print(f"           action={action} obs={obs!r}")


def run_memory_case() -> None:
    """记忆链路：真实 LLM 单会话摘要提取 → 跨会话聚合 → 注入 system prompt。"""
    print("\n===== 记忆链路（单会话摘要 → 聚合 → 注入） =====")
    client = OpenAICompatibleModelClient(AgentParams(cwd=".", max_turns=5))

    with tempfile.TemporaryDirectory() as tmp:
        memory_dir = Path(tmp) / "memories"
        print(f"记忆目录（临时，避免污染真实 memories/）：{memory_dir}")

        # 1) 模拟一次会话：真实主循环跑一轮，按 CLI 同款逻辑存档轨迹消息。
        mgr = MemoryManager(memory_dir=memory_dir, llm=client)
        loop = AgentLoop(
            AgentParams(cwd=".", max_turns=5),
            llm=client,
            tools=ToolExecutor("."),
            memory=mgr,
        )
        user_input = (
            "创建一个文件 smoke_memory_probe.txt，内容写「记忆冒烟测试」，"
            "然后告诉我完成了。"
        )
        mgr.append_message(Message(role="user", content=user_input))
        resp = loop.run(AgentRequest(user_input=user_input))
        print("stop_reason:", resp.stop_reason.value)
        for step in resp.steps:
            for message in step_to_messages(step):
                mgr.append_message(message)
        mgr.close_session()

        # 2) sweep：真实调用 LLM 做单会话摘要 + 跨会话聚合。
        print("→ sweep()：真实 LLM 提取单会话摘要 + 聚合 summary.md")
        mgr.sweep()

        session_jsonl = list(memory_dir.glob("session_*.jsonl"))
        summary_files = list(memory_dir.glob("*.summary.md"))
        aggregate_file = memory_dir / "summary.md"
        print(f"会话存档：{len(session_jsonl)} 个")
        print(f"单会话摘要：{len(summary_files)} 个")
        if summary_files:
            print(f"--- {summary_files[0].name} ---")
            print(summary_files[0].read_text(encoding="utf-8")[:800])
        print(f"聚合 summary.md 存在：{aggregate_file.exists()}")
        if aggregate_file.exists():
            print(f"--- summary.md（前 600 字符）---")
            print(aggregate_file.read_text(encoding="utf-8")[:600])

        # 3) 注入验证：新会话的 system prompt 应包含跨会话记忆段。
        mgr2 = MemoryManager(memory_dir=memory_dir, llm=client)
        loop2 = AgentLoop(
            AgentParams(cwd=".", max_turns=3),
            llm=client,
            memory=mgr2,
        )
        resp2 = loop2.run(AgentRequest(user_input="在 system prompt 里能看到跨会话记忆吗？"))
        print("注入后 stop_reason:", resp2.stop_reason.value)
        if resp2.final_answer:
            print("注入后 final:", resp2.final_answer)

        # 4) 幂等验证：再次 sweep 不应重新调 LLM 改摘要（观察 .state done 状态）。
        mgr2.sweep()
        print("幂等：第二次 sweep 后状态", mgr2._load_state(mgr2.session_id))
        print("幂等：summary.md mtime 未变 =", aggregate_file.stat().st_mtime if aggregate_file.exists() else "n/a")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "all"
    if case in ("all", "plain"):
        run_case("纯对话（无工具）", "用一句话介绍你自己", False)
    if case in ("all", "tool"):
        run_case("工具链路（list_files -> final）", "请查看当前目录下的文件列表，并告诉我有哪些文件", True)
    if case in ("all", "env"):
        run_case(
            "环境感知（直接用 workspace tree，不调工具）",
            "根据工作区信息，直接告诉我当前目录有哪些顶层文件和目录，不要调用任何工具。",
            False,
        )
    if case in ("all", "write"):
        run_case(
            "写入链路（create_file + read_file + delete_file）",
            "请创建一个文件 test/smoke_probe.txt 内容为 probe，读取确认内容后再删除它，最后告诉我完成了。",
            True,
        )
    if case in ("all", "deny"):
        run_case(
            "审批拒绝链路（所有工具被拒）",
            "请查看当前目录的文件列表并告诉我有哪些文件。",
            True,
            approval_gate=DenyGate(),
        )
    if case in ("all", "memory"):
        run_memory_case()
