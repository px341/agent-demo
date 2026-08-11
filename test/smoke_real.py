"""真实链路冒烟：DeepSeek 端点 + 主循环 + 工具链路（一次性验证脚本，会真实调用 API 消耗额度）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams
from myagent.agent_loop import AgentLoop
from myagent.contracts import AgentRequest, StopReason
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
        obs = (s.observation or "")[:80]
        raw = (s.raw_output or "").replace("\n", " ")[:150]
        print(f"  turn{s.turn}: {raw}")
        print(f"           action={type(s.action).__name__} obs={obs!r}")


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
