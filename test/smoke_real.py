"""真实链路冒烟：DeepSeek 端点 + 主循环 + 工具链路（一次性验证脚本，会真实调用 API 消耗额度）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams
from myagent.agent_loop import AgentLoop
from myagent.contracts import AgentRequest, StopReason
from myagent.provider import OpenAICompatibleModelClient
from myagent.tools import execute_tool


def run_case(title: str, user_input: str, use_tools: bool) -> None:
    print(f"\n===== {title} =====")
    params = AgentParams(cwd=".", max_turns=5)
    client = OpenAICompatibleModelClient(params)
    loop = AgentLoop(
        params,
        llm=client,
        # execute_tool 是模块级函数，适配成带 execute 方法的对象。
        tools=SimpleNamespace(execute=execute_tool) if use_tools else None,
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
