from .agent_config import AgentParams
from .provider import agent_turn
from .agent_config import Message
from .context_manager import ContextManager
from .tools import execute_tool

from pathlib import Path


class AgentLoop:
    # 单轮内层"思考-行动"循环上限，防止模型反复 tool_call / retry 死循环。
    MAX_INNER_TURNS = 20

    def __init__(self, agent_params: AgentParams):
        self.agent_params = agent_params
        self.cwd = Path(agent_params.cwd).resolve()
        self.max_turns = agent_params.max_turns
        self.context_manager = ContextManager(agent_params)

    def run(self):
        """进入交互式多轮对话循环。"""
        print(f"工作目录：{self.cwd}")

        for turn in range(self.max_turns):
            user_input = input("你：").strip()

            if user_input in ("/exit", "/quit"):
                print("已退出。")
                break

            if not user_input:
                print("没有输入内容，请重新输入。")
                continue

            self.context_manager.add_user_message(user_input)

            # 单轮"思考-行动"循环：模型可能连续调用工具/重试，直到给出 final 答案。
            for _ in range(self.MAX_INNER_TURNS):
                assistant = agent_turn(
                    self.agent_params,
                    user_input="",
                    provider=None,
                    max_new_tokens=self.agent_params.token_out_limit,
                )
                self.context_manager.add_message(assistant)

                action = assistant.metadata.get("action")

                if action == "tool_call":
                    call = assistant.tool_calls[0]
                    print(f"⚙️ 调用工具 {call['name']}，参数：{call['arguments']}")
                    result = execute_tool(call["name"], call["arguments"])
                    self.context_manager.add_message(
                        Message(
                            role="tool",
                            content=result,
                            tool_call_id=call["name"],
                        )
                    )
                    continue

                if action == "retry":
                    reason = assistant.metadata.get("reason", "")
                    print(f"⚠️ 输出需重试：{reason}")
                    self.context_manager.add_message(
                        Message(
                            role="user",
                            content=(
                                "输出不符合契约，请重新输出合法 JSON。"
                                f"原因：{reason}"
                            ),
                        )
                    )
                    continue

                # final：给出最终答案，结束本轮。
                print(f"🤖 {assistant.content}")
                break
            else:
                print("⚠️ 本轮内层循环达到上限，已停止。")

