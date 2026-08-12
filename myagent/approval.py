"""命令行审批闸门：工具执行前在终端询问用户 y/n。

满足 contracts.ApprovalGate 协议，供 cli 注入主循环：

- 低风险（默认 ``read`` 只读）工具**免询问**直接放行；
- 其余工具输入 ``y``/``yes`` 允许、``n``/``no`` 拒绝，非法输入重问；
- EOF（Ctrl-D）按**拒绝**处理（安全默认）。

风险等级在工具注册时标注（ToolSpec.risk：read/write/delete），
可通过 ``auto_approve_risks`` 调整免审集合（如 ``set()`` 恢复全询问）。
"""
from __future__ import annotations

import json
from typing import Any

from .tools.registry import TOOLS

ALLOWED = ("y", "yes")
DENIED = ("n", "no")


class ConsoleApprovalGate:
    """在标准输入上询问用户是否允许调用工具（read 类免询问）。"""

    def __init__(self, auto_approve_risks: set[str] | None = None):
        #: 免询问的风险等级集合；默认只读工具直接放行。
        self.auto_approve_risks = (
            auto_approve_risks if auto_approve_risks is not None else {"read"}
        )

    def request(self, name: str, args: dict[str, Any]) -> bool:
        spec = TOOLS.get(name)
        if spec is not None and spec.risk in self.auto_approve_risks:
            return True
        args_text = json.dumps(args, ensure_ascii=False) if args else ""
        while True:
            try:
                answer = input(
                    f"是否允许调用工具 {name}（{args_text}）？[y/n] "
                ).strip().lower()
            except EOFError:
                print("（输入中止，默认拒绝）")
                return False
            if answer in ALLOWED:
                return True
            if answer in DENIED:
                return False
            print("请输入 y（允许）或 n（拒绝）")

    def request_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> bool:
        """一次性审批多个工具调用（并行）。

        全部为免审风险（默认 read）时直接放行；否则一次性列出全部调用
        与参数询问用户 y/n。
        """
        for name, _ in calls:
            spec = TOOLS.get(name)
            if spec is None or spec.risk not in self.auto_approve_risks:
                break
        else:
            return True

        lines = []
        for name, args in calls:
            args_text = json.dumps(args, ensure_ascii=False) if args else ""
            lines.append(f"  - {name}（{args_text}）")
        prompt = "是否允许调用以下工具？\n" + "\n".join(lines) + "\n[y/n] "
        while True:
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                print("（输入中止，默认拒绝）")
                return False
            if answer in ALLOWED:
                return True
            if answer in DENIED:
                return False
            print("请输入 y（允许）或 n（拒绝）")
