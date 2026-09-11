"""环境 prompt 构建测试：workspace tree 生成 + 占位符替换 + 主循环集成。

不触网：FakeLLM 捕获系统消息断言环境信息注入。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.agent_config import AgentParams
from myagent.agent_loop import AgentLoop
from myagent.contracts import AgentRequest, LLMResponse
from myagent.environment import (
    build_environment_prompt,
    build_workspace_tree,
)

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "myagent" / "prompts"
)


class FakeLLM:
    def __init__(self, output: str):
        self.output = output
        self.calls: list = []

    def complete(self, messages, *, tools=None, max_new_tokens=None):
        self.calls.append(list(messages))
        return LLMResponse(text=self.output)


class WorkspaceTreeTest(unittest.TestCase):
    def test_tree_includes_files_and_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("")
            (root / "sub").mkdir()
            (root / "sub" / "b.py").write_text("")

            tree = build_workspace_tree(root)

            self.assertIn("a.py", tree)
            self.assertIn("sub/", tree)
            self.assertIn("b.py", tree)

    def test_tree_ignores_internal_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv").mkdir()
            (root / ".venv" / "x").write_text("")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("")
            (root / "ok.py").write_text("")

            tree = build_workspace_tree(root)

            self.assertIn("ok.py", tree)
            self.assertNotIn(".venv", tree)
            self.assertNotIn(".git", tree)

    def test_tree_depth_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "d1" / "d2" / "d3").mkdir(parents=True)
            (root / "d1" / "d2" / "d3" / "deep.py").write_text("")

            shallow = build_workspace_tree(root, max_depth=1)
            self.assertIn("d1/", shallow)
            self.assertNotIn("d2", shallow)

            # max_depth=N：展示到第 N 层目录的名字，不展开其内容。
            mid = build_workspace_tree(root, max_depth=3)
            self.assertIn("d3/", mid)
            self.assertNotIn("deep.py", mid)

            # deep.py 在第 4 层，需要 max_depth=4。
            deep = build_workspace_tree(root, max_depth=4)
            self.assertIn("deep.py", deep)

            # max_depth<=0 被 clamp 到 1，与 max_depth=1 等价。
            self.assertEqual(
                build_workspace_tree(root, max_depth=0),
                shallow,
            )

    def test_missing_root(self):
        tree = build_workspace_tree(Path("/nonexistent/path-xyz"))
        self.assertIn("不存在", tree)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = build_workspace_tree(Path(tmp))
            self.assertEqual(tree, ".")

    def test_symlink_shown_as_leaf(self):
        """symlink 目录按叶子显示，不跟随递归（防循环）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "real").mkdir()
            (root / "real" / "inner.py").write_text("")
            try:
                (root / "loop").symlink_to(root, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"当前平台不允许创建符号链接：{exc}")

            tree = build_workspace_tree(root)

            self.assertIn("loop", tree)
            self.assertIn("inner.py", tree)
            # 不跟随 symlink：不会出现 loop/ 目录下的内容（即不展开）。
            self.assertNotIn("loop/", tree)


class EnvironmentPromptTest(unittest.TestCase):
    def test_placeholders_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hello.py").write_text("")

            prompt = build_environment_prompt(PROMPTS_DIR, root)

            self.assertIn("角色定义", prompt)
            self.assertIn("工作目录（cwd）", prompt)
            self.assertIn(str(root.resolve()), prompt)
            self.assertIn("hello.py", prompt)
            # 占位符全部替换，无残留。
            for placeholder in ("{cwd}", "{os}", "{python_version}",
                                "{git_repo}", "{now}", "{workspace_tree}"):
                self.assertNotIn(placeholder, prompt)


class AgentLoopEnvTest(unittest.TestCase):
    def test_system_message_contains_environment(self):
        llm = FakeLLM('{"action": "final", "answer": "好"}')
        loop = AgentLoop(AgentParams(), llm=llm)
        loop.run(AgentRequest(user_input="hi"))

        system = llm.calls[0][0]
        self.assertEqual(system.role, "system")
        # 环境 prompt：cwd + 工作区树条目（真实工作区含 myagent/）。
        self.assertIn("工作目录（cwd）", system.content)
        self.assertIn("myagent/", system.content)
        # 文件树不含 .venv 目录（忽略列表仅出现在说明文字中，目录带 / 后缀）。
        self.assertNotIn(".venv/", system.content)
        # 工具 prompt 拼接在环境 prompt 之后；工具列表由 function calling
        # 协议提供（不再嵌进 prompt）。
        self.assertIn("## 五、规则", system.content)
        self.assertIn("## 二、工具使用方式", system.content)
        self.assertNotIn("{tool_list}", system.content)
        self.assertNotIn("### delete_dir", system.content)

    def test_environment_regenerated_each_run(self):
        """每次 run 重新生成系统提示（时间戳逐次刷新，内容有效）。"""
        llm = FakeLLM('{"action": "final", "answer": "好"}')
        loop = AgentLoop(AgentParams(), llm=llm)
        loop.run(AgentRequest(user_input="a"))
        loop.run(AgentRequest(user_input="b"))

        system_a = llm.calls[0][0].content
        system_b = llm.calls[1][0].content
        # 每次都是新生成的实例：关键内容都在，时间戳可能因跨秒不同。
        self.assertIn("工作目录（cwd）", system_a)
        self.assertIn("工作目录（cwd）", system_b)
        self.assertIn("myagent/", system_a)
        self.assertIn("myagent/", system_b)


if __name__ == "__main__":
    unittest.main()
