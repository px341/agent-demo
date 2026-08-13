"""工具系统测试：注册表、执行器、文件/目录工具、路径安全边界。

全部在临时目录内执行，不触网、不触碰真实工作区。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.tools import TOOLS, ToolExecutor, list_tools, register_tool
from myagent.tools.registry import to_openai_tools
from myagent.errors import (
    ExecutionError,
    NotFoundError,
    ToolError,
    ToolPermissionError,
    ToolTimeoutError,
    ValidationError,
)

EXPECTED_TOOLS = {
    "read_file",
    "create_file",
    "write_file",
    "edit_file",
    "delete_file",
    "list_files",
    "create_dir",
    "write_dir",
    "rename_dir",
    "delete_dir",
    "run_shell",
    "git_status",
    "git_diff",
    "git_apply_patch",
}


class RegistryTest(unittest.TestCase):
    def test_builtin_tools_registered(self):
        self.assertEqual(set(TOOLS), EXPECTED_TOOLS)

    def test_list_tools_sorted(self):
        names = list_tools()
        self.assertEqual(names, sorted(names))
        self.assertEqual(set(names), EXPECTED_TOOLS)

    def test_tool_spec_metadata(self):
        spec = TOOLS["read_file"]
        self.assertEqual(spec.name, "read_file")
        self.assertIn("读取", spec.description)
        self.assertIn("path", spec.parameters)

    def test_risk_levels(self):
        """风险等级标注：read 只读 / write 写入 / delete 破坏性。"""
        self.assertEqual(TOOLS["read_file"].risk, "read")
        self.assertEqual(TOOLS["list_files"].risk, "read")
        self.assertEqual(TOOLS["write_file"].risk, "write")
        self.assertEqual(TOOLS["create_file"].risk, "write")
        self.assertEqual(TOOLS["edit_file"].risk, "write")
        self.assertEqual(TOOLS["create_dir"].risk, "write")
        self.assertEqual(TOOLS["write_dir"].risk, "write")
        self.assertEqual(TOOLS["rename_dir"].risk, "write")
        self.assertEqual(TOOLS["delete_file"].risk, "delete")
        self.assertEqual(TOOLS["delete_dir"].risk, "delete")

    def test_register_decorator(self):
        @register_tool("_test_tmp_tool", description="临时")
        def _tmp(args, cwd):
            return "ok"

        self.assertIn("_test_tmp_tool", TOOLS)
        self.assertEqual(TOOLS["_test_tmp_tool"].description, "临时")
        del TOOLS["_test_tmp_tool"]

    def test_register_rejects_unknown_risk(self):
        with self.assertRaises(ValueError):
            register_tool("_bad_risk", risk="explode")

        self.assertNotIn("_bad_risk", TOOLS)


class ExecutorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.executor = ToolExecutor(self._tmp.name)

    def test_unknown_tool(self):
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("no_such_tool", {})
        self.assertIn("read_file", str(ctx.exception))

    def test_missing_required_param(self):
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("read_file", {})
        self.assertIn("缺少参数", str(ctx.exception))
        self.assertIn("path", str(ctx.exception))

    def test_param_type_validation(self):
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("read_file", {"path": 123})
        self.assertIn("必须是字符串", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("delete_dir", {"path": "d", "recursive": "yes"})
        self.assertIn("必须是布尔值", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("read_file", {"path": "a.py", "start_line": "x"})
        self.assertIn("必须是整数", str(ctx.exception))
        # bool 是 int 子类，需显式拒绝。
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("read_file", {"path": "a.py", "start_line": True})
        self.assertIn("必须是整数", str(ctx.exception))

    def test_error_as_observation_text(self):
        with self.assertRaises(NotFoundError):
            self.executor.execute("read_file", {"path": "missing.txt"})

    def test_unknown_exception_wrapped_as_execution_error(self):
        """工具抛非 ToolError 的未知异常 → 包装为 ExecutionError。"""
        from myagent.tools.registry import register_tool

        @register_tool("_boom_tool")
        def _boom(args, cwd):
            raise RuntimeError("内部爆炸")

        try:
            with self.assertRaises(ExecutionError) as ctx:
                self.executor.execute("_boom_tool", {})
            self.assertIn("内部爆炸", str(ctx.exception))
        finally:
            del TOOLS["_boom_tool"]

    def test_timeout_raises_tool_timeout(self):
        """超时机制：短 timeout + 卡住工具 → ToolTimeoutError。"""
        from myagent.tools.registry import register_tool

        @register_tool("_slow_tool")
        def _slow(args, cwd):
            import time

            time.sleep(5)
            return "太慢了"

        slow_executor = ToolExecutor(self._tmp.name, timeout=0.1)
        try:
            with self.assertRaises(ToolTimeoutError) as ctx:
                slow_executor.execute("_slow_tool", {})
            self.assertIn("超过", str(ctx.exception))
        finally:
            del TOOLS["_slow_tool"]

    def test_to_openai_tools(self):
        """OpenAI tools 数组应覆盖全部注册工具，含正确的 name/description/parameters。"""
        tools = to_openai_tools()
        names = [t["function"]["name"] for t in tools]
        for name in EXPECTED_TOOLS:
            self.assertIn(name, names)
        self.assertNotIn("list_dir", names)
        for tool in tools:
            self.assertEqual(tool["type"], "function")
            function = tool["function"]
            self.assertTrue(function["description"])
            # parameters 是合法 JSON Schema 对象（顶层 type: object）。
            self.assertIsInstance(function["parameters"], dict)
            self.assertEqual(function["parameters"]["type"], "object")
            self.assertIsInstance(function["parameters"]["properties"], dict)

    def test_to_openai_tools_required_and_strip(self):
        """参数级 required 汇总到顶层，required/default 键从属性中剥离。"""
        tools = to_openai_tools()
        read_file = next(
            t for t in tools if t["function"]["name"] == "read_file"
        )
        params = read_file["function"]["parameters"]
        self.assertEqual(params["required"], ["path"])
        # 属性内不允许残留 required / default 键。
        for prop in params["properties"].values():
            self.assertNotIn("required", prop)
            self.assertNotIn("default", prop)

    def test_execute_returns_string_on_success(self):
        """合法参数下工具正常返回字符串，不抛异常。"""
        self.executor.execute("write_file", {"path": "x.txt", "content": "c"})
        out = self.executor.execute("read_file", {"path": "x.txt", "start_line": 1})
        self.assertIsInstance(out, str)
        self.assertIn("x.txt", out)

    def test_failures_raise_tool_error_not_bare(self):
        """失败抛 ToolError 子类，而非裸异常（如 KeyError/ValueError）。"""
        with self.assertRaises(ToolError):
            self.executor.execute("read_file", {"path": "missing.txt"})
        with self.assertRaises(ToolError):
            self.executor.execute("read_file", {"path": "../escape"})
        with self.assertRaises(ToolError):
            self.executor.execute("read_file", {"path": "x.txt", "start_line": "bad"})


class FileToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.executor = ToolExecutor(self.root)

    def test_read_create_write_edit_delete_flow(self):
        # create
        out = self.executor.execute("create_file", {"path": "a.txt", "content": "hello world"})
        self.assertIn("已创建", out)
        self.assertTrue((self.root / "a.txt").is_file())
        # create 已存在 → ValidationError
        with self.assertRaises(ValidationError):
            self.executor.execute("create_file", {"path": "a.txt"})
        # write 覆盖
        self.executor.execute("write_file", {"path": "a.txt", "content": "hello myagent\n第二行"})
        self.assertEqual((self.root / "a.txt").read_text(), "hello myagent\n第二行")
        # read（含行区间）
        out = self.executor.execute("read_file", {"path": "a.txt", "start_line": 1, "end_line": 1})
        self.assertIn("hello myagent", out)
        self.assertIn("第 1-1 行", out)
        # edit 唯一匹配
        out = self.executor.execute("edit_file", {"path": "a.txt", "old_text": "myagent", "new_text": "agent"})
        self.assertIn("已修改", out)
        self.assertIn("hello agent", (self.root / "a.txt").read_text())
        # delete
        out = self.executor.execute("delete_file", {"path": "a.txt"})
        self.assertIn("已删除", out)
        self.assertFalse((self.root / "a.txt").exists())
        # delete 不存在 → NotFoundError
        with self.assertRaises(NotFoundError):
            self.executor.execute("delete_file", {"path": "a.txt"})

    def test_edit_requires_unique_match(self):
        self.executor.execute("write_file", {"path": "b.txt", "content": "x x x"})
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("edit_file", {"path": "b.txt", "old_text": "x", "new_text": "y"})
        self.assertIn("出现 3 次", str(ctx.exception))

    def test_write_file_creates_parents(self):
        self.executor.execute("write_file", {"path": "deep/nested/f.txt", "content": "hi"})
        self.assertTrue((self.root / "deep" / "nested" / "f.txt").is_file())


class DirToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.executor = ToolExecutor(self.root)

    def test_list_create_write_rename_delete_flow(self):
        # create_dir 单层（父目录是根，存在）
        self.executor.execute("create_dir", {"path": "d1"})
        self.assertTrue((self.root / "d1").is_dir())
        # create_dir 父目录缺失 → NotFoundError（提示用 write_dir）
        with self.assertRaises(NotFoundError) as ctx:
            self.executor.execute("create_dir", {"path": "x/y"})
        self.assertIn("write_dir", str(ctx.exception))
        # write_dir 递归
        self.executor.execute("write_dir", {"path": "a/b/c"})
        self.assertTrue((self.root / "a" / "b" / "c").is_dir())
        # 写入文件后 list_files 应显示
        self.executor.execute("write_file", {"path": "d1/f.txt", "content": "x"})
        out = self.executor.execute("list_files", {"path": "d1"})
        self.assertIn("f.txt", out)
        # rename_dir
        out = self.executor.execute("rename_dir", {"src": "d1", "dst": "d2"})
        self.assertIn("已重命名", out)
        self.assertTrue((self.root / "d2").is_dir())
        self.assertFalse((self.root / "d1").exists())
        # delete_dir 非空默认失败 → ValidationError
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute("delete_dir", {"path": "d2"})
        self.assertIn("非空", str(ctx.exception))
        # delete_dir 递归
        out = self.executor.execute("delete_dir", {"path": "d2", "recursive": True})
        self.assertIn("递归", out)
        self.assertFalse((self.root / "d2").exists())
        # delete_dir 空目录
        self.executor.execute("create_dir", {"path": "empty"})
        out = self.executor.execute("delete_dir", {"path": "empty"})
        self.assertIn("已删除", out)

    def test_list_files_empty(self):
        out = self.executor.execute("list_files", {"path": "."})
        self.assertIn("空目录", out)


class PathSafetyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: self.outside and self.outside.exists() and None)
        self.executor = ToolExecutor(self.root)

    def test_parent_traversal_rejected(self):
        with self.assertRaises(ToolPermissionError) as ctx:
            self.executor.execute("read_file", {"path": "../outside.txt"})
        self.assertIn("路径越界", str(ctx.exception))

    def test_absolute_path_outside_rejected(self):
        outside_file = self.outside / "secret.txt"
        outside_file.write_text("secret")
        with self.assertRaises(ToolPermissionError) as ctx:
            self.executor.execute("read_file", {"path": str(outside_file)})
        self.assertIn("路径越界", str(ctx.exception))

    def test_relative_path_inside_allowed(self):
        (self.root / "in.txt").write_text("ok")
        out = self.executor.execute("read_file", {"path": "in.txt"})
        self.assertIn("ok", out)

    def test_delete_rejects_escape(self):
        with self.assertRaises(ToolPermissionError) as ctx:
            self.executor.execute("delete_file", {"path": "../victim.txt"})
        self.assertIn("路径越界", str(ctx.exception))

    def test_symlink_escape_rejected(self):
        """symlink 指向 cwd 之外：resolve 展开后应被拒。"""
        outside_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: outside_dir.exists() and __import__("shutil").rmtree(outside_dir))
        (outside_dir / "secret.txt").write_text("secret")
        link = self.root / "link"
        link.symlink_to(outside_dir, target_is_directory=True)

        with self.assertRaises(ToolPermissionError) as ctx:
            self.executor.execute("read_file", {"path": "link/secret.txt"})
        self.assertIn("路径越界", str(ctx.exception))

    def test_relative_path_resolution_inside(self):
        """sub/../in.txt 这类合法回落路径应放行（解析后仍在 cwd 内）。"""
        (self.root / "sub").mkdir()
        (self.root / "in.txt").write_text("ok")
        out = self.executor.execute("read_file", {"path": "sub/../in.txt"})
        self.assertIn("ok", out)

    def test_invalid_param_value_is_param_error(self):
        (self.root / "a.txt").write_text("line1\nline2")
        with self.assertRaises(ValidationError) as ctx:
            self.executor.execute(
                "read_file", {"path": "a.txt", "start_line": "abc"}
            )
        self.assertIn("必须是整数", str(ctx.exception))

    def test_delete_dir_rejects_cwd(self):
        with self.assertRaises(ToolPermissionError):
            self.executor.execute("delete_dir", {"path": "."})
        with self.assertRaises(ToolPermissionError):
            self.executor.execute("delete_dir", {"path": ".", "recursive": True})

    def test_rename_dir_rejects_cwd(self):
        with self.assertRaises(ToolPermissionError):
            self.executor.execute("rename_dir", {"src": ".", "dst": "moved"})


if __name__ == "__main__":
    unittest.main()
