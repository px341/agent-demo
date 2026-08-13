"""shell / git 工具测试：命令执行安全、git diff hunk 裁剪、patch 应用。

全部在临时目录 + 临时 git 仓库内执行，不触碰真实工作区。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myagent.context.tokens import clip_diff_hunks
from myagent.errors import (
    CommandError,
    ToolPermissionError,
    ToolTimeoutError,
)
from myagent.tools import TOOLS, ToolExecutor


class ShellToolTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.executor = ToolExecutor(self._tmp.name)

    def test_success_output(self):
        out = self.executor.execute("run_shell", {"command": "echo hello"})
        self.assertIn("hello", out)

    def test_nonzero_exit_raises_command_error(self):
        with self.assertRaises(CommandError) as ctx:
            self.executor.execute("run_shell", {"command": "python3 -c 'import sys; sys.exit(3)'"})
        self.assertIn("退出码 3", str(ctx.exception))
        self.assertTrue(ctx.exception.recoverable)
        self.assertEqual(ctx.exception.error_type, "CommandError")

    def test_blacklist_rm_rf(self):
        with self.assertRaises(ToolPermissionError) as ctx:
            self.executor.execute("run_shell", {"command": "rm -rf /tmp/x"})
        self.assertIn("禁止", str(ctx.exception))

    def test_blacklist_dangerous_command(self):
        with self.assertRaises(ToolPermissionError):
            self.executor.execute("run_shell", {"command": "sudo ls"})

    def test_blacklist_pipeline(self):
        with self.assertRaises(ToolPermissionError):
            self.executor.execute("run_shell", {"command": "ls | grep py"})

    def test_timeout(self):
        with self.assertRaises(ToolTimeoutError):
            self.executor.execute("run_shell", {"command": "sleep 5", "timeout": 1})

    def test_cwd_restricted_to_workdir(self):
        # 子进程 cwd 被限制在工具目录内：在工具目录建文件，pwd 应指向它。
        (Path(self._tmp.name) / "marker.txt").write_text("x")
        out = self.executor.execute("run_shell", {"command": "pwd"})
        self.assertIn(self._tmp.name, out)

    def test_blacklist_git_reset_hard(self):
        with self.assertRaises(ToolPermissionError):
            self.executor.execute("run_shell", {"command": "git reset --hard HEAD"})

    def test_shlex_argument_list_no_shell(self):
        # 参数列表方式执行：含空格/引号的参数不会被 shell 二次解析。
        out = self.executor.execute("run_shell", {"command": 'echo "a b" c'})
        self.assertIn("a b c", out)


class GitToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.repo, check=True)
        self.executor = ToolExecutor(self.repo)

    def _write(self, name: str, content: str) -> None:
        (self.repo / name).write_text(content, encoding="utf-8")

    def test_not_a_git_repo(self):
        plain = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: plain.exists() and None)
        ex = ToolExecutor(plain)
        with self.assertRaises(CommandError):
            ex.execute("git_status", {})

    def test_git_status_clean_and_dirty(self):
        self._write("a.txt", "one")
        subprocess.run(["git", "add", "a.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.repo, check=True)
        out = self.executor.execute("git_status", {})
        self.assertIn("工作区干净", out)
        # 改文件 + 新建未跟踪 → 变更出现。
        self._write("a.txt", "two")
        self._write("b.txt", "new")
        out2 = self.executor.execute("git_status", {})
        self.assertIn("a.txt", out2)
        self.assertIn("b.txt", out2)

    def test_git_diff_stat_and_hunks(self):
        self._write("a.txt", "line1\nline2\nline3\n")
        subprocess.run(["git", "add", "a.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.repo, check=True)
        self._write("a.txt", "line1\nline2 CHANGED\nline3\nNEW LINE\n")
        out = self.executor.execute("git_diff", {})
        self.assertIn("diff --git", out)
        self.assertIn("@@", out)  # hunk 头
        self.assertIn("line2 CHANGED", out)

    def test_git_diff_path_filter(self):
        self._write("a.txt", "a1\na2\n")
        self._write("b.txt", "b1\nb2\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.repo, check=True)
        self._write("a.txt", "a1\na2 CHANGED\n")
        self._write("b.txt", "b1\nb2 CHANGED\n")
        out = self.executor.execute("git_diff", {"path": "a.txt"})
        self.assertIn("diff --git a/a.txt", out)
        self.assertNotIn("b.txt", out.split("diff --git")[-1])

    def test_git_diff_cached(self):
        self._write("a.txt", "one\n")
        subprocess.run(["git", "add", "a.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.repo, check=True)
        self._write("a.txt", "two\n")
        subprocess.run(["git", "add", "a.txt"], cwd=self.repo, check=True)
        out = self.executor.execute("git_diff", {"cached": True})
        self.assertIn("+two", out)

    def test_git_apply_patch(self):
        self._write("a.txt", "line1\nline2\n")
        subprocess.run(["git", "add", "a.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.repo, check=True)
        # 生成一个 patch（对 a.txt 的修改）。
        self._write("a.txt", "line1\nline2 MODIFIED\n")
        raw = subprocess.run(
            ["git", "diff"], cwd=self.repo, capture_output=True, text=True, check=True
        ).stdout
        subprocess.run(["git", "checkout", "--", "a.txt"], cwd=self.repo, check=True)
        # 应用 patch → 内容恢复修改。
        out = self.executor.execute("git_apply_patch", {"patch": raw})
        self.assertIn("成功", out)
        self.assertIn("line2 MODIFIED", (self.repo / "a.txt").read_text())

    def test_git_apply_invalid_patch(self):
        with self.assertRaises(CommandError):
            self.executor.execute("git_apply_patch", {"patch": "这不是有效 patch"})

    def test_git_diff_hunk_clipped(self):
        """大 diff：hunk 对齐裁剪，保留完整 hunk + 标记。"""
        self._write("big.txt", "".join(f"base line {i}\n" for i in range(2000)))
        subprocess.run(["git", "add", "big.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.repo, check=True)
        # 修改多处 → 多 hunk diff。
        lines = (self.repo / "big.txt").read_text().split("\n")
        for i in (0, 500, 1000, 1500, 1999):
            lines[i] = f"MODIFIED {i}"
        (self.repo / "big.txt").write_text("\n".join(lines), encoding="utf-8")
        out = self.executor.execute("git_diff", {"max_tokens": 300})
        # 保留 hunk 头，且含裁剪标记（大 diff 必然超预算）。
        self.assertIn("diff --git", out)
        self.assertIn("@@", out)
        self.assertIn("已裁剪", out)


class ClipDiffHunksTest(unittest.TestCase):
    """clip_diff_hunks 单测：hunk 完整保留、预算内原样。"""

    def test_within_budget_unchanged(self):
        diff = (
            "diff --git a/a.txt b/a.txt\n"
            "--- a/a.txt\n+++ b/a.txt\n"
            "@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        self.assertEqual(clip_diff_hunks(diff, 10000), diff)

    def test_hunks_preserved_head_tail(self):
        diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n"
        hunks = []
        for n in range(20):
            hunks.append(f"@@ -{n},1 +{n},1 @@\n+ line{n}\n")
        full = diff + "".join(hunks)
        clipped = clip_diff_hunks(full, 200)
        # 保留完整的 hunk（每个 hunk 的行都完整，不切中间）。
        self.assertIn("已裁剪", clipped)
        self.assertIn("line0", clipped)
        # 头部 hunk 优先保留，尾部在预算允许时补（此预算下尾 hunk 可能被裁）。
        self.assertIn("…[已裁剪", clipped)
        # 所有保留的 hunk 都完整：每个变更行（+ 前缀，排除文件头 +++）前都有 @@ 头。
        body = clipped.split("diff --git", 1)[1]
        plus_lines = [l for l in body.split("\n") if l.startswith("+") and not l.startswith("+++")]
        hunk_headers = [l for l in body.split("\n") if l.startswith("@@")]
        self.assertEqual(len(plus_lines), len(hunk_headers))

    def test_zero_budget(self):
        self.assertEqual(clip_diff_hunks("diff", 0), "")

    def test_no_hunks_falls_back_head_tail(self):
        diff = "diff --git a/x b/x\n" + "a" * 3000
        clipped = clip_diff_hunks(diff, 100)
        self.assertIn("已裁剪", clipped)


if __name__ == "__main__":
    unittest.main()
