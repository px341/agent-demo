"""Timeout and runtime isolation contracts without paid API requests."""
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from myagent import sandbox_runner
from myagent.benchmark_container import DockerSandbox, RecoverableTimeout, benchmark_schema
from myagent.benchmark_container import prepare_runtime
from myagent.errors import ToolTimeoutError


def test_timeout_restarts_namespace_and_has_finite_recoveries():
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.deadline = time.monotonic() + 60
    sandbox.timeout = 1
    sandbox.max_timeout_recoveries = 1
    sandbox.timeouts = 0
    sandbox.events = []
    sandbox.container = Mock()
    sandbox._invoke = Mock(return_value={"error": "TimeoutError", "message": "killed"})
    with pytest.raises(RecoverableTimeout):
        sandbox.execute("run_shell", {"command": "sleep 60"})
    with pytest.raises(ToolTimeoutError) as caught:
        sandbox.execute("run_shell", {"command": "sleep 60"})
    assert not caught.value.recoverable
    assert sandbox.container.restart.call_count == 2


def test_shell_kills_background_process_group():
    with tempfile.TemporaryDirectory() as temporary, patch.object(sandbox_runner, "ROOT", Path(temporary)):
        result = sandbox_runner.command(["/bin/bash", "-c", "sleep 30 & wait"], 0.1)
        assert result["error"] == "TimeoutError"
        result = sandbox_runner.command(["/bin/bash", "-c", "printf 'ok' | tr a-z A-Z > output && cat output"], 2)
        assert result["output"] == "OK"
        assert result["exit_code"] == 0


def test_benchmark_schema_supports_shell_pipelines():
    shell = next(t for t in benchmark_schema() if t["function"]["name"] == "run_shell")
    assert "pipes and redirects" in shell["function"]["description"]


def test_uploaded_base_is_editable_and_keeps_only_tracked_files(tmp_path):
    import io
    import tarfile
    (tmp_path / "source.py").write_text("value = 1\n")
    (tmp_path / "untracked-answer.txt").write_text("private")
    (tmp_path / "link.py").symlink_to("source.py")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "source.py", "link.py"], cwd=tmp_path, check=True)
    client = Mock()
    client.images.get.return_value.id = "sha256:clean"
    client.images.get.return_value.attrs = {"Config": {}}
    container = client.containers.run.return_value
    container.attrs = {"Mounts": [], "HostConfig": {"NetworkMode": "none"}}
    container.put_archive.return_value = True
    spec = {"image_id": "sha256:clean", "audit": "runtime-only-v1", "source_module": "task",
            "python": "/usr/bin/python", "check_command": "python -V"}
    with patch.object(DockerSandbox, "execute", return_value="ok"):
        sandbox = DockerSandbox(tmp_path, spec, client=client)
    with tarfile.open(fileobj=io.BytesIO(container.put_archive.call_args.args[1])) as archive:
        assert set(archive.getnames()) == {"testbed", "testbed/source.py", "testbed/link.py"}
        source = archive.getmember("testbed/source.py")
        assert (source.uid, source.gid) == (0, 0)
        assert source.mode & 0o200
        assert archive.getmember("testbed/link.py").issym()
    sandbox.close()


def test_runtime_export_drops_answers_history_caches_and_installed_task_code():
    import io
    import tarfile
    buffer = io.BytesIO()
    paths = ["usr/lib/runtime.so", "testbed/source.py", "testbed/.git/objects/answer",
             "root/.cache/huggingface/answer", "tmp/test_patch", "eval.sh",
             "usr/share/.git/objects/answer", "usr/lib/answer.patch",
             "opt/miniconda3/envs/testbed/lib/site-packages/django/fix.py",
             "opt/miniconda3/envs/testbed/lib/site-packages/dependency/runtime.py",
             "opt/miniconda3/envs/testbed-backup/answer"]
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in paths:
            member = tarfile.TarInfo(name)
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
    container = Mock()
    container.export.return_value = [buffer.getvalue()]
    client = Mock()
    client.containers.create.return_value = container
    client.images.get.return_value.id = "sha256:clean"
    retained = []
    def import_file(filename, **kwargs):
        with tarfile.open(filename) as archive:
            retained.extend(archive.getnames())
        return b'{"status":"imported"}'
    client.api.import_image_from_file.side_effect = import_file
    assert prepare_runtime("source", client, "clean", source_module="django") == "sha256:clean"
    assert "usr/lib/runtime.so" in retained
    assert "opt/miniconda3/envs/testbed/lib/site-packages/dependency/runtime.py" in retained
    assert not set(paths[1:9]).intersection(retained)
    assert paths[-1] not in retained
    container.remove.assert_called_once_with(force=True)
