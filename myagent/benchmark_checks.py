"""No-model isolation acceptance checks against actual host canaries."""
from __future__ import annotations

import json
import secrets
import shlex
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def isolation_check(sandbox) -> dict:
    with tempfile.TemporaryDirectory(prefix="myagent-host-answer-") as temporary:
        directory = Path(temporary)
        canary = directory / "answer-cache.txt"
        marker = secrets.token_hex(32)
        canary.write_text(marker, encoding="utf-8")
        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2)
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(marker.encode())
            def log_message(self, *args):
                pass
        # A real host Git object containing the canary, outside the transferred snapshot.
        subprocess.run(["git", "init", "-q", str(directory)], check=True, capture_output=True)
        blob = subprocess.run(["git", "hash-object", "-w", str(canary)], cwd=directory,
                              check=True, capture_output=True, text=True).stdout.strip()
        server = HTTPServer(("0.0.0.0", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        script = """
import json, os, pathlib, socket, urllib.request
secret = pathlib.Path(HOST_PATH)
results = {}
for label, target in [('absolute', secret), ('parent', pathlib.Path('/testbed/..') / str(secret).lstrip('/'))]:
    try:
        target.read_bytes()
        results[label] = False
    except (FileNotFoundError, PermissionError):
        results[label] = True
link = pathlib.Path('/testbed/.isolation-link')
link.symlink_to(secret)
try:
    open(link, 'rb').read()
    results['symlink_python_open'] = False
except (FileNotFoundError, PermissionError):
    results['symlink_python_open'] = True
finally:
    link.unlink()
for name, url in [('host_network', HOST_URL), ('public_network', 'http://1.1.1.1')]:
    try:
        urllib.request.urlopen(url, timeout=2).read(1)
        results[name] = False
    except Exception:
        results[name] = True
results['docker_socket'] = not pathlib.Path('/var/run/docker.sock').exists()
results['secrets'] = not any('KEY' in key or 'TOKEN' in key or 'SECRET' in key for key in os.environ)
print(json.dumps(results))
""".replace("HOST_PATH", repr(str(canary))).replace("HOST_URL", repr(f"http://host.docker.internal:{server.server_port}/answer-cache.txt"))
        try:
            output = sandbox.execute("run_shell", {"command": "python -c " + shlex.quote(script)})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        checks = json.loads(output)
        # Separate shell/git attacks also have to fail with no marker in their output.
        for label, cmd in {
            "shell_absolute": "cat " + shlex.quote(str(canary)),
            "external_git_dir": f"git --git-dir={shlex.quote(str(directory / '.git'))} cat-file -p {blob}",
        }.items():
            result = sandbox._invoke("run_shell", {"command": cmd}, 5)
            checks[label] = bool(result.get("exit_code")) and marker not in result.get("output", "")
        # Reproduce the confirmed historical entry points without returning any
        # reference contents to the agent. These probes run before inference.
        project = Path(__file__).resolve().parents[1]
        legacy_dataset = project / ".swebench-work/runs/verified-mini-full-isolated-20261007-01/dataset.jsonl"
        cache = Path.home() / ".cache/huggingface"
        cached_arrow = next(cache.rglob("*.arrow"), None) if cache.exists() else None
        for label, target in (("historical_dataset", legacy_dataset), ("historical_hf_cache", cached_arrow)):
            if target is not None and target.is_file():
                probe = "with open(" + repr(str(target)) + ", 'rb') as file: file.read(1)"
                result = sandbox._invoke("run_shell", {"command": "python -c " + shlex.quote(probe)}, 5)
                checks[label] = bool(result.get("exit_code"))
        legacy_git = project / ".swebench-work/repos/sphinx.git"
        if legacy_git.is_dir():
            command = "git --git-dir=" + shlex.quote(str(legacy_git)) + " show 31ec519d3808617e333794493e0aabc5fe5d4f81 --format= --stat"
            result = sandbox._invoke("run_shell", {"command": command}, 5)
            checks["historical_future_git_commit"] = bool(result.get("exit_code"))
        sandbox.execute("write_file", {"path": ".isolation-edit", "content": "ok\n"})
        checks["normal_edit"] = "ok" in sandbox.execute("read_file", {"path": ".isolation-edit"})
        checks["shell_pipeline"] = sandbox.execute("run_shell", {"command": "cat .isolation-edit | tr a-z A-Z > .isolation-out && cat .isolation-out"}).strip() == "OK"
        sandbox.execute("delete_file", {"path": ".isolation-edit"})
        sandbox.execute("delete_file", {"path": ".isolation-out"})
        checks["single_base_commit"] = sandbox.execute("run_shell", {"command": "git rev-list --all --count"}).strip() == "1"
        checks["no_git_remote"] = not sandbox.execute("run_shell", {"command": "git remote"}).strip()
        # New-file writes alone miss ownership errors in transferred base files.
        script = """
import pathlib, subprocess
names = subprocess.check_output(['git', 'ls-files', '-z']).decode().split(chr(0))
path = next(pathlib.Path(name) for name in names if name and pathlib.Path(name).is_file() and not pathlib.Path(name).is_symlink())
original = path.read_bytes()
try:
    path.write_bytes(original + b'\\n')
    assert path.read_bytes() == original + b'\\n'
finally:
    path.write_bytes(original)
print('existing tracked file edited and restored')
"""
        checks["existing_file_edit"] = "edited and restored" in sandbox.execute(
            "run_shell", {"command": "python -c " + shlex.quote(script)})
        sandbox.execute("write_file", {"path": ".isolation-new-source.py", "content": "value = 1\n"})
        checks["new_file_patch"] = "+value = 1" in sandbox.capture_patch()
        sandbox.execute("delete_file", {"path": ".isolation-new-source.py"})
        # Exercise an escaped child session, not just a single sleeping process.
        # This happens before inference and does not consume the agent's allowance.
        from .errors import ToolTimeoutError
        original_timeouts = sandbox.timeouts
        child_script = ("import subprocess,time; "
                        "p=subprocess.Popen(['python','-c','import time; time.sleep(60)'],start_new_session=True); "
                        "open('.isolation-child-pid','w').write(str(p.pid)); time.sleep(60)")
        checks["timeout_feedback"] = False
        try:
            sandbox.execute("run_shell", {"command": "python -c " + shlex.quote(child_script), "timeout": 1})
        except ToolTimeoutError as exc:
            checks["timeout_feedback"] = exc.recoverable
        child_pid = int(sandbox.execute("run_shell", {"command": "cat .isolation-child-pid"}).strip())
        script = f"import os; print(not os.path.exists('/proc/{child_pid}'))"
        checks["timeout_child_cleanup"] = sandbox.execute("run_shell", {"command": "python -c " + shlex.quote(script)}).strip() == "True"
        sandbox.execute("delete_file", {"path": ".isolation-child-pid"})
        sandbox.timeouts = original_timeouts
        # Existing task test runner availability is part of the environment preflight.
        checks["environment_preflight"] = bool(sandbox.preflight.strip())
        if not all(checks.values()):
            raise RuntimeError("Isolation acceptance failed: " + json.dumps(checks))
        return checks
