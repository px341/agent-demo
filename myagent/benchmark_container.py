"""Per-task Docker boundary. No host directories or secrets are mounted."""
from __future__ import annotations

import base64
import io
import json
import shlex
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path, PurePosixPath

from .errors import CommandError, ExecutionError, ToolTimeoutError, ValidationError
from .tools.executor import _validate_params
from .tools.registry import TOOLS, to_openai_tools


class RecoverableTimeout(ToolTimeoutError):
    recoverable = True


def load_environments(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != 1 or not value.get("environments"):
        raise ValueError("Missing audited generation environments; run prepare-environments")
    return value["environments"]


def benchmark_schema():
    allowed = {"read_file", "create_file", "write_file", "edit_file", "delete_file", "list_files",
               "create_dir", "write_dir", "rename_dir", "delete_dir", "run_shell", "git_status", "git_diff", "git_apply_patch"}
    result = [tool for tool in to_openai_tools() if tool["function"]["name"] in allowed]
    for tool in result:
        if tool["function"]["name"] == "run_shell":
            tool["function"]["description"] = "Run bash inside the offline task container; pipes and redirects are supported. Search only /testbed."
            tool["function"]["parameters"]["properties"]["command"]["description"] = "Bash command inside /testbed"
    return result


class DockerSandbox:
    def __init__(self, checkout: Path, spec: dict, *, timeout=60, instance_seconds=1800,
                 max_timeout_recoveries=2, client=None):
        if client is None:
            import docker
            client = docker.from_env(timeout=30)
        self.client = client
        self.container = None
        self.timeout = timeout
        self.max_timeout_recoveries = max_timeout_recoveries
        self.timeouts = 0
        self.deadline = time.monotonic() + instance_seconds
        self.spec = spec
        self.runner = Path(__file__).with_name("sandbox_runner.py").read_text(encoding="utf-8")
        self.events = []
        try:
            image = client.images.get(spec["image_id"])
            if image.id != spec["image_id"] or spec.get("audit") != "runtime-only-v1":
                raise ValueError("Image digest/audit mismatch")
            config = image.attrs["Config"]
            if config.get("Volumes") or config.get("Entrypoint") or config.get("Env"):
                raise ValueError("Generation image has unaudited config")
            self.container = client.containers.run(
                image.id, ["/bin/sleep", "infinity"], detach=True, entrypoint=[],
                network_mode="none", volumes={}, cap_drop=["ALL"],
                labels={"myagent.purpose": "benchmark-generation"},
                security_opt=["no-new-privileges:true"], pids_limit=256,
                mem_limit="4g", nano_cpus=2_000_000_000, working_dir="/",
                environment={"PATH": str(PurePosixPath(spec["python"]).parent) + ":/usr/local/bin:/usr/bin:/bin",
                             "HOME": "/tmp", "PYTHONPATH": "/testbed:/testbed/src",
                             "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
                             "PYTHONDONTWRITEBYTECODE": "1"})
            self.container.reload()
            host = self.container.attrs["HostConfig"]
            if self.container.attrs.get("Mounts") or host["NetworkMode"] != "none" or host.get("Privileged"):
                raise ValueError("Container boundary verification failed")
            self.boundary = {"network_mode": host["NetworkMode"], "mounts": self.container.attrs.get("Mounts", []),
                             "privileged": host.get("Privileged", False), "cap_drop": host.get("CapDrop"),
                             "security_opt": host.get("SecurityOpt"), "pids_limit": host.get("PidsLimit")}
            # Task code must be supplied only by the base snapshot, never a wheel
            # or source copy baked into the dependency environment.
            module = spec["source_module"]
            audit_script = ("import importlib.util, pathlib; "
                            f"assert importlib.util.find_spec({module!r}) is None, 'Task code baked into image'; "
                            "assert not list(pathlib.Path('/testbed').iterdir()), 'Source/scoring residue in image'; "
                            "print('Runtime image audit passed')")
            self.image_audit = self.execute("run_shell", {"command": "python -c " + shlex.quote(audit_script)})
            # Only tracked base files, never .git/history, run outputs or data.
            import subprocess
            names = subprocess.run(["git", "ls-files", "-z"], cwd=checkout,
                                   check=True, capture_output=True, timeout=60).stdout.split(b"\0")
            buffer = io.BytesIO()
            def container_owner(member):
                # Host checkout UID must not prevent ordinary edits after all
                # capabilities (including DAC_OVERRIDE and CHOWN) are dropped.
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                return member
            with tarfile.open(fileobj=buffer, mode="w", dereference=False) as dest:
                root = tarfile.TarInfo("testbed")
                root.type = tarfile.DIRTYPE
                root.mode = 0o755
                dest.addfile(root)
                # git archive can omit tracked tests via export-ignore attributes.
                # Copy every tracked base file, preserving symlinks without following them.
                for raw in names:
                    if raw:
                        name = raw.decode("utf-8", errors="surrogateescape")
                        dest.add(checkout / name, arcname="testbed/" + name, recursive=False,
                                 filter=container_owner)
            if not self.container.put_archive("/", buffer.getvalue()):
                raise RuntimeError("Cannot upload base snapshot")
            self.execute("run_shell", {"command": "git init -q && git config user.name Benchmark && git config user.email benchmark@localhost && git add -A -f -- . && git commit -qm base"})
            self.base_sha = self.execute("run_shell", {"command": "git rev-parse HEAD"}).strip()
            self.preflight = self.execute("run_shell", {"command": spec["check_command"], "timeout": timeout})
        except Exception:
            self.close()
            raise

    def _invoke(self, name, args, timeout):
        payload = base64.b64encode(json.dumps({"name": name, "args": args, "timeout": timeout}).encode()).decode()
        script = self.runner + "\nimport base64\nprint(json.dumps(main(json.loads(base64.b64decode('" + payload + "')))))"
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.container.exec_run, [self.spec["python"], "-c", script], workdir="/testbed")
        try:
            result = future.result(timeout=timeout + 5)
        except TimeoutError:
            # Docker exec has no cancellation API. Restart kills the entire task PID namespace.
            self.container.restart(timeout=0)
            raise RecoverableTimeout("Docker exec watchdog expired; all container processes terminated")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if result.exit_code:
            raise ExecutionError("Container dispatcher failed: " + result.output.decode(errors="replace")[:2000])
        return json.loads(result.output)

    def execute(self, name, args):
        if name != "capture_patch":
            spec = TOOLS.get(name)
            if spec is None:
                raise ValidationError(f"Unknown benchmark tool: {name}")
            _validate_params(spec, args)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ToolTimeoutError("Per-instance time limit reached")
        timeout = max(0.01, min(float(args.get("timeout", self.timeout)), self.timeout, remaining))
        started = time.monotonic()
        try:
            result = self._invoke(name, args, timeout)
            if result.get("error") == "TimeoutError":
                self.container.restart(timeout=0)
                raise RecoverableTimeout(result["message"])
            if result.get("error"):
                raise ValidationError(result["message"])
            if result.get("exit_code"):
                raise CommandError(f"Exit {result['exit_code']}: {result['output'][:8000]}")
            return result["output"] if name == "capture_patch" else result["output"][:16000]
        except RecoverableTimeout as exc:
            self.timeouts += 1
            if self.timeouts > self.max_timeout_recoveries:
                raise ToolTimeoutError("Timeout recovery limit exhausted") from exc
            raise
        finally:
            self.events.append({"tool": name, "elapsed_seconds": round(time.monotonic() - started, 3),
                                "timeout_count": self.timeouts})

    def capture_patch(self):
        # Finalization still works after the task time limit, with a bounded command.
        result = self._invoke("capture_patch", {"base": self.base_sha}, 60)
        if result.get("error") or result.get("exit_code"):
            raise RuntimeError("Patch capture failed: " + str(result))
        return result["output"]

    def close(self):
        if self.container is not None:
            self.container.remove(force=True)
            self.container = None
        self.client.close()


def prepare_runtime(image_name: str, client, destination: str, *, source_module: str) -> str:
    """Flatten an evaluation image into a new allowlisted runtime-only image.

    This runs before inference, never with agent tools or an API key in a container.
    Source trees, Git objects, setup scripts, homes, caches and scoring outputs are dropped.
    """
    import tempfile
    image = client.images.get(image_name)
    container = client.containers.create(image.id, entrypoint=[], command=["/bin/true"], network_mode="none")
    try:
        with tempfile.TemporaryDirectory() as temp:
            raw, clean = Path(temp) / "raw.tar", Path(temp) / "clean.tar"
            with raw.open("wb") as file:
                for chunk in container.export():
                    file.write(chunk)
            with tarfile.open(raw) as source, tarfile.open(clean, "w") as target:
                for member in source:
                    parts = PurePosixPath(member.name).parts
                    if not parts or parts[0] not in {"bin", "sbin", "lib", "lib32", "lib64", "libx32", "usr", "etc", "opt"}:
                        continue
                    if ".." in parts:
                        raise ValueError("Unsafe path in source image export")
                    if parts[0] == "opt" and not (member.name == "opt" or parts[:4] == ("opt", "miniconda3", "envs", "testbed")):
                        continue
                    if any(p in {".git", ".cache", "__pycache__", "pkgs"} for p in parts):
                        continue
                    if parts[-1] in {"eval.sh", "test_output.txt", "predictions.jsonl", "generation-summary.json", "dataset.jsonl"}:
                        continue
                    if "site-packages" in parts:
                        at = parts.index("site-packages") + 1
                        package = parts[at].lower() if at < len(parts) else ""
                        if package == source_module.lower() or package.startswith((source_module.lower() + ".", source_module.lower() + "-")):
                            continue
                    if member.name.startswith(("etc/ssh/", "etc/shadow", "etc/gshadow")):
                        continue
                    if member.isfile() and member.name.endswith((".diff", ".patch")):
                        continue
                    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                        continue
                    target.addfile(member, source.extractfile(member) if member.isfile() else None)
                for name in ("tmp", "testbed", "proc", "sys", "dev", "run", "var", "var/tmp"):
                    member = tarfile.TarInfo(name)
                    member.type, member.mode = tarfile.DIRTYPE, 0o1777 if name.endswith("tmp") else 0o755
                    target.addfile(member)
            response = client.api.import_image_from_file(str(clean), repository=destination, tag="latest")
            if isinstance(response, bytes):
                response = response.decode()
            for line in str(response).splitlines():
                if json.loads(line).get("error"):
                    raise RuntimeError(line)
            return client.images.get(destination + ":latest").id
    finally:
        container.remove(force=True)
