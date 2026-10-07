"""Standard-library-only tool dispatcher, executed exclusively inside Docker."""
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/testbed")


def command(argv, timeout, *, input_text=None):
    proc = subprocess.Popen(argv, cwd=str(ROOT), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return {"error": "TimeoutError", "message": f"Command exceeded {timeout}s; process group killed"}
    finally:
        # Also remove background children left behind by an otherwise successful shell.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"output": stdout + ("\n(stderr)\n" + stderr if stderr else ""),
            "exit_code": proc.returncode}


def path(value):
    candidate = (ROOT / value).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError("Path must stay inside /testbed")
    return candidate


def patch(timeout, base):
    # A temporary index includes new files without changing the agent's index.
    # Ignored files (caches, environments, compiled output) remain excluded.
    with tempfile.TemporaryDirectory() as directory:
        old = os.environ.get("GIT_INDEX_FILE")
        os.environ["GIT_INDEX_FILE"] = directory + "/index"
        try:
            for args in (["read-tree", base], ["add", "-A", "--", "."]):
                result = command(["git", *args], timeout)
                if result.get("error") or result["exit_code"]:
                    return result
            return command(["git", "diff", "--cached", "--binary", "--no-ext-diff", base], timeout)
        finally:
            if old is None:
                os.environ.pop("GIT_INDEX_FILE", None)
            else:
                os.environ["GIT_INDEX_FILE"] = old


def dispatch(name, args, timeout):
    if name == "run_shell":
        return command(["/bin/bash", "--noprofile", "--norc", "-c", args["command"]], timeout)
    if name == "capture_patch":
        return patch(timeout, args["base"])
    if name in {"git_status", "git_diff", "git_apply_patch"}:
        if name == "git_status":
            argv = ["git", "status", "--short"]
        elif name == "git_apply_patch":
            argv = ["git", "apply", "--whitespace=nowarn"] + (["--cached"] if args.get("cached") else [])
        else:
            argv = ["git", "diff"] + (["--cached"] if args.get("cached") else [])
            if args.get("path"):
                argv += ["--", str(path(args["path"]).relative_to(ROOT))]
        return command(argv, timeout, input_text=args.get("patch"))
    target = path(args.get("path", "."))
    if name == "read_file":
        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, args.get("start_line", 1))
        end = args.get("end_line", len(lines))
        output = "\n".join(f"{i + start}: {line}" for i, line in enumerate(lines[start - 1:end]))
    elif name == "list_files":
        output = "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir()))
    elif name in {"write_file", "create_file"}:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x" if name == "create_file" else "w", encoding="utf-8") as file:
            file.write(args.get("content", ""))
        output = "File written"
    elif name == "edit_file":
        content = target.read_text(encoding="utf-8")
        old = args["old_text"]
        if not old or content.count(old) != 1:
            raise ValueError("old_text must match exactly once")
        target.write_text(content.replace(old, args.get("new_text", "")), encoding="utf-8")
        output = "File edited"
    elif name == "delete_file":
        target.unlink()
        output = "File deleted"
    elif name in {"create_dir", "write_dir"}:
        target.mkdir(parents=name == "write_dir", exist_ok=name == "write_dir")
        output = "Directory created"
    elif name == "rename_dir":
        source, destination = path(args["src"]), path(args["dst"])
        if source == ROOT or destination.exists():
            raise ValueError("Invalid directory rename")
        source.rename(destination)
        output = "Directory renamed"
    elif name == "delete_dir":
        import shutil
        if target == ROOT:
            raise ValueError("Cannot delete project root")
        if args.get("recursive"):
            shutil.rmtree(target)
        else:
            target.rmdir()
        output = "Directory deleted"
    else:
        raise ValueError("Unknown tool")
    return {"output": output, "exit_code": 0}


def main(payload):
    try:
        return dispatch(payload["name"], payload["args"], payload["timeout"])
    except (ValueError, KeyError, TypeError) as exc:
        return {"error": "ValidationError", "message": str(exc)}
    except OSError as exc:
        return {"error": "CommandError", "message": str(exc)}
