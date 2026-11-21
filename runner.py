"""runner.py - out-of-process agent runner with JSON-Lines stdio isolation.

Runs an untrusted agent in a separate OS process while keeping ``AuditEnv``,
the ``Evaluator``, ``seed`` and ``truth`` strictly inside the runner process.

Protocol (UTF-8, one JSON object per line, matching ``agent_protocol.py``):

    runner -> agent   {"type": "reset", "observation": {...}, "info": {...}}
    agent  -> runner  {"action": {...}}
    runner -> agent   {"type": "step", "observation": {...}, "reward": 0.0,
                       "terminated": false, "truncated": false, "info": {...}}

In ``mode="eval"``, ``AuditEnv`` returns ``reward = 0.0`` and empty
``info["diagnostics"]`` on every step including the final one; the real
``EvalResult`` stays inside ``RunResult.eval_result`` on the runner side.

Cross-platform (Windows and Linux, Python 3.11+, stdlib only): uses binary
pipes with bounded ``readline(max_line_bytes + 1)`` in daemon threads and
``queue.Queue`` timeouts (no Unix-only ``select`` or ``fcntl``).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import schemas as S
from env import AuditEnv

DEFAULT_TIMEOUT_SEC = 60.0
DEFAULT_MAX_LINE_BYTES = 256 * 1024
MAX_STDERR_BYTES = 16 * 1024
GRACEFUL_EXIT_SEC = 2.0
EOF_WAIT_SEC = 1.0

AGENT_PROTOCOL_PATH = Path(__file__).resolve().with_name("agent_protocol.py")
DUMMY_AGENTS = ("silent", "random", "crash", "slow", "badjson", "eof", "oversize", "hostile")

_WINDOWS_SAFE_ENV = (
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
    "OS", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)
_FORBIDDEN_ENV_RE = re.compile(r"seed|truth|provenance|private|audit", re.IGNORECASE)


@dataclass(frozen=True)
class RunResult:
    """Structured outcome of running one agent process on one episode."""

    episode_id: str
    eval_result: dict
    trajectory: list
    elapsed_sec: float
    agent_exit_code: Optional[int]
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "eval_result": copy.deepcopy(self.eval_result),
            "trajectory": copy.deepcopy(self.trajectory),
            "elapsed_sec": self.elapsed_sec,
            "agent_exit_code": self.agent_exit_code,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Process environment, command isolation and pipe helpers
# --------------------------------------------------------------------------- #
def build_clean_env() -> dict:
    """Minimal OS environment without secrets, user paths or PYTHONPATH."""
    env: dict = {}
    if os.name == "nt":
        for key in _WINDOWS_SAFE_ENV:
            value = os.environ.get(key)
            if value is not None and not _FORBIDDEN_ENV_RE.search(f"{key}={value}"):
                env[key] = value
    else:
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        env["LANG"] = "C.UTF-8"
        env["LC_ALL"] = "C.UTF-8"
    return env


def _is_python_executable(executable: str) -> bool:
    name = Path(executable).name.lower()
    if name.startswith("python") or name in ("py", "py.exe"):
        return True
    try:
        return os.path.abspath(executable) == os.path.abspath(sys.executable)
    except OSError:
        return False


def _prepare_command(agent_cmd: Sequence[str], isolate_python: bool) -> list:
    if not isinstance(agent_cmd, (list, tuple)) or not agent_cmd:
        raise ValueError("agent_cmd must be a non-empty sequence of strings")
    if not all(isinstance(part, str) and part != "" for part in agent_cmd):
        raise ValueError("every element of agent_cmd must be a non-empty string")
    cmd = list(agent_cmd)
    if not isolate_python or not _is_python_executable(cmd[0]):
        return cmd

    # Scan existing interpreter flags before the script/module/command token.
    existing_flags: set = set()
    idx = 1
    while idx < len(cmd) and cmd[idx].startswith("-") and cmd[idx] != "-":
        flag = cmd[idx]
        if flag in ("-c", "-m"):
            break
        if flag in ("-W", "-X") and idx + 1 < len(cmd):
            idx += 2
            continue
        if not flag.startswith("--"):
            existing_flags.update(flag[1:])
        idx += 1

    to_insert: list = []
    if "I" not in existing_flags:
        to_insert.append("-I")
    if "B" not in existing_flags:
        to_insert.append("-B")
    return [cmd[0], *to_insert, *cmd[1:]]


def _read_lines(stream: Any, max_line_bytes: int, out_queue: "queue.Queue[tuple]") -> None:
    """Bounded binary line reader for ``proc.stdout``."""
    try:
        while True:
            raw = stream.readline(max_line_bytes + 1)
            if not raw:
                out_queue.put(("eof", None))
                return
            if len(raw) > max_line_bytes:
                out_queue.put(("oversize", len(raw)))
                return
            out_queue.put(("line", raw))
    except (OSError, ValueError):
        out_queue.put(("eof", None))


def _write_loop(stream: Any, in_queue: "queue.Queue[Optional[bytes]]",
                ack_queue: "queue.Queue[str]") -> None:
    """Dedicated writer thread so a blocked pipe still honours ``timeout_sec``."""
    while True:
        data = in_queue.get()
        if data is None:
            return
        try:
            stream.write(data)
            stream.flush()
            ack_queue.put("ok")
        except (BrokenPipeError, OSError, ValueError):
            ack_queue.put("broken")
            return


def _drain_stderr(stream: Any, chunks: list, cap: int = MAX_STDERR_BYTES) -> None:
    """Drain ``proc.stderr`` so a verbose crash can never deadlock the child."""
    total = 0
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            if total < cap:
                keep = chunk[:cap - total]
                chunks.append(keep)
                total += len(keep)
    except (OSError, ValueError):
        return


def _parse_agent_line(raw: bytes) -> tuple:
    """Parse one agent JSON line. Returns ``(valid_envelope, action_or_none)``."""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False, None
    if not isinstance(obj, dict) or set(obj.keys()) != {"action"}:
        return False, None
    return True, obj["action"]


def _close_quietly(stream: Any) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except OSError:
        pass


def _kill_and_wait(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _terminate_then_kill(proc: subprocess.Popen, wait_sec: float) -> None:
    if proc.poll() is not None:
        return
    if wait_sec > 0:
        try:
            proc.wait(timeout=wait_sec)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        pass
    _kill_and_wait(proc)


# --------------------------------------------------------------------------- #
# Subprocess runner
# --------------------------------------------------------------------------- #
class SubprocessRunner:
    """Runs an external agent process against ``AuditEnv`` over JSON-Lines."""

    def __init__(
        self,
        agent_cmd: Sequence[str],
        *,
        mode: str = "eval",
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
        isolate_python: bool = True,
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if not isinstance(timeout_sec, (int, float)) or isinstance(timeout_sec, bool) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be a positive number")
        if not S.is_int(max_line_bytes) or max_line_bytes < 64:
            raise ValueError("max_line_bytes must be an integer >= 64")
        self.cmd = _prepare_command(agent_cmd, isolate_python)
        self.mode = mode
        self.timeout_sec = float(timeout_sec)
        self.max_line_bytes = max_line_bytes
        self.isolate_python = isolate_python

    def run(self, seed: int, config: Any = None, *, id_scheme: Any = None) -> RunResult:
        env = AuditEnv(mode=self.mode)
        obs, info = env.reset(seed, config, id_scheme=id_scheme)
        episode_id = env.episode_id or ""
        trajectory: list = [{
            "episode_id": episode_id,
            "step": 0,
            "action": None,
            "observation": copy.deepcopy(obs),
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }]

        started = time.monotonic()
        deadline = started + self.timeout_sec
        workdir = tempfile.mkdtemp(prefix="audit_agent_")
        proc: Optional[subprocess.Popen] = None
        in_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        ack_queue: "queue.Queue[str]" = queue.Queue()
        line_queue: "queue.Queue[tuple]" = queue.Queue()
        stderr_chunks: list = []
        threads: list = []
        abort_reason: Optional[str] = None
        error: Optional[str] = None
        agent_exit_code: Optional[int] = None

        def send_message(payload: dict) -> str:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            in_queue.put(data)
            try:
                return ack_queue.get(timeout=remaining)
            except queue.Empty:
                return "timeout"

        try:
            try:
                proc = subprocess.Popen(
                    self.cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=workdir,
                    env=build_clean_env(),
                    bufsize=0,
                )
            except OSError as exc:
                abort_reason = "agent_exited"
                error = f"spawn_failed: {exc}"

            if proc is not None:
                reader = threading.Thread(
                    target=_read_lines, args=(proc.stdout, self.max_line_bytes, line_queue),
                    daemon=True)
                writer = threading.Thread(
                    target=_write_loop, args=(proc.stdin, in_queue, ack_queue),
                    daemon=True)
                err_reader = threading.Thread(
                    target=_drain_stderr, args=(proc.stderr, stderr_chunks),
                    daemon=True)
                threads = [reader, writer, err_reader]
                for t in threads:
                    t.start()

                write_status = send_message({"type": "reset", "observation": obs, "info": info})
                if write_status == "timeout":
                    abort_reason = "timeout"
                elif write_status == "broken":
                    abort_reason = "agent_exited"

                while abort_reason is None and not env.done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        abort_reason = "timeout"
                        break
                    try:
                        event, payload = line_queue.get(timeout=remaining)
                    except queue.Empty:
                        abort_reason = "timeout"
                        break

                    if event == "eof":
                        abort_reason = "agent_exited"
                        break
                    if event == "oversize":
                        abort_reason = "oversized_message"
                        break

                    valid_envelope, action = _parse_agent_line(payload)
                    step_action = action if valid_envelope else None
                    obs, reward, terminated, truncated, info = env.step(step_action)
                    trajectory.append({
                        "episode_id": episode_id,
                        "step": info["step"],
                        "action": copy.deepcopy(step_action),
                        "observation": copy.deepcopy(obs),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    })

                    step_msg = {
                        "type": "step",
                        "observation": obs,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                    write_status = send_message(step_msg)
                    if terminated or truncated:
                        break
                    if write_status == "timeout":
                        abort_reason = "timeout"
                        break
                    if write_status == "broken":
                        abort_reason = "agent_exited"
                        break

            if abort_reason is not None and not env.done:
                if proc is not None and abort_reason in ("timeout", "oversized_message"):
                    _kill_and_wait(proc)
                obs, reward, terminated, truncated, info = env.abort(abort_reason)
                trajectory.append({
                    "episode_id": episode_id,
                    "step": info["step"] + 1,
                    "action": None,
                    "observation": copy.deepcopy(obs),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                })
                if error is None:
                    error = abort_reason
        finally:
            in_queue.put(None)
            if proc is not None:
                _close_quietly(proc.stdin)
                if abort_reason in ("timeout", "oversized_message"):
                    _kill_and_wait(proc)
                elif abort_reason == "agent_exited":
                    _terminate_then_kill(proc, EOF_WAIT_SEC)
                else:
                    wait_budget = min(GRACEFUL_EXIT_SEC, max(0.2, deadline - time.monotonic()))
                    _terminate_then_kill(proc, wait_budget)
                agent_exit_code = proc.returncode
                _close_quietly(proc.stdout)
                _close_quietly(proc.stderr)
            for t in threads:
                t.join(timeout=0.5)
            shutil.rmtree(workdir, ignore_errors=True)

        elapsed_sec = time.monotonic() - started
        eval_result = env.evaluator.final_result()
        assert eval_result is not None
        return RunResult(
            episode_id=episode_id,
            eval_result=eval_result,
            trajectory=trajectory,
            elapsed_sec=elapsed_sec,
            agent_exit_code=agent_exit_code,
            error=error,
        )


def dummy_agent_command(
    agent: str,
    *,
    python_executable: Optional[str] = None,
    agent_seed: int = 0,
    steps: int = 12,
    actions_path: Optional[Any] = None,
    sleep_sec: Optional[float] = None,
) -> list:
    """Build the command line for one of the built-in agents in ``agent_protocol.py``."""
    cmd = [python_executable or sys.executable, str(AGENT_PROTOCOL_PATH), "--agent", agent]
    if agent == "random":
        cmd += ["--agent-seed", str(agent_seed), "--steps", str(steps)]
    elif agent == "replay":
        if actions_path is None:
            raise ValueError("actions_path is required for the replay agent")
        cmd += ["--actions", str(actions_path)]
    elif agent == "slow" and sleep_sec is not None:
        cmd += ["--sleep", str(sleep_sec)]
    return cmd


def run_episode_subprocess(
    seed: int,
    agent_cmd: Sequence[str],
    *,
    config: Any = None,
    id_scheme: Any = None,
    mode: str = "eval",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    isolate_python: bool = True,
) -> RunResult:
    """Convenience wrapper around ``SubprocessRunner.run``."""
    runner = SubprocessRunner(
        agent_cmd,
        mode=mode,
        timeout_sec=timeout_sec,
        max_line_bytes=max_line_bytes,
        isolate_python=isolate_python,
    )
    return runner.run(seed, config, id_scheme=id_scheme)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_config_arg(value: Optional[str]) -> dict:
    if value is None:
        return {}
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _parse_agent_cmd_arg(tokens: Sequence[str]) -> list:
    if len(tokens) == 1 and " " in tokens[0] and not Path(tokens[0]).exists():
        parts = shlex.split(tokens[0], posix=(os.name != "nt"))
        if os.name == "nt":
            parts = [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"') else p
                     for p in parts]
        return parts
    return list(tokens)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an AuditEnv episode with the agent isolated in a subprocess.")
    parser.add_argument("--seed", type=int, default=0, help="episode seed")
    parser.add_argument("--mode", choices=("train", "eval"), default="eval")
    parser.add_argument("--config", help="JSON string or path to a JSON EpisodeConfig file")
    parser.add_argument("--id-scheme", choices=S.ID_SCHEMES, help="override config.id_scheme")
    parser.add_argument("--id-salt", type=int, help="override config.id_salt")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-line-bytes", type=int, default=DEFAULT_MAX_LINE_BYTES)
    parser.add_argument("--trajectory-out", help="optional path to write trajectories.jsonl")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dummy-agent", choices=DUMMY_AGENTS,
                       help="run a built-in agent from agent_protocol.py")
    group.add_argument("--agent-cmd", nargs="+", help="command to launch an external agent")

    args = parser.parse_args(argv)
    config = _load_config_arg(args.config)
    if args.id_scheme is not None:
        config["id_scheme"] = args.id_scheme
    if args.id_salt is not None:
        config["id_salt"] = args.id_salt

    if args.dummy_agent is not None:
        cmd = dummy_agent_command(args.dummy_agent)
    else:
        cmd = _parse_agent_cmd_arg(args.agent_cmd)

    result = run_episode_subprocess(
        args.seed,
        cmd,
        config=config or None,
        mode=args.mode,
        timeout_sec=args.timeout,
        max_line_bytes=args.max_line_bytes,
    )
    if args.trajectory_out:
        out_path = Path(args.trajectory_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(S.to_jsonl(result.trajectory), encoding="utf-8", newline="")

    summary = {
        "episode_id": result.episode_id,
        "eval_result": result.eval_result,
        "steps": len(result.trajectory) - 1,
        "elapsed_sec": round(result.elapsed_sec, 4),
        "agent_exit_code": result.agent_exit_code,
        "error": result.error,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
