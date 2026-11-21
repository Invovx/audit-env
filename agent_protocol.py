"""agent_protocol.py - agent-side half of the runner JSON-Lines protocol.

This file is executed *inside the agent process*.  It is deliberately
self-contained (standard library only, no project imports), so it works when
started with ``python -I`` from an empty working directory.

Protocol (one JSON object per line, UTF-8):

    runner -> agent   {"type": "reset", "observation": {...}, "info": {...}}
    agent  -> runner  {"action": {...}}
    runner -> agent   {"type": "step", "observation": {...}, "reward": 0.0,
                       "terminated": false, "truncated": false, "info": {...}}

The agent answers every ``reset``/``step`` message with exactly one action
until a message has ``terminated`` or ``truncated`` set; then it exits.

Writing your own agent::

    from agent_protocol import run_policy

    def policy(message):              # message = last runner message
        return {"tool": "submit", "findings": {
            "episode_id": message["observation"]["manifest"]["episode_id"]
            if message["type"] == "reset" else EPISODE_ID, "findings": []}}

    run_policy(policy)

Built-in agents for tests and demos (``python agent_protocol.py --agent NAME``):
``silent``, ``random``, ``replay``, ``crash``, ``slow``, ``badjson``, ``eof``,
``oversize``, ``hostile``.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import random
import re
import sys
import time
from typing import Any, Callable, Optional

TOOLS = ("list_records", "get_record", "get_document", "calculator", "submit")
TABLES = ("ledger", "invoices", "payments", "fx")


class AgentSession:
    """Line-oriented JSON channel over stdin/stdout (UTF-8, no buffering surprises)."""

    def __init__(self, stdin: Any = None, stdout: Any = None) -> None:
        self._in = stdin if stdin is not None else io.TextIOWrapper(
            sys.stdin.buffer, encoding="utf-8", errors="replace")
        self._out = stdout if stdout is not None else io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", newline="\n", write_through=True)
        self.received: list = []          # raw lines, for self-inspection by tests

    def recv(self) -> Optional[dict]:
        line = self._in.readline()
        if not line:
            return None
        self.received.append(line)
        return json.loads(line)

    def send_raw(self, text: str) -> None:
        self._out.write(text if text.endswith("\n") else text + "\n")
        self._out.flush()

    def send_action(self, action: Any) -> None:
        self.send_raw(json.dumps({"action": action}, ensure_ascii=False, separators=(",", ":")))


def is_final(message: dict) -> bool:
    return bool(message.get("terminated")) or bool(message.get("truncated"))


def run_policy(policy: Callable[[dict], Any], session: Optional[AgentSession] = None) -> int:
    """Drive ``policy(message) -> action`` until the episode ends. Returns an exit code."""
    session = session or AgentSession()
    message = session.recv()
    while message is not None and not is_final(message):
        session.send_action(policy(message))
        message = session.recv()
    return 0


# --------------------------------------------------------------------------- #
# Built-in agents
# --------------------------------------------------------------------------- #
class _EpisodeMemory:
    """Remembers the episode id from the first observation."""

    def __init__(self) -> None:
        self.episode_id = ""

    def update(self, message: dict) -> None:
        if message.get("type") == "reset":
            self.episode_id = message["observation"]["manifest"]["episode_id"]

    def submit(self, findings: list) -> dict:
        return {"tool": "submit", "findings": {"episode_id": self.episode_id, "findings": findings}}


def agent_silent(args: argparse.Namespace) -> int:
    """Submits an empty report immediately (the correct answer on clean books)."""
    memory = _EpisodeMemory()

    def policy(message: dict) -> dict:
        memory.update(message)
        return memory.submit([])

    return run_policy(policy)


def agent_random(args: argparse.Namespace) -> int:
    """Random exploration (valid and invalid calls), then an empty submit."""
    rng = random.Random(args.agent_seed)
    memory = _EpisodeMemory()
    seen_ids: list = []
    budget = rng.randint(3, max(3, args.steps))
    state = {"used": 0}

    def policy(message: dict) -> dict:
        memory.update(message)
        obs = message.get("observation", {})
        for record in obs.get("records", []) or []:
            for key in ("row_id", "doc_id", "fx_id"):
                if key in record:
                    seen_ids.append(record[key])
        state["used"] += 1
        if state["used"] > budget:
            return memory.submit([])
        roll = rng.randint(1, 6)
        if roll <= 2 or not seen_ids:
            return {"tool": "list_records", "table": rng.choice(TABLES), "cursor": None,
                    "limit": rng.randint(1, 50)}
        if roll == 3:
            return {"tool": "get_record", "table": rng.choice(TABLES), "id": rng.choice(seen_ids)}
        if roll == 4:
            return {"tool": "get_document", "doc_id": rng.choice(seen_ids)}
        if roll == 5:
            return {"tool": "calculator", "expression": f"{rng.randint(1, 999)} * {rng.randint(1, 99)}"}
        return {"tool": rng.choice(("search", "sql", "open_file"))}

    return run_policy(policy)


def agent_replay(args: argparse.Namespace) -> int:
    """Replays actions from a JSON file; the string "$EPISODE_ID" is substituted.

    Used by tests to emulate an oracle *from outside*: the test harness (which
    knows the answer) writes the actions; the agent process itself never sees
    the reference data or the seed.
    """
    with open(args.actions, "r", encoding="utf-8") as handle:
        actions = json.load(handle)
    memory = _EpisodeMemory()
    queue = list(actions)

    def substitute(value: Any) -> Any:
        if value == "$EPISODE_ID":
            return memory.episode_id
        if isinstance(value, list):
            return [substitute(v) for v in value]
        if isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        return value

    def policy(message: dict) -> dict:
        memory.update(message)
        return substitute(queue.pop(0)) if queue else memory.submit([])

    return run_policy(policy)


def agent_crash(args: argparse.Namespace) -> int:
    session = AgentSession()
    session.recv()
    session.send_action({"tool": "list_records", "table": "ledger", "limit": 2})
    session.recv()
    raise RuntimeError("simulated agent crash")


def agent_slow(args: argparse.Namespace) -> int:
    session = AgentSession()
    session.recv()
    time.sleep(args.sleep)
    session.send_action({"tool": "calculator", "expression": "1+1"})
    return 0


def agent_badjson(args: argparse.Namespace) -> int:
    """Sends broken lines first, then behaves like ``silent``."""
    session = AgentSession()
    message = session.recv()
    episode_id = message["observation"]["manifest"]["episode_id"]
    for garbage in ("this is not json", "{\"action\": ", "[1, 2, 3]", "{\"no_action\": 1}"):
        session.send_raw(garbage)
        message = session.recv()
    session.send_action({"tool": "submit", "findings": {"episode_id": episode_id, "findings": []}})
    session.recv()
    return 0


def agent_eof(args: argparse.Namespace) -> int:
    """Reads the first observation and closes its output without acting."""
    session = AgentSession()
    session.recv()
    sys.stdout.close()
    return 0


def agent_oversize(args: argparse.Namespace) -> int:
    session = AgentSession()
    session.recv()
    session.send_raw(json.dumps({"action": {"tool": "calculator", "expression": "1" * (400 * 1024)}}))
    time.sleep(5)
    return 0


_SUSPICIOUS = re.compile(r"truth|provenance|issue_id|ISS-\d|valid_evidence|expected_amount|"
                         r"seed|private|[A-Za-z]:[\\/]|/home/|/tmp/|/Users/", re.IGNORECASE)


def _iter_strings(node: Any):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_strings(v)


def agent_hostile(args: argparse.Namespace) -> int:
    """Looks everywhere it can for the reference answer and reports what it found.

    The report goes into the explanation of an (empty-handed) submission, so
    the harness can inspect it in the trajectory.
    """
    session = AgentSession()
    message = session.recv()
    episode_id = message["observation"]["manifest"]["episode_id"]
    # spend one step to also see a normal observation
    session.send_action({"tool": "list_records", "table": "ledger", "limit": 3})
    session.recv()

    names = {"AuditEnv", "Evaluator", "Episode", "EvalResult", "Books", "SubprocessRunner"}
    live = sorted({type(o).__name__ for o in gc.get_objects() if type(o).__name__ in names})
    modules = sorted(m for m in ("env", "evaluator", "generator", "runner", "schemas")
                     if m in sys.modules)
    env_hits = sorted(k for k, v in os.environ.items()
                      if re.search(r"seed|truth|provenance|private|audit", k + "=" + v, re.I))
    cwd_entries = sorted(os.listdir("."))
    files = [p for p in ("truth.json", "provenance.json", os.path.join("private", "truth.json"),
                         os.path.join("..", "private", "truth.json")) if os.path.exists(p)]
    leak_hits = set()
    for raw_line in session.received:
        for text_val in _iter_strings(json.loads(raw_line)):
            for m in _SUSPICIOUS.finditer(text_val):
                leak_hits.add(m.group(0).lower())
    leaks = sorted(leak_hits)
    report = {"live": live, "modules": modules, "env": env_hits, "cwd": cwd_entries,
              "files": files, "leaks": leaks, "argv": sys.argv[1:],
              "env_keys": sorted(os.environ)}
    text = json.dumps(report, separators=(",", ":"))[:1000]
    session.send_action({"tool": "submit", "findings": {"episode_id": episode_id, "findings": [{
        "type": "duplicate_invoice", "affected_ids": ["X"], "claimed_amount_minor": 0,
        "currency": "KZT", "evidence_ids": [], "explanation": text}]}})
    session.recv()
    return 0


AGENTS = {
    "silent": agent_silent, "random": agent_random, "replay": agent_replay,
    "crash": agent_crash, "slow": agent_slow, "badjson": agent_badjson, "eof": agent_eof,
    "oversize": agent_oversize, "hostile": agent_hostile,
}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Built-in AuditEnv agents (JSON-Lines over stdio).")
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--agent-seed", type=int, default=0, help="RNG seed of the random agent")
    parser.add_argument("--steps", type=int, default=12, help="max exploration steps (random)")
    parser.add_argument("--actions", help="JSON file with actions (replay)")
    parser.add_argument("--sleep", type=float, default=30.0, help="seconds to sleep (slow)")
    args = parser.parse_args(argv)
    if args.agent == "replay" and not args.actions:
        parser.error("--actions is required for the replay agent")
    return AGENTS[args.agent](args)


if __name__ == "__main__":
    sys.exit(main())
