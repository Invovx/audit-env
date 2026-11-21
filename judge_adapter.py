"""judge_adapter.py - the jury's single entry point (Genesis FinTech check protocol, 20.11.2026).

    generate_episode(seed, config, output_dir) -> None
    make_env() -> object with reset(seed, config) and step(action)
    score_episode(episode_dir, findings, inspection_steps) -> dict

Translation only: the adapter never chooses actions and never changes the task.

* Action envelope ``{"name": <action>, "args": {...}}`` -> internal
  ``{"tool": <action>, **args}``; names: list_records, get_record, get_document,
  calculator, submit (args ``{"findings": <findings.json object>}``).
* First observation ``{"task", "manifest", "policy", "remaining_steps": 100}``;
  then ``{"result", "error", "remaining_steps"}``.
* reset config ``{"mode": "closed"|"train", "episode_dir": ..., "team_config": {...}}``.
  With ``episode_dir`` the episode is loaded from disk exactly as given (never
  regenerated, ids untouched); without it the team generator runs with
  ``seed`` and ``team_config``.  ``closed`` = reward 0 for the policy,
  including the terminal step; the true reward stays with the evaluator.
* 100 actions per episode including submit; submit is not an inspection step.
  60 s wall clock from reset (policy time included) -> truncated, empty answer.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import episode_io
import generator
import schemas as S
from env import AuditEnv
from evaluator import Evaluator

ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")
TIME_LIMIT_SECONDS = 60.0
_MODES = {"closed": "eval", "eval": "eval", "train": "train"}


# --------------------------------------------------------------------------- #
def _team_config(config: Any) -> dict:
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise S.SchemaError("config: must be an object")
    if "team_config" in config:
        inner = config["team_config"]
        return dict(inner) if isinstance(inner, dict) else {}
    return {k: v for k, v in config.items() if k not in ("mode", "episode_dir")}


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Write one episode: ``output_dir/public/*`` (agent-visible files) and
    ``output_dir/private/{truth.json, provenance.json}`` (jury only)."""
    episode = generator.generate_episode(seed, _team_config(config))
    root = Path(output_dir)
    generator.write_episode(episode, root)
    (root / "config.json").write_text(
        S.dump_json({"seed": seed, "team_config": _team_config(config),
                     "generator_version": S.GENERATOR_VERSION}), encoding="utf-8")


# --------------------------------------------------------------------------- #
class JudgeEnv:
    """reset/step in the jury's envelope around ``env.AuditEnv``."""

    def __init__(self) -> None:
        self._env: Optional[AuditEnv] = None
        self._policy: dict = {}
        self._t0 = 0.0
        self._clock = time.monotonic           # replaceable in tests (time-limit check C18)

    # ------------------------------------------------------------------ #
    def reset(self, seed: Any = None, config: Any = None) -> tuple:
        config = config or {}
        if not isinstance(config, dict):
            raise S.SchemaError("config: must be an object")
        mode = _MODES.get(config.get("mode", "train"))
        if mode is None:
            raise S.SchemaError("config.mode: must be 'closed' or 'train'")
        self._env = AuditEnv(mode)
        if config.get("episode_dir"):
            self._env.reset_from_dir(config["episode_dir"])
            self._policy = episode_io.load_public(config["episode_dir"])["policy"]
        else:
            episode = generator.generate_episode(seed, _team_config(config))
            self._env._install({"manifest": episode.manifest, "policy": episode.policy,
                                "ledger": episode.ledger, "invoices": episode.invoices,
                                "payments": episode.payments, "fx": episode.fx}, episode.truth)
            self._policy = episode.policy
        self._t0 = self._clock()
        observation = {
            "task": self._task(),
            "manifest": self._env._manifest,
            "policy": self._policy,
            "remaining_steps": self._env.steps_remaining,
        }
        return _jsonify(observation), {}

    @staticmethod
    def _task() -> str:
        from env import TASK_DESCRIPTION
        return TASK_DESCRIPTION

    # ------------------------------------------------------------------ #
    def step(self, action: Any) -> tuple:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        env = self._env
        if not env.done and self._clock() - self._t0 > TIME_LIMIT_SECONDS:
            obs, reward, terminated, truncated, info = env.abort("timeout")
            return self._wrap(obs), float(reward), bool(terminated), bool(truncated), self._public_info(info)

        internal, submit_error = self._translate(action)
        obs, reward, terminated, truncated, info = env.step(internal)
        wrapped = self._wrap(obs)
        if obs.get("kind") == "submitted" and submit_error is not None:
            wrapped = {"result": None, "error": {"code": "invalid_submission", "message": submit_error},
                       "remaining_steps": env.steps_remaining}
        return wrapped, float(reward), bool(terminated), bool(truncated), self._public_info(info)

    def _translate(self, action: Any) -> tuple:
        """Envelope -> internal action.  Malformed envelopes become an internal
        invalid action, so they cost one step like any other error."""
        if not isinstance(action, dict) or set(action) - {"name", "args"} or "name" not in action:
            return {"tool": None}, None
        name, args = action.get("name"), action.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return {"tool": None}, None
        if name == "submit":
            if set(args) != {"findings"}:
                return {"tool": "submit", "findings": None}, "args must be {\"findings\": {...}}"
            try:
                S.parse_submission(args["findings"], self._env._manifest["episode_id"])
                error = None
            except S.SchemaError as exc:                 # structure only - never correctness
                error = str(exc)[:200]
            return {"tool": "submit", "findings": args["findings"]}, error
        if "tool" in args:
            return {"tool": "__invalid__"}, None
        return {"tool": name, **args}, None

    def _wrap(self, obs: dict) -> dict:
        kind = obs.get("kind")
        remaining = obs.get("steps_remaining", self._env.steps_remaining)
        result: Any = None
        error = None
        if kind == "records":
            result = {"records": obs["records"], "next_cursor": obs["next_cursor"]}
        elif kind == "record":
            result = obs["record"]
        elif kind == "document":
            result = obs["document"]
        elif kind == "calculation":
            result = obs["result"]
        elif kind == "submitted":
            result = {"accepted": True}
        elif kind == "error":
            error = {"code": obs["error"]["code"], "message": obs["error"]["message"]}
        return _jsonify({"result": result, "error": error, "remaining_steps": remaining})

    def _public_info(self, info: dict) -> dict:
        out = {"steps_used": info.get("step")}
        if self._env.mode == "train" and info.get("diagnostics"):
            out["diagnostics"] = info["diagnostics"]          # train only; empty in closed mode
        return _jsonify(out)


def make_env() -> JudgeEnv:
    return JudgeEnv()


# --------------------------------------------------------------------------- #
def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    """Closed evaluator call.  Valid submit -> tp, fp, fn, precision, recall, f1,
    return.  Invalid structure -> return = -1 and the other six fields null."""
    data = episode_io.load_public(episode_dir)
    truth = episode_io.load_truth(episode_dir)
    payload = findings
    if isinstance(findings, dict) and set(findings) == {"findings"} and isinstance(findings["findings"], dict):
        payload = findings["findings"]           # accept the submit args wrapper too
    evaluator = Evaluator(data["manifest"]["episode_id"], truth, episode_io.tolerance(data),
                          episode_io.public_ids(data))
    result = evaluator.finalize(payload, inspection_steps)      # finalize() logs invalid structure
    if not result.valid_submission:
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                "f1": None, "return": -1.0}
    return {"tp": result.tp, "fp": result.fp, "fn": result.fn, "precision": result.precision,
            "recall": result.recall, "f1": result.f1, "return": result.reward}


def _jsonify(value: Any) -> Any:
    """Deep copy restricted to JSON types (dict/list/str/int/float/bool/None)."""
    import json
    return json.loads(json.dumps(value, ensure_ascii=False))
