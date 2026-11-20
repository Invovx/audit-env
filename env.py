"""env.py - ``AuditEnv``: a JSON-compatible, Gymnasium-style environment.

``reset(seed, config=None) -> (observation, info)``
``step(action) -> (observation, reward, terminated, truncated, info)``

Isolation: the environment keeps only the *public* tables.  The reference
errors live inside the ``Evaluator`` and are never copied into observations,
errors or ``info``.  Observations contain no seed, no file system paths and no
reference ids.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

import safe_calc
import schemas as S
from evaluator import Evaluator, EvalResult
from generator import generate_episode

TASK_DESCRIPTION = (
    "You are auditing the books of one fictional company for one reporting period. "
    "Inspect the public tables (ledger, invoices, payments, fx) with the tools below and report "
    "only errors you can prove from records. A clean book legitimately has no errors; "
    "unusual-looking but legitimate cases exist (split or combined payments, returns/credit "
    "notes, recurring identical amounts, boundary and out-of-period documents).\n"
    "Tools (exactly five), passed as {\"tool\": ..., ...}:\n"
    "1. list_records {table: ledger|invoices|payments|fx, cursor: str|null, limit: int<=50}\n"
    "2. get_record {table, id}  (ids: ledger->row_id, invoices/payments->doc_id, fx->fx_id)\n"
    "3. get_document {doc_id}\n"
    "4. calculator {expression}  (decimal arithmetic: + - * / // % ** and parentheses)\n"
    "5. submit {findings: {episode_id, findings: [...]}}\n"
    "Budget: 100 steps in total including submit. If step 100 is not submit, your answer is "
    "empty. Every inspection step slightly lowers the final reward. Submit exactly once.\n"
    "Each finding: {type, affected_ids, claimed_amount_minor, currency, evidence_ids, "
    "explanation}; at most 30 findings, 10 evidence_ids each, explanation up to 1000 "
    "characters. Money is an integer in minor units of the stated currency.\n"
    "The ledger is kept in the base currency; documents keep their own currency (see policy for "
    "the FX translation and settlement rules).\n"
    "Error types:\n"
    "- duplicate_invoice: the same invoice entered twice (two invoice documents plus duplicated "
    "postings). affected_ids = doc_ids of both invoice documents; claimed_amount_minor = "
    "total_minor of the invoice; currency = invoice currency; evidence_ids = ids of records "
    "that prove it (documents and/or ledger rows).\n"
    "- period_cutoff_error: an invoice document dated outside the reporting period "
    "(compare absolute instants, see policy) whose postings are booked inside the period. "
    "affected_ids = [doc_id of that invoice]; claimed_amount_minor = its total_minor; "
    "currency = its currency; evidence_ids = the invoice doc_id plus a ledger row_id of its "
    "postings.\n"
    "- partial_payment_allocation: a payment document applies its amount to an invoice, but the "
    "ledger applies only part of it (or none) to receivables and leaves the rest in unapplied cash "
    "(2100). affected_ids = [payment doc_id, invoice doc_id]; claimed_amount_minor = the "
    "unapplied amount in the payment currency; currency = payment currency; evidence_ids = the "
    "payment doc_id plus the ledger row_id that shows the unapplied or short-applied amount. "
    "A legitimate overpayment (payment larger than the invoice) also leaves cash in 2100."
)

_MAX_MESSAGE = 200


def _clip(text: Any, limit: int = 60) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


class _ToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message[:_MAX_MESSAGE]


class AuditEnv:
    """Financial-audit environment (one company, one reporting period per episode)."""

    metadata = {"name": "AuditEnv-v1", "modes": ("train", "eval")}
    MAX_STEPS = S.MAX_STEPS

    def __init__(self, mode: str = "train") -> None:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self._evaluator: Optional[Evaluator] = None
        self._tables: dict = {}
        self._key: dict = {}
        self._index: dict = {}
        self._documents: dict = {}
        self._manifest: dict = {}
        self._policy_summary: dict = {}
        self._steps_used = 0
        self._done = False
        self._terminated = False
        self._truncated = False

    # ------------------------------------------------------------------ #
    @property
    def evaluator(self) -> Evaluator:
        """The grader. Harness-side access to the final result in ``eval`` mode."""
        if self._evaluator is None:
            raise RuntimeError("reset() must be called first")
        return self._evaluator

    @property
    def steps_remaining(self) -> int:
        return self.MAX_STEPS - self._steps_used

    # ------------------------------------------------------------------ #
    def reset(self, seed: int, config: Any = None) -> tuple:
        episode = generate_episode(seed, config)   # validates seed/config, may raise
        fx_rows = [{"fx_id": S.fx_record_id(r), **r} for r in episode.fx]
        tables = {
            "ledger": copy.deepcopy(episode.ledger),
            "invoices": copy.deepcopy(episode.invoices),
            "payments": copy.deepcopy(episode.payments),
            "fx": fx_rows,
        }
        key = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "fx_id"}
        self._tables = tables
        self._key = key
        self._index = {t: {row[key[t]]: i for i, row in enumerate(rows)} for t, rows in tables.items()}
        self._documents = {d["doc_id"]: ("invoice", d) for d in tables["invoices"]}
        self._documents.update({d["doc_id"]: ("payment", d) for d in tables["payments"]})
        self._manifest = copy.deepcopy(episode.manifest)
        self._policy_summary = S.policy_summary(episode.policy)
        self._evaluator = Evaluator(episode.episode_id, episode.truth,
                                    episode.policy["amount_tolerance_minor"])
        self._steps_used = 0
        self._done = self._terminated = self._truncated = False
        observation = {
            "kind": "task",
            "task": TASK_DESCRIPTION,
            "manifest": copy.deepcopy(self._manifest),
            "policy_summary": copy.deepcopy(self._policy_summary),
            "steps_remaining": self.MAX_STEPS,
        }
        return observation, self._info()

    # ------------------------------------------------------------------ #
    def step(self, action: Any) -> tuple:
        if self._evaluator is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            obs = self._error_obs("episode_finished", "The episode is over; call reset().")
            return obs, 0.0, self._terminated, self._truncated, self._info()

        self._steps_used += 1
        inspection_steps = self._steps_used - 1
        is_submit = isinstance(action, dict) and action.get("tool") == "submit"

        if is_submit:
            payload = action["findings"] if set(action) == {"tool", "findings"} else None
            result = self._evaluator.finalize(payload, inspection_steps)
            self._done = self._terminated = True
            obs = {"kind": "submitted", "message": "Submission received. The episode is finished.",
                   "steps_remaining": self.steps_remaining}
            return obs, *self._terminal(result, terminated=True)

        if self._steps_used >= self.MAX_STEPS:
            # every one of the 100 steps was an inspection step; the answer is empty
            result = self._evaluator.finalize(None, self._steps_used, truncated=True)
            self._done = self._truncated = True
            obs = {"kind": "truncated",
                   "message": "Step budget exhausted without submit; the answer is empty.",
                   "steps_remaining": 0}
            return obs, *self._terminal(result, terminated=False)

        try:
            obs = self._dispatch(action)
        except _ToolError as exc:
            obs = self._error_obs(exc.code, exc.message)
        except safe_calc.CalcError as exc:
            obs = self._error_obs("calculator_error", str(exc))
        return obs, 0.0, False, False, self._info()

    # ------------------------------------------------------------------ #
    def _terminal(self, result: EvalResult, terminated: bool) -> tuple:
        if self.mode == "train":
            return result.reward, terminated, not terminated, self._info(result.public_diagnostics())
        return 0.0, terminated, not terminated, self._info()

    def _info(self, diagnostics: Optional[dict] = None) -> dict:
        return {"mode": self.mode, "step": self._steps_used,
                "steps_remaining": self.steps_remaining,
                "diagnostics": diagnostics if diagnostics is not None else {}}

    def _error_obs(self, code: str, message: str) -> dict:
        return {"kind": "error", "error": {"code": code, "message": message[:_MAX_MESSAGE]},
                "steps_remaining": self.steps_remaining}

    # ------------------------------------------------------------------ #
    def _dispatch(self, action: Any) -> dict:
        if not isinstance(action, dict):
            raise _ToolError("invalid_action", "Action must be an object with a 'tool' field.")
        tool = action.get("tool")
        if not isinstance(tool, str):
            raise _ToolError("invalid_action", "Action must contain a string 'tool'.")
        handlers = {
            "list_records": self._list_records,
            "get_record": self._get_record,
            "get_document": self._get_document,
            "calculator": self._calculator,
        }
        handler = handlers.get(tool)
        if handler is None:
            raise _ToolError("unknown_tool", f"Unknown tool '{_clip(tool)}'. Available tools: "
                             "list_records, get_record, get_document, calculator, submit.")
        return handler(action)

    @staticmethod
    def _check_keys(action: dict, required: tuple, optional: tuple = ()) -> None:
        missing = [k for k in required if k not in action]
        if missing:
            raise _ToolError("invalid_argument", f"Missing argument(s): {', '.join(missing)}.")
        extra = [str(k) for k in action if k != "tool" and k not in required and k not in optional]
        if extra:
            raise _ToolError("invalid_argument", f"Unexpected argument(s): {_clip(', '.join(extra))}.")

    @staticmethod
    def _table_arg(action: dict) -> str:
        table = action.get("table")
        if not isinstance(table, str) or table not in S.TABLES:
            raise _ToolError("invalid_argument",
                             "table must be one of: " + ", ".join(S.TABLES) + ".")
        return table

    def _list_records(self, action: dict) -> dict:
        self._check_keys(action, ("table",), ("cursor", "limit"))
        table = self._table_arg(action)
        limit = action.get("limit", S.MAX_LIST_LIMIT)
        if not S.is_int(limit) or limit < 1:
            raise _ToolError("invalid_argument", "limit must be an integer >= 1.")
        limit = min(limit, S.MAX_LIST_LIMIT)
        rows = self._tables[table]
        start = 0
        cursor = action.get("cursor")
        if cursor is not None:
            if not isinstance(cursor, str):
                raise _ToolError("invalid_argument", "cursor must be a string or null.")
            prefix, sep, last_id = cursor.partition(":")
            if sep != ":" or prefix != table or last_id not in self._index[table]:
                raise _ToolError("invalid_argument", "Invalid cursor for this table.")
            start = self._index[table][last_id] + 1
        page = rows[start:start + limit]
        has_more = start + len(page) < len(rows)
        next_cursor = f"{table}:{page[-1][self._key[table]]}" if (page and has_more) else None
        return {"kind": "records", "table": table, "records": copy.deepcopy(page),
                "next_cursor": next_cursor, "has_more": has_more,
                "steps_remaining": self.steps_remaining}

    def _get_record(self, action: dict) -> dict:
        self._check_keys(action, ("table", "id"))
        table = self._table_arg(action)
        record_id = action.get("id")
        if not isinstance(record_id, str):
            raise _ToolError("invalid_argument", "id must be a string.")
        position = self._index[table].get(record_id)
        if position is None:
            raise _ToolError("unknown_id", f"No record '{_clip(record_id)}' in table '{table}'.")
        return {"kind": "record", "table": table,
                "record": copy.deepcopy(self._tables[table][position]),
                "steps_remaining": self.steps_remaining}

    def _get_document(self, action: dict) -> dict:
        self._check_keys(action, ("doc_id",))
        doc_id = action.get("doc_id")
        if not isinstance(doc_id, str):
            raise _ToolError("invalid_argument", "doc_id must be a string.")
        found = self._documents.get(doc_id)
        if found is None:
            raise _ToolError("unknown_id", f"No document '{_clip(doc_id)}'.")
        doc_type, document = found
        return {"kind": "document", "doc_type": doc_type, "document": copy.deepcopy(document),
                "steps_remaining": self.steps_remaining}

    def _calculator(self, action: dict) -> dict:
        self._check_keys(action, ("expression",))
        expression = action.get("expression")
        if not isinstance(expression, str):
            raise _ToolError("invalid_argument", "expression must be a string.")
        result = safe_calc.evaluate(expression)       # CalcError handled by step()
        return {"kind": "calculation", "expression": expression.strip(), "result": result,
                "steps_remaining": self.steps_remaining}
