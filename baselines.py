"""baselines.py - the four reference policies required by the case.

Every policy sees only observations returned by ``AuditEnv`` (the JSON
contract).  The single exception is ``OraclePolicy``: it is built by the
*harness* from the closed reference data and exists only to check the maximum
reward; it is never given to an agent.

Interface: ``policy.reset(first_observation)`` then ``policy.act(observation)``
returns the next action (a JSON-compatible dict).
"""
from __future__ import annotations

import random
from typing import Any, Optional, Sequence

import schemas as S

KNOWN_TYPES = S.ALL_ISSUE_TYPES


def _submit(episode_id: str, findings: list) -> dict:
    return {"tool": "submit", "findings": {"episode_id": episode_id, "findings": findings[:S.MAX_FINDINGS]}}


class Policy:
    name = "base"

    def reset(self, observation: dict) -> None:
        self.manifest = observation["manifest"]
        self.episode_id = self.manifest["episode_id"]

    def act(self, observation: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class _TableReader(Policy):
    """Helper: pages through whole tables with stable cursors."""

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self.rows: dict = {t: [] for t in S.TABLES}
        self._queue: list = []
        self._table: Optional[str] = None
        self._cursor: Optional[str] = None

    def _read(self, observation: dict) -> Optional[dict]:
        """Consume the last page and return the next list_records action, or None when done."""
        if observation.get("kind") == "records" and observation.get("table") == self._table:
            self.rows[self._table].extend(observation["records"])
            if observation["next_cursor"] is not None:
                self._cursor = observation["next_cursor"]
                return {"tool": "list_records", "table": self._table, "cursor": self._cursor,
                        "limit": S.MAX_LIST_LIMIT}
            self._table = None
        if self._table is None and self._queue:
            self._table, self._cursor = self._queue.pop(0), None
            return {"tool": "list_records", "table": self._table, "cursor": None,
                    "limit": S.MAX_LIST_LIMIT}
        return None


class RandomExplorer(Policy):
    """Random inspection with a fixed seed, then a random (usually wrong) guess."""

    name = "random"

    def __init__(self, seed: int = 20261119) -> None:
        self.base_seed = seed

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self.rng = random.Random(f"{self.base_seed}|{self.episode_id}")
        self.budget = self.rng.randint(5, 30)
        self.seen_docs: list = []
        self.seen_rows: list = []
        self.steps = 0

    def act(self, observation: dict) -> dict:
        if observation.get("kind") == "records":
            for rec in observation["records"]:
                if "doc_id" in rec and "invoice_id" in rec:
                    self.seen_docs.append(rec)
                elif "row_id" in rec:
                    self.seen_rows.append(rec["row_id"])
        self.steps += 1
        if self.steps > self.budget:
            findings = []
            for _ in range(self.rng.randint(0, 2)):
                if not self.seen_docs:
                    break
                doc = self.rng.choice(self.seen_docs)
                evidence = [doc["doc_id"]] + ([self.rng.choice(self.seen_rows)] if self.seen_rows else [])
                findings.append({"type": self.rng.choice(KNOWN_TYPES), "affected_ids": [doc["doc_id"]],
                                 "claimed_amount_minor": doc["total_minor"], "currency": doc["currency"],
                                 "evidence_ids": evidence, "explanation": "random guess"})
            return _submit(self.episode_id, findings)
        choice = self.rng.random()
        if choice < 0.6:
            return {"tool": "list_records", "table": self.rng.choice(S.TABLES), "cursor": None,
                    "limit": self.rng.randint(1, S.MAX_LIST_LIMIT)}
        if choice < 0.85 and self.seen_rows:
            return {"tool": "get_record", "table": "ledger", "id": self.rng.choice(self.seen_rows)}
        a, b = self.rng.randint(1, 10_000), self.rng.randint(1, 10_000)
        return {"tool": "calculator", "expression": f"{a} * {b} / 100"}


class ReconcilePolicy(_TableReader):
    """Simple reconciliation: read invoices and ledger, apply two textbook checks.

    * duplicate_invoice: two invoice documents with identical counterparty,
      date, currency and line items;
    * period_cutoff_error: invoice dated outside the period (absolute instant)
      whose postings are inside the period;
    * partial_payment_allocation (naive): any payment with a row in unapplied cash
      (2100), claiming the 2100 amount - it does not separate legitimate
      overpayments and does not translate currencies.
    """

    name = "reconcile"

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self._queue = ["invoices", "payments", "ledger"]
        self.start = S.parse_timestamp(self.manifest["period_start"])
        self.end = S.parse_timestamp(self.manifest["period_end"])

    def act(self, observation: dict) -> dict:
        nxt = self._read(observation)
        if nxt is not None and observation.get("steps_remaining", 0) > 1:
            return nxt
        return _submit(self.episode_id, self._findings())

    def _findings(self) -> list:
        invoices = [d for d in self.rows["invoices"] if d["total_minor"] > 0]
        out: list = []
        groups: dict = {}
        for d in invoices:
            key = (d["entity_id"], d["issued_at"], d["currency"], d["total_minor"],
                   tuple((l["description"], l["quantity"], l["unit_price_minor"]) for l in d["line_items"]))
            groups.setdefault(key, []).append(d)
        for docs in groups.values():
            if len(docs) == 2:
                out.append({"type": S.ISSUE_DUPLICATE_INVOICE,
                            "affected_ids": sorted(d["doc_id"] for d in docs),
                            "claimed_amount_minor": docs[0]["total_minor"], "currency": docs[0]["currency"],
                            "evidence_ids": sorted(d["doc_id"] for d in docs),
                            "explanation": "Two invoice documents with identical counterparty, date and lines."})
        rows_by_invoice: dict = {}
        for r in self.rows["ledger"]:
            if r["invoice_id"] and r["payment_id"] is None:
                rows_by_invoice.setdefault(r["invoice_id"], []).append(r)
        for d in invoices:
            if self.start <= S.parse_timestamp(d["issued_at"]) <= self.end:
                continue
            rows = [r for r in rows_by_invoice.get(d["invoice_id"], [])
                    if self.start <= S.parse_timestamp(r["timestamp"]) <= self.end]
            if rows:
                out.append({"type": S.ISSUE_PERIOD_CUTOFF, "affected_ids": [d["doc_id"]],
                            "claimed_amount_minor": d["total_minor"], "currency": d["currency"],
                            "evidence_ids": [d["doc_id"], rows[0]["row_id"]],
                            "explanation": "Invoice dated outside the period but posted inside it."})
        by_invoice_id = {d["invoice_id"]: d for d in self.rows["invoices"]}
        payments = {d["payment_id"]: d for d in self.rows["payments"]}
        for r in self.rows["ledger"]:
            if r["account"] != "2100" or r["payment_id"] not in payments:
                continue
            pay = payments[r["payment_id"]]
            invoice = by_invoice_id.get(pay["invoice_refs"][0])
            if invoice is None:
                continue
            out.append({"type": S.ISSUE_PARTIAL_PAYMENT,
                        "affected_ids": sorted([pay["doc_id"], invoice["doc_id"]]),
                        "claimed_amount_minor": r["credit_minor"], "currency": pay["currency"],
                        "evidence_ids": [pay["doc_id"], r["row_id"]],
                        "explanation": "Cash from this payment sits in unapplied cash."})
        return out


class FlagEverything(_TableReader):
    """Attack: accuse every invoice of every error type (capped at 30 findings)."""

    name = "flag_all"

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self._queue = ["invoices"]

    def act(self, observation: dict) -> dict:
        nxt = self._read(observation)
        if nxt is not None:
            return nxt
        findings = []
        for d in self.rows["invoices"]:
            for t in KNOWN_TYPES:
                findings.append({"type": t, "affected_ids": [d["doc_id"]],
                                 "claimed_amount_minor": d["total_minor"], "currency": d["currency"],
                                 "evidence_ids": [d["doc_id"]], "explanation": "everything is wrong"})
        return _submit(self.episode_id, findings)


class OraclePolicy(Policy):
    """Evaluator-side only: submits the reference findings with evidence at step 1."""

    name = "oracle"

    def __init__(self, truth: Sequence[dict]) -> None:
        self._truth = [dict(t) for t in truth]

    def act(self, observation: dict) -> dict:
        findings = [{"type": t["type"], "affected_ids": list(t["affected_ids"]),
                     "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
                     "evidence_ids": list(t["valid_evidence_sets"][0]),
                     "explanation": "reference finding"} for t in self._truth]
        return _submit(self.episode_id, findings)


AGENT_POLICIES = (RandomExplorer, ReconcilePolicy, FlagEverything)
