"""Base-currency ledger: FX translation and settlement, recomputed only from public files."""
from __future__ import annotations

from datetime import date

import pytest

import schemas as S
from generator import generate_episode

FOREIGN = {"n_ledger_rows": 900, "n_documents": 120, "n_errors": 0}


def rate_lookup(ep):
    series: dict = {}
    for row in ep.fx:
        series.setdefault(row["source_currency"], []).append((row["date"][:10], row["rate"]))
    for v in series.values():
        v.sort()

    def rate(currency: str, local: str) -> str:
        best = None
        for d, r in series[currency]:
            if d <= local:
                best = r
        assert best is not None
        return best
    return rate


def local(ts: str, tz: str) -> str:
    return S.parse_timestamp(ts).astimezone(S.parse_offset(tz)).date().isoformat()


def convert(amount, currency, rate, base):
    if currency == base:
        return amount
    return S.convert_minor(amount, rate, S.SUPPORTED_CURRENCIES[currency], S.SUPPORTED_CURRENCIES[base])


@pytest.mark.parametrize("seed", range(6))
def test_ledger_is_in_base_currency_and_translated_by_the_published_rule(seed):
    ep = generate_episode(seed, FOREIGN)
    base, tz = ep.manifest["base_currency"], ep.manifest["timezone"]
    rate = rate_lookup(ep)
    assert {r["currency"] for r in ep.ledger} == {base}
    assert sum(r["debit_minor"] for r in ep.ledger) == sum(r["credit_minor"] for r in ep.ledger)

    invoices = {d["invoice_id"]: d for d in ep.invoices}
    payments = {d["payment_id"]: d for d in ep.payments}
    checked_inv = checked_pay = 0
    for r in ep.ledger:
        if r["account"] == "1100" and r["debit_minor"] > 0 and r["payment_id"] is None:
            doc = invoices[r["invoice_id"]]
            expected = convert(doc["total_minor"], doc["currency"],
                               doc["currency"] != base and rate(doc["currency"], local(r["timestamp"], tz)), base)
            assert r["debit_minor"] == expected
            checked_inv += 1
        if r["account"] == "1000" and r["payment_id"] is not None:
            doc = payments[r["payment_id"]]
            expected = convert(doc["amount_minor"], doc["currency"],
                               doc["currency"] != base and rate(doc["currency"], local(doc["timestamp"], tz)), base)
            assert r["debit_minor"] == expected
            checked_pay += 1
    assert checked_inv > 0 and checked_pay == len(ep.payments)


def test_foreign_documents_cover_weekends_and_fx_differences():
    weekend = fx_rows = foreign = 0
    for seed in range(8):
        ep = generate_episode(seed, FOREIGN)
        base, tz = ep.manifest["base_currency"], ep.manifest["timezone"]
        for d in ep.invoices + ep.payments:
            if d["currency"] != base:
                foreign += 1
                ts = d.get("issued_at") or d["timestamp"]
                if date.fromisoformat(local(ts, tz)).weekday() >= 5:
                    weekend += 1
        fx_rows += sum(1 for r in ep.ledger if r["account"] == "4900")
    assert foreign > 50 and weekend > 0 and fx_rows > 0


@pytest.mark.parametrize("seed", range(6))
def test_settled_invoices_leave_no_residue_in_receivables(seed):
    ep = generate_episode(seed, FOREIGN)
    ar: dict = {}
    for r in ep.ledger:
        if r["account"] == "1100" and r["invoice_id"]:
            ar[r["invoice_id"]] = ar.get(r["invoice_id"], 0) + r["debit_minor"] - r["credit_minor"]
    for d in ep.invoices:
        if d["total_minor"] < 0:
            target = d["line_items"][0]["ref_invoice_id"]
            ar[target] = ar.get(target, 0) + ar.pop(d["invoice_id"], 0)
    # an invoice whose payments (+ credit notes) cover its total has zero receivable balance
    allocated: dict = {}
    for p in ep.payments:
        if len(p["invoice_refs"]) == 1:
            allocated[p["invoice_refs"][0]] = allocated.get(p["invoice_refs"][0], 0) + p["amount_minor"]
    for d in ep.invoices:
        if d["total_minor"] <= 0:
            continue
        returned = -sum(x["total_minor"] for x in ep.invoices
                        if x["total_minor"] < 0 and x["line_items"][0]["ref_invoice_id"] == d["invoice_id"])
        if allocated.get(d["invoice_id"], 0) + returned == d["total_minor"]:
            assert ar.get(d["invoice_id"], 0) == 0, d["invoice_id"]


# --------------------------------------------------------------------------- #
# fx_rate_error and interacting-error scenarios
# --------------------------------------------------------------------------- #
from test_core import _truth_is_semantically_true  # noqa: E402
from evaluator import Evaluator  # noqa: E402

INTERACT = {"n_ledger_rows": 900, "n_documents": 120, "n_errors": 4,
            "interactions": ["partial_fx_boundary", "cutoff_fx"]}


def _oracle_reward(ep) -> float:
    ev = Evaluator(ep.manifest["episode_id"], ep.truth, ep.policy["amount_tolerance_minor"])
    findings = [{"type": t["type"], "affected_ids": t["affected_ids"],
                 "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
                 "evidence_ids": t["valid_evidence_sets"][0], "explanation": "oracle"} for t in ep.truth]
    ev.finalize({"episode_id": ep.manifest["episode_id"], "findings": findings}, 0)
    return ev.final_result()["reward"]


@pytest.mark.parametrize("seed", range(12))
def test_interacting_scenarios_are_two_linked_provable_issues(seed):
    ep = generate_episode(seed, INTERACT)
    _truth_is_semantically_true(ep)
    scenarios = {}
    for m in ep.provenance["mutations"]:
        if "scenario" in m:
            scenarios.setdefault(m["scenario"], []).append(m)
    assert set(scenarios) == {"partial_fx_boundary", "cutoff_fx"}
    truth = {t["issue_id"]: t for t in ep.truth}
    for name, muts in scenarios.items():
        assert len(muts) == 2
        a, b = (truth[m["issue_id"]] for m in muts)
        assert set(a["affected_ids"]) & set(b["affected_ids"]), name     # same records, two errors
        assert {a["type"], b["type"]} == ({"partial_payment_allocation", "fx_rate_error"}
                                          if name == "partial_fx_boundary"
                                          else {"period_cutoff_error", "fx_rate_error"})
    # the boundary payment is on the last local day of the period, in a foreign currency
    pay_doc = next(t for t in ep.truth if t["type"] == "fx_rate_error"
                   and any(t["affected_ids"][0] == p["doc_id"] for p in ep.payments))
    pay = next(p for p in ep.payments if p["doc_id"] == pay_doc["affected_ids"][0])
    assert pay["currency"] != ep.manifest["base_currency"]
    assert local(pay["timestamp"], ep.manifest["timezone"]) == local(ep.manifest["period_end"], ep.manifest["timezone"])
    assert _oracle_reward(ep) == 1.0


@pytest.mark.parametrize("seed", range(10))
def test_fx_rate_error_alone_is_provable_and_deterministic(seed):
    cfg = {"n_ledger_rows": 600, "n_documents": 60, "n_errors": 3, "error_types": ["fx_rate_error"]}
    ep = generate_episode(seed, cfg)
    assert [t["type"] for t in ep.truth] == ["fx_rate_error"] * 3
    _truth_is_semantically_true(ep)
    assert ep.public_files() == generate_episode(seed, cfg).public_files()
    assert _oracle_reward(ep) == 1.0


def test_interaction_config_validation():
    with pytest.raises(S.SchemaError):
        S.coerce_config({"interactions": ["unknown"]})
    with pytest.raises(S.SchemaError):
        S.coerce_config({"interactions": ["cutoff_fx"], "foreign_currencies": []})
    with pytest.raises(S.SchemaError):
        S.coerce_config({"n_errors": 10, "interactions": ["cutoff_fx", "partial_fx_boundary"],
                         "n_documents": 200, "n_ledger_rows": 2000})
