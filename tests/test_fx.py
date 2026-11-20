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
