"""schemas.py - contracts and strict validation for the audit simulator.

Everything that defines a *contract* lives here: constants and limits, the
episode configuration, validators for every public file, the ``submit``
payload parser, CSV/JSONL (de)serialisation helpers and the money/FX helpers.

Rules enforced everywhere:
* money is ``int`` in minor units (``bool`` is rejected, ``float`` is rejected);
* timestamps are ISO-8601 with an explicit UTC offset;
* FX rates are decimal *strings* (``"485.2500"``) handled through ``Decimal``;
* missing references are ``None`` in memory / JSON ``null`` / empty CSV cell.
"""
from __future__ import annotations

import csv
import decimal
import io
import json
import re
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable, Mapping, Optional, Sequence

# --------------------------------------------------------------------------- #
# Versions, limits, vocabularies
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.1.0"

MAX_STEPS = 100
MAX_LIST_LIMIT = 50
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000
STEP_PENALTY = 0.002

TABLES = ("ledger", "invoices", "payments", "fx")

ID_SCHEMES = ("sequential", "randomized")
PRESENTATION_FIELDS = ("id_scheme", "id_salt")

ISSUE_DUPLICATE_INVOICE = "duplicate_invoice"
ISSUE_PERIOD_CUTOFF = "period_cutoff_error"
ISSUE_PARTIAL_PAYMENT = "partial_payment_allocation"
STAGE1_ISSUE_TYPES = (ISSUE_DUPLICATE_INVOICE, ISSUE_PERIOD_CUTOFF)
ALL_ISSUE_TYPES = (ISSUE_DUPLICATE_INVOICE, ISSUE_PERIOD_CUTOFF, ISSUE_PARTIAL_PAYMENT)

# currency -> number of minor-unit digits (explicitly published in manifest.json)
SUPPORTED_CURRENCIES = {"KZT": 2, "USD": 2, "EUR": 2}
ROUNDING_RULE = "ROUND_HALF_UP"

CHART_OF_ACCOUNTS = {
    "1000": "Cash",
    "1100": "Accounts Receivable",
    "2100": "Unapplied Cash Receipts",
    "2300": "Payroll Taxes Payable",
    "3000": "Owner Equity",
    "4000": "Sales Revenue",
    "4100": "Sales Returns",
    "4900": "Foreign Exchange Gain/Loss",
    "5100": "Rent Expense",
    "5200": "Payroll Expense",
    "5300": "Utilities Expense",
    "5400": "Office Expense",
    "5500": "Marketing Expense",
}

LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
)
INVOICE_KEYS = (
    "doc_id", "invoice_id", "entity_id", "issued_at", "due_at",
    "currency", "total_minor", "line_items",
)
LINE_ITEM_KEYS = (
    "line_no", "description", "quantity", "unit_price_minor",
    "amount_minor", "ref_invoice_id",
)
PAYMENT_KEYS = (
    "doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")
MANIFEST_KEYS = (
    "schema_version", "episode_id", "base_currency", "currency_minor_units",
    "period_start", "period_end", "timezone", "counts", "rounding_rule",
)
COUNT_KEYS = ("ledger_rows", "invoices", "payments", "documents", "fx_rates")
POLICY_REQUIRED_KEYS = (
    "fx_rate_date_rule", "period", "rounding_rule", "amount_tolerance_minor",
)
TRUTH_KEYS = (
    "issue_id", "type", "affected_ids", "expected_amount_minor", "currency",
    "valid_evidence_sets",
)
FINDING_KEYS = (
    "type", "affected_ids", "claimed_amount_minor", "currency",
    "evidence_ids", "explanation",
)
SUBMIT_KEYS = ("episode_id", "findings")


class SchemaError(ValueError):
    """Raised whenever data violates a contract defined in this module."""


# --------------------------------------------------------------------------- #
# Primitive validators
# --------------------------------------------------------------------------- #
def is_int(value: Any) -> bool:
    """True for real integers only (``bool`` is *not* an integer here)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _require_keys(obj: Any, keys: Sequence[str], where: str) -> None:
    _require(isinstance(obj, Mapping), f"{where}: must be an object")
    missing = [k for k in keys if k not in obj]
    extra = [k for k in obj if k not in keys]
    _require(not missing, f"{where}: missing keys {missing}")
    _require(not extra, f"{where}: unexpected keys {extra}")


def _require_str(value: Any, where: str, *, non_empty: bool = True) -> str:
    _require(isinstance(value, str), f"{where}: must be a string")
    _require(not non_empty or value != "", f"{where}: must not be empty")
    return value


def _require_opt_str(value: Any, where: str) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, where)


def _require_money(value: Any, where: str) -> int:
    _require(is_int(value), f"{where}: money must be an int in minor units")
    return value


def _require_currency(value: Any, where: str) -> str:
    _require(isinstance(value, str) and value in SUPPORTED_CURRENCIES,
             f"{where}: unsupported currency")
    return value


_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RATE_RE = re.compile(r"^\d{1,6}\.\d{4}$")


def parse_timestamp(value: Any, where: str = "timestamp") -> datetime:
    """Parse an ISO-8601 timestamp that carries an explicit UTC offset."""
    _require(isinstance(value, str) and _TS_RE.match(value) is not None,
             f"{where}: must be ISO-8601 with UTC offset, e.g. 2026-03-15T14:30:00+05:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise SchemaError(f"{where}: not a valid timestamp") from None
    _require(parsed.utcoffset() is not None, f"{where}: UTC offset is required")
    return parsed


def format_timestamp(value: datetime) -> str:
    """Format an aware datetime as ISO-8601 with offset (whole seconds)."""
    _require(value.utcoffset() is not None, "datetime must be timezone-aware")
    return value.replace(microsecond=0).isoformat()


def parse_offset(value: Any) -> timezone:
    """Parse a fixed offset such as ``+05:00`` into a ``timezone``."""
    _require(isinstance(value, str), "timezone: must be a string like +05:00")
    match = _OFFSET_RE.match(value)
    _require(match is not None, "timezone: must look like +05:00")
    sign, hours, minutes = match.group(1), int(match.group(2)), int(match.group(3))
    _require(hours <= 14 and minutes < 60, "timezone: offset out of range")
    delta = timedelta(hours=hours, minutes=minutes)
    return timezone(-delta if sign == "-" else delta)


def parse_rate(value: Any, where: str = "rate") -> Decimal:
    """FX rates are decimal strings with exactly four fractional digits."""
    _require(isinstance(value, str) and _RATE_RE.match(value) is not None,
             f"{where}: must be a decimal string like '485.2500'")
    rate = Decimal(value)
    _require(rate > 0, f"{where}: must be positive")
    return rate


_MONEY_CTX = decimal.Context(prec=60, rounding=ROUND_HALF_UP)


def convert_minor(amount_minor: int, rate: Any, from_digits: int, to_digits: int) -> int:
    """Convert minor units of one currency to minor units of another.

    ``rate`` is a decimal string (or ``Decimal``) quoted as *major target units
    per one major source unit*. Rounding is ROUND_HALF_UP applied once, after
    the exact (60-digit) conversion; no ``float`` is involved.
    """
    _require_money(amount_minor, "amount_minor")
    rate_dec = parse_rate(rate) if isinstance(rate, str) else Decimal(rate)
    scale = _MONEY_CTX.power(Decimal(10), Decimal(to_digits - from_digits))
    exact = _MONEY_CTX.multiply(_MONEY_CTX.multiply(Decimal(amount_minor), rate_dec), scale)
    return int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP, context=_MONEY_CTX))


def fx_record_id(row: Mapping[str, Any]) -> str:
    """Public id of an FX row (fx.csv has no id column)."""
    return f"FX-{row['date'][:10]}-{row['source_currency']}-{row['target_currency']}"


# --------------------------------------------------------------------------- #
# Episode configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EpisodeConfig:
    """Size and shape of one generated episode.

    Hard ranges (from the specification): 200-2000 ledger rows, 20-200
    documents (invoices + payments), 0-12 errors.  Two cross-field
    feasibility rules keep every configuration constructible:
    ``n_documents >= 4 * n_errors`` and ``n_ledger_rows >= 3 * n_documents``.
    """

    n_ledger_rows: int = 600
    n_documents: int = 60
    n_errors: int = 3
    error_types: tuple = ALL_ISSUE_TYPES
    base_currency: str = "KZT"
    foreign_currencies: tuple = ("USD", "EUR")
    timezone: str = "+05:00"
    period_start: str = "2026-01-01"
    period_days: int = 90
    amount_tolerance_minor: int = 100
    include_distractors: bool = True
    # Presentation-only fields: they change ids and row order, never the history.
    id_scheme: str = "sequential"
    id_salt: int = 0

    def __post_init__(self) -> None:
        _require(isinstance(self.id_scheme, str) and self.id_scheme in ID_SCHEMES,
                 f"config.id_scheme: must be one of {list(ID_SCHEMES)}")
        _require(is_int(self.id_salt) and 0 <= self.id_salt < 2 ** 63,
                 "config.id_salt: must be an integer in 0..2**63-1")
        for name in ("n_ledger_rows", "n_documents", "n_errors", "period_days",
                     "amount_tolerance_minor"):
            _require(is_int(getattr(self, name)), f"config.{name}: must be an integer")
        _require(200 <= self.n_ledger_rows <= 2000, "config.n_ledger_rows: must be in 200..2000")
        _require(20 <= self.n_documents <= 200, "config.n_documents: must be in 20..200")
        _require(0 <= self.n_errors <= 12, "config.n_errors: must be in 0..12")
        _require(self.n_documents >= 4 * self.n_errors,
                 "config: n_documents must be >= 4 * n_errors")
        _require(self.n_ledger_rows >= 3 * self.n_documents,
                 "config: n_ledger_rows must be >= 3 * n_documents")
        _require(30 <= self.period_days <= 366, "config.period_days: must be in 30..366")
        _require(0 <= self.amount_tolerance_minor <= 10_000,
                 "config.amount_tolerance_minor: must be in 0..10000")
        _require(isinstance(self.include_distractors, bool),
                 "config.include_distractors: must be a bool")
        for name in ("error_types", "foreign_currencies"):
            value = getattr(self, name)
            _require(isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value),
                     f"config.{name}: must be a list of strings")
            object.__setattr__(self, name, tuple(value))
        _require(len(set(self.error_types)) == len(self.error_types),
                 "config.error_types: must not repeat")
        _require(self.n_errors == 0 or len(self.error_types) > 0,
                 "config.error_types: must not be empty when n_errors > 0")
        _require_currency(self.base_currency, "config.base_currency")
        for cur in self.foreign_currencies:
            _require_currency(cur, "config.foreign_currencies")
        _require(self.base_currency not in self.foreign_currencies,
                 "config.foreign_currencies: must not contain the base currency")
        _require(len(set(self.foreign_currencies)) == len(self.foreign_currencies),
                 "config.foreign_currencies: must not repeat")
        parse_offset(self.timezone)
        _require(isinstance(self.period_start, str) and _DATE_RE.match(self.period_start) is not None,
                 "config.period_start: must be YYYY-MM-DD")
        try:
            date.fromisoformat(self.period_start)
        except ValueError:
            raise SchemaError("config.period_start: not a valid date") from None

    def to_dict(self) -> dict:
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def history_dict(self) -> dict:
        """Fields that shape the company history (presentation fields excluded)."""
        return {k: v for k, v in self.to_dict().items() if k not in PRESENTATION_FIELDS}

    def history_json(self) -> str:
        """Canonical JSON of ``history_dict`` - the RNG material of the history.

        It is byte-identical to the Stage 1 ``canonical_json`` for configs that
        do not use the presentation fields, so old episodes are unchanged.
        """
        return json.dumps(self.history_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeConfig":
        _require(isinstance(data, Mapping), "config: must be an object")
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(str(k) for k in data if k not in allowed)
        _require(not unknown, f"config: unknown keys {unknown}")
        return cls(**dict(data))


def coerce_config(config: Any) -> EpisodeConfig:
    if config is None:
        return EpisodeConfig()
    if isinstance(config, EpisodeConfig):
        return config
    if isinstance(config, Mapping):
        return EpisodeConfig.from_dict(config)
    raise SchemaError("config: must be None, a dict or an EpisodeConfig")


def validate_seed(seed: Any) -> int:
    _require(is_int(seed), "seed: must be an integer")
    return seed


# --------------------------------------------------------------------------- #
# Public file validators
# --------------------------------------------------------------------------- #
def validate_manifest(manifest: Any) -> None:
    _require_keys(manifest, MANIFEST_KEYS, "manifest")
    _require(manifest["schema_version"] == SCHEMA_VERSION, "manifest.schema_version: mismatch")
    _require_str(manifest["episode_id"], "manifest.episode_id")
    base = _require_currency(manifest["base_currency"], "manifest.base_currency")
    units = manifest["currency_minor_units"]
    _require(isinstance(units, Mapping) and base in units, "manifest.currency_minor_units: must include base")
    for cur, digits in units.items():
        _require_currency(cur, "manifest.currency_minor_units")
        _require(is_int(digits) and 0 <= digits <= 6, "manifest.currency_minor_units: bad digits")
    start = parse_timestamp(manifest["period_start"], "manifest.period_start")
    end = parse_timestamp(manifest["period_end"], "manifest.period_end")
    _require(start < end, "manifest: period_start must precede period_end")
    parse_offset(manifest["timezone"])
    _require_keys(manifest["counts"], COUNT_KEYS, "manifest.counts")
    for key in COUNT_KEYS:
        _require(is_int(manifest["counts"][key]) and manifest["counts"][key] >= 0,
                 f"manifest.counts.{key}: must be a non-negative int")
    _require(manifest["counts"]["documents"]
             == manifest["counts"]["invoices"] + manifest["counts"]["payments"],
             "manifest.counts.documents: must equal invoices + payments")
    _require(manifest["rounding_rule"] == ROUNDING_RULE, "manifest.rounding_rule: unsupported")


def validate_policy(policy: Any) -> None:
    _require(isinstance(policy, Mapping), "policy: must be an object")
    for key in POLICY_REQUIRED_KEYS:
        _require(key in policy, f"policy: missing key {key}")
    _require(is_int(policy["amount_tolerance_minor"]) and policy["amount_tolerance_minor"] >= 0,
             "policy.amount_tolerance_minor: must be a non-negative int")
    period = policy["period"]
    _require(isinstance(period, Mapping), "policy.period: must be an object")
    parse_timestamp(period.get("start"), "policy.period.start")
    parse_timestamp(period.get("end"), "policy.period.end")


def validate_ledger_row(row: Any) -> None:
    _require_keys(row, LEDGER_COLUMNS, "ledger row")
    _require_str(row["row_id"], "ledger.row_id")
    parse_timestamp(row["timestamp"], "ledger.timestamp")
    _require_str(row["account"], "ledger.account")
    debit = _require_money(row["debit_minor"], "ledger.debit_minor")
    credit = _require_money(row["credit_minor"], "ledger.credit_minor")
    _require(debit >= 0 and credit >= 0, "ledger: amounts must be non-negative")
    _require((debit > 0) != (credit > 0), "ledger: exactly one of debit/credit must be positive")
    _require_currency(row["currency"], "ledger.currency")
    for key in ("entity_id", "invoice_id", "payment_id"):
        _require_opt_str(row[key], f"ledger.{key}")


def validate_invoice(doc: Any) -> None:
    _require_keys(doc, INVOICE_KEYS, "invoice")
    _require_str(doc["doc_id"], "invoice.doc_id")
    _require_str(doc["invoice_id"], "invoice.invoice_id")
    _require_str(doc["entity_id"], "invoice.entity_id")
    issued = parse_timestamp(doc["issued_at"], "invoice.issued_at")
    due = parse_timestamp(doc["due_at"], "invoice.due_at")
    _require(due >= issued, "invoice: due_at must not precede issued_at")
    _require_currency(doc["currency"], "invoice.currency")
    total = _require_money(doc["total_minor"], "invoice.total_minor")
    lines = doc["line_items"]
    _require(isinstance(lines, list) and lines, "invoice.line_items: must be a non-empty list")
    running = 0
    for idx, line in enumerate(lines, start=1):
        _require_keys(line, LINE_ITEM_KEYS, "invoice line item")
        _require(line["line_no"] == idx and is_int(line["line_no"]), "invoice line_no: must be sequential")
        _require_str(line["description"], "line.description")
        _require(is_int(line["quantity"]), "line.quantity: must be an int")
        unit = _require_money(line["unit_price_minor"], "line.unit_price_minor")
        amount = _require_money(line["amount_minor"], "line.amount_minor")
        _require(amount == line["quantity"] * unit, "line: amount_minor != quantity * unit_price_minor")
        _require_opt_str(line["ref_invoice_id"], "line.ref_invoice_id")
        running += amount
    _require(running == total, "invoice: total_minor != sum of line amounts")


def validate_payment(doc: Any) -> None:
    _require_keys(doc, PAYMENT_KEYS, "payment")
    _require_str(doc["doc_id"], "payment.doc_id")
    _require_str(doc["payment_id"], "payment.payment_id")
    parse_timestamp(doc["timestamp"], "payment.timestamp")
    _require_currency(doc["currency"], "payment.currency")
    _require_money(doc["amount_minor"], "payment.amount_minor")
    refs = doc["invoice_refs"]
    _require(isinstance(refs, list) and refs and all(isinstance(r, str) and r for r in refs),
             "payment.invoice_refs: must be a non-empty list of ids")


def validate_fx_row(row: Any) -> None:
    _require_keys(row, FX_COLUMNS, "fx row")
    parse_timestamp(row["date"], "fx.date")
    _require_currency(row["source_currency"], "fx.source_currency")
    _require_currency(row["target_currency"], "fx.target_currency")
    _require(row["source_currency"] != row["target_currency"], "fx: currencies must differ")
    parse_rate(row["rate"], "fx.rate")


def validate_public(manifest: Mapping, policy: Mapping, ledger: Sequence[Mapping],
                    invoices: Sequence[Mapping], payments: Sequence[Mapping],
                    fx: Sequence[Mapping]) -> None:
    """Validate every public file and all cross-file invariants."""
    validate_manifest(manifest)
    validate_policy(policy)
    for row in ledger:
        validate_ledger_row(row)
    for doc in invoices:
        validate_invoice(doc)
    for doc in payments:
        validate_payment(doc)
    for row in fx:
        validate_fx_row(row)

    counts = manifest["counts"]
    _require(counts["ledger_rows"] == len(ledger), "manifest.counts.ledger_rows mismatch")
    _require(counts["invoices"] == len(invoices), "manifest.counts.invoices mismatch")
    _require(counts["payments"] == len(payments), "manifest.counts.payments mismatch")
    _require(counts["fx_rates"] == len(fx), "manifest.counts.fx_rates mismatch")

    row_ids = [r["row_id"] for r in ledger]
    _require(len(set(row_ids)) == len(row_ids), "ledger.row_id: not unique")
    doc_ids = [d["doc_id"] for d in invoices] + [d["doc_id"] for d in payments]
    _require(len(set(doc_ids)) == len(doc_ids), "doc_id: not unique across invoices/payments")
    pay_ids = [d["payment_id"] for d in payments]
    _require(len(set(pay_ids)) == len(pay_ids), "payment_id: not unique")
    fx_ids = [fx_record_id(r) for r in fx]
    _require(len(set(fx_ids)) == len(fx_ids), "fx rows: duplicate (date, source, target)")

    invoice_ids = {d["invoice_id"] for d in invoices}
    payment_id_set = set(pay_ids)
    for doc in payments:
        for ref in doc["invoice_refs"]:
            _require(ref in invoice_ids, "payment.invoice_refs: dangling reference")
    for row in ledger:
        _require(row["invoice_id"] is None or row["invoice_id"] in invoice_ids,
                 "ledger.invoice_id: dangling reference")
        _require(row["payment_id"] is None or row["payment_id"] in payment_id_set,
                 "ledger.payment_id: dangling reference")
        _require(row["currency"] in manifest["currency_minor_units"],
                 "ledger.currency: missing from manifest.currency_minor_units")

    totals: dict = {}
    for row in ledger:
        debit, credit = totals.get(row["currency"], (0, 0))
        totals[row["currency"]] = (debit + row["debit_minor"], credit + row["credit_minor"])
    for cur, (debit, credit) in totals.items():
        _require(debit == credit, f"ledger: not balanced in {cur}")


def policy_summary(policy: Mapping[str, Any]) -> dict:
    """The short policy view shown in the first observation."""
    period = policy["period"]
    return {
        "period_start": period["start"],
        "period_end": period["end"],
        "period_rule": period.get("rule"),
        "fx_rate_date_rule": policy["fx_rate_date_rule"].get("rule")
        if isinstance(policy["fx_rate_date_rule"], Mapping) else policy["fx_rate_date_rule"],
        "fx_fallback": policy["fx_rate_date_rule"].get("fallback")
        if isinstance(policy["fx_rate_date_rule"], Mapping) else None,
        "rounding_rule": policy["rounding_rule"].get("mode")
        if isinstance(policy["rounding_rule"], Mapping) else policy["rounding_rule"],
        "amount_tolerance_minor": policy["amount_tolerance_minor"],
    }


# --------------------------------------------------------------------------- #
# Truth validation (used by the evaluator)
# --------------------------------------------------------------------------- #
def validate_truth_entry(entry: Any) -> None:
    _require_keys(entry, TRUTH_KEYS, "truth entry")
    _require_str(entry["issue_id"], "truth.issue_id")
    _require_str(entry["type"], "truth.type")
    ids = entry["affected_ids"]
    _require(isinstance(ids, list) and ids and all(isinstance(i, str) and i for i in ids),
             "truth.affected_ids: must be a non-empty list of ids")
    _require_money(entry["expected_amount_minor"], "truth.expected_amount_minor")
    _require_currency(entry["currency"], "truth.currency")
    sets = entry["valid_evidence_sets"]
    _require(isinstance(sets, list) and sets, "truth.valid_evidence_sets: must be a non-empty list")
    for ev in sets:
        _require(isinstance(ev, list) and ev and all(isinstance(i, str) and i for i in ev),
                 "truth.valid_evidence_sets: each set must be a non-empty list of ids")


# --------------------------------------------------------------------------- #
# Submission parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Finding:
    """A validated and canonicalised finding (sorted, de-duplicated id lists)."""

    type: str
    affected_ids: tuple
    claimed_amount_minor: int
    currency: str
    evidence_ids: tuple
    explanation: str


def _id_list(value: Any, where: str, max_items: Optional[int]) -> list:
    _require(isinstance(value, list), f"{where}: must be a list of strings")
    _require(all(isinstance(v, str) for v in value), f"{where}: every item must be a string")
    if max_items is not None:
        _require(len(value) <= max_items, f"{where}: more than {max_items} items")
    return value


def parse_submission(payload: Any, episode_id: str) -> list:
    """Validate a ``submit`` payload and return canonical ``Finding`` objects.

    Raises ``SchemaError`` when the structure is invalid or any limit
    (30 findings, 10 evidence_ids per finding, 1000 explanation chars) is hit.
    """
    _require(isinstance(payload, Mapping), "submit.findings: must be an object")
    _require_keys(payload, SUBMIT_KEYS, "submit.findings")
    _require(payload["episode_id"] == episode_id, "submit.findings.episode_id: mismatch")
    items = payload["findings"]
    _require(isinstance(items, list), "submit.findings.findings: must be a list")
    _require(len(items) <= MAX_FINDINGS, f"submit.findings.findings: more than {MAX_FINDINGS} items")
    out = []
    for idx, item in enumerate(items):
        where = f"findings[{idx}]"
        _require_keys(item, FINDING_KEYS, where)
        ftype = _require_str(item["type"], f"{where}.type")
        affected = _id_list(item["affected_ids"], f"{where}.affected_ids", None)
        amount = _require_money(item["claimed_amount_minor"], f"{where}.claimed_amount_minor")
        currency = _require_str(item["currency"], f"{where}.currency")
        evidence = _id_list(item["evidence_ids"], f"{where}.evidence_ids", MAX_EVIDENCE_IDS)
        explanation = _require_str(item["explanation"], f"{where}.explanation", non_empty=False)
        _require(len(explanation) <= MAX_EXPLANATION_CHARS,
                 f"{where}.explanation: longer than {MAX_EXPLANATION_CHARS} characters")
        out.append(Finding(ftype, tuple(sorted(set(affected))), amount, currency,
                           tuple(sorted(set(evidence))), explanation))
    return out


# --------------------------------------------------------------------------- #
# Serialisation helpers (deterministic text)
# --------------------------------------------------------------------------- #
def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def to_jsonl(rows: Iterable[Mapping]) -> str:
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows]
    return "".join(line + "\n" for line in lines)


def parse_jsonl(text: str) -> list:
    out = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            raise SchemaError(f"jsonl line {number}: invalid JSON") from None
    return out


def _csv_text(columns: Sequence[str], rows: Iterable[Mapping]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row[c] is None else row[c] for c in columns])
    return buf.getvalue()


def ledger_to_csv(rows: Iterable[Mapping]) -> str:
    return _csv_text(LEDGER_COLUMNS, rows)


def fx_to_csv(rows: Iterable[Mapping]) -> str:
    return _csv_text(FX_COLUMNS, rows)


def _read_csv(text: str, columns: Sequence[str], where: str) -> list:
    reader = csv.DictReader(io.StringIO(text))
    _require(reader.fieldnames is not None and list(reader.fieldnames) == list(columns),
             f"{where}: header must be {','.join(columns)}")
    return list(reader)


def parse_ledger_csv(text: str) -> list:
    """Read ledger.csv: money columns -> int, empty reference cells -> None."""
    out = []
    for raw in _read_csv(text, LEDGER_COLUMNS, "ledger.csv"):
        row: dict = dict(raw)
        for key in ("debit_minor", "credit_minor"):
            try:
                row[key] = int(row[key])
            except (TypeError, ValueError):
                raise SchemaError(f"ledger.csv: {key} must be an integer") from None
        for key in ("entity_id", "invoice_id", "payment_id"):
            if row[key] == "":
                row[key] = None
        out.append(row)
    return out


def parse_fx_csv(text: str) -> list:
    return [dict(raw) for raw in _read_csv(text, FX_COLUMNS, "fx.csv")]
