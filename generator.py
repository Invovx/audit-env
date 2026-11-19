"""generator.py - deterministic episode generator (seed + config).

Pipeline (each stage uses its own forked RNG stream, so adding a stage later
does not perturb the others):

1. plan      - decide which errors will be injected;
2. history   - build a fully consistent, balanced company history
               (invoices, credit notes, split/combined payments, recurring
               billing, boundary cases, filler journals, FX table);
3. verify    - prove the clean books are consistent (balance, AR tie-out);
4. mutate    - apply each error through a ``Mutator`` that changes the clean
               history in one causal way and records the change;
5. finalize  - derive the ledger from the records, assign ids through an
               ``IdScheme`` and emit public files, ``truth`` and ``provenance``.

Extension points for Stage 2:
* new error types   -> subclass ``Mutator`` and add it to ``MUTATORS``;
* interacting errors-> a ``Mutator`` may target records already used by
                       another one (``Books.used``) and return several ``Issue``;
* row shuffling / id renaming -> subclass ``IdScheme`` (truth is resolved
                       *after* ids are assigned, via object references, so
                       semantics survive any renaming or reordering).
"""
from __future__ import annotations

import bisect
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import schemas as S

UTC = timezone.utc

ACC_CASH = "1000"
ACC_AR = "1100"
ACC_PAYROLL_TAX = "2300"
ACC_EQUITY = "3000"
ACC_REVENUE = "4000"
ACC_RETURNS = "4100"
ACC_PAYROLL = "5200"
EXPENSE_ACCOUNTS = ("5100", "5300", "5400", "5500")

# reference rates, in units of 1e-4 KZT per one major unit
_REFERENCE_RATES_E4 = {"KZT": 10_000, "USD": 4_852_500, "EUR": 5_284_000}

# (description, unit price in minor units per currency)
CATALOG = (
    ("Consulting services (hour)", {"KZT": 2_500_000, "USD": 5_500, "EUR": 5_000}),
    ("Software license (month)", {"KZT": 12_000_000, "USD": 25_000, "EUR": 23_000}),
    ("Cloud hosting (month)", {"KZT": 8_000_000, "USD": 16_500, "EUR": 15_000}),
    ("Data migration package", {"KZT": 45_000_000, "USD": 95_000, "EUR": 88_000}),
    ("Training workshop (day)", {"KZT": 30_000_000, "USD": 64_000, "EUR": 59_000}),
    ("Support plan (quarter)", {"KZT": 20_000_000, "USD": 42_000, "EUR": 39_000}),
    ("Integration module", {"KZT": 75_000_000, "USD": 160_000, "EUR": 145_000}),
    ("Reporting add-on", {"KZT": 6_000_000, "USD": 13_000, "EUR": 12_000}),
)


class GeneratorError(RuntimeError):
    """Internal consistency failure or an infeasible configuration."""


# --------------------------------------------------------------------------- #
# Deterministic RNG (independent of the Python version's `random` module)
# --------------------------------------------------------------------------- #
class DetRng:
    """SplitMix64 stream seeded from a text material; ``fork`` gives substreams."""

    _MASK = (1 << 64) - 1

    def __init__(self, material: str) -> None:
        self._material = material
        self._state = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")

    def fork(self, label: str) -> "DetRng":
        return DetRng(f"{self._material}/{label}")

    def _next(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & self._MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
        return z ^ (z >> 31)

    def randint(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("empty range")
        span = high - low + 1
        limit = (1 << 64) - ((1 << 64) % span)
        while True:
            x = self._next()
            if x < limit:
                return low + x % span

    def chance(self, percent: int) -> bool:
        return self.randint(1, 100) <= percent

    def choice(self, items: Sequence):
        return items[self.randint(0, len(items) - 1)]

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]

    def sample(self, items: Sequence, k: int) -> list:
        pool = list(items)
        self.shuffle(pool)
        return pool[:k]


# --------------------------------------------------------------------------- #
# Records (mutable in-memory history; ids are assigned only at finalize time)
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Party:
    entity_id: str
    currency: str


@dataclass(eq=False)
class LineItem:
    description: str
    quantity: int
    unit_price_minor: int
    ref: Optional["InvoiceRec"] = None

    @property
    def amount_minor(self) -> int:
        return self.quantity * self.unit_price_minor


@dataclass(eq=False)
class InvoiceRec:
    kind: str                      # "invoice" | "credit_note"
    entity_id: str
    currency: str
    issued_at: datetime
    due_at: datetime
    lines: list
    booked_at: Optional[datetime]  # None => not posted to this period's ledger
    display_offset: timezone
    story: int
    order: int
    tags: list = field(default_factory=list)
    alias_of: Optional["InvoiceRec"] = None   # re-uses the invoice_id of another record
    invoice_id: str = ""
    doc_id: str = ""

    @property
    def total_minor(self) -> int:
        return sum(line.amount_minor for line in self.lines)


@dataclass(eq=False)
class PaymentRec:
    ts: datetime
    currency: str
    entity_id: str
    allocations: list              # [(InvoiceRec, amount_minor)]
    story: int
    order: int
    payment_id: str = ""
    doc_id: str = ""

    @property
    def amount_minor(self) -> int:
        return sum(amount for _, amount in self.allocations)


@dataclass(eq=False)
class JournalRec:
    ts: datetime
    currency: str
    legs: list                     # [(account, debit, credit, entity_id|None)]
    label: str
    order: int


@dataclass(eq=False)
class LedgerRow:
    ts: datetime
    offset: timezone
    account: str
    debit: int
    credit: int
    currency: str
    entity_id: Optional[str]
    invoice: Optional[InvoiceRec]
    payment: Optional[PaymentRec]
    owner: Any
    role: str
    seq: int
    row_id: str = ""


@dataclass(frozen=True)
class LedgerRef:
    """Symbolic reference to a ledger row (resolved after ids are assigned)."""

    owner: Any
    role: str


@dataclass
class Issue:
    """One injected error, expressed with object references."""

    type: str
    affected: list
    amount_minor: int
    currency: str
    evidence_sets: list
    mutation: dict


@dataclass(frozen=True)
class PeriodCtx:
    tz: timezone
    start: datetime
    end: datetime
    days: int
    base: str

    @classmethod
    def from_config(cls, cfg: S.EpisodeConfig) -> "PeriodCtx":
        tz = S.parse_offset(cfg.timezone)
        d = date.fromisoformat(cfg.period_start)
        start = datetime(d.year, d.month, d.day, tzinfo=tz)
        end = start + timedelta(days=cfg.period_days) - timedelta(seconds=1)
        return cls(tz, start, end, cfg.period_days, cfg.base_currency)


@dataclass(eq=False)
class Books:
    config: S.EpisodeConfig
    ctx: PeriodCtx
    customers: list = field(default_factory=list)
    suppliers: list = field(default_factory=list)
    invoices: list = field(default_factory=list)
    payments: list = field(default_factory=list)
    journals: list = field(default_factory=list)
    fx: list = field(default_factory=list)
    story_log: list = field(default_factory=list)
    used: set = field(default_factory=set)
    _order: int = 0
    _story: int = 0

    def next_order(self) -> int:
        self._order += 1
        return self._order

    def new_story(self) -> int:
        self._story += 1
        return self._story

    def log_story(self, story: int, kind: str, spec: "StorySpec", records: list) -> None:
        self.story_log.append({"story": story, "kind": kind, "tags": list(spec.tags),
                               "records": records})


# --------------------------------------------------------------------------- #
# Story building (clean history)
# --------------------------------------------------------------------------- #
@dataclass
class StorySpec:
    kind: str
    tags: tuple = ()
    window: Optional[str] = None   # "head" | "tail" | "not_late" | None
    max_docs: Optional[int] = None


def _tod(rng: DetRng) -> int:
    return rng.randint(9 * 3600, 18 * 3600 - 1)


def _pick_day(ctx: PeriodCtx, rng: DetRng, window: Optional[str], hi: Optional[int] = None) -> int:
    last = ctx.days - 1
    hi = last if hi is None else max(0, min(hi, last))
    if window == "head":
        return rng.randint(0, min(4, hi))
    if window == "tail":
        return rng.randint(max(0, last - 4), last)
    if window == "not_late":
        return rng.randint(0, max(0, min(hi, last - 2)))
    return rng.randint(0, hi)


def _make_lines(rng: DetRng, currency: str) -> list:
    count = rng.choice((1, 1, 2, 3))
    return [LineItem(desc, rng.randint(1, 12), prices[currency])
            for desc, prices in rng.sample(CATALOG, count)]


def _copy_lines(lines: list) -> list:
    return [LineItem(l.description, l.quantity, l.unit_price_minor, None) for l in lines]


def _make_invoice(books: Books, rng: DetRng, party: Party, issued: datetime, lines: list,
                  story: int, spec: StorySpec, *, kind: str = "invoice", booked: bool = True,
                  offset: Optional[timezone] = None, extra_tags: Sequence[str] = (),
                  terms_days: Optional[int] = None) -> InvoiceRec:
    terms = terms_days if terms_days is not None else rng.choice((14, 30, 45))
    due = issued if kind == "credit_note" else issued + timedelta(days=terms)
    inv = InvoiceRec(kind=kind, entity_id=party.entity_id, currency=party.currency,
                     issued_at=issued, due_at=due, lines=lines,
                     booked_at=issued if booked else None,
                     display_offset=offset or books.ctx.tz, story=story,
                     order=books.next_order(), tags=list(spec.tags) + list(extra_tags))
    books.invoices.append(inv)
    return inv


def _make_payment(books: Books, party: Party, ts: datetime, allocations: list,
                  story: int) -> PaymentRec:
    pay = PaymentRec(ts=ts, currency=party.currency, entity_id=party.entity_id,
                     allocations=allocations, story=story, order=books.next_order())
    books.payments.append(pay)
    return pay


def _split_amount(rng: DetRng, total: int, parts: int) -> list:
    weights = [rng.randint(1, 5) for _ in range(parts)]
    amounts = [total * w // sum(weights) for w in weights[:-1]]
    amounts.append(total - sum(amounts))
    return amounts


def _build_open(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    ctx = books.ctx
    story = books.new_story()
    party = rng.choice(books.customers)
    day = _pick_day(ctx, rng, spec.window)
    issued = ctx.start + timedelta(days=day, seconds=_tod(rng))
    inv = _make_invoice(books, rng, party, issued, _make_lines(rng, party.currency), story, spec)
    books.log_story(story, "open", spec, [inv])
    return 1


def _build_paid(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    if max_docs < 2:
        return _build_open(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = rng.choice(books.customers)
    lag = rng.randint(3, min(40, ctx.days - 2))
    day = _pick_day(ctx, rng, spec.window, hi=ctx.days - 1 - lag)
    issued = ctx.start + timedelta(days=day, seconds=_tod(rng))
    inv = _make_invoice(books, rng, party, issued, _make_lines(rng, party.currency), story, spec)
    ts = ctx.start + timedelta(days=day + lag, seconds=_tod(rng))
    pay = _make_payment(books, party, ts, [(inv, inv.total_minor)], story)
    books.log_story(story, "paid", spec, [inv, pay])
    return 2


def _build_split(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    """Legitimate split payment: one invoice settled by 2-3 instalments."""
    if max_docs < 2:
        return _build_open(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = rng.choice(books.customers)
    parts = rng.choice((2, 3))
    lags = [rng.randint(3, 12) for _ in range(parts)]
    keep = min(parts, max_docs - 1)
    while keep > 1 and sum(lags[:keep]) > ctx.days - 2:
        keep -= 1
    cumulative = [sum(lags[:i + 1]) for i in range(keep)]
    day = _pick_day(ctx, rng, spec.window, hi=ctx.days - 1 - cumulative[-1])
    issued = ctx.start + timedelta(days=day, seconds=_tod(rng))
    inv = _make_invoice(books, rng, party, issued, _make_lines(rng, party.currency), story, spec,
                        extra_tags=("split_payment",))
    amounts = _split_amount(rng, inv.total_minor, parts)
    records: list = [inv]
    for i in range(keep):
        ts = ctx.start + timedelta(days=day + cumulative[i], seconds=_tod(rng))
        records.append(_make_payment(books, party, ts, [(inv, amounts[i])], story))
    books.log_story(story, "split", spec, records)
    return 1 + keep


def _build_return(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    """Legitimate return: invoice, partial credit note, payment of the net amount."""
    if max_docs < 3:
        return _build_paid(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = rng.choice(books.customers)
    lag_cn = rng.randint(2, 12)
    lag_pay = rng.randint(3, 15)
    day = _pick_day(ctx, rng, spec.window, hi=ctx.days - 1 - lag_cn - lag_pay)
    issued = ctx.start + timedelta(days=day, seconds=_tod(rng))
    inv = _make_invoice(books, rng, party, issued, _make_lines(rng, party.currency), story, spec,
                        extra_tags=("has_return",))
    returned = max(1, inv.total_minor * rng.randint(10, 40) // 100)
    cn_ts = ctx.start + timedelta(days=day + lag_cn, seconds=_tod(rng))
    note = _make_invoice(books, rng, party, cn_ts,
                         [LineItem("Credit note: returned goods/services", 1, -returned, inv)],
                         story, spec, kind="credit_note", extra_tags=("credit_note",))
    pay_ts = ctx.start + timedelta(days=day + lag_cn + lag_pay, seconds=_tod(rng))
    pay = _make_payment(books, party, pay_ts, [(inv, inv.total_minor - returned)], story)
    books.log_story(story, "return", spec, [inv, note, pay])
    return 3


def _build_combined(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    """Legitimate combined payment: one payment settling two invoices."""
    if max_docs < 3:
        return _build_paid(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = rng.choice(books.customers)
    gap = rng.randint(0, 10)
    lag = rng.randint(3, 15)
    day1 = _pick_day(ctx, rng, spec.window, hi=ctx.days - 1 - gap - lag)
    day2 = day1 + gap
    inv1 = _make_invoice(books, rng, party, ctx.start + timedelta(days=day1, seconds=_tod(rng)),
                         _make_lines(rng, party.currency), story, spec, extra_tags=("combined_payment",))
    inv2 = _make_invoice(books, rng, party, ctx.start + timedelta(days=day2, seconds=_tod(rng)),
                         _make_lines(rng, party.currency), story, spec, extra_tags=("combined_payment",))
    ts = ctx.start + timedelta(days=day2 + lag, seconds=_tod(rng))
    pay = _make_payment(books, party, ts,
                        [(inv1, inv1.total_minor), (inv2, inv2.total_minor)], story)
    books.log_story(story, "combined", spec, [inv1, inv2, pay])
    return 3


def _build_recurring(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    """Legitimate repeated amounts: the same customer billed the same amount 3 times."""
    if max_docs < 3:
        return _build_paid(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = rng.choice(books.customers)
    template = _make_lines(rng, party.currency)
    spacing = rng.randint(7, min(30, (ctx.days - 4) // 2))
    day0 = rng.randint(0, ctx.days - 1 - 2 * spacing)
    tod = _tod(rng)
    docs = 0
    records: list = []
    invoices = []
    for i in range(3):
        issued = ctx.start + timedelta(days=day0 + i * spacing, seconds=tod)
        inv = _make_invoice(books, rng, party, issued, _copy_lines(template), story, spec,
                            extra_tags=("recurring",), terms_days=30)
        invoices.append((inv, day0 + i * spacing))
        records.append(inv)
        docs += 1
    for inv, day in invoices:
        if docs < max_docs and rng.chance(70):
            lag = rng.randint(3, 20)
            if day + lag <= ctx.days - 1:
                ts = ctx.start + timedelta(days=day + lag, seconds=_tod(rng))
                records.append(_make_payment(books, party, ts, [(inv, inv.total_minor)], story))
                docs += 1
    books.log_story(story, "recurring", spec, records)
    return docs


def _special_builder(kind: str) -> Callable[[Books, DetRng, StorySpec, int], int]:
    """Single-document distractors that look suspicious but are legitimate."""

    def build(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
        ctx = books.ctx
        story = books.new_story()
        party = rng.choice(books.customers)
        offset: Optional[timezone] = None
        booked = True
        if kind == "boundary_start":
            issued = ctx.start
        elif kind == "boundary_end":
            issued = ctx.end
        elif kind == "tz_head_utc":
            # inside the period, but printed in UTC it looks like the previous year/month
            issued = ctx.start + timedelta(seconds=rng.randint(600, 4 * 3600))
            offset = UTC
        elif kind == "out_of_next":
            issued = ctx.end + timedelta(seconds=rng.randint(3600, 5 * 86400))
            booked = False
            offset = UTC if rng.chance(50) else None
        else:  # out_of_prev
            issued = ctx.start - timedelta(seconds=rng.randint(3600, 5 * 86400))
            booked = False
            offset = UTC if rng.chance(50) else None
        inv = _make_invoice(books, rng, party, issued, _make_lines(rng, party.currency), story,
                            spec, booked=booked, offset=offset,
                            extra_tags=(f"distractor:{kind}",))
        books.log_story(story, kind, spec, [inv])
        return 1

    return build


STORY_BUILDERS: dict = {
    "open": _build_open,
    "paid": _build_paid,
    "split": _build_split,
    "return": _build_return,
    "combined": _build_combined,
    "recurring": _build_recurring,
    "boundary_start": _special_builder("boundary_start"),
    "boundary_end": _special_builder("boundary_end"),
    "tz_head_utc": _special_builder("tz_head_utc"),
    "out_of_next": _special_builder("out_of_next"),
    "out_of_prev": _special_builder("out_of_prev"),
}
STORY_MIN_DOCS = {"open": 1, "paid": 2, "split": 2, "return": 3, "combined": 3, "recurring": 3,
                  "boundary_start": 1, "boundary_end": 1, "tz_head_utc": 1,
                  "out_of_next": 1, "out_of_prev": 1}
STORY_MAX_DOCS = {"open": 1, "paid": 2, "split": 4, "return": 3, "combined": 3, "recurring": 6,
                  "boundary_start": 1, "boundary_end": 1, "tz_head_utc": 1,
                  "out_of_next": 1, "out_of_prev": 1}
FORCED_DISTRACTORS = ("recurring", "split", "return", "combined", "boundary_start",
                      "boundary_end", "tz_head_utc", "out_of_next", "out_of_prev")
REGULAR_WEIGHTS = (("open", 25), ("paid", 40), ("split", 15), ("return", 8),
                   ("combined", 6), ("recurring", 6))


def _build_parties(books: Books, rng: DetRng) -> None:
    cfg = books.config
    count = min(40, max(5, cfg.n_documents // 4))
    for i in range(1, count + 1):
        if cfg.foreign_currencies and rng.chance(40):
            currency = rng.choice(sorted(cfg.foreign_currencies))
        else:
            currency = cfg.base_currency
        books.customers.append(Party(f"CUST-{i:03d}", currency))
    books.suppliers = [f"SUP-{i:03d}" for i in range(1, 7)]


def _run_story(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    used = STORY_BUILDERS[spec.kind](books, rng, spec, max_docs)
    if not 1 <= used <= max_docs:
        raise GeneratorError(f"story '{spec.kind}' used {used} documents (budget {max_docs})")
    return used


def _build_clean_stories(books: Books, rng: DetRng, counts: dict) -> None:
    cfg = books.config
    extra_docs = sum(MUTATORS[name].extra_documents(c) for name, c in counts.items())
    initial = cfg.n_documents - extra_docs
    remaining = initial

    reserved: list = []
    for name in sorted(counts):
        reserved.extend(MUTATORS[name].reserved_specs(counts[name], rng.fork(f"reserve-{name}")))
    rng.shuffle(reserved)
    if len(reserved) > remaining:
        raise GeneratorError("document budget too small for the requested errors")
    for index, spec in enumerate(reserved):
        later = len(reserved) - index - 1
        cap = min(spec.max_docs or STORY_MAX_DOCS[spec.kind], remaining - later)
        remaining -= _run_story(books, rng, spec, cap)

    if books.config.include_distractors:
        kinds = list(FORCED_DISTRACTORS)
        rng.shuffle(kinds)
        budget = initial * 60 // 100
        for kind in kinds:
            if not rng.chance(85):
                continue
            room = min(remaining, budget)
            if room < STORY_MIN_DOCS[kind]:
                continue
            used = _run_story(books, rng, StorySpec(kind), min(STORY_MAX_DOCS[kind], room))
            remaining -= used
            budget -= used

    table = [name for name, weight in REGULAR_WEIGHTS for _ in range(weight)]
    while remaining > 0:
        eligible = [k for k in table if STORY_MIN_DOCS[k] <= remaining]
        kind = rng.choice(eligible)
        remaining -= _run_story(books, rng, StorySpec(kind), min(STORY_MAX_DOCS[kind], remaining))
    if remaining != 0:
        raise GeneratorError("document budget accounting error")


# --------------------------------------------------------------------------- #
# FX table
# --------------------------------------------------------------------------- #
def _build_fx(books: Books, rng: DetRng) -> None:
    ctx, cfg = books.ctx, books.config
    rows = []
    for cur in sorted(cfg.foreign_currencies):
        cur_rng = rng.fork(cur)
        rate = _REFERENCE_RATES_E4[cur] * 10_000 // _REFERENCE_RATES_E4[ctx.base]
        for offset in range(-7, ctx.days + 7):
            day = ctx.start + timedelta(days=offset)
            if day.weekday() >= 5:      # no rates on weekends -> fallback rule is exercised
                continue
            rate = max(1, rate + rate * cur_rng.randint(-30, 30) // 10_000)
            rows.append({"date": S.format_timestamp(day), "source_currency": cur,
                         "target_currency": ctx.base,
                         "rate": f"{rate // 10_000}.{rate % 10_000:04d}"})
    rows.sort(key=lambda r: (r["date"], r["source_currency"]))
    books.fx = rows


class FxLookup:
    """Rate lookup implementing the published rule (local date, else latest earlier)."""

    def __init__(self, rows: Sequence[dict]) -> None:
        grouped: dict = {}
        for row in rows:
            grouped.setdefault((row["source_currency"], row["target_currency"]), []).append(
                (row["date"][:10], row["rate"]))
        self._dates = {k: [d for d, _ in sorted(v)] for k, v in grouped.items()}
        self._rates = {k: [r for _, r in sorted(v)] for k, v in grouped.items()}

    def rate(self, source: str, target: str, local_date: str) -> str:
        key = (source, target)
        if key not in self._dates:
            raise GeneratorError(f"no FX series {source}->{target}")
        idx = bisect.bisect_right(self._dates[key], local_date) - 1
        if idx < 0:
            raise GeneratorError("no FX rate on or before the requested date")
        return self._rates[key][idx]


# --------------------------------------------------------------------------- #
# Ledger derivation
# --------------------------------------------------------------------------- #
def build_ledger(books: Books) -> tuple:
    """Derive ledger rows from the records.  Returns (rows sorted by time, index)."""
    ctx = books.ctx
    rows: list = []

    def add(ts, offset, account, debit, credit, currency, entity, invoice, payment, owner, role):
        rows.append(LedgerRow(ts, offset, account, debit, credit, currency, entity, invoice,
                              payment, owner, role, len(rows)))

    for jr in books.journals:
        for account, debit, credit, entity in jr.legs:
            add(jr.ts, ctx.tz, account, debit, credit, jr.currency, entity, None, None, jr, "journal")
    for inv in books.invoices:
        if inv.booked_at is None:
            continue
        amount = inv.total_minor
        if inv.kind == "invoice":
            add(inv.booked_at, inv.display_offset, ACC_AR, amount, 0, inv.currency,
                inv.entity_id, inv, None, inv, "ar")
            add(inv.booked_at, inv.display_offset, ACC_REVENUE, 0, amount, inv.currency,
                inv.entity_id, inv, None, inv, "revenue")
        else:
            add(inv.booked_at, inv.display_offset, ACC_RETURNS, -amount, 0, inv.currency,
                inv.entity_id, inv, None, inv, "contra")
            add(inv.booked_at, inv.display_offset, ACC_AR, 0, -amount, inv.currency,
                inv.entity_id, inv, None, inv, "ar")
    for pay in books.payments:
        add(pay.ts, ctx.tz, ACC_CASH, pay.amount_minor, 0, pay.currency, pay.entity_id,
            None, pay, pay, "cash")
        for position, (inv, amount) in enumerate(pay.allocations):
            add(pay.ts, ctx.tz, ACC_AR, 0, amount, pay.currency, pay.entity_id, inv, pay,
                pay, f"ar:{position}")
    rows.sort(key=lambda r: (r.ts, r.seq))
    index = {(id(r.owner), r.role): r for r in rows}
    return rows, index


def _build_filler(books: Books, rng: DetRng, counts: dict) -> None:
    """Balanced operating journals so that the ledger has exactly n_ledger_rows rows."""
    cfg, ctx = books.config, books.ctx
    doc_rows = len(build_ledger(books)[0])
    extra = sum(MUTATORS[name].extra_ledger_rows(c) for name, c in counts.items())
    free = cfg.n_ledger_rows - doc_rows - extra
    if free < 0 or free == 1 or free == 3:
        raise GeneratorError(f"cannot fill ledger: {free} filler rows required")
    if free == 0:
        return
    remaining = free - 2          # two rows are reserved for the opening capital entry
    cash_out = 0
    while remaining > 0:
        ts = ctx.start + timedelta(days=rng.randint(0, ctx.days - 1), seconds=_tod(rng))
        if remaining >= 3 and (remaining == 3 or (remaining >= 5 and rng.chance(25))):
            gross = rng.randint(2_000_000, 20_000_000)
            tax = gross * 10 // 100
            net = gross - tax
            legs = [(ACC_PAYROLL, gross, 0, None), (ACC_CASH, 0, net, None),
                    (ACC_PAYROLL_TAX, 0, tax, None)]
            label, cash_out, remaining = "payroll", cash_out + net, remaining - 3
        else:
            amount = rng.randint(50_000, 5_000_000)
            supplier = rng.choice(books.suppliers)
            legs = [(rng.choice(EXPENSE_ACCOUNTS), amount, 0, supplier),
                    (ACC_CASH, 0, amount, supplier)]
            label, cash_out, remaining = "operating_expense", cash_out + amount, remaining - 2
        books.journals.append(JournalRec(ts, ctx.base, legs, label, books.next_order()))
    capital = cash_out + rng.randint(1_000_000, 10_000_000)
    books.journals.append(JournalRec(
        ctx.start + timedelta(minutes=30), ctx.base,
        [(ACC_CASH, capital, 0, None), (ACC_EQUITY, 0, capital, None)],
        "opening_capital", books.next_order()))


def _verify_clean(books: Books) -> dict:
    """Prove the clean history is consistent before any error is injected."""
    ctx = books.ctx
    rows, _ = build_ledger(books)
    balance: dict = {}
    ar_ledger: dict = {}
    for r in rows:
        d, c = balance.get(r.currency, (0, 0))
        balance[r.currency] = (d + r.debit, c + r.credit)
        if r.account == ACC_AR:
            ar_ledger[r.currency] = ar_ledger.get(r.currency, 0) + r.debit - r.credit
    for cur, (d, c) in balance.items():
        if d != c:
            raise GeneratorError(f"clean ledger not balanced in {cur}")

    open_by_inv: dict = {}
    ar_documents: dict = {}
    for inv in books.invoices:
        if inv.booked_at is None:
            continue
        open_by_inv[id(inv)] = inv.total_minor if inv.kind == "invoice" else 0
        ar_documents[inv.currency] = ar_documents.get(inv.currency, 0) + inv.total_minor
        if inv.kind == "credit_note":
            target = inv.lines[0].ref
            open_by_inv[id(target)] += inv.total_minor
    for pay in books.payments:
        for inv, amount in pay.allocations:
            open_by_inv[id(inv)] -= amount
            ar_documents[pay.currency] = ar_documents.get(pay.currency, 0) - amount
    if any(v < 0 for v in open_by_inv.values()):
        raise GeneratorError("an invoice is over-settled")
    for cur in set(ar_ledger) | set(ar_documents):
        if ar_ledger.get(cur, 0) != ar_documents.get(cur, 0):
            raise GeneratorError(f"AR sub-ledger does not tie out in {cur}")

    lookup = FxLookup(books.fx)
    digits = S.SUPPORTED_CURRENCIES
    revenue_base = 0
    for inv in books.invoices:
        if inv.booked_at is None or inv.kind != "invoice":
            continue
        if inv.currency == ctx.base:
            revenue_base += inv.total_minor
        else:
            local = inv.booked_at.astimezone(ctx.tz).date().isoformat()
            rate = lookup.rate(inv.currency, ctx.base, local)
            revenue_base += S.convert_minor(inv.total_minor, rate, digits[inv.currency],
                                            digits[ctx.base])
    return {
        "balanced_by_currency": {c: {"debit_minor": d, "credit_minor": cr}
                                 for c, (d, cr) in sorted(balance.items())},
        "ar_open_minor_by_currency": dict(sorted(ar_documents.items())),
        "booked_revenue_in_base_minor": revenue_base,
        "ledger_rows_before_errors": len(rows),
        "documents_before_errors": len(books.invoices) + len(books.payments),
    }


# --------------------------------------------------------------------------- #
# Mutators (error injection)
# --------------------------------------------------------------------------- #
class Mutator(ABC):
    """One error type.  ``apply`` changes the clean history causally and returns an Issue."""

    name: str

    @abstractmethod
    def reserved_specs(self, count: int, rng: DetRng) -> list:
        """Ordinary-looking stories that must exist so ``count`` mutations are possible."""

    def extra_documents(self, count: int) -> int:
        return 0

    def extra_ledger_rows(self, count: int) -> int:
        return 0

    @abstractmethod
    def apply(self, books: Books, rng: DetRng, ordinal: int) -> Issue:
        ...


class DuplicateInvoice(Mutator):
    """The same invoice is entered twice: extra document + extra (balanced) postings."""

    name = S.ISSUE_DUPLICATE_INVOICE
    TAG = "reserved:duplicate_invoice"

    def reserved_specs(self, count: int, rng: DetRng) -> list:
        return [StorySpec(rng.choice(("open", "paid", "paid", "split")), (self.TAG,),
                          window="not_late", max_docs=3)
                for _ in range(count)]

    def extra_documents(self, count: int) -> int:
        return count

    def extra_ledger_rows(self, count: int) -> int:
        return 2 * count

    def apply(self, books: Books, rng: DetRng, ordinal: int) -> Issue:
        ctx = books.ctx
        candidates = [
            inv for inv in books.invoices
            if self.TAG in inv.tags and id(inv) not in books.used and inv.kind == "invoice"
            and inv.booked_at is not None
            and (ctx.end - inv.booked_at).total_seconds() >= 6 * 3600
        ]
        if not candidates:
            raise GeneratorError("no candidate invoice for duplicate_invoice")
        original = rng.choice(candidates)
        books.used.add(id(original))
        variant = rng.choice(("same_invoice_id", "renumbered"))
        room = int((ctx.end - original.booked_at).total_seconds())
        delay = rng.randint(3600, min(3 * 86400, room))
        dup = InvoiceRec(
            kind="invoice", entity_id=original.entity_id, currency=original.currency,
            issued_at=original.issued_at, due_at=original.due_at,
            lines=_copy_lines(original.lines),
            booked_at=original.booked_at + timedelta(seconds=delay),
            display_offset=original.display_offset, story=original.story,
            order=books.next_order(), tags=["mutation:duplicate_invoice"],
            alias_of=original if variant == "same_invoice_id" else None)
        books.invoices.append(dup)
        books.used.add(id(dup))

        evidence = [[original, dup]]
        if variant == "same_invoice_id":
            evidence.append([LedgerRef(original, "revenue"), LedgerRef(dup, "revenue")])
            evidence.append([LedgerRef(original, "ar"), LedgerRef(dup, "ar")])
        mutation = {
            "type": self.name, "variant": variant, "ordinal": ordinal,
            "reason": "The invoice was entered a second time; the copy has its own document "
                      "and its own postings, so receivables and revenue are overstated.",
            "ops": [{"op": "clone_invoice", "source": original, "created": dup,
                     "reuses_invoice_id": variant == "same_invoice_id",
                     "copy_booked_at": S.format_timestamp(
                         dup.booked_at.astimezone(original.display_offset))}],
        }
        return Issue(self.name, [original, dup], original.total_minor, original.currency,
                     evidence, mutation)


class PeriodCutoffError(Mutator):
    """An invoice dated outside the period is still booked inside the period."""

    name = S.ISSUE_PERIOD_CUTOFF
    TAG = "reserved:period_cutoff_error"

    def reserved_specs(self, count: int, rng: DetRng) -> list:
        specs = []
        for _ in range(count):
            variant = rng.choice(("late", "early"))
            specs.append(StorySpec("open", (f"{self.TAG}:{variant}",),
                                   window="tail" if variant == "late" else "head", max_docs=1))
        return specs

    def apply(self, books: Books, rng: DetRng, ordinal: int) -> Issue:
        ctx = books.ctx
        candidates = [inv for inv in books.invoices
                      if any(t.startswith(self.TAG) for t in inv.tags)
                      and id(inv) not in books.used and inv.booked_at is not None]
        if not candidates:
            raise GeneratorError("no candidate invoice for period_cutoff_error")
        inv = rng.choice(candidates)
        books.used.add(id(inv))
        variant = "late" if f"{self.TAG}:late" in inv.tags else "early"
        shift = timedelta(seconds=rng.randint(3600, 4 * 86400))
        terms = inv.due_at - inv.issued_at
        old_issued, old_due = inv.issued_at, inv.due_at
        inv.issued_at = ctx.end + shift if variant == "late" else ctx.start - shift
        inv.due_at = inv.issued_at + terms
        inv.tags.append("mutation:period_cutoff_error")
        fmt = lambda dt: S.format_timestamp(dt.astimezone(inv.display_offset))  # noqa: E731
        mutation = {
            "type": self.name, "variant": variant, "ordinal": ordinal,
            "reason": "The invoice document belongs to the "
                      + ("following" if variant == "late" else "previous")
                      + " period but its postings stay inside the reporting period.",
            "ops": [
                {"op": "set_field", "record": inv, "field": "issued_at",
                 "old": fmt(old_issued), "new": fmt(inv.issued_at)},
                {"op": "set_field", "record": inv, "field": "due_at",
                 "old": fmt(old_due), "new": fmt(inv.due_at)},
            ],
        }
        evidence = [[inv, LedgerRef(inv, "ar")], [inv, LedgerRef(inv, "revenue")]]
        return Issue(self.name, [inv], inv.total_minor, inv.currency, evidence, mutation)


MUTATORS: dict = {m.name: m for m in (DuplicateInvoice(), PeriodCutoffError())}


# --------------------------------------------------------------------------- #
# Id schemes (ordering + naming); Stage 2 can subclass to shuffle / rename
# --------------------------------------------------------------------------- #
@dataclass
class Layout:
    ledger: list
    invoices: list
    payments: list


class IdScheme(ABC):
    """Strategy that orders records and assigns every public identifier."""

    @abstractmethod
    def assign(self, books: Books, ledger_rows: list) -> Layout:
        ...


class SequentialIdScheme(IdScheme):
    """Chronological order, readable sequential ids."""

    def assign(self, books: Books, ledger_rows: list) -> Layout:
        year = books.ctx.start.year
        ledger = list(ledger_rows)
        for n, row in enumerate(ledger, start=1):
            row.row_id = f"L{n:06d}"

        invoices = sorted(books.invoices, key=lambda i: (i.issued_at, i.order))
        payments = sorted(books.payments, key=lambda p: (p.ts, p.order))
        merged = sorted([(i.issued_at, 0, i.order, i) for i in invoices]
                        + [(p.ts, 1, p.order, p) for p in payments], key=lambda t: t[:3])
        for n, (_, _, _, doc) in enumerate(merged, start=1):
            doc.doc_id = f"DOC-{n:05d}"

        n_invoice = n_note = 0
        for inv in invoices:
            if inv.alias_of is not None:
                continue
            if inv.kind == "invoice":
                n_invoice += 1
                inv.invoice_id = f"INV-{year}-{n_invoice:04d}"
            else:
                n_note += 1
                inv.invoice_id = f"CN-{year}-{n_note:04d}"
        for inv in invoices:
            if inv.alias_of is not None:
                inv.invoice_id = inv.alias_of.invoice_id
        for n, pay in enumerate(payments, start=1):
            pay.payment_id = f"PAY-{year}-{n:04d}"
        return Layout(ledger, invoices, payments)


# --------------------------------------------------------------------------- #
# Episode container and finalisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Episode:
    episode_id: str
    manifest: dict
    policy: dict
    ledger: list
    invoices: list
    payments: list
    fx: list
    truth: list
    provenance: dict

    def public_files(self) -> dict:
        return {
            "manifest.json": S.dump_json(self.manifest),
            "policy.json": S.dump_json(self.policy),
            "ledger.csv": S.ledger_to_csv(self.ledger),
            "invoices.jsonl": S.to_jsonl(self.invoices),
            "payments.jsonl": S.to_jsonl(self.payments),
            "fx.csv": S.fx_to_csv(self.fx),
        }

    def private_files(self) -> dict:
        return {
            "truth.json": S.dump_json(self.truth),
            "provenance.json": S.dump_json(self.provenance),
        }


def write_episode(episode: Episode, root: Any) -> tuple:
    """Write ``root/public`` (agent-visible) and ``root/private`` (evaluator only)."""
    root = Path(root)
    public_dir, private_dir = root / "public", root / "private"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    for name, text in episode.public_files().items():
        (public_dir / name).write_text(text, encoding="utf-8", newline="")
    for name, text in episode.private_files().items():
        (private_dir / name).write_text(text, encoding="utf-8", newline="")
    return public_dir, private_dir


def _resolve(value: Any, index: dict) -> Any:
    if isinstance(value, InvoiceRec):
        return {"table": "invoices", "id": value.doc_id}
    if isinstance(value, PaymentRec):
        return {"table": "payments", "id": value.doc_id}
    if isinstance(value, LedgerRef):
        return {"table": "ledger", "id": index[(id(value.owner), value.role)].row_id}
    if isinstance(value, (list, tuple)):
        return [_resolve(v, index) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, index) for k, v in value.items()}
    return value


def _rid(obj: Any, index: dict) -> str:
    return _resolve(obj, index)["id"]


def _fmt(ts: datetime, offset: timezone) -> str:
    return S.format_timestamp(ts.astimezone(offset))


def _build_policy(ctx: PeriodCtx, cfg: S.EpisodeConfig) -> dict:
    return {
        "policy_version": S.SCHEMA_VERSION,
        "period": {
            "start": S.format_timestamp(ctx.start),
            "end": S.format_timestamp(ctx.end),
            "start_inclusive": True,
            "end_inclusive": True,
            "timezone": cfg.timezone,
            "rule": "A document or posting belongs to the reporting period if and only if "
                    "period.start <= its timestamp <= period.end, comparing absolute instants "
                    "(the UTC offset written in each timestamp must be honoured).",
        },
        "fx_rate_date_rule": {
            "rule": "local_calendar_date_of_document_timestamp",
            "timezone": cfg.timezone,
            "fallback": "latest_earlier_available_date",
            "description": "To convert a foreign-currency amount, use the fx.csv rate dated on the "
                           "company-local calendar date of the document timestamp; if that date "
                           "has no rate (e.g. weekend), use the latest earlier date. Rows in "
                           "fx.csv are dated at local midnight.",
        },
        "rounding_rule": {
            "mode": S.ROUNDING_RULE,
            "unit": "minor unit of the target currency",
            "applied": "once, after the full-precision conversion",
        },
        "amount_tolerance_minor": cfg.amount_tolerance_minor,
        "ledger_conventions": {
            "currency": "Ledger amounts are denominated in the row currency; debits equal "
                        "credits within every currency.",
            "documents": "invoices.jsonl holds invoices and credit notes (CN-*, negative "
                         "total_minor). An invoice dated outside the period without ledger "
                         "postings is correctly excluded from this period.",
            "references": "Missing references are null (empty cell in ledger.csv).",
        },
        "chart_of_accounts": dict(S.CHART_OF_ACCOUNTS),
    }


def _finalize(seed: int, cfg: S.EpisodeConfig, episode_id: str, books: Books, issues: list,
              plan: list, clean_checks: dict, scheme: IdScheme) -> Episode:
    ctx = books.ctx
    rows, index = build_ledger(books)
    layout = scheme.assign(books, rows)

    ledger = [{
        "row_id": r.row_id, "timestamp": _fmt(r.ts, r.offset), "account": r.account,
        "debit_minor": r.debit, "credit_minor": r.credit, "currency": r.currency,
        "entity_id": r.entity_id,
        "invoice_id": r.invoice.invoice_id if r.invoice is not None else None,
        "payment_id": r.payment.payment_id if r.payment is not None else None,
    } for r in layout.ledger]
    invoices = [{
        "doc_id": inv.doc_id, "invoice_id": inv.invoice_id, "entity_id": inv.entity_id,
        "issued_at": _fmt(inv.issued_at, inv.display_offset),
        "due_at": _fmt(inv.due_at, inv.display_offset),
        "currency": inv.currency, "total_minor": inv.total_minor,
        "line_items": [{
            "line_no": n, "description": line.description, "quantity": line.quantity,
            "unit_price_minor": line.unit_price_minor, "amount_minor": line.amount_minor,
            "ref_invoice_id": line.ref.invoice_id if line.ref is not None else None,
        } for n, line in enumerate(inv.lines, start=1)],
    } for inv in layout.invoices]
    payments = [{
        "doc_id": p.doc_id, "payment_id": p.payment_id, "timestamp": _fmt(p.ts, ctx.tz),
        "currency": p.currency, "amount_minor": p.amount_minor,
        "invoice_refs": [inv.invoice_id for inv, _ in p.allocations],
    } for p in layout.payments]
    fx = [dict(r) for r in books.fx]

    manifest = {
        "schema_version": S.SCHEMA_VERSION,
        "episode_id": episode_id,
        "base_currency": ctx.base,
        "currency_minor_units": {c: S.SUPPORTED_CURRENCIES[c]
                                 for c in [ctx.base, *sorted(cfg.foreign_currencies)]},
        "period_start": S.format_timestamp(ctx.start),
        "period_end": S.format_timestamp(ctx.end),
        "timezone": cfg.timezone,
        "counts": {"ledger_rows": len(ledger), "invoices": len(invoices),
                   "payments": len(payments), "documents": len(invoices) + len(payments),
                   "fx_rates": len(fx)},
        "rounding_rule": S.ROUNDING_RULE,
    }
    policy = _build_policy(ctx, cfg)
    S.validate_public(manifest, policy, ledger, invoices, payments, fx)
    if len(ledger) != cfg.n_ledger_rows or manifest["counts"]["documents"] != cfg.n_documents:
        raise GeneratorError("generated sizes do not match the configuration")

    truth = []
    for n, issue in enumerate(issues, start=1):
        sets = sorted({tuple(sorted({_rid(o, index) for o in ev})) for ev in issue.evidence_sets})
        truth.append({
            "issue_id": f"ISS-{n:04d}",
            "type": issue.type,
            "affected_ids": sorted({_rid(o, index) for o in issue.affected}),
            "expected_amount_minor": issue.amount_minor,
            "currency": issue.currency,
            "valid_evidence_sets": [list(s) for s in sets],
        })
    for entry in truth:
        S.validate_truth_entry(entry)

    mutations = []
    for n, issue in enumerate(issues, start=1):
        record = _resolve(issue.mutation, index)
        record["mutation_id"] = f"MUT-{n:04d}"
        record["issue_id"] = f"ISS-{n:04d}"
        mutations.append(record)

    provenance = {
        "generator_version": S.GENERATOR_VERSION,
        "schema_version": S.SCHEMA_VERSION,
        "episode_id": episode_id,
        "seed": seed,
        "config": cfg.to_dict(),
        "error_plan": list(plan),
        "company": {
            "customers": [{"entity_id": p.entity_id, "currency": p.currency} for p in books.customers],
            "suppliers": list(books.suppliers),
        },
        "stories": [{"story": s["story"], "kind": s["kind"], "tags": s["tags"],
                     "records": _resolve(s["records"], index)} for s in books.story_log],
        "clean_books_checks": clean_checks,
        "mutations": mutations,
        "final_counts": manifest["counts"],
    }
    return Episode(episode_id, manifest, policy, ledger, invoices, payments, fx, truth, provenance)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_episode(seed: int, config: Any = None, *, id_scheme: Optional[IdScheme] = None) -> Episode:
    """Generate one episode.  Same (seed, config, GENERATOR_VERSION) -> identical bytes."""
    S.validate_seed(seed)
    cfg = S.coerce_config(config)
    unknown = [t for t in cfg.error_types if t not in MUTATORS]
    if unknown:
        raise S.SchemaError(f"config.error_types: unsupported types {unknown}")

    material = f"{S.GENERATOR_VERSION}|{seed}|{cfg.canonical_json()}"
    episode_id = "EP-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12].upper()
    root = DetRng(material)
    ctx = PeriodCtx.from_config(cfg)
    books = Books(config=cfg, ctx=ctx)

    plan_rng = root.fork("plan")
    types = list(cfg.error_types)
    plan_rng.shuffle(types)
    plan = [types[i % len(types)] for i in range(cfg.n_errors)] if cfg.n_errors else []
    plan_rng.shuffle(plan)
    counts = {name: plan.count(name) for name in sorted(set(plan))}

    _build_parties(books, root.fork("parties"))
    _build_clean_stories(books, root.fork("stories"), counts)
    _build_fx(books, root.fork("fx"))
    _build_filler(books, root.fork("filler"), counts)
    clean_checks = _verify_clean(books)

    mutation_rng = root.fork("mutations")
    issues = [MUTATORS[name].apply(books, mutation_rng.fork(str(i)), i)
              for i, name in enumerate(plan)]
    return _finalize(seed, cfg, episode_id, books, issues, plan, clean_checks,
                     id_scheme or SequentialIdScheme())
