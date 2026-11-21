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
ACC_UNAPPLIED = "2100"
ACC_REVENUE = "4000"
ACC_FX = "4900"
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
    posted: Optional[list] = None  # foreign amounts actually posted to AR per allocation (mutations)
    excess: int = 0                # legitimate overpayment kept as unapplied cash

    @property
    def amount_minor(self) -> int:
        return sum(amount for _, amount in self.allocations) + self.excess


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


@dataclass(frozen=True)
class FxRef:
    """Reference to an fx.csv row (its id is derived from its content)."""

    fx_id: str


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
    translator: Any = None
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
    foreign: bool = False          # counterparty must invoice in a foreign currency


def _customer(books: "Books", rng: "DetRng", spec: StorySpec) -> "Party":
    if not spec.foreign:
        return rng.choice(books.customers)
    pool = [p for p in books.customers if p.currency != books.ctx.base]
    if not pool:
        raise GeneratorError("no foreign-currency customer for a foreign story")
    return rng.choice(pool)


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
    party = _customer(books, rng, spec)
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
    party = _customer(books, rng, spec)
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
    party = _customer(books, rng, spec)
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
    party = _customer(books, rng, spec)
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
    party = _customer(books, rng, spec)
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


def _build_overpaid(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    """Legitimate overpayment: the customer pays more than the invoice; the excess stays
    in unapplied cash (2100) until it is refunded or applied to a future invoice."""
    if max_docs < 2:
        return _build_open(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = _customer(books, rng, spec)
    lag = rng.randint(3, min(30, ctx.days - 2))
    day = _pick_day(ctx, rng, spec.window, hi=ctx.days - 1 - lag)
    issued = ctx.start + timedelta(days=day, seconds=_tod(rng))
    inv = _make_invoice(books, rng, party, issued, _make_lines(rng, party.currency), story, spec,
                        extra_tags=("overpaid",))
    ts = ctx.start + timedelta(days=day + lag, seconds=_tod(rng))
    pay = _make_payment(books, party, ts, [(inv, inv.total_minor)], story)
    pay.excess = max(1, inv.total_minor * rng.randint(2, 30) // 100)
    books.log_story(story, "overpaid", spec, [inv, pay])
    return 2


def _build_boundary_paid(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    """Invoice mid-period, paid on the last local evening of the period."""
    if max_docs < 2:
        return _build_open(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = _customer(books, rng, spec)
    day = rng.randint(0, max(0, ctx.days - 10))
    issued = ctx.start + timedelta(days=day, seconds=_tod(rng))
    inv = _make_invoice(books, rng, party, issued, _make_lines(rng, party.currency), story, spec)
    ts = ctx.start + timedelta(days=ctx.days - 1, seconds=rng.randint(20 * 3600, 23 * 3600 + 3000))
    pay = _make_payment(books, party, ts, [(inv, inv.total_minor)], story)
    books.log_story(story, "boundary_paid", spec, [inv, pay])
    return 2


def _build_recurring(books: Books, rng: DetRng, spec: StorySpec, max_docs: int) -> int:
    """Legitimate repeated amounts: the same customer billed the same amount 3 times."""
    if max_docs < 3:
        return _build_paid(books, rng, spec, max_docs)
    ctx = books.ctx
    story = books.new_story()
    party = _customer(books, rng, spec)
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
        party = _customer(books, rng, spec)
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
    "overpaid": _build_overpaid,
    "boundary_paid": _build_boundary_paid,
    "boundary_start": _special_builder("boundary_start"),
    "boundary_end": _special_builder("boundary_end"),
    "tz_head_utc": _special_builder("tz_head_utc"),
    "out_of_next": _special_builder("out_of_next"),
    "out_of_prev": _special_builder("out_of_prev"),
}
STORY_MIN_DOCS = {"open": 1, "paid": 2, "split": 2, "return": 3, "combined": 3, "recurring": 3,
                  "overpaid": 2, "boundary_paid": 2,
                  "boundary_start": 1, "boundary_end": 1, "tz_head_utc": 1,
                  "out_of_next": 1, "out_of_prev": 1}
STORY_MAX_DOCS = {"open": 1, "paid": 2, "split": 4, "return": 3, "combined": 3, "recurring": 6,
                  "overpaid": 2, "boundary_paid": 2,
                  "boundary_start": 1, "boundary_end": 1, "tz_head_utc": 1,
                  "out_of_next": 1, "out_of_prev": 1}
FORCED_DISTRACTORS = ("recurring", "split", "return", "combined", "overpaid", "boundary_paid",
                      "boundary_start",
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
    foreign = sorted(cfg.foreign_currencies)
    if foreign and sum(p.currency != cfg.base_currency for p in books.customers) < 2:
        for k, party in enumerate(books.customers[:2]):
            party.currency = foreign[k % len(foreign)]
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
def local_date(ts: datetime, ctx: PeriodCtx) -> str:
    """Company-local calendar date of an instant (the FX rate-date rule)."""
    return ts.astimezone(ctx.tz).date().isoformat()


class Translator:
    """Translates foreign-currency documents into the base (functional) currency.

    Published rules (policy.json):
    * an invoice is booked at the rate of the local date of its posting;
    * a credit note and every allocation of a payment relieve receivables at the
      historical rate of the invoice they refer to; the allocation that settles an
      invoice completely relieves exactly the remaining base balance (no residue);
    * cash is booked at the rate of the payment date;
    * the difference between cash and relieved receivables goes to 4900.
    """

    def __init__(self, books: Books) -> None:
        self.books = books
        self.ctx = books.ctx
        self.lookup = FxLookup(books.fx)
        self.digits = S.SUPPORTED_CURRENCIES
        self.rate_overrides: dict = {}   # (id(record), purpose) -> rate string (FX mutators)

    def rate(self, currency: str, ts: datetime, record: Any = None, purpose: str = "") -> str:
        override = self.rate_overrides.get((id(record), purpose))
        if override is not None:
            return override
        return self.lookup.rate(currency, self.ctx.base, local_date(ts, self.ctx))

    def convert(self, amount: int, currency: str, rate: str) -> int:
        if currency == self.ctx.base:
            return amount
        return S.convert_minor(amount, rate, self.digits[currency], self.digits[self.ctx.base])

    def invoice_rate(self, inv: InvoiceRec) -> Optional[str]:
        if inv.currency == self.ctx.base or inv.booked_at is None:
            return None
        return self.rate(inv.currency, inv.booked_at, inv, "booking")

    def plan(self) -> dict:
        """Base amounts for every posting, computed in chronological order."""
        out: dict = {"invoice": {}, "credit": {}, "relief": {}, "unapplied": {}, "excess": {},
                     "cash": {}}
        outstanding: dict = {}       # id(invoice) -> [foreign, base]
        events = []
        for inv in self.books.invoices:
            if inv.booked_at is None:
                continue
            events.append((inv.booked_at, 0, inv.order, inv))
        for pay in self.books.payments:
            events.append((pay.ts, 1, pay.order, pay))
        events.sort(key=lambda e: e[:3])

        def relieve(inv: InvoiceRec, amount: int) -> int:
            state = outstanding.get(id(inv))
            if state is None:              # allocation to an invoice not booked in this period
                rate = self.rate(inv.currency, inv.issued_at, inv, "booking")
                return self.convert(amount, inv.currency, rate)
            if amount == state[0]:
                base = state[1]
            else:
                base = self.convert(amount, inv.currency, self.invoice_rate(inv) or "1.0000")
            state[0] -= amount
            state[1] -= base
            return base

        for _, kind, _, rec in events:
            if kind == 0 and rec.kind == "invoice":
                base = self.convert(rec.total_minor, rec.currency, self.invoice_rate(rec) or "1.0000")
                out["invoice"][id(rec)] = base
                outstanding[id(rec)] = [rec.total_minor, base]
            elif kind == 0:                                     # credit note
                target = rec.lines[0].ref
                out["credit"][id(rec)] = relieve(target, -rec.total_minor)
            else:
                rate = None if rec.currency == self.ctx.base else self.rate(rec.currency, rec.ts, rec, "cash")
                out["cash"][id(rec)] = self.convert(rec.amount_minor, rec.currency, rate or "1.0000")
                for position, (inv, amount) in enumerate(rec.allocations):
                    posted = amount if rec.posted is None else rec.posted[position]
                    out["relief"][(id(rec), position)] = relieve(inv, posted) if posted > 0 else None
                    left = amount - posted
                    out["unapplied"][(id(rec), position)] = (
                        self.convert(left, rec.currency, rate or "1.0000") if left > 0 else None)
                out["excess"][id(rec)] = (self.convert(rec.excess, rec.currency, rate or "1.0000")
                                          if rec.excess > 0 else None)
        out["outstanding"] = outstanding
        return out


def build_ledger(books: Books, translator: Optional[Translator] = None) -> tuple:
    """Derive ledger rows (all in the base currency).  Returns (rows sorted by time, index)."""
    ctx = books.ctx
    translator = translator or getattr(books, "translator", None) or Translator(books)
    plan = translator.plan()
    base = ctx.base
    rows: list = []

    def add(ts, offset, account, debit, credit, entity, invoice, payment, owner, role):
        rows.append(LedgerRow(ts, offset, account, debit, credit, base, entity, invoice,
                              payment, owner, role, len(rows)))

    for jr in books.journals:
        for account, debit, credit, entity in jr.legs:
            add(jr.ts, ctx.tz, account, debit, credit, entity, None, None, jr, "journal")
    for inv in books.invoices:
        if inv.booked_at is None:
            continue
        if inv.kind == "invoice":
            amount = plan["invoice"][id(inv)]
            add(inv.booked_at, inv.display_offset, ACC_AR, amount, 0, inv.entity_id, inv, None, inv, "ar")
            add(inv.booked_at, inv.display_offset, ACC_REVENUE, 0, amount, inv.entity_id, inv, None,
                inv, "revenue")
        else:
            amount = plan["credit"][id(inv)]
            add(inv.booked_at, inv.display_offset, ACC_RETURNS, amount, 0, inv.entity_id, inv, None,
                inv, "contra")
            add(inv.booked_at, inv.display_offset, ACC_AR, 0, amount, inv.entity_id, inv, None, inv, "ar")
    for pay in books.payments:
        cash = plan["cash"][id(pay)]
        add(pay.ts, ctx.tz, ACC_CASH, cash, 0, pay.entity_id, None, pay, pay, "cash")
        relieved = 0
        for position, (inv, _amount) in enumerate(pay.allocations):
            relief = plan["relief"][(id(pay), position)]
            if relief is not None:
                relieved += relief
                add(pay.ts, ctx.tz, ACC_AR, 0, relief, pay.entity_id, inv, pay, pay, f"ar:{position}")
            unapplied = plan["unapplied"][(id(pay), position)]
            if unapplied is not None:
                relieved += unapplied
                add(pay.ts, ctx.tz, ACC_UNAPPLIED, 0, unapplied, pay.entity_id, None, pay, pay,
                    f"unapplied:{position}")
        excess = plan["excess"][id(pay)]
        if excess is not None:
            relieved += excess
            add(pay.ts, ctx.tz, ACC_UNAPPLIED, 0, excess, pay.entity_id, None, pay, pay, "excess")
        diff = cash - relieved
        if diff > 0:
            add(pay.ts, ctx.tz, ACC_FX, 0, diff, pay.entity_id, None, pay, pay, "fx")
        elif diff < 0:
            add(pay.ts, ctx.tz, ACC_FX, -diff, 0, pay.entity_id, None, pay, pay, "fx")
    rows.sort(key=lambda r: (r.ts, r.seq))
    index = {(id(r.owner), r.role): r for r in rows}
    return rows, index


def _build_filler(books: Books, rng: DetRng) -> None:
    """Balanced operating journals so that the ledger has exactly n_ledger_rows rows.

    Runs after the error injection, so it absorbs any row-count change caused by
    a mutation (journals are independent of documents)."""
    cfg, ctx = books.config, books.ctx
    doc_rows = len(build_ledger(books)[0])
    free = cfg.n_ledger_rows - doc_rows
    if free < 0 or free == 1:
        raise GeneratorError(f"cannot fill ledger: {free} filler rows required")
    if free == 0:
        return
    opening = 3 if free == 3 else 2
    remaining = free - opening
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
    if opening == 2:
        legs = [(ACC_CASH, capital, 0, None), (ACC_EQUITY, 0, capital, None)]
    else:
        first = capital // 2
        legs = [(ACC_CASH, capital, 0, None), (ACC_EQUITY, 0, first, "OWNER-1"),
                (ACC_EQUITY, 0, capital - first, "OWNER-2")]
    books.journals.append(JournalRec(ctx.start + timedelta(minutes=30), ctx.base, legs,
                                     "opening_capital", books.next_order()))


def _check_books(books: Books, *, allow_mutations: bool) -> dict:
    """Recompute the books independently and prove they are consistent.

    Clean books: balanced, every invoice/payment translated by the published
    rule, receivables per invoice tie out to the documents, nothing over-settled
    and nothing left in unapplied cash."""
    ctx = books.ctx
    rows, _ = build_ledger(books)
    debit = sum(r.debit for r in rows)
    credit = sum(r.credit for r in rows)
    if debit != credit:
        raise GeneratorError("ledger not balanced")
    plan = books.translator.plan()
    lookup = FxLookup(books.fx)
    digits = S.SUPPORTED_CURRENCIES

    def independent(amount: int, currency: str, ts: datetime) -> int:
        if currency == ctx.base:
            return amount
        rate = lookup.rate(currency, ctx.base, local_date(ts, ctx))
        return S.convert_minor(amount, rate, digits[currency], digits[ctx.base])

    ar_by_inv: dict = {}
    for r in rows:
        if r.account == ACC_AR and r.invoice is not None:
            key = r.invoice.lines[0].ref if r.invoice.kind == "credit_note" else r.invoice
            ar_by_inv[id(key)] = ar_by_inv.get(id(key), 0) + r.debit - r.credit
    revenue_base = 0
    open_base = 0
    for inv in books.invoices:
        if inv.booked_at is None or inv.kind != "invoice":
            continue
        foreign, base = plan["outstanding"][id(inv)]
        if not allow_mutations:
            if plan["invoice"][id(inv)] != independent(inv.total_minor, inv.currency, inv.booked_at):
                raise GeneratorError("invoice translated with a wrong rate")
            if foreign < 0:
                raise GeneratorError("an invoice is over-settled")
            if foreign == 0 and base != 0:
                raise GeneratorError("settled invoice leaves a residue in receivables")
        if ar_by_inv.get(id(inv), 0) != base:
            raise GeneratorError("receivables do not tie out to the documents")
        revenue_base += plan["invoice"][id(inv)]
        open_base += base
    unapplied = sum(r.credit - r.debit for r in rows if r.account == ACC_UNAPPLIED)
    if not allow_mutations:
        for pay in books.payments:
            if plan["cash"][id(pay)] != independent(pay.amount_minor, pay.currency, pay.ts):
                raise GeneratorError("payment translated with a wrong rate")
        legit = sum(v for v in plan["excess"].values() if v is not None)
        if unapplied != legit:
            raise GeneratorError("clean books hold unapplied cash beyond overpayments")
    return {
        "balanced": {"currency": ctx.base, "debit_minor": debit, "credit_minor": credit},
        "ar_open_base_minor": open_base,
        "unapplied_cash_base_minor": unapplied,
        "booked_revenue_in_base_minor": revenue_base,
        "fx_gain_loss_base_minor": sum(r.credit - r.debit for r in rows if r.account == ACC_FX),
        "ledger_rows": len(rows),
        "documents": len(books.invoices) + len(books.payments),
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
        candidates = [inv for inv in books.invoices
                      if any(t.startswith(self.TAG) for t in inv.tags)
                      and id(inv) not in books.used and inv.booked_at is not None]
        if not candidates:
            raise GeneratorError("no candidate invoice for period_cutoff_error")
        inv = rng.choice(candidates)
        books.used.add(id(inv))
        variant = "late" if f"{self.TAG}:late" in inv.tags else "early"
        return shift_out_of_period(books, rng, inv, variant, ordinal)


def shift_out_of_period(books: Books, rng: DetRng, inv: InvoiceRec, variant: str, ordinal: int,
                        min_shift_s: int = 3600) -> Issue:
    """Move the document date of a booked invoice out of the period (postings unchanged)."""
    ctx = books.ctx
    shift = timedelta(seconds=rng.randint(min_shift_s, 4 * 86400))
    terms = inv.due_at - inv.issued_at
    old_issued, old_due = inv.issued_at, inv.due_at
    inv.issued_at = ctx.end + shift if variant == "late" else ctx.start - shift
    inv.due_at = inv.issued_at + terms
    inv.tags.append("mutation:period_cutoff_error")
    fmt = lambda dt: S.format_timestamp(dt.astimezone(inv.display_offset))  # noqa: E731
    mutation = {
        "type": S.ISSUE_PERIOD_CUTOFF, "variant": variant, "ordinal": ordinal,
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
    return Issue(S.ISSUE_PERIOD_CUTOFF, [inv], inv.total_minor, inv.currency, evidence, mutation)


class PartialPaymentAllocation(Mutator):
    """A payment is received in full, but only part of it (or none) is applied to the
    invoice it names; the rest sits in unapplied cash, so the invoice stays open."""

    name = S.ISSUE_PARTIAL_PAYMENT
    TAG = "reserved:partial_payment_allocation"

    def reserved_specs(self, count: int, rng: DetRng) -> list:
        return [StorySpec(rng.choice(("paid", "paid", "split")), (self.TAG,), max_docs=3)
                for _ in range(count)]

    def apply(self, books: Books, rng: DetRng, ordinal: int) -> Issue:
        candidates = [pay for pay in books.payments
                      if id(pay) not in books.used and pay.posted is None and pay.excess == 0
                      and len(pay.allocations) == 1 and self.TAG in pay.allocations[0][0].tags
                      and id(pay.allocations[0][0]) not in books.used]
        if not candidates:
            raise GeneratorError("no candidate payment for partial_payment_allocation")
        pay = rng.choice(candidates)
        inv, amount = pay.allocations[0]
        books.used.update({id(pay), id(inv)})
        variant = "missing" if amount < 4 or rng.chance(35) else "partial"
        posted = 0 if variant == "missing" else amount * rng.randint(20, 80) // 100
        pay.posted = [posted]
        unapplied = amount - posted
        evidence = [[pay, LedgerRef(pay, "unapplied:0")]]
        if variant == "partial":
            evidence.append([pay, LedgerRef(pay, "ar:0")])
        mutation = {
            "type": self.name, "variant": variant, "ordinal": ordinal,
            "reason": "The payment document applies the full amount to the invoice, but the "
                      "posting applies only part of it; the remainder was left in unapplied cash "
                      "and the invoice still looks open in receivables.",
            "ops": [{"op": "set_posted_allocation", "record": pay, "invoice": inv,
                     "document_amount_minor": amount, "posted_amount_minor": posted}],
        }
        return Issue(self.name, [pay, inv], unapplied, pay.currency, evidence, mutation)


def fx_row_for(books: Books, currency: str, local: str) -> dict:
    """The fx.csv row the published rule selects for a local date (latest earlier)."""
    best = None
    for row in books.fx:
        if row["source_currency"] == currency and row["date"][:10] <= local:
            best = row
    if best is None:
        raise GeneratorError("no fx row for the requested date")
    return best


def misdate_rate(books: Books, rng: DetRng, record: Any, purpose: str, ts: datetime,
                 amount: int, currency: str, wrong_dates: Sequence[tuple], ordinal: int,
                 affected: list, evidence_roles: Sequence[str]) -> Issue:
    """Translate one posting at the rate of a wrong date (an fx_rate_error).

    ``wrong_dates`` is a list of (variant, local_date) candidates tried in order;
    the first one whose rate changes the base amount by more than the matching
    tolerance is used."""
    ctx = books.ctx
    digits = S.SUPPORTED_CURRENCIES
    right_local = local_date(ts, ctx)
    right_row = fx_row_for(books, currency, right_local)
    right_base = S.convert_minor(amount, right_row["rate"], digits[currency], digits[ctx.base])
    floor = max(100, books.config.amount_tolerance_minor + 1)
    for variant, wrong_local in wrong_dates:
        wrong_row = fx_row_for(books, currency, wrong_local)
        wrong_base = S.convert_minor(amount, wrong_row["rate"], digits[currency], digits[ctx.base])
        if wrong_row["rate"] != right_row["rate"] and abs(wrong_base - right_base) >= floor:
            break
    else:
        raise GeneratorError("no wrong-date rate changes the posting enough")
    books.translator.rate_overrides[(id(record), purpose)] = wrong_row["rate"]
    if hasattr(record, "tags"):
        record.tags.append("mutation:fx_rate_error")
    fx_ref = FxRef(S.fx_record_id(right_row))
    mutation = {
        "type": S.ISSUE_FX_RATE, "variant": variant, "ordinal": ordinal,
        "reason": "The posting was translated at the rate of the wrong date; the published "
                  "rule selects the rate of the company-local date of the posting.",
        "ops": [{"op": "override_rate", "record": record, "purpose": purpose,
                 "correct_rate_date": right_row["date"][:10], "correct_rate": right_row["rate"],
                 "used_rate_date": wrong_row["date"][:10], "used_rate": wrong_row["rate"],
                 "correct_base_minor": right_base, "posted_base_minor": wrong_base}],
    }
    evidence = [[record, LedgerRef(record, role), fx_ref] for role in evidence_roles]
    return Issue(S.ISSUE_FX_RATE, affected, abs(wrong_base - right_base), ctx.base, evidence, mutation)


def _shifted(local: str, days: int) -> str:
    return (date.fromisoformat(local) + timedelta(days=days)).isoformat()


def invoice_wrong_dates(books: Books, rng: DetRng, inv: InvoiceRec) -> list:
    right = local_date(inv.booked_at, books.ctx)
    options = [("due_date", local_date(inv.due_at, books.ctx))]
    options += [("previous_days", _shifted(right, -d)) for d in range(1, 11)]
    options += [("next_days", _shifted(right, d)) for d in range(1, 11)]
    head, tail = options[:1], options[1:]
    rng.shuffle(tail)
    return head + tail if rng.chance(40) else tail + head


class FxRateError(Mutator):
    """A foreign-currency invoice is booked at the rate of the wrong date."""

    name = S.ISSUE_FX_RATE
    TAG = "reserved:fx_rate_error"

    def reserved_specs(self, count: int, rng: DetRng) -> list:
        # one spare story: a tiny invoice may not move enough under any wrong date
        return [StorySpec(rng.choice(("open", "paid")), (self.TAG,), max_docs=2, foreign=True)
                for _ in range(count + 1)]

    def apply(self, books: Books, rng: DetRng, ordinal: int) -> Issue:
        candidates = [inv for inv in books.invoices
                      if self.TAG in inv.tags and id(inv) not in books.used
                      and inv.booked_at is not None and inv.currency != books.ctx.base]
        if not candidates:
            raise GeneratorError("no candidate invoice for fx_rate_error")
        rng.shuffle(candidates)
        for inv in candidates:                      # small invoices may not move enough
            try:
                issue = misdate_rate(books, rng, inv, "booking", inv.booked_at, inv.total_minor,
                                     inv.currency, invoice_wrong_dates(books, rng, inv), ordinal,
                                     [inv], ("ar", "revenue"))
            except GeneratorError:
                continue
            books.used.add(id(inv))
            return issue
        raise GeneratorError("no candidate invoice for fx_rate_error moves enough")


class Scenario(Mutator):
    """Interacting errors: one causal story, two linked reference issues."""

    def apply(self, books: Books, rng: DetRng, ordinal: int) -> list:  # type: ignore[override]
        raise NotImplementedError


class PartialFxBoundary(Scenario):
    """A foreign-currency payment received on the last evening of the period is only
    partly applied AND its cash is translated at the next day's rate."""

    name = "scenario:partial_fx_boundary"
    TAG = "reserved:scenario:partial_fx_boundary"

    def reserved_specs(self, count: int, rng: DetRng) -> list:
        return [StorySpec("boundary_paid", (self.TAG,), max_docs=2, foreign=True) for _ in range(count)]

    def apply(self, books: Books, rng: DetRng, ordinal: int) -> list:  # type: ignore[override]
        candidates = [pay for pay in books.payments
                      if id(pay) not in books.used and len(pay.allocations) == 1
                      and self.TAG in pay.allocations[0][0].tags]
        if not candidates:
            raise GeneratorError("no candidate payment for partial_fx_boundary")
        pay = rng.choice(candidates)
        inv, amount = pay.allocations[0]
        books.used.update({id(pay), id(inv)})
        posted = amount * rng.randint(30, 80) // 100
        pay.posted = [posted]
        partial = Issue(
            S.ISSUE_PARTIAL_PAYMENT, [pay, inv], amount - posted, pay.currency,
            [[pay, LedgerRef(pay, "unapplied:0")], [pay, LedgerRef(pay, "ar:0")]],
            {"type": S.ISSUE_PARTIAL_PAYMENT, "variant": "partial_at_period_end", "ordinal": ordinal,
             "scenario": "partial_fx_boundary",
             "reason": "Payment received on the last evening of the period was applied only in part.",
             "ops": [{"op": "set_posted_allocation", "record": pay, "invoice": inv,
                      "document_amount_minor": amount, "posted_amount_minor": posted}]})
        right = local_date(pay.ts, books.ctx)
        wrong = [("next_period_day", _shifted(right, d)) for d in range(1, 11)]
        fx = misdate_rate(books, rng, pay, "cash", pay.ts, pay.amount_minor, pay.currency, wrong,
                          ordinal, [pay], ("cash",))
        fx.mutation["scenario"] = "partial_fx_boundary"
        return [partial, fx]


class CutoffFx(Scenario):
    """An invoice dated in the next period is booked in this one AND translated at the
    rate of its document date instead of its posting date."""

    name = "scenario:cutoff_fx"
    TAG = "reserved:scenario:cutoff_fx"

    def reserved_specs(self, count: int, rng: DetRng) -> list:
        return [StorySpec("open", (self.TAG,), window="tail", max_docs=1, foreign=True)
                for _ in range(count)]

    def apply(self, books: Books, rng: DetRng, ordinal: int) -> list:  # type: ignore[override]
        candidates = [inv for inv in books.invoices
                      if self.TAG in inv.tags and id(inv) not in books.used and inv.booked_at is not None]
        if not candidates:
            raise GeneratorError("no candidate invoice for cutoff_fx")
        inv = rng.choice(candidates)
        books.used.add(id(inv))
        cutoff = shift_out_of_period(books, rng, inv, "late", ordinal, min_shift_s=86400)
        cutoff.mutation["scenario"] = "cutoff_fx"
        right = local_date(inv.booked_at, books.ctx)
        doc_day = local_date(inv.issued_at, books.ctx)
        wrong = [("document_date", doc_day)] + [("document_date_next", _shifted(doc_day, d))
                                                for d in range(1, 11)]
        fx = misdate_rate(books, rng, inv, "booking", inv.booked_at, inv.total_minor, inv.currency,
                          wrong, ordinal, [inv], ("ar", "revenue"))
        fx.mutation["scenario"] = "cutoff_fx"
        return [cutoff, fx]


SCENARIOS: dict = {m.name: m for m in (PartialFxBoundary(), CutoffFx())}


MUTATORS: dict = {m.name: m for m in (DuplicateInvoice(), PeriodCutoffError(),
                                       PartialPaymentAllocation(), FxRateError(),
                                       *SCENARIOS.values())}


# --------------------------------------------------------------------------- #
# Id schemes (ordering + naming)
# --------------------------------------------------------------------------- #
@dataclass
class Layout:
    """Public order of every table and the naming of entities.

    ``fx=None`` keeps the generator's FX order; ``entity_map`` renames
    customers/suppliers (missing keys keep their original name).
    """

    ledger: list
    invoices: list
    payments: list
    fx: Optional[list] = None
    entity_map: dict = field(default_factory=dict)


class IdScheme(ABC):
    """Strategy that orders records and assigns every public identifier.

    An id scheme is *presentation only*: it may reorder rows and rename ids
    but never changes amounts, timestamps, links or the set of records.
    The reference answer is resolved after ``assign`` through object
    references, so it follows any renaming automatically.
    """

    name = "abstract"

    def fingerprint(self) -> str:
        """Stable text describing the scheme (enters the episode id)."""
        return self.name

    @abstractmethod
    def assign(self, books: Books, ledger_rows: list, rng: Optional[DetRng] = None) -> Layout:
        ...


class SequentialIdScheme(IdScheme):
    """Chronological order, readable sequential ids (Stage 1 behaviour)."""

    name = "sequential"

    def assign(self, books: Books, ledger_rows: list, rng: Optional[DetRng] = None) -> Layout:
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


class _HexIds:
    """Collision-free 6-hex-digit ids per namespace, drawn from a DetRng."""

    SPACE = 0xFFFFFF

    def __init__(self, rng: DetRng) -> None:
        self._rng = rng
        self._used: dict = {}

    def draw(self, namespace: str) -> str:
        used = self._used.setdefault(namespace, set())
        if len(used) > self.SPACE // 2:
            raise GeneratorError(f"id space exhausted for {namespace}")
        while True:
            value = f"{self._rng.randint(0, self.SPACE):06x}"
            if value not in used:
                used.add(value)
                return value


class RandomizedIdScheme(IdScheme):
    """Shuffled row order and random-looking hex ids (``DOC-91b0e4``).

    Deterministic: the same (seed, history config, salt) always gives the
    same order and the same ids.  ``salt`` selects an independent renaming
    of the very same history, which enables paired invariance checks.
    """

    name = "randomized"

    def __init__(self, salt: int = 0) -> None:
        if not S.is_int(salt) or not 0 <= salt < 2 ** 63:
            raise S.SchemaError("id_salt: must be an integer in 0..2**63-1")
        self.salt = salt

    def fingerprint(self) -> str:
        return f"{self.name}:{self.salt}"

    def assign(self, books: Books, ledger_rows: list, rng: Optional[DetRng] = None) -> Layout:
        if rng is None:
            raise GeneratorError("RandomizedIdScheme needs the episode RNG")
        base = rng.fork(f"randomized/{self.salt}")
        ids = _HexIds(base.fork("ids"))

        ledger = list(ledger_rows)
        base.fork("order/ledger").shuffle(ledger)
        invoices = list(books.invoices)
        base.fork("order/invoices").shuffle(invoices)
        payments = list(books.payments)
        base.fork("order/payments").shuffle(payments)
        fx = [dict(r) for r in books.fx]
        base.fork("order/fx").shuffle(fx)

        for row in ledger:
            row.row_id = f"L-{ids.draw('row')}"
        documents = invoices + payments
        base.fork("order/doc-ids").shuffle(documents)
        for doc in documents:
            doc.doc_id = f"DOC-{ids.draw('doc')}"
        for inv in invoices:
            if inv.alias_of is None:
                prefix = "INV" if inv.kind == "invoice" else "CN"
                inv.invoice_id = f"{prefix}-{ids.draw('invoice_number')}"
        for inv in invoices:
            if inv.alias_of is not None:
                inv.invoice_id = inv.alias_of.invoice_id
        for pay in payments:
            pay.payment_id = f"PAY-{ids.draw('payment_number')}"

        entity_map = {}
        for party in books.customers:
            entity_map[party.entity_id] = f"CUST-{ids.draw('entity')}"
        for supplier in books.suppliers:
            entity_map[supplier] = f"SUP-{ids.draw('entity')}"
        return Layout(ledger, invoices, payments, fx=fx, entity_map=entity_map)


def scheme_from_config(cfg: S.EpisodeConfig) -> IdScheme:
    if cfg.id_scheme == "randomized":
        return RandomizedIdScheme(cfg.id_salt)
    return SequentialIdScheme()


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
    if isinstance(value, FxRef):
        return {"table": "fx", "id": value.fx_id}
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
            "rule": "local_calendar_date_of_posting_timestamp",
            "timezone": cfg.timezone,
            "fallback": "latest_earlier_available_date",
            "description": "The ledger is kept in the base currency. A foreign-currency invoice is "
                           "translated at the fx.csv rate of the company-local calendar date of its "
                           "posting; if that date has no rate (e.g. weekend), use the latest earlier "
                           "date. Rows in fx.csv are dated at local midnight; rate = base major units "
                           "per one foreign major unit.",
            "settlement": "Cash received is translated at the rate of the payment date. Credit notes "
                          "and payment allocations relieve receivables at the historical rate of the "
                          "invoice they refer to; the allocation that settles an invoice completely "
                          "relieves exactly its remaining base balance. The difference between cash "
                          "and relieved receivables is posted to account 4900.",
        },
        "rounding_rule": {
            "mode": S.ROUNDING_RULE,
            "unit": "minor unit of the target currency",
            "applied": "once, after the full-precision conversion",
        },
        "amount_tolerance_minor": cfg.amount_tolerance_minor,
        "ledger_conventions": {
            "currency": "All ledger rows are in the base currency; debits equal credits. "
                        "Documents keep their own currency.",
            "payments": "A payment allocation is posted as a credit to receivables (1100) with the "
                        "invoice_id; cash not applied to an invoice is posted to 2100.",
            "documents": "invoices.jsonl holds invoices and credit notes (CN-*, negative "
                         "total_minor). An invoice dated outside the period without ledger "
                         "postings is correctly excluded from this period.",
            "references": "Missing references are null (empty cell in ledger.csv).",
        },
        "chart_of_accounts": dict(S.CHART_OF_ACCOUNTS),
    }


def _id_registry(layout: Layout, books: Books) -> dict:
    """public id -> internal record key (identical across id schemes of one history)."""
    def entity(name: Optional[str]) -> Optional[str]:
        return None if name is None else layout.entity_map.get(name, name)

    invoice_numbers = {}
    for inv in layout.invoices:
        owner = inv.alias_of if inv.alias_of is not None else inv
        invoice_numbers[inv.invoice_id] = f"invoice_number#{owner.order}"
    entities = {entity(p.entity_id): f"entity#{p.entity_id}" for p in books.customers}
    entities.update({entity(s): f"entity#{s}" for s in books.suppliers})
    return {
        "ledger_rows": {r.row_id: f"row#{r.seq}" for r in layout.ledger},
        "documents": {**{i.doc_id: f"invoice#{i.order}" for i in layout.invoices},
                      **{p.doc_id: f"payment#{p.order}" for p in layout.payments}},
        "invoice_ids": invoice_numbers,
        "payment_ids": {p.payment_id: f"payment_number#{p.order}" for p in layout.payments},
        "entities": entities,
    }


def _finalize(seed: int, cfg: S.EpisodeConfig, episode_id: str, books: Books, issues: list,
              plan: list, clean_checks: dict, scheme: IdScheme,
              id_rng: Optional[DetRng] = None) -> Episode:
    ctx = books.ctx
    rows, index = build_ledger(books)
    layout = scheme.assign(books, rows, id_rng)
    emap = layout.entity_map

    def entity(name: Optional[str]) -> Optional[str]:
        return None if name is None else emap.get(name, name)

    ledger = [{
        "row_id": r.row_id, "timestamp": _fmt(r.ts, r.offset), "account": r.account,
        "debit_minor": r.debit, "credit_minor": r.credit, "currency": r.currency,
        "entity_id": entity(r.entity_id),
        "invoice_id": r.invoice.invoice_id if r.invoice is not None else None,
        "payment_id": r.payment.payment_id if r.payment is not None else None,
    } for r in layout.ledger]
    invoices = [{
        "doc_id": inv.doc_id, "invoice_id": inv.invoice_id, "entity_id": entity(inv.entity_id),
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
    fx = [dict(r) for r in (layout.fx if layout.fx is not None else books.fx)]

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
        "id_scheme": scheme.fingerprint(),
        "error_plan": list(plan),
        "company": {
            "customers": [{"entity_id": entity(p.entity_id), "currency": p.currency}
                          for p in books.customers],
            "suppliers": [entity(s) for s in books.suppliers],
        },
        "stories": [{"story": s["story"], "kind": s["kind"], "tags": s["tags"],
                     "records": _resolve(s["records"], index)} for s in books.story_log],
        "clean_books_checks": clean_checks,
        "mutations": mutations,
        "final_counts": manifest["counts"],
        "id_registry": _id_registry(layout, books),
    }
    return Episode(episode_id, manifest, policy, ledger, invoices, payments, fx, truth, provenance)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_episode(seed: int, config: Any = None, *, id_scheme: Optional[IdScheme] = None) -> Episode:
    """Generate one episode.

    Same (seed, config, GENERATOR_VERSION) -> identical bytes.

    The *history* (records, amounts, errors) depends only on the seed and the
    history fields of the config.  The id scheme - ``id_scheme`` argument, or
    ``config.id_scheme``/``config.id_salt`` - only reorders rows and renames
    ids.  ``manifest.episode_id`` is derived from the history material plus
    the scheme fingerprint: the sequential scheme keeps the Stage 1 id, and
    every randomized variant gets its own id, so an agent cannot recognise
    two presentations of the same history by their episode id.
    """
    S.validate_seed(seed)
    cfg = S.coerce_config(config)
    unknown = [t for t in cfg.error_types if t not in MUTATORS]
    if unknown:
        raise S.SchemaError(f"config.error_types: unsupported types {unknown}")
    scheme = id_scheme if id_scheme is not None else scheme_from_config(cfg)
    if not isinstance(scheme, IdScheme):
        raise S.SchemaError("id_scheme: must be an IdScheme instance")

    material = f"{S.GENERATOR_VERSION}|{seed}|{cfg.history_json()}"
    fingerprint = scheme.fingerprint()
    id_material = material if fingerprint == SequentialIdScheme.name else f"{material}|ids={fingerprint}"
    episode_id = "EP-" + hashlib.sha256(id_material.encode("utf-8")).hexdigest()[:12].upper()
    root = DetRng(material)
    ctx = PeriodCtx.from_config(cfg)
    books = Books(config=cfg, ctx=ctx)

    plan_rng = root.fork("plan")
    types = list(cfg.error_types)
    plan_rng.shuffle(types)
    plan = [types[i % len(types)] for i in range(cfg.n_errors)] if cfg.n_errors else []
    plan_rng.shuffle(plan)
    plan += [f"scenario:{name}" for name in cfg.interactions]
    counts = {name: plan.count(name) for name in sorted(set(plan))}

    _build_parties(books, root.fork("parties"))
    _build_clean_stories(books, root.fork("stories"), counts)
    _build_fx(books, root.fork("fx"))
    books.translator = Translator(books)
    clean_checks = _check_books(books, allow_mutations=False)   # before any error exists

    mutation_rng = root.fork("mutations")
    issues = []
    for i, name in enumerate(plan):
        result = MUTATORS[name].apply(books, mutation_rng.fork(str(i)), i)
        issues.extend(result if isinstance(result, list) else [result])
    _build_filler(books, root.fork("filler"))
    clean_checks["after_errors"] = _check_books(books, allow_mutations=True)
    return _finalize(seed, cfg, episode_id, books, issues, plan, clean_checks,
                     scheme, root.fork("ids"))
