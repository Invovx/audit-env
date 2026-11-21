"""episode_io.py - read an episode directory written by ``generator.write_episode``
(or prepared by the jury from one: rows permuted, ids remapped).

Layout accepted (both are documented in the README):

    <episode_dir>/public/{manifest.json, policy.json, ledger.csv, invoices.jsonl,
                          payments.jsonl, fx.csv}
    <episode_dir>/private/{truth.json, provenance.json}

or the same files directly in ``<episode_dir>`` (flat layout).

Ids are treated as opaque strings: no format is assumed, so remapped copies load
exactly like originals.  The public loader never touches ``truth.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import schemas as S

PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl",
                "payments.jsonl", "fx.csv")


class EpisodeDirError(S.SchemaError):
    """The directory does not hold a readable episode."""


def _public_dir(episode_dir: Any) -> Path:
    root = Path(episode_dir)
    for candidate in (root / "public", root):
        if (candidate / "manifest.json").is_file():
            return candidate
    raise EpisodeDirError("episode_dir: manifest.json not found")


def _private_dir(episode_dir: Any) -> Path:
    root = Path(episode_dir)
    for candidate in (root / "private", root):
        if (candidate / "truth.json").is_file():
            return candidate
    raise EpisodeDirError("episode_dir: truth.json not found")


def load_public(episode_dir: Any) -> dict:
    """Public tables of an episode: manifest, policy, ledger, invoices, payments, fx."""
    pub = _public_dir(episode_dir)
    missing = [name for name in PUBLIC_FILES if not (pub / name).is_file()]
    if missing:
        raise EpisodeDirError(f"episode_dir: missing public files {missing}")
    read = lambda name: (pub / name).read_text(encoding="utf-8")  # noqa: E731
    try:
        data = {
            "manifest": json.loads(read("manifest.json")),
            "policy": json.loads(read("policy.json")),
            "ledger": S.parse_ledger_csv(read("ledger.csv")),
            "invoices": S.parse_jsonl(read("invoices.jsonl")),
            "payments": S.parse_jsonl(read("payments.jsonl")),
            "fx": S.parse_fx_csv(read("fx.csv")),
        }
    except json.JSONDecodeError:
        raise EpisodeDirError("episode_dir: invalid JSON in a public file") from None
    if not isinstance(data["manifest"], dict) or not isinstance(data["policy"], dict):
        raise EpisodeDirError("episode_dir: manifest and policy must be objects")
    return data


def load_truth(episode_dir: Any) -> list:
    """Closed reference errors (evaluator side only)."""
    truth = json.loads((_private_dir(episode_dir) / "truth.json").read_text(encoding="utf-8"))
    if not isinstance(truth, list):
        raise EpisodeDirError("truth.json: must be a list")
    return truth


def public_ids(data: dict) -> frozenset:
    """Every id an agent can reach with the tools: ledger row_id, document doc_id
    (invoices and payments) and the content-derived fx id."""
    ids = {row["row_id"] for row in data["ledger"]}
    ids |= {doc["doc_id"] for doc in data["invoices"]}
    ids |= {doc["doc_id"] for doc in data["payments"]}
    ids |= {S.fx_record_id(row) for row in data["fx"]}
    return frozenset(ids)


def tolerance(data: dict) -> int:
    """Published amount tolerance of the episode (policy.json), default exact."""
    value = data["policy"].get("amount_tolerance_minor", 0)
    if not S.is_int(value) or value < 0:
        raise EpisodeDirError("policy.amount_tolerance_minor: must be a non-negative int")
    return value
