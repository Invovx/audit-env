"""Contract, isolation and reward tests for the audit simulator (Stage 1)."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import safe_calc  # noqa: E402
import schemas as S  # noqa: E402
from env import AuditEnv  # noqa: E402
from evaluator import Evaluator, compute_reward  # noqa: E402
from generator import generate_episode, write_episode  # noqa: E402

DIRTY = {"n_ledger_rows": 400, "n_documents": 48, "n_errors": 4}
CLEAN = {"n_ledger_rows": 400, "n_documents": 48, "n_errors": 0}
SEEDS = (1, 2, 3, 5, 8)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def finding_from_truth(entry: dict, **overrides) -> dict:
    out = {
        "type": entry["type"],
        "affected_ids": list(entry["affected_ids"]),
        "claimed_amount_minor": entry["expected_amount_minor"],
        "currency": entry["currency"],
        "evidence_ids": list(entry["valid_evidence_sets"][0]),
        "explanation": "proved by the listed records",
    }
    out.update(overrides)
    return out


def perfect_findings(episode) -> list:
    return [finding_from_truth(t) for t in episode.truth]


def false_accusation(episode) -> dict:
    row = episode.ledger[0]["row_id"]
    doc = episode.invoices[0]["doc_id"]
    return {"type": "duplicate_invoice", "affected_ids": [doc, episode.invoices[1]["doc_id"]],
            "claimed_amount_minor": 12345, "currency": episode.manifest["base_currency"],
            "evidence_ids": [doc, row], "explanation": "made up"}


def submit_action(episode_id: str, findings: list) -> dict:
    return {"tool": "submit", "findings": {"episode_id": episode_id, "findings": findings}}


INSPECTION = [{"tool": "list_records", "table": "ledger", "limit": 5},
              {"tool": "calculator", "expression": "1+1"},
              {"tool": "list_records", "table": "invoices", "limit": 5}]


def run_submission(seed, config, findings_fn, mode="train", inspection=INSPECTION):
    episode = generate_episode(seed, config)
    env = AuditEnv(mode)
    env.reset(seed, config)
    for action in inspection:
        env.step(action)
    out = env.step(submit_action(episode.episode_id, findings_fn(episode)))
    return env, out


def parse_ts(value: str):
    return S.parse_timestamp(value)


# --------------------------------------------------------------------------- #
# 1. determinism
# --------------------------------------------------------------------------- #
def test_generator_is_bitwise_deterministic():
    a = generate_episode(77, DIRTY)
    b = generate_episode(77, DIRTY)
    assert a.public_files() == b.public_files()
    assert a.private_files() == b.private_files()
    assert generate_episode(78, DIRTY).public_files() != a.public_files()
    other = dict(DIRTY, n_errors=3)
    assert generate_episode(77, other).public_files() != a.public_files()


def test_reset_and_full_run_are_reproducible():
    script = [
        {"tool": "list_records", "table": "ledger", "limit": 50},
        {"tool": "list_records", "table": "fx", "limit": 7},
        {"tool": "get_record", "table": "ledger", "id": "L000001"},
        {"tool": "get_document", "doc_id": "DOC-00001"},
        {"tool": "calculator", "expression": "485.2500 * 1000"},
        {"tool": "no_such_tool"},
        {"tool": "get_record", "table": "ledger", "id": "nope"},
    ]

    def transcript(mode):
        env = AuditEnv(mode)
        log = [env.reset(42, DIRTY)]
        for action in script:
            log.append(env.step(action))
        episode = generate_episode(42, DIRTY)
        log.append(env.step(submit_action(episode.episode_id, perfect_findings(episode))))
        return json.dumps(log, sort_keys=True)

    assert transcript("train") == transcript("train")
    assert transcript("eval") == transcript("eval")


def test_reset_fully_isolates_episode_state():
    env_a, env_b = AuditEnv(), AuditEnv()
    obs_a, _ = env_a.reset(1, DIRTY)
    env_b.reset(2, CLEAN)
    for _ in range(10):
        env_b.step({"tool": "list_records", "table": "ledger", "limit": 3})
    obs_a2, info = env_a.reset(1, DIRTY)
    assert obs_a == obs_a2 and info["steps_remaining"] == 100
    out = env_a.step({"tool": "list_records", "table": "ledger", "limit": 1})
    assert out[0]["steps_remaining"] == 99                      # untouched by env_b
    first = env_a.step({"tool": "get_record", "table": "ledger", "id": "L000001"})[0]["record"]
    first["debit_minor"] = -1                                     # mutating a copy ...
    again = env_a.step({"tool": "get_record", "table": "ledger", "id": "L000001"})[0]["record"]
    assert again["debit_minor"] != -1                             # ... must not affect state


def test_first_observation_contract():
    obs, info = AuditEnv().reset(5, DIRTY)
    assert obs["steps_remaining"] == 100
    assert set(obs) >= {"task", "manifest", "policy_summary", "steps_remaining"}
    assert list(obs["manifest"]) == list(S.MANIFEST_KEYS)
    assert "amount_tolerance_minor" in obs["policy_summary"]
    assert json.loads(json.dumps(obs)) == obs                     # strictly JSON-compatible
    assert json.loads(json.dumps(info)) == info


# --------------------------------------------------------------------------- #
# generator contracts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("config", [
    {"n_ledger_rows": 200, "n_documents": 20, "n_errors": 0},
    {"n_ledger_rows": 200, "n_documents": 20, "n_errors": 5},
    {"n_ledger_rows": 600, "n_documents": 60, "n_errors": 12},
    {"n_ledger_rows": 2000, "n_documents": 200, "n_errors": 12},
    {"n_ledger_rows": 777, "n_documents": 93, "n_errors": 7, "timezone": "-03:30",
     "base_currency": "USD", "foreign_currencies": ["EUR", "KZT"]},
])
@pytest.mark.parametrize("seed", [0, 11, 12345])
def test_generated_files_follow_the_contract(seed, config):
    ep = generate_episode(seed, config)
    cfg = S.EpisodeConfig.from_dict(config)
    m = ep.manifest
    assert m["counts"]["ledger_rows"] == cfg.n_ledger_rows == len(ep.ledger)
    assert m["counts"]["documents"] == cfg.n_documents
    assert len(ep.truth) == cfg.n_errors
    assert list(m) == list(S.MANIFEST_KEYS)
    assert m["currency_minor_units"][m["base_currency"]] == 2
    S.validate_public(ep.manifest, ep.policy, ep.ledger, ep.invoices, ep.payments, ep.fx)

    for row in ep.ledger:                                   # money is int, no float anywhere
        assert type(row["debit_minor"]) is int and type(row["credit_minor"]) is int
    for doc in ep.invoices:
        assert type(doc["total_minor"]) is int
        S.parse_timestamp(doc["issued_at"]), S.parse_timestamp(doc["due_at"])
    for doc in ep.payments:
        assert type(doc["amount_minor"]) is int
    for row in ep.fx:
        assert isinstance(row["rate"], str) and re.fullmatch(r"\d+\.\d{4}", row["rate"])

    totals = {}
    for row in ep.ledger:
        d, c = totals.get(row["currency"], (0, 0))
        totals[row["currency"]] = (d + row["debit_minor"], c + row["credit_minor"])
    assert all(d == c for d, c in totals.values())          # balanced per currency

    def no_floats(node):
        if isinstance(node, float):
            raise AssertionError("float found in public data")
        if isinstance(node, dict):
            for v in node.values():
                no_floats(v)
        elif isinstance(node, list):
            for v in node:
                no_floats(v)

    no_floats([ep.ledger, ep.invoices, ep.payments, ep.fx, ep.manifest, ep.policy])


def test_files_on_disk_isolation_and_null_mapping(tmp_path):
    ep = generate_episode(9, DIRTY)
    public_dir, private_dir = write_episode(ep, tmp_path)
    assert sorted(p.name for p in public_dir.iterdir()) == sorted(
        ["manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"])
    assert sorted(p.name for p in private_dir.iterdir()) == ["provenance.json", "truth.json"]
    assert public_dir.parent == private_dir.parent and public_dir != private_dir
    for p in public_dir.iterdir():
        low = p.read_text(encoding="utf-8").lower()
        assert "iss-" not in low and "truth" not in low and "provenance" not in low

    ledger = S.parse_ledger_csv((public_dir / "ledger.csv").read_text(encoding="utf-8"))
    assert ledger == ep.ledger                                # empty cell <-> None round trip
    assert any(r["invoice_id"] is None for r in ledger) and any(r["payment_id"] is None for r in ledger)
    assert S.parse_jsonl((public_dir / "invoices.jsonl").read_text(encoding="utf-8")) == ep.invoices
    assert S.parse_jsonl((public_dir / "payments.jsonl").read_text(encoding="utf-8")) == ep.payments
    assert S.parse_fx_csv((public_dir / "fx.csv").read_text(encoding="utf-8")) == ep.fx
    header = (public_dir / "ledger.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id"
    truth = json.loads((private_dir / "truth.json").read_text(encoding="utf-8"))
    assert truth == ep.truth
    prov = json.loads((private_dir / "provenance.json").read_text(encoding="utf-8"))
    assert prov["mutations"] and prov["clean_books_checks"]["balanced_by_currency"]


def _truth_is_semantically_true(ep) -> None:
    inv = {d["doc_id"]: d for d in ep.invoices}
    ledger = {r["row_id"]: r for r in ep.ledger}
    known = set(inv) | {d["doc_id"] for d in ep.payments} | set(ledger)
    start, end = parse_ts(ep.manifest["period_start"]), parse_ts(ep.manifest["period_end"])
    policy_tol = ep.policy["amount_tolerance_minor"]
    assert isinstance(policy_tol, int)

    cutoff_ids, dup_groups = set(), []
    for t in ep.truth:
        assert t["valid_evidence_sets"] and all(set(s) <= known for s in t["valid_evidence_sets"])
        assert set(t["affected_ids"]) <= known
        if t["type"] == "duplicate_invoice":
            a, b = (inv[i] for i in t["affected_ids"])
            for key in ("entity_id", "issued_at", "due_at", "currency", "total_minor", "line_items"):
                assert key == "line_items" or a[key] == b[key]
            assert [l["amount_minor"] for l in a["line_items"]] == [l["amount_minor"] for l in b["line_items"]]
            assert t["expected_amount_minor"] == a["total_minor"] and t["currency"] == a["currency"]
            dup_groups.append(set(t["affected_ids"]))
            for ev in t["valid_evidence_sets"]:
                if ev[0] in ledger:
                    rows = [ledger[i] for i in ev]
                    assert len({r["invoice_id"] for r in rows}) == 1
        else:
            doc = inv[t["affected_ids"][0]]
            assert not (start <= parse_ts(doc["issued_at"]) <= end)
            rows = [r for r in ep.ledger if r["invoice_id"] == doc["invoice_id"]]
            assert rows and all(start <= parse_ts(r["timestamp"]) <= end for r in rows)
            assert t["expected_amount_minor"] == doc["total_minor"]
            cutoff_ids.add(doc["doc_id"])
            for ev in t["valid_evidence_sets"]:
                assert doc["doc_id"] in ev and any(i in ledger for i in ev)

    # no *other* out-of-period invoice is booked, no other duplicate exists
    booked = {r["invoice_id"] for r in ep.ledger if r["invoice_id"]}
    outside_booked = {d["doc_id"] for d in ep.invoices
                      if d["invoice_id"] in booked
                      and not (start <= parse_ts(d["issued_at"]) <= end)}
    assert outside_booked == cutoff_ids
    groups: dict = {}
    for d in ep.invoices:
        groups.setdefault((d["entity_id"], d["issued_at"], d["currency"], d["total_minor"]), set()).add(d["doc_id"])
    assert sorted(map(sorted, (g for g in groups.values() if len(g) > 1))) == sorted(map(sorted, dup_groups))


@pytest.mark.parametrize("seed", range(8))
def test_truth_is_backed_by_causal_provenance_and_data(seed):
    ep = generate_episode(seed, {"n_ledger_rows": 500, "n_documents": 70, "n_errors": 6})
    _truth_is_semantically_true(ep)
    prov = ep.provenance
    assert len(prov["mutations"]) == len(ep.truth) == 6
    assert {m["type"] for m in prov["mutations"]} == {"duplicate_invoice", "period_cutoff_error"}
    for m, t in zip(prov["mutations"], ep.truth):
        assert m["issue_id"] == t["issue_id"] and m["ops"] and m["reason"]
    assert prov["seed"] == seed


def test_clean_books_have_no_truth_but_contain_legitimate_distractors():
    kinds = set()
    for seed in range(6):
        ep = generate_episode(seed, {"n_ledger_rows": 900, "n_documents": 150, "n_errors": 0})
        assert ep.truth == [] and ep.provenance["mutations"] == []
        _truth_is_semantically_true(ep)
        start, end = parse_ts(ep.manifest["period_start"]), parse_ts(ep.manifest["period_end"])
        pay_count: dict = {}
        for p in ep.payments:
            for ref in p["invoice_refs"]:
                pay_count[ref] = pay_count.get(ref, 0) + 1
            if len(p["invoice_refs"]) > 1:
                kinds.add("combined_payment")
        if any(c > 1 for c in pay_count.values()):
            kinds.add("split_payment")
        if any(d["total_minor"] < 0 and d["invoice_id"].startswith("CN-") for d in ep.invoices):
            kinds.add("return")
        totals = {}
        for d in ep.invoices:
            totals.setdefault((d["entity_id"], d["total_minor"]), set()).add(d["issued_at"])
        if any(len(v) > 1 for v in totals.values()):
            kinds.add("repeated_amount")
        if any(parse_ts(d["issued_at"]) in (start, end) for d in ep.invoices):
            kinds.add("boundary")
        if any(not (start <= parse_ts(d["issued_at"]) <= end) for d in ep.invoices):
            kinds.add("out_of_period_unbooked")
        if any(d["issued_at"].endswith("+00:00") and ep.manifest["timezone"] != "+00:00" for d in ep.invoices):
            kinds.add("utc_rendering")
    assert kinds == {"combined_payment", "split_payment", "return", "repeated_amount",
                     "boundary", "out_of_period_unbooked", "utc_rendering"}


def test_invalid_configs_are_rejected():
    for bad in ({"n_documents": 19}, {"n_documents": 201}, {"n_ledger_rows": 199},
                {"n_ledger_rows": 2001}, {"n_errors": 13}, {"n_errors": -1}, {"unknown": 1},
                {"n_errors": True}, {"timezone": "Asia/Almaty"}, {"error_types": ["nope"]},
                {"n_documents": 40, "n_errors": 11}):
        with pytest.raises(ValueError):
            generate_episode(1, bad)
    with pytest.raises(ValueError):
        generate_episode(True, None)


# --------------------------------------------------------------------------- #
# 2. no leaks
# --------------------------------------------------------------------------- #
def _collect_everything(mode: str, tmp_path, seed: int):
    episode = generate_episode(seed, DIRTY)
    write_episode(episode, tmp_path)                      # paths exist on disk during the run
    env = AuditEnv(mode)
    log = [env.reset(seed, DIRTY)]
    actions = [{"tool": "list_records", "table": t, "limit": 50} for t in S.TABLES]
    actions += [{"tool": "get_record", "table": "ledger", "id": episode.ledger[3]["row_id"]},
                {"tool": "get_record", "table": "fx", "id": "FX-NOPE"},
                {"tool": "get_document", "doc_id": episode.invoices[0]["doc_id"]},
                {"tool": "get_document", "doc_id": "../../x/y.json"},
                {"tool": "calculator", "expression": "__import__('os').getcwd()"},
                {"tool": "calculator", "expression": "1/0"},
                {"tool": "calculator", "expression": "open('x/y.json')"},
                {"tool": "frobnicate", "path": str(tmp_path)},
                {"tool": "list_records", "table": "../x", "limit": 3},
                "garbage", None]
    for a in actions:
        log.append(env.step(a))
    log.append(env.step(submit_action(episode.episode_id, perfect_findings(episode))))
    return episode, env, log


def _all_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _all_strings(k)
            yield from _all_strings(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _all_strings(v)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_no_leak_of_truth_seed_or_paths(mode, tmp_path):
    seed = 918273645
    episode, env, log = _collect_everything(mode, tmp_path, seed)
    blob = json.dumps(log)
    low = blob.lower()
    for word in ("truth", "provenance", "seed", "private", "mutation"):
        assert word not in low
    assert str(seed) not in blob
    assert str(tmp_path) not in blob and str(tmp_path).replace("\\", "\\\\") not in blob
    strings = list(_all_strings(log))
    assert strings
    for text in strings:                                  # real strings, not JSON-escaped text
        assert not re.search(r"[A-Za-z]:[\\/]", text) and "/tmp" not in text
        assert "/home/" not in text and "/Users/" not in text and str(tmp_path) not in text
    for t in episode.truth:
        assert t["issue_id"] not in blob
    for _, _, _, _, info in log[1:]:
        assert set(info) == {"mode", "step", "steps_remaining", "diagnostics"}
    for _, info in [log[0]]:
        assert set(info) == {"mode", "step", "steps_remaining", "diagnostics"}


def test_evaluator_module_is_isolated_from_generator_and_env():
    tree = ast.parse((ROOT / "evaluator.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "copy", "json", "dataclasses", "pathlib", "typing", "schemas"}
    env_src = (ROOT / "env.py").read_text(encoding="utf-8")
    assert "truth.json" not in env_src and "provenance" not in env_src


def test_eval_mode_keeps_reward_and_diagnostics_inside_the_evaluator():
    env_t, out_t = run_submission(3, DIRTY, perfect_findings, "train")
    env_e, out_e = run_submission(3, DIRTY, perfect_findings, "eval")
    obs, reward, terminated, truncated, info = out_e
    assert (reward, terminated, truncated, info["diagnostics"]) == (0.0, True, False, {})
    assert out_t[1] > 0.9 and out_t[4]["diagnostics"]["f1"] == 1.0
    assert env_e.evaluator.final_result()["reward"] == out_t[1]
    assert env_e.evaluator.final_result()["f1"] == 1.0


# --------------------------------------------------------------------------- #
# 3. calculator safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("expr, expected", [
    ("0.1 + 0.2", "0.3"), ("485.2500 * 1000", "485250.0000"), ("10 / 4", "2.5"),
    ("2 ** 10", "1024"), ("-(3 + 4) * 2", "-14"), ("7 // 2", "3"), ("7 % 4", "3"),
    ("-2 ** 2", "-4"), ("1e3 + 1", "1001"), ("(1_000 + 1) * 2", "2002"), ("2 ** -2", "0.25"),
    ("123456789012345678901234567890 + 1", "123456789012345678901234567891"),
])
def test_calculator_exact_decimal_arithmetic(expr, expected):
    assert safe_calc.evaluate(expr) == expected


def test_calculator_never_uses_float():
    assert safe_calc.evaluate("0.1 * 3") == "0.3"
    assert safe_calc.evaluate("1.10 + 2.20") == "3.30"
    assert str(safe_calc.safe_eval("1/3")).startswith("0.3333333333")


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "__import__('os')", "import os", "os.system('ls')",
    "exec('1')", "eval('1')", "open('x')", "().__class__", "[1, 2]", "{1: 2}", "'a' * 5",
    "1 if 1 else 2", "lambda: 1", "x + 1", "True + 1", "None", "1 < 2", "not 1", "~1", "1 << 2",
    "1j", "0x10", "1e999", "10 ** 1000", "2 ** 0.5", "2 ** 1000", "", "   ", "(", "1 +", "1 2",
    "1/0", "1//0", "1%0", "0/0", "0 ** -1", "9" * 300, "(" * 150 + "1" + ")" * 150, "1;2",
    "print(1)", "a.b", "1\n+2\n__import__('os')",
])
def test_calculator_rejects_unsafe_or_invalid_expressions(expr):
    with pytest.raises(safe_calc.CalcError):
        safe_calc.evaluate(expr)


def test_calculator_rejects_non_strings():
    for bad in (5, None, ["1+1"], b"1+1"):
        with pytest.raises(safe_calc.CalcError):
            safe_calc.evaluate(bad)  # type: ignore[arg-type]


def test_calculator_errors_through_env_cost_one_step_and_do_not_crash():
    env = AuditEnv()
    env.reset(1, CLEAN)
    remaining = 100
    for expr in ("1/0", "__import__('os')", "import os", "2 +", ""):
        obs, reward, terminated, truncated, info = env.step({"tool": "calculator", "expression": expr})
        remaining -= 1
        assert obs["kind"] == "error" and obs["error"]["code"] == "calculator_error"
        assert len(obs["error"]["message"]) <= 200
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert obs["steps_remaining"] == remaining == info["steps_remaining"]
    obs = env.step({"tool": "calculator", "expression": "485.25 * 2"})[0]
    assert obs["kind"] == "calculation" and obs["result"] == "970.50"


# --------------------------------------------------------------------------- #
# environment contract
# --------------------------------------------------------------------------- #
def test_list_records_pagination_limits_and_stable_cursors():
    env = AuditEnv()
    env.reset(4, DIRTY)
    big = env.step({"tool": "list_records", "table": "ledger", "limit": 1000})[0]
    assert len(big["records"]) == 50 and big["has_more"] and big["next_cursor"]
    default = env.step({"tool": "list_records", "table": "ledger"})[0]
    assert default["records"] == big["records"]

    for table, expected in (("ledger", 400), ("invoices", None), ("payments", None), ("fx", None)):
        seen, cursor = [], None
        while True:
            obs = env.step({"tool": "list_records", "table": table, "cursor": cursor, "limit": 37})[0]
            assert obs["kind"] == "records" and len(obs["records"]) <= 37
            again = env.step({"tool": "list_records", "table": table, "cursor": cursor, "limit": 37})[0]
            assert again["records"] == obs["records"] and again["next_cursor"] == obs["next_cursor"]
            seen.extend(obs["records"])
            cursor = obs["next_cursor"]
            if cursor is None:
                assert not obs["has_more"]
                break
        key = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "fx_id"}[table]
        ids = [r[key] for r in seen]
        assert len(ids) == len(set(ids))
        if expected:
            assert len(ids) == expected
    # a cursor is only valid for its own table
    cur = env.step({"tool": "list_records", "table": "ledger", "limit": 2})[0]["next_cursor"]
    assert env.step({"tool": "list_records", "table": "fx", "cursor": cur})[0]["kind"] == "error"
    assert env.step({"tool": "list_records", "table": "ledger", "cursor": "ledger:zzz"})[0]["kind"] == "error"


def test_get_record_and_get_document():
    ep = generate_episode(6, DIRTY)
    env = AuditEnv()
    env.reset(6, DIRTY)
    row = env.step({"tool": "get_record", "table": "ledger", "id": ep.ledger[10]["row_id"]})[0]
    assert row["record"] == ep.ledger[10]
    fx = env.step({"tool": "get_record", "table": "fx", "id": S.fx_record_id(ep.fx[0])})[0]
    assert fx["record"]["rate"] == ep.fx[0]["rate"]
    doc = env.step({"tool": "get_document", "doc_id": ep.payments[0]["doc_id"]})[0]
    assert doc["doc_type"] == "payment" and doc["document"] == ep.payments[0]
    inv = env.step({"tool": "get_document", "doc_id": ep.invoices[0]["doc_id"]})[0]
    assert inv["doc_type"] == "invoice" and inv["document"] == ep.invoices[0]
    inv2 = env.step({"tool": "get_record", "table": "invoices", "id": ep.invoices[0]["doc_id"]})[0]
    assert inv2["record"] == ep.invoices[0]


BAD_ACTIONS = [
    None, 5, "list_records", [], {}, {"tool": 7}, {"tool": "nope"}, {"tool": "list_records"},
    {"tool": "list_records", "table": "secrets"}, {"tool": "list_records", "table": ["ledger"]},
    {"tool": "list_records", "table": "ledger", "limit": 0},
    {"tool": "list_records", "table": "ledger", "limit": True},
    {"tool": "list_records", "table": "ledger", "limit": "5"},
    {"tool": "list_records", "table": "ledger", "cursor": 5},
    {"tool": "list_records", "table": "ledger", "extra": 1},
    {"tool": "get_record", "table": "ledger"}, {"tool": "get_record", "table": "ledger", "id": 1},
    {"tool": "get_record", "table": "ledger", "id": "x" * 5000},
    {"tool": "get_document"}, {"tool": "get_document", "doc_id": ["a"]},
    {"tool": "get_document", "doc_id": "DOC-99999"},
    {"tool": "calculator"}, {"tool": "calculator", "expression": 5},
]


def test_invalid_actions_cost_one_step_and_never_crash():
    env = AuditEnv()
    env.reset(2, CLEAN)
    remaining = 100
    for action in BAD_ACTIONS:
        obs, reward, terminated, truncated, info = env.step(action)
        remaining -= 1
        assert obs["kind"] == "error", action
        assert 0 < len(obs["error"]["message"]) <= 200 and len(json.dumps(obs)) < 1000
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert obs["steps_remaining"] == remaining == info["steps_remaining"]
        json.dumps(obs)


def test_exploration_steps_always_give_zero_reward_in_both_modes():
    for mode in ("train", "eval"):
        env = AuditEnv(mode)
        env.reset(3, DIRTY)
        for action in INSPECTION + BAD_ACTIONS[:5]:
            _, reward, terminated, truncated, info = env.step(action)
            assert reward == 0.0 and not terminated and not truncated and info["diagnostics"] == {}


def test_only_five_tools_are_accepted():
    env = AuditEnv()
    env.reset(3, CLEAN)
    for tool in ("search", "run_sql", "reset", "state", "LIST_RECORDS", "submit_findings"):
        assert env.step({"tool": tool})[0]["error"]["code"] == "unknown_tool"


# --------------------------------------------------------------------------- #
# 4. reward properties and step budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_perfect_answer_beats_silence_on_erroneous_books(seed):
    _, perfect = run_submission(seed, DIRTY, perfect_findings)
    _, silent = run_submission(seed, DIRTY, lambda ep: [])
    assert perfect[2] and not perfect[3] and silent[2] and not silent[3]
    assert perfect[1] > silent[1]
    assert silent[1] < -0.9                                  # F1 = 0 -> about -1
    expected = compute_reward(1.0, len(INSPECTION))
    assert perfect[1] == pytest.approx(expected) and expected == pytest.approx(1 - 0.002 * 3)


@pytest.mark.parametrize("seed", SEEDS)
def test_silence_beats_false_accusations_on_clean_books(seed):
    _, silent = run_submission(seed, CLEAN, lambda ep: [])
    _, accuse = run_submission(seed, CLEAN, lambda ep: [false_accusation(ep)])
    assert silent[1] == pytest.approx(1 - 0.002 * 3)
    assert accuse[1] == -1.0                                  # F1 = 0, clamped at the floor
    assert silent[1] > accuse[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_duplicate_findings_never_raise_the_reward(seed):
    ep = generate_episode(seed, DIRTY)
    assert len(ep.truth) >= 2
    one = [finding_from_truth(ep.truth[0])]
    cases = {
        "one_then_dup": (lambda e: one, lambda e: one + one + one),
        "perfect_then_dup": (perfect_findings, lambda e: perfect_findings(e) + [perfect_findings(e)[0]]),
    }
    for base_fn, dup_fn in cases.values():
        _, base = run_submission(seed, DIRTY, base_fn)
        _, dup = run_submission(seed, DIRTY, dup_fn)
        assert dup[1] < base[1]
        assert dup[4]["diagnostics"]["tp"] == base[4]["diagnostics"]["tp"]
        assert dup[4]["diagnostics"]["fp"] > base[4]["diagnostics"]["fp"]


def test_evaluator_scores_duplicates_as_false_positives():
    ep = generate_episode(1, DIRTY)
    ev = Evaluator(ep.episode_id, ep.truth, ep.policy["amount_tolerance_minor"])
    f = finding_from_truth(ep.truth[0])
    payload = {"episode_id": ep.episode_id, "findings": [f, dict(f), dict(f)]}
    res = ev.evaluate(payload, 0)
    assert (res.tp, res.fp, res.fn) == (1, 2, len(ep.truth) - 1)


INVALID_PAYLOADS = {
    "too_many_findings": lambda f: [f] * 31,
    "too_many_evidence": lambda f: [{**f, "evidence_ids": [f"E{i}" for i in range(11)]}],
    "explanation_too_long": lambda f: [{**f, "explanation": "x" * 1001}],
    "missing_key": lambda f: [{k: v for k, v in f.items() if k != "currency"}],
    "extra_key": lambda f: [{**f, "confidence": 0.9}],
    "float_amount": lambda f: [{**f, "claimed_amount_minor": 100.0}],
    "bool_amount": lambda f: [{**f, "claimed_amount_minor": True}],
    "string_amount": lambda f: [{**f, "claimed_amount_minor": "100"}],
    "affected_not_list": lambda f: [{**f, "affected_ids": "DOC-00001"}],
    "evidence_not_strings": lambda f: [{**f, "evidence_ids": [1, 2]}],
    "finding_not_object": lambda f: ["finding"],
}


@pytest.mark.parametrize("name", sorted(INVALID_PAYLOADS))
def test_invalid_submission_gives_minus_one_and_is_logged(name):
    def build(ep):
        return INVALID_PAYLOADS[name](finding_from_truth(ep.truth[0]))

    env, (obs, reward, terminated, truncated, info) = run_submission(1, DIRTY, build)
    assert reward == -1.0 and terminated and not truncated
    assert info["diagnostics"] == {"valid_submission": False}
    assert env.evaluator.error_log and "invalid submission" in env.evaluator.error_log[0]
    json.dumps(obs)


def test_invalid_submit_envelopes_give_minus_one():
    ep = generate_episode(1, DIRTY)
    envelopes = [
        {"tool": "submit"},
        {"tool": "submit", "findings": []},
        {"tool": "submit", "findings": None},
        {"tool": "submit", "findings": {"episode_id": ep.episode_id}},
        {"tool": "submit", "findings": {"episode_id": "EP-WRONG", "findings": []}},
        {"tool": "submit", "findings": {"episode_id": ep.episode_id, "findings": {}}},
        {"tool": "submit", "findings": {"episode_id": ep.episode_id, "findings": [], "x": 1}},
        {"tool": "submit", "findings": {"episode_id": ep.episode_id, "findings": []}, "extra": 1},
    ]
    for action in envelopes:
        env = AuditEnv()
        env.reset(1, DIRTY)
        _, reward, terminated, truncated, _ = env.step(action)
        assert (reward, terminated, truncated) == (-1.0, True, False), action


def test_limits_are_inclusive_and_valid_at_the_boundary():
    ep = generate_episode(1, DIRTY)
    f = finding_from_truth(ep.truth[0])
    boundary = [{**f, "evidence_ids": f["evidence_ids"] + [f"X{i}" for i in range(10 - len(f["evidence_ids"]))],
                 "explanation": "y" * 1000}] + [false_accusation(ep)] * 29
    assert len(boundary) == 30 and len(boundary[0]["evidence_ids"]) == 10
    env = AuditEnv()
    env.reset(1, DIRTY)
    env.step(submit_action(ep.episode_id, boundary))
    res = env.evaluator.final_result()
    assert res["valid_submission"] is True and res["tp"] == 1


def test_matching_rules_tp_conditions():
    ep = generate_episode(2, DIRTY)
    ev = Evaluator(ep.episode_id, ep.truth, ep.policy["amount_tolerance_minor"])
    tol = ep.policy["amount_tolerance_minor"]
    t0 = ep.truth[0]
    other_cur = "USD" if t0["currency"] != "USD" else "EUR"
    other_type = "period_cutoff_error" if t0["type"] == "duplicate_invoice" else "duplicate_invoice"

    def tp(finding):
        return ev.evaluate({"episode_id": ep.episode_id, "findings": [finding]}, 0).tp

    base = finding_from_truth(t0)
    assert tp(base) == 1
    assert tp({**base, "type": other_type}) == 0
    assert tp({**base, "currency": other_cur}) == 0
    assert tp({**base, "affected_ids": base["affected_ids"][:-1] or ["DOC-X"]}) == 0
    assert tp({**base, "affected_ids": base["affected_ids"] + ["DOC-EXTRA"]}) == 0
    assert tp({**base, "claimed_amount_minor": base["claimed_amount_minor"] + tol}) == 1
    assert tp({**base, "claimed_amount_minor": base["claimed_amount_minor"] - tol}) == 1
    assert tp({**base, "claimed_amount_minor": base["claimed_amount_minor"] + tol + 1}) == 0
    assert tp({**base, "claimed_amount_minor": base["claimed_amount_minor"] - tol - 1}) == 0
    # canonicalisation: order and duplicates do not matter
    assert tp({**base, "affected_ids": list(reversed(base["affected_ids"])) + base["affected_ids"],
               "evidence_ids": list(reversed(base["evidence_ids"])) * 2}) == 1
    # evidence must contain a *complete* valid set; extra ids are harmless
    sets = t0["valid_evidence_sets"]
    if len(sets[0]) > 1:
        assert tp({**base, "evidence_ids": sets[0][:-1]}) == 0
    assert tp({**base, "evidence_ids": []}) == 0
    assert tp({**base, "evidence_ids": ["L999999"]}) == 0
    assert tp({**base, "evidence_ids": sets[0] + ["NOISE-1", "NOISE-2"]}) == 1
    for alt in sets:                                         # every listed set is sufficient alone
        assert tp({**base, "evidence_ids": list(alt)}) == 1


def _truth(issue_id, amount, sets, affected=("A",), type_="duplicate_invoice", cur="KZT"):
    return {"issue_id": issue_id, "type": type_, "affected_ids": list(affected),
            "expected_amount_minor": amount, "currency": cur,
            "valid_evidence_sets": [list(s) for s in sets]}


def _finding(amount, evidence, affected=("A",), type_="duplicate_invoice", cur="KZT"):
    return {"type": type_, "affected_ids": list(affected), "claimed_amount_minor": amount,
            "currency": cur, "evidence_ids": list(evidence), "explanation": ""}


def test_metrics_formulas_and_reward_formula():
    truth = [_truth("T1", 100, [["e1"]], ("A",)), _truth("T2", 200, [["e2"]], ("B",))]
    ev = Evaluator("EP", truth, 0)
    pay = lambda fs: {"episode_id": "EP", "findings": fs}  # noqa: E731
    r = ev.evaluate(pay([_finding(100, ["e1"], ("A",)), _finding(5, ["z"], ("C",))]), 7)
    assert (r.tp, r.fp, r.fn) == (1, 1, 1)
    assert r.precision == 0.5 and r.recall == 0.5 and r.f1 == pytest.approx(2 / 4)
    assert r.reward == pytest.approx(2 * 0.5 - 1 - 0.002 * 7)
    none = ev.evaluate(pay([]), 0)
    assert (none.tp, none.fp, none.fn, none.precision, none.recall, none.f1) == (0, 0, 2, 0.0, 0.0, 0.0)
    assert none.reward == -1.0
    allr = ev.evaluate(pay([_finding(100, ["e1"], ("A",)), _finding(200, ["e2"], ("B",))]), 10)
    assert allr.f1 == 1.0 and allr.reward == pytest.approx(1 - 0.02)
    assert compute_reward(1.0, 0) == 1.0 and compute_reward(0.0, 1000) == -1.0
    assert compute_reward(1.0, -10) == 1.0                    # clamped from above


def test_clean_book_special_case_and_zero_denominators():
    ev = Evaluator("EP", [], 0)
    ok = ev.evaluate({"episode_id": "EP", "findings": []}, 4)
    assert ok.f1 == 1.0 and ok.reward == pytest.approx(1 - 0.008)
    bad = ev.evaluate({"episode_id": "EP", "findings": [_finding(1, ["x"])]}, 4)
    assert bad.f1 == 0.0 and bad.precision == 0.0 and bad.recall == 0.0
    assert bad.reward == -1.0


def test_matching_is_truly_one_to_one_and_optimal():
    # F_b matches both references, F_a only the first: a greedy matcher would score 1.
    truth = [_truth("T1", 100, [["e1"], ["e2"]]), _truth("T2", 100, [["e1"]])]
    ev = Evaluator("EP", truth, 0)
    fa, fb = _finding(100, ["e2"]), _finding(100, ["e1"])
    res = ev.evaluate({"episode_id": "EP", "findings": [fb, fa]}, 0)
    assert (res.tp, res.fp, res.fn) == (2, 0, 0)
    res = ev.evaluate({"episode_id": "EP", "findings": [fb, fb, fb]}, 0)
    assert (res.tp, res.fp, res.fn) == (2, 1, 0)             # each reference matched at most once


def test_evaluator_is_single_shot_and_logs_truncation():
    ev = Evaluator("EP", [], 0)
    assert ev.final_result() is None
    ev.finalize(None, 100, truncated=True)
    assert ev.final_result()["truncated"] is True and ev.error_log
    with pytest.raises(RuntimeError):
        ev.finalize({"episode_id": "EP", "findings": []}, 1)


def test_hundredth_non_submit_step_truncates_with_empty_answer():
    for config, expected in ((DIRTY, -1.0), (CLEAN, compute_reward(1.0, 100))):
        env = AuditEnv("train")
        env.reset(5, config)
        for i in range(99):
            obs, reward, terminated, truncated, info = env.step({"tool": "calculator", "expression": "1+1"})
            assert (reward, terminated, truncated) == (0.0, False, False)
            assert obs["steps_remaining"] == 99 - i
        obs, reward, terminated, truncated, info = env.step({"tool": "calculator", "expression": "1+1"})
        assert (terminated, truncated) == (False, True)
        assert reward == pytest.approx(expected) and obs["steps_remaining"] == 0
        assert env.evaluator.final_result()["truncated"] is True
        # stepping beyond the limit keeps reporting the finished state without crashing
        again = env.step({"tool": "calculator", "expression": "1+1"})
        assert (again[1], again[2], again[3]) == (0.0, False, True)
        assert again[0]["error"]["code"] == "episode_finished"
    env = AuditEnv("eval")
    env.reset(5, DIRTY)
    for _ in range(99):
        env.step({"tool": "list_records", "table": "fx", "limit": 1})
    _, reward, terminated, truncated, info = env.step({"tool": "list_records", "table": "fx", "limit": 1})
    assert (reward, terminated, truncated, info["diagnostics"]) == (0.0, False, True, {})
    assert env.evaluator.final_result()["reward"] == -1.0


def test_submit_on_the_hundredth_step_is_still_a_normal_termination():
    ep = generate_episode(5, DIRTY)
    env = AuditEnv("train")
    env.reset(5, DIRTY)
    for _ in range(99):
        env.step({"tool": "list_records", "table": "fx", "limit": 1})
    obs, reward, terminated, truncated, _ = env.step(submit_action(ep.episode_id, perfect_findings(ep)))
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(compute_reward(1.0, 99)) and obs["steps_remaining"] == 0


def test_submit_always_terminates_and_charges_inspection_steps_only_before_submit():
    ep = generate_episode(7, DIRTY)
    env = AuditEnv("train")
    env.reset(7, DIRTY)
    for _ in range(10):
        env.step({"tool": "calculator", "expression": "2*2"})
    obs, reward, terminated, truncated, info = env.step(submit_action(ep.episode_id, perfect_findings(ep)))
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(1 - 0.002 * 10)
    assert env.evaluator.final_result()["inspection_steps"] == 10
    after = env.step({"tool": "calculator", "expression": "1"})
    assert (after[1], after[2], after[3]) == (0.0, True, False)


def test_step_requires_reset_and_schema_helpers():
    with pytest.raises(RuntimeError):
        AuditEnv().step({"tool": "calculator", "expression": "1"})
    with pytest.raises(ValueError):
        AuditEnv("debug")
    assert S.convert_minor(100_000, "485.2500", 2, 2) == 48_525_000
    assert S.convert_minor(1, "0.5000", 2, 2) == 1                     # ROUND_HALF_UP
    assert S.convert_minor(3, "0.5000", 2, 2) == 2
    with pytest.raises(S.SchemaError):
        S.parse_rate("485.25")
    with pytest.raises(S.SchemaError):
        S.parse_timestamp("2026-03-15T14:30:00")
    assert S.parse_timestamp("2026-03-15T14:30:00+05:00").utcoffset().total_seconds() == 5 * 3600
