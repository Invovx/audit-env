"""Checks from the published jury protocol (Genesis FinTech, 20.11.2026), run
through ``judge_adapter.py`` exactly as the jury connects to the container.

Block 1: contract tests C01-C20 on fixtures C (clean), E (one error), L (>=101 rows).
Block 3: matcher/reward tests 1-20 and open examples M01-M12.
Block 4: paired checks - the jury's own permutation and id remapping of a copied
episode directory, loaded through ``episode_dir``.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

import judge_adapter as J
import schemas as S
from evaluator import Evaluator

CLEAN = {"n_ledger_rows": 400, "n_documents": 48, "n_errors": 0}
ONE = {"n_ledger_rows": 400, "n_documents": 48, "n_errors": 1}
LARGE = {"n_ledger_rows": 900, "n_documents": 120, "n_errors": 4}


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    out = {}
    for name, seed, cfg in (("C", 11, CLEAN), ("E", 12, ONE), ("L", 13, LARGE)):
        d = tmp_path_factory.mktemp(name)
        J.generate_episode(seed, cfg, str(d))
        out[name] = d
    return out


def closed(env, d):
    return env.reset(0, {"mode": "closed", "episode_dir": str(d)})


def act(name, **args):
    return {"name": name, "args": args}


def public(d):
    p = Path(d) / "public"
    return {"ledger": S.parse_ledger_csv((p / "ledger.csv").read_text()),
            "invoices": S.parse_jsonl((p / "invoices.jsonl").read_text()),
            "payments": S.parse_jsonl((p / "payments.jsonl").read_text()),
            "manifest": json.loads((p / "manifest.json").read_text())}


def truth(d):
    return json.loads((Path(d) / "private" / "truth.json").read_text())


def oracle_findings(d):
    eid = public(d)["manifest"]["episode_id"]
    return {"episode_id": eid, "findings": [
        {"type": t["type"], "affected_ids": t["affected_ids"], "claimed_amount_minor": t["expected_amount_minor"],
         "currency": t["currency"], "evidence_ids": t["valid_evidence_sets"][0], "explanation": "oracle"}
        for t in truth(d)]}


def is_json(value) -> bool:
    return json.loads(json.dumps(value)) == value


# --------------------------------------------------------------------------- #
# Block 1 - contract tests
# --------------------------------------------------------------------------- #
def test_C01_C02_C03_reset_and_first_observation(dirs):
    env = J.make_env()
    obs, info = closed(env, dirs["C"])
    assert is_json(obs) and is_json(info)
    assert set(obs) == {"task", "manifest", "policy", "remaining_steps"} and obs["remaining_steps"] == 100
    m, pub = obs["manifest"], public(dirs["C"])
    for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end",
                "timezone", "counts", "rounding_rule"):
        assert key in m
    assert m["counts"]["ledger_rows"] == len(pub["ledger"])
    assert m["counts"]["documents"] == len(pub["invoices"]) + len(pub["payments"])
    blob = json.dumps(obs)
    assert "seed" not in blob.lower().replace("seeded", "") and "ISS-" not in blob


def test_C04_step_signature(dirs):
    env = J.make_env()
    closed(env, dirs["C"])
    out = env.step(act("list_records", table="ledger", cursor=None, limit=10))
    obs, reward, terminated, truncated, info = out
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert is_json(obs) and is_json(info) and reward == 0
    assert set(obs) == {"result", "error", "remaining_steps"} and obs["remaining_steps"] == 99


def test_C05_C06_pagination_and_stable_cursors(dirs):
    env = J.make_env()
    closed(env, dirs["L"])
    seen, cursor, pages = [], None, []
    while True:
        obs = env.step(act("list_records", table="ledger", cursor=cursor, limit=50))[0]
        assert len(obs["result"]["records"]) <= 50
        pages.append((cursor, obs["result"]["records"]))
        seen += [r["row_id"] for r in obs["result"]["records"]]
        cursor = obs["result"]["next_cursor"]
        if cursor is None:
            break
    rows = [r["row_id"] for r in public(dirs["L"])["ledger"]]
    assert len(rows) >= 101 and seen == rows and len(pages) >= 3
    again = env.step(act("list_records", table="ledger", cursor=pages[1][0], limit=50))[0]
    assert again["result"]["records"] == pages[1][1]


def test_C07_C08_get_record_and_document(dirs):
    env = J.make_env()
    closed(env, dirs["C"])
    pub = public(dirs["C"])
    r1, d1 = pub["ledger"][3], pub["invoices"][2]
    assert env.step(act("get_record", table="ledger", id=r1["row_id"]))[0]["result"] == r1
    assert env.step(act("get_document", doc_id=d1["doc_id"]))[0]["result"] == d1


def test_C09_C10_calculator(dirs):
    env = J.make_env()
    closed(env, dirs["C"])
    assert env.step(act("calculator", expression="0.1+0.2"))[0]["result"] == "0.3"
    for expr in ("open('/etc/passwd').read()", "__import__('os').system('ls')", "1;2"):
        obs = env.step(act("calculator", expression=expr))[0]
        assert obs["result"] is None and obs["error"]["code"] == "calculator_error"


def test_C11_C12_C13_C14_bounded_errors_cost_a_step(dirs):
    env = J.make_env()
    closed(env, dirs["C"])
    pub = public(dirs["C"])
    unknown = "U-" + "".join(r["row_id"] for r in pub["ledger"][:2])     # absent by construction
    remaining = 100
    for action in (act("get_record", table="ledger", id=unknown), act("get_document", doc_id=unknown),
                   act("no_such_action"), {"bad": "envelope"}):
        obs = env.step(action)[0]
        remaining -= 1
        assert obs["result"] is None and obs["error"]["code"] and obs["remaining_steps"] == remaining
        assert len(obs["error"]["message"]) <= 200 and "truth" not in obs["error"]["message"]
    ok = env.step(act("get_record", table="ledger", id=pub["ledger"][0]["row_id"]))[0]
    assert ok["error"] is None and ok["remaining_steps"] == remaining - 1


def test_C15_submit_is_final(dirs):
    env = J.make_env()
    closed(env, dirs["E"])
    f = oracle_findings(dirs["E"])
    obs, reward, terminated, truncated, _ = env.step(act("submit", findings=f))
    assert obs["result"] == {"accepted": True} and terminated and not truncated and reward == 0
    before = env._env.evaluator.final_result()
    obs2, _, term2, trunc2, _ = env.step(act("submit", findings={"episode_id": f["episode_id"], "findings": []}))
    assert obs2["error"] is not None and term2 and env._env.evaluator.final_result() == before


def test_C16_submit_as_action_100_is_accepted(dirs):
    env = J.make_env()
    closed(env, dirs["E"])
    for _ in range(99):
        env.step(act("calculator", expression="1+1"))
    obs, _, terminated, truncated, _ = env.step(act("submit", findings=oracle_findings(dirs["E"])))
    final = env._env.evaluator.final_result()
    assert obs["result"] == {"accepted": True} and terminated and not truncated
    assert final["inspection_steps"] == 99 and final["tp"] == 1


def test_C17_100_inspections_truncate(dirs):
    env = J.make_env()
    closed(env, dirs["E"])
    for i in range(100):
        out = env.step(act("calculator", expression="1+1"))
    assert out[3] is True and out[2] is False
    final = env._env.evaluator.final_result()
    assert final["fn"] == 1 and final["tp"] == 0 and final["truncated"]
    late = env.step(act("submit", findings=oracle_findings(dirs["E"])))
    assert late[0]["error"] is not None and env._env.evaluator.final_result() == final


def test_C18_time_limit_truncates(dirs):
    env = J.make_env()
    now = [0.0]
    env._clock = lambda: now[0]
    closed(env, dirs["E"])
    env.step(act("calculator", expression="1+1"))
    now[0] = 61.0
    obs, _, terminated, truncated, _ = env.step(act("submit", findings=oracle_findings(dirs["E"])))
    final = env._env.evaluator.final_result()
    assert truncated and not terminated and final["tp"] == 0 and final["truncated"]


def test_C19_closed_mode_hides_reward_train_shows_it(dirs):
    for mode, expect_reward in (("closed", False), ("train", True)):
        env = J.make_env()
        env.reset(0, {"mode": mode, "episode_dir": str(dirs["E"])})
        rewards = [env.step(act("list_records", table="invoices", cursor=None, limit=5))[1] for _ in range(3)]
        obs, reward, _, _, info = env.step(act("submit", findings=oracle_findings(dirs["E"])))
        assert rewards == [0.0, 0.0, 0.0]
        if expect_reward:
            assert reward > 0.9 and info["diagnostics"]["tp"] == 1
        else:
            assert reward == 0 and "diagnostics" not in info and obs == {
                "result": {"accepted": True}, "error": None, "remaining_steps": 96}


def test_C20_no_closed_data_anywhere_in_the_policy_channel(dirs):
    for name in ("C", "E"):
        env = J.make_env()
        obs, info = closed(env, dirs[name])
        stream = [obs, info]
        for table in ("ledger", "invoices", "payments", "fx"):
            cursor = None
            while True:
                o = env.step(act("list_records", table=table, cursor=cursor, limit=50))
                stream += [o[0], o[4]]
                cursor = o[0]["result"]["next_cursor"]
                if cursor is None or o[0]["remaining_steps"] < 5:
                    break
        stream.append(env.step(act("get_document", doc_id="../private/truth.json"))[0])
        blob = json.dumps(stream)
        for t in truth(dirs[name]):
            assert t["issue_id"] not in blob
        for marker in ("truth", "provenance", "valid_evidence_sets", "expected_amount_minor", str(dirs[name])):
            assert marker not in blob


def test_reset_from_dir_restarts_and_never_regenerates(dirs):
    env = J.make_env()
    a = closed(env, dirs["L"])[0]
    env.step(act("calculator", expression="2*2"))
    b = closed(env, dirs["L"])[0]
    assert a == b and b["remaining_steps"] == 100
    assert b["manifest"]["episode_id"] == public(dirs["L"])["manifest"]["episode_id"]


# --------------------------------------------------------------------------- #
# Block 3 - matcher and reward (hand-built fixture, tolerance 0)
# --------------------------------------------------------------------------- #
T, T2 = "duplicate_invoice", "period_cutoff_error"
IDS = ["R1", "R2", "R3", "R4", "R5", "D1", "D2", "D3"]


def ref(issue="ISS-A", affected=("R1",), amount=10000, evidence=(("D1", "D2"),), typ=T):
    return {"issue_id": issue, "type": typ, "affected_ids": list(affected), "expected_amount_minor": amount,
            "currency": "KZT", "valid_evidence_sets": [list(e) for e in evidence]}


def fnd(affected=("R1",), amount=10000, evidence=("D1", "D2"), typ=T, currency="KZT", explanation="ok"):
    return {"type": typ, "affected_ids": list(affected), "claimed_amount_minor": amount,
            "currency": currency, "evidence_ids": list(evidence), "explanation": explanation}


def grade(refs, findings, steps=10, tol=0):
    ev = Evaluator("EP-X", refs, tol, IDS)
    return ev.evaluate({"episode_id": "EP-X", "findings": findings}, steps).as_dict()


def close(a, b):
    return abs(a - b) <= 1e-9


@pytest.mark.parametrize("findings,refs,tp,fp,fn,reward", [
    ([fnd()], [ref()], 1, 0, 0, 0.98),                                        # 1
    ([], [ref()], 0, 0, 1, -1.0),                                             # 2
    ([], [], 0, 0, 0, 0.98),                                                  # 3
    ([fnd()], [], 0, 1, 0, -1.0),                                             # 4
    ([fnd(), fnd()], [ref()], 1, 1, 0, 47 / 150),                             # 5
    ([fnd(typ=T2)], [ref()], 0, 1, 1, -1.0),                                  # 6
    ([fnd(affected=("R2",))], [ref()], 0, 1, 1, -1.0),                        # 7
    ([fnd(amount=10001)], [ref()], 0, 1, 1, -1.0),                            # 8
    ([fnd(currency="USD")], [ref()], 0, 1, 1, -1.0),                          # 9
    ([fnd(evidence=("D1",))], [ref()], 0, 1, 1, -1.0),                        # 10
    ([fnd(evidence=("D1", "U"))], [ref()], 0, 1, 1, -1.0),                    # 11
    ([fnd()], [ref(), ref("ISS-B", affected=("R3",))], 1, 0, 1, 47 / 150),    # 13
    ([fnd(), fnd(affected=("R5",))], [ref(), ref("ISS-B", affected=("R3",))], 1, 1, 1, -0.02),  # 14
])
def test_block3_matcher(findings, refs, tp, fp, fn, reward):
    r = grade(refs, findings)
    assert (r["tp"], r["fp"], r["fn"]) == (tp, fp, fn) and close(r["reward"], reward)


def test_block3_12_mass_flagging():
    types = [T, T2, "partial_payment_allocation", "fx_rate_error"]
    findings = [fnd(affected=(r,), typ=t) for r in IDS[:5] for t in types]
    r = grade([ref()], findings)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 19, 0) and close(r["reward"], -871 / 1050)


@pytest.mark.parametrize("steps,reward", [(0, 1.0), (50, 0.90), (100, 0.80)])
def test_block3_15_16_17_step_cost(steps, reward):
    assert close(grade([ref()], [fnd()], steps)["reward"], reward)


@pytest.mark.parametrize("findings", [
    [fnd()] * 31,                                                             # 18
    [fnd(evidence=["D1", "D2"] + [f"R{i % 5 + 1}" for i in range(9)])],      # 19 (11 ids)
    [fnd(explanation="a" * 1001)],                                            # 20
])
def test_block3_18_19_20_structure_limits(findings, tmp_path, dirs):
    r = grade([ref()], findings)
    assert r["valid_submission"] is False and r["reward"] == -1.0


# --------------------------------------------------------------------------- #
# Open examples M01-M12 (A: affected {R1,R2}, 10000 KZT, evidence {D1,D2}, tau 0)
# --------------------------------------------------------------------------- #
A = ref(affected=("R1", "R2"))


@pytest.mark.parametrize("finding,tp", [
    (fnd(affected=("R2", "R1", "R1")), 1),                                    # M01
    (fnd(affected=("R1", "R2", "R3")), 0),                                    # M02
    (fnd(affected=("R1", "R2"), evidence=("D2", "D1", "D3")), 1),            # M03
    (fnd(affected=("R1", "R2"), evidence=("D1", "D2", "U")), 0),             # M04
    (fnd(affected=("R1", "R2"), currency="kzt"), 0),                          # M06
    (fnd(affected=("R1", "R2"), currency="KZT "), 0),                         # M06
    (fnd(affected=("R1", "R2"), amount=10001), 0),                            # M07
])
def test_M_examples_matching(finding, tp):
    r = grade([A], [finding])
    assert r["valid_submission"] and r["tp"] == tp and r["fp"] == 1 - tp and r["fn"] == 1 - tp


def test_M05_M09_structure_errors():
    assert grade([A], [fnd(affected=("R1", "R2"), evidence=["D1"] * 10 + ["D2"])])["reward"] == -1.0
    for amount in ("10000", True, 10000.0):
        assert grade([A], [fnd(affected=("R1", "R2"), amount=amount)])["valid_submission"] is False


def test_M08_tolerance_one():
    assert grade([A], [fnd(affected=("R1", "R2"), amount=10001)], tol=1)["tp"] == 1
    assert grade([A], [fnd(affected=("R1", "R2"), amount=10002)], tol=1)["tp"] == 0


def test_M10_M11_M12_maximum_matching():
    a, b = ref("ISS-A", evidence=(("D1",),)), ref("ISS-B", evidence=(("D2",),))
    f0, f1 = fnd(evidence=("D1", "D2")), fnd(evidence=("D1",))
    assert grade([a, b], [f0, f1])["tp"] == 2                                 # M10
    assert grade([a, b], [f1, f0])["tp"] == 2
    both = fnd(evidence=("D1", "D2"))
    r1, r2 = grade([a, b], [both, both]), grade([b, a], [both, both])         # M11
    assert r1["tp"] == r2["tp"] == 2 and r1["reward"] == r2["reward"]
    r = grade([ref()], [fnd(), fnd()])                                        # M12
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)


def test_score_episode_reads_tolerance_and_ids_from_episode_dir(dirs, tmp_path):
    d = tmp_path / "tau"
    shutil.copytree(dirs["E"], d)
    policy_path = d / "public" / "policy.json"
    policy = json.loads(policy_path.read_text())
    f = oracle_findings(d)
    f["findings"][0]["claimed_amount_minor"] += 1
    policy["amount_tolerance_minor"] = 0
    policy_path.write_text(json.dumps(policy))
    assert J.score_episode(str(d), f, 0)["tp"] == 0
    policy["amount_tolerance_minor"] = 1
    policy_path.write_text(json.dumps(policy))
    assert J.score_episode(str(d), f, 0)["tp"] == 1
    bad = {"episode_id": f["episode_id"], "findings": [{"type": "x"}]}
    assert J.score_episode(str(d), bad, 0) == {"tp": None, "fp": None, "fn": None, "precision": None,
                                               "recall": None, "f1": None, "return": -1.0}


# --------------------------------------------------------------------------- #
# Block 4 - paired checks: the jury's own transformation of a copied directory
# --------------------------------------------------------------------------- #
def jury_transform(src: Path, dst: Path, rng: random.Random, permute: bool, remap: bool) -> dict:
    """Copy an episode, permute rows and/or remap ids consistently everywhere."""
    shutil.copytree(src, dst)
    p = dst / "public"
    ledger = S.parse_ledger_csv((p / "ledger.csv").read_text())
    invoices = S.parse_jsonl((p / "invoices.jsonl").read_text())
    payments = S.parse_jsonl((p / "payments.jsonl").read_text())
    fx = S.parse_fx_csv((p / "fx.csv").read_text())
    mapping: dict = {}

    def new(space: str, old):
        if old is None or not remap:
            return old
        key = (space, old)
        if key not in mapping:
            mapping[key] = f"{space[:2].upper()}-{rng.getrandbits(40):010x}"
        return mapping[key]

    for r in ledger:
        r["row_id"] = new("row", r["row_id"])
        r["entity_id"] = new("entity", r["entity_id"])
        r["invoice_id"] = new("invoice", r["invoice_id"])
        r["payment_id"] = new("payment", r["payment_id"])
    for d in invoices:
        d["doc_id"] = new("doc", d["doc_id"])
        d["invoice_id"] = new("invoice", d["invoice_id"])
        d["entity_id"] = new("entity", d["entity_id"])
        for line in d["line_items"]:
            line["ref_invoice_id"] = new("invoice", line["ref_invoice_id"])
    for d in payments:
        d["doc_id"] = new("doc", d["doc_id"])
        d["payment_id"] = new("payment", d["payment_id"])
        d["invoice_refs"] = [new("invoice", x) for x in d["invoice_refs"]]
        if "entity_id" in d:
            d["entity_id"] = new("entity", d["entity_id"])
    if permute:
        for rows in (ledger, invoices, payments, fx):
            rng.shuffle(rows)
    (p / "ledger.csv").write_text(S.ledger_to_csv(ledger))
    (p / "invoices.jsonl").write_text(S.to_jsonl(invoices))
    (p / "payments.jsonl").write_text(S.to_jsonl(payments))
    (p / "fx.csv").write_text(S.fx_to_csv(fx))
    ids = lambda x: mapping.get(("row", x)) or mapping.get(("doc", x)) or x  # noqa: E731
    t = truth(dst)
    for entry in t:
        entry["affected_ids"] = [ids(x) for x in entry["affected_ids"]]
        entry["valid_evidence_sets"] = [[ids(x) for x in ev] for ev in entry["valid_evidence_sets"]]
    (dst / "private" / "truth.json").write_text(json.dumps(t))
    return mapping


PAIR_CFG = {"n_ledger_rows": 900, "n_documents": 120, "n_errors": 4,
            "interactions": ["partial_fx_boundary", "cutoff_fx"]}


@pytest.mark.parametrize("pair", range(20))
def test_block4_paired_checks(pair, tmp_path):
    cfg = CLEAN if pair < 5 else PAIR_CFG
    src = tmp_path / "src"
    J.generate_episode(3000 + pair, cfg, str(src))
    permute, remap = (True, False) if pair < 5 else (False, True) if pair < 10 else (True, True)
    dst = tmp_path / "dst"
    jury_transform(src, dst, random.Random(pair), permute, remap)
    S.validate_public(json.loads((dst / "public/manifest.json").read_text()),
                      json.loads((dst / "public/policy.json").read_text()),
                      S.parse_ledger_csv((dst / "public/ledger.csv").read_text()),
                      S.parse_jsonl((dst / "public/invoices.jsonl").read_text()),
                      S.parse_jsonl((dst / "public/payments.jsonl").read_text()),
                      S.parse_fx_csv((dst / "public/fx.csv").read_text()))

    def run(d):
        env = J.make_env()
        trace = [closed(env, d)]
        for action in (act("list_records", table="invoices", cursor=None, limit=7),
                       act("calculator", expression="1/3"),
                       act("submit", findings=oracle_findings(d))):
            trace.append(env.step(action))
        return trace, J.score_episode(str(d), oracle_findings(d), 2)

    t1, s1 = run(src)
    t1b, s1b = run(src)
    t2, s2 = run(dst)
    assert t1 == t1b and s1 == s1b                    # exact replay on the original
    assert s1 == s2                                   # same TP/FP/FN/P/R/F1/return after the transform
    assert s1["f1"] == 1.0
