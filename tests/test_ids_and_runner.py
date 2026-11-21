"""Tests for RandomizedIdScheme invariance (20 paired checks) and runner.py isolation."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runner as R  # noqa: E402
import schemas as S  # noqa: E402
from env import AuditEnv  # noqa: E402
from evaluator import compute_reward  # noqa: E402
from generator import RandomizedIdScheme, SequentialIdScheme, generate_episode, write_episode  # noqa: E402

DIRTY_BASE = {"n_ledger_rows": 400, "n_documents": 48, "n_errors": 4}
CLEAN_BASE = {"n_ledger_rows": 400, "n_documents": 48, "n_errors": 0}

RE_ROW = re.compile(r"^L-[0-9a-f]{6}$")
RE_DOC = re.compile(r"^DOC-[0-9a-f]{6}$")
RE_INV = re.compile(r"^INV-[0-9a-f]{6}$")
RE_CN = re.compile(r"^CN-[0-9a-f]{6}$")
RE_PAY = re.compile(r"^PAY-[0-9a-f]{6}$")
RE_CUST = re.compile(r"^CUST-[0-9a-f]{6}$")
RE_ENTITY = re.compile(r"^(CUST|SUP)-[0-9a-f]{6}$")

TRAJECTORY_KEYS = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def oracle_findings(episode) -> list:
    return [{
        "type": entry["type"],
        "affected_ids": list(entry["affected_ids"]),
        "claimed_amount_minor": entry["expected_amount_minor"],
        "currency": entry["currency"],
        "evidence_ids": list(entry["valid_evidence_sets"][0]),
        "explanation": "oracle proof",
    } for entry in episode.truth]


def submit_action(episode_id: str, findings: list) -> dict:
    return {"tool": "submit", "findings": {"episode_id": episode_id, "findings": findings}}


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


def _canonical_truth(episode) -> list:
    """Resolve truth ids to internal record keys (invariant across id schemes)."""
    reg = episode.provenance["id_registry"]
    lookup = {**reg["ledger_rows"], **reg["documents"]}
    out = []
    for entry in episode.truth:
        out.append((
            entry["type"],
            entry["expected_amount_minor"],
            entry["currency"],
            tuple(sorted(lookup[i] for i in entry["affected_ids"])),
            tuple(sorted(tuple(sorted(lookup[i] for i in ev)) for ev in entry["valid_evidence_sets"])),
        ))
    return sorted(out)


def _duplicate_alias_signature(episode) -> list:
    """For every duplicate_invoice, record whether both copies share invoice_id."""
    inv_by_doc = {d["doc_id"]: d for d in episode.invoices}
    sig = []
    for entry in episode.truth:
        if entry["type"] == "duplicate_invoice":
            a, b = (inv_by_doc[doc_id] for doc_id in entry["affected_ids"])
            sig.append((entry["expected_amount_minor"], entry["currency"],
                        a["invoice_id"] == b["invoice_id"]))
    return sorted(sig)


def _assert_trajectory_schema(trajectory: list, episode_id: str) -> None:
    assert len(trajectory) >= 2
    steps = [entry["step"] for entry in trajectory]
    assert steps == list(range(len(trajectory)))
    assert trajectory[0]["step"] == 0 and trajectory[0]["action"] is None
    assert trajectory[0]["observation"]["kind"] == "task"
    for entry in trajectory:
        assert set(entry.keys()) == TRAJECTORY_KEYS
        assert entry["episode_id"] == episode_id
        json.dumps(entry)
    assert trajectory[-1]["terminated"] or trajectory[-1]["truncated"]
    assert not (trajectory[-1]["terminated"] and trajectory[-1]["truncated"])
    for entry in trajectory[:-1]:
        assert not entry["terminated"] and not entry["truncated"]


def _assert_no_leaks(trajectory: list, episode, seed: int, forbidden_path: str = "") -> None:
    for entry in trajectory:
        for text in _all_strings(entry):
            low = text.lower()
            for word in ("truth", "provenance", "seed", "private", "mutation"):
                assert word not in low, f"leaked {word!r} in {text!r}"
            assert str(seed) not in text
            assert not re.search(r"[A-Za-z]:[\\/]", text), f"leaked windows path in {text!r}"
            assert "/tmp/" not in text and "/home/" not in text and "/Users/" not in text
            if forbidden_path:
                assert forbidden_path not in text
            for t in episode.truth:
                assert t["issue_id"] not in text


# --------------------------------------------------------------------------- #
# Part 1: 20 paired invariance checks (SequentialIdScheme vs RandomizedIdScheme)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(20))
def test_twenty_paired_id_invariance_checks(seed):
    seq_cfg = dict(DIRTY_BASE)
    rnd_cfg = dict(DIRTY_BASE, id_scheme="randomized", id_salt=seed + 1)

    seq_ep = generate_episode(seed, seq_cfg)
    rnd_ep = generate_episode(seed, rnd_cfg)

    # 1. Public & truth schemas validate cleanly.
    S.validate_public(rnd_ep.manifest, rnd_ep.policy, rnd_ep.ledger,
                      rnd_ep.invoices, rnd_ep.payments, rnd_ep.fx)
    for entry in rnd_ep.truth:
        S.validate_truth_entry(entry)

    # 2. All randomized IDs match the hex format.
    for row in rnd_ep.ledger:
        assert RE_ROW.match(row["row_id"])
        if row["entity_id"] is not None:
            assert RE_ENTITY.match(row["entity_id"])
        if row["invoice_id"] is not None:
            assert RE_INV.match(row["invoice_id"]) or RE_CN.match(row["invoice_id"])
        if row["payment_id"] is not None:
            assert RE_PAY.match(row["payment_id"])

    for doc in rnd_ep.invoices:
        assert RE_DOC.match(doc["doc_id"])
        assert RE_CUST.match(doc["entity_id"])
        if doc["total_minor"] >= 0:
            assert RE_INV.match(doc["invoice_id"])
        else:
            assert RE_CN.match(doc["invoice_id"])
        for line in doc["line_items"]:
            if line["ref_invoice_id"] is not None:
                assert RE_INV.match(line["ref_invoice_id"])

    for doc in rnd_ep.payments:
        assert RE_DOC.match(doc["doc_id"])
        assert RE_PAY.match(doc["payment_id"])
        for ref in doc["invoice_refs"]:
            assert RE_INV.match(ref) or RE_CN.match(ref)

    # 3. Distinct episode_id and shuffled internal order for ledger, invoices, payments.
    assert rnd_ep.episode_id != seq_ep.episode_id
    seq_reg = seq_ep.provenance["id_registry"]
    rnd_reg = rnd_ep.provenance["id_registry"]

    seq_ledger_order = [seq_reg["ledger_rows"][r["row_id"]] for r in seq_ep.ledger]
    rnd_ledger_order = [rnd_reg["ledger_rows"][r["row_id"]] for r in rnd_ep.ledger]
    assert set(seq_ledger_order) == set(rnd_ledger_order)
    assert seq_ledger_order != rnd_ledger_order

    seq_inv_order = [seq_reg["documents"][d["doc_id"]] for d in seq_ep.invoices]
    rnd_inv_order = [rnd_reg["documents"][d["doc_id"]] for d in rnd_ep.invoices]
    assert set(seq_inv_order) == set(rnd_inv_order)
    assert seq_inv_order != rnd_inv_order

    seq_pay_order = [seq_reg["documents"][d["doc_id"]] for d in seq_ep.payments]
    rnd_pay_order = [rnd_reg["documents"][d["doc_id"]] for d in rnd_ep.payments]
    assert set(seq_pay_order) == set(rnd_pay_order)
    if len(seq_pay_order) >= 4:
        assert seq_pay_order != rnd_pay_order

    # 4. Semantic equivalence: counts, truth (via internal keys) and duplicate alias_of.
    assert rnd_ep.manifest["counts"] == seq_ep.manifest["counts"]
    assert _canonical_truth(rnd_ep) == _canonical_truth(seq_ep)
    assert _duplicate_alias_signature(rnd_ep) == _duplicate_alias_signature(seq_ep)

    # 5. Oracle receives the EXACT same reward and F1 == 1.0 on both variants.
    inspection = [
        {"tool": "list_records", "table": "ledger", "limit": 10},
        {"tool": "list_records", "table": "invoices", "limit": 10},
        {"tool": "calculator", "expression": "100 + 200"},
    ]

    def run_oracle(cfg, ep):
        env = AuditEnv("train")
        env.reset(seed, cfg)
        for act in inspection:
            env.step(act)
        obs, reward, terminated, truncated, info = env.step(
            submit_action(ep.episode_id, oracle_findings(ep)))
        return reward, terminated, truncated, info, env.evaluator.final_result()

    r_seq, term_s, trunc_s, info_s, res_s = run_oracle(seq_cfg, seq_ep)
    r_rnd, term_r, trunc_r, info_r, res_r = run_oracle(rnd_cfg, rnd_ep)
    assert (term_s, trunc_s) == (True, False) == (term_r, trunc_r)
    assert res_s["f1"] == 1.0 and res_r["f1"] == 1.0
    assert r_rnd == pytest.approx(r_seq)
    assert r_rnd == pytest.approx(compute_reward(1.0, len(inspection)))
    assert info_s["diagnostics"] == info_r["diagnostics"]

    # Submitting sequential IDs to the randomized episode must not match.
    env_mismatch = AuditEnv("train")
    env_mismatch.reset(seed, rnd_cfg)
    _, r_bad, _, _, _ = env_mismatch.step(
        submit_action(rnd_ep.episode_id, oracle_findings(seq_ep)))
    assert env_mismatch.evaluator.final_result()["tp"] == 0
    assert r_bad == -1.0


def test_randomized_determinism_salt_and_clean_books():
    cfg_a = dict(DIRTY_BASE, id_scheme="randomized", id_salt=7)
    ep1 = generate_episode(42, cfg_a)
    ep2 = generate_episode(42, cfg_a)
    assert ep1.public_files() == ep2.public_files()
    assert ep1.private_files() == ep2.private_files()

    # Explicit id_scheme parameter matches config-selected scheme.
    ep3 = generate_episode(42, DIRTY_BASE, id_scheme=RandomizedIdScheme(7))
    assert ep3.public_files() == ep1.public_files()
    assert ep3.truth == ep1.truth

    # Different salt gives different IDs & order, same semantic history.
    ep_other = generate_episode(42, dict(DIRTY_BASE, id_scheme="randomized", id_salt=8))
    assert ep_other.episode_id != ep1.episode_id
    assert ep_other.public_files() != ep1.public_files()
    assert _canonical_truth(ep_other) == _canonical_truth(ep1)

    # Clean book invariance.
    clean_seq = generate_episode(13, CLEAN_BASE)
    clean_rnd = generate_episode(13, dict(CLEAN_BASE, id_scheme="randomized", id_salt=99))
    assert clean_seq.truth == [] and clean_rnd.truth == []
    env_s, env_r = AuditEnv("train"), AuditEnv("train")
    env_s.reset(13, CLEAN_BASE)
    env_r.reset(13, dict(CLEAN_BASE, id_scheme="randomized", id_salt=99))
    _, rew_s, *_ = env_s.step(submit_action(clean_seq.episode_id, []))
    _, rew_r, *_ = env_r.step(submit_action(clean_rnd.episode_id, []))
    assert rew_s == rew_r == 1.0

    # Invalid id_scheme / id_salt values are rejected.
    for bad in ({"id_scheme": "uuid"}, {"id_scheme": 1}, {"id_salt": -1}, {"id_salt": True}):
        with pytest.raises(S.SchemaError):
            generate_episode(1, bad)
    with pytest.raises(S.SchemaError):
        RandomizedIdScheme(-5)
    with pytest.raises(S.SchemaError):
        generate_episode(1, DIRTY_BASE, id_scheme="randomized")  # type: ignore[arg-type]


def test_cursor_pagination_and_lookups_on_shuffled_tables():
    cfg = dict(DIRTY_BASE, id_scheme="randomized", id_salt=17)
    ep = generate_episode(11, cfg)
    env = AuditEnv("train")
    env.reset(11, cfg)

    expected_tables = {
        "ledger": (ep.ledger, "row_id"),
        "invoices": (ep.invoices, "doc_id"),
        "payments": (ep.payments, "doc_id"),
        "fx": ([{"fx_id": S.fx_record_id(r), **r} for r in ep.fx], "fx_id"),
    }
    for table, (expected_rows, key) in expected_tables.items():
        collected = []
        cursor = None
        while True:
            obs = env.step({"tool": "list_records", "table": table, "cursor": cursor, "limit": 29})[0]
            assert obs["kind"] == "records"
            # Cursor stability: repeating the same request returns identical page & cursor.
            if cursor is not None and len(collected) < 60:
                repeat = env.step({"tool": "list_records", "table": table, "cursor": cursor, "limit": 29})[0]
                assert repeat["records"] == obs["records"]
                assert repeat["next_cursor"] == obs["next_cursor"]
            collected.extend(obs["records"])
            cursor = obs["next_cursor"]
            if cursor is None:
                assert not obs["has_more"]
                break
        assert [r[key] for r in collected] == [r[key] for r in expected_rows]
        assert len({r[key] for r in collected}) == len(expected_rows)

    # Direct lookups with randomized IDs.
    env.reset(11, cfg)
    sample_row = ep.ledger[5]
    sample_inv = ep.invoices[2]
    sample_pay = ep.payments[0]
    assert env.step({"tool": "get_record", "table": "ledger", "id": sample_row["row_id"]})[0]["record"] == sample_row
    assert env.step({"tool": "get_record", "table": "invoices", "id": sample_inv["doc_id"]})[0]["record"] == sample_inv
    doc_obs = env.step({"tool": "get_document", "doc_id": sample_pay["doc_id"]})[0]
    assert doc_obs["doc_type"] == "payment" and doc_obs["document"] == sample_pay


def test_env_reset_accepts_explicit_id_scheme():
    env = AuditEnv("train")
    obs, _ = env.reset(3, DIRTY_BASE, id_scheme=RandomizedIdScheme(12))
    ep = generate_episode(3, DIRTY_BASE, id_scheme=RandomizedIdScheme(12))
    assert obs["manifest"]["episode_id"] == ep.episode_id
    first = env.step({"tool": "list_records", "table": "ledger", "limit": 1})[0]["records"][0]
    assert RE_ROW.match(first["row_id"])


# --------------------------------------------------------------------------- #
# Part 2: runner.py subprocess isolation tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_runner_silent_agent_clean_and_dirty(mode):
    cmd = R.dummy_agent_command("silent", python_executable=sys.executable)

    clean_res = R.run_episode_subprocess(5, cmd, config=CLEAN_BASE, mode=mode)
    assert clean_res.error is None and clean_res.agent_exit_code == 0
    assert clean_res.eval_result["f1"] == 1.0
    assert clean_res.eval_result["reward"] == pytest.approx(1.0)
    assert clean_res.eval_result["truncated"] is False
    _assert_trajectory_schema(clean_res.trajectory, clean_res.episode_id)
    assert clean_res.trajectory[-1]["terminated"] is True
    if mode == "eval":
        assert all(step["reward"] == 0.0 for step in clean_res.trajectory)
    else:
        assert clean_res.trajectory[-1]["reward"] == pytest.approx(1.0)

    dirty_res = R.run_episode_subprocess(5, cmd, config=DIRTY_BASE, mode=mode)
    assert dirty_res.error is None and dirty_res.agent_exit_code == 0
    assert dirty_res.eval_result["f1"] == 0.0
    assert dirty_res.eval_result["reward"] == -1.0
    assert dirty_res.eval_result["fn"] == 4
    _assert_trajectory_schema(dirty_res.trajectory, dirty_res.episode_id)


def test_runner_random_agent_explores_and_completes():
    cmd = R.dummy_agent_command("random", python_executable=sys.executable, agent_seed=7, steps=8)
    res = R.run_episode_subprocess(3, cmd, config=DIRTY_BASE, mode="eval")
    assert res.error is None and res.agent_exit_code == 0
    assert res.eval_result["valid_submission"] is True
    assert res.eval_result["truncated"] is False
    assert res.eval_result["inspection_steps"] >= 3
    _assert_trajectory_schema(res.trajectory, res.episode_id)
    assert res.trajectory[-1]["terminated"] is True
    assert all(step["reward"] == 0.0 for step in res.trajectory)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_runner_replay_oracle_on_sequential_and_randomized(tmp_path, mode):
    seed = 918273645
    seq_cfg = dict(DIRTY_BASE)
    rnd_cfg = dict(DIRTY_BASE, id_scheme="randomized", id_salt=5)

    def run_replay(cfg, name):
        ep = generate_episode(seed, cfg)
        write_episode(ep, tmp_path / name)  # private files exist on disk during run
        actions = [
            {"tool": "list_records", "table": "ledger", "limit": 5},
            {"tool": "calculator", "expression": "485.2500 * 2"},
            {"tool": "submit", "findings": {"episode_id": "$EPISODE_ID",
                                            "findings": oracle_findings(ep)}},
        ]
        actions_file = tmp_path / f"{name}_actions.json"
        actions_file.write_text(json.dumps(actions), encoding="utf-8")
        cmd = R.dummy_agent_command("replay", python_executable=sys.executable,
                                    actions_path=actions_file)
        res = R.run_episode_subprocess(seed, cmd, config=cfg, mode=mode)
        return ep, res

    seq_ep, seq_res = run_replay(seq_cfg, "seq")
    rnd_ep, rnd_res = run_replay(rnd_cfg, "rnd")

    for ep, res in ((seq_ep, seq_res), (rnd_ep, rnd_res)):
        assert res.error is None and res.agent_exit_code == 0
        assert res.eval_result["f1"] == 1.0
        assert res.eval_result["tp"] == 4
        assert res.eval_result["fp"] == 0 and res.eval_result["fn"] == 0
        assert res.eval_result["inspection_steps"] == 2
        assert res.eval_result["reward"] == pytest.approx(compute_reward(1.0, 2))
        _assert_trajectory_schema(res.trajectory, ep.episode_id)
        _assert_no_leaks(res.trajectory, ep, seed, str(tmp_path))
        if mode == "eval":
            assert all(step["reward"] == 0.0 for step in res.trajectory)
        else:
            assert res.trajectory[-1]["reward"] == pytest.approx(compute_reward(1.0, 2))
            assert all(step["reward"] == 0.0 for step in res.trajectory[:-1])

    assert seq_res.eval_result["reward"] == pytest.approx(rnd_res.eval_result["reward"])


def test_runner_hostile_agent_finds_nothing(monkeypatch, tmp_path):
    seed = 777123456
    monkeypatch.setenv("AUDIT_SECRET_SEED", str(seed))
    monkeypatch.setenv("TRUTH_PATH", str(tmp_path / "private" / "truth.json"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT))

    cmd = R.dummy_agent_command("hostile", python_executable=sys.executable)
    res = R.run_episode_subprocess(seed, cmd, config=DIRTY_BASE, mode="eval")
    assert res.error is None and res.agent_exit_code == 0
    _assert_trajectory_schema(res.trajectory, res.episode_id)
    assert all(step["reward"] == 0.0 for step in res.trajectory)

    explanation = res.trajectory[-1]["action"]["findings"]["findings"][0]["explanation"]
    report = json.loads(explanation)
    assert report["live"] == []
    assert report["modules"] == []
    assert report["env"] == []
    assert report["cwd"] == []
    assert report["files"] == []
    assert report["leaks"] == []
    assert "AUDIT_SECRET_SEED" not in report["env_keys"]
    assert "TRUTH_PATH" not in report["env_keys"]
    assert "PYTHONPATH" not in report["env_keys"]


def test_runner_isolated_agent_cannot_import_project_modules(tmp_path):
    probe = tmp_path / "import_probe.py"
    probe.write_text(
        "import json, sys\n"
        "msg = json.loads(sys.stdin.readline())\n"
        "ep = msg['observation']['manifest']['episode_id']\n"
        "blocked = []\n"
        "for mod in ('env', 'evaluator', 'generator', 'runner', 'schemas'):\n"
        "    try:\n"
        "        __import__(mod)\n"
        "    except ModuleNotFoundError:\n"
        "        blocked.append(mod)\n"
        "out = {'action': {'tool': 'submit', 'findings': {'episode_id': ep, 'findings': [{'type': 'duplicate_invoice', 'affected_ids': ['X'], 'claimed_amount_minor': 0, 'currency': 'KZT', 'evidence_ids': [], 'explanation': ','.join(blocked)}]}} }\n"
        "sys.stdout.write(json.dumps(out) + '\\n'); sys.stdout.flush()\n"
        "sys.stdin.readline()\n",
        encoding="utf-8",
    )
    res = R.run_episode_subprocess(1, [sys.executable, str(probe)], config=CLEAN_BASE, mode="eval")
    assert res.error is None and res.agent_exit_code == 0
    blocked = res.trajectory[-1]["action"]["findings"]["findings"][0]["explanation"]
    assert blocked == "env,evaluator,generator,runner,schemas"


def test_runner_badjson_survives_and_reaches_submit():
    cmd = R.dummy_agent_command("badjson", python_executable=sys.executable)
    res = R.run_episode_subprocess(2, cmd, config=CLEAN_BASE, mode="train")
    assert res.error is None and res.agent_exit_code == 0
    assert res.eval_result["valid_submission"] is True
    assert res.eval_result["truncated"] is False
    assert res.eval_result["f1"] == 1.0
    assert res.eval_result["inspection_steps"] == 4
    assert res.eval_result["reward"] == pytest.approx(compute_reward(1.0, 4))
    _assert_trajectory_schema(res.trajectory, res.episode_id)
    # Steps 1..4 are invalid_action errors; step 5 is the final submit.
    for step_idx in range(1, 5):
        assert res.trajectory[step_idx]["action"] is None
        assert res.trajectory[step_idx]["observation"]["kind"] == "error"
        assert res.trajectory[step_idx]["observation"]["error"]["code"] == "invalid_action"
    assert res.trajectory[5]["terminated"] is True


def test_runner_crash_truncates_cleanly():
    cmd = R.dummy_agent_command("crash", python_executable=sys.executable)
    res = R.run_episode_subprocess(2, cmd, config=DIRTY_BASE, mode="eval")
    assert res.error == "agent_exited"
    assert res.agent_exit_code is not None and res.agent_exit_code != 0
    assert res.eval_result["truncated"] is True
    assert res.eval_result["inspection_steps"] == 1
    assert res.eval_result["reward"] == -1.0
    _assert_trajectory_schema(res.trajectory, res.episode_id)
    assert res.trajectory[-1]["truncated"] is True
    assert all(step["reward"] == 0.0 for step in res.trajectory)


def test_runner_eof_truncates_cleanly():
    cmd = R.dummy_agent_command("eof", python_executable=sys.executable)
    res = R.run_episode_subprocess(2, cmd, config=CLEAN_BASE, mode="eval")
    assert res.error == "agent_exited"
    assert res.eval_result["truncated"] is True
    assert res.eval_result["inspection_steps"] == 0
    assert res.eval_result["f1"] == 1.0  # empty answer on clean book
    _assert_trajectory_schema(res.trajectory, res.episode_id)
    assert res.trajectory[-1]["truncated"] is True


def test_runner_oversize_kills_process_and_truncates():
    cmd = R.dummy_agent_command("oversize", python_executable=sys.executable)
    res = R.run_episode_subprocess(2, cmd, config=DIRTY_BASE, mode="eval", timeout_sec=10.0)
    assert res.error == "oversized_message"
    assert res.eval_result["truncated"] is True
    assert res.eval_result["reward"] == -1.0
    assert res.elapsed_sec < 3.5  # killed promptly, did not sleep for 5s
    _assert_trajectory_schema(res.trajectory, res.episode_id)
    assert res.trajectory[-1]["truncated"] is True


def test_runner_timeout_kills_slow_process_quickly():
    cmd = R.dummy_agent_command("slow", python_executable=sys.executable, sleep_sec=10.0)
    res = R.run_episode_subprocess(2, cmd, config=DIRTY_BASE, mode="eval", timeout_sec=1.0)
    assert res.error == "timeout"
    assert res.eval_result["truncated"] is True
    assert res.eval_result["reward"] == -1.0
    assert 0.8 <= res.elapsed_sec < 3.5
    _assert_trajectory_schema(res.trajectory, res.episode_id)
    assert res.trajectory[-1]["truncated"] is True


def test_runner_spawn_failure_is_handled_gracefully():
    res = R.run_episode_subprocess(2, ["nonexistent_agent_binary_xyz_12345"],
                                   config=DIRTY_BASE, mode="eval", isolate_python=False)
    assert res.error is not None and res.error.startswith("spawn_failed:")
    assert res.agent_exit_code is None
    assert res.eval_result["truncated"] is True
    _assert_trajectory_schema(res.trajectory, res.episode_id)


def test_runner_cli_and_trajectory_export(tmp_path, capsys):
    out_file = tmp_path / "traj.jsonl"
    rc = R.main([
        "--seed", "4",
        "--mode", "eval",
        "--id-scheme", "randomized",
        "--id-salt", "3",
        "--dummy-agent", "silent",
        "--trajectory-out", str(out_file),
    ])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["agent_exit_code"] == 0 and summary["error"] is None
    lines = S.parse_jsonl(out_file.read_text(encoding="utf-8"))
    _assert_trajectory_schema(lines, summary["episode_id"])
    assert all(step["reward"] == 0.0 for step in lines)
