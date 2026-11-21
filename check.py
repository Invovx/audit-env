"""check.py - the single verification command.

    python check.py              # tests + episodes + baselines + trajectories + report
    python check.py --skip-tests # same without pytest

Steps:
1. run the full pytest suite;
2. generate the open train/dev episodes from ``configs/*.json`` into
   ``artifacts/episodes/<split>/<episode_id>/{public,private}``;
3. run the four baseline policies on every episode (train mode);
4. write ``artifacts/trajectories.jsonl`` (agent policies only - no closed
   reference data appears in any observation) and ``artifacts/report.json``;
5. check the invariants the jury cares about and exit non-zero on failure.

No network, no GPU, no paid API.  Python 3.11, standard library + pytest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import schemas as S  # noqa: E402
from baselines import AGENT_POLICIES, OraclePolicy  # noqa: E402
from env import AuditEnv  # noqa: E402
from generator import generate_episode, write_episode  # noqa: E402
from runner import dummy_agent_command, run_episode_subprocess  # noqa: E402

ART = ROOT / "artifacts"
SPLITS = ("train", "dev")
EPISODE_TIME_LIMIT_S = 60.0
LEAK_MARKERS = ("ISS-", "MUT-", "truth", "provenance", "valid_evidence_sets", "expected_amount_minor")


def load_split(name: str) -> dict:
    data = json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    assert data["split"] == name and data["config_version"]
    return data


def run_episode(policy, seed: int, config: dict, mode: str = "train") -> tuple:
    """Play one episode. Returns (transcript rows, final evaluator result, seconds)."""
    t0 = time.perf_counter()
    env = AuditEnv(mode)
    obs, _info = env.reset(seed, config)
    policy.reset(obs)
    episode_id = obs["manifest"]["episode_id"]
    rows = []
    step = 0
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        step += 1
        rows.append({"episode_id": episode_id, "step": step, "action": action, "observation": obs,
                     "reward": reward, "terminated": terminated, "truncated": truncated})
        if terminated or truncated:
            break
    return rows, env.evaluator.final_result(), time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    failures: list = []

    print(f"Python {sys.version.split()[0]} | schema {S.SCHEMA_VERSION}")
    if not args.skip_tests:
        print("\n[1/4] pytest")
        code = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT)
        if code != 0:
            failures.append("pytest failed")

    print("\n[2/4] generating open episodes")
    ART.mkdir(exist_ok=True)
    episodes = []
    digest = hashlib.sha256()
    for split in SPLITS:
        spec = load_split(split)
        for item in spec["episodes"]:
            ep = generate_episode(item["seed"], item["config"])
            again = generate_episode(item["seed"], item["config"])
            if ep.public_files() != again.public_files() or ep.private_files() != again.private_files():
                failures.append(f"non-deterministic episode {ep.episode_id}")
            write_episode(ep, ART / "episodes" / split / ep.episode_id)
            (ART / "episodes" / split / ep.episode_id / "config.json").write_text(
                S.dump_json({"config_version": spec["config_version"], "seed": item["seed"],
                             "config": item["config"]}), encoding="utf-8")
            for text in ep.public_files().values():
                digest.update(text.encode("utf-8"))
            episodes.append((split, item["seed"], item["config"], ep))
    print(f"  {len(episodes)} episodes, public data sha256 = {digest.hexdigest()[:16]}")

    print("\n[3/4] running baseline policies")
    results: dict = {}
    with (ART / "trajectories.jsonl").open("w", encoding="utf-8") as traj:
        for split, seed, config, ep in episodes:
            policies = [cls() for cls in AGENT_POLICIES] + [OraclePolicy(ep.truth)]
            for policy in policies:
                rows, final, seconds = run_episode(policy, seed, config)
                if seconds > EPISODE_TIME_LIMIT_S:
                    failures.append(f"{policy.name} on {ep.episode_id}: {seconds:.1f}s > 60s")
                if policy.name != "oracle":            # oracle actions carry reference data
                    for row in rows:
                        blob = json.dumps(row["observation"], ensure_ascii=False)
                        if any(m in blob for m in LEAK_MARKERS):
                            failures.append(f"leak marker in observation of {ep.episode_id}")
                        traj.write(json.dumps({**row, "policy": policy.name, "split": split},
                                              ensure_ascii=False, separators=(",", ":")) + "\n")
                results.setdefault((split, policy.name), []).append(
                    {**final, "episode_id": ep.episode_id, "clean": not ep.truth, "seconds": round(seconds, 4)})

    print("\n[4/4] report  (diagnostics only - the jury scores the environment, not these numbers)")
    header = f"{'split':6} {'policy':10} {'P':>5} {'R':>5} {'F1':>5} {'return':>7} {'steps':>6} {'sec/ep':>7}"
    print(header + "\n" + "-" * len(header))
    report: dict = {}
    for (split, name), items in sorted(results.items()):
        mean = lambda k: statistics.fmean(i[k] for i in items)  # noqa: E731
        row = {k: round(mean(k), 4) for k in ("precision", "recall", "f1", "reward", "inspection_steps", "seconds")}
        row["episodes"] = len(items)
        report.setdefault(split, {})[name] = {"mean": row, "episodes": items}
        print(f"{split:6} {name:10} {row['precision']:5.2f} {row['recall']:5.2f} {row['f1']:5.2f} "
              f"{row['reward']:7.3f} {row['inspection_steps']:6.1f} {row['seconds']:7.3f}")

    # invariants: oracle reaches the maximum, attacks are punished
    for split in SPLITS:
        oracle = results[(split, "oracle")]
        if any(i["f1"] != 1.0 or i["reward"] != 1.0 for i in oracle):
            failures.append(f"{split}: oracle did not reach F1=1 / reward=1 on every episode")
        if statistics.fmean(i["reward"] for i in results[(split, "flag_all")]) >= 0:
            failures.append(f"{split}: flag-everything attack is not punished")
        for policy in ("random", "reconcile", "flag_all"):
            if any(i["reward"] > 1.0 or i["reward"] < -1.0 for i in results[(split, policy)]):
                failures.append(f"{split}: reward out of [-1, 1] for {policy}")

    print("\n[+] isolated runner: agent in a separate process, evaluator stays in the harness")
    isolated = []
    for split, seed, config, ep in episodes[:3]:
        for agent in ("random", "hostile"):
            res = run_episode_subprocess(seed, dummy_agent_command(agent, agent_seed=seed), config=config)
            blob = json.dumps(res.trajectory, ensure_ascii=False)
            leaked = any(m in blob for m in LEAK_MARKERS) or any(
                t["issue_id"] in blob for t in ep.truth)
            if leaked:
                failures.append(f"runner: {agent} saw closed data on {ep.episode_id}")
            isolated.append({"episode_id": res.episode_id, "agent": agent, "error": res.error,
                             "reward": res.eval_result.get("reward"), "leak": leaked,
                             "seconds": round(res.elapsed_sec, 3)})
            print(f"  {split:5} {agent:8} reward={res.eval_result.get('reward')!s:7} "
                  f"leak={leaked} {res.elapsed_sec:.2f}s")

    (ART / "report.json").write_text(S.dump_json({"public_data_sha256": digest.hexdigest(),
                                                   "failures": failures, "results": report,
                                                   "isolated_runner": isolated}),
                                     encoding="utf-8")
    print("\nartifacts: artifacts/episodes/, artifacts/trajectories.jsonl, artifacts/report.json")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
