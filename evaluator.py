"""evaluator.py - isolated grader (1-to-1 matching, F1, terminal reward).

This module imports only the standard library and ``schemas``.  It never
imports the generator or the environment: it receives the reference errors
(the content of ``truth.json``) as plain data.  The environment owns an
``Evaluator`` instance; nothing it knows is ever copied into observations.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import schemas as S


@dataclass(frozen=True)
class EvalResult:
    """Outcome of grading one episode."""

    reward: float
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    inspection_steps: int
    valid_submission: bool
    truncated: bool
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "reward": self.reward, "f1": self.f1, "precision": self.precision,
            "recall": self.recall, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "inspection_steps": self.inspection_steps,
            "valid_submission": self.valid_submission, "truncated": self.truncated,
        }

    def public_diagnostics(self) -> dict:
        """Counters only: no ids, no reference content."""
        if not self.valid_submission:
            return {"valid_submission": False}
        return {
            "valid_submission": True, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
        }


@dataclass(frozen=True)
class _Reference:
    type: str
    affected: tuple
    amount: int
    currency: str
    evidence_sets: tuple  # tuple of frozensets


def compute_reward(f1: float, inspection_steps: int) -> float:
    """reward = max(-1, min(1, 2*F1 - 1 - 0.002 * inspection_steps))."""
    return max(-1.0, min(1.0, 2.0 * f1 - 1.0 - S.STEP_PENALTY * inspection_steps))


def _matches(finding: S.Finding, ref: _Reference, tolerance: int,
             public_ids: Optional[frozenset] = None) -> bool:
    if finding.type != ref.type:
        return False
    if finding.affected_ids != ref.affected:
        return False
    if finding.currency != ref.currency:
        return False
    if abs(finding.claimed_amount_minor - ref.amount) > tolerance:
        return False
    evidence = set(finding.evidence_ids)
    if public_ids is not None and not evidence <= public_ids:
        # a nonexistent / unavailable evidence id makes the finding unmatched (M04)
        return False
    return any(valid <= evidence for valid in ref.evidence_sets)


def _max_matching(findings: Sequence[S.Finding], refs: Sequence[_Reference], tolerance: int,
                  public_ids: Optional[frozenset] = None) -> int:
    """Size of a maximum 1-to-1 matching (augmenting paths, deterministic order).

    TP/FP/FN depend only on the size of the matching, which is unique, so the
    order of findings can change which identical finding is matched but never
    the counts or the reward."""
    adjacency = [[j for j, ref in enumerate(refs) if _matches(f, ref, tolerance, public_ids)]
                 for f in findings]
    owner = [-1] * len(refs)

    def augment(i: int, seen: set) -> bool:
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(1 for i in range(len(findings)) if augment(i, set()))


class Evaluator:
    """Grades a submission against the reference errors of one episode."""

    def __init__(self, episode_id: str, truth: Sequence[Mapping[str, Any]],
                 amount_tolerance_minor: int, public_ids: Optional[Any] = None) -> None:
        """``public_ids``: every id an agent can reach with the tools (ledger
        row_id, document doc_id, fx id).  When given, a finding that cites any
        other evidence id is not matched."""
        if not S.is_int(amount_tolerance_minor) or amount_tolerance_minor < 0:
            raise S.SchemaError("amount_tolerance_minor: must be a non-negative int")
        if not isinstance(truth, (list, tuple)):
            raise S.SchemaError("truth: must be a list")
        refs = []
        seen_ids = set()
        for entry in copy.deepcopy(list(truth)):
            S.validate_truth_entry(entry)
            if entry["issue_id"] in seen_ids:
                raise S.SchemaError("truth: duplicate issue_id")
            seen_ids.add(entry["issue_id"])
            refs.append(_Reference(
                entry["type"], tuple(sorted(set(entry["affected_ids"]))),
                entry["expected_amount_minor"], entry["currency"],
                tuple(frozenset(ev) for ev in entry["valid_evidence_sets"])))
        self._episode_id = episode_id
        # reference errors ordered by issue_id (published tie-break rule); counts never depend on it
        order = sorted(range(len(refs)), key=lambda i: truth[i]["issue_id"])
        self._refs = tuple(refs[i] for i in order)
        self._tolerance = amount_tolerance_minor
        self._public_ids = frozenset(public_ids) if public_ids is not None else None
        self._final: Optional[EvalResult] = None
        self._log: list = []

    @classmethod
    def from_files(cls, truth_path: Any, episode_id: str, amount_tolerance_minor: int) -> "Evaluator":
        truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
        return cls(episode_id, truth, amount_tolerance_minor)

    # ------------------------------------------------------------------ #
    def evaluate(self, payload: Any, inspection_steps: int, *, truncated: bool = False) -> EvalResult:
        """Pure scoring (no state change).

        ``truncated=True`` grades the empty answer used when the step budget
        runs out; ``payload`` is ignored in that case.
        """
        if truncated:
            payload = {"episode_id": self._episode_id, "findings": []}
        try:
            findings = S.parse_submission(payload, self._episode_id)
        except S.SchemaError as exc:
            return EvalResult(-1.0, 0.0, 0.0, 0.0, 0, 0, len(self._refs), inspection_steps,
                              False, truncated, error=str(exc))

        tp = _max_matching(findings, self._refs, self._tolerance, self._public_ids)
        fp = len(findings) - tp
        fn = len(self._refs) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if not self._refs:
            f1 = 1.0 if not findings else 0.0
        else:
            f1 = (2.0 * tp) / (2 * tp + fp + fn)
        reward = compute_reward(f1, inspection_steps)
        return EvalResult(reward, f1, precision, recall, tp, fp, fn, inspection_steps,
                          True, truncated)

    def finalize(self, payload: Any, inspection_steps: int, *, truncated: bool = False,
                 reason: Optional[str] = None) -> EvalResult:
        """Grade the episode once and keep the result inside the evaluator."""
        if self._final is not None:
            raise RuntimeError("episode already graded")
        result = self.evaluate(payload, inspection_steps, truncated=truncated)
        if result.error is not None:
            self._log.append(f"invalid submission: {result.error}")
        if truncated:
            if reason is None:
                self._log.append("step budget exhausted: empty answer graded")
            else:
                self._log.append(f"episode aborted ({reason}): empty answer graded")
        self._final = result
        return result

    # ------------------------------------------------------------------ #
    def final_result(self) -> Optional[dict]:
        """Final result (reward and diagnostics) or ``None`` before grading."""
        return None if self._final is None else self._final.as_dict()

    @property
    def error_log(self) -> list:
        """Evaluator-side journal (e.g. invalid submissions)."""
        return list(self._log)
