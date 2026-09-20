import math

import pytest

from decision_model_bench.core import InvalidDecision, request_for, validate_decision
from decision_model_bench.metrics import paired_bootstrap, summarize

TASK = {
    "id": "test",
    "question": {
        "type": "choice",
        "instructions": "Judge the claim.",
        "criteria": {"T": "supported", "F": "contradicted", "N": "unknown"},
    },
    "label_groups": {"T": "yes", "F": "no", "N": "no"},
    "positive_group": "yes",
}
CASES = [
    {"id": "1", "group": "story", "state": "A fact", "target": "T"},
    {"id": "2", "group": "story", "state": "Another fact", "target": "N"},
]


def record(case_id, choice, probs, attempt=1):
    return {
        "case_id": case_id,
        "attempt": attempt,
        "status": "ok",
        "decision": {"choice": choice, "probabilities": dict(zip(["T", "F", "N"], probs))},
        "latency_s": 0.2,
    }


def test_target_never_reaches_adapter():
    request = request_for({**CASES[0], "private_metadata": "do not pass"}, TASK)
    assert set(request) == {"id", "state", "questions"}
    assert "target" not in request and "private_metadata" not in request


def test_merge_happens_after_original_choice():
    # T wins the original 3-way decision, but F+N has more mass. Accuracy must use T.
    s = summarize(TASK, CASES[:1], [record("1", "T", [0.4, 0.35, 0.25])])
    assert s["accuracy"] == 1
    assert s["event_brier"] == pytest.approx(0.36)
    assert s["multiclass_brier"] == pytest.approx(0.72)


def test_unknown_is_not_scored_as_separate_class_in_turtlebench():
    s = summarize(TASK, CASES[1:], [record("2", "F", [0.1, 0.8, 0.1])])
    assert s["accuracy"] == 1
    assert s["event_brier"] == pytest.approx(0.01)


def test_failure_and_unattempted_are_not_silently_dropped():
    rows = [{"case_id": "1", "attempt": 1, "status": "error", "latency_s": None}]
    s = summarize(TASK, CASES, rows)
    assert s["accuracy"] == 0 and s["attempted"] == 1 and s["total"] == 2
    assert s["event_brier"] is None and s["valid_latest"] == 0


def test_retry_reports_first_attempt_and_final_accuracy():
    rows = [{"case_id": "1", "attempt": 1, "status": "error"}, record("1", "T", [1, 0, 0], 2)]
    s = summarize(TASK, CASES[:1], rows)
    assert s["accuracy"] == 1 and s["first_attempt_accuracy"] == 0 and s["failed_attempts"] == 1


def test_rounding_normalization_is_explicit_and_raw_score_retained():
    s = summarize(TASK, CASES[:1], [record("1", "T", [0.5, 0.3, 0.19])])
    assert s["renormalized_probability_count"] == 1
    assert s["event_brier"] == pytest.approx((0.5 / 0.99 - 1) ** 2)
    assert s["raw_event_brier"] == 0.25


def test_zero_probability_nll_is_clipped_and_reported():
    s = summarize(TASK, CASES[:1], [record("1", "F", [0, 1, 0])])
    assert s["zero_target_probability_count"] == 1
    assert s["clipped_nll"] == pytest.approx(-math.log(1e-12))


@pytest.mark.parametrize(
    "probs",
    [
        [float("nan"), 0.5, 0.5],
        [float("inf"), 0, 0],
        [-0.1, 0.6, 0.5],
        [True, 0, 0],
        [0.2, 0.2, 0.2],
        [0, 0, 0],
    ],
)
def test_invalid_distributions_rejected(probs):
    with pytest.raises(InvalidDecision):
        validate_decision(record("1", "T", probs)["decision"], TASK)


def test_unknown_option_and_nonargmax_rejected():
    for choice in ["invented", "N"]:
        with pytest.raises(InvalidDecision):
            validate_decision(record("1", choice, [0.8, 0.1, 0.1])["decision"], TASK)


def test_selective_risk_keeps_ties_and_zero_coverage_is_null():
    s = summarize(TASK, CASES, [record("1", "T", [0.5, 0.3, 0.2]), record("2", "T", [0.5, 0.3, 0.2])])
    at_half = next(x for x in s["selective_risk"] if x["threshold"] == 0.5)
    assert at_half["coverage"] == 1 and at_half["error_rate"] == 0.5
    assert s["selective_risk"][-1]["error_rate"] is None


def test_bootstrap_uses_story_groups_and_rejects_incomplete():
    good = [record("1", "T", [1, 0, 0]), record("2", "N", [0, 0, 1])]
    bad = [record("1", "F", [0, 1, 0]), record("2", "T", [1, 0, 0])]
    result = paired_bootstrap(TASK, CASES, good, bad, repeats=100)
    assert result["groups"] == 1 and result["interval_95"] == [1, 1]
    with pytest.raises(ValueError):
        paired_bootstrap(TASK, CASES, good[:1], bad, repeats=100)
