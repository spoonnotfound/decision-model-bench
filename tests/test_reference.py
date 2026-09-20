from pathlib import Path

from decision_model_bench.core import load_task, read_jsonl, request_digest, request_for
from decision_model_bench.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_published_reference_recomputes_without_any_model_or_key():
    task, cases, _ = load_task(ROOT / "tasks/turtlebench-en.json")
    expected_hashes = {c["id"]: request_digest(request_for(c, task)) for c in cases}
    assert len(cases) == 1532 and len({c["group"] for c in cases}) == 32
    for model, correct in [("jev", 1299), ("laya-typed-decisions", 956), ("laya-english", 1009)]:
        rows = read_jsonl(ROOT / "results/turtlebench-en-20260920" / (model + ".jsonl"))
        assert all(r["request_hash"] == expected_hashes[r["case_id"]] for r in rows)
        s = summarize(task, cases, rows)
        assert s["correct"] == correct and s["attempted"] == 1532
        if model == "jev":
            assert s["failed_attempts"] == 2 and s["first_attempt_accuracy"] < s["accuracy"]
