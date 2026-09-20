import json

import pytest
from test_core_metrics import CASES, TASK

from decision_model_bench.adapters import AdapterError
from decision_model_bench.core import read_jsonl
from decision_model_bench.runner import run


class Fake:
    def __init__(self, fail=False):
        self.metadata = {"adapter": "fake"}
        self.fail = fail
        self.calls = []

    def predict(self, request):
        self.calls.append(request)
        if self.fail:
            raise AdapterError("test_failure", 503)
        return {"choice": "T", "probabilities": {"T": 1.0, "F": 0.0, "N": 0.0}}, {}

    def close(self):
        pass


def test_resume_never_repeats_success_and_rejects_changed_input(tmp_path):
    fake = Fake()
    config = {"adapter": "baseline"}
    run(TASK, CASES, config, tmp_path, factory=lambda _: fake)
    assert len(fake.calls) == 2
    run(TASK, CASES, config, tmp_path, factory=lambda _: pytest.fail("Should not create adapter"))
    changed = [dict(CASES[0], state="Changed"), CASES[1]]
    with pytest.raises(ValueError, match="Resume refused"):
        run(TASK, changed, config, tmp_path, factory=lambda _: fake)


def test_failures_are_retained_across_explicit_retry(tmp_path):
    config = {"adapter": "baseline"}
    run(TASK, CASES[:1], config, tmp_path, factory=lambda _: Fake(True))
    out = run(TASK, CASES[:1], config, tmp_path, retry_failed=True, factory=lambda _: Fake())
    assert out["attempt_count"] == 2 and out["failed_attempts"] == 1
    assert out["accuracy"] == 1 and out["first_attempt_accuracy"] == 0


def test_remote_run_requires_explicit_live_flag(tmp_path):
    with pytest.raises(ValueError, match="--live"):
        run(TASK, CASES, {"adapter": "typesafe"}, tmp_path)


def test_corrupt_log_is_not_silently_skipped(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": true}\n{"partial":')
    with pytest.raises(ValueError, match="Invalid JSON"):
        read_jsonl(p)


def test_exception_details_and_secrets_are_not_logged(tmp_path):
    class Broken(Fake):
        def predict(self, request):
            raise RuntimeError("Authorization: Bearer PRIVATE_TEST_SECRET")

    run(TASK, CASES[:1], {"adapter": "baseline"}, tmp_path, factory=lambda _: Broken())
    assert "PRIVATE_TEST_SECRET" not in (tmp_path / "attempts.jsonl").read_text()
    assert json.loads((tmp_path / "attempts.jsonl").read_text())["error"]["code"] == "adapter_internal_error"


def test_credentials_in_config_are_refused(tmp_path):
    with pytest.raises(ValueError, match="environment"):
        run(TASK, CASES, {"adapter": "baseline", "api_key": "do-not-log"}, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_second_writer_is_refused(tmp_path):
    (tmp_path / ".writer.lock").write_text('{"pid": 123}')
    with pytest.raises(ValueError, match="locked"):
        run(TASK, CASES, {"adapter": "baseline"}, tmp_path, factory=lambda _: Fake())
    assert (tmp_path / ".writer.lock").exists()


def test_snapshots_and_label_hash_are_retained(tmp_path):
    from decision_model_bench.core import digest, load_task

    run(TASK, CASES, {"adapter": "baseline"}, tmp_path, factory=lambda _: Fake())
    snapshot_task, snapshot_cases, _ = load_task(tmp_path / "task.json")
    assert snapshot_cases == CASES and snapshot_task["id"] == TASK["id"]
    rows = read_jsonl(tmp_path / "attempts.jsonl")
    assert rows[0]["case_hash"] == digest(CASES[0])
    assert not (tmp_path / ".writer.lock").exists()


def test_unused_secret_bearing_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="configuration fields"):
        run(TASK, CASES, {"adapter": "baseline", "headers": {"Authorization": "secret"}}, tmp_path)
    assert not (tmp_path / "manifest.json").exists()
