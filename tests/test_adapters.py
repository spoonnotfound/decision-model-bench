import io
import json
import urllib.error

import pytest
from test_core_metrics import CASES, TASK

from decision_model_bench.adapters import AdapterError, Replay, TypeSafe
from decision_model_bench.core import request_digest, request_for


def test_direct_transport_sends_no_gold_label_and_no_redirect(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-secret")
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured["body"] = json.loads(request.data)
            captured["url"] = request.full_url
            return io.BytesIO(
                json.dumps(
                    {
                        "model": "jev-test",
                        "answers": {
                            "judge": {"choice": "T", "probabilities": {"T": 0.8, "F": 0.1, "N": 0.1}}
                        },
                        "usage": {"input_tokens": 12},
                    }
                ).encode()
            )

    def opener(handler):
        assert handler.redirect_request(None) is None
        return Opener()

    monkeypatch.setattr("urllib.request.build_opener", opener)
    adapter = TypeSafe({"model": "jev-test"})
    decision, metadata = adapter.predict(request_for(CASES[0], TASK))
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert set(captured["body"]) == {"model", "state", "questions"}
    assert "test-secret" not in json.dumps(captured["body"])
    assert decision["choice"] == "T" and metadata["model_returned"] == "jev-test"


def test_direct_transport_sanitizes_http_error(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-secret")

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 503, "secret-body", {}, None)

    monkeypatch.setattr("urllib.request.build_opener", lambda _: Opener())
    with pytest.raises(AdapterError) as info:
        TypeSafe({"model": "jev-test"}).predict(request_for(CASES[0], TASK))
    assert info.value.status == 503 and str(info.value) == "http_error"


def test_replay_rejects_changed_question_even_when_id_matches(tmp_path):
    request = request_for(CASES[0], TASK)
    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "1",
                "status": "ok",
                "request_hash": request_digest(request),
                "decision": {"choice": "T", "probabilities": {"T": 1, "F": 0, "N": 0}},
            }
        )
        + "\n"
    )
    adapter = Replay({"records": str(path)})
    assert adapter.predict(request)[0]["choice"] == "T"
    changed = {**request, "state": "A different question"}
    with pytest.raises(AdapterError, match="replay_request_mismatch"):
        adapter.predict(changed)
