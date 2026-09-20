"""Import a historical study into the schema without copying credentials or host paths."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_model_bench.core import digest, load_task, read_jsonl, request_digest, request_for, write_json
from decision_model_bench.metrics import summarize


def main(source, output):
    root = Path(__file__).resolve().parents[1]
    source, output = Path(source), Path(output)
    task, cases, task_hash = load_task(root / "tasks/turtlebench-en.json")
    output.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((source / "protocol.json").read_text())
    assert source_protocol["question"] == task["question"]
    lookup = {c["id"]: c for c in cases}
    failures = read_jsonl(source / "jev-errors.jsonl") if (source / "jev-errors.jsonl").exists() else []
    summaries = {}
    for name in ["jev", "laya-typed-decisions", "laya-english"]:
        original = read_jsonl(source / f"{name}-predictions.jsonl")
        assert len(original) == 1532 and len({r["id"] for r in original}) == 1532
        imported = []
        for old in sorted(original, key=lambda x: x["id"]):
            case_id = str(old["id"])
            case = lookup[case_id]
            assert old["state"] == case["state"] and old["label"] == case["target"]
            request_hash = request_digest(request_for(case, task))
            attempts = [f for f in failures if f["case_id"] == old["id"]] if name == "jev" else []
            for i, error in enumerate(attempts, 1):
                imported.append(
                    {
                        "case_id": case_id,
                        "attempt": i,
                        "request_hash": request_hash,
                        "case_hash": digest(case),
                        "status": "error",
                        "latency_s": None,
                        "error": {"code": "historical_gateway_error", "http_status": error["status"]},
                    }
                )
            ans = old["response"]["answers"]["judge"]
            imported.append(
                {
                    "case_id": case_id,
                    "attempt": len(attempts) + 1,
                    "request_hash": request_hash,
                    "case_hash": digest(case),
                    "status": "ok",
                    "decision": {"choice": ans["choice"], "probabilities": ans["probabilities"]},
                    "latency_s": old.get("latency_s"),
                    "metadata": {
                        "usage": old["response"]["usage"],
                        "historical_import": True,
                        "batch_size": old.get("batch_size"),
                        "batch_latency_s": old.get("batch_latency_s"),
                    },
                }
            )
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in imported)
        )
        summaries[name] = summarize(task, cases, imported)
    write_json(output / "summary.json", summaries)
    manifest = {
        "kind": "historical exploratory study; imported, not rerun by this new harness",
        "study_date": "2026-09-20",
        "task_hash": task_hash,
        "dataset_revision": source_protocol["dataset_commit"],
        "laya_revision": source_protocol["model_revision"],
        "jev_requested_and_returned": "typesafe-ai/jev",
        "jev_backend_revision": None,
        "jev_route": "Vercel AI Gateway",
        "jev_concurrency": "probe 1; main 4; remaining 127 cases serial",
        "laya_platform": "Apple M1, float32, official full weights; no quantization",
        "laya_english_execution": "1-550 MPS (restart after 406); 551-985 CPU single; 986-1532 CPU batch 4",
        "parity": "12 CPU/MPS and 16 batch/single probes: same choices and SDK-rounded probabilities",
        "timing_comparable": False,
        "note": "Accuracy comparison. Unknown and Incorrect merged. English transfer test is not typed-decisions claim replication.",
    }
    write_json(output / "study.json", manifest)
    for filename in ["weights-audit.json", "laya-english-cpu-parity.json", "laya-english-batch-parity.json"]:
        write_json(output / filename, json.loads((source / filename).read_text()))
    print({k: v["correct"] for k, v in summaries.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Local historical study directory; never committed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.source, args.out)
