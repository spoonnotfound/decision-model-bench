import importlib.metadata
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from .adapters import AdapterError, create_adapter
from .core import (
    InvalidDecision,
    digest,
    read_jsonl,
    request_digest,
    request_for,
    validate_decision,
    write_json,
)
from .metrics import summarize


def now():
    return datetime.now(UTC).isoformat()


def _run(
    task,
    cases,
    config,
    output,
    *,
    live=False,
    retry_failed=False,
    max_errors=1,
    max_attempts=3,
    factory=create_adapter,
):
    if max_errors < 1 or max_attempts < 1:
        raise ValueError("Error and attempt limits must be positive")
    if config["adapter"] in {"gateway", "typesafe"} and not live:
        raise ValueError("Remote adapters require --live; inspect the task/config first")
    if any(k.lower() in {"api_key", "token", "secret", "password", "authorization"} for k in config):
        raise ValueError("Store credentials in environment variables, never in configs")
    allowed = {
        "baseline": {"adapter", "choice"},
        "replay": {"adapter", "records"},
        "laya": {"adapter", "model", "revision", "subfolder", "device", "threads", "max_len"},
        "typesafe": {"adapter", "model", "key_env", "timeout"},
        "gateway": {"adapter", "model", "bridge", "timeout", "node"},
    }
    if config["adapter"] not in allowed or set(config) - allowed[config["adapter"]]:
        raise ValueError("Unknown adapter configuration fields; credentials belong in the environment")
    versions = {}
    for package in ["decision-model-bench", "laya", "torch", "transformers"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources = {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
    fingerprint = digest(
        {"task": task, "cases": cases, "config": config, "source": digest(sources), "packages": versions}
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    log = output / "attempts.jsonl"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: task, cases, adapter, code or dependency versions changed")
    else:
        if log.exists():
            raise ValueError("Refusing orphan log without manifest")
        manifest = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "task_id": task["id"],
            "created_at": now(),
            "config": config,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": versions,
            "task": task,
            "case_ids": [c["id"] for c in cases],
            "sessions": [],
        }
        write_json(manifest_path, manifest)
        write_json(output / "task.json", {**task, "data": "cases.jsonl"})
        (output / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    rows = read_jsonl(log) if log.exists() else []
    expected_requests = {c["id"]: request_digest(request_for(c, task)) for c in cases}
    expected_cases = {c["id"]: digest(c) for c in cases}
    for row in rows:
        if row.get("case_hash") != expected_cases.get(row["case_id"]):
            raise ValueError("Resume refused: case labels or metadata changed")
        if row.get("request_hash") != expected_requests.get(row["case_id"]):
            raise ValueError("Resume refused: logged request does not match current input")
    summarize(task, cases, rows)  # Validate IDs, attempt ordering and decisions before reusing anything.
    attempts = {}
    for row in rows:
        attempts.setdefault(row["case_id"], []).append(row)
    pending = [
        c
        for c in cases
        if not attempts.get(c["id"])
        or (
            retry_failed and attempts[c["id"]][-1]["status"] != "ok" and len(attempts[c["id"]]) < max_attempts
        )
    ]
    if not pending:
        result = summarize(task, cases, rows)
        write_json(output / "summary.json", result)
        return result
    adapter = factory(config)
    manifest["sessions"].append(
        {
            "started_at": now(),
            "adapter": adapter.metadata,
            "retry_failed": retry_failed,
            "max_errors": max_errors,
            "max_attempts_per_case": max_attempts,
        }
    )
    write_json(manifest_path, manifest)
    errors = 0
    try:
        with log.open("a", encoding="utf-8") as stream:
            for case in pending:
                request = request_for(case, task)
                row = {
                    "case_id": case["id"],
                    "attempt": len(attempts.get(case["id"], [])) + 1,
                    "request_hash": request_digest(request),
                    "case_hash": digest(case),
                    "started_at": now(),
                }
                start = time.perf_counter()
                try:
                    raw, metadata = adapter.predict(request)
                    decision = validate_decision(raw, task)
                    row.update(status="ok", decision=decision, metadata=metadata)
                except InvalidDecision:
                    row.update(status="invalid", error={"code": "invalid_decision"})
                    errors += 1
                except AdapterError as exc:
                    row.update(status="error", error={"code": exc.code, "http_status": exc.status})
                    errors += 1
                except Exception:  # noqa: BLE001 -- sanitize arbitrary adapter exception text
                    # Exception messages/bodies can contain tokens or provider request headers.
                    row.update(status="error", error={"code": "adapter_internal_error"})
                    errors += 1
                row["latency_s"] = None if config["adapter"] == "replay" else time.perf_counter() - start
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                print(
                    f"{len({r['case_id'] for r in rows})}/{len(cases)} {case['id']} {row['status']}",
                    flush=True,
                )
                if errors >= max_errors:
                    break
    finally:
        adapter.close()
        manifest["sessions"][-1]["finished_at"] = now()
        write_json(manifest_path, manifest)
        write_json(output / "summary.json", summarize(task, cases, rows))
    return summarize(task, cases, rows)


def run(task, cases, config, output, **kwargs):
    """One writer per output directory. A killed process leaves an explicit stale lock."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            "Run directory is locked; verify the writer has stopped before removing .writer.lock"
        ) from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "started_at": now()}, stream)
        return _run(task, cases, config, output, **kwargs)
    finally:
        lock.unlink()
