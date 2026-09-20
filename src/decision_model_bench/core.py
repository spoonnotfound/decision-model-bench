from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class InvalidDecision(ValueError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {Path(path).name}:{number}; preserve/repair before resume"
                ) from exc
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_task(path):
    path = Path(path)
    task = json.loads(path.read_text(encoding="utf-8"))
    question = task["question"]
    if question.get("type") != "choice" or not isinstance(question.get("instructions"), str):
        raise ValueError("v0.1 supports choice tasks with explicit instructions")
    options = question.get("criteria")
    if (
        not isinstance(options, dict)
        or len(options) < 2
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in options.items())
    ):
        raise ValueError("criteria must map at least two option names to descriptions")
    groups = task.get("label_groups", {k: k for k in options})
    if set(groups) != set(options) or not all(isinstance(v, str) for v in groups.values()):
        raise ValueError("label_groups must cover every option exactly")
    if task.get("positive_group") is not None and (
        len(set(groups.values())) != 2 or task["positive_group"] not in groups.values()
    ):
        raise ValueError("positive_group requires a binary grouped task")
    rows = read_jsonl(path.parent / task["data"])
    ids = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Case IDs must be unique strings")
        ids.add(row["id"])
        if row.get("target") not in options or not isinstance(row.get("state"), str):
            raise ValueError("Each case needs a known target and string state")
        if not isinstance(row.get("group", row["id"]), str):
            raise TypeError("Case group must be a string")
    if not rows:
        raise ValueError("Empty task")
    # Including all case content means changed labels also invalidate cached scoring.
    return task, rows, digest({"task": task, "cases": rows})


def request_for(case, task):
    # Deliberately never pass target, group or dataset metadata to a model.
    return {"id": case["id"], "state": case["state"], "questions": {"judge": task["question"]}}


def request_digest(request):
    return digest({"state": request["state"], "questions": request["questions"]})


def validate_decision(decision, task):
    options = task["question"]["criteria"]
    choice = decision.get("choice")
    probabilities = decision.get("probabilities")
    if not isinstance(choice, str) or choice not in options:
        raise InvalidDecision("Unknown choice")
    if not isinstance(probabilities, dict) or set(probabilities) != set(options):
        raise InvalidDecision("Probability keys do not match options")
    if any(
        isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise InvalidDecision("Probabilities must be finite numbers in [0, 1]")
    mass = sum(probabilities.values())
    tolerance = task.get("probability_sum_tolerance", 0.011)
    if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 0.05:
        raise ValueError("Probability tolerance must be between 0 and 0.05")
    if mass <= 0 or abs(mass - 1) > tolerance:
        raise InvalidDecision("Probability mass outside declared rounding tolerance")
    if probabilities[choice] + 1e-8 < max(probabilities.values()):
        raise InvalidDecision("Choice must be an argmax of the reported options (ties allowed)")
    return {"choice": choice, "probabilities": dict(probabilities), "probability_mass": mass}
