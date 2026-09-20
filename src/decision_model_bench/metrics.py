"""Scoring is model-independent. Failed and unattempted cases remain visible."""

import collections
import math
import random
import statistics

from .core import validate_decision


def summarize(task, cases, records):
    by_id = {case["id"]: [] for case in cases}
    for row in records:
        if row["case_id"] not in by_id:
            raise ValueError("Unknown case in results")
        prior = by_id[row["case_id"]]
        if row["attempt"] != len(prior) + 1:
            raise ValueError("Attempt sequence must be contiguous per case")
        prior.append(row)
    groups = task.get("label_groups", {k: k for k in task["question"]["criteria"]})
    classes = sorted(set(groups.values()))
    first_correct = latest_correct = 0
    valid = []
    story_scores = collections.defaultdict(list)
    predictions = collections.Counter()
    class_stats = {k: {"n": 0, "correct": 0} for k in classes}
    for case in cases:
        attempts = by_id[case["id"]]
        target = groups[case["target"]]
        class_stats[target]["n"] += 1

        def is_correct(row, target=target):
            if row.get("status") != "ok":
                return False
            decision = validate_decision(row["decision"], task)
            return groups[decision["choice"]] == target

        correct = bool(attempts) and is_correct(attempts[-1])
        latest_correct += correct
        first_correct += bool(attempts) and is_correct(attempts[0])
        class_stats[target]["correct"] += correct
        story_scores[case.get("group", case["id"])].append(int(correct))
        if attempts and attempts[-1].get("status") == "ok":
            row = attempts[-1]
            decision = validate_decision(row["decision"], task)
            raw = decision["probabilities"]
            mass = sum(raw.values())
            probs = {k: sum(v for option, v in raw.items() if groups[option] == k) / mass for k in classes}
            chosen = groups[decision["choice"]]
            item = {
                "correct": correct,
                "confidence": probs[chosen],
                "probabilities": probs,
                "target": target,
                "raw": raw,
                "mass": mass,
                "row": row,
            }
            valid.append(item)
            predictions[decision["choice"]] += 1

    def mean(values):
        values = list(values)
        return statistics.mean(values) if values else None

    bins = []
    for i in range(10):
        subset = [x for x in valid if min(int(x["confidence"] * 10), 9) == i]
        bins.append(
            {
                "low": i / 10,
                "high": (i + 1) / 10,
                "n": len(subset),
                "mean_probability": mean(x["confidence"] for x in subset),
                "accuracy": mean(x["correct"] for x in subset),
            }
        )
    selective = []
    for threshold in [0, 0.5, 0.7, 0.9, 0.95]:
        subset = [x for x in valid if x["confidence"] >= threshold]
        selective.append(
            {
                "threshold": threshold,
                "n": len(subset),
                "coverage": len(subset) / len(cases),
                "error_rate": mean(not x["correct"] for x in subset),
            }
        )
    latencies = [x["row"]["latency_s"] for x in valid if isinstance(x["row"].get("latency_s"), (int, float))]
    labels = collections.Counter(groups[x["target"]] for x in cases)
    result = {
        "total": len(cases),
        "attempted": sum(bool(x) for x in by_id.values()),
        "valid_latest": len(valid),
        "attempt_count": len(records),
        "failed_attempts": sum(r.get("status") != "ok" for r in records),
        "correct": latest_correct,
        "accuracy": latest_correct / len(cases),
        "first_attempt_accuracy": first_correct / len(cases),
        "macro_group_accuracy": mean(mean(v) for v in story_scores.values()),
        "majority_baseline": max(labels.values()) / len(cases),
        "class_recall": {k: v["correct"] / v["n"] if v["n"] else None for k, v in class_stats.items()},
        "prediction_counts": dict(predictions),
        "multiclass_brier": mean(
            sum((x["probabilities"][k] - int(x["target"] == k)) ** 2 for k in classes) for x in valid
        ),
        "clipped_nll": mean(-math.log(max(x["probabilities"][x["target"]], 1e-12)) for x in valid),
        "zero_target_probability_count": sum(x["probabilities"][x["target"]] == 0 for x in valid),
        "ece_10_bins": sum(b["n"] * abs(b["mean_probability"] - b["accuracy"]) for b in bins if b["n"])
        / len(valid)
        if valid
        else None,
        "calibration_bins": bins,
        "selective_risk": selective,
        "probability_count": len(valid),
        "renormalized_probability_count": sum(abs(x["mass"] - 1) > 1e-8 for x in valid),
        "latency_count": len(latencies),
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
    }
    if task.get("positive_group"):
        positive = task["positive_group"]
        result["event_brier"] = mean(
            (x["probabilities"][positive] - int(x["target"] == positive)) ** 2 for x in valid
        )
        result["raw_event_brier"] = mean(
            (sum(p for k, p in x["raw"].items() if groups[k] == positive) - int(x["target"] == positive)) ** 2
            for x in valid
        )
    return result


def paired_bootstrap(task, cases, left, right, repeats=2000, seed=20260920):
    """Resample whole story groups, preserving paired questions within each group."""
    # Validate all logs before comparing; incomplete runs should not enter a leaderboard.
    for rows in [left, right]:
        if summarize(task, cases, rows)["attempted"] != len(cases):
            raise ValueError("Paired comparison requires complete attempted coverage")

    def latest(rows):
        return {row["case_id"]: row for row in rows}

    l, r = latest(left), latest(right)
    mapping = task.get("label_groups", {k: k for k in task["question"]["criteria"]})
    groups = collections.defaultdict(list)
    for c in cases:

        def score(row, target=c["target"]):
            return int(row["status"] == "ok" and mapping[row["decision"]["choice"]] == mapping[target])

        groups[c.get("group", c["id"])].append(score(l[c["id"]]) - score(r[c["id"]]))
    rng = random.Random(seed)
    keys = list(groups)
    draws = []
    for _ in range(repeats):
        chosen = [rng.choice(keys) for _ in keys]
        draws.append(sum(sum(groups[k]) for k in chosen) / sum(len(groups[k]) for k in chosen))
    draws.sort()
    return {
        "difference_left_minus_right": sum(map(sum, groups.values())) / len(cases),
        "interval_95": [draws[int(repeats * 0.025)], draws[min(repeats - 1, int(repeats * 0.975))]],
        "resampling_unit": "case.group",
        "groups": len(groups),
        "repeats": repeats,
        "seed": seed,
    }
