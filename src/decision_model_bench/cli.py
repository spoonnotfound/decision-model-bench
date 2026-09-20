import argparse
import json
import tomllib
from pathlib import Path

from .core import load_task, read_jsonl, write_json
from .metrics import paired_bootstrap, summarize
from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Decision Model Bench: explicit tasks, adapters and scoring")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate a task and its dataset without calling any model")
    validate.add_argument("--task", required=True)
    execute = sub.add_parser("run", help="Run or resume an evaluation; remote models require --live")
    execute.add_argument("--task", required=True)
    execute.add_argument("--config", required=True)
    execute.add_argument("--out", required=True)
    execute.add_argument("--limit", type=int)
    execute.add_argument("--live", action="store_true")
    execute.add_argument("--retry-failed", action="store_true")
    execute.add_argument("--max-errors", type=int, default=1)
    execute.add_argument("--max-attempts", type=int, default=3)
    score = sub.add_parser("score", help="Recompute metrics from saved attempts; no model call")
    score.add_argument("--task", required=True)
    score.add_argument("--records", required=True)
    score.add_argument("--out")
    compare = sub.add_parser("compare", help="Paired comparison using dataset groups as resampling units")
    compare.add_argument("--task", required=True)
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--out")
    args = parser.parse_args()
    task, cases, task_hash = load_task(args.task)
    if args.command == "validate":
        result = {"task": task["id"], "cases": len(cases), "task_hash": task_hash}
    elif args.command == "run":
        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit must be positive")
            cases = cases[: args.limit]
        path = Path(args.config)
        config = tomllib.loads(path.read_text())
        for key in ["records", "bridge"]:
            if key in config:
                config[key] = str((path.parent / config[key]).resolve())
        result = run(
            task,
            cases,
            config,
            args.out,
            live=args.live,
            retry_failed=args.retry_failed,
            max_errors=args.max_errors,
            max_attempts=args.max_attempts,
        )
    else:
        paths = [args.records] if args.command == "score" else [args.left, args.right]
        records = [read_jsonl(path) for path in paths]
        # Score/compare refuses a matching ID with a different input or question.
        from .core import digest, request_digest, request_for

        hashes = {c["id"]: request_digest(request_for(c, task)) for c in cases}
        case_hashes = {c["id"]: digest(c) for c in cases}
        for rows in records:
            if any(r.get("case_hash") != case_hashes.get(r["case_id"]) for r in rows):
                raise ValueError("Result case hashes do not match target labels or case metadata")
            if any(r.get("request_hash") != hashes.get(r["case_id"]) for r in rows):
                raise ValueError("Result request hashes do not match this task")
        result = (
            summarize(task, cases, records[0])
            if args.command == "score"
            else paired_bootstrap(task, cases, records[0], records[1])
        )
        if args.out:
            write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.command == "run" and (
        result["attempted"] < result["total"] or result["valid_latest"] < result["total"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
