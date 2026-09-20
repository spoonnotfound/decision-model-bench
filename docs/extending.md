# Add tasks and models

## A new task

Copy `tasks/demo.json` and supply UTF-8 JSONL cases:

```json
{"id":"case-1","group":"source-1","state":"The material to judge","target":"OptionA"}
```

`question.criteria` maps option names to their descriptions. Targets must be those names. `label_groups` optionally maps every original option to a scored class; omit it for ordinary multiclass classification. `positive_group` enables binary event metrics. Keep names and order stable across models. Task paths resolve relative to the task JSON file.

Run `dmb validate --task tasks/your-task.json` before any remote calls. Include dataset license, upstream revision, transformation rules, and provenance. Keep training, threshold fitting and evaluation data separate by source/story where related examples could leak.

## A new adapter

Implement `predict(request) -> (decision, metadata)` and `close()`. A request contains only `id`, `state` and a `questions.judge` choice schema. Return:

```json
{"choice":"OptionA","probabilities":{"OptionA":0.8,"OptionB":0.2}}
```

Register it in `create_adapter`. Keep heavyweight dependencies inside the adapter constructor; the core and offline re-scoring must remain usable without a GPU library. Read keys from environment variables. Return selected nonsecret metadata only. Raise `AdapterError` with a stable sanitized code rather than propagating request headers or exception bodies.

The Laya adapter wraps its SDK and refuses truncation. The direct TypeSafe adapter uses the native typed endpoint. The gateway adapter runs a persistent Node evaluation bridge. Replay verifies a request hash and must never be presented as a new live measurement. The constant baseline is an explicit baseline, not a mock AI result.

Test bad distributions, parsing failures, secret-safe logging, interrupted runs and changed-config resume rejection. Validate a new transport with a small real request separately from offline mocked tests.

## Planned extensions

Additional licensed task suites (including a carefully documented typed-decisions task), generic structured-output LLM adapters, ordinal scores, full probability distributions for calibration research, and controlled throughput benchmarks can be added independently. These are not claimed as existing features of v0.1.
