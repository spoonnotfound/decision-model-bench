# Decision Model Bench

A reproducible accuracy benchmark for **discrete decision models**.

**[中文说明](README.zh-CN.md) · [Methodology](docs/methodology.md) · [Add a task/model](docs/extending.md)**

Models receive a state and named options. The benchmark compares the accuracy of their choices with reproducible per-case results. It supports dedicated decision models and any adapter that satisfies the same contract.

**v0.1 scope:** categorical choices, including binary decisions and explicitly grouped labels. Ordinal `score`, continuous outputs, multi-turn agents, and cross-task overall rankings are not implemented.

## Start without a GPU or API key

```bash
git clone https://github.com/spoonnotfound/decision-model-bench.git
cd decision-model-bench
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

dmb validate --task tasks/turtlebench-en.json
dmb score --task tasks/turtlebench-en.json \
  --records results/turtlebench-en-20260920/jev.jsonl
pytest -q
```

The repository includes normalized, pinned TurtleBench English inputs and sanitized historical predictions. **Scoring these files is offline; it is not a new model run.** A small routing demo tests the harness plumbing:

```bash
dmb run --task tasks/demo.json --config configs/baseline.toml --out runs/demo
```

## Reference study: English TurtleBench

An exploratory study conducted on 2026-09-20, **before this harness was built**, was imported and independently re-scored. All three models received the same 1,532 English cases across 32 stories from [TurtleBench on GitHub](https://github.com/mazzzystar/TurtleBench). This comparison focuses on decision accuracy. The study asks a model to judge a player's guess with the full story provided.

| Model | Correct | Accuracy |
|---|---:|---:|
| Jev via Vercel AI Gateway | 1,299 / 1,532 | 84.79% |
| Laya English base | 1,009 / 1,532 | 65.86% |
| Laya typed-decisions | 956 / 1,532 | 62.40% |
| Always choose a negative label | 886 / 1,532 | 57.83% |

**These are task-specific findings, not a universal model leaderboard or a reproduction of Laya's advertised typed-decisions result.** Laya's task-specific fine-tuning did not improve accuracy over its English base checkpoint on this transfer task.

The original scoring merges `Incorrect` and `Unknown` **after** the model chooses among three options. Thus this score cannot show whether a model distinguishes contradiction from insufficient information. TurtleBench also filters easy cases: the measured error rate does not represent ordinary gameplay.

Important provenance:

- Laya: official complete checkpoints at revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, no quantization or task-specific tuning in this study. Checkpoint temperature settings were retained.
- Jev: the gateway returned only `typesafe-ai/jev`; **the precise backend revision is unknown**.
- The English compact prompt was chosen after exploratory Chinese tests. No English labels were changed. Training contamination of the public dataset is unknown.

See [study provenance](results/turtlebench-en-20260920/study.json), [full metrics](results/turtlebench-en-20260920/summary.json), and [metric definitions](docs/methodology.md).

## Run a model

### Laya

```bash
pip install -e '.[laya]'
# Two-case smoke test; first use downloads the selected official checkpoint.
dmb run --task tasks/turtlebench-en.json --config configs/laya-typed.toml \
  --out runs/laya-typed-smoke --limit 2
# Full run uses a new output directory.
dmb run --task tasks/turtlebench-en.json --config configs/laya-typed.toml \
  --out runs/laya-typed-full
```

Use `configs/laya-english.toml` or `configs/laya-multilingual.toml` for other checkpoints. The default device is CPU; explicitly set `device = "cuda"` or `"mps"` for your hardware. Model revision and checkpoint temperatures are recorded. Input, instruction, and option truncation cause a visible failed attempt, rather than silently discarding text. An explicit `max_len` override is a new configuration and requires a new run directory.

### Jev via the same gateway as the reference study

Requires Node.js 24+ and an existing Vercel AI Gateway key.

```bash
npm ci --prefix bridge
# Set AI_GATEWAY_API_KEY in your environment; never put it in a task or config.
dmb run --task tasks/turtlebench-en.json --config configs/jev-gateway.toml \
  --out runs/jev-smoke --limit 2 --live
```

For a direct TypeSafe account, set `TYPESAFE_API_KEY` and use `configs/jev-direct.toml`. That adapter is unit-tested but was **not the transport used in the published study**; confirm that the requested model version remains available to your account.

Remote requests require `--live`. They are sequential in this initial harness, have no hidden retries, and stop after the configured error limit.

### Resume and retry

```bash
# Skips completed attempts. Requests, config, code and dependency fingerprints must match.
dmb run --task tasks/turtlebench-en.json --config configs/jev-gateway.toml \
  --out runs/jev-smoke --limit 2 --live
# Retry failed cases explicitly, preserving earlier attempts.
dmb run --task tasks/turtlebench-en.json --config configs/jev-gateway.toml \
  --out runs/jev-smoke --limit 2 --live --retry-failed
```

Each run writes `manifest.json`, input snapshots `task.json` / `cases.jsonl`, append-only `attempts.jsonl`, and `summary.json`. Request and case hashes protect both model inputs and target labels. A writer lock prevents concurrent processes from writing to the same directory; after a forced kill, verify no writer is running before removing a stale `.writer.lock`. Incomplete or invalid runs exit with status 2. Credentials are loaded from environment variables. Logs store selected response fields and sanitized error codes, not transport headers or exception bodies. New run directories are ignored by Git; review before publishing your own data.

## Compare saved runs

```bash
dmb compare --task tasks/turtlebench-en.json \
  --left results/turtlebench-en-20260920/jev.jsonl \
  --right results/turtlebench-en-20260920/laya-typed-decisions.jsonl \
  --out runs/comparison.json
```

The paired bootstrap resamples whole stories, preserving correlated questions within each story.

## Browse cases offline

```bash
python scripts/build_report.py --task tasks/turtlebench-en.json \
  --run Jev=results/turtlebench-en-20260920/jev.jsonl \
  --run Laya-Typed=results/turtlebench-en-20260920/laya-typed-decisions.jsonl \
  --run Laya-English=results/turtlebench-en-20260920/laya-english.jsonl \
  --out runs/reference.html
```

Open `runs/reference.html` in a browser to filter disagreements and inspect original probabilities. It is a standalone file with no external analytics or network dependencies.

## Project layout

```text
src/decision_model_bench/  task validation, adapters, runner, metrics, CLI
tasks/                    versioned task definitions and attributed datasets
configs/                  model settings, with no credentials
bridge/                   optional Vercel AI SDK evaluation transport
results/                  sanitized, auditable reference study
tests/                    scoring, logging, resume and reference regression tests
docs/                     methodology, extension contract, study limitations
scripts/                  pinned data preparation and historical import
```

## Design references and attribution

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): separate task definitions from model backends and keep heavyweight backends optional.
- [Inspect](https://inspect.aisi.org.uk/tasks.html): separate datasets, model execution and scoring; retain per-sample logs and explicit failure policies.
- [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks): probability-aware evaluation, calibration, selective risk and reproducible comparisons.
- [TurtleBench](https://github.com/mazzzystar/TurtleBench) ([paper](https://arxiv.org/abs/2410.05262)): the first substantive dataset, pinned to revision `6547a5f0cb7097292b900b256d5e824a280acd33`.

This is an independent implementation, not a fork, endorsement or official benchmark of those projects or model vendors. Code is Apache-2.0; bundled TurtleBench data retains its upstream attribution and license. See [NOTICE](NOTICE).
