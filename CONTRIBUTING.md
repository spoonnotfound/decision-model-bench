# Contributing

Run `pip install -e '.[dev]'`, `ruff check src tests scripts`, and `pytest -q` before submitting changes. Keep core scoring independent of model libraries and credentials.

A new task should include provenance, a license, exact transformations, and scoring examples that expose ambiguity. A new adapter should preserve probabilities and failure records, refuse silent truncation, and include transport tests. Report whether tests are mocked, replayed or live.

Do not upload credentials, private inputs, model weights or mutable leaderboard claims. Publish complete runs and disclose failed attempts, exploratory prompt changes, unknown API versions and different hardware conditions. Open an issue to discuss a new evaluation contract before adding incompatible primitives.
