# Methodology and metric contract

## Unit of evaluation

One case has a stable string `id`, a string `state`, a target option, and an optional `group` (a story in TurtleBench). Only the ID, state and task question are sent to adapters; target/group metadata is never included in a model request. IDs identify results and are not used as task evidence.

The task question declares named criteria. A valid decision has a known choice, exactly those probability keys, finite numeric probabilities in [0,1], an approximately unit total, and an argmax choice (ties allowed). Missing values, booleans, NaN and unsupported choices are rejected.

`probability_sum_tolerance` explicitly permits small interface rounding drift (0.011 in the TurtleBench task). Raw values are retained. Probability metrics normalize accepted distributions by their sum and report the number normalized. This does not recalibrate the model. `raw_event_brier` additionally preserves the first study's unnormalized returned-probability convention.

## Label grouping

Original option choice happens **before** task-defined label grouping. With probabilities Correct=.4, Incorrect=.35, Unknown=.25, the selected original option is Correct, even though the merged negative probability is .6. Accuracy uses that actual choice. Group probabilities are summed for probability scoring.

TurtleBench merges Incorrect and Unknown; the framework must not report this as a three-class accuracy. This convention cannot certify a full game host's handling of unknown answers.

## Reported scores

- Accuracy: correct latest attempts / all selected cases. Unattempted and invalid cases contribute zero; attempted coverage and valid counts are shown separately so incomplete runs cannot masquerade as complete scores.
- First-attempt accuracy: first attempt per case, with failures counted wrong. Retries never erase first-attempt failures.
- Macro group accuracy: mean accuracy across case groups, rather than allowing large stories to dominate.
- Class recall and constant-majority baseline: identify class imbalance and degenerate predictions. The majority baseline is descriptive, not a trained model.
- Multiclass Brier: mean sum over scored classes of `(p - one_hot_target)^2`. It ranges from 0 to 2; do not confuse this with the binary event Brier convention.
- Event Brier (binary grouped tasks): mean `(P(positive_group) - y)^2`, range 0–1. Brier reflects discrimination and calibration, not calibration alone.
- Clipped NLL: `-log(max(p_target, 1e-12))`. The zero-target-probability count is reported separately; the clipping is numerical reporting, not a claim of nonzero model belief.
- ECE: ten equal-width bins using the reported probability of the **chosen scored group**, compared with its correctness. Empty bins have null statistics. ECE depends on bins and sample size; it is not a universal trust certificate.
- Selective risk: fixed thresholds 0, .5, .7, .9, .95 on that same chosen-group probability. Coverage is accepted / all cases, including failed or unattempted cases in the denominator. Risk is error fraction among accepted cases; zero coverage has null risk. Ties are retained together. No thresholds are fitted on test labels.

The vendor's `confidence` statistic is not substituted for answer probability. `Unknown` is an answer about insufficient story evidence; choosing not to automate a low-confidence case is a separate system action.

## Runtime and resume

Input/task/config/code/dependency fingerprints guard resumes. Case hashes also bind labels and grouping metadata; score/compare rejects relabeled cases. Runs snapshot the task and cases so later source edits do not destroy the original inputs. One writer lock guards each run directory; remove a stale lock only after verifying its writer has stopped. Every attempt is appended before proceeding. Transport errors and invalid decisions remain visible. A partial JSON line is not silently discarded: preserve the original file, inspect it, and repair deliberately.

Remote runs require `--live`; there are no implicit retries. Credentials come from environment variables and are not serialized. Error logs deliberately omit arbitrary exception messages and HTTP bodies. Additional adapters must follow this contract. Do not publish private datasets simply because a run completed.


## Comparison and provenance

`dmb compare` requires matching request hashes and complete attempted coverage. It computes paired differences and resamples whole case groups 2,000 times with a fixed seed. Questions within a story are not treated as independent. Confidence intervals cover resampling variation, not prompt uncertainty, public-test contamination, hardware differences, or mutable model aliases.

Publish exact dataset and model revisions, prompt, grouping policy, precision, hardware, calibration state, dependencies, errors and raw probabilities. If an API only returns an alias, report the exact version as unknown. Keep exploratory experiments and predeclared comparisons distinct; do not select whichever prompt makes a preferred model win.
