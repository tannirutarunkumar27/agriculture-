# Local verification and recovery guide for the regression restart

This guide applies to the existing Windows checkout and its installed virtual
environment. It is not a Linux deployment validation, portable release package,
network server, secure-aggregation deployment or production privacy claim.
The active scientific task is conditional harvested seed dry matter regression
in t/ha. Historical crop-classification results are a separate development study.

## Environment and immutable evidence

Run from the repository checkout root using the environment-specific Python
launcher (for this Windows checkout: `.venv/Scripts/python.exe`). The observed
training environment is recorded in
`paper2/requirements-runtime-observed.txt` and each execution manifest. Figure
generation additionally used installed Matplotlib3.11.1. These observations are
not proof of compatibility on a different OS or a fresh installation.

Frozen manifests bind source hashes, dependency versions, data files, partitions,
settings and parent studies. Changing a frozen source/runtime must fail
verification. Do not bypass checks or rewrite old manifests to accept changes.
Implement an amendment under new code/package/result identities if justified.
The current manifests include absolute paths; moving the checkout is not a
validated portability workflow. A portable release needs separate preparation
and verification, without modifying this evidence.

Keep Paper1, raw data, the open manuscript, frozen snapshots, models, checkpoints,
prediction caches, journals and ledgers intact. New outputs must use a fresh
directory beneath `paper2/results`; commands reject existing output paths to
avoid replacement. Never clear an inference intent to make a retry proceed.

## Completed studies and audit-only commands

The1120 original inner candidates,320 original outer evaluations,320 support-
guard sensitivity evaluations,224 expanded inner candidates,40 R13 controls,
and160 R14 matched runs have completed execution. R14 independent final audit
status must be checked in PROJECT_STATUS.md rather than inferred from this list.
The old v1/v2 privacy simulations are closed; v3's200 runs are complete.
Do not rerun a completed study to regenerate plots.

Use new output directory versions for any fresh audit. These commands verify
cached predictions and provenance, without model forward calls or training:

```powershell
.\.venv\Scripts\python.exe -m paper2.restart.matched_noise_protocol verify
.\.venv\Scripts\python.exe -m paper2.restart.verify_public_refit --output paper2/results/refit_audit_NEW_VERSION
.\.venv\Scripts\python.exe -m paper2.restart.verify_matched_noise --output paper2/results/matched_audit_NEW_VERSION
```

The protocol verifier also checks its earlier cached parent studies. Final R14
acceptance requires160 verified new runs,40 bound cached controls, strict
attempt receipts,23174 combined events and unchanged8564 historical event rows.
These tests establish bookkeeping/software consistency, not a DP guarantee.

## Summary and figure regeneration

Full-study summaries reject incomplete matrices. They use the unchanged tested
aggregation, preserve equal-group versus pooled metrics, and retain all variants
and support panels. The following commands perform cached audits first:

```powershell
.\.venv\Scripts\python.exe -m paper2.restart.summarize_matched_noise --output paper2/results/matched_analysis_NEW_VERSION
.\.venv\Scripts\python.exe -m paper2.restart.publication_figures --study matched --output paper2/results/matched_figures_NEW_VERSION
```

For R13 figures use `--study refit`. Inspect all PNG figures before accepting
the output, retain the generated exact `plot_values.json`, captions and hashes,
and use the accompanying vector PDF/SVG files for publication. Descriptive
seed SD is not a confidence interval over independent regions. Unseen support
is limited to265 Auzeville systems. No figure alone establishes field benefit.

## Recovery rules

Inspect `PROJECT_STATUS.md`, the current execution report, processes and the
study writer lock first. An existing lock file alone does not mean a process is
active. Never remove it or terminate an unrelated process to force execution.
The runner uses an OS lock and rejects concurrent writers. During a long batch,
`progress.json` may retain its previous batch count; completion files/journals
are live evidence, but their counts are not independent audit acceptance.

If an incomplete batch has no active writer, use the existing frozen runner
with explicit `--execute --resume --cap N`, where N is a positive bound on NEW
runs. Existing completions are verified and reused. Do not alter the plan or
increase per-run rounds based on test scores. A failed training attempt requires
traceback diagnosis and explicit `--retry-failed`; preserve the failed attempt.
An inference intent without valid completion is ambiguous and cannot be retried
automatically. Escalate that specific run for methodological review.

State and noise events commit together in checkpoints. The shared ledger
reconciles those events, preserving old rows and rejecting removed/changed
committed events. Never count a completed resumed round twice. Deterministic
public simulation RNG is for reproducibility, not secrecy or certified DP.

## Relevant tests

Each test harness writes a log and verification JSON to a new output directory:

```powershell
.\.venv\Scripts\python.exe -m paper2.restart.test_matched_noise_protocol --output paper2/results/protocol_tests_NEW_VERSION
.\.venv\Scripts\python.exe -m paper2.restart.test_matched_noise_training --output paper2/results/training_tests_NEW_VERSION
.\.venv\Scripts\python.exe -m paper2.restart.test_matched_noise_runner --output paper2/results/runner_tests_NEW_VERSION
.\.venv\Scripts\python.exe -m paper2.restart.test_matched_noise_summary --output paper2/results/summary_tests_NEW_VERSION
.\.venv\Scripts\python.exe -m paper2.restart.test_publication_figures --output paper2/results/figure_tests_NEW_VERSION
```

Use tests appropriate to actual code changes. Do not repeat completed model
experiments merely because a test or scheduled check ran. Synthetic test failures
used to verify recovery must not be counted as real study failures.

Outstanding gates: independent agronomic knowledge validity, ranking utility
comparability, production privacy/SecAgg, Linux/remote infrastructure and a
portable reproduction package. Existing local verification opens no ports,
uses no credentials and incurs no cloud charges.
