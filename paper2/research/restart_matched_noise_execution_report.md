# R14 starter audit and remaining-run launch

Superseded by `restart_matched_noise_completion_report.md`: all160 completed
and independently audited,zero failures;full analysis/figures accepted. Do not
resume the historical running notice below.

2026-10-14. R14a protocol and R14b execution acceptance are VERIFIED_COMPLETE.
Overall R14 is IN_PROGRESS. The four-run real starter completed and independently
audited PASS_PARTIAL_MATCHED_STUDY at `results/restart_matched_noise_verified_v1`.
Four COMPLETE journals, one attempt each, zero failed attempts. These are
outer0/Angers seed123, one clipped control and noise multipliers2/4/8.
No inference was repeated by the audit. Historical8564 events are unchanged;
399 new committed noise rounds produce8963 total events.156 runs remain after
this audited starter. No full-study utility finding is claimed from one group/seed.

Next authorized execution has been LAUNCHED with the unchanged frozen package:

```text
.venv/Scripts/python.exe -m paper2.restart.matched_noise_runner --execute --resume --cap 156
```

Unified execution session at launch:69296. Do not assume it is still running
merely from this report. Check actual process/writer lock and saved journals/
checkpoints/completions before resuming. Never launch a duplicate writer.
The saved progress.json is written at batch completion or failure and may retain
the four-run starter count while this batch is running. Checkpoint events remain
authoritative and are reconciled into the shared ledger before test inference,
after failures where possible, and on safe resume. Do not terminate an active
writer just to update status or perform the independent audit.

Commands already completed successfully:

```text
.venv/Scripts/python.exe -m paper2.restart.matched_noise_runner
.venv/Scripts/python.exe -m paper2.restart.matched_noise_runner --execute --cap 4
.venv/Scripts/python.exe -m paper2.restart.verify_matched_noise --output paper2/results/restart_matched_noise_verified_v1
.venv/Scripts/python.exe -m paper2.restart.test_matched_noise_summary --output paper2/results/restart_matched_noise_summary_tests_v1
```

The completed dry-run verified the execution/source/protocol chain.41 scoped
acceptance tests passed before real training. Added `summarize_matched_noise.py`
and11 passing summary tests (3 new matched tests and8 existing aggregation/panel
tests), total52 relevant tests this session, zero failures/errors. Summary code
requires all160 audited runs plus40 matched cached R13 references; verifies
cross-variant IDs/targets/support; reuses the existing aggregation unchanged;
reports descriptive paired seed differences, geography panels, equal-group and
pooled metrics, saved training-fit diagnostics and cumulative accounting.
No summary of incomplete real data has been generated.

After the active batch exits successfully:

```text
.venv/Scripts/python.exe -m paper2.restart.verify_matched_noise --output paper2/results/restart_matched_noise_verified_v2
.venv/Scripts/python.exe -m paper2.restart.summarize_matched_noise --output paper2/results/restart_matched_noise_analysis_v1
```

Use a NEW versioned output if either path already exists. Audit must verify160
complete runs, expected23174 combined events, retained8564 prior events, strict
journals and all cached metrics/support. If the batch fails, preserve traceback,
inspect exact failure, add regression tests and version any frozen-source fix;
never silently change rounds/noise/clipping or retry ambiguous test inference.
Training-only retry needs explicit --retry-failed after diagnosis.

Remaining: complete R14 execution/audit/summary, then evidence-based R10 figures,
manuscript and reproducibility deliverables. R06/R07 agronomic/ranking eligibility
and R09 remote infrastructure remain blocked. R08 certified privacy/SecAgg is a
separate open requirement. Scope/caveats are in the preparation report. Nothing
in this simulation establishes field utility, novel crop recommendation or
application privacy. Paper1, raw data, open LaTeX and earlier frozen work remain
unchanged. No dependency installs, cloud costs or network services.
