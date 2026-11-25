# Paper 2 Reproducibility Readiness

The current repository has a verified local evidence chain for the Paper 2 manuscript, but it is **not yet a portable clean-room release**. The readiness audit inventories the manuscript, source registry, claim matrix, prepared-data evidence, completed study audits, accepted figure manifests, and runtime record. It hashes only the essential evidence files and does not retrain models or repeat test inference.

Run the checks from the repository root:

```powershell
python -B -m unittest paper2.tests.test_reproducibility_readiness -v
python -B -m paper2.restart.reproducibility_readiness --output paper2/results/restart_reproducibility_readiness_v1
```

The versioned result is `paper2/results/restart_reproducibility_readiness_v1/readiness.json`. A passing local audit has the status `PASS_LOCAL_EVIDENCE_BLOCKED_PORTABLE_REPRODUCTION`. This is intentionally not called a clean-room reproduction pass.

## Remaining release blockers

- Historical manifests contain checkout-specific absolute paths.
- The runtime file records one working Windows environment and is not a cross-platform dependency lock.
- No fresh environment or Linux end-to-end execution has been verified.
- Raw-source redistribution permissions and release licensing require review.
- Production privacy accounting, secure aggregation, and external deployment infrastructure remain outside the verified implementation.

These blockers do not invalidate the cached local analyses, but they prevent a defensible portability or independent-reproduction claim.
