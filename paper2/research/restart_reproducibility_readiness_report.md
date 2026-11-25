# Restart reproducibility-readiness audit

## Scope

This audit checks the integrity and presence of the essential local evidence used by the Paper 2 manuscript. It does not rerun any of the 1,120 inner-validation candidates, 320 original outer evaluations, 320 post-hoc guard evaluations, 224 fixed-public search candidates, 40 unnoised refits, or 160 matched-noise evaluations. It also performs no model inference.

## Acceptance criteria

The local evidence check passes only when every registered asset exists, each JSON asset parses, all frozen completion counters and status fields match their audited values, the runtime record contains the six core packages, and the runtime record retains its explicit non-lock disclaimer. The audit hashes every registered asset and detects Windows absolute paths as portability findings.

## Interpretation

A local pass establishes that the current checkout contains a coherent, hashable evidence chain for manuscript verification. It does not establish a portable release. The clean-room status remains blocked by checkout-specific paths, the absence of a cross-platform dependency lock, the absence of fresh-environment and Linux verification, unresolved redistribution/licensing review, and the absence of production privacy and deployment infrastructure.

## Safety properties

The checker is standard-library-only, refuses to overwrite a versioned output directory, fails closed for missing or malformed required evidence, and records that no training or inference occurred.
