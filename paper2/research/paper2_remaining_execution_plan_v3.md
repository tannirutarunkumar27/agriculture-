# Paper 2 remaining execution plan v3

## Current local implementation state

The local implementation now has portable remote bundles, corrected DP
accounting, frozen one-round and multi-round Flower SecAgg+ protocols, remote
preflight launchers, a network-deployment evidence gate, agronomic-rule schema
validation, complete cultivar/density/mixture recovery, an experiment-level
ranking review packet, and a fail-closed ranking approval gate. These artifacts
prepare and validate external work; they do not substitute for remote execution
or independent domain approval.

## 1. Remote GPU RR06 and RR07

1. Infrastructure owner supplies a non-local SSH Linux GPU host and credentials.
2. Run the clean 504-candidate `restart_matched_neural_execution_v10` study
   against `restart_matched_neural_protocol_v7` with
   `paper2/remote_gpu/run_remote_rr06.ps1`.
3. Retrieve and independently verify all 504 receipts. The archived 441 v7 CPU
   receipts remain separate and cannot be rebound into v10.
4. Finalize RR06 selection only after the complete audit.
5. Freeze, run, retrieve, and audit the 160-run RR07 outer evaluation. Do not
   inspect outer results during RR06 selection.

## 2. Secure aggregation and differential privacy

1. On the remote Linux environment, execute the corrected one-round Flower
   SecAgg+ and central-DP pilot in `restart_dp_secagg_pilot_protocol_v8`.
2. Retrieve receipts and run the independent execution verifier.
3. Execute the prospectively frozen 20-round study in
   `restart_dp_secagg_study_protocol_v2`, retrieve receipts, and audit every
   released aggregate and budget event.
4. Perform an independent security/privacy review covering transport,
   authentication, threat model, client clipping, every adaptive release,
   preprocessing, selection, metrics, checkpoints, and final models.
5. Keep the end-to-end privacy claim disabled until that review passes.

## 3. Authenticated Linux network deployment

Deploy Flower SuperLink/SuperNodes on distinct remote hosts or independently
isolated nodes using TLS and registered SuperNode identities. Record the CA
fingerprint, public-key registrations, online-node receipt, run identifier,
SecAgg completion, DP audit, and result hash. The current network gate remains
`BLOCKED_NO_NETWORK_DEPLOYMENT_EVIDENCE` because no remote hosts or receipts
were supplied.

## 4. Agronomic rules and restricted ranking

The agronomic registry contains zero approved independent rules, so knowledge
guidance remains disabled. Domain reviewers must add source-located,
unit-compatible, region/stage/water-scoped rules and document approval.

The ranking review established 47 blocks, 450 management packages, 26/37
experiments with references, and 11/37 without. Fifteen candidate blocks lack
a reference. All 47 approval rows remain pending. Reviewers must decide every
block using the original publications and design records. Only an audited PASS
from `ranking_approval_gate.py` may authorize restricted observed-set ranking.

## 5. Publication and clean reproduction

Do not finalize numerical tables or the IEEE manuscript before RR06, RR07,
privacy/SecAgg, network, and domain-review evidence is reconciled. After those
gates close, regenerate tables/figures from audited artifacts, convert the
manuscript to IEEE LaTeX, render and inspect the PDF, verify every citation and
license, and reproduce the release from a clean Linux environment.

## Remaining external inputs

- SSH-accessible Linux GPU infrastructure and credentials.
- TLS certificates and registered Flower SuperNode identities.
- Independent security/privacy reviewer.
- Original experimental-design evidence for the 47 ranking blocks, including
  missing publication links.
- Independent agronomic/domain reviewer and approved rule sources.
