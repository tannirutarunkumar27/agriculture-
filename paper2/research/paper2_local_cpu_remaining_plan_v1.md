# Paper 2 local-CPU completion plan

Date: 2026-11-02

## Scope decision

The frozen RR06 and RR07 studies are compute-device agnostic: their scientific
identity is defined by data, splits, source hashes, hyperparameters, seeds,
receipts and audits, not by GPU use. The local laptop completed RR06 on CPU
(504/504) and RR07 on CPU (160/160), and both audits passed. A GPU repetition
may measure runtime or cross-device numerical portability, but it is no longer
a Paper 2 completion requirement.

The local laptop has 12 logical processors and 15.73 GiB RAM. The Azure
Standard B2ats v2 host used for the smoke test has 2 vCPUs and approximately
1 GiB RAM. Therefore, the laptop is the preferred machine for the remaining
CPU-feasible work. It has no verified CUDA runtime, so all plans must remain
CPU bounded.

## Completed and removed from the remaining list

1. RR06 matched-neural inner study: 504/504 complete and audited.
2. RR06 model-selection finalization: complete and bound to the audit.
3. RR07 development outer evaluation: 160/160 complete and audited.
4. Earlier data/split, baseline, control-search, refit, matched-noise, figures
   and Markdown manuscript work recorded as verified in PROJECT_STATUS.md.

## Required remaining work

### L1. Local Linux runtime on the laptop

Use WSL2/Ubuntu on the same laptop for Flower. Flower's simulation runtime uses
Ray; native Windows Ray support is experimental and the recorded Windows run
hung during startup. WSL2 is a local execution environment, not a cloud server.

Acceptance evidence: Linux platform receipt, pinned dependency lock with wheel
hashes, source/data hashes, bounded-resource settings, interruption recovery,
and result-retrieval verification.

### L2. Secure aggregation and differential privacy

1. Re-freeze the one-round Flower SecAgg+ plus central-DP pilot after the local
   WSL environment is fixed.
2. Execute and independently audit the one-round pilot.
3. Execute the already designed 20-round prospective study and audit every
   released aggregate and privacy-accounting event.
4. Complete an independent privacy/security review covering preprocessing,
   clipping, noise, adaptive releases, selection, metrics, checkpoints and
   final models.

The existing Windows attempts are terminal failed/aborted receipts and must not
be overwritten or presented as successful SecAgg evidence.

### L3. Authenticated multi-process network verification

Run Flower SuperLink and seven registered SuperNodes as distinct local Linux
processes or containers with TLS and node authentication. Record CA and key
fingerprints, registered identities, process boundaries, run ID, completion
receipt and retrieved-result hash. This validates the software deployment path
on one laptop; it must not be described as a multi-farm field deployment.

### L4. Agronomic-rule validation

Obtain independent domain decisions for all eight candidate rules, including
taxonomy, units, geography, crop stage/cultivar applicability and source
independence. Freeze include/exclude decisions before enabling guidance.

This step cannot be completed by additional model training alone.

### L5. Restricted ranking validation

Resolve the outstanding experiment references, review all 47 candidate
treatment blocks, and freeze the utility definition plus replicate/tie rules.
Run restricted ranking only for blocks approved by the review gate. Do not
claim general crop recommendation or counterfactual optimality.

### L6. Licensing and clean reproduction

Document dataset redistribution permission, select an approved project code
license, complete the Linux dependency-license inventory, and pass a clean WSL
reproduction using the final lock. Then re-run the release/reproducibility gate.

### L7. Publication packaging (deferred by the user)

After L1-L6 are reconciled, update the verified results/limitations, convert the
manuscript to IEEE LaTeX, render and inspect the PDF, and perform final figure,
table, bibliography and claim-to-evidence checks.

## Execution order

L1 -> L2 -> L3 -> L6 can be performed on the laptop. L4 and L5 depend on
independent agronomic/source review and may proceed in parallel with the local
technical work. L7 starts only after the evidence gates are frozen.
