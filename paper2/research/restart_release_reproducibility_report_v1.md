# Release and reproducibility gate report v1

## Result

Status: `RELEASE_REPRODUCIBILITY_BLOCKED`.

The RR06 GPU bundle and DP/SecAgg bundle both pass their local file-hash,
declared-file-count, path-containment, and byte-count checks. This establishes
bundle integrity only. It does not establish a successful Linux installation,
GPU execution, authenticated deployment, or permission to redistribute source
data and derived artifacts.

## Blocking findings

1. `paper2/requirements-runtime-observed.txt` documents one working Windows
   environment and explicitly states that it is not a cross-platform lock.
2. No final Linux dependency lock has been produced from the target GPU host.
3. No clean Linux end-to-end reproduction receipt exists.
4. The official Zenodo v2.0.0 record at
   `https://zenodo.org/records/14160885` identifies the dataset, DOI, seven
   files, and source hashes, but its Rights section does not name a license.
   Raw files therefore remain excluded from any release package until rights
   are documented.
5. The project repository has no approved top-level code license. The small
   SecAgg pilot subpackages' prototype licenses do not license the whole
   project.
6. No dependency-license inventory has been generated and independently
   reviewed in the final Linux environment.
7. RR06/RR07 GPU results, authenticated network deployment, agronomic-rule
   review, and ranking-design review remain incomplete.

## Enforced release policy

The licensing registry authorizes no redistribution. Until the dataset rights
are confirmed, a future code release must direct users to the versioned DOI
instead of bundling the raw dataset. The project owner must separately choose a
code license after ownership review. Models, predictions, caches, tables, and
figures require artifact-specific review because some may embed or reconstruct
source information.

## Next verification sequence

After a real Linux GPU host is supplied: install the frozen execution
dependencies, record the resolved platform-specific lock, verify the bundle,
run RR06/RR07, execute the DP/SecAgg studies, and preserve environment and
result receipts. Then create a new clean environment from the lock and perform
the independent reproduction. Generate the third-party license inventory in
that exact environment. Release remains disabled unless all gates pass.
