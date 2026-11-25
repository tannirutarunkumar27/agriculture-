# Corrected DP + SecAgg multi-round protocol

## Accounting correction

The earlier pilot ledger treated Flower's noise multiplier as a multiplier of
whole-client replacement sensitivity. Flower 1.42 actually adds Gaussian noise
with standard deviation `z*C/m`, while replacement sensitivity for the equal
mean of clipped whole-client updates is `2C/m`. Consequently `z=8` gives a
noise-to-sensitivity ratio of 4, not 8. The prior one-round epsilon was too
optimistic and must not be reported.

With `C=1`, seven fixed clients, `z=8`, and `delta=1e-5`, the corrected zCDP
upper bounds are:

- one adaptive release: epsilon 1.2308814780;
- twenty adaptive releases: epsilon 5.9899150657.

No client-sampling amplification is assumed. Every adaptive round is counted,
even when only the final model is retained.

## Frozen study

`restart_dp_secagg_study_protocol_v2` freezes 20 rounds, seven fixed clients,
equal client weights, client-side fixed clipping, Flower SecAgg+, and central
Gaussian noise. It disables client evaluation, private validation release,
checkpoint release, noise-seed release, hyperparameter selection, and outer
test inference. The transform/data manifest is fixed public study data.

The portable remote bundle is `restart_dp_secagg_remote_bundle_v1`. The pilot
and multi-round launchers verify every transferred file, the Flower 1.42 API,
and the relocated data-manifest hash before execution. Dry runs perform no
training. Actual execution remains pending an SSH-accessible Linux host.

This is a logical-client research study, not application privacy certification,
remote-farm deployment evidence, or an independent security audit.
