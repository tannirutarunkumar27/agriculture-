# RR03 feature-timing audit

## Purpose

This audit closes the data-availability question for the current Paper 2
regression study. It is a model-free audit: it reads the frozen data manifest,
records a conservative timing classification for every feature, and does not
change or retrain any model.

## Result

The generated registry is
`paper2/results/restart_feature_timing_audit_v1/registry.json`. It covers all
21 manifest features. The audit deliberately leaves
`pre_sowing_recommendation_enabled` set to `false`.

The current evidence supports only a retrospective conditional regression
interpretation. Recorded management variables (N amount, P amount, irrigation
amount, and pest control) are labelled retrospective-only. Soil, sowing, and
previous-crop fields are treated as unverified availability candidates rather
than silently promoted to pre-decision inputs. Mixture indicators are marked
as unresolved for cultivar and density because the columns do not establish
those quantities.

## Research consequence

Paper 2 must not claim a pre-sowing crop recommendation from this dataset.
That claim requires documented prediction-cutoff availability and explicit
measurements for cultivar, planting density, mixture proportion, and treatment
timing. Until those records are supplied, the responsible next step is to
keep the study scoped to retrospective federated/conditional yield modelling.
