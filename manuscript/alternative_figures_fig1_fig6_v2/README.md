# Alternative plots for manuscript Figures 1 and 6

All plots use the reconciled saved probability arrays for the designated
330-record test partition. No model was retrained and no probabilities were
changed or synthesized.

## Figure 1 alternatives

1. `fig1_alt_small_multiples_reliability`: four separate reliability panels.
   This is the recommended direct replacement because it preserves the familiar
   reliability-diagram interpretation while removing crossing-curve and legend
   clutter. Bubble size reflects populated-bin sample count.
2. `fig1_alt_calibration_gap_heatmap`: signed binwise calibration gap with
   counts. This is the recommended compact companion/supplement because sparse
   one-to-three-record bins are explicit and over-/under-confidence direction
   is easier to compare.

Suggested caption for the small-multiple version:

> Top-label reliability on the designated 330-record saved-checkpoint test
> partition. Each panel shows empirical accuracy versus mean confidence for
> populated bins; marker size reflects bin sample count and the dashed diagonal
> denotes ideal calibration. ECE values use the same reconciled probability
> arrays as the reported probability metrics.

## Figure 6 alternatives

1. `fig6_alt_confidence_error_distribution`: confidence histogram and ranked
   confidence trace with the single error marked. This is the recommended main
   replacement because the ordinary risk-coverage line is nearly degenerate:
   329/330 predictions are correct and the error enters only at rank 328
   (99.4% coverage).
2. `fig6_alt_threshold_operating_curve`: threshold-versus-coverage and retained
   error count. This is useful when the paper discusses a deployable confidence
   threshold, but it must remain separate from XGBoost early-exit routing and
   latency.

Suggested caption for the distribution/rank version:

> Confidence distribution for direct corrected noncompact KG-Hybrid predictions
> on the designated 330-record test partition. The upper panel locates the
> single misclassification at confidence 0.563; the lower panel orders records
> by maximum predicted probability and shows that the error enters at rank 328,
> corresponding to 99.4% retained coverage. This is a selective-prediction
> diagnostic, not a cascade-routing or latency result.

PNG files are for inspection; PDF and SVG files are the publication-ready
vector versions. Exact plotted values and source hashes are retained alongside
the figures.
