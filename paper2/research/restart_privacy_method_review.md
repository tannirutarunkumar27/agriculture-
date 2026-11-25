# Method review for the regression restart

Checked 2026-10-14 using primary sources. This focused review does not establish
an exhaustive novelty search. Historical crop-classification metrics do not
provide a comparison for conditional harvest regression.

| Source | Verified relevance | Project implication |
| --- | --- | --- |
| [McMahan et al., Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963) | User-level protection is integrated with federated updates; task is language modeling | Clipped/noisy federated updates are established methodology; the protected unit must match our regional clients |
| [Bun and Steinke, Concentrated Differential Privacy](https://arxiv.org/html/1605.02065) | Gaussian privacy cost, adaptive composition and conversion to epsilon/delta | Use full-participation accounting with explicit sensitivity; no sampling amplification claim |
| [Mironov, Renyi Differential Privacy](https://arxiv.org/html/1702.07476) | Privacy cost curves and composition framework | Supports later comparison with maintained accounting tools; current closed-form bound is conservative |
| [Bonawitz et al., CCS 2017](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/) | Practical secure aggregation for machine learning | Secure aggregation needs vetted protocol integration and explicit assumptions; it is not implemented by averaging local tensors |

Knowledge guidance still requires independently validated regression-compatible
agronomic sources. Neither the support guard nor combining established DP and
federated training establishes a new agricultural algorithm. Candidate research
questions: how does whole-client clipping/noise affect equal-region error and
unseen-composition behavior in this small, uneven client population, and how
does a separately valid agronomic mechanism change that trade-off?

These are hypotheses for prospective experiments, not completed findings.
No novel method or privacy-preserving crop-recommendation claim is established.
