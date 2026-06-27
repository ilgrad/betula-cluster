# Best GMM E-step for clustering on CF summaries (measured)

Experiment: `research/gmm_cf_estep.py`. Data — a GMM with known labels (K=4, N=20000, d=2),
compressed into `m` micro-clusters → CF `(n_i, μ_i, Σ_i)`, then GMM-EM on the CFs with three
different E-steps (identical init and M-step, so only the E-step varies). Metric — ARI of the
original points against ground truth.

| scenario | gold(raw) | kmeans-CF | A: plug-in | B: convolution (paper) | **C: expected-log** |
|----------|-----------|-----------|------------|-------------------------|---------------------|
| bal sep=2.5 m=40 | 0.817 | 0.774 | 0.775 | 0.526 | **0.792** |
| bal sep=2.5 m=150 | 0.766 | 0.680 | 0.756 | 0.750 | **0.756** |
| bal sep=4.0 m=40 | 0.989 | 0.986 | 0.986 | 0.986 | 0.986 |
| imb sep=2.5 m=40 | 0.837 | 0.503 | 0.779 | 0.719 | **0.794** |
| imb sep=2.5 m=150 | 0.696 | 0.777 | 0.892 | 0.892 | **0.892** |
| imb sep=4.0 m=40 | 0.969 | 0.527 | 0.960 | 0.960 | 0.960 |
| (sep≥6 everywhere) | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 |

## Conclusion
- **C (expected-log with the `−½ tr(Σ_k⁻¹ Σ_i)` correction) is the best E-step.** Closest to
  gold(raw-GMM), consistently ≥ A and B, and far better than k-means under imbalance (0.794 vs 0.503
  at m=40).
- **B (convolution, the BETULA paper's approach) is the worst on coarse CFs** (0.526 at m=40):
  inflating `Σ_k + Σ_i` washes out the components' separability. On fine CFs (m=150) `Σ_i` is small,
  so A/B/C converge.
- The difference shows up exactly under **overlap + coarse summaries + imbalance** — i.e. where CF
  compression genuinely stresses the E-step. On well-separated data all three are equivalent.

## Decision for the implementation
GMM-on-CF E-step: **`log r_ik = log π_k + log N(μ_i|μ_k,Σ_k) − ½ tr(Σ_k⁻¹ Σ_i)`** (variant C), with
log-sum-exp normalization; the M-step folds in `Σ_i`: `Σ_k = Σ_i w_ik(Σ_i+(μ_i−μ_k)(μ_i−μ_k)ᵀ)/N_k`.
This confirms `math_improove/06`; the paper's convolution approach is NOT used for the responsibility.
