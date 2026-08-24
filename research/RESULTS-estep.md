# Best GMM E-step for clustering on CF summaries (measured)

> Decision record for the expected-log E-step the library ships (`docs/MATH.md` § *GMM E-step*).
> Re-measured **2026-08-24** over three seeds; the previous edition of this page was a **single run**
> of 2026-06-26 on a pre-0.1.0 prototype, and three of its headline numbers turn out to have been
> ends of a seed spread rather than typical values. The decision it reached still stands, but for a
> narrower reason than it claimed. Reproduce with
> `uv run --no-sync python research/gmm_cf_estep.py` (`--seeds` to change them).

Experiment: `research/gmm_cf_estep.py`. Data — a GMM with known labels (K=4, N=20000, d=2),
compressed into `m` micro-clusters → CF `(n_i, μ_i, Σ_i)`, then GMM-EM on the CFs with three
different E-steps (identical init and M-step, so only the E-step varies). Metric — ARI of the
original points against ground truth. Medians of seeds `0xBE7012A + {0,1,2}`, `[min, max]` beneath.

| scenario | gold(raw) | kmeans-CF | A: plug-in | B: convolution (paper) | **C: expected-log** |
|---|---|---|---|---|---|
| bal sep=2.5 m=40 | 0.817 | 0.774 | 0.775 | 0.743 | **0.792** |
| | 0.78–0.86 | 0.66–0.84 | 0.75–0.84 | 0.53–0.82 | 0.74–0.84 |
| bal sep=2.5 m=150 | 0.772 | 0.725 | 0.766 | 0.765 | 0.766 |
| | 0.77–0.83 | 0.68–0.81 | 0.76–0.82 | 0.75–0.82 | 0.76–0.82 |
| bal sep=4.0 m=40 | 0.989 | 0.986 | 0.986 | 0.986 | 0.986 |
| bal sep=4.0 m=150 | 0.990 | 0.987 | 0.990 | 0.990 | 0.990 |
| imb sep=2.5 m=40 | 0.850 | 0.818 | 0.818 | 0.797 | 0.819 |
| | 0.84–0.90 | 0.50–0.90 | 0.78–0.92 | 0.72–0.89 | 0.79–0.93 |
| imb sep=2.5 m=150 | 0.739 | 0.777 | **0.885** | **0.885** | **0.885** |
| | 0.70–0.84 | 0.72–0.79 | 0.83–0.89 | 0.82–0.89 | 0.83–0.89 |
| imb sep=4.0 m=40 | 0.995 | 0.979 | 0.989 | 0.989 | 0.989 |
| imb sep=4.0 m=150 | 0.989 | 0.843 | 0.920 | 0.920 | 0.919 |
| (sep = 6.0 everywhere) | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 |

Counting every one of the 36 `scenario × seed` cells rather than the medians, because a variant that
wins narrowly everywhere and one that wins hugely once are different results:

| | best or tied | median shortfall against gold(raw) |
|---|---:|---:|
| A: plug-in | 32/36 | +0.002 |
| B: convolution | 22/36 | +0.002 |
| **C: expected-log** | **33/36** | +0.002 |

## Conclusion

- **B (convolution, the BETULA paper's approach) is the weakest**, and that is the finding that
  survives re-measurement: 22/36 against 32–33/36. Inflating each component by `Σ_k + Σ_i` washes out
  separability, and it costs most where the summary is coarse. Its worst cell (bal sep=2.5 m=40)
  spans 0.53–0.82 — far wider than any other variant's, so the failure is also *unreliable*, which is
  worse than being uniformly slightly behind.
- **C vs A is a tie at three seeds — 33/36 against 32/36.** The previous edition claimed C was
  "consistently ≥ A and B"; against A that is not supported. C leads A in exactly one regime, coarse
  summaries with overlapping components (bal sep=2.5 m=40, 0.792 against 0.775), which is where the
  `−½ tr(Σ_k⁻¹ Σ_i)` correction is theoretically supposed to matter — `Σ_i` is largest relative to
  `Σ_k` there. Elsewhere they agree to the third decimal, and at imb sep=4.0 m=150 A is ahead by
  0.001.
- **All three CF variants beat gold(raw) under imbalance at m=150** (0.885 against 0.739). The
  summary is acting as a regulariser: raw EM on 20 000 points with a 0.5/0.25/0.15/0.10 mixture
  collapses a small component more often than EM on 150 weighted micro-clusters does.
- **Two dramatic numbers from the 2026-06-26 edition were single-seed artefacts.** `kmeans-CF` at
  imb sep=2.5 m=40 was reported as 0.503, against a three-seed median of 0.818 and a range of
  0.50–0.90 — the old number was the bottom of the spread. Likewise B's 0.526 at bal sep=2.5 m=40,
  against a median of 0.743 and a range of 0.53–0.82. The qualitative claims they were used to
  support (k-means-CF is fragile under imbalance; B is worst on coarse CFs) both hold on the medians,
  but not with the margins the single run suggested.

## Decision for the implementation

Unchanged: **`log r_ik = log π_k + log N(μ_i|μ_k,Σ_k) − ½ tr(Σ_k⁻¹ Σ_i)`** (variant C), log-sum-exp
normalised, with the M-step folding `Σ_i` back in as
`Σ_k = Σ_i w_ik(Σ_i + (μ_i−μ_k)(μ_i−μ_k)ᵀ)/N_k`. Shipped in `src/clustering/gmm.rs` for both the
diagonal and full heads (`trace_under` supplies the correction).

The justification is now narrower and should be stated as such: C is chosen over **B** on measured
ARI, and over **A** because it is the correct expected log-likelihood under the leaf model and costs
one trace that the second moment is already carrying — not because it measurably clusters better in
general. A is a legitimate alternative that would perform the same on this experiment outside the
coarse-and-overlapping corner.
