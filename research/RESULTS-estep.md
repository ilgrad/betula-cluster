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

## Independent check against the authors' own implementation (2026-08-24)

The experiment above is synthetic, two-dimensional and written by the same project it justifies.
ELKI 0.8.0 — the reference implementation of BETULA, by the algorithm's authors — ships two GMM
heads over the same cluster features: `BetulaGMM` and `BetulaGMMWeighted`. Running them against
this library's `method="gmm"` at matched CF-tree parameters is the outside check this page lacked.

Harness: `local/scratch/elki/cross_check.py`, median of seeds 0/1/2, `feature="diagonal"` ↔
`VVIFeature`, branching 32, `threshold=0`, `max_iter=100`. Two CF-tree geometries are run because
the projects ship different defaults — D0/D0 (`CentroidEuclideanDistance` routing and absorption,
this library's default) and D4/R (`VarianceIncreaseDistance` + `RadiusDistance`, ELKI's).

| dataset | geometry | `betula gmm` | ELKI `BetulaGMM` | ELKI `BetulaGMMWeighted` |
|---|---|---|---|---|
| digits (1797×64, 200 leaves) | D0/D0 | **0.5239** | 0.2305 | 0.3528 |
| digits | D4/R | **0.5210** | 0.1627 | 0.4586 |
| covtype-50k (54-D, 2000 leaves) | D0/D0 | **0.1062** | 0.0244 | 0.0523 |
| covtype-50k | D4/R | **0.0852** | 0.0555 | 0.0575 |

The shipped head leads at the median in all four cells, and reaches the lower within-cluster sum of
squares in all four as well. **The margin is not all E-step.** This library's GMM head keeps the best
of four EM restarts by log-likelihood (`GMM_N_INIT`, `src/clustering/gmm.rs`); a bare ELKI
`BetulaGMM` runs one, and ELKI 0.8.0 has no `BestOfMultipleKMeans` equivalent for EM to equalise it
with. Compared instead against the **best of ELKI's three seeds across both its variants** — a
generous handicap that absorbs much of that difference — the result splits evenly: the shipped head
leads on digits/D0/D0 (0.5239 vs 0.3744) and covtype/D0/D0 (0.1062 vs 0.0696), and trails on
digits/D4/R (0.5210 vs 0.5294) and covtype/D4/R (0.0852 vs 0.1193). So: a clear lead at equal seed
budget, and a two-two split against ELKI's luckiest seed — the honest reading is parity once the
restart budget is accounted for, not superiority.

What this does and does not license. It **does** retire the objection that the E-step decision rests
on one synthetic fixture from a pre-0.1.0 prototype: on two real datasets, in two tree geometries,
the shipped formulation is never behind both of the authors' variants at an equal seed budget, and
is at worst level with the better of them once ELKI is given its luckiest seed. It does **not** isolate
the E-step as the cause — restarts, initialisation and the M-step differ too. Isolating it would need
variants A/B/C wired behind a flag in this crate and run on the same fixtures, which is worth doing
if the decision is ever revisited. One directional signal is worth recording: `BetulaGMMWeighted`
beats plain `BetulaGMM` in all four cells, which is the same direction as this page's argument —
folding the leaf's own mass and scatter into the E-step helps.
