# ADR 001 — Toeplitz / AR-structured GMM covariance head (`gmm-toeplitz`)

**Status:** Accepted · **implemented** (experimental, off by default). The **AR(w)** head
(`method="gmm-toeplitz"`) ships in **0.2.0**; the general (non-AR) **`gmm-toeplitz-full`** head in
**0.3.0**; the full-order **Gohberg-Semencul MLE** head (`method="gmm-toeplitz-gs"`) in **0.5.0** — the
three-rung ladder is complete. Validated in Rust (`clustering::gmm_toeplitz`) and Python.

## Context

betula's Gaussian-mixture heads model each component covariance as **diagonal** (`method="gmm"`) or
**full** (`method="gmm-full"`). On **ordered, wide-sense-stationary (WSS) signals** — fixed-length
time-series windows, trajectories, sensor / audio / vibration waveforms, lag features of one process —
neither is a good prior:

- **Diagonal** ignores correlation between neighbouring positions, which for a stationary signal *is*
  the discriminative structure (two AR processes with the same marginal variance are identical to a
  diagonal model).
- **Full** has `O(d²)` parameters per component and is ill-conditioned exactly in the common regime
  `N_k ≪ d` (few windows per cluster, long windows). The 0.2.0 per-dimension covariance floor patches
  conditioning but adds no structure.

For a WSS signal the covariance is (approximately) **Toeplitz** — `Σ_{ij} = c(|i−j|)` — so it is
determined by an autocovariance sequence, not a dense `d×d` matrix. arXiv:2311.14995 estimates a
**positive-definiteness-guaranteed** Toeplitz *inverse* covariance from few samples via the
**Gohberg-Semencul (GS)** parameterization plus a cheap closed form — i.e. exactly the *precision* a
GMM E-step consumes.

Crucially, the CF-tree already materializes the component weighted scatter `S_k` (`gmm-full` builds
it), so the pooled autocovariance `r(τ) = mean of the τ-th diagonal of S_k` is available **with no new
CF machinery** — the same reason the existing GMM heads compose with the compression.

## Decision

Add an experimental `method="gmm-toeplitz"`: a GMM whose per-component covariance is **AR(w) /
Toeplitz-structured**.

- **Estimator.** Per component, pool the biased autocovariance `r(0..w)` from the CF scatter, run
  **Levinson-Durbin** → AR coefficients `a` and innovation variance `σ²`. Represent the precision as
  the banded whitening filter `A` (unit diagonal, `−a_j` on the j-th sub-diagonal):
  `Γ = AᵀA / σ²`, **positive-definite by construction** (`σ² > 0`). Pick order `w` by **BIC** over a
  small bounded grid (`w ≤ ~8`).
- **E-step, all `O(d·w)`, via the exact Gohberg-Semencul precision** (no dense `d×d`): the exact
  finite-sample AR log-likelihood by the **prediction-error decomposition** — position `t` uses the
  order-`min(t, w)` Levinson predictor and its own error variance, so the `w` boundary positions are
  modelled *exactly* (the GS `−ZZᵀ` corner term) instead of dropped by a conditional likelihood.
  Measured **+0.04 ARI at short windows** (`d = 32 / 64`) over the conditional form, identical at
  `d ≫ w`. PD is guaranteed by Levinson's reflection-coefficient clamp (the GS box constraint).
- **Parameters:** `O(w)` per component instead of `O(d²)` — well-posed at `N_k ≪ d`.
- **Scope guard:** documented for ordered / stationary signals **only**; explicitly *not* for generic
  embeddings, where coordinate order is meaningless and a permutation destroys the Toeplitz structure
  (expected effect there is negative). Off by default.
- **No new dependencies** — Levinson-Durbin + banded matvec are pure arithmetic (stays NumPy-only).
- **General (non-AR) rung** (`method="gmm-toeplitz-full"`, 0.3.0). AR(w) has a *banded* precision, so it
  cannot represent an autocovariance whose support exceeds order `w` (e.g. a single echo at lag `K > w`).
  A second head forms the dense Toeplitz covariance directly from the **biased** autocovariance
  `r_b(0..d−1)` — positive-semidefinite by construction (periodogram-consistent; the `÷d` bias shrinks
  high lags, a free regularization at `N_k ≪ d`), ridge-regularized to strict PD, and Cholesky-factored
  for an exact multivariate-Gaussian E-step. `O(d²)` parameters / `O(d³)` per component — opt-in, chosen
  by the user (not per-component BIC, to keep the AR head's `O(d·w)` cost). Measured: on a
  lag-`K∈{16,28,40}` echo mixture (all `> w_max=10`) it recovers the components (ARI 0.70→0.97 as the
  window grows) where the AR head sits at chance, and it matches the AR head on AR-generated signals.

## Validation

`research/gmm_toeplitz_prototype.py` — a mixture of AR processes that differ **only** in autocovariance
(each rescaled to unit marginal variance, so the signal is entirely in the covariance structure), 30
windows per component, ARI vs the window length `d`:

| d (window) | N_k / d | gmm-diag | gmm-full | **gmm-toeplitz** |
|---|---|---|---|---|
| 32  | 0.94 | −0.01 | 0.03  | **0.53** |
| 64  | 0.47 | −0.02 | −0.01 | **0.79** |
| 128 | 0.23 | −0.01 | −0.02 | **1.00** |
| 256 | 0.12 |  0.02 | 0.02  | **1.00** |

Diagonal is blind (marginals equalized); full is at chance (`N_k < d`, ill-conditioned); the AR/Toeplitz
model *improves with `d`* (more positions to pool the autocovariance) to perfect separation — decisive
in precisely the regime where both existing heads fail. Random-initialised EM (best of 8 by
log-likelihood), no ground-truth used in fitting.

## Consequences

- **+** A genuine differentiator — no mainstream clustering library ships a Toeplitz-GMM head; it
  strengthens the streaming / sensor / time-series story and is the one regime (`N_k ≪ d`, ordered
  signal) where both diagonal and full covariance fail.
- **+** Reuses the CF scatter already built; cheap (`O(d·w)`); PD by construction (no reg_covar
  guessing).
- **−** Narrow audience (time-series-window clustering), *not* generic clustering; it does **not**
  improve the existing embedding / tabular benchmark (digits / covtype / mnist are not Toeplitz).
- **−** New public method + covariance model in the E/M-step + BIC-over-`w` + Rust core + Python
  wrapper + stubs + tests + a synthetic AR-mixture benchmark + a notebook — a Tier-2/3 effort.
- **−** The AR(w) banded precision *approximates* the exact Toeplitz precision (boundary / edge
  effects at the window ends); acceptable for clustering, to be documented.

## Alternatives considered

1. **Full covariance + shrinkage toward a Toeplitz target.** Still `O(d²)`, still needs `N_k > d` to
   escape the floor; less structured. Rejected — the `O(w)` AR parameterization is the entire point.
2. **Reduce dimensionality first (SVD/PCA), then cluster with an existing head.** Destroys the
   neighbour correlation the prior exploits; we already measure this failure on SVD-reduced sparse text
   (`bench/RESULTS.md`). Rejected.
3. **The paper's GS machinery — exact precision *and* covariance-method estimator, both implemented.**
   The exact Gohberg-Semencul precision `Γ = (1/σ²)(BBᵀ − ZZᵀ)` is realised via the prediction-error
   decomposition (see the E-step; the reflection-coefficient clamp is the paper's PD box constraint),
   and the **covariance-method** estimator is realised as the *unbiased* (per-lag `d − τ`)
   autocovariance projected back to a stable AR by that same clamp — the CF-compatible form of
   `â = S⁻¹_{≥1,≥1} S_{≥1,0}` with the paper's PD projection. Measured: the exact precision adds **+0.04
   ARI** at short windows; the covariance method adds **+0.01–0.04 mean ARI** *and a better worst case*
   across seeds at every `d` tested. The one variant left is a **general (non-AR) Toeplitz** precision,
   deferred *with evidence*: a probe over the cases where AR(w) is intuitively suspect shows banded
   AR(w, BIC) is not restrictive for clustering — periodic signals (two sinusoids of different period +
   white control) ARI **1.00** (a single sinusoid is an AR(2) resonance, so period needs no order),
   MA(2) (infinite AR order) **0.93–1.00**, four-sinusoid signals whose true AR order (~8) exceeds
   `w_max = 6` **1.00**. Clustering only needs the per-component fits to *differ*, not to be exact, so
   the extra generality is a parameter-efficiency refinement, not a
   capability gap; the cheap lever if a hard signal appears is to raise `w_max` (an internal constant).
   The three-rung ladder, **rungs 1–2 now implemented in 0.3.0**, escalated only on measured evidence:
   (1) **done** — raised `TOEPLITZ_W_MAX` 6 → 10 (AR(w) approaches any WSS precision as `w↑`, stays
   closed-form; BIC keeps the smallest sufficient order, so easy signals are unchanged); (2) **done** —
   the general `gmm-toeplitz-full` head, a dense PD Toeplitz covariance from the **biased** autocovariance
   (PSD by construction — simpler *and* cheaper than the alternating {Toeplitz} ∩ {PSD} projection first
   sketched, with no numerical optimizer), which recovers a long-lag-echo mixture (lag `K > w_max`) that
   AR cannot (ARI 0.70 → 0.97 vs chance); (3) **done (0.5.0)** — `method="gmm-toeplitz-gs"`, the full-order
   exact-GS **MLE** precision: a Yule-Walker (Levinson) warm start at order ≤ 16, refined by coordinate
   ascent of the exact log-likelihood over the reflection coefficients (PD by `|k| < 1`, deterministic) —
   faithful to the paper's likelihood-based estimator, made cheap by warm-starting from the moment
   estimator rather than optimizing the GS generators from scratch. Measured: competitive with the AR head
   on AR signals, and recovers mid-lag echo (lags 11–16 > `w_max`) the banded head misses, while the
   covariance-route `gmm-toeplitz-full` still covers arbitrarily long lags (the order cap is the trade-off
   for the `O(m·d·p)` precision E-step). The three routes are complementary: banded AR (cheapest, short
   memory), dense-covariance full (any lag), MLE precision (likelihood-optimal, cheaper E-step than full
   at large `d`).
4. **Do nothing; tell time-series users to preprocess.** The weak default, rejected: the head is a
   genuine capability no preprocessing recovers. This ADR records the design and the measured evidence
   alongside the 0.2.0 implementation.

## References

- arXiv:2311.14995 — *Positive-definiteness-ensuring likelihood-based estimation of Toeplitz
  (Gaussian-stationary) covariance and inverse-covariance matrices from few samples.*
- Prototype: `research/gmm_toeplitz_prototype.py`. GMM math: [`docs/MATH.md`](../MATH.md).
