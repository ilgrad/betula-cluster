# ADR 001 — Toeplitz / AR-structured GMM covariance head (`gmm-toeplitz`)

**Status:** Proposed · target **0.3.0**, experimental (off by default) · not scheduled for 0.2.0.

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
- **E-step, all `O(d·w)`** (no dense `d×d`): `logdet Σ = −d log σ²` (from the Levinson
  prediction-error product), Mahalanobis `δᵀΓδ = ‖A δ‖² / σ²` (whitening-residual energy), and the
  within-leaf correction `tr(Γ Σ_i)` from the banded `Γ` against the CF second moment.
- **Parameters:** `O(w)` per component instead of `O(d²)` — well-posed at `N_k ≪ d`.
- **Scope guard:** documented for ordered / stationary signals **only**; explicitly *not* for generic
  embeddings, where coordinate order is meaningless and a permutation destroys the Toeplitz structure
  (expected effect there is negative). Off by default.
- **No new dependencies** — Levinson-Durbin + banded matvec are pure arithmetic (stays NumPy-only).

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
3. **The paper's exact GS closed-form estimator** (general Toeplitz precision, not just AR(w)). More
   faithful to arXiv:2311.14995 and strictly more expressive, but heavier to implement and to tune the
   PD constraint set. **Deferred** — ship the validated AR/Levinson route first; revisit the GS
   closed form only if AR(w) proves too restrictive on real signals.
4. **Do nothing; tell time-series users to preprocess.** The weak default. This ADR records the design
   so it is ready when 0.3.0 scope is set, without blocking the 0.2.0 release.

## References

- arXiv:2311.14995 — *Positive-definiteness-ensuring likelihood-based estimation of Toeplitz
  (Gaussian-stationary) covariance and inverse-covariance matrices from few samples.*
- Prototype: `research/gmm_toeplitz_prototype.py`. GMM math: [`docs/MATH.md`](../MATH.md).
