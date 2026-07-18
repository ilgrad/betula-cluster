# Mathematical foundation & improvements

Every formula below is verified symbolically (Maxima) and/or numerically (mpmath/Julia ground
truth) — see [`DESIGN.md`](https://github.com/ilgrad/betula-cluster/blob/main/DESIGN.md) and `research/`.

## Numerically stable cluster features `(n, μ, S)`

Classic BIRCH stores $(N,\ LS = \sum x,\ SS = \sum x^2)$ and recovers variance as $SS/N - (LS/N)^2$ — a
**difference of two large, nearly-equal numbers**. On real data with an offset (timestamps, money,
geo-coordinates, un-centered embeddings) this **catastrophically cancels**: in `f64` the variance
collapses to noise — and can go *negative* — around coordinate magnitude `1e7`, silently corrupting
every downstream radius, covariance, and label.

betula-cluster stores $(n, \mu, S)$ — weight, mean, and the sum of squared deviations
$S = \sum w\,(x - \mu)(x - \mu)^\top$. $S$ is a **sum of non-negative terms**, so the variance $S/n \ge 0$ and the
covariance is **positive-semidefinite by construction** — there is nothing to cancel. The updates are
algebraically exact (not approximations):

$$
\begin{aligned}
\text{add (Welford/West):}\quad & W' = W + w, \quad \mu \leftarrow \mu + \tfrac{w}{W'}\,\delta, \quad S \leftarrow S + w\Bigl(1 - \tfrac{w}{W'}\Bigr)(\delta \odot \delta), \quad \delta = x - \mu_\text{old} \\
\text{merge (Chan):}\quad & W = W_A + W_B, \quad \mu = \mu_A + \tfrac{W_B}{W}\,\Delta, \quad S = S_A + S_B + \tfrac{W_A W_B}{W}(\Delta \odot \Delta), \quad \Delta = \mu_B - \mu_A
\end{aligned}
$$

(full covariance: $\delta \odot \delta \to \delta\delta^\top$, $\Delta \odot \Delta \to \Delta\Delta^\top$). Tested bit-stable at offset `1e7–1e8` against an
mpmath reference, where the classic $(N, LS, SS)$ form loses all significant digits.

## GMM E-step: expected-log (variant C), not the paper's convolution

Running EM on the leaf CFs, each leaf is a mini-Gaussian $\mathcal{N}(\mu_i, \Sigma_i)$. The textbook / paper move
treats it by **convolution** — $\mathcal{N}(\mu_i \mid \mu_k, \Sigma_k + \Sigma_i)$ — which inflates each component by the
leaf's own spread and **washes out components on a coarse CF-tree**. betula-cluster instead uses the
**expected log-likelihood** responsibility (measured to give higher ARI — `research/RESULTS-estep.md`):

$$
\log r_{ik} = \log \pi_k + \log \mathcal{N}(\mu_i \mid \mu_k, \Sigma_k) - \tfrac{1}{2}\operatorname{tr}(\Sigma_k^{-1}\Sigma_i) \qquad \text{(then log-sum-exp normalized)}
$$

The M-step folds the within-leaf scatter back in, so the fitted components stay exact:

$$
\Sigma_k = \frac{1}{N_k}\sum_i w_{ik}\bigl(\Sigma_i + (\mu_i - \mu_k)(\mu_i - \mu_k)^\top\bigr), \qquad w_{ik} = n_i\,r_{ik}
$$

with NIW/MAP regularization $\Sigma_k = (\Psi + \dots)/(\nu + N_k + d + 1)$ so a 1-point leaf never yields a
singular covariance.

**High-dimensional floor.** With few effective leaves per component, $\Sigma_k$ can still go
near-singular along a low-variance direction, which makes the $-\tfrac12\operatorname{tr}(\Sigma_k^{-1}\Sigma_i)$
correction over-confident and *starves* the component — its responsibility collapses to zero and the
recovered count drops below `k`. A per-dimension floor on each component's covariance **diagonal** at
$10^{-3}\,(\Sigma_\text{global})_{dd}$ — relative to the global *per-dimension* variance (not the mean
scale, which between-cluster separation inflates), with off-diagonals / orientation left untouched —
keeps every $\Sigma_k$ well-conditioned. On 64-dimensional `digits` it holds all 10 components (an
unfloored fit starves one to 9) and raises full-covariance ARI $0.39 \to 0.51$, past scikit-learn's
`GaussianMixture` (0.40), while low-dimensional and rotated-anisotropic fits are unchanged. The
diagonal GMM floors its per-dimension variance the same way ($10^{-3}\,(\Sigma_\text{global})_{dd}$).

## Toeplitz / AR covariance for stationary signals

For an ordered **wide-sense-stationary** signal the covariance is Toeplitz, $\Sigma_{ts} = c(|t-s|)$, so
it is fixed by an autocovariance sequence rather than a dense $d\times d$ matrix. `method="gmm-toeplitz"`
models each component covariance as an **AR(w)** process. The pooled biased autocovariance

$$
r_c(\tau) = \frac{1}{N_c\,d}\sum_i w_{ic}\Bigl[\textstyle\sum_t \delta_{it}\,\delta_{i,t+\tau} + [\tau=0]\operatorname{tr}\Sigma_i\Bigr],
\qquad \delta_i = \mu_i - \mu_c,
$$

(from the leaf mean deviations, with the within-leaf variance folded into the zero lag) is mapped by the
**Levinson-Durbin** recursion, kept at *every* intermediate order $m = 0..w$ (predictor $\phi_m$, error
variance $v_m$). The precision is the **exact Gohberg-Semencul** form $\Gamma = (1/\sigma^2)(BB^\top - ZZ^\top)$
($B$ lower-triangular Toeplitz from the AR coefficients; $Z$ the corner correction), evaluated by the
**prediction-error decomposition**
$\log p(\delta) = -\tfrac12\sum_{t=0}^{d-1}\bigl[\ln(2\pi v_{m_t}) + (\delta_t - \phi_{m_t}\!\cdot\delta_{t-1:t-m_t})^2/v_{m_t}\bigr]$,
$m_t = \min(t, w)$ — so the first $w$ boundary positions are modelled *exactly* (the $-ZZ^\top$ term)
rather than dropped by a conditional likelihood (measured $+0.04$ ARI at short windows). It is
**positive-definite by construction** — the reflection-coefficient clamp $|k_m| \le 0.999$ ($\Rightarrow v_m > 0$)
is the GS box constraint — with $O(w)$ parameters and $O(d\,w)$ cost per leaf. The order $w$ is chosen
per component by BIC, and the mean is a single stationary scalar (one parameter,
not $d$). This is well-posed at $N_k \ll d$, where full covariance is singular and a diagonal model is
blind to the neighbour correlation. Ordered coordinates only — a permutation destroys the structure.
(Gohberg-Semencul Toeplitz-precision estimation, arXiv:2311.14995; see
[`docs/adr/001-gmm-toeplitz.md`](adr/001-gmm-toeplitz.md).)

## Directional clustering: spherical k-means & von Mises–Fisher

On L2-normalized data every point lies on the unit sphere `S^{d-1}`, where cosine — not Euclidean —
similarity is the meaningful geometry (CLIP / face / sentence / speaker embeddings). Two heads cluster
by direction: **spherical k-means** (hard) and a **mixture of von Mises–Fisher** distributions (soft).

**Exact merge on the sphere.** A leaf summarizing points `{xₚ}` on the sphere is reduced to its
weighted mean `μ_i = (Σ xₚ)/n_i`, whose length `‖μ_i‖ = R̄_i ∈ [0, 1]` is the leaf's *mean resultant
length* — a direct measure of within-leaf angular concentration. The cluster resultant is
`R_c = Σ_{i∈c} n_i μ_i = Σ_{p∈c} xₚ`, additive across leaves and independent of how points were
grouped: the BETULA exact-merge property carries through unchanged. The MLE mean direction is
`μ̂_c = R_c / ‖R_c‖` and `R̄_c = ‖R_c‖ / N_c`. Keeping `μ_i` **un-normalized** is essential —
re-normalizing each leaf to a unit direction discards `R̄_i`, makes the compression look artificially
concentrated, over-estimates `κ`, and fragments the mixture.

**Concentration.** The vMF concentration uses the Banerjee et al. (2005) closed form
`κ̂ ≈ R̄(d − R̄²)/(1 − R̄²)`, which avoids inverting the Bessel ratio. The normalizer
`C_d(κ) = κ^{d/2−1} / ((2π)^{d/2} I_{d/2−1}(κ))` still needs `log I_ν(κ)`; we take it from the
all-positive power series in log-space — pull out `(κ/2)^ν`, accumulate the term ratio
`(κ/2)² / (m(ν+m))` with an online log-sum-exp — which is stable for large `κ` and needs no Bessel
library (the crate stays NumPy-only). `κ` is capped for numerical safety.

The EM E-step is the exact expected log-likelihood of a leaf's points under component `c`,
`n_i·[ln π_c + log C_d(κ_c)] + κ_c · μ_c · R_i` with `R_i = n_i μ_i` the raw resultant, so a
spread-out leaf contributes proportionally weaker evidence — the directional analogue of the
full-covariance GMM's within-leaf `−½ tr(Σ_c⁻¹ Σ_i)` correction. `predict_proba` returns this true
posterior; `n_clusters=0` selects the component count by BIC.

## Geometry-aware graph (GeoBETULA) and scale-space modes

Two heads exploit the geometry *within* each microcluster, on the `M ≪ N` leaves.

**GeoBETULA (`method="leiden"`).** The self-tuning k-NN affinity graph normally uses the centroid
distance `‖μ_i − μ_j‖²`. Two optional terms make it geometry-aware: a **log-Euclidean** covariance
term `β·‖logΣ_i − logΣ_j‖²_F` (the SPD-manifold metric, `covariance_weight`) so neighbours agree in
*shape*, and a **Grassmann** term `γ·d²_Gr(U_i, U_j)` with `d²_Gr = r − ‖U_iᵀ U_j‖²_F` — the
projection distance between the two rank-`r` principal subspaces `U_i` (top-`r` eigenvectors of `Σ_i`),
`tangent_weight` — so neighbours agree in *manifold orientation*. This separates crossing or adjacent
manifolds that share a centroid neighbourhood but differ in local tangent. Both reuse the in-house
Jacobi eigensolver; both default to `0` (plain centroid affinity).

**Scale-space modes (`method="scale-space"`).** Treat the leaves as a weighted sample and take the
modes of the KDE `ρ_h(x) = Σ_j n_j exp(−‖x−μ_j‖²/2h²)` (found by mean-shift) as clusters. Increasing
the bandwidth `h` merges modes — a one-parameter Morse filtration. Rather than fix `h` (or `k`), the
head sweeps `h` log-spaced and reports the labelling at the **most persistent** mode count: the widest
plateau of the "number of modes vs `log h`" curve, with the trivial fully-merged tail winning only when
no multi-mode structure is at least as persistent. At each scale, raw mean-shift modes separated by
only a **shallow density valley** (`ρ` along the connecting segment stays ≥ `VALLEY_RATIO = 0.8` of the
lower peak) are merged by prominence — this collapses the spurious sub-peaks a single cluster produces
at fine bandwidths, cleaning the curve so the persistent plateau is unambiguous (robust from 2 to ~8+
clusters). This is parameter-free and non-convex-aware.

## Other verified improvements

- **Distances `D0`–`D4`** are the BIRCH measures re-derived on $(n, \mu, S)$ (Maxima-verified
  *equivalent*, computed stably). Variance-increase / Ward is $D_4 = \tfrac{n_A n_B}{n_{AB}}\,\|\Delta\mu\|^2$ — the $S$
  terms cancel by König–Huygens, so it is an exact centroid measure (no Lance-Williams approximation).
- **k-means on CFs** minimizes the true point objective, not the leaf-centroid proxy:
  $\text{SSE} = \sum_i \bigl[S_i + n_i\|\mu_i - c\|^2\bigr]$ folds each leaf's own scatter $S_i$ back in, so compressing to a
  CF-tree first does not change what is being optimized.
- **Full covariance** uses a matrix Welford (PSD) with on-demand Cholesky for `logdet` /
  `mahalanobis`; the packed upper-triangular index is the tested $(j-1)j/2$ form — a reference
  implementation shipped a $(j-1)\cdot\mathrm{dim}/2$ variant that silently corrupts $\mathrm{dim} \ge 4$.
- **$\chi^2$ absorption gate** (`absorb="chi2"`): a mass-invariant Mahalanobis-$\chi^2$ threshold with a
  Normal-Inverse-Gamma prior $\text{var}_\text{eff} = (S + \kappa s_0)/(n + \kappa)$, finite at $n = 1$. Fixes the
  size-imbalance failure where a 12-point vs a $10^4$-point cluster decide differently (sklearn #22854).
- **Frequent-Directions sketch** (high $d$): the full-cov GMM consumes it in **low-rank** form —
  $\operatorname{tr}(\Sigma_k^{-1}\Sigma_i) = \sum_r \|L_k^{-1} f_r\|^2$ — so it never materializes a $d \times d$ matrix per leaf and keeps
  $O(\ell d)$ memory through clustering. Identical math to the dense path.
- **Rebuild threshold** is the within-leaf mean nearest-sibling gap (ELKI/BETULA-standard,
  $O(M \cdot \text{capacity})$), raised monotonically — no global all-pairs scan, no over-growth collapse.
- **Robust insertion** (`huber_k = k`, optional): before a point $x$ is folded into its target
  microcluster $(n, \mu, S)$, each coordinate is winsorized to the cluster's own scale,

  $\tilde{x}_j = \operatorname{clip}\bigl(x_j,\ \mu_j - k\sigma_j,\ \mu_j + k\sigma_j\bigr), \quad \sigma_j = \sqrt{S_j / n}$

  and the stable Welford update runs on $\tilde{x}$ instead of $x$. A coordinate with $\sigma_j = 0$ (no scale
  yet) passes through unchanged, and the clip is skipped until the target holds ≥ 5 points (so the
  scale estimate is trustworthy). This bounds any single point's pull on the centroid to $O(k\sigma/n)$
  — outliers can no longer drag a centroid or inflate a radius — while leaving the CF a valid
  $(n, \mu, S)$ triple, so every downstream head is unchanged. The clipped value flows identically
  into the leaf entry and its ancestors, preserving the "each node = merge of its subtree" invariant.

## Relation to BIRCH and BETULA

This library is a from-scratch Rust implementation of the **BETULA** cluster feature — the
numerically stable $(n, \mu, S)$ summary introduced by Lang & Schubert to replace classic BIRCH's
cancellation-prone $(N, LS, SS)$:

- **BETULA: Numerically Stable CF-Trees for BIRCH Clustering** — Andreas Lang & Erich Schubert,
  *SISAP 2020* ([arXiv:2006.12881](https://arxiv.org/abs/2006.12881) ·
  [Springer](https://link.springer.com/chapter/10.1007/978-3-030-60936-8_22)); extended journal
  version, *Information Systems* 2022
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306437921001253)).
- **BIRCH** — Zhang, Ramakrishnan & Livny, *SIGMOD 1996*.

Reference implementations: [ELKI](https://elki-project.github.io/) (Java), and
[**betulars**](https://pypi.org/project/betulars/) ([source](https://github.com/andiwg/betula)) — a
Rust+PyO3 package by paper co-author **Andreas Lang**. betulars is a faithful, highly optimised
**Phase-1 CF-tree builder**: it builds the tree and exposes leaf cluster statistics, but (as of
v0.1.0) produces **no cluster labels** and **no global clustering** — k-means / GMM / hierarchical are
listed as planned. If you need *just* the canonical BETULA CF-tree primitive, fast, use betulars.

**betula-cluster is a different thing: an end-to-end clustering library.** It re-derives the same
stable CF from scratch and then adds everything betulars leaves to the user:

- the full Phase-2 pipeline — k-means / GMM / **full-covariance** GMM / Ward / HDBSCAN with
  automatic `k` — producing **per-point labels** and `predict`, behind a real **scikit-learn API**
  (the de-facto Python BIRCH, `sklearn.cluster.Birch`, is *classic* BIRCH: the unstable CF);
- `f32` trees, streaming `partial_fit`, a mass-invariant **χ² absorption gate**, a
  Frequent-Directions sketch for high `d`, `normalize=True` for embeddings, an inspection API
  (outliers / near-duplicates / representatives / geometry), and serde persistence;
- auto-vectorized distance kernels (tight inline reductions) and rayon-parallel build + labeling.

The concrete, reproducible quality/speed/memory comparison is against the labeled scikit-learn
clusterers practitioners actually reach for: at **matching ARI**, betula labels 1 M points **~40×
faster** than `sklearn.cluster.Birch` (8.0 s → 0.20 s) and **~17×** faster than `KMeans`, while
streaming memory stays flat at ~57 MB; see [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md) and the
[method-comparison notebook](https://github.com/ilgrad/betula-cluster/blob/main/examples/04_method_comparison.ipynb). (betulars produces no labels, so
it is not in that comparison; on the raw Phase-1 *build* the two are at parity — betula-cluster builds
an **identical tree** at every `N` and, with matched `target-cpu=native` flags, matches betulars'
wall-clock to within ~2 %.)
