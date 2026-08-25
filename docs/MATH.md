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
**expected log-likelihood** responsibility, which is measured to beat the convolution on ARI and is
the correct expected log-likelihood under the leaf model; against the simpler plug-in that drops the
trace it is a wash outside coarse, overlapping summaries (`research/RESULTS-estep.md`, three seeds):

$$
\log r_{ik} = \log \pi_k + \log \mathcal{N}(\mu_i \mid \mu_k, \Sigma_k) - \tfrac{1}{2}\mathrm{tr}(\Sigma_k^{-1}\Sigma_i) \qquad \text{(then log-sum-exp normalized)}
$$

The M-step folds the within-leaf scatter back in, so the fitted components stay exact:

$$
\Sigma_k = \frac{1}{N_k}\sum_i w_{ik}\bigl(\Sigma_i + (\mu_i - \mu_k)(\mu_i - \mu_k)^\top\bigr), \qquad w_{ik} = n_i\,r_{ik}
$$

with NIW/MAP regularization $\Sigma_k = (\Psi + \dots)/(\nu + N_k + d + 1)$ so a 1-point leaf never yields a
singular covariance.

**High-dimensional floor.** With few effective leaves per component, $\Sigma_k$ can still go
near-singular along a low-variance direction, which makes the $-\tfrac12\mathrm{tr}(\Sigma_k^{-1}\Sigma_i)$
correction over-confident and *starves* the component — its responsibility collapses to zero and the
recovered count drops below `k`. A per-dimension floor on each component's covariance **diagonal** at
$10^{-3}\,(\Sigma_\text{global})_{dd}$ — relative to the global *per-dimension* variance (not the mean
scale, which between-cluster separation inflates), with off-diagonals / orientation left untouched —
keeps every $\Sigma_k$ well-conditioned. On 64-dimensional `digits` it holds all 10 components (an
unfloored fit starves one to 9) and the floored full-covariance head reaches ARI **0.575** against
scikit-learn's `GaussianMixture` at **0.463** (median of seeds 0/1/2), while low-dimensional and
rotated-anisotropic fits are unchanged. The unfloored figure of 0.39 that this pair was first measured
against dates from a 0.2.0 build and has not been re-taken; the floored-vs-scikit-learn gap has. The
diagonal GMM floors its per-dimension variance the same way ($10^{-3}\,(\Sigma_\text{global})_{dd}$).

## Toeplitz / AR covariance for stationary signals

For an ordered **wide-sense-stationary** signal the covariance is Toeplitz, $\Sigma_{ts} = c(|t-s|)$, so
it is fixed by an autocovariance sequence rather than a dense $d\times d$ matrix. `method="gmm-toeplitz"`
models each component covariance as an **AR(w)** process. The pooled biased autocovariance

$$
r_c(\tau) = \frac{1}{N_c\,d}\sum_i w_{ic}\Bigl[\textstyle\sum_t \delta_{it}\,\delta_{i,t+\tau} + [\tau=0]\mathrm{tr}\Sigma_i\Bigr],
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

**General (non-AR) Toeplitz covariance** (`method="gmm-toeplitz-full"`). AR(w) has a *banded* precision,
so it cannot represent an autocovariance whose support exceeds order $w$ (e.g. a single echo at lag
$K > w$). The general head drops the banded assumption: it forms the dense Toeplitz covariance
$\Sigma_{ts} = r^{\mathrm b}_c(|t-s|)$ directly from the **biased** ($\div d$) autocovariance $r^{\mathrm b}_c$.
The biased sequence is periodogram-consistent, so $\Sigma$ is **positive-semidefinite by construction**
(each leaf contributes the autocorrelation of its zero-padded deviation, a nonnegative spectrum; a
nonnegative-weighted sum stays PSD), and the $\div d$ bias shrinks high-lag terms — a free regularization
exactly where $N_k \ll d$. A ridge (the within-leaf variance plus $10^{-6}\,r^{\mathrm b}_c(0)$) makes it
strictly PD; a Cholesky factor gives the exact multivariate-Gaussian log-density
$-\tfrac12(d\ln 2\pi + \ln|\Sigma| + \delta^\top\Sigma^{-1}\delta)$. Cost is $O(d^2)$ parameters and $O(d^3)$
per component (vs AR's $O(d\,w)$), so it is the opt-in rung for signals a low-order AR cannot fit: on a
long-lag-echo mixture ($K\in\{16,28,40\}>w_{\max}=10$) it reaches ARI $0.73\!\to\!1.00$ as the window grows
where the AR head sits at chance, and it matches the AR head on AR-generated signals
(`bench/toeplitz_ar_mixture.py`).

A third route, `method="gmm-toeplitz-gs"`, estimates the general Toeplitz **precision** by the paper's
**Gohberg-Semencul MLE**: a full-order (`≤ 16`) Yule-Walker (Levinson) fit refined by coordinate ascent of
the exact log-likelihood over the reflection coefficients $k_m$, positive-definite by $|k_m| < 1$ and
deterministic. The reflection-coefficient parameterization makes the constraint free — the step-up
recursion maps any $|k| < 1$ to a stable predictor with $v_m > 0$ — so the MLE refines the moment
estimator toward the likelihood optimum without leaving the PD cone. Its $O(m \cdot d \cdot p)$ E-step is
cheaper than the dense $O(d^3)$ covariance route at large $d$; it captures structure up to its order cap
(mid-lag echoes the banded head misses), while the covariance route covers arbitrarily long lags.

## CF-weighted NMF for nonnegative data

For **nonnegative** features (TF-IDF, bag-of-words, event counts, spectrogram magnitudes, histograms) a
nonnegative low-rank factorization $X \approx W H$, $W, H \ge 0$, is often the natural representation — its
parts $H$ are interpretable and additive. Factorizing the raw $N \times d$ matrix is $O(N d r)$ per
iteration and holds an $N \times r$ code matrix, defeating BETULA's compression. `projection="weighted-nmf"`
factorizes the $M \ll N$ leaf **centroids** instead. Assigning every point in leaf $C_j$ the same code
$z_j$ (the hard-leaf approximation every Phase-3 head already makes), König-Huygens gives

$$
\sum_{x_i \in C_j} \lVert x_i - z_j H \rVert^2 = \underbrace{\sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2}_{\text{within-leaf scatter (const in } z,H)} + n_j \lVert \mu_j - z_j H \rVert^2 ,
$$

so minimizing the full-data objective is equivalent — up to that constant — to the **weighted centroid**
problem $\min_{Z,H \ge 0} \sum_j n_j \lVert \mu_j - z_j H \rVert^2 = \lVert \tilde X - \tilde Z H \rVert_F^2$
with $\tilde X_j = \sqrt{n_j}\,\mu_j$, $\tilde Z_j = \sqrt{n_j}\,z_j$. The factorization runs over the
microclusters ($O(M d r)$ per sweep, memory-bounded) and any head then clusters the nonnegative codes
$z_j$. The solver is a dependency-free weighted **HALS** (coordinate descent, reusing the Gram /
cross-product matrices $HH^\top$, $\tilde X H^\top$, $W^\top W$, $W^\top \tilde X$ across sweeps); because
$M$ is small the matrices are tiny, so no BLAS is needed — **the compression, not a fast NMF kernel, is the
speed-up**. Nonnegative input only; signed data is rejected rather than shifted (a shift would destroy
angles). For signed embeddings use the directional heads or reduce with PCA / TruncatedSVD first.

For **count** data (word counts, event tallies) the Frobenius objective assumes Gaussian noise, which is
mis-specified — counts are Poisson. `projection="weighted-nmf-kl"` minimizes the generalized-KL
(I-divergence) $\sum_{ij} [X_{ij}\ln(X_{ij}/(WH)_{ij}) - X_{ij} + (WH)_{ij}]$ by Lee-Seung multiplicative
updates over the raw centroids, with the shared components $H$ weighted by leaf mass $n_j$ (the per-row
$W$ update is weight-invariant, each row minimized independently) — the Poisson maximum-likelihood fit.
The gain is rate-dependent: largest at **sparse** counts (measured up to **+0.5 ARI** over Frobenius on a
Poisson-count mixture at mean rate $< 0.5$), narrowing to a few points as the mean count grows past $\sim1.5$
and the central-limit theorem pulls Poisson toward Gaussian.

**Initialization and the scale gauge.** Both solvers start from **NNDSVDar** (Boutsidis & Gallopoulos,
2008): a rank-$r$ truncated SVD — obtained from a randomized range finder (Halko-Martinsson-Tropp:
Gaussian sketch, two power iterations, then a small eigendecomposition of $BB^\top$, so no LAPACK is
needed) — whose $k$-th singular triplet yields a nonnegative pair by keeping whichever of its positive
/ negative parts carries more energy. The resulting zeros must be filled: zero is a fixed point of both
HALS and the multiplicative updates, so an entry left at zero could never recover.

Two details of that fill decide whether the factorization is usable at all, because **zero is absorbing**
— a component driven to zero on one sweep is gone permanently:

* *Rank-deficient triplets.* The right vector is recovered as $v = B^\top u / \sigma$, so once $\sigma$
  falls to the noise floor the division amplifies round-off into a vector of arbitrary magnitude. Such
  triplets are cut at the LAPACK numerical-rank threshold $\sigma \le \sigma_{\max}\,\max(M,d)\,\varepsilon$
  and reported as exact zeros, to be seeded by the fill instead.
* *Fill scale.* A filled component is a rank-1 block of constant magnitude $f^2$ per entry, and $r$ of
  them add up against data entries of size $\operatorname{mean}(X)$. The plain `a` variant's
  $f = \operatorname{mean}(X)$ therefore swamps the data as $r$ grows — measured on a rank-12 matrix at
  $r = 32$: initial relative residual **13.5**, and the first sweep annihilated **28 of 32** components.
  The `ar` variant's $f = \operatorname{mean}(X)\cdot U(0,1)/100$ keeps the fill a perturbation, and the
  randomness breaks the degeneracy a constant fill would create (identical, linearly dependent columns
  that no coordinate descent can separate). Measured effect: 0 components dead, converged residual
  **270×** lower, and on `digits` at $r = 24$ the reconstruction error drops **0.33 → 0.20** — matching
  `scikit-learn`'s own NMF to three decimals at every rank tested ($r = 10, 16, 24, 32$).

Both objectives are invariant to $(WD, D^{-1}H)$ for any positive diagonal $D$ — the optimizer therefore
leaves an arbitrary per-component scale (measured spreads of $70\times$ between components on a converged
fit). That gauge freedom is harmless for the reconstruction but not for us: $W$ leaves the factorizer as a
**Euclidean feature vector** for the Phase-3 head, where a per-component scale is a per-dimension weight,
so the head silently clusters along whichever component drew the largest number. The returned
factorization is therefore canonical — $\lVert H_k \rVert_2 = 1$ with the scale absorbed into $W$, and
components ordered by descending energy. Measured on a 4-topic nonnegative mixture over 8 seeds, at
$N = 8\,000 / 40\,000 / 160\,000$: median ARI **0.81 / 0.99 / 0.97 → 1.00** and seed spread
**±0.37 → ±0.00**. The gain is determinism, not accuracy in the mean. Convergence is tested on the **size of the update** — the total coordinate movement of a sweep, against
the first sweep's — not on the size of the objective. A relative test on the residual never fires here:
HALS converges sublinearly, so it keeps buying more than $\texttt{tol}$ of relative improvement for
hundreds of sweeps and the iteration budget ends up the only brake. The movement is scale-free, does
converge, and falls out of the sweep at no extra cost.

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
`C_d(κ) = κ^{d/2−1} / ((2π)^{d/2} I_{d/2−1}(κ))` still needs `log I_ν(κ)`, and no Bessel library is
pulled in for it (the crate stays NumPy-only) — two evaluators are split at `κ = 10⁴`:

- **below**, the all-positive power series in log-space — pull out `(κ/2)^ν`, accumulate the term
  ratio `(κ/2)² / (m(ν+m))` with an online log-sum-exp. Nothing overflows, but the peak term sits at
  `m ≈ κ/2`, so the cost is `O(κ)` and the loop's own stop truncates *before* the peak above
  `κ ≈ 4·10⁵`. Large `κ` is not a stability problem for this series; it is a cost problem that turns
  into a correctness problem.
- **above**, DLMF 10.41.3, the uniform asymptotic expansion for large order, in `O(1)`:
  `I_ν(νz) ~ e^{νη}/(√(2πν)·(1+z²)^{1/4}) · Σₖ Uₖ(p)/ν^k` with `η = √(1+z²) + ln(z/(1+√(1+z²)))` and
  `p = 1/√(1+z²)`. Three terms (`U₀..U₂`) are exactly what f64 needs from `κ = 10⁴` up: measured
  against 50-digit arithmetic over `ν ∈ [1, 2047]`, the f64 result lands within **0.8 ulp**, while
  stopping at `U₁` costs some 300 ulps and a fourth term changes nothing.

The expansion is written in `z = κ/ν` and divides by `ν = d/2 − 1`, so it exists only for `d ≥ 4`.
Below that the series is the sole evaluator, and the concentration cap is what keeps it in range:
**10⁶ for `d ≥ 4`, 10⁴ for `d ≤ 3`**. The cap is a limit of the normalizer, not of the model — a
cluster tighter than it is already effectively a point. Extending `d ≤ 3` would need a separate
small-order expansion and its own validation.

The EM E-step is the exact expected log-likelihood of a leaf's points under component `c`,
`n_i·[ln π_c + log C_d(κ_c)] + κ_c · μ_c · R_i` with `R_i = n_i μ_i` the raw resultant, so a
spread-out leaf contributes proportionally weaker evidence — the directional analogue of the
full-covariance GMM's within-leaf `−½ tr(Σ_c⁻¹ Σ_i)` correction. `predict_proba` returns this true
posterior; `n_clusters=0` selects the component count by BIC.

## Labelling a raw point

The CF-tree is a *summary*, not the model. A head fits its parameters to the leaves, and the label of
a new point follows from that head's own objective — not from where the tree happens to route it:

| head | rule |
|---|---|
| `kmeans`, `spherical-kmeans` | $\arg\min_c \lVert x - c \rVert^2$ over the cluster centres (spherical compares unit-normalized centres, where the Euclidean argmin and the cosine argmax agree) |
| `gmm`, `gmm-full`, `vmf`, `gmm-toeplitz{,-full,-gs}` | $\arg\max_c\ \ln \pi_c + \ln p(x \mid \theta_c)$ |
| `ward`, `spectral`, `leiden`, `hdbscan`, `scale-space` | nearest leaf entry, then that entry's label |

The third row is not a fallback but the only defined answer: those clusters need not be convex, so any
centre or density rule would impose a partition the head exists to avoid. The first two rows *were*
computed that way until they were fixed, and the tree descent is greedy — an approximate
nearest-microcluster search that disagreed with the model on 3–28% of points.

The mixture densities are the plain component log-densities: the E-step's within-leaf correction
$-\tfrac{1}{2}\operatorname{tr}(\Sigma_c^{-1}\Sigma_i)$ exists because a *leaf* has scatter, and a single
observation has none. Each head keeps the exact numbers its own EM converged to — the floored diagonal
variances, the ridge-regularized Cholesky, the AR predictor bank or Toeplitz factor — so the point rule
cannot drift from the fit. `predict_proba` normalizes the same scores, which makes
`predict_proba(X).argmax(1)` equal to `predict(X)` by construction. A component that ends up claiming
no leaf is silenced, so `predict` can only name a label the fitted partition actually uses.

**Where the diagonal head is weak.** Being the model's own rule is not the same as being the most
accurate one, and the gap shows up wherever the model is a poor fit. `method="gmm"` treats every
dimension as independent, so scoring a raw point sums `d` separate penalties and a per-dimension
modelling error accumulates across all of them; the leaf-level score damps this through the
$-\tfrac{1}{2}\operatorname{tr}(\Sigma_c^{-1}\Sigma_i)$ term, which a single observation does not have.
On raw image pixels — strongly correlated, and exactly what a diagonal covariance cannot represent —
that costs accuracy: MNIST-20k ARI **0.340** by leaf descent against **0.185** by posterior, where
TF-IDF text gains (20-newsgroups `gmm` **0.027 → 0.054**) and so does `digits` (**0.489 → 0.507**). The
fitted covariance is what costs it: on the same fit a nearest-centre rule, which ignores the covariance
entirely, scores **0.378** — above both. The loss is specific to a fine leaf budget, not to the head:
at `max_leaves=300` the posterior wins for `gmm` (**0.206 → 0.239**) and for `gmm-full`
(**0.207 → 0.212**) alike. For raw images prefer `kmeans` or a `projection`.

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

**Scale-space modes (`method="scale-space"`).** Treat the leaves as a weighted sample of **Gaussians,
not points** — a leaf is a cloud, and convolving that cloud with the kernel is what the data's own
kernel density is, $N(\mu_j, \Sigma_j) * N(0, h^2 I) = N(\mu_j, \Sigma_j + h^2 I)$. With
$\sigma_j^2 = S_j/(n_j d)$ from the summary, every leaf carries its own width $s_j^2 = h^2 + \sigma_j^2$:

$$\rho_h(x) = \sum_j n_j\, s_j^{-d}\, \exp\bigl(-\|x - \mu_j\|^2 / 2 s_j^2\bigr)$$

The $s_j^{-d}$ amplitude is what stops a fat leaf peaking as high as a tight one, and the mean-shift
fixed point becomes $x = \sum_j (w_j/s_j^2)\mu_j / \sum_j (w_j/s_j^2)$ (Comaniciu, Ramesh & Meer 2001);
at zero leaf scatter both collapse to the point-kernel form exactly. Take the
modes of this KDE (found by mean-shift) as clusters. Increasing
the bandwidth `h` merges modes — a one-parameter Morse filtration. Rather than fix `h` (or `k`), the
head sweeps `h` log-spaced and reports the labelling at the **most persistent** mode count: the widest
plateau of the "number of modes vs `log h`" curve, with the trivial fully-merged tail winning only when
no multi-mode structure is at least as persistent. At each scale, raw mean-shift modes separated by
only a **shallow density valley** (`ρ` along the connecting segment stays ≥ `VALLEY_RATIO = 0.8` of the
lower peak) are merged by prominence — this collapses the spurious sub-peaks a single cluster produces
at fine bandwidths, cleaning the curve so the persistent plateau is unambiguous. This is parameter-free
and non-convex-aware — and **its operating envelope is narrow**: measured over 52
`(PCA dimension × leaf budget)` cells on `digits`, 38 return a single cluster, and the plateau selector
rather than the kernel is what the measurement indicts. See `bench/RESULTS.md` before relying on it.

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
  $\mathrm{tr}(\Sigma_k^{-1}\Sigma_i) = \sum_r \|L_k^{-1} f_r\|^2$ — so it never materializes a $d \times d$ matrix per leaf and keeps
  $O(\ell d)$ memory through clustering. Identical math to the dense path. The shrink step subtracts
  the lower-median $\sigma^2$ from every direction; that trace is banked rather than discarded, so
  $S = \mathrm{tr}(B^\mathsf{T}B) + \text{lost}$ is the leaf's *exact* scatter, and the covariance is
  reported as $\Sigma = \frac{S}{\mathrm{tr}(B^\mathsf{T}B)}\cdot\frac{B^\mathsf{T}B}{n}$ — the retained
  directions scaled to carry the missing mass. Measured against the isotropic alternative in
  `bench/RESULTS.md`; without either, the sketch reports as little as 66 % of its own scatter.
- **Height-bounded exact HAC** (`clustering::dendrogram_below`, Rust API): every merge below a chosen
  height $h_{\max}$, computed over a candidate graph instead of all pairs, and **exact** rather than
  approximate. The whole argument is one inequality — the minimum over cross pairs is at most their
  (mass-weighted) mean, and that mean is
  $\|\Delta\mu\|^2 + S_A/n_A + S_B/n_B$ for *any* pair of clusters. What differs per linkage is how
  the linkage value bounds it:
  - **average / weighted**: the value *is* that mean, so $\min \le h$ and the radius is $r^2 = h$;
  - **Ward**: the value is $2\frac{n_A n_B}{n_A + n_B}\|\Delta\mu\|^2$ and a cluster's own scatter is
    the sum of the $D_4$ of the merges inside it, $S_A \le (m_A - 1)h/2$, so
    $\min \le \frac{h}{2}\bigl(\frac{m_A}{n_A} + \frac{m_B}{n_B}\bigr) \le h/w_{\min}$ with $w_{\min}$
    the lightest leaf;
  - **centroid / median**: the value is $\|\Delta\mu\|^2$ alone, which bounds neither spread — two
    concentric shells have coincident centroids and no close cross pair at all. **No radius exists**,
    at any height, and the call returns an error rather than a silently wrong dendrogram.

  Exactness then follows in two lines: the candidate set is a subset of the pairs, so the candidate
  minimum is $\ge$ the true minimum; and if the true minimum is $\le h_{\max}$ the radius puts its
  edge in the graph, so the candidate minimum is $\le$ it. Equality at every step below $h_{\max}$ —
  and the moment the candidate minimum exceeds $h_{\max}$ the same argument says the true one does
  too, so stopping there is right rather than heuristic. The graph must therefore be the **exact**
  radius graph: an approximate k-NN index may omit the one edge the certificate depends on, which
  costs the guarantee and leaves an approximation indistinguishable from an exact answer. See
  `bench/RESULTS.md` for where this pays and where it does not.
- **Rebuild** merges the $k$ closest within-leaf sibling pairs, where $k$ is what the leaf budget asks
  for, and raises the threshold to the widest gap it took (monotone, $O(M \cdot \text{capacity})$ scan,
  no global all-pairs). Two consequences. *In place*: merging two entries inside one leaf node leaves
  every node CF exactly equal to the merge of its subtree, so no ancestor is touched; the reinsertion
  that follows merges nothing and only re-routes, which is the one thing compaction cannot do (shrink a
  mixed leaf, yes; split it, no). *Cliff-safe*: in high dimension distances
  concentrate, so the leaf count is near-discontinuous in the threshold (measured on 3000-d TF-IDF:
  7755 leaves at $\tau = 1.0$, 12 at $\tau = 1.3$) and any threshold-first policy either fails to reduce
  or collapses the tree; choosing $k$ instead caps a rebuild at one merge per entry.
- **Mass-balanced budget** (`balance = b`, optional): the absorption gate above is purely geometric,
  so nothing bounds how much *mass* one leaf accumulates. With `balance` set, a leaf entry is a
  candidate for absorption or for a compaction merge only while its weight stays under

  $w_{\max} = \max\bigl(b \cdot W / M,\ 2\bigr), \qquad W = \text{total mass seen so far},\ M = \texttt{max\_leaves}$

  so $W/M$ is the perfectly balanced ideal and $b$ the slack allowed above it. Reading $W$ from the
  root CF makes the cap self-tuning on a stream — it tightens as data arrives — and the floor of 2
  keeps it from forbidding the merge of two singletons during warm-up. The cap is enforced at both
  sites because either one alone undoes the other: absorption refuses a full entry and starts a new
  one, and the rebuild skips a pair that would overflow. It is **soft** and $M$ is **hard**: a rebuild
  that cannot reach its target under the cap merges over it rather than leave the tree over budget.
- **Robust insertion** (`huber_k = k`, optional): before a point $x$ is folded into its target
  microcluster $(n, \mu, S)$, each coordinate is winsorized to the cluster's own scale,

  $\tilde{x}_j = \mathrm{clip}\bigl(x_j,\ \mu_j - k\sigma_j,\ \mu_j + k\sigma_j\bigr), \quad \sigma_j = \sqrt{S_j / n}$

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
- AVX2/FMA distance kernels with a scalar fallback, and rayon-parallel build + labeling.

The concrete, reproducible quality/speed/memory comparison is against the labeled scikit-learn
clusterers practitioners actually reach for: at **matching ARI**, betula labels 1 M points **30×
faster** than `sklearn.cluster.Birch` (8.01 s → 0.26 s) and **9×** faster than `KMeans`, while
streaming memory stays flat at ~60 MB; see [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md) and the
[method-comparison notebook](https://github.com/ilgrad/betula-cluster/blob/main/examples/04_method_comparison.ipynb). (betulars produces no labels, so
it is not in that comparison; on the raw Phase-1 *build* the two are at parity — betula-cluster builds
an **identical tree** at every `N` and, with matched `target-cpu=native` flags, matches betulars'
wall-clock to within ~2 %.)
