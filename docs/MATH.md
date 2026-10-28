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

**Why this is the right bound, and not merely the winning one.** $\log\mathcal{N}(x \mid \mu_k, \Sigma_k)$
is a quadratic form in $x$, so its expectation under any distribution supported on leaf $i$ depends on
that distribution *only* through its first two moments — which are exactly what the cluster feature
stores. $\mathbb{E}_{x \sim i}[\log \mathcal{N}]$ is therefore **exact** given $(n_i, \mu_i, \Sigma_i)$,
with no error contributed by the unknown within-leaf shape. Running EM on that quantity is exact
**variational EM** on the point-level likelihood under the constraint that all points in a leaf share
one responsibility vector: a genuine ELBO whose only slack is the tying. The two alternatives are not
bounds of that kind. The convolution form $\mathcal{N}(\mu_i \mid \mu_k, \Sigma_k + \Sigma_i)$ computes
$\log \mathbb{E}[p] \ge \mathbb{E}[\log p]$ by Jensen — the wrong side, and a different functional —
which is why it washes out components once the summary is coarse. The plug-in form is the zeroth-order
truncation of the same expansion, dropping precisely the
$-\tfrac{1}{2}\mathrm{tr}(\Sigma_k^{-1}\Sigma_i)$ term above, so it agrees wherever $\Sigma_i$ is
negligible against $\Sigma_k$ — which is what the measurement shows (`research/RESULTS-estep.md`: 33/36
cells against 32/36 for the plug-in and 22/36 for the convolution). The same theorem in its Bregman
generalization — $\mathbb{E}_{x \in i}[d_\varphi(x, \mu_k)] = S_i/n_i + d_\varphi(\mu_i, \mu_k)$, exact
for every $\varphi$ — is what the `BregmanBetula` head's monotone `elbo` reports.

**Prior art.** This tied bound is published. Verbeek, Nunnink & Vlassis (*Data Min. Knowl. Discov.*
2006) run EM on it over kd-tree cells and show that refining the partition cannot lower it;
Kurihara, Welling & Vlassis (NIPS 2006) use the same bound for variational Dirichlet-process
mixtures. EMACF (Jin, Leung, Wong & Xu 2005) is its diagonal-covariance case. Two other summary
E-steps are not lower bounds: EMADS (Jin, Wong & Leung 2005) keeps only the first eigen-component of
the trace term, dropping $\sum_{j \ge 2} \lambda_j c_j^\top \Sigma_k^{-1} c_j \ge 0$ and so
over-stating the expected log-density; scalable EM (Bradley, Fayyad & Reina 1998) is the plug-in at
the summary mean. No published E-step on the same $(n, \mu, S)$ is tighter.

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
$M$ is small the matrices are tiny, so **no BLAS is needed** — the compression is what removes the
dependency and the memory growth, not a fast NMF kernel. The wall-clock effect is a **crossover, not a
uniform win**: measured against `sklearn.decomposition.NMF` → k-means the CF-weighted path is *slower*
at small $N$ and faster once $N$ passes roughly $10^5$ (0.2× / 0.7× / 1.3× at $N$ = 8 k / 40 k / 160 k,
`bench/RESULTS.md`), because its factorization cost is set by the leaf count while scikit-learn's grows
with $N$. What it buys unconditionally is a code
matrix that stops growing ($M \times r$, not $N \times r$) and a **canonical** factorization — unit-L2
parts, energy-ordered — whose ARI spread across seeds is 0.000 where scikit-learn's is 0.37.
Nonnegative input only; signed data is rejected rather than shifted (a shift would destroy
angles). For signed embeddings use the directional heads or reduce with PCA / TruncatedSVD first.

For **count** data (word counts, event tallies) the Frobenius objective assumes Gaussian noise, which is
mis-specified — counts are Poisson. `projection="weighted-nmf-kl"` minimizes the generalized-KL
(I-divergence) $\sum_{ij} [X_{ij}\ln(X_{ij}/(WH)_{ij}) - X_{ij} + (WH)_{ij}]$ by Lee-Seung multiplicative
updates over the raw centroids, with the shared components $H$ weighted by leaf mass $n_j$ (the per-row
$W$ update is weight-invariant, each row minimized independently) — the Poisson maximum-likelihood fit.
The gain is rate-dependent and, on the current tree, **small**: largest at **sparse** counts (+0.04 ARI
over Frobenius on a Poisson-count mixture at mean rate 0.2 — 0.892 against 0.850) and zero to three
decimals once the mean count passes $\sim0.4$, where the central-limit theorem pulls Poisson toward
Gaussian (`examples/17_nmf_topics.ipynb`). **Correction to the previous edition, which reported up to
+0.5 ARI (0.83 against 0.24).** That gap did not close because KL got worse — it closed because
Frobenius went from 0.24 to 0.85 at the same setting, once 0.6.0 moved the k-means head that consumes
the codes to its own partition and the mixture heads to maximum-posterior labelling. Most of what the
KL objective appeared to be buying was a labelling defect downstream of the factorization. The
objective remains the correct likelihood for Poisson counts; the measured margin over Frobenius is now
worth about four hundredths at the sparsest rate.

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
| `gmm`, `gmm-full`, `vmf`, `watson`, `gmm-toeplitz{,-full,-gs}` | $\arg\max_c\ \ln \pi_c + \ln p(x \mid \theta_c)$ |
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
plateau of the "number of modes vs `log h`" curve. At each scale, raw mean-shift modes separated by
only a **shallow density valley** (`ρ` along the connecting segment stays ≥ `VALLEY_RATIO = 0.8` of the
lower peak) are merged by prominence — this collapses the spurious sub-peaks a single cluster produces
at fine bandwidths, cleaning the curve so the persistent plateau is unambiguous.

Persistence is only a usable selector on the part of the curve that carries the merge cascade. The
swept range is $[\tfrac12 \operatorname{med}_j \delta_j,\ \tfrac12 \operatorname{diam}]$ with $\delta_j$
the nearest-leaf gap, and the last two clusters join far below that ceiling, so a single grid spends
most of its points on a trivial $k=1$ tail — always the longest flat run of a decreasing curve, hence
the winner whenever the cascade above it is strictly decreasing. Two things fix that: the sweep
**stops at the first single-mode scale**, leaving the trivial run one point long, and the range is then
**narrowed onto the cascade and re-swept**, so a plateau inside it has grid points to be measured on.
Narrowing is guarded on having seen at least two multi-mode scales, because refining below the merge
scale manufactures structure — a single Gaussian blob's refined curve reads `[3,3,3,2,2,…]` over a 3 %
span of `h`. Measured over 52 `(PCA dimension × leaf budget)` cells on `digits`, the single grid
answered $k=1$ in 42; the two-pass sweep does so in 3, at 2.45× the cost. This is parameter-free and
non-convex-aware, but still the weakest head on `covtype`, where every cell scores a negative ARI. See
`bench/RESULTS.md` before relying on it.

## Other verified improvements

- **Distances `D0`–`D4`** are the BIRCH measures re-derived on $(n, \mu, S)$ (Maxima-verified
  *equivalent*, computed stably). Variance-increase / Ward is $D_4 = \tfrac{n_A n_B}{n_{AB}}\,\|\Delta\mu\|^2$ — the $S$
  terms cancel by König–Huygens, so it is an exact centroid measure (no Lance-Williams approximation).
- **k-means on CFs** minimizes the true point objective, not the leaf-centroid proxy:
  $\text{SSE} = \sum_i \bigl[S_i + n_i\|\mu_i - c\|^2\bigr]$ folds each leaf's own scatter $S_i$ back in, so compressing to a
  CF-tree first does not change what is being optimized.
- **The boundary of the summary.** A cluster feature is a sum-decomposition
  $f(X) = \varphi\bigl(\sum_u \psi(x_u)\bigr)$ (Deep Sets, Zaheer et al. 2017), so it carries the
  permutation-invariant polynomials of degree $\le 2$ and nothing else. WCSS, Ward, Calinski–Harabasz,
  RMS Davies–Bouldin, $D_{4,\varphi}$ and both Gaussian E-steps are therefore *exact* on the summary;
  the exact silhouette, a single-linkage dendrogram and a DBSCAN labelling provably are not — two
  four-point sets with **bitwise-identical** features have different pairwise distance sets. The
  counterexample, the exactness residuals and the leaf-budget error rate are in
  `research/RESULTS-cf-boundary.md`.
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
- **Rebuild** merges the $k$ cheapest within-leaf sibling pairs, where $k$ is what the leaf budget asks
  for, and raises the threshold to the widest gap it took (monotone, $O(M \cdot \text{capacity})$ scan,
  no global all-pairs). Cheapest is by the exact cost of the merge,
  $\Delta\mathrm{SSE} = \frac{w_a w_b}{w_a + w_b}\lVert\mu_a - \mu_b\rVert^2$, whatever `distance`
  and `absorption` are set to: those decide whether a point belongs in a leaf, this decides which
  merge loses the least information, and under the default $D_0$ geometry the two disagree — the
  unweighted gap is mass-blind, so it merges the dense core into a few heavy leaves and leaves the
  rest as singletons. Two consequences. *In place*: merging two entries inside one leaf node leaves
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

## Symmetries of a (feature, head) pair

A clustering head answers a question about the *data*, not about the coordinates it arrived in.
Which change of frame it is allowed to ignore is a property of its model — and one that is easy to
lose by accident, to a bandwidth in absolute data units, a covariance floor added before whitening,
or an initialisation that reads coordinate 0. The groups the crate claims, and enforces in
`tests/equivariance.rs`:

| head | translate | rotate | uniform scale | swap axes |
|---|---|---|---|---|
| `kmeans`, `xmeans`, `ward_hac`, `agglomerative`, `dyn_msc`, `kmedoids`, `fuzzy_cmeans`, `hdbscan` | yes | yes | yes | yes |
| `gmm_full`, `mppca`, `spectral`, `scale_space` | yes | yes | yes | yes |
| `gmm_diagonal` | yes | **no** | yes | yes |
| `spherical_kmeans`, `movmf`, `watson` | **no** | yes | yes | yes |
| `gmm_toeplitz` | yes | **no** | yes | **no** |

Three rows are exceptions by construction, not defects. An axis-aligned covariance is a claim *about
the axes*, so rotation is not a symmetry of `gmm_diagonal`'s model; rotating a diagonal fit and
expecting the same answer is the error. The directional heads read a direction from the origin, so
translating the data moves it relative to the very thing they measure. And `gmm_toeplitz` is neither
Euclidean nor directional: an AR($w$) covariance says the coordinates are an evenly spaced
*sequence*, so shifting or rescaling the whole sequence leaves the model alone while reordering the
axes is a different model. The audit exhibits that on two AR processes that differ in *nothing but*
their autocovariance — each window standardised, so the marginal mean and variance carry no signal —
which the head separates exactly; sending lag-1 neighbours 64 apart then destroys the band structure
and the answer with it.

The head is only half of the pair. A symmetry the **leaf summary** has already discarded cannot be
recovered by any head reading it, so the group of a `(feature, head)` pair is the *intersection* of
the two:

| feature | what it keeps of the scatter | rotate |
|---|---|---|
| `Spherical` | its trace only | yes — nothing anisotropic is left to rotate |
| `Diagonal` | its axis-aligned part | **no** — the off-diagonal scatter is gone |
| `Full`, `FdSketch` | all of it (`FdSketch` up to sketch error) | yes |

So `gmm_full` — a rotation-invariant model — has three different symmetry groups depending on what
it is handed: invariant on `Full` leaves, *not* invariant on `Diagonal` ones, and invariant again on
`Spherical` ones for the trivial reason that an isotropic summary has nothing a rotation can change.
Choosing `feature="diagonal"` for memory is therefore also a choice about which frames the answer is
allowed to depend on.

The audit found no head violating its claimed group. That is only worth reporting because the tests
carry their own controls: each exception is asserted to *differ* rather than quietly skipped, and a
fixture check confirms the comparison can tell two answers apart at all — blobs separated far enough
that every head finds them in any frame would pass an invariance suite without exercising a single
symmetry.

## What the heads can and cannot guarantee

Apart from `dc-center` and `dc-median`, which are exact on their own objective, no head here
carries an approximation factor for the objective it is usually judged on, and for several of
those objectives no polynomial algorithm can carry one unless P = NP. This section says which, so
that a benchmark number is not read as a bound. Each entry names the exact objective, because the
hardness results below are about specific objectives and do not move to a neighbouring one. Five
of the sources are 2026 preprints from OpenAI's `openai/math` release, listed at the end of the
section; they are not peer-reviewed, and each statement taken from them is the preprint's claim.

Some of those claims are also machine-checked: the release formalises selected theorems in Lean and
states the scope of each formal statement (`lean/docs/` in the `openai/math` repository). A formal
check covers the stated theorem, not the paper around it, and it was not re-run here.

| Cited result | Machine-checked in its source? |
|---|---|
| Propeller bound $9/(8\pi)$ (propeller preprint, Theorem 1.1) | yes — every dimension, every number of cells |
| Metric k-median threshold $1 + 2/e$ (k-median preprint, Theorem 1.1, Corollary 1.2) | yes — with candidate facilities, the infimum under P ≠ NP |
| The Unique Games theorem | yes — the gap reduction from 3SAT |
| Uniform sparsest cut, constant-factor hardness (sparsest-cut preprint, Theorem 1.1) | no — only a companion paper's Goemans–Linial integrality gaps are formalised |
| Kernel clustering below $(8\pi/9)(1 - 1/k)$ (propeller preprint, Theorem 6.1) | no — the propeller formalisation leaves the kernel-clustering consequences out |
| Three-state threshold (three-state preprint, Corollary 1.2) | no — only reconstruction above the Kesten–Stigum bound on trees is formalised; non-reconstruction at or below it, and the block-model consequences, are not |

**`kmeans` — a closed-form target, not a guarantee.** For $N(0, I_d)$ the population k-means cost
per point has a closed form at $k = 2$ and $k = 3$. A planar sector of half-angle $\alpha$ has the
unnormalised Gaussian first moment $z = \tfrac{\sin\alpha}{\sqrt{2\pi}}\,(1, 0)$ (the propeller
preprint, §1.1). For $k$ cells of equal mass the between-cluster sum of squares per point is
$k \sum_i \lVert z(A_i)\rVert^2$, so

$$
\text{three } 120^\circ \text{ sectors:}\quad d - \frac{27}{8\pi} \approx d - 1.0743,
\qquad\qquad \text{two half-planes:}\quad d - \frac{2}{\pi} \approx d - 0.6366,
$$

with centres at radius $3^{3/2}/(2^{3/2}\sqrt{\pi}) \approx 1.0365$ and $\sqrt{2/\pi}$. Both
partitions are Lloyd fixed points by symmetry. That the three sectors are the *global* optimum is
supported numerically, not proven: the propeller bound caps $\sum_i \lVert z(A_i)\rVert^2$ at
$9/(8\pi)$ over every partition, but the k-means objective divides each term by the cell's mass.
The tests hold the tree plus the head within 0.01 of both values at $d = 2$ and $2\cdot 10^5$
rows (`tests/integration_api.rs`, `tests/test_python.py`), which needs no labels and no reference
implementation.

**`spectral` — the cut it relaxes is hard to approximate.** Uniform sparsest cut,
$\Phi(G) = \min_S \operatorname{cut}(S) / \bigl(|S|\,(N - |S|)\bigr)$ on a graph with nonnegative
rational capacities, is NP-hard to approximate within any fixed factor $C > 1$ (the sparsest-cut
preprint, Theorem 1.1). The two-way RatioCut is $N$ times the same ratio,

$$
\frac{\operatorname{cut}(S)}{|S|} + \frac{\operatorname{cut}(S)}{N - |S|}
= \frac{N\,\operatorname{cut}(S)}{|S|\,(N - |S|)},
$$

so the objective that *unnormalised* spectral clustering relaxes has no constant-factor polynomial
algorithm unless P = NP. The `spectral` head is Ng–Jordan–Weiss: it relaxes the *normalised* cut,
and the theorem is not about NCut. What carries over is the general point — the cut objectives
spectral methods relax are hard to approximate, so the eigenvector embedding is a heuristic for
them, not an approximation algorithm.

**`kmedoids` and `dc-median` — metric k-median.** For metric k-median with candidate facilities,
a $(1 + 2/e + \varepsilon)$-approximation exists for every fixed $\varepsilon > 0$, and under
P ≠ NP the infimum factor is exactly $1 + 2/e \approx 1.736$ (the k-median preprint, Theorem 1.1
and Corollary 1.2).
Neither head needs it. `dc-median` is exact: in the density-connectivity ultrametric the k-median
is a dendrogram knapsack (`src/clustering/dcdist.rs`). `kmedoids` minimises *squared* Euclidean
deviation, which violates the triangle inequality, so no metric bound applies to it, and FasterPAM
is a local search with no stated factor.

**Kernel clustering — cited only to stop a conflation.** Identity-target kernel clustering
maximises $\sum_{p,q} a_{pq}\, \mathbf{1}\{\sigma(p) = \sigma(q)\}$ over $\sigma : [N] \to [k]$ for a
centred positive-semidefinite $A$. For each fixed $k \ge 3$ it is NP-hard to approximate within any
loss factor below $(8\pi/9)(1 - 1/k)$: 1.862 at $k = 3$, tending to 2.793 (the propeller preprint,
Theorem 6.1, which also rests on the companion *The Unique Games theorem*). With $A$ a centred Gram
matrix this looks like k-means, but it is a different objective: k-means divides each cluster's sum
by the cluster's size, and this one does not. No head here optimises it.

**`ward`, `average` — factors a random tree also earns, or that need separation.** Take
$d_{ij} = \lVert x_i - x_j\rVert^2$ and the tree value $\sum_{i<j} d_{ij}\,|T_{ij}|$, where
$|T_{ij}|$ is the size of the smallest cluster holding both points. Average linkage scores at least
half of the trivial bound $N \sum_{i<j} d_{ij}$ (Cohen-Addad, Kanade, Mallmann-Trenn & Mathieu,
[arXiv:1704.02147](https://arxiv.org/abs/1704.02147), Theorem 6.2). Charikar, Chatziafratis &
Niazadeh ([arXiv:1808.02227](https://arxiv.org/abs/1808.02227)) cite $2/3$ for the published
analysis, show it is tight, and show that a uniformly random tree earns $2/3$ as well. On leaves the
`average` head is average linkage on $n_a$ coincident copies of each leaf, so the factor reaches the
points scaled by $1 - \rho$, with $\rho = \sum_a n_a S_a / (N S_{\text{tot}}) \le \max_a n_a / N$.
Ward's $k$-cut is a 2-approximation of the $k$-means cost only when the optimal clustering is weakly
$(2 + 2\sqrt2 + \varepsilon)$-centre-separated; without separation the factor is
$\Omega((3/2)^d)$ (Großwendt, Röglin & Schmidt, [arXiv:1907.05094](https://arxiv.org/abs/1907.05094),
Theorem 10). Dasgupta's cost admits no constant factor under the Small-Set Expansion hypothesis
(Charikar & Chatziafratis, [arXiv:1609.09548](https://arxiv.org/abs/1609.09548)), and no head
claims one. One exact statement does hold: Ward heights are monotone SSE increments, so among all
partitions of a Ward tree into $k$ subtrees the horizontal cut has the least SSE, and any score
monotone in SSE at fixed $k$ (Calinski–Harabasz among them) is maximised by it. `n_clusters=0`
therefore loses nothing to non-horizontal cuts under `ward`. Average linkage has no such property:
two leaves of mass 100 that sit 0.9 apart and two of mass 1 that sit 1 apart give a 3-cluster
horizontal cut of SSE $81/2$ against $1/2$ for the antichain that splits the heavy pair.

**`leiden` — no factor, and a measured detection gap.** Approximating maximum modularity within
any positive multiplicative factor is NP-hard (Dinh, Li & Thai,
[arXiv:1602.01016](https://arxiv.org/abs/1602.01016)). Detection has an exact limit on the symmetric
sparse three-community stochastic block model, with rates $a/n$ inside a community, $b/n$ between,
and mean degree $(a + 2b)/3 > 1$. Weak recovery is possible exactly when

$$
s = \frac{(a - b)^2}{3\,(a + 2b)} > 1 .
$$

At or below the line every estimator's expected overlap tends to chance, and above it an
$O(n \log n)$ algorithm exists (the three-state preprint, Corollary 1.2, through the Mossel–Sly–Sohn
tree-to-graph transfer and Abbe–Sandon). The impossibility half is the preprint's claim and is not
in its formal scope; in the peer-reviewed literature the Kesten–Stigum bound is known to be tight
for three states only on regular trees of large degree (Sly, *Ann. Probab.* 39 (2011) 1365–1406).
The core is a deterministic Leiden variant, with greedy refinement and no well-connectedness
checks, so it carries no $\gamma$-connectivity guarantee beyond what its tests check. The
modularity core ($\gamma = 1$) measured at mean degree 6, ARI medians of graph seeds 0/1/2,
communities in contiguous id blocks:

| $s$ | 0.6 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| ARI, $n = 3\,000$ | 0.0015 | 0.0029 | 0.0196 | 0.1537 | 0.6565 | 0.8715 | 0.9965 |
| ARI, $n = 30\,000$ | 0.0002 | 0.0003 | 0.0014 | 0.0167 | 0.5626 | 0.7978 | 0.9963 |
| communities, $n = 30\,000$ | 114 | 122 | 119 | 112 | 100 | 108 | 77 |

The core needs $s \approx 4$, four times the threshold signal. At $s = 3$ it reads 0.0167 at
$n = 30\,000$, against 0.07–0.31 at the smaller $n$ measured (1 500 to 12 000). With the node ids
shuffled, both rows move by at most 0.062, in both directions, and most where the curve is
steepest. Most of the returned communities are isolated vertices: about $e^{-6} n \approx 74$ at
$n = 30\,000$. The in-module test of `src/clustering/community.rs` uses both ends as controls: at
$s = 0.6$ an ARI above 0.005 could only come from the node ids, and at $s = 5$ the blocks must come
back. The `leiden` head builds its own kNN affinity graph, so this tests the core, not the user
path.

The table is one Leiden pass, the default and the only behaviour before `n_iterations`. One pass
stops at a local optimum of its last aggregate graph; the guarantees of Traag, Waltman & van Eck
(2019) are about the iterated algorithm. `n_iterations` (Rust: `clustering::leiden_iterated`)
starts every further pass from the previous partition, and on the same graphs (medians of seeds
0/1/2) it closes most of the gap at $s \ge 4$:

| | $s = 0.6$ | $s = 4$ | $s = 5$ |
|---|---|---|---|
| ARI, $n = 3\,000$: one pass / two / until stable | 0.0015 / 0.0026 / 0.0024 | 0.6565 / 0.8967 / 0.9325 | 0.8715 / 0.9776 / 0.9766 |
| ARI, $n = 30\,000$: one pass / two / until stable | 0.0002 / 0.0002 / 0.0002 | 0.5626 / 0.8696 / 0.9014 | 0.7978 / 0.9779 / 0.9796 |
| passes until stable, $n = 30\,000$ | 53 / 51 / 69 | 29 / 32 / 18 | 9 / 5 / 9 |

As an outside reference, python-igraph 1.0
(`community_leiden`, modularity, $\beta = 0.01$, `n_iterations=-1`) reads ARI 0.923 / 0.910 / 0.915
on the same graphs at $n = 30\,000$, $s = 4$, against 0.901 / 0.904 / 0.901 here, and both settle
at modularity 0.548. Modularity never fell from one pass to the next in any of these runs. Below the threshold the
passes raise modularity by fitting noise (0.369 to 0.415 at $n = 30\,000$) and recover nothing,
so the $s = 0.6$ control holds at every pass count. At $s = 3$, $n = 30\,000$, two passes move
ARI only from 0.017 to 0.032, and the partition is still changing after 143 to more than 200
passes, at 0.11–0.23; `n_iterations < 0` stops at 100 passes. igraph does not settle there either:
on the same graphs three of four of its runs were still changing after 300 passes and the fourth
stopped at 235, so the cap decides nothing a longer run would. It stays at 100. On the benchmark's own leaf graphs
the passes matter much less: with the `bench/comprehensive.py` call (800 leaves, 20 000-row draws,
seeds 0/1/2) the median ARI over one pass / two / until stable reads 0.728 / 0.731 / 0.731 on
digits, 0.281 / 0.283 / 0.311 on MNIST and 0.046 / 0.046 / 0.046 on covtype. The default stays at
one pass, because a second one changes the labels of every 1.x release.

**Mass-weighted leaf graphs.** The leaf graph above gives every leaf one node of size 1, joined to
its neighbours by the affinity $w_{ij}$ of the leaf means. Leaf masses are far from equal (more
than half of the digits leaves at 800 hold one point), so the leaf objective is that of no point
graph. `leaf_graph="mass"` partitions the quotient of the point graph instead (Traag, Waltman &
van Eck 2019, App. A): with every cross-leaf pair of points joined at its leaves' $w_{ij}$, the
edge between leaves $i$ and $j$ is $n_i n_j w_{ij}$ and node $i$ has size $n_i$. A partition of the
leaves then scores exactly what the same partition of their points scores, under modularity and
CPM alike; `the_mass_graph_scores_a_leaf_partition_as_the_point_graph_scores_its_points` checks it
against the expanded graph. Within-leaf pairs are left out. Modelling them needs a kernel inside a
leaf, which the leaf means do not set: the Jensen bound $e^{-2r_i^2/\sigma_i^2}$ lifts digits at 800
leaves by 0.048 and loses up to 0.012 elsewhere, and kernels corrected for the leaf spread lose up
to 0.10 on MNIST, where that spread at $d = 784$ dwarfs the $\sigma$ the means set.

Median ARI, seeds 0/1/2, 20 000-row draws, threshold 0, one pass / until stable:

| leaves | 400 | 800 | 1 600 | 3 200 |
|---|---|---|---|---|
| MNIST, `unit` | 0.225 / 0.224 | 0.281 / 0.282 | 0.419 / 0.426 | 0.443 / 0.446 |
| MNIST, `mass` | 0.307 / 0.325 | 0.420 / 0.415 | 0.438 / 0.456 | 0.503 / 0.504 |
| digits, `unit` | 0.654 / 0.662 | 0.728 / 0.731 | 0.819 / 0.831 | 0.811 / 0.822 |
| digits, `mass` | 0.681 / 0.674 | 0.693 / 0.696 | 0.824 / 0.828 | 0.811 / 0.822 |
| covtype, `unit` | 0.055 / 0.055 | 0.045 / 0.045 | 0.043 / 0.043 | 0.040 / 0.040 |
| covtype, `mass` | 0.045 / 0.045 | 0.043 / 0.043 | 0.033 / 0.033 | 0.030 / 0.030 |

Two references bound what a graph change can do. The optimiser is not the limit: python-igraph's
Leiden on the same unit leaf graphs reads within 0.03 of these rows in eleven cells of twelve. The
data is: Leiden on a point-level graph (15 nearest neighbours, Jaccard weights) reads 0.46 on
MNIST, 0.79 on digits and 0.02 on covtype, so on covtype no leaf graph has anything to recover and
the `mass` row moves inside noise. Digits at 3 200 leaves is a control: 1 797 rows give every leaf
one point, and the two graphs coincide. `unit` stays the default for 1.x because `mass` changes the
labels.

The spectral head takes the same parameter. Its `mass` operator is KASP's (Yan, Huang & Jordan 2009,
Lemma 8): the normalized affinity of the graph that replicates leaf $i$ into $n_i$ points has the
spectrum of $R^{1/2} D^{-1/2} A D^{-1/2} R^{1/2}$ with $R = \mathrm{diag}(n_i)$, which is the
symmetric normalisation of the quotient $n_i n_j w_{ij}$, plus one zero per extra replica;
`the_mass_operator_has_the_spectrum_of_the_replicated_point_graph` checks it against the expanded
graph. A replica's embedding row is its leaf's scaled by $n_i^{-1/2}$, which the row normalisation
removes, so the k-means that rounds the embedding weighs leaf $i$ by $n_i$. Rounding the same
operator unweighted loses 0.05–0.09 on MNIST at 800 and 1 600 leaves, gains 0.04 at 3 200, and
changes digits by at most 0.004, so the digits loss below comes from the operator, not the rounding.

Spectral, median ARI over seeds 0/1/2, 20 000-row draws, threshold 0, the true `k`:

| leaves | 400 | 800 | 1 600 | 3 200 |
|---|---|---|---|---|
| MNIST, `unit` | 0.089 | 0.205 | 0.284 | 0.322 |
| MNIST, `mass` | 0.336 | 0.411 | 0.435 | 0.382 |
| digits, `unit` | 0.625 | 0.669 | 0.680 | 0.669 |
| digits, `mass` | 0.644 | 0.653 | 0.659 | 0.669 |
| covtype, `unit` | 0.008 | 0.073 | 0.070 | 0.030 |
| covtype, `mass` | 0.003 | 0.041 | 0.035 | 0.027 |

Two moons and two circles at 20 000 points read 1.000 on both graphs at every budget; one seed of
three falls to 0.55 on the moons at 3 200 leaves with `mass`.

Sources (2026 preprints, OpenAI):

- *The Gaussian propeller bound in every dimension*, 24 September 2026 —
  [paper](https://github.com/openai/math/blob/main/preprints/The-Gaussian-Propeller-Bound-in-Every-Dimension-September-24-2026/main.pdf).
- *Constant-factor hardness of uniform sparsest cut*, 24 September 2026 —
  [paper](https://github.com/openai/math/blob/main/preprints/Constant-factor-hardness-of-uniform-sparsest-cut-September-24-2026/Constant-factor-hardness-of-uniform-sparsest-cut-September-24-2026.pdf).
- *The approximation threshold for metric k-median*, 24 September 2026 —
  [paper](https://github.com/openai/math/blob/main/preprints/The-Approximation-Threshold-for-Metric-k-Median-September-24-2026/main.pdf).
- *The exact reconstruction threshold for the three-state symmetric channel*, 25 September 2026 —
  [paper](https://github.com/openai/math/blob/main/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/paper.pdf).
- *The Unique Games theorem*, 23 September 2026 —
  [paper](https://github.com/openai/math/blob/main/preprints/The-Unique-Games-Theorem-September-23-2026/paper.pdf).

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
clusterers practitioners actually reach for: at **matching ARI**, betula labels 1 M points **37×
faster** than `sklearn.cluster.Birch` (8.04 s → 0.22 s) and **10.8×** faster than `KMeans`, while
streaming memory stays flat at ~53 MB; see [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md) and the
[method-comparison notebook](https://github.com/ilgrad/betula-cluster/blob/main/examples/04_method_comparison.ipynb). (betulars produces no labels, so
it is not in that comparison; on the raw Phase-1 *build* the two are at parity — betula-cluster builds
an **identical tree** at every `N` and, with matched `target-cpu=native` flags, matches betulars'
wall-clock to within ~2 %.)
