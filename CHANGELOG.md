# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **CF-tree: the rebuild spent a third of the leaf budget, and collapsed the tree outright in high
  dimension.** `max_leaves` is a resolution budget — the summary handed to the global clustering is
  only as fine as the leaves actually kept — but the rebuild grew the threshold to the *mean*
  within-leaf nearest-sibling gap and then merged whatever fell under it, which is a prediction, not a
  target. Measured utilization of the budget: **65.6–96.2%** (mean 80.6%) across digits, blobs at
  n = 20 000/100 000 and d = 10/50, and uniform d = 20. On 3000-dimensional TF-IDF it failed
  catastrophically instead — **3 leaves against a 2000 budget**, one of them holding 85.7% of the mass,
  ARI 0.0 for every method and every seed. The cause is concentration of measure: the achievable leaf
  count is near-discontinuous in the threshold (7755 leaves at `threshold=1.0`, **12** at `1.3`), so no
  threshold-first policy has a safe value there. The rebuild now merges the `k` closest sibling pairs
  with `k` set by the budget and reads the grown threshold off the widest gap it took — `k` is exact,
  and merging is capped at one pair per entry, so the cliff cannot be stepped over. Utilization is now
  **90.0–99.0%** (mean 93.5%) on the same datasets and **90.0/94.1%** on TF-IDF (was 0.6/0.2%), where
  ARI goes 0.0 → 0.071 with the `spherical-kmeans` head.
- **CF-tree: the rebuild conflated reducing the entry count with rebalancing the node structure.**
  Merging two entries *inside their own leaf node* leaves every node CF in the tree exactly unchanged —
  a node's CF is the merge of its subtree, and merging two of its children does not change that
  multiset union — so mass is conserved per node, no ancestor needs touching, and no leaf can be
  emptied. The count reduction is therefore done in place, in one `O(Σ_leaf child_count²)` scan plus an
  `O(m log m)` sort, and the reinsertion pass that follows now merges **nothing**: it only re-routes
  entries so that leaves re-partition around the geometry the data actually has, which compaction
  alone cannot do (it can shrink a leaf that mixes two clusters but never split it). Keeping absorption
  on during that pass is what walks off the concentration cliff — measured, it collapses a d = 50 blob
  mixture to **9 leaves against a 500 budget**. Rebuilds became more frequent (each shaves 10% rather
  than overshooting) and individually cheaper; end-to-end build time moved **−6% to +98%** across the
  eight configurations, with the two largest (blobs n = 100 000, d = 50) essentially unchanged at
  +1% and +3%. `n_rebuilds_` counts a different, cheaper unit of work than before and is not comparable
  across the change.

  Quality follows the head, not the leaf count, and the finer summary is not uniformly better: on
  digits (5 seeds, median) the `kmeans` head goes **0.646 → 0.667** while the `gmm` head goes
  **0.593 → 0.458** at `max_leaves=500` — the diagonal-covariance head degrades as leaves get smaller
  and is best on that dataset at ~230 leaves, which is now a `max_leaves` choice rather than something
  the rebuild does behind the caller's back. Blob mixtures stay at ARI 1.000 throughout.
- **CF-weighted NMF: the solver stopped after 5 sweeps regardless of `max_iter`.** The convergence test
  compared `|prev − err|` against `tol · prev` with `prev` seeded at `+inf`; IEEE-754 says `inf <= inf`,
  so the first check always fired and the iteration budget was dead code. Every other EM loop in the
  crate guards this with `it > 0`; `nmf.rs` was the only one that did not. The budget is now honoured,
  and a regression test asserts the residual keeps falling as the budget grows.
- **CF-weighted NMF: most components collapsed to zero and never recovered.** Two independent causes,
  both fatal because zero is an absorbing state for HALS and for the multiplicative updates. (1) The
  randomized SVD recovers the right vector as `v = Bᵀu/σ`; below the noise floor that division amplified
  round-off into a vector of arbitrary magnitude, so a rank-deficient triplet seeded a component wildly
  out of scale. Such triplets are now cut at the LAPACK numerical-rank threshold and reported as exact
  zeros. (2) NNDSVD's zero-fill used `mean(X)`, but a filled component is a rank-1 block of constant
  magnitude and `r` of them add up, so the fill swamped the data as the rank grew. The init now uses the
  `ar` fill, `mean(X)·U(0,1)/100` — small enough to stay a perturbation, random enough to break the
  degeneracy a constant fill creates. Measured on a rank-12 matrix at rank 32: initial relative residual
  **13.5 → 1.2**, dead components **28/32 → 3/32** (3 is the honest count — the data is rank 12), and the
  converged residual **270×** lower. On `digits` at rank 24: **15/24 → 0** dead, reconstruction error
  **0.33 → 0.20**, downstream ARI **0.54 → 0.61**. The reconstruction error now matches `scikit-learn`'s
  own NMF to three decimals at every rank tested (10, 16, 24, 32), against being **2.3× worse** before.
- **CF-weighted NMF: an arbitrary per-component scale reweighted the downstream clustering.** NMF is
  invariant to `(W D, D⁻¹H)`, so the optimizer left whatever split it landed on — measured spreads of
  70× between component scales on a converged fit. Because the codes are consumed as a Euclidean
  feature vector by the Phase-3 head, that scale acts as a per-dimension weight and the head clustered
  along whichever component drew the largest number. The factorization is now canonical (component rows
  unit-L2, scale absorbed into the codes, components ordered by descending energy). Measured on a
  4-topic nonnegative mixture over 8 seeds: median ARI **0.63 → 1.00**, seed spread **0.37 → 0.00**;
  against the shipped 0.5.0 behaviour, median **0.92–1.00 → 1.00** with spread **0.31 → 0.00** at
  N = 160 000. `scikit-learn`'s own `NMF` shows the same 0.37 spread on this data, unfixed.

### Added
- **NNDSVDar initialization** for both NMF solvers (Boutsidis & Gallopoulos, 2008), built on a
  self-contained randomized range finder (Halko-Martinsson-Tropp) — deterministic given the seed and far
  better conditioned than the previous random start, which decided the basin a non-convex coordinate
  descent lands in.
- `Betula.components_` — the NMF parts `H` as `(projection_dim, dim)`, unit-L2 rows ordered by
  descending energy, so a row reads directly as a topic over the input features.
- `Betula.reconstruction_err_` — relative reconstruction error `‖X̃ − W H‖_F / ‖X̃‖_F` of the projection
  over the leaf centroid matrix.
- `projection_max_iter` (default 100) — the factorizer's own sweep budget, independent of the head's
  `max_iter`. Previously the two shared one number, so raising the clustering budget silently paid for
  NMF sweeps too.
- `projection="weighted-nmf"` / `"weighted-nmf-kl"` now accept **sparse CSR** input; only the stored
  values are checked for negativity (implicit zeros are already nonnegative). Measured ARI 1.00 on a
  sparse topic mixture, matching the dense path.
- A convergence check for the KL solver (it previously always burned the full `max_iter`). The Frobenius
  solver's check now follows the **size of the update** rather than the size of the objective, compared
  against the first sweep's — sklearn's rule. A relative test on the residual provably never fired: HALS
  converges sublinearly and keeps buying more than `tol` of relative improvement for hundreds of sweeps,
  so `max_iter` was the only brake (measured: a 4× budget cost 6× the time). The movement falls out of
  the sweep for free, so the check is also cheaper than the residual it replaced.

### Changed
- **`WᵀX`, the NMF sweep's hot loop, restructured for locality: measured 2.4-3.4× faster.** Evaluating
  it per output cell (`Σ_j w[j][k]·x[j][c]`) walks a whole column of `X` for each of the `r·d` cells,
  striding `d` floats per step through a matrix far larger than L2. Accumulating into the small,
  cache-resident `r×d` output while reading each row of `X` once, sequentially, computes the same
  product with the access pattern the hardware wants (verified equal to 1e-9 by a unit test).
- The NMF rank is capped by both centroid-matrix dimensions (`min(rank, d, M)`): `rank > M` makes the
  factorization rank-deficient by construction and leaves whole components with nothing to fit.

### Notes
- **A-HALS extrapolation (Ang & Gillis, 2019) was implemented, measured and rejected.** It lost to plain
  HALS at every sweep budget on two planted problems (up to 445× worse residual at 200 sweeps): the
  accept/reject test for the extrapolation parameter has to judge the objective at the *feasible*
  iterate, but the Gram factors on hand belong to the extrapolated point, so making it honest costs the
  extra `O(Mdr)` product the acceleration was meant to save. The cheap exact convergence check that came
  out of the attempt was kept.

## [0.5.0] — 2026-07-18

### Added
- `method="gmm-toeplitz-gs"` — **full-order Gohberg-Semencul MLE** Toeplitz-precision GMM for ordered
  stationary signals: a Yule-Walker (Levinson) warm start at order ≤ 16, then coordinate ascent of the
  exact log-likelihood over the reflection coefficients (positive-definite by the `|k| < 1` constraint) —
  the likelihood-optimal general precision (arXiv:2311.14995), completing the three-rung Toeplitz ladder
  of [`docs/adr/001-gmm-toeplitz.md`](docs/adr/001-gmm-toeplitz.md). Reuses the exact GS precision E-step;
  BIC auto-`k`; a true `predict_proba`. Measured: competitive with the AR head on AR signals and recovers
  mid-lag echo structure (lags 11–16, beyond the banded cap `w_max = 10`) the banded head is blind to,
  while the dense-covariance `gmm-toeplitz-full` covers arbitrarily long lags (`bench/toeplitz_ar_mixture.py`).
- `projection="weighted-nmf-kl"` — **KL-divergence (I-divergence) CF-weighted NMF** for **count** data
  (word counts / event tallies / Poisson observations), where the Frobenius objective (Gaussian noise) is
  mis-specified. Lee-Seung multiplicative updates with the shared components weighted by leaf mass — the
  Poisson maximum-likelihood fit. The advantage is largest where counts are **sparse**: measured **up to
  +0.5 ARI over Frobenius** on a Poisson-count mixture at mean rate < 0.5 (0.83 vs 0.24), narrowing to a
  few points as counts grow and Poisson → Gaussian (`examples/17_nmf_topics.ipynb`).

### Notes
- NMF Phase-2 warm-start / randomized range-finder were assessed and **deferred** (see
  `plans/nmf-cf-weighted.md`): in the CF-compressed regime the factorization already runs over the
  `M ≪ N` leaves, so the compression — not these — is the speedup; they would add API/state for a marginal
  gain and are not shipped.

## [0.4.0] — 2026-07-18

### Added
- `projection="weighted-nmf"` (+ `projection_dim`) — **CF-weighted nonnegative matrix factorization** as
  a Phase-3 reducer for **nonnegative** data (TF-IDF / bag-of-words / event counts / spectrogram
  magnitudes / histograms). Rather than factorizing the raw `N×d` matrix (which defeats the compression),
  it factorizes the `M ≪ N` leaf **centroids** weighted by their mass, `X̃_j = √n_j·μ_j`: by
  König-Huygens the full-data NMF objective equals the weighted-centroid one up to the within-leaf
  scatter constant, so the expensive factorization runs over the microclusters — memory-bounded, `O(M·d·r)`
  — and any head (k-means / GMM / Leiden) then clusters the nonnegative codes. The solver is a
  dependency-free weighted **HALS** (coordinate descent, Gram-reuse across sweeps); the compression, not a
  fast NMF, is the speedup. Available on the one-shot `fit_predict` and the streaming `Betula` estimator.
  Signed input is rejected (no silent shifting — use `vmf` / `spherical-kmeans` or PCA / TruncatedSVD for
  embeddings); dense input in this release. See [`docs/MATH.md`](docs/MATH.md).

## [0.3.0] — 2026-07-18

### Added
- `method="gmm-toeplitz-full"` — **general (non-AR) positive-definite Toeplitz-covariance GMM** for
  ordered wide-sense-stationary signals whose autocovariance a low-order AR cannot capture (e.g. a
  long-lag echo / narrowband structure beyond the AR order). Each component covariance is the dense
  Toeplitz matrix built from the **biased (periodogram-consistent) autocovariance** — positive-
  semidefinite by construction, made strictly positive-definite by the within-leaf variance plus a
  small ridge — factored by Cholesky for an exact multivariate-Gaussian E-step. `O(d²)` parameters,
  `O(d³)` per component; BIC auto-`k` at `n_clusters=0`; a true posterior via `predict_proba`. This is
  the general (non-AR) rung of the Toeplitz ladder recorded in
  [`docs/adr/001-gmm-toeplitz.md`](docs/adr/001-gmm-toeplitz.md). On a long-lag-echo mixture (echo lag
  `K ∈ {16, 28, 40}`, all beyond the AR order) it recovers the components (ARI 0.70 → 0.97 as the window
  grows) where the banded `gmm-toeplitz` sits at chance; on AR-generated signals the two match
  (`bench/toeplitz_ar_mixture.py`).

### Changed
- `gmm-toeplitz`: raised the internal AR-order cap `w_max` 6 → 10. BIC still selects the smallest
  sufficient order (easy signals are bit-for-bit unchanged); higher-order / MA-like signals gain
  headroom before the general `gmm-toeplitz-full` head is needed.

## [0.2.0] — 2026-07-18

### Added
- `method="spherical-kmeans"` / `method="vmf"` — **directional clustering on the unit hypersphere**
  for L2-normalized embeddings (CLIP / face / sentence / speaker vectors), where cosine — not
  Euclidean — geometry is what matters. `spherical-kmeans` is hard cosine assignment
  (`argmax_c μ̂·μ_c`, centers re-normalized to the sphere); `vmf` is a soft **mixture of
  von Mises–Fisher** distributions (EM, a true posterior for `predict_proba`, and BIC auto-`k` when
  `n_clusters=0`). Both reduce each leaf to its weighted mean `(n_i, μ_i)`, so the cluster resultant
  `R_c = Σ n_i μ_i` stays **exactly mergeable** — the BETULA property carries through to the sphere —
  and the within-leaf spread `‖μ_i‖` feeds the concentration `κ` (Banerjee et al. 2005), so
  microcluster compression does not over-estimate it. The engine L2-normalizes input automatically
  for these methods (`get_params` stays verbatim). The `κ` normalizer uses a dependency-free,
  numerically stable `log I_ν(κ)` (log-space series) — no Bessel library, the crate stays NumPy-only.
  Available on the dense one-shot / streaming estimator, the sparse (`O(nnz)`) path, and as the
  `spherical_kmeans` / `movmf` / `movmf_auto` Rust functions.
- `covariance_weight` (`method="leiden"` / `"leiden-cpm"`, `feature="full"`) — **covariance-aware
  community detection**. `β > 0` adds a **log-Euclidean** shape term `β·‖logΣ_i − logΣ_j‖²_F` to the
  microcluster affinity graph, so two microclusters must be close in **both** centroid *and*
  covariance to be neighbours — useful when clusters differ by orientation / shape (covariance
  descriptors, motion / time-series windows, anisotropic blobs). `logΣ` is computed with the in-house
  Jacobi eigensolver (new `linalg::matrix_log`) — no new dependency; `β = 0` (default) is the
  existing centroid-only affinity, bit-for-bit unchanged.
- `tangent_weight` / `tangent_rank` (`method="leiden"` / `"leiden-cpm"`, `feature="full"`) —
  **GeoBETULA manifold-aware community detection**. `γ > 0` adds a **Grassmann** term
  `γ · d²_Gr(U_i, U_j)` (projection distance between each microcluster's rank-`tangent_rank` principal
  subspace) to the affinity, so communities must agree in centroid, covariance **and** manifold
  orientation — separating crossing / adjacent manifolds that share a centroid neighbourhood. Reuses
  the in-house Jacobi eigensolver (no new dependency); `γ = 0` (default) leaves the graph unchanged.
- `method="scale-space"` — **scale-space (Morse-persistence) density-mode clustering**. Treats the
  microclusters as a weighted point set and clusters the modes of the KDE
  `ρ_h(x) = Σ_j n_j exp(−‖x−μ_j‖²/2h²)`; it **sweeps the bandwidth `h` and keeps the labelling at the
  most persistent mode count** (the widest plateau of the modes-vs-`log h` curve), so it needs **no
  `k` and no bandwidth** and finds non-convex, arbitrary-count structure. A **prominence**-based mode
  merge (collapse peaks separated by only a shallow density valley) cleans the mode-count curve, so it
  is robust from 2 to ~8+ well-separated clusters and on unequal densities. Pure-Rust mean-shift over
  the `M ≪ N` leaves — cost bounded by the leaf budget, not `N`.
- `method="gmm-toeplitz"` — **AR / Toeplitz-structured GMM for ordered, wide-sense-stationary
  signals** (fixed-length time-series windows, trajectories, sensor / audio / vibration waveforms).
  Each component's covariance is an **AR(w)** process: the pooled **unbiased (covariance-method)**
  autocovariance is mapped by
  **Levinson-Durbin** to the exact **Gohberg-Semencul** precision `Γ = (1/σ²)(BBᵀ − ZZᵀ)`, evaluated by
  the prediction-error decomposition so the `w` boundary positions are modelled exactly — **positive-
  definite by construction** (the reflection-coefficient clamp is the GS box constraint), `O(w)`
  parameters, order `w` chosen by BIC — so it stays well-posed in the
  `N_k ≪ d` regime where full covariance is singular and a diagonal model is blind to neighbour
  correlation. Reuses the CF scatter (no new tree machinery); a scalar stationary mean; BIC auto-`k`
  at `n_clusters=0`; a true posterior via `predict_proba`; parallel EM restarts (`parallel` feature).
  **For ordered coordinates only** — on generic embeddings the Toeplitz prior is
  wrong (use `gmm` / `gmm-full`). Based on the Gohberg-Semencul Toeplitz-precision estimator of
  arXiv:2311.14995; design and validation in [`docs/adr/001-gmm-toeplitz.md`](docs/adr/001-gmm-toeplitz.md).

### Fixed
- **High-dimensional GMM regularization** — the expected-log E-step adds a within-leaf correction
  (`−½ Σ_d (Σ_i)_dd/σ²_kd` for diagonal, `−½ tr(Σ_k⁻¹ Σ_i)` for full covariance) that turns
  *over-confident* when a component's own covariance goes near-singular along a low-variance
  direction — which is the norm in high dimensions with few effective microclusters per component.
  Two floors now keep the component covariances well-conditioned:
  - `method="gmm"` (diagonal): per-dimension variance floor raised from `1e-6·gvar_d` to
    `1e-3·gvar_d`. `digits` (64-D) ARI 0.372 → 0.396, now ahead of scikit-learn's
    `GaussianMixture(covariance_type="diag")` (0.324).
  - `method="gmm-full"` (full covariance): added a per-dimension floor on each component's covariance
    **diagonal** at `1e-3·gcov_dd` (off-diagonals — orientation — untouched). Previously a component
    could be starved to zero responsibility and the recovered count dropped below `k`; on `digits`
    the fit collapsed to 9 clusters at ARI 0.391, and now holds all 10 at ARI 0.511 — ahead of
    scikit-learn's `GaussianMixture(covariance_type="full")` (0.402).

  The floors are relative to the **per-dimension** global variance (not the global mean scale, which
  is inflated by between-cluster separation and would over-regularize tight clusters), so
  low-dimensional and anisotropic fits are unchanged (well-separated blobs still ARI 1.00; the
  rotated-anisotropic 2-D case still ties `GaussianMixture` at 0.887). No API change.

## [0.1.5] — 2026-07-04

### Added
- `method="leiden"` / `method="leiden-cpm"` — **graph clustering / community detection** over the
  microcluster affinity graph via the full **Leiden** algorithm (Traag, Waltman & van Eck 2019):
  local moving → refinement (sub-communities grown from singletons *along edges*, so each is
  connected by construction — Leiden's guarantee over Louvain) → aggregation seeded from the
  pre-refinement partition. It **discovers the community count** — no `k` (like the density head).
  A `resolution` (`γ`) knob trades community count against size; the **modularity** objective
  (`"leiden"`, γ = 1 default) has a resolution limit, the **CPM** objective (`"leiden-cpm"`) is
  resolution-limit-free (γ on a smaller, density scale). Pure Rust — no eigensolver, NumPy-only.
  Best for community/blob structure at a moderate `threshold`; use `method="spectral"` for elongated
  manifolds. The self-tuning k-NN affinity graph is shared between the spectral and Leiden heads.
- `betula_cluster.consensus(X, n_clusters, n_runs=…)` — clusters `X` under several random
  insertion-order permutations and votes, turning the CF-tree's **insertion-order sensitivity**
  (Known Limitation #1) into a measurable quantity: a consensus labelling plus a **per-point
  stability score** in `[0, 1]` (`ConsensusResult.confidence` — low on unstable boundaries, high
  where every order agrees). NumPy-only; for the partitional heads at a fixed `n_clusters`. The
  independent runs parallelize across threads with `n_jobs` (the Rust core releases the GIL).
- `method="spectral"` — spectral clustering over the CF-tree leaf microclusters for **non-convex /
  manifold** clusters (moons, rings, spirals) that the centroid heads cannot separate. Self-tuning
  symmetric k-NN affinity (Zelnik-Manor & Perona local scaling), the normalized Laplacian embedding
  (Ng-Jordan-Weiss) via the in-house Jacobi eigensolver — no LAPACK/ARPACK, the crate stays
  NumPy-only — with a k-means landmark reduction above 256 microclusters so the `O(m³)` solve stays
  bounded. Dense input only; pair it with a small `threshold` (many leaves) so the microclusters
  resolve the manifold. No built-in cluster-count selection: `n_clusters=0` defaults to 2.
- `threshold="auto"` for the `Betula` estimator — removes the one hyperparameter users most often
  have to guess. A bounded-subsample pilot fits a `threshold=0` tree at the same `max_leaves` and
  reads the threshold it converges to, warm-starting the full fit near-converged instead of growing
  it from zero (fewer rebuild passes, lower peak leaf count on large `n`). Cached across refits /
  streaming batches; below the pilot cap it is a no-op (growing from zero is already cheap), and it
  is dense-only (raises on sparse input).

### Changed
- Benchmarks now cover every head (spectral, Leiden added to `bench/comprehensive.py`) and the
  compression heads run at `max_leaves = 4000`: betula-kmeans is at *exact* parity with scikit-learn
  (blobs 0.861 = 0.861) and Ward beats raw Ward while running the full `N`. Docs / README / docs site
  surface the spectral, Leiden and consensus additions; test counts reconciled (190 Python, 158
  Rust). The docs site now renders the CHANGELOG and redeploys on every published release.

## [0.1.4] — 2026-07-04

### Added
- `MapperGraph.persistence_diagram` / `MapperGraph.persistence(filtration=…)` — 0-D persistent homology
  of the Mapper nerve by single-linkage union-find (elder rule, `O(E log E)`, pure Rust). Two
  filtrations: `"overlap"` (the `1 − edge_overlap` Bhattacharyya gap — a finite bar's death is the depth
  of a bottleneck, ranking the boolean `bridges`) and `"lens"` (the lens sublevel diagram). Essential
  connected-component classes carry `inf` death.
- Greedy weighted k-means++ init (scikit-learn's default): lower-inertia, lower-variance seeds at
  ~`ln k`× the negligible init cost over the leaves.
- `objective="dbcv"` for `tune` — Density-Based Clustering Validation (Moulavi et al. 2014, in
  `[-1, 1]`). Unlike the convex Calinski-Harabasz / Davies-Bouldin metrics (which *penalise* correct
  non-convex partitions), DBCV validates variable-density / non-convex clusters, so it is the right
  selection metric for the HDBSCAN-CF and DbStream density heads. NumPy-only, computed over a
  subsample.

### Changed
- `fit_predict_sparse` / the `_core` CSR entry points now cap `n_features` (`MAX_SPARSE_FEATURES`) and
  validate CSR arrays through the pure-Rust `sparse::validate_csr`, closing an unbounded-allocation DoS
  where a hostile caller could force an ~8 EB allocation with a single-nonzero row.
- Docs reconciled to the current suite: **172**-case Python suite, **147** Rust tests (143 unit + 4
  integration under default features; the `python` / `persistence` / `cli` surfaces add more, 155 total).

### Tests
- Mutation-testing infrastructure (`cargo-mutants` scoped to the CF math core, `mutmut` for the Python
  wrapper, a weekly non-blocking workflow) plus a CSR-fuzzing proptest and the two coverage gaps it
  surfaced (the CF-tree absorption boundary, exact tune-metric values).

## [0.1.3] — 2026-07-04

### Added
- `betula_cluster.tune` — memory-aware hyperparameter search over the CF knobs, scored by an internal
  metric (Calinski-Harabasz / Davies-Bouldin) or ARI, with a multi-objective **quality / memory /
  speed** Pareto mode. NumPy-only by default; an optional Optuna backend (TPE / NSGA-II) via
  `pip install 'betula-cluster[tune]'`.
- Property-based tests (`proptest`, dev-only) for the CF-tree invariants: the clustering feature is a
  commutative monoid (`merge` is associative/commutative and equals a sequential build), folding a
  tree's leaf features reconstructs the whole-dataset feature, the full-covariance upper-triangular
  index is a bijection (incl. `dim ≥ 4`), and the Frequent-Directions sketch is lossless on low-rank
  data and never overshoots the exact scatter.
- Sparse-text benchmark (20 newsgroups, TF-IDF): the `O(nnz)` `fit_predict_sparse` path and the
  standard reduce-then-cluster pipeline (TruncatedSVD / NMF → k-means) vs scikit-learn, written up
  honestly in `bench/RESULTS.md` (raw high-`d` TF-IDF concentrates for every fast clusterer; on NMF
  topics betula matches sklearn).
- `MapperGraph.edge_overlap` — a Bhattacharyya coefficient in `(0, 1]` per Mapper edge, from the pooled
  diagonal-Gaussian summaries of the two nodes' member microclusters. Surfaced on `to_networkx()` edges
  as `overlap=…`, so a bridge between well-separated regions reads as a lower-weight edge than one
  inside a dense blob.
- Documentation site (MkDocs Material + `mkdocstrings` API autodoc, MathJax-rendered math) built from
  `docs/`, with a GitHub Pages deploy workflow; `pip install 'betula-cluster[docs]'` for the toolchain.

### Changed
- Coverage floor (`cargo llvm-cov`, ≥95 % lines) now also measures the `persistence` and `cli` feature
  sets, not just the default core.
- Declared `rust-version = "1.82"` (MSRV) and lowered the real floor to it — the streaming heads had an
  implicit 1.87 dependency (`u64::is_multiple_of`), now rewritten. Added `Documentation` / `Changelog`
  project URLs.
- Docs reconciled to the current suite: **167**-case Python suite, **141** Rust tests (137 unit + 4
  integration), and **five** end-to-end use cases (README, DESIGN.md).
- Repository hardening: `macOS` / `Windows` CI test legs, an sdist install smoke test, a nightly
  `cargo audit` cron, Dependabot, and `SECURITY.md` / `CONTRIBUTING.md` / issue templates.

## [0.1.2] — 2026-06-28

### Added
- `betula_cluster.__version__`, resolved from the installed package metadata.

### Changed
- README repositioned: compress-then-cluster framing, the test/coverage story surfaced at the top, a
  "When to use it" section, and a **stable-core / experimental** capability split. HDBSCAN is labelled
  **HDBSCAN-CF** consistently in prose (the `method="hdbscan"` API string is unchanged).

### Fixed
- Stale docs: the Python suite is **153** cases (was written as 123); `betula-index` references now
  point to `lexindex` (the indexing companion's published name).

## [0.1.1] — 2026-06-28

### Fixed
- PyPI project description: README links to the docs, benchmarks, and examples are now absolute GitHub
  URLs so they resolve on the PyPI page (relative links only worked in the GitHub-rendered README).

## [0.1.0] — 2026-06-28

First public release.

### Added
- Numerically stable BETULA clustering features `(n, μ, S)` (Welford/Chan updates) with four
  covariance models: spherical, diagonal, full (PSD via Cholesky), and a Frequent-Directions sketch
  (`O(ℓ·d)` per leaf) for very high-dimensional data.
- Memory-bounded CF-tree (Phase 1) with auto-rebuild under a `max_leaves` cap; optional parallel
  shard+merge build (`n_jobs`); EWMA `decay` for streaming concept drift.
- Global clustering heads: Hamerly-accelerated exact k-means, diagonal & full-covariance GMM-EM
  (expected-log E-step + NIW/MAP), Ward-HAC (nearest-neighbour chain), and HDBSCAN-on-CF; automatic
  cluster count at `n_clusters=0` (BIC / X-means / dendrogram cut).
- χ² / Mahalanobis mass-invariant absorption gate (`absorb="chi2"`).
- `normalize=True` for cosine/direction clustering of embeddings (L2-normalized rows on the unit
  sphere; squared-Euclidean is monotone in cosine). Doubles as the **high-dimensional fix**: at d≫100
  raw Euclidean distances concentrate and the CF-tree collapses, but direction stays discriminative —
  on MNIST-784 it lifts ARI 0.04 → 0.44, beating scikit-learn (benchmarked in
  `bench/results_real_normalize.csv`). Off by default (magnitude is signal on tabular data).
- Inline auto-vectorized distance kernels (the compiler vectorizes the tight reductions per call
  site; `target-cpu=native` opts into AVX2 / AVX-512 — see `.cargo/config.toml`); rayon-parallel
  labeling.
- Python bindings: abi3 wheel (CPython 3.11+), zero-copy NumPy, `float32`/`float64` (no upcast), GIL
  released during compute; one-shot `fit_predict` and a scikit-learn-style streaming `Betula`
  estimator (`partial_fit` / `fit` / `predict` / `fit_predict`).
- Full scikit-learn parameter protocol (`get_params` / `set_params`) — works with `clone`,
  `Pipeline`, and `GridSearchCV`. PEP 561 typed (`py.typed` + stubs).
- Dataset-structure inspection: `microcluster_centers_`/`_weights_`/`_radii_`,
  `cluster_centers_`/`_radii_`/`_sizes_`, `outlier_scores`, `find_outliers`, `find_near_duplicates`,
  `near_duplicate_pairs` (scored cosine pairs, exact within each leaf-block — the scalable
  counterpart to an O(N²) all-pairs scan), `sample_representatives`, `assign_microclusters`,
  `summary`, and `n_rebuilds_` / `threshold_` diagnostics.
- **Mapper topological skeleton** (`topology::mapper` → `Betula.mapper()` → `MapperGraph`): a lens
  (`density` / `radius` / `l2norm` / `coordinate` / `eccentricity`) over the microclusters, an
  overlapping cover, per-bin single-linkage at a data-adaptive (median-NN) scale, and a nerve graph with branch
  points and bridges (Tarjan); optional `to_networkx()`. Exploration of structure / RAG leakage /
  dedup, not a partition. `mapper_stability()` sweeps the resolution and reports the topology's
  persistence across scale (β₀ components, β₁ loops, branch points, bridges per resolution).
- **Soft assignment & confidence**: `predict_proba` (true posterior for the GMM heads via the
  per-leaf responsibility matrix `microcluster_proba_`; a documented centroid-distance softmax
  *heuristic* for k-means / Ward / HDBSCAN) and `assignment_confidence`.
- **Coreset / diagnostics**: `export_coreset()` → `Coreset` (leaves as weighted points — a streaming
  coreset), `diagnostics()` (compression ratio, radius percentiles, cluster mass spread),
  `representatives(method=medoid|boundary|outlier|diverse)`, and `cluster_profile()` (JSON-able
  geometry for LLM cluster naming).
- **`memory_budget_mb`**: size `max_leaves` from a target tree-resident memory (MiB) at fit time
  instead of tuning it by hand; the resolved value is exposed as `effective_max_leaves_`.
- **Drift monitoring & curation**: `snapshot()` + `Betula.compare_snapshots(before, after)`
  (nearest-centroid match → centroid shifts / mass ratios) and `active_learning_batch(strategy=
  "uncertain"|"outlier")` (rows to review/label).
- **`DenStream`** streaming density clusterer (Cao et al., SDM 2006) over fading spherical
  micro-clusters built on the stable CFs (decay is centroid/radius-invariant); `partial_fit` /
  `cluster` / `fit` / `fit_predict` / `predict` (`-1` = noise) + microcluster getters, sklearn-style.
- **`DbStream`** streaming DBSTREAM clusterer (Hahsler & Bolaños, 2016): fading micro-clusters
  connected by **shared density** (faded overlap mass) rather than distance, so it recovers
  arbitrarily-shaped clusters and keeps close-but-disconnected dense regions apart. Fixed-radius
  multi-assignment online; offline connects a pair when their overlap mass is `≥ alpha·min_weight`.
  Same fading-CF core and sklearn-style API as `DenStream`; `core::stream::DbStream` in Rust.
- **Streaming quantile sketches** (`betula-sketch`, in `src/sketch/`): `KllSketch` (Karnin–Lang–
  Liberty, rank-error) and `DdSketch` (Masson et al., relative-error) — `update` / `update_many` /
  `merge` / `quantile` / `quantiles`, mergeable, bounded memory.
- **Sparse input**: `fit` / `fit_predict` / `partial_fit` / `predict` accept a `scipy.sparse` matrix
  (CSR-routed, rows expanded one at a time — the dense `N × d` matrix is never materialized). f64;
  this dense-tree path keeps the cancellation-free guarantee, compute `O(N·d)`.
- **`O(nnz)` sparse-native** (`fit_predict_sparse`): one-shot clustering of a `scipy.sparse` matrix
  that touches only the non-zeros. Rows summarize into spherical micro-clusters keeping
  `(n, ΣX, ‖ΣX‖², S)` (so the mean, cached `‖μ‖²`, and centroid distance are `O(nnz)`) via a flat
  leader pass bounded by `max_leaves`, then a parametric head (`kmeans` default — robust for
  high-`d` sparse) labels each row. Uses the *expanded* squared-distance form, so unlike the dense
  path it is not cancellation-free (accurate for sparse rows far from the dense centroid);
  `core::sparse::{summarize_sparse, nearest_sparse}` is the Rust API.
- **Robust insertion** (`huber_k`): optional Huber/winsorized point updates on the streaming
  estimator — each point is clamped to within `huber_k` per-dimension standard deviations of its
  target microcluster before the Welford fold-in, bounding any single point's pull on the centroid
  (`O(k·σ/n)`) so stream outliers cannot stretch a centroid or inflate a radius. Off by default;
  zero-variance dimensions pass through and a 5-point warm-up gates the clip. The result is still a
  valid `(n, μ, S)` triple, so every downstream head is unchanged.
- **Constrained clustering** (`must_link` / `cannot_link`): semi-supervised COP-KMeans (Wagstaff et
  al., 2001) over the leaf microclusters — `fit(X, must_link=..., cannot_link=...)` /
  `fit_predict(...)` take `(m, 2)` row-index pairs. Must-link is transitively closed; cannot-link is
  enforced per assignment. Constraints are honoured at the microcluster granularity, so a cannot-link
  inside one leaf (or contradictory / over-constrained inputs) raises `ValueError` rather than being
  silently dropped. `method="kmeans"`, dense input; `core::clustering::cop_kmeans` exposes the Rust
  API with a typed `ConstraintError`.
- **Mixed numeric + categorical clustering** (`KPrototypes`): k-prototypes (Huang, 1997) for mixed
  data. A *mixed CF* (`MixedCf`) pairs the stable numeric `(n, μ, S)` with a per-attribute category
  histogram (mode = categorical centroid); distance is `‖Δnumeric‖² + γ·(categorical mismatch)`, with
  `γ` defaulting to Huang's heuristic. Rows are leader-summarized into bounded mixed micro-clusters,
  then clustered. Standalone scikit-learn-style estimator (`categorical` column indices,
  `fit`/`fit_predict`/`predict`, `cluster_centroids_`/`cluster_modes_`); `core::clustering::{MixedCf,
  kprototypes, summarize_mixed}` is the Rust API.
- **Command-line interface** (`betula`, behind the `cli` feature): a dependency-free binary that
  clusters a delimited numeric file or stdin and writes one label per row; flags mirror the library
  (`--clusters` / `--method` / `--feature` / `--threshold` / … ; `--clusters 0` auto-selects `k`).
- `save` / `load` + pickle (`joblib`-compatible) persistence (serde + CBOR via ciborium,
  schema-versioned).
- NaN/Inf input validation at the boundary.

### Fixed
- `estimate_threshold` now measures the mean nearest-sibling distance **within each leaf node**
  (ELKI/BETULA-standard, `O(M·capacity)`) instead of a global all-pairs scan; the rebuild threshold
  rises monotonically (no multiplicative bump that compounded across rebuilds and collapsed the tree
  far below `max_leaves`), and rebuilds reinsert in reverse-DFS leaf order. The CF-tree build is now
  byte-for-byte the reference (`betulars`) tree shape and at speed parity with matched build flags.

[Unreleased]: https://github.com/ilgrad/betula-cluster/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ilgrad/betula-cluster/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.5
[0.1.4]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.4
[0.1.3]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.3
[0.1.2]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.2
[0.1.1]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.1
[0.1.0]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.0
