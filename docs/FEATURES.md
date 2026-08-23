# Features — full reference

A capability-by-capability reference. For runnable code see [`USAGE.md`](USAGE.md) and the
[example notebooks](https://github.com/ilgrad/betula-cluster/blob/main/examples/README.md); for the math behind these, see [`MATH.md`](MATH.md).

- CF-tree (BIRCH/BETULA Phase 1) with auto-rebuild and covariance models — spherical, diagonal,
  full (PSD-by-construction via Cholesky), and a **Frequent-Directions sketch** for very
  high-dimensional data ($O(\ell d)$ memory per leaf instead of $O(d^2)$; trades speed for memory, for
  `d` so large the full covariance does not fit).
- auto-vectorized distance kernels (tight inline reductions the compiler vectorizes — measured
  faster than runtime SIMD dispatch on the small-`d` hot path); rayon-parallel point
  labeling and rebuild-threshold estimation (deterministic — bit-identical labels to the serial
  path; `parallel` feature, on by default, `--no-default-features` for a serial build).
- Global clustering heads: weighted **k-means** (k-means++ + exact Lloyd), **diagonal &
  full-covariance GMM-EM** (expected-log E-step + NIW/MAP regularization + a per-dimension covariance
  floor that keeps components well-conditioned in high dimensions — no starved-component collapse,
  full covariance captures rotated/correlated clusters, **BIC auto-selects the component count** when
  `n_clusters=0`), an **AR / Toeplitz-structured GMM** (`method="gmm-toeplitz"`) for **ordered,
  wide-sense-stationary signals** — fixed-length time-series windows, trajectories, sensor / audio
  waveforms — where each component covariance is an AR(w) process (Levinson-Durbin → a banded
  positive-definite precision `Γ = AᵀA/σ²`, `O(w)` parameters, order `w` by BIC), well-posed at
  `N_k ≪ d` where full covariance is singular and a diagonal model is blind to neighbour correlation
  (ordered coordinates only — not generic embeddings; based on the Gohberg-Semencul estimator of
  arXiv:2311.14995, see [`docs/adr/001-gmm-toeplitz.md`](adr/001-gmm-toeplitz.md)) plus a general
  (non-AR) **`gmm-toeplitz-full`** head — a dense positive-definite Toeplitz covariance from the biased
  autocovariance — for signals whose autocovariance a low-order AR cannot represent (e.g. a long-lag
  echo: it recovers such a mixture where the banded AR head sits at chance), and a **`gmm-toeplitz-gs`**
  head — the full-order **Gohberg-Semencul MLE** precision (Yule-Walker warm start + exact-likelihood
  coordinate ascent, PD by `|k| < 1`), the likelihood-optimal general precision with a cheaper `O(m·d·p)`
  E-step than the dense route,
  **Ward agglomerative HAC** (exact, via nearest-neighbour chain; dendrogram-cut auto-k),
  **spectral clustering** (self-tuning k-NN affinity + normalized Laplacian embedding via the
  in-house Jacobi eigensolver, k-means-landmark-reduced above 256 microclusters — separates
  non-convex / manifold clusters the centroid heads cannot; pair it with a small `threshold` so the
  microclusters resolve the manifold),
  **Leiden community detection** (graph clustering, Traag et al. 2019) over the microcluster affinity
  graph — local moving → refinement (each community connected by construction) → seeded aggregation;
  **discovers the community count**, no `k` needed; a `resolution` γ knob with **modularity**
  (`"leiden"`) or resolution-limit-free **CPM** (`"leiden-cpm"`) objectives; pure Rust — pair it with
  a moderate `threshold`, a very fine graph over-splits per modularity's resolution limit;
  `covariance_weight > 0` makes the affinity **covariance-aware** via a log-Euclidean shape term
  (`feature="full"`), so communities agree in both centroid and covariance; `tangent_weight > 0` adds
  a **Grassmann** tangent-subspace term (GeoBETULA) for manifold-aware communities that separate
  crossing / adjacent structures),
  **directional clustering on the unit hypersphere** — hard **spherical k-means**
  (`"spherical-kmeans"`) and a soft **mixture of von Mises–Fisher** distributions (`"vmf"`, EM with a
  true posterior and BIC auto-`k`) for L2-normalized embeddings (CLIP / face / sentence / speaker),
  where cosine — not Euclidean — geometry matters; each leaf keeps its weighted mean so the resultant
  `R_c = Σ n_i μ_i` is exactly mergeable (BETULA on the sphere) and the concentration `κ`
  (Banerjee 2005) is estimated without a Bessel library, with input auto-L2-normalized, and
  **HDBSCAN-style density clustering over the CF microclusters** (mass-aware mutual-reachability +
  mass-weighted stability → non-convex clusters and noise, automatic count; an *approximation* of
  raw-point HDBSCAN over the $M \ll N$ microclusters, not identical to it), and
  **scale-space (Morse-persistence) density-mode clustering** (`method="scale-space"` — mean-shift over
  the microcluster KDE, with the bandwidth *and* the cluster count chosen by **mode persistence** across
  scale, so no `k` or bandwidth is required; non-convex, arbitrary count).
- **Soft assignment & confidence**: `predict_proba` (the point's own posterior under the fitted
  mixture for the GMM, **vMF** and Toeplitz heads, so `predict_proba(X).argmax(1) == predict(X)`; a
  documented centroid-distance softmax *heuristic* for k-means / Ward / spectral / Leiden / HDBSCAN),
  `assignment_confidence`,
  `microcluster_proba_` (per-microcluster GMM responsibilities, GMM heads only), `export_coreset` (the
  leaves as weighted points), `diagnostics`, `representatives`, `cluster_profile`.
- **`DenStream`** — a separate streaming density clusterer (Cao et al., SDM 2006) over *fading*
  micro-clusters, for evolving streams where old data should decay out: `partial_fit` chunks, then
  `predict` (`-1` = noise). Reuses the same numerically stable CFs (decay is exact and leaves the
  centroid/radius untouched, only the weight).
- **`DbStream`** — a streaming **DBSTREAM** clusterer (Hahsler & Bolaños, 2016) that connects fading
  micro-clusters by **shared density** (the mass of points within radius `r` of *both*), not mere
  proximity: it recovers arbitrarily-shaped clusters as chains of overlapping micro-clusters and —
  unlike a distance-only rule — keeps two close-but-disconnected dense regions apart (an empty gap
  carries zero shared density). Same fading-CF core as `DenStream`; `partial_fit` / `predict`.
- **Streaming quantile sketches** (`KllSketch`, `DdSketch`) — compact, mergeable summaries that
  answer quantile / rank queries over a stream in bounded memory: **KLL** with a rank-error guarantee
  (uniform across the distribution) and **DDSketch** with a relative-error guarantee (ideal for
  skewed / positive / long-tailed data such as latencies).
- **Sparse input** — `fit` / `fit_predict` / `partial_fit` / `predict` accept a `scipy.sparse`
  matrix directly; rows are expanded one at a time, so the dense `N × d` matrix is **never
  materialized** (cluster a million-row sparse matrix that would never fit dense). This dense-tree
  path keeps the cancellation-free guarantee; compute scales with the feature count (the CF centroid
  is dense, as in every CF-tree method — sklearn-Birch included).
- **$O(\mathrm{nnz})$ sparse-native** (`fit_predict_sparse`) — for very high-dimensional sparse data, a
  one-shot path that touches only the non-zeros: rows summarize into spherical micro-clusters keeping
  $(n, \Sigma X, \|\Sigma X\|^2, S)$ so updates and centroid distances are $O(\mathrm{nnz})$, then a parametric head
  (`kmeans` default) clusters them. It uses the *expanded* squared-distance form for speed and so does
  **not** carry the dense path's cancellation-free guarantee — accurate for sparse rows far from the
  dense centroid; use the dense `Betula` path when you need cancellation-free scatter.
- **CF-weighted NMF reduction** (`projection="weighted-nmf"`, `projection_dim`) — for **nonnegative**
  data (TF-IDF / bag-of-words / event counts / spectrogram magnitudes / histograms), a nonnegative
  low-rank projection applied over the $M \ll N$ leaf **centroids**, not the raw $N \times d$ matrix: by
  König-Huygens the weighted-centroid NMF equals the full-data NMF up to the within-leaf scatter
  constant, so it runs NMF at BETULA scale and bounded memory (something point-level NMF cannot), then
  any head clusters the nonnegative codes. Dependency-free weighted **HALS** (no BLAS — the matrices are
  small because $M \ll N$); `projection="weighted-nmf-kl"` switches to the **KL-divergence** variant
  (Lee-Seung multiplicative updates) — the Poisson maximum-likelihood objective for **count** data. The
  advantage is largest where counts are **sparse** (measured up to **+0.5 ARI** over Frobenius on
  Poisson counts at mean rate < 0.5), converging to Frobenius as counts grow and Poisson → Gaussian.
  Both solvers start from a deterministic **NNDSVDar** initialization (a randomized range finder, so no
  LAPACK; rank-deficient triplets cut at the numerical-rank threshold rather than amplified) and return a
  **canonical** factorization — component rows unit-L2, ordered by descending
  energy. That last part is load-bearing, not cosmetic: NMF is invariant to `(W D, D⁻¹H)`, so an
  unpinned split lets one component's arbitrary scale dominate the Euclidean geometry the head then
  clusters (measured over 8 seeds at N = 8k/40k/160k: median ARI 0.81/0.99/0.97 → 1.00, seed spread
  ±0.37 → ±0.00 — the gain is determinism, not accuracy in the mean). `components_`
  and `reconstruction_err_` expose the parts and the fit; `projection_max_iter` is the solver's own
  budget, independent of the head's `max_iter`. Dense **and** sparse CSR input; signed input is
  rejected, not shifted. See [`MATH.md`](MATH.md).
- **Robust insertion** (`huber_k`) — optional Huber/winsorized point updates: each incoming point is
  clamped to within `huber_k` per-dimension standard deviations of its target microcluster *before*
  it is folded in, so a single extreme value cannot stretch a centroid or inflate a radius. Off by
  default; most valuable for streaming, where you cannot go back and re-fit on cleaned data. See the
  formula in [`MATH.md`](MATH.md).
- **Constrained clustering** (`must_link` / `cannot_link`) — semi-supervised **COP-KMeans** (Wagstaff
  et al., 2001): pass pairwise row-index constraints to `fit` / `fit_predict` and points that *must*
  share a cluster are kept together and points that *cannot* are kept apart. Constraints are honoured
  at the microcluster granularity (a cannot-link between two points the tree compressed into one leaf
  is reported as infeasible — lower `threshold` to separate them); contradictory or over-constrained
  inputs raise rather than silently violate. `method="kmeans"` only, dense input.
- **Mixed numeric + categorical** (`KPrototypes`) — **k-prototypes** (Huang, 1997) for data that is
  part numeric, part categorical. Each cluster is a *mixed CF*: the stable numeric $(n, \mu, S)$ plus a
  category-count histogram per categorical attribute (its mode is the categorical centroid). Distance
  is $\|\Delta_\text{numeric}\|^2 + \gamma \cdot (\text{categorical mismatch})$, with $\gamma$ auto-set to Huang's heuristic. Rows are
  leader-summarized into bounded mixed micro-clusters first, so it scales like the rest of the library.
- Python bindings: abi3 wheel, zero-copy numpy (one-shot `fit_predict` takes **float32 or
  float64** — `f32` data is clustered in `f32`, halving memory on embeddings), GIL released during
  compute, plus a scikit-learn-style `Betula` estimator with `partial_fit` (float32 or float64 — an
  `f32` tree halves resident memory) for streaming / out-of-core data at bounded memory, and
  `save` / `load` + pickle (`joblib`-compatible) persistence of a fitted model. The estimator
  implements the full scikit-learn parameter protocol (`get_params` / `set_params`), so it drops
  into `clone`, `Pipeline`, and `GridSearchCV`; the wheel is typed (PEP 561 `py.typed` + stubs).
  Inputs are validated at the boundary — a `NaN` / `Inf` raises instead of silently corrupting the
  tree.
- Dataset-structure inspection (not just labels) — the estimator exposes its microcluster and
  cluster geometry (`microcluster_centers_` / `_weights_` / `_radii_`, `cluster_centers_` /
  `_radii_` / `_sizes_`) and, on top of it, `summary()`, `outlier_scores(X)` (distance to the
  assigned centroid ÷ cluster radius), `find_outliers`, `find_near_duplicates` (unscored groups),
  `near_duplicate_pairs(X, threshold)` (scored cosine pairs, exact within each leaf-block — the
  scalable counterpart to an $O(N^2)$ all-pairs scan), `sample_representatives`, and
  `assign_microclusters` — for embedding dataset cleaning, deduplication, and outlier discovery,
  reusing the CF-tree already built (no extra passes).
- **Mapper topological skeleton** (`mapper()` → `MapperGraph`) — TDA Mapper specialised to the
  microclusters: a lens (`density` / `radius` / `l2norm` / `coordinate` / `eccentricity`) is covered
  by overlapping bins, microclusters in each bin are single-linked at a data-adaptive scale, and the
  nerve graph exposes **branch points** and **bridges** (thin links between otherwise separate
  regions — topic leakage / merges in embeddings). Each edge also carries a **CF-aware Bhattacharyya
  overlap** (`edge_overlap ∈ (0, 1]`) from the two nodes' pooled diagonal-Gaussian summaries, so a
  bridge across a sparse neck scores lower than an edge inside one dense blob — distributional, not a
  bare shared-microcluster count. Runs over the $M \ll N$ microclusters, with an optional
  `to_networkx()` (edges carry `weight` / `overlap` / `bridge`) for plotting; `mapper_stability()`
  sweeps the resolution to find the topologically stable scale. An exploration tool (structure, RAG
  curation, dedup), not a partition — complementary to the HDBSCAN density head.
- **Memory-aware hyperparameter tuning** (`tune` → `TuneResult`) — searches betula's CF-representation
  knobs (`max_leaves`, covariance model, `normalize`) for the best clustering into `n_clusters`,
  scored by an internal metric (Calinski-Harabasz / Davies-Bouldin) or ARI against ground-truth
  labels, with an optional **quality / memory (`n_leaves`) / speed (fit-time)** Pareto mode. NumPy-only
  by default (random search); an optional Optuna backend (TPE / NSGA-II) via
  `pip install 'betula-cluster[tune]'`. Because betula fits are cheap, a few hundred trials run in
  seconds — the search is over the compression, so cost is bounded by the microcluster count, not `N`.
- **Consensus & stability** (`consensus` → `ConsensusResult`) — clusters `X` under several random
  insertion-order permutations and votes, converting the CF-tree's insertion-order sensitivity into a
  measurable signal: a consensus labelling plus a **per-point stability score** in `[0, 1]` (low on
  unstable boundaries, high where every order agrees). NumPy-only; partitional heads at a fixed
  `n_clusters`.

## Architecture (crate layout)

| module | role |
|--------|------|
| `types` | `Real` numeric trait (`f32` / `f64`) |
| `linalg` | Cholesky / triangular solve / logdet / Mahalanobis / Jacobi eigensolver (no LAPACK) |
| `stats` | χ² quantile (inverse regularized incomplete gamma) for Mahalanobis gates |
| `feature` | clustering features: `Spherical` / `Diagonal` / `Full` / `FdSketch` (high-d) |
| `kernels` | auto-vectorized distance kernels (inline reductions) |
| `distance` | D0–D4, radius, Mahalanobis (stable forms) |
| `tree` | arena CF-tree + budget-targeting auto-rebuild |
| `clustering` | `kmeans` / `cop_kmeans`, `gmm_diagonal`, `gmm_full`, `gmm_toeplitz{,_full,_gs}`, `ward_hac`, `spectral`, `leiden`, `spherical_kmeans`, `movmf`, `scale_space`, `hdbscan`, `kprototypes`, `nmf` (the `projection` reducer) |
| `mixture` | fitted-mixture kernels (diagonal / full-Cholesky / stationary / vMF) that score a raw point — what `predict` / `predict_proba` label by |
| `stream` | `DenStream` + `DbStream` fading-microcluster density heads |
| `sparse` | `O(nnz)` sparse-native summarisation (`fit_predict_sparse`) |
| `sketch` | KLL + DDSketch mergeable quantile sketches |
| `topology` | Mapper nerve + 0-D persistence |
| `model` | end-to-end `Model::fit` / `predict`; the `Method` enum and the per-head assignment rule |
| `python` | PyO3 bindings: one-shot `fit_predict` + streaming `Betula` estimator |

See [`DESIGN.md`](https://github.com/ilgrad/betula-cluster/blob/main/DESIGN.md) for the full design and the verified mathematical foundation.
