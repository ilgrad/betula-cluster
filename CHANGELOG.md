# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

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
  `sample_representatives`, `assign_microclusters`, `summary`, and `n_rebuilds_` / `threshold_`
  diagnostics.
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

[0.1.0]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.0
