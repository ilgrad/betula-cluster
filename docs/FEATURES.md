# Features — full reference

A capability-by-capability reference. For runnable code see [`USAGE.md`](USAGE.md) and the
[example notebooks](../examples/README.md); for the math behind these, see [`MATH.md`](MATH.md).

- CF-tree (BIRCH/BETULA Phase 1) with auto-rebuild and covariance models — spherical, diagonal,
  full (PSD-by-construction via Cholesky), and a **Frequent-Directions sketch** for very
  high-dimensional data (`O(ℓ·d)` memory per leaf instead of `O(d²)`; trades speed for memory, for
  `d` so large the full covariance does not fit).
- auto-vectorized distance kernels (tight inline reductions the compiler vectorizes — measured
  faster than runtime SIMD dispatch on the small-`d` hot path); rayon-parallel point
  labeling and rebuild-threshold estimation (deterministic — bit-identical labels to the serial
  path; `parallel` feature, on by default, `--no-default-features` for a serial build).
- Global clustering heads: weighted **k-means** (k-means++ + exact Lloyd), **diagonal &
  full-covariance GMM-EM** (expected-log E-step + NIW/MAP regularization, full covariance captures
  rotated/correlated clusters, **BIC auto-selects the component count** when `n_clusters=0`),
  **Ward agglomerative HAC** (exact, via nearest-neighbour chain; dendrogram-cut auto-k), and
  **HDBSCAN-style density clustering over the CF microclusters** (mass-aware mutual-reachability +
  mass-weighted stability → non-convex clusters and noise, automatic count; an *approximation* of
  raw-point HDBSCAN over the `M ≪ N` microclusters, not identical to it).
- **Soft assignment & confidence**: `predict_proba` (true posterior for the GMM heads; a documented
  centroid-distance softmax *heuristic* for k-means / Ward / HDBSCAN), `assignment_confidence`,
  `export_coreset` (the leaves as weighted points), `diagnostics`, `representatives`, `cluster_profile`.
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
- **`O(nnz)` sparse-native** (`fit_predict_sparse`) — for very high-dimensional sparse data, a
  one-shot path that touches only the non-zeros: rows summarize into spherical micro-clusters keeping
  `(n, ΣX, ‖ΣX‖², S)` so updates and centroid distances are `O(nnz)`, then a parametric head
  (`kmeans` default) clusters them. It uses the *expanded* squared-distance form for speed and so does
  **not** carry the dense path's cancellation-free guarantee — accurate for sparse rows far from the
  dense centroid; use the dense `Betula` path when you need cancellation-free scatter.
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
  part numeric, part categorical. Each cluster is a *mixed CF*: the stable numeric `(n, μ, S)` plus a
  category-count histogram per categorical attribute (its mode is the categorical centroid). Distance
  is `‖Δnumeric‖² + γ·(categorical mismatch)`, with `γ` auto-set to Huang's heuristic. Rows are
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
  assigned centroid ÷ cluster radius), `find_outliers`, `find_near_duplicates`,
  `sample_representatives`, and `assign_microclusters` — for embedding dataset cleaning,
  deduplication, and outlier discovery, reusing the CF-tree already built (no extra passes).
- **Mapper topological skeleton** (`mapper()` → `MapperGraph`) — TDA Mapper specialised to the
  microclusters: a lens (`density` / `radius` / `l2norm` / `coordinate` / `eccentricity`) is covered
  by overlapping bins, microclusters in each bin are single-linked at a data-adaptive scale, and the
  nerve graph exposes **branch points** and **bridges** (thin links between otherwise separate
  regions — topic leakage / merges in embeddings). Runs over the `M ≪ N` microclusters, with an
  optional `to_networkx()` for plotting. An exploration tool (structure, RAG curation, dedup), not a
  partition — complementary to the HDBSCAN density head.

## Architecture (crate layout)

| module | role |
|--------|------|
| `types` | `Real` numeric trait (`f32` / `f64`) |
| `linalg` | Cholesky / triangular solve / logdet / Mahalanobis / Jacobi eigensolver (no LAPACK) |
| `stats` | χ² quantile (inverse regularized incomplete gamma) for Mahalanobis gates |
| `feature` | clustering features: `Spherical` / `Diagonal` / `Full` / `FdSketch` (high-d) |
| `kernels` | auto-vectorized distance kernels (inline reductions) |
| `distance` | D0–D4, radius, Mahalanobis (stable forms) |
| `tree` | arena CF-tree + auto-rebuild |
| `clustering` | `kmeans`, `gmm_diagonal`, `gmm_full`, `ward_hac`, `hdbscan` |
| `model` | end-to-end `Model::fit` / `predict` |
| `python` | PyO3 bindings: one-shot `fit_predict` + streaming `Betula` estimator |

See [`DESIGN.md`](../DESIGN.md) for the full design and the verified mathematical foundation.
