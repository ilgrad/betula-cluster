# betula-cluster

[![PyPI](https://img.shields.io/pypi/v/betula-cluster.svg)](https://pypi.org/project/betula-cluster/)
[![Python](https://img.shields.io/pypi/pyversions/betula-cluster.svg)](https://pypi.org/project/betula-cluster/)
[![CI](https://github.com/ilgrad/betula-cluster/actions/workflows/ci.yml/badge.svg)](https://github.com/ilgrad/betula-cluster/actions/workflows/ci.yml)
[![Python coverage 100%](https://img.shields.io/badge/python%20coverage-100%25-brightgreen.svg)](https://github.com/ilgrad/betula-cluster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)
[![Rust core · PyO3](https://img.shields.io/badge/Rust%20core-PyO3-orange.svg)](https://github.com/ilgrad/betula-cluster)

> **Fast, memory-bounded clustering for large tabular & embedding data.** A numerically stable
> **BETULA** CF-tree with a full set of clustering heads — k-means · GMM (diagonal & full) · Ward ·
> HDBSCAN · Mapper — plus streaming `partial_fit` and a scikit-learn API. From-scratch **Rust** core,
> **PyO3** bindings, no LAPACK, no SciPy at runtime.

```bash
pip install betula-cluster
```

## At a glance — honest benchmarks

Measured against scikit-learn on `StandardScaler`-normalized data, each method in its own subprocess
with peak RSS sampled from `/proc/self/statm`. Full methodology, every metric, and all tables (wins
**and** losses) live in [**`bench/RESULTS.md`**](bench/RESULTS.md).

- 🎯 **Quality at parity.** betula's k-means / GMM / Ward land within **≈0.01 ARI** of their
  scikit-learn counterparts; full-covariance GMM recovers anisotropic clusters just as well
  (0.90 vs 0.90); HDBSCAN-on-CF nails non-convex moons & circles (**ARI 1.00**).
- ⚡ **13–39× faster at N = 1 000 000.** betula-kmeans labels a million points in **0.20 s** vs
  scikit-learn KMeans 2.7 s (13×), Birch 7.9 s (39×), GaussianMixture 4.9 s (25×).
- 🪶 **Bounded memory.** Streaming 10 M points peaks at **~57 MB — flat in N** — while an in-core
  KMeans must hold the array and peaks at **~5 GB** (**≈88× less**, and the gap grows without limit).

| ![Fit time vs N](https://raw.githubusercontent.com/ilgrad/betula-cluster/main/bench/plots/scaling_time.png) | ![Peak memory vs N](https://raw.githubusercontent.com/ilgrad/betula-cluster/main/bench/plots/memory_streaming.png) |
|:--:|:--:|
| Phase-3 clusters only the ~2 000 leaf microclusters, not the raw points, so every head finishes 1 M points in **under ⅓ s**. | The CF-tree is capped by `max_leaves`, so streaming memory stays **flat** — it clusters data larger than RAM. |

## Why

Clustering libraries tend to either not scale (full GMM/HDBSCAN on raw points), lose precision
(classic BIRCH computes variance as `SS − ‖LS‖²/n`, which catastrophically cancels far from the
origin), or blow up in memory (BIRCH-family subcluster explosion in high dimensions). betula-cluster
addresses all three:

- **Numerically stable** — clustering features `(n, μ, S)` via Welford / Chan updates; the covariance
  is PSD by construction. Classic BIRCH loses all digits near coordinate `1e7`; betula does not.
- **Memory-bounded by design** — the CF-tree caps its leaves (`max_leaves`) and rebuilds, so it never
  explodes; streaming memory is flat in `N` and clusters data larger than RAM.
- **Complete** — one stable engine spanning k-means / GMM (diag & full) / Ward / HDBSCAN-style /
  Mapper, with streaming `partial_fit`, a scikit-learn API, and dataset-structure inspection.

The math (stable CF, the expected-log GMM E-step, distance derivations, relation to BIRCH/BETULA) is
written up — verified symbolically and numerically — in [**`docs/MATH.md`**](docs/MATH.md).

## Quick start

```python
import numpy as np
import betula_cluster

X = np.random.default_rng(0).normal(size=(100_000, 10))

labels = betula_cluster.fit_predict(X, n_clusters=10, method="kmeans")
labels = betula_cluster.fit_predict(X, n_clusters=0, feature="full", method="gmm-full")  # auto-k via BIC
labels = betula_cluster.fit_predict(X, method="hdbscan", min_cluster_size=25)             # -1 == noise
```

Streaming / out-of-core — feed chunks, finalize, predict; memory stays bounded by `max_leaves`:

```python
est = betula_cluster.Betula(method="gmm", memory_budget_mb=512)
for chunk in stream_of_arrays:        # each chunk is a 2-D float32/float64 array
    est.partial_fit(chunk)
est.partial_fit()                     # finalize the global clustering over everything seen
labels = est.predict(X_query)
```

Constraints (COP-KMeans), mixed numeric+categorical (`KPrototypes`), streaming density (`DenStream` /
`DbStream`), quantile sketches, `scipy.sparse` input, soft assignment / coresets / diagnostics, the
Rust API, and the CLI — all in the [**usage guide**](docs/USAGE.md).

## Capabilities

- **Clustering heads** — weighted k-means, GMM (diagonal & full covariance, BIC auto-`k`), exact Ward
  HAC, HDBSCAN-style density over CF microclusters, and a Mapper topological skeleton.
- **Streaming** — `partial_fit` at bounded memory; `DenStream` & `DbStream` for evolving streams;
  mergeable `KllSketch` / `DdSketch` quantiles.
- **Data types** — dense `f32`/`f64`, `scipy.sparse` (never densified), `O(nnz)` sparse-native, and
  mixed numeric+categorical (k-prototypes).
- **Beyond labels** — `predict_proba`, coresets, diagnostics, outliers / near-duplicates /
  representatives, drift snapshots, COP-KMeans constraints, and robust (Huber) insertion.
- **Engineering** — scikit-learn API (`Pipeline` / `clone` / `GridSearchCV`), typed abi3 wheel,
  `save` / `load` + pickle, a dependency-free CLI, and a reusable Rust core.

Full reference: [**`docs/FEATURES.md`**](docs/FEATURES.md).

## Examples

**Twelve** executed, plotted notebooks — one per capability — live in
[`examples/`](examples/README.md) (render on GitHub):

- **Core** — [quickstart](examples/01_quickstart.ipynb),
  [embeddings & inspection](examples/02_embeddings_and_inspection.ipynb),
  [streaming & persistence](examples/03_streaming_and_persistence.ipynb),
  [method comparison](examples/04_method_comparison.ipynb),
  [Mapper topology](examples/05_topology_mapper.ipynb).
- **Streaming density** — [`DenStream` & `DbStream`](examples/06_streaming_density.ipynb).
- **Mixed data** — [`KPrototypes`](examples/07_mixed_data_kprototypes.ipynb).
- **Sketches** — [`KllSketch` & `DdSketch`](examples/08_quantile_sketches.ipynb).
- **Semi-supervised** — [must-link / cannot-link](examples/09_semisupervised_constraints.ipynb).
- **Sparse / high-dim** — [`scipy.sparse` + `fit_predict_sparse`](examples/10_sparse_highdim.ipynb).
- **Soft assignment & coresets** —
  [`predict_proba`, coresets, diagnostics](examples/11_soft_assignment_coreset_diagnostics.ipynb).
- **Production ops** — [drift, active learning, robust, memory budgets](examples/12_drift_robust_memory.ipynb).

And three **end-to-end use cases** (each scored against ground truth):

- 🧹 [**Embedding dedup**](examples/usecases/usecase_01_embedding_dedup.ipynb) — collapse a repost-heavy corpus to representatives.
- 🚨 [**Log anomaly detection**](examples/usecases/usecase_02_log_anomaly_detection.ipynb) — batch outlier scoring + streaming `DbStream` flags.
- 👥 [**Customer segmentation**](examples/usecases/usecase_03_customer_segmentation.ipynb) — mixed RFM + categorical personas with `KPrototypes`.

## Documentation

- [**Usage guide**](docs/USAGE.md) — runnable snippets for every interface.
- [**Features**](docs/FEATURES.md) — full capability reference + crate architecture.
- [**Math**](docs/MATH.md) — stable CF, GMM E-step, distance derivations, relation to BIRCH/BETULA.
- [**Benchmarks**](bench/RESULTS.md) — methodology, every metric, all tables, honest wins & losses.
- [**Design**](DESIGN.md) — internal design, invariants, and testing strategy.

Verified: **128** Rust unit/integration tests + a **121-case** Python suite at **100%** wrapper
coverage (Rust ≥95%, CI-enforced), `clippy -D warnings` + `fmt` clean across all feature sets, on
Python 3.11–3.14 (single abi3 wheel).

## Known limitations

Honest scope — inherent to a CF-compression + streaming design, not bugs:

1. **Insertion-order sensitive** — like every BIRCH-family streaming method, the labels depend on the
   order points arrive (the parallel build differs from the serial one, as a different order would).
2. **`threshold` / `max_leaves` are real hyperparameters** — they trade compression against
   resolution; `n_rebuilds_` / `threshold_` expose thrashing / over-coarsening.
3. **CF-level heads approximate raw-data clustering** — Phase-3 runs on the `M ≪ N` microclusters;
   quality degrades when clusters overlap at the compression scale. Mitigation: more leaves.
4. **HDBSCAN-on-CF ≠ raw-point HDBSCAN** — mass-aware HDBSCAN over microclusters: fast and close, but
   an approximation (weaker on *overlapping* blobs; see the benchmarks).
5. **The expected-log GMM optimizes a CF-level objective**, not pointwise EM (a deliberate, measured
   choice for coarse CFs).
6. **Frequent-Directions is an approximate low-rank covariance** (exact only up to its rank `ℓ`).

## License

MIT.
