# betula-cluster

**Numerically stable, memory-bounded clustering** — a from-scratch Rust core with a scikit-learn API.

betula-cluster compresses raw data or a stream into a small set of numerically stable BETULA
*microclusters* (a CF-tree), then runs the clustering head over the compressed summary. Cost scales
with the compression size, not with `N` — so it labels **a million points in ~0.2 s** and streams
**ten million in a flat ~57 MB of RAM**, at quality on par with scikit-learn.

```bash
pip install betula-cluster
```

```python
import numpy as np
import betula_cluster

X = np.random.default_rng(0).normal(size=(1_000_000, 10))
labels = betula_cluster.fit_predict(X, n_clusters=10, method="kmeans")
```

## What's here

- **[Usage guide](USAGE.md)** — every API with runnable snippets (fit/predict, streaming, sparse,
  constraints, mixed data, sketches, tuning).
- **[Features](FEATURES.md)** — the full capability list (stable core + experimental heads).
- **[The math](MATH.md)** — why the stable $(n, \mu, S)$ cluster feature avoids catastrophic
  cancellation, and the derivations betula adds on top.
- **[API reference](api.md)** — the typed public surface.

## Why it's different

- **Stable core.** Classic BIRCH `(N, LS, SS)` loses precision on real, offset data; BETULA's
  $(n, \mu, S)$ moments never form the cancelling difference (positive-semidefinite by construction).
- **Bounded memory.** `max_leaves` / `memory_budget_mb` cap the tree, so `partial_fit` streams data
  larger than RAM.
- **One engine.** k-means · GMM (diagonal & full) · Ward · spectral · Leiden community detection ·
  directional (spherical k-means / von Mises–Fisher) · scale-space density modes · AR/Toeplitz GMM
  (time-series windows) · HDBSCAN-CF · Mapper topology — plus dedup, outliers, representatives,
  coresets, `consensus` stability and memory-aware `tune`, over the same stable CF-tree.
- **Lean.** A single typed abi3 wheel, zero LAPACK / SciPy at runtime; the only dependency is NumPy.

Benchmarks (speed, memory, quality — wins *and* losses) are reproducible from `bench/comprehensive.py`
and written up in [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md).
