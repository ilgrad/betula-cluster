# Usage guide

Runnable snippets for every interface. For executed, plotted walk-throughs see the
[example notebooks](../examples/README.md); for the full capability list see [`FEATURES.md`](FEATURES.md).

## One-shot — `fit_predict`

```python
import numpy as np
import betula_cluster

X = np.random.default_rng(0).normal(size=(100_000, 10))

labels = betula_cluster.fit_predict(X, n_clusters=10, feature="diagonal", method="gmm")
labels = betula_cluster.fit_predict(X, n_clusters=10, feature="full", method="gmm-full")
labels = betula_cluster.fit_predict(X, n_clusters=0, method="gmm")  # auto-select k via BIC
labels = betula_cluster.fit_predict(X, n_clusters=10, method="kmeans")
labels = betula_cluster.fit_predict(X, method="hdbscan", min_samples=10, min_cluster_size=25)
# hdbscan: label -1 == noise
```

Keyword args: `feature ∈ {spherical, diagonal, full, fd}`, `method ∈ {kmeans, gmm, gmm-full, ward, hdbscan}`,
`distance ∈ {euclidean, manhattan, ward, average}` (routing measure),
`absorb ∈ {euclidean, chi2}` (`chi2` = mass-invariant Mahalanobis gate at level `chi2_p` with
`chi2_scale` = within-cluster variance; fixes the BIRCH size-imbalance bug), `decay` (EWMA factor
for streaming concept drift), `normalize` (L2-normalize rows → cluster by *direction* for embeddings;
on the unit sphere squared-Euclidean is monotone in cosine, so the tree clusters by angle),
`n_jobs` (parallel shard+merge tree build — `>1` gives ~4–5× on large
`N`), `threshold`, `branching`, `leaf_cap`, `max_leaves`, `max_iter`, `min_samples`,
`min_cluster_size`, `seed`. `n_clusters=0` ⇒ automatic `k` for every parametric head (BIC for
k-means/GMM, dendrogram cut for Ward).

## Streaming / out-of-core — the `Betula` estimator

Feed chunks with `partial_fit`, finalize with a no-arg `partial_fit()`, then `predict`. Memory stays
bounded by `max_leaves` no matter how much data streams through (the CF-tree rebuilds, it never grows
without limit) — or set **`memory_budget_mb`** and let it size `max_leaves` for you (a target for the
tree's resident size; most meaningful for streaming, where the data is transient and the tree is what
grows). Set **`huber_k`** (e.g. `2.0`) to winsorize each incoming point to ±`k·σ` of its target
microcluster before folding it in, so outliers in the stream cannot drag a centroid or inflate a radius.

```python
est = betula_cluster.Betula(method="gmm", memory_budget_mb=512)   # don't think about max_leaves
for chunk in stream_of_arrays:        # each chunk is a 2-D float64 array
    est.partial_fit(chunk)
est.partial_fit()                     # finalize the global clustering over everything seen
labels = est.predict(X_query)         # est.n_clusters_ / est.n_leaves_ / est.effective_max_leaves_
```

## Soft assignment, coresets, diagnostics, drift

All over the microclusters the tree already holds (no extra data passes):

```python
proba = est.predict_proba(X_query)            # (n, k): GMM posterior; centroid-softmax heuristic for other heads
conf  = est.assignment_confidence(X_query)    # (n,) in [0, 1] — low flags boundary / ambiguous points
coreset = est.export_coreset()                # coreset.centers / .weights / .radii — fit any weighted model on these
report  = est.diagnostics()                   # compression_ratio, radius p50/p90/p99, cluster mass spread, n_rebuilds
reps    = est.representatives(X_query, cluster_id=0, method="medoid")   # or "boundary" / "outlier" / "diverse"
profile = est.cluster_profile(0)              # JSON-able geometry + nearest clusters (e.g. to LLM-name a cluster)
batch   = est.active_learning_batch(X_query, n=100, strategy="uncertain")  # rows to review/label

snap = est.snapshot()                         # cluster geometry now; later, detect drift:
drift = betula_cluster.Betula.compare_snapshots(snap, est_next.snapshot())  # shifts / mass ratios / births
```

## Semi-supervised — COP-KMeans constraints

Constraints are `(row_i, row_j)` index pairs into `X`:

```python
est = betula_cluster.Betula(n_clusters=4, method="kmeans")
labels = est.fit_predict(
    X,
    must_link=[(0, 5), (0, 9)],      # rows 0, 5, 9 end up in the same cluster
    cannot_link=[(0, 42)],           # rows 0 and 42 end up in different clusters
)
# Infeasible (e.g. a cannot-link inside one microcluster, or more mutually-cannot-linked
# groups than n_clusters) raises ValueError — constraints are never silently violated.
```

## Mixed numeric + categorical — `KPrototypes`

Name the categorical column indices; their values are integer codes:

```python
from betula_cluster import KPrototypes

# X columns: [age, income, city_code, plan_code]; columns 2 and 3 are categorical
kp = KPrototypes(n_clusters=5, categorical=[2, 3])    # gamma auto = ½·mean numeric σ
labels = kp.fit_predict(X)
kp.cluster_centroids_   # numeric centroids (n_clusters × n_numeric)
kp.cluster_modes_       # categorical modes   (n_clusters × n_categorical)
```

## Evolving streams — `DenStream` & `DbStream`

For an *evolving* stream where stale data should fade, use the separate `DenStream` head:

```python
from betula_cluster import DenStream

ds = DenStream(eps=1.5, decay=0.05, beta=0.5, mu=4)   # eps = micro-cluster radius (tune to scale)
for chunk in stream_of_arrays:
    ds.partial_fit(chunk)                              # old micro-clusters fade as new data arrives
labels = ds.predict(X_query)                          # -1 = noise; finalizes the offline step once
```

For arbitrarily-shaped clusters on a stream (or to avoid bridging close-but-disconnected regions),
use `DbStream`, which connects micro-clusters by shared density rather than distance:

```python
from betula_cluster import DbStream

ds = DbStream(r=1.5, decay=0.05, alpha=0.1)   # r = micro radius; alpha = shared-density bridge
for chunk in stream_of_arrays:
    ds.partial_fit(chunk)
labels = ds.predict(X_query)                  # -1 = noise; finalizes the shared-density graph once
```

## Streaming quantiles — `KllSketch` & `DdSketch`

Bounded-memory, mergeable across shards:

```python
from betula_cluster import KllSketch, DdSketch

kll = KllSketch(k=256)          # rank-error (uniform); DdSketch(alpha=0.01) for relative-error
for chunk in stream_of_values:
    kll.update_many(chunk)      # 1-D float64 array
p50, p99 = kll.quantile(0.5), kll.quantile(0.99)
kll.merge(other_shard_sketch)  # combine sketches computed in parallel
```

## Sparse input

Transparent — pass a `scipy.sparse` matrix to any of `fit` / `fit_predict` / `partial_fit` / `predict`:

```python
import scipy.sparse as sp

X = sp.csr_matrix(one_hot_features)          # never densified to N × d
labels = betula_cluster.Betula(method="kmeans", feature="diagonal").fit_predict(X)
```

For very high-dimensional sparse data (text TF-IDF, large one-hot), the `O(nnz)` sparse-native
one-shot touches only the non-zeros:

```python
from betula_cluster import fit_predict_sparse

labels = fit_predict_sparse(X, n_clusters=20, threshold=0.5)   # kmeans by default; O(nnz) per row
```

## Rust

```rust
use betula_cluster::distance::CentroidEuclidean;
use betula_cluster::feature::Spherical;
use betula_cluster::model::{Method, Model};
use betula_cluster::tree::CFTree;

let mut tree: CFTree<f64, Spherical<f64>, _, _> =
    CFTree::new(2, 32, 32, 0.0, 2000, CentroidEuclidean, CentroidEuclidean);
for p in &points {
    tree.insert(p);
}
let model = Model::fit(tree, 4, Method::Gmm, 100, 0);
let label = model.predict(&points[0]);
```

## Command line

A dependency-free `betula` binary (behind the `cli` feature) clusters a delimited numeric file (or
stdin) and writes one label per row to stdout:

```sh
cargo install --path . --features cli          # or: cargo build --release --features cli
betula --clusters 4 --method gmm data.csv      # reads a comma-delimited matrix
cat data.csv | betula -k 0 --method kmeans      # k=0 → auto-select k; reads stdin
betula --help                                   # all options
```

Flags mirror the library: `--feature`, `--threshold`, `--branching`, `--leaf-cap`, `--max-leaves`,
`--max-iter`, `--seed`, `--delimiter`, `--header`.

## Build from source

Prebuilt `abi3` wheels (Python 3.11+) ship for Linux, macOS, and Windows, so normally no Rust
toolchain is needed. To build from source instead:

```bash
# Python wheel (needs a Rust toolchain)
maturin build --release --features python
pip install target/wheels/betula_cluster-*.whl

# Rust library: add betula-cluster as a path / git dependency in Cargo.toml
```

For a build pinned to *your own* CPU, add `target-cpu=native` for ~8 % off the CF-tree build from
AVX2 / AVX-512 vectorization of the distance kernels (this is what brings the build to parity with
betulars, whose wheels ship with it):

```bash
RUSTFLAGS="-C target-cpu=native" maturin build --release --features python
```

The published wheels deliberately stay portable (a `target-cpu=native` wheel raises `SIGILL` on any
CPU older than the build host), so this is a local/private build only — see
[`.cargo/config.toml`](../.cargo/config.toml).
