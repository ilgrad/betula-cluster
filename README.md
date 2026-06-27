# betula-cluster

Fast, **memory-bounded streaming clustering** for large tabular / embedding data, built around
**numerically stable CF microclusters** (BETULA) — a from-scratch Rust core with parametric and
density/topological global clustering heads and a scikit-learn API, exposed to Python via PyO3.

## Why

Clustering libraries tend to either (a) not scale (full GMM/HDBSCAN on raw points), (b) lose
precision — classic BIRCH computes variance as `SS − ‖LS‖²/n`, which suffers catastrophic
cancellation far from the origin, or (c) blow up in memory (BIRCH-family subcluster explosion in
high dimensions). betula-cluster addresses all three:

- **Numerically stable** clustering features `(n, μ, S)` via Welford / Chan updates — no
  cancellation (classic BIRCH loses all digits near coordinate `1e8`; betula does not).
- **Memory-bounded by design** — the CF-tree caps its leaves (`max_leaves`) and rebuilds with a
  grown threshold, so it never explodes. On 100k × 10-d data it uses ~150 MB where scikit-learn
  Birch needs gigabytes (or OOMs).
- **Two-phase**: compress `N` points into a small CF-tree, then cluster the few leaf summaries.
- **Best formula, measured**: the GMM E-step uses an expected-log responsibility — a *CF-level
  objective* (not the only correct EM) measured to beat the convolution form on coarse CFs
  (`research/RESULTS-estep.md`).

## Mathematical foundation & improvements

Every formula below is verified symbolically (Maxima) and/or numerically (mpmath/Julia ground
truth) — see [`DESIGN.md`](DESIGN.md) and `research/`.

### Numerically stable cluster features `(n, μ, S)`

Classic BIRCH stores `(N, LS = Σx, SS = Σx²)` and recovers variance as `SS/N − (LS/N)²` — a
**difference of two large, nearly-equal numbers**. On real data with an offset (timestamps, money,
geo-coordinates, un-centered embeddings) this **catastrophically cancels**: in `f64` the variance
collapses to noise — and can go *negative* — around coordinate magnitude `1e7`, silently corrupting
every downstream radius, covariance, and label.

betula-cluster stores **`(n, μ, S)`** — weight, mean, and the sum of squared deviations
`S = Σ w (x − μ)(x − μ)ᵀ`. `S` is a **sum of non-negative terms**, so the variance `S/n ≥ 0` and the
covariance is **positive-semidefinite by construction** — there is nothing to cancel. The updates are
algebraically exact (not approximations):

```
add (Welford/West):  W' = W + w;   μ ← μ + (w/W')·δ;     S ← S + w·(1 − w/W')·(δ⊙δ)     [δ = x − μ_old]
merge (Chan):        W  = W_A+W_B; μ = μ_A + (W_B/W)·Δ;  S = S_A + S_B + (W_A·W_B/W)·(Δ⊙Δ) [Δ = μ_B − μ_A]
```

(full covariance: `δ⊙δ → δδᵀ`, `Δ⊙Δ → ΔΔᵀ`). Tested bit-stable at offset `1e7–1e8` against an
mpmath reference, where the classic `(N, LS, SS)` form loses all significant digits.

### GMM E-step: expected-log (variant C), not the paper's convolution

Running EM on the leaf CFs, each leaf is a mini-Gaussian `N(μ_i, Σ_i)`. The textbook / paper move
treats it by **convolution** — `N(μ_i | μ_k, Σ_k + Σ_i)` — which inflates each component by the
leaf's own spread and **washes out components on a coarse CF-tree**. betula-cluster instead uses the
**expected log-likelihood** responsibility (measured to give higher ARI — `research/RESULTS-estep.md`):

```
log r_ik = log π_k + log N(μ_i | μ_k, Σ_k) − ½·tr(Σ_k⁻¹ Σ_i)        (then log-sum-exp normalized)
```

The M-step folds the within-leaf scatter back in, so the fitted components stay exact:

```
Σ_k = Σ_i w_ik·(Σ_i + (μ_i − μ_k)(μ_i − μ_k)ᵀ) / N_k,    w_ik = n_i·r_ik
```

with NIW/MAP regularization `Σ_k = (Ψ + …)/(ν + N_k + d + 1)` so a 1-point leaf never yields a
singular covariance.

### Other verified improvements

- **Distances `D0–D4`** are the BIRCH measures re-derived on `(n, μ, S)` (Maxima-verified
  *equivalent*, computed stably). Variance-increase / Ward is `D4 = (n_A·n_B / n_AB)·‖Δμ‖²` — the `S`
  terms cancel by König–Huygens, so it is an exact centroid measure (no Lance-Williams approximation).
- **k-means on CFs** minimizes the true point objective, not the leaf-centroid proxy:
  `SSE = Σ_i [S_i + n_i·‖μ_i − c‖²]` folds each leaf's own scatter `S_i` back in, so compressing to a
  CF-tree first does not change what is being optimized.
- **Full covariance** uses a matrix Welford (PSD) with on-demand Cholesky for `logdet` /
  `mahalanobis`; the packed upper-triangular index is the tested `(j−1)·j/2` form — a reference
  implementation shipped a `(j−1)·dim/2` variant that silently corrupts `dim ≥ 4`.
- **χ² absorption gate** (`absorb="chi2"`): a mass-invariant Mahalanobis-χ² threshold with a
  Normal-Inverse-Gamma prior `var_eff = (S + κ·s₀)/(n + κ)`, finite at `n = 1`. Fixes the
  size-imbalance failure where a 12-point vs a 10⁴-point cluster decide differently (sklearn #22854).
- **Frequent-Directions sketch** (high `d`): the full-cov GMM consumes it in **low-rank** form —
  `tr(Σ_k⁻¹ Σ_i) = Σ_r ‖L_k⁻¹ f_r‖²` — so it never materializes a `d×d` matrix per leaf and keeps
  `O(ℓ·d)` memory through clustering. Identical math to the dense path.
- **Rebuild threshold** is the within-leaf mean nearest-sibling gap (ELKI/BETULA-standard,
  `O(M·capacity)`), raised monotonically — no global all-pairs scan, no over-growth collapse.
- **Robust insertion** (`huber_k = k`, optional): before a point `x` is folded into its target
  microcluster `(n, μ, S)`, each coordinate is winsorized to the cluster's own scale,

  ```
  x̃_j = clip(x_j, μ_j − k·σ_j, μ_j + k·σ_j),   σ_j = √(S_j / n)
  ```

  and the stable Welford update runs on `x̃` instead of `x`. A coordinate with `σ_j = 0` (no scale
  yet) passes through unchanged, and the clip is skipped until the target holds ≥ 5 points (so the
  scale estimate is trustworthy). This bounds any single point's pull on the centroid to `O(k·σ/n)`
  — outliers can no longer drag a centroid or inflate a radius — while leaving the CF a valid
  `(n, μ, S)` triple, so every downstream head is unchanged. The clipped value flows identically
  into the leaf entry and its ancestors, preserving the "each node = merge of its subtree" invariant.

## Relation to BIRCH and BETULA

This library is a from-scratch Rust implementation of the **BETULA** cluster feature — the
numerically stable `(n, μ, S)` summary introduced by Lang & Schubert to replace classic BIRCH's
cancellation-prone `(N, LS, SS)`:

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

The concrete, reproducible quality/speed/memory comparison is against **scikit-learn's Birch** — the
labeled BIRCH practitioners actually reach for — where betula uses **~34× less memory** and runs
**~46× faster** at equal ARI (and the naive Birch radius **OOMs**); see
[`bench/RESULTS.md`](bench/RESULTS.md) and the
[method-comparison notebook](examples/04_method_comparison.ipynb). (betulars produces no labels, so it
is not in that comparison; on the raw Phase-1 *build* the two are at parity — betula-cluster builds an
**identical tree** at every `N` and, with matched `target-cpu=native` flags, matches betulars'
wall-clock to within ~2 %. See [`bench/RESULTS.md`](bench/RESULTS.md).)

## Features

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
  formula in [Mathematical foundation](#mathematical-foundation--improvements).
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

## Install

```bash
pip install betula-cluster
```

Prebuilt `abi3` wheels (Python 3.11+) ship for Linux, macOS, and Windows, so no Rust toolchain is
needed. Build from source instead with:

```bash
# Python wheel (needs a Rust toolchain)
maturin build --release --features python
pip install target/wheels/betula_cluster-*.whl

# Rust library
# add betula-cluster as a path / git dependency in Cargo.toml
```

For a build pinned to *your own* CPU, add `target-cpu=native` for ~8 % off the CF-tree build from
AVX2 / AVX-512 vectorization of the distance kernels (this is what brings the build to parity with
betulars, whose wheels ship with it):

```bash
RUSTFLAGS="-C target-cpu=native" maturin build --release --features python
```

The published wheels deliberately stay portable (a `target-cpu=native` wheel raises `SIGILL` on any
CPU older than the build host), so this is a local/private build only — see [`.cargo/config.toml`](.cargo/config.toml).

## Quick start — Python

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

For streaming / out-of-core data use the scikit-learn-style estimator: feed chunks with
`partial_fit`, finalize with a no-arg `partial_fit()`, then `predict`. Memory stays bounded by
`max_leaves` no matter how much data streams through (the CF-tree rebuilds, it never grows without
limit) — or set **`memory_budget_mb`** and let it size `max_leaves` for you (a target for the tree's
resident size; most meaningful for streaming, where the data is transient and the tree is what grows).
Set **`huber_k`** (e.g. `2.0`) to winsorize each incoming point to ±`k·σ` of its target microcluster
before folding it in, so outliers in the stream cannot drag a centroid or inflate a radius.

```python
est = betula_cluster.Betula(method="gmm", memory_budget_mb=512)   # don't think about max_leaves
for chunk in stream_of_arrays:        # each chunk is a 2-D float64 array
    est.partial_fit(chunk)
est.partial_fit()                     # finalize the global clustering over everything seen
labels = est.predict(X_query)         # est.n_clusters_ / est.n_leaves_ / est.effective_max_leaves_
```

Soft assignment, confidence, a weighted-point coreset, and structure diagnostics — all over the
microclusters the tree already holds (no extra data passes):

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

Semi-supervised clustering with a few known pairwise relations (COP-KMeans). Constraints are
`(row_i, row_j)` index pairs into `X`:

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

Mixed numeric + categorical data (k-prototypes) — name the categorical column indices; their values
are integer codes:

```python
from betula_cluster import KPrototypes

# X columns: [age, income, city_code, plan_code]; columns 2 and 3 are categorical
kp = KPrototypes(n_clusters=5, categorical=[2, 3])    # gamma auto = ½·mean numeric σ
labels = kp.fit_predict(X)
kp.cluster_centroids_   # numeric centroids (n_clusters × n_numeric)
kp.cluster_modes_       # categorical modes   (n_clusters × n_categorical)
```

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

Bounded-memory streaming quantiles (standalone, mergeable across shards):

```python
from betula_cluster import KllSketch, DdSketch

kll = KllSketch(k=256)          # rank-error (uniform); DdSketch(alpha=0.01) for relative-error
for chunk in stream_of_values:
    kll.update_many(chunk)      # 1-D float64 array
p50, p99 = kll.quantile(0.5), kll.quantile(0.99)
kll.merge(other_shard_sketch)  # combine sketches computed in parallel
```

Sparse input is transparent — pass a `scipy.sparse` matrix to any of `fit` / `fit_predict` /
`partial_fit` / `predict`:

```python
import scipy.sparse as sp

X = sp.csr_matrix(one_hot_features)          # never densified to N × d
labels = Betula(method="kmeans", feature="diagonal").fit_predict(X)
```

For very high-dimensional sparse data (text TF-IDF, large one-hot), the `O(nnz)` sparse-native
one-shot touches only the non-zeros:

```python
from betula_cluster import fit_predict_sparse

labels = fit_predict_sparse(X, n_clusters=20, threshold=0.5)   # kmeans by default; O(nnz) per row
```

## Examples (notebooks)

**Twelve** executed, plotted notebooks — one per capability — live in
[`examples/`](examples/README.md) (seaborn plots, pandas tables, networkx graphs; render on GitHub):

- **Core** — [quickstart](examples/01_quickstart.ipynb) (every head + auto-`k`),
  [embeddings & inspection](examples/02_embeddings_and_inspection.ipynb),
  [streaming & persistence](examples/03_streaming_and_persistence.ipynb),
  [method comparison](examples/04_method_comparison.ipynb),
  [Mapper topology](examples/05_topology_mapper.ipynb).
- **Streaming density** — [`DenStream` & `DbStream`](examples/06_streaming_density.ipynb).
- **Mixed data** — [`KPrototypes`](examples/07_mixed_data_kprototypes.ipynb) (numeric + categorical).
- **Sketches** — [`KllSketch` & `DdSketch`](examples/08_quantile_sketches.ipynb) quantiles.
- **Semi-supervised** — [must-link / cannot-link](examples/09_semisupervised_constraints.ipynb).
- **Sparse / high-dim** — [`scipy.sparse` + `fit_predict_sparse`](examples/10_sparse_highdim.ipynb).
- **Soft assignment & coresets** —
  [`predict_proba`, coresets, diagnostics](examples/11_soft_assignment_coreset_diagnostics.ipynb).
- **Production ops** —
  [drift, active learning, robust, memory budgets](examples/12_drift_robust_memory.ipynb).

## Quick start — Rust

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

## Architecture

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

See `DESIGN.md` for the full design and the verified mathematical foundation.

## Results

**Tests.** 46 Rust unit/integration tests pass (incl. Hamerly ≡ brute Lloyd, X-means/auto-k
recovery, FD-vs-full, decay invariants, exact parallel-build summary); `cargo clippy --all-targets -D warnings` and
`cargo fmt --check` are clean across the `parallel`, serial, `persistence`, and `python` feature
sets. A `pytest` suite (`tests/test_python.py`, 61 tests) covers every binding — all heads +
features, auto-k, `f32` (one-shot **and** streaming), χ²/routing distances, decay, `normalize`,
the streaming estimator, save/load + pickle, scikit-learn `clone` / `Pipeline` compatibility, the
inspection API (microcluster/cluster geometry, outlier scores, near-duplicates), and the error
contract (incl. NaN/Inf and dtype rejection). Python end-to-end: k-means & GMM
recover separated blobs at **ARI = 1.000**; HDBSCAN separates two-moons at **ARI = 0.97**.

**Benchmark vs scikit-learn** (`bench/benchmark.py`; each method runs in an isolated subprocess
with a memory cap so a runaway baseline reports `OOM` instead of crashing — full table in
`bench/RESULTS.md`):

blobs, n = 100 000, d = 10:

| method | ARI | time (s) | peak RSS (MB) |
|--------|----:|---------:|--------------:|
| betula-kmeans | 1.000 | 0.27 | **147** |
| betula-kmeans (n_jobs=8) | 1.000 | **0.10** | 148 |
| betula-gmm | 1.000 | 0.28 | **147** |
| betula-gmm-full | 1.000 | 0.29 | 148 |
| sklearn-kmeans | 1.000 | 0.19 | 157 |
| sklearn-birch (tuned) | 1.000 | 24.9 | 5212 |
| sklearn-birch (radius 0.5) | — | — | **OOM** |
| sklearn-gmm | 1.000 | 0.71 | 194 |

two-moons, n = 20 000 (non-convex):

| method | ARI | time (s) | peak RSS (MB) |
|--------|----:|---------:|--------------:|
| betula-hdbscan | 0.976 | 0.19 | **132** |
| sklearn-kmeans | 0.255 | 0.09 | 132 |
| sklearn-hdbscan | 0.995 | 1.62 | 140 |

→ Matching quality at **~35× less memory** and **~90× faster** than sklearn-Birch on high-d (its naive
radius **OOMs**); the parallel Phase-1 build (`n_jobs=8`) adds a further **~2.6–3×** at equal
ARI/memory; HDBSCAN-on-CF solves non-convex shapes at comparable quality **~8.5× faster** than
sklearn-HDBSCAN; and on high-d full covariance (30k × 64) `feature="diagonal"` + `gmm-full` beats
sklearn-gmm **~4×** at less memory (full *component* covariance from diagonal leaves). FD trades speed
for memory — use it only when even a full `d×d`-per-leaf tree will not fit. Full table + honest notes
in `bench/RESULTS.md`.

## Mathematical foundation

Every formula is verified symbolically (Maxima) and/or numerically (Julia, Python + mpmath) in a
companion math workspace — the catastrophic-cancellation condition number, distance-form equivalence
(BIRCH ↔ BETULA), square-root (Cholesky) covariance, and the measured-best GMM E-step. Highlights:
stable `(n, μ, S)`; `D4 = Ward` increment with no `S` term (König–Huygens); covariance PSD by
construction; expected-log responsibility `log r_ik = log π_k + log N(μ_i|μ_k,Σ_k) − ½ tr(Σ_k⁻¹ Σ_i)`.

## Known limitations

Honest scope — these are inherent to a CF-compression + streaming design, not bugs:

1. **Insertion-order sensitive.** Like every BIRCH-family streaming method, the CF-tree (and so the
   labels) depends on the order points arrive; the parallel build differs from the serial one, as a
   different order would.
2. **`threshold` / `max_leaves` are real hyperparameters.** They trade compression against
   resolution; `n_rebuilds_` / `threshold_` expose when the tree is thrashing or over-coarsening.
3. **CF-level heads approximate raw-data clustering.** Phase-3 runs on the `M ≪ N` microclusters, not
   the raw points; quality degrades when clusters overlap at the compression scale. Mitigation: more
   leaves.
4. **HDBSCAN-on-CF ≠ raw-point HDBSCAN.** It is mass-aware HDBSCAN over the microclusters — fast and
   close, but an approximation.
5. **The expected-log GMM optimizes a CF-level objective**, not pointwise EM on the raw data (it is
   measured to be the better choice on coarse CFs, but it is a deliberate objective choice).
6. **Frequent-Directions is an approximate low-rank covariance** (exact only up to its rank `ℓ`);
   it trades accuracy and speed for bounded memory at very high `d`.

## Status & roadmap

**Done:** stable CF core, SIMD distances + rayon parallelism (incl. parallel Phase-1 build),
CF-tree + rebuild, Hamerly-accelerated k-means / diagonal & full GMM (with BIC auto-k) / Ward-HAC /
HDBSCAN, χ² absorption option, Frequent-Directions high-d sketch, EWMA decay, end-to-end model,
Python wheel (typed, scikit-learn `get_params`/`set_params` + `Pipeline`/`clone`/`GridSearchCV`) +
streaming `partial_fit` estimator + serde save/load/pickle persistence, NaN/Inf input validation,
`normalize=True` for cosine/direction clustering of embeddings, dataset-structure inspection
(microcluster/cluster geometry, outlier scores, near-duplicates, `n_rebuilds_`/`threshold_`
diagnostics), scikit-learn benchmark, CI + multi-platform wheel release.
**Measured & rejected:** a native cosine CF — on the unit sphere squared-Euclidean is monotone in
cosine (`d² = 2 − 2cosθ`), so `normalize=True` *is* spherical clustering and a separate cosine code
path adds nothing (`bench/cosine_spike.py`). **Known limit:** like all BIRCH-family compressors,
quality degrades when clusters overlap at the compression scale (within-≈between-cluster distance);
raise `max_leaves` (watch `n_rebuilds_`). **Planned:** GPU kernels (CUDA / OpenCL),
persistence-diagram exploration (the Mapper skeleton ships now via `mapper()`). See `DESIGN.md`.

## License

MIT.
