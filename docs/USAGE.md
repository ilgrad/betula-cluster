# Usage guide

Runnable snippets for every interface. For executed, plotted walk-throughs see the
[example notebooks](https://github.com/ilgrad/betula-cluster/blob/main/examples/README.md); for the full capability list see [`FEATURES.md`](FEATURES.md).

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

Keyword args: `feature ∈ {spherical, diagonal, full, fd}`, `method ∈ {kmeans, gmm, gmm-full, ward, spectral, leiden, leiden-cpm, spherical-kmeans, vmf, gmm-toeplitz, gmm-toeplitz-full, gmm-toeplitz-gs, hdbscan, scale-space}`,
`distance ∈ {euclidean, manhattan, ward, average}` (routing measure),
`absorb ∈ {euclidean, manhattan, average, diameter, ward, radius, chi2}` (see *Absorption criteria*
below; `chi2` = mass-invariant Mahalanobis gate at level `chi2_p` with `chi2_scale` = within-cluster
variance; fixes the BIRCH size-imbalance bug), `decay` (EWMA factor
for streaming concept drift), `normalize` (L2-normalize rows → cluster by *direction*; on the unit
sphere squared-Euclidean is monotone in cosine, so the tree clusters by angle. It earns its keep on
`digits`-64 (k-means **0.467 → 0.569**, ward **0.643 → 0.699**, median of three seeds); on MNIST-784
it is now a wash — 0.307 → 0.346, inside the seed spread and sign-flipping between seeds, since the
tree-rebuild fix removed most of the Euclidean collapse it used to compensate for. Leave it off for
tabular data where magnitude is signal: it takes covtype ward to **−0.049**, worse than random),
`n_jobs` (parallel shard+merge tree build — `>1` gives ~4–5× on large
`N`), `threshold`, `branching`, `leaf_cap`, `max_leaves`, `max_iter`, `min_samples`
(for `method="hdbscan"`, the core-distance neighbourhood **counting the microcluster itself** —
the convention of Campello's Def. 3.1, `sklearn.cluster.HDBSCAN` and ELKI, so `min_samples=1`
leaves every core distance at 0 and HDBSCAN\* degenerates to single linkage;
`scikit-learn-contrib/hdbscan` excludes it, where the same number means one neighbour more),
`min_cluster_size`, `resolution` (Leiden γ — granularity for `method="leiden"` / `"leiden-cpm"`, higher
⇒ more communities), `covariance_weight` (Leiden β — a log-Euclidean covariance/shape term in the
affinity, `feature="full"`; `0` = off, the centroid-only default), `tangent_weight` / `tangent_rank`
(Leiden γ — a Grassmann tangent-subspace term of rank `tangent_rank` for manifold-aware communities,
`feature="full"`; `0` = off), `projection` / `projection_dim` / `projection_max_iter` (reduce the leaf centroids to
`projection_dim` codes before the head; `"none"` = off. **`"weighted-nmf"`**, or
**`"weighted-nmf-kl"`** for count data, gives nonnegative CF-weighted NMF codes — for **nonnegative**
data only: TF-IDF / counts / spectrograms, dense or CSR. **`"svd"`** gives a CF-weighted PCA, accepts
signed data, and is the one-call text pipeline — see *Text: reduce and cluster in one call* below.
After a fit, `components_` gives the `(projection_dim, dim)` parts and `reconstruction_err_` the
relative fit error), `refine` (BIRCH Phase 4 — see below), `seed`. `n_clusters=0` ⇒ automatic `k` for every parametric head (BIC for
k-means/GMM, dendrogram cut for Ward). `threshold="auto"` (dense only) drops the one knob users most
often have to guess: a subsample pilot estimates a warm-start absorption radius, so the full fit
starts near-converged instead of growing the threshold from zero.

### Absorption criteria

`absorb` decides when a point joins an existing leaf rather than starting a new one — the single
choice that shapes the whole summary. The full BIRCH grid is available, plus this crate's own gate:

| `absorb` | BIRCH name | what it measures | `threshold` units |
|---|---|---|---|
| `euclidean` *(default)* | D0 | squared distance to the centroid | squared |
| `manhattan` | D1 | L1 distance to the centroid | **L1, not squared** |
| `average` | D2 | mean squared distance between the two clusters' points | squared |
| `diameter` | D3 | mean squared distance *within* the merged cell | squared |
| `ward` | D4 | variance increase from the merge | squared |
| `radius` | R | mean squared radius of the merged cell | squared |
| `chi2` | — | Mahalanobis-χ² with a variance prior | χ²`dim` quantile via `chi2_p` |

`threshold` is read in the chosen criterion's own units, so a value tuned for one does **not**
transfer to another — retune when you switch. `D2`, `D3` and `R` read the leaves' second moments, so
they grow with a cell's scatter; `D0`, `D1` and `D4` are centroid-only and therefore the most
numerically stable.

**The default is deliberate, and the alternatives optimise a different objective.** Lang's thesis
tunes absorption for minimum variance and finds D4 × D2 best on Gaussian data; this crate chose
mass-invariance instead, because the variance-minimising criteria inherit BIRCH's size-imbalance bug
(scikit-learn [#22854](https://github.com/scikit-learn/scikit-learn/issues/22854) — a large cluster
swallows a distant point because its average radius barely moves). The two objectives genuinely
conflict: our own measurements have the radius criterion over-absorbing exactly where `euclidean` and
`chi2` correctly reject. Pick `radius` or `diameter` if you want BIRCH's published behaviour, `chi2`
if your clusters differ wildly in size, and leave the default alone otherwise.

### Choosing a head

| your data / goal | `method` | needs `k`? |
|---|---|---|
| compact/spherical groups, fastest | `kmeans` | yes |
| elliptical / correlated / anisotropic, soft assignment | `gmm` (diag) or `gmm-full` | yes (or `0` = BIC) |
| **L2-normalized embeddings** (CLIP / face / sentence / speaker), cosine geometry | `vmf` (soft) or `spherical-kmeans` (hard) | yes (or `0` = BIC, `vmf`) |
| a cluster *hierarchy* / merge structure | `ward` | yes (or `0` = dendrogram cut) |
| **non-convex / manifold** shapes (moons, rings, spirals) | `spectral` | yes (pair with a **small** `threshold`) |
| **community / graph structure**, unknown count | `leiden` (or `leiden-cpm`) | **no** — count is discovered; tune `resolution` |
| variable-density clusters **+ noise**, unknown count | `hdbscan` | no |
| **density peaks**, arbitrary count, no `k` *or* bandwidth to pick | `scale-space` | **no** — scale chosen by mode persistence |
| **ordered / stationary signals** (time-series windows, trajectories, sensor waveforms), covariance *shape* | `gmm-toeplitz` | yes (or `0` = BIC) |
| ordered signals with structure **beyond a low-order AR** (long-lag echo, narrowband) | `gmm-toeplitz-full` (any lag) or `gmm-toeplitz-gs` (likelihood-optimal precision, ≤ order 16) | yes (or `0` = BIC) |
| topological skeleton / #components / loops | [`mapper()`](FEATURES.md) | no |

`n_clusters=0` auto-selects `k` for the parametric heads; `leiden` / `hdbscan` always discover it
(`leiden` reads the count off the graph — tune granularity with `resolution` γ, higher ⇒ more).
For a robustness score per point, wrap any partitional head in `consensus` (see below).

### Refining on the raw points — `refine`

`refine=n` runs BIRCH's Phase 4: `n` Lloyd sweeps over the raw rows, warm-started from the Phase-3
centres. It is **off by default**, applies only to the centroid heads (`kmeans`, `spherical-kmeans`),
and only to the in-memory `fit` / `fit_predict` — `partial_fit` keeps a tree, not the data, and the
sparse path would have to densify the matrix it exists to avoid.

It moves the objective where the summary is coarse relative to the data. MNIST (first 20 000 rows,
784-D, `StandardScaler`, k=10, spherical CF, `threshold=0`, `max_leaves=4000`; median of seeds
0/1/2 — `local/scratch/refine_claims.py`):

| | ARI | k-means objective | time |
|---|---|---|---|
| `refine=0` | **0.315** | 11 750 563 | 4.2 s |
| `refine=5` | 0.311 | 11 720 402 | 4.6 s |
| `refine=20` | 0.309 | **11 710 630** | 6.4 s |
| `sklearn KMeans(n_init=10)` | 0.324 | 11 671 351 | 19.3 s |

**Read that table before enabling the parameter: the objective falls monotonically and the ARI falls
with it.** Twenty sweeps buy 0.34 % of objective for a 52 % time premium and cost 0.006 ARI. Phase 4
does exactly what it says — Lloyd is monotone in the objective — but on this dataset the objective and
the ground truth point in opposite directions, which is the caveat two paragraphs down and not a bug
in the sweep.

**Two regimes get nothing, for structural reasons rather than weak refinement.** When
`max_leaves ≥ N` the tree holds one leaf per point — `digits` at `max_leaves=4000` realises 1 797
leaves for 1 797 rows — so Phase 3 *is* exact k-means on the raw data and Phase 4 starts at its fixed
point; the labels are bit-identical at every `refine`. Raising the budget does the same thing more
gradually: MNIST at `max_leaves=16000` reaches ARI 0.3237 unrefined at objective 11 671 813 —
scikit-learn's own answer, in 15.8 s against its 19.3 s — and twenty sweeps then move the ARI by
0.0001. `covtype` at `max_leaves=4000` is the same story from the other end: 0.1993 → 0.1998, and the
centres move by 7e-5 relative.

**A lower objective is not a better partition, and on this benchmark it is reliably worse.** On
`covtype` (same probe), `sklearn KMeans(n_init=10)` reaches the better objective (827 314 against
`n_init=1`'s 832 081) and **0.174 ARI against 0.277**. `digits` shows the same inversion — `n_init=1`
scores 0.559 at objective 69 749, `n_init=10` scores 0.468 at 69 405. Refinement optimizes the
objective faithfully; whether that is what you want is a property of your data, so measure it rather
than assuming.

## Streaming / out-of-core — the `Betula` estimator

Feed chunks with `partial_fit`, finalize with a no-arg `partial_fit()`, then `predict`. Memory stays
bounded by `max_leaves` no matter how much data streams through (the CF-tree rebuilds, it never grows
without limit) — or set **`memory_budget_mb`** and let it size `max_leaves` for you (a target for the
tree's resident size; most meaningful for streaming, where the data is transient and the tree is what
grows). Set **`huber_k`** (e.g. `2.0`) to winsorize each incoming point to $\pm k\sigma$ of its target
microcluster before folding it in, so outliers in the stream cannot drag a centroid or inflate a radius.

```python
est = betula_cluster.Betula(method="gmm", memory_budget_mb=512)   # don't think about max_leaves
for chunk in stream_of_arrays:        # each chunk is a 2-D float64 array
    est.partial_fit(chunk)
est.partial_fit()                     # finalize the global clustering over everything seen
labels = est.predict(X_query)         # est.n_clusters_ / est.n_leaves_ / est.effective_max_leaves_
```

### Sizing `max_leaves` against `n_clusters`

The summary has to be finer than the partition you ask of it. Below **two leaves per cluster** the
head has essentially no freedom — every cluster is one leaf and the answer is the tree's, not the
head's — and quality collapses: over three seeds on the `ward` head, well-separated synthetic data
loses 29 % (`k`=50) and 55 % (`k`=200) of its achievable ARI at ≈1 leaf per cluster, while `digits`
and `covtype` score 0.000 and 0.003 there. betula raises a `UserWarning` naming the realised leaf
count, `n_clusters` and the current `max_leaves` whenever it lands under that floor.

Two is a **floor, not a target**. More resolution is not monotonically better: on `covtype` the same
sweep peaks at ≈8 leaves per cluster and declines after, while `digits` keeps improving to ≈60. If the
warning fires, raise `max_leaves` (or lower `threshold`, or lower `n_clusters`) — then tune the ratio
on your own data rather than assuming higher is better. The warning reads the **realised** leaf count,
not the cap: the tree routinely settles below `max_leaves`, and when `N < max_leaves` the cap never
binds at all.

## Soft assignment, coresets, diagnostics, drift

All over the microclusters the tree already holds (no extra data passes):

```python
proba = est.predict_proba(X_query)            # (n, k): the point's own mixture posterior (argmax == predict); centroid-softmax heuristic for the non-generative heads
conf  = est.assignment_confidence(X_query)    # (n,) in [0, 1] — low flags boundary / ambiguous points
coreset = est.export_coreset()                # coreset.centers / .weights / .radii — fit any weighted model on these
report  = est.diagnostics()                   # compression_ratio, radius p50/p90/p99, cluster mass spread, n_rebuilds
reps    = est.representatives(X_query, cluster_id=0, method="medoid")   # or "boundary" / "outlier" / "diverse"
profile = est.cluster_profile(0)              # JSON-able geometry + nearest clusters (e.g. to LLM-name a cluster)
batch   = est.active_learning_batch(X_query, n=100, strategy="uncertain")  # rows to review/label

snap = est.snapshot()                         # cluster geometry now; later, detect drift:
drift = betula_cluster.Betula.compare_snapshots(snap, est_next.snapshot())  # matched clusters: centroid shifts / mass ratios
```

## Topological structure — `mapper()`

A TDA-Mapper skeleton over the microclusters: non-convex shape, **branch points**, and **bridges**
(thin links that flag topic leakage / merges in embeddings). It runs over the $M \ll N$ microclusters,
so it is cheap — an exploration tool, not a partition.

```python
est = betula_cluster.Betula(n_clusters=8).fit(X)
g = est.mapper(lens="density", resolution=10, gain=0.3)  # lens: density|radius|l2norm|coordinate|eccentricity

g.n_nodes, g.n_edges          # skeleton size
g.branch_points               # nodes where the shape splits (degree >= 3)
g.bridges                     # indices into g.edges whose removal disconnects the graph
g.edge_overlap                # (n_edges,) Bhattacharyya overlap in (0, 1]: a bridge across a sparse
                              # neck reads LOWER than an edge inside one dense blob — distributional,
                              # not just a shared-microcluster count

nxg = g.to_networkx()         # optional (needs networkx); edges carry weight / overlap / bridge

# sweep resolution to find the topologically stable scale (β0 / branch / bridge counts vs resolution)
curve = est.mapper_stability(resolutions=[8, 12, 16])
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

For very high-dimensional sparse data (text TF-IDF, large one-hot), the $O(\mathrm{nnz})$ sparse-native
one-shot touches only the non-zeros:

```python
from betula_cluster import fit_predict_sparse

labels = fit_predict_sparse(X, n_clusters=20, threshold=0.5)   # kmeans by default; O(nnz) per row
```

### Text: reduce and cluster in one call — `projection="svd"`

Clustering TF-IDF in its own geometry does not work, and the size of the failure is worth stating:
on 20-newsgroups the unprojected sparse path scores **ARI 0.003**. The standard fix is to reduce
first, and `projection="svd"` does it inside the same call — a CF-weighted PCA of the **leaf
summary**, so the factorization runs over `M ≈ 10³` micro-clusters rather than `N` documents.

```python
labels = fit_predict_sparse(
    X, n_clusters=20, method="spherical-kmeans",   # cosine geometry on the codes -- see below
    max_leaves=256, projection="svd", projection_dim=50,
)
```

20-newsgroups TF-IDF (18 846 × 2 000, `k`=20, rank 50, median of seeds 0/1/2, one BLAS thread):

| | ARI | time |
|---|---|---|
| sparse path, no projection | 0.003 | 8.1 s |
| `projection="svd"`, `max_leaves=256` | 0.130 | 0.30 s |
| `projection="svd"`, `max_leaves=512` | 0.144 | 0.58 s |
| `projection="svd"`, `max_leaves=2048` | 0.152 | 5.4 s |
| `TruncatedSVD(50)` + `KMeans` on the raw rows | 0.143 | 0.54 s |

Two things decide whether this works for you.

**Use a cosine head on the codes.** `method="kmeans"` on the same codes scores **0.014** against
`spherical-kmeans`'s 0.152 — an eleven-fold difference, because the leading principal direction of a
TF-IDF corpus is document length, and only an angular objective ignores it.

**The leaf budget is the cost, not the projection.** Sweeping the rank from 1 to 100 moves the total
by 1.2 s; sweeping `max_leaves` from 256 to 2048 moves it from 0.30 s to 5.4 s, because the sparse
summarizer compares each row against every micro-cluster it has so far. Buy resolution deliberately.

The basis is not a compromise for being built from a summary: labelling raw rows in it scores 0.159
against 0.143 for `TruncatedSVD`'s own basis on the same rows. Under the spherical cluster feature the
discarded within-leaf scatter is isotropic, so it shifts eigenvalues and leaves the directions alone.

Unlike `weighted-nmf`, a PCA is a linear map, so each row is labelled by **its own** code
(`(x − x̄)Vᵀ`, computed from its non-zeros) rather than by its micro-cluster's. That distinction is
worth 0.062 ARI here, and it is why the NMF projection cannot be given the same treatment: its code
is the solution of a per-row nonnegative least squares, not a matrix product.

## Hyperparameter tuning — memory-aware, dependency-free

`betula_cluster.tune` searches the CF-representation knobs (compression resolution, covariance model,
`normalize`) for the best clustering — with an internal metric, or ARI when you have labels. It is
NumPy-only; its **multi-objective** mode returns the **quality / memory / speed** Pareto front, so you
pick the point that fits your accuracy, footprint and latency budget.

```python
import numpy as np

import betula_cluster

X = np.random.default_rng(0).normal(size=(20_000, 16))

# single-objective: maximize the internal Calinski-Harabasz score, then refit with the winner
best = betula_cluster.tune(X, n_clusters=8, n_trials=40)
labels = betula_cluster.fit_predict(X, n_clusters=8, **best.best_params)

# multi-objective: the accuracy / memory / speed Pareto front
result = betula_cluster.tune(X, n_clusters=8, multi_objective=True)
for t in result.pareto:
    print(t.params, f"score={t.score:.1f} leaves={t.n_leaves} time={t.time_s:.3f}s")
```

The **Optuna** backend drops in for random search at the same trial budget — usually better trials
for the same cost. It is an optional extra (`pip install 'betula-cluster[tune]'`); the default path
above needs only NumPy.

```python
# needs: pip install 'betula-cluster[tune]'
best = betula_cluster.tune(
    X,
    n_clusters=8,
    sampler="optuna",          # TPE (single-objective) / NSGA-II (multi_objective Pareto)
    n_trials=60,
    space={                    # optional: override the default search space
        "max_leaves": ("int_log", 256, 8192),          # log-uniform integer
        "feature": ("cat", ["spherical", "diagonal", "full"]),
        "normalize": ("cat", [False, True]),
    },
)
labels = betula_cluster.fit_predict(X, n_clusters=8, **best.best_params)
```

Objectives: `"calinski_harabasz"` (default, higher better), `"davies_bouldin"` (lower better), or
`"ari"` (needs `y=`). Because betula fits are cheap, hundreds of trials stay fast — and every trial is
scored for memory (`n_leaves`) and time, not just quality.

## Consensus & stability — `consensus`

The CF-tree depends on insertion order. `consensus` clusters several random permutations of the input
and votes, so you get a robust labelling **and** a per-point stability score — low where a point sits
on an unstable boundary, high where every insertion order groups it the same way.

```python
res = betula_cluster.consensus(X, n_clusters=8, n_runs=5, method="kmeans", n_jobs=-1)  # -1 = all cores
res.labels          # (n,) consensus label per point
res.confidence      # (n,) in [0, 1] — per-point agreement across runs
res.mean_confidence # scalar robustness summary
stable = X[res.confidence == 1.0]   # points every insertion order agrees on
```

For the partitional heads (`kmeans` / `gmm` / `ward` / `spectral`) at a fixed `n_clusters`; extra
kwargs are forwarded to `fit_predict`.

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
[`.cargo/config.toml`](https://github.com/ilgrad/betula-cluster/blob/main/.cargo/config.toml).
