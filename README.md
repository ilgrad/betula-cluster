# betula-cluster

[![PyPI](https://img.shields.io/pypi/v/betula-cluster)](https://pypi.org/project/betula-cluster/)
[![Python](https://img.shields.io/pypi/pyversions/betula-cluster)](https://pypi.org/project/betula-cluster/)
[![CI](https://github.com/ilgrad/betula-cluster/actions/workflows/ci.yml/badge.svg)](https://github.com/ilgrad/betula-cluster/actions/workflows/ci.yml)
[![Python coverage 100%](https://img.shields.io/badge/python%20coverage-100%25-brightgreen.svg)](https://github.com/ilgrad/betula-cluster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ilgrad/betula-cluster/blob/main/LICENSE-MIT)
[![Rust core · PyO3](https://img.shields.io/badge/Rust%20core-PyO3-orange.svg)](https://github.com/ilgrad/betula-cluster)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21427331.svg)](https://doi.org/10.5281/zenodo.21427331)

> **Rust-powered, memory-bounded clustering for large embeddings & tabular streams.** It compresses raw
> data into numerically stable **BETULA** microclusters, then runs the clustering head on the
> *compressed* representation — k-means · GMM (diagonal & full) · Ward · **spectral** · **Leiden**
> community detection · **directional** (von Mises–Fisher / spherical) · HDBSCAN-CF · **scale-space**
> modes · Mapper — so cost scales with the microcluster count, not `N`.
> Streaming `partial_fit`, a scikit-learn API, from-scratch
> **Rust** core + **PyO3**, no LAPACK or SciPy at runtime.

```bash
pip install betula-cluster
```

**Verified:** a **457-case** Python suite at **100% wrapper coverage** + **728** Rust tests,
`clippy -D warnings` + `fmt` clean across all feature sets, CI on CPython 3.11–3.14 (one abi3 wheel).

## At a glance — honest benchmarks

Measured against scikit-learn on `StandardScaler`-normalized data, each method in its own subprocess
with peak RSS sampled from `/proc/self/statm`. Full methodology, every metric, and all tables (wins
**and** losses) live in [**`bench/RESULTS.md`**](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md).

> **Re-measured 2026-08-24 against the working tree after 0.7.0.** Every quality figure is the
> **median of seeds 0, 1, 2** — clustering
> quality is seed-dependent and a single run is not a result. Ranges per cell are in
> `bench/results_*_spread.csv`; on the synthetic sets every row moves by more than 0.05 ARI across the
> three seeds, so a margin below that is a tie and is written as one here.

- ⚡🪶 **Always faster — and lighter — at scale (the unconditional win).** betula labels **1 M points
  in 0.26 s**: 9× faster than scikit-learn KMeans, 15× vs GaussianMixture, 30× vs Birch — and
  streams **10 M in a flat ~60 MB** where an in-core KMeans needs **~5.0 GB** (**82× less**, and the
  gap grows without bound). This holds for *every* method at *every* size.
- 🎯 **Parity on the centroid heads, ahead on the structured ones.** betula's k-means ties
  scikit-learn (blobs 0.793 vs 0.794, `digits` 0.467 vs 0.468); full-covariance GMM **beats** it on
  anisotropic data (**0.961 vs 0.902**) and on real 64-D `digits` (**0.575 vs 0.463**, via the
  high-dimensional covariance floor); betula-ward clusters 1 M in 0.38 s where `O(N²)` sklearn-ward
  can't run past ~10 k; and on non-convex moons & circles the **spectral** and HDBSCAN heads hit
  **ARI 1.00**. Spectral matches `SpectralClustering`'s quality at 1.0–1.5× its speed — the durable
  claim there is *scaling* (cost set by `max_leaves`, not `N`), not a constant factor.
- 🌍 **Real data, and two losses stated plainly.** betula's diagonal GMM overtakes scikit-learn on hard
  `covtype` (**0.104 vs 0.080** at adequate leaf resolution — at the default 4 000-leaf budget the two
  are a tie inside their seed spreads) and it clusters **full covtype (581 k rows) 4.7× faster** at no
  worse ARI (0.070 vs 0.049). But `sklearn-birch` beats **every** betula
  head on both `covtype` (0.131) and MNIST (0.426 vs 0.377). On `covtype` that is a loss on the merits
  — tested in both directions, and the mechanism is the leaf budget's unequal cell *mass*; on MNIST
  Birch simply does not compress (20 000 subclusters for 20 000 points), and at equal compression the
  gap falls from 0.059 to 0.010. HDBSCAN-on-CF likewise trails raw HDBSCAN on overlapping density.
  [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md) reports
  both rather than hiding them.

| ![Fit time vs N](https://raw.githubusercontent.com/ilgrad/betula-cluster/main/bench/plots/scaling_time.png) | ![Peak memory vs N](https://raw.githubusercontent.com/ilgrad/betula-cluster/main/bench/plots/memory_streaming.png) |
|:--:|:--:|
| Phase-3 clusters only the ~2 000 leaf microclusters, not the raw points, so every head finishes 1 M points in **under 0.9 s** (k-means in 0.26 s). | The CF-tree is capped by `max_leaves`, so streaming memory stays **flat** — it clusters data larger than RAM. |

## Why

**Who it's for:** practitioners clustering large embedding or tabular datasets — in batch or as a
stream — who need bounded memory *and* the numerical stability that classic BIRCH and in-core
scikit-learn don't provide together.

Clustering libraries tend to either not scale (full GMM/HDBSCAN on raw points), lose precision
(classic BIRCH computes variance as `SS − ‖LS‖²/n`, which catastrophically cancels far from the
origin), or blow up in memory (BIRCH-family subcluster explosion in high dimensions). betula-cluster
addresses all three:

- **Numerically stable** — clustering features `(n, μ, S)` via Welford / Chan updates; the covariance
  is PSD by construction. Classic BIRCH loses all digits near coordinate `1e7`; betula does not.
- **Memory-bounded by design** — the CF-tree caps its leaves (`max_leaves`) and rebuilds, so it never
  explodes; streaming memory is flat in `N` and clusters data larger than RAM.
- **Complete** — one stable engine spanning k-means / GMM (diag & full) / Ward / spectral / Leiden
  community detection / HDBSCAN-style / Mapper, with streaming `partial_fit`, a scikit-learn API,
  and dataset-structure inspection.

The math (stable CF, the expected-log GMM E-step, distance derivations, relation to BIRCH/BETULA) is
written up — verified symbolically and numerically — in [**`docs/MATH.md`**](https://github.com/ilgrad/betula-cluster/blob/main/docs/MATH.md).

## When to use it

**Reach for betula-cluster when** the data is large or streaming, memory must stay bounded, you want
fast `predict` on new points, or you want one numerically stable engine spanning k-means / GMM / Ward /
density / topology plus dedup / outliers / representatives — especially on **embeddings and tabular
streams**.

**Use raw scikit-learn instead when** `N` fits comfortably in RAM and you want the exact point-level
algorithm with no compression: at small `N` the two-phase overhead removes the speed edge, and raw
HDBSCAN is stronger on overlapping density. betula-cluster trades a CF-compression approximation for
scale and bounded memory — if you need neither, a plain in-core clusterer is simpler.

## Installation

```bash
pip install betula-cluster            # prebuilt abi3 wheels, CPython 3.11–3.14
pip install 'betula-cluster[tune]'    # + Optuna backend for memory-aware tuning
```

NumPy is the only runtime dependency — no SciPy, LAPACK, or BLAS. Prebuilt wheels ship for Linux
(x86-64 + aarch64), macOS (Intel + Apple Silicon), and Windows (x64); one abi3 wheel covers every
supported Python. Building from source needs a Rust toolchain — `maturin develop --release` (or
`pip install .`) in a clone.

## Quick start

```python
import numpy as np
import betula_cluster

X = np.random.default_rng(0).normal(size=(100_000, 10))

labels = betula_cluster.fit_predict(X, n_clusters=10, method="kmeans")
labels = betula_cluster.fit_predict(X, n_clusters=0, feature="full", method="gmm-full")  # auto-k via BIC
labels = betula_cluster.fit_predict(X, n_clusters=8, method="spectral", threshold=0.0)   # non-convex / manifold
labels = betula_cluster.fit_predict(X, method="leiden", threshold=0.4)                    # graph communities; count auto-discovered
labels = betula_cluster.fit_predict(X, method="hdbscan", min_cluster_size=25)            # HDBSCAN-CF; -1 = noise
labels = betula_cluster.fit_predict(X, n_clusters=10, method="vmf")                       # directional / cosine (input auto-L2-normalized)
labels = betula_cluster.fit_predict(X, n_clusters=4, method="watson", feature="full")     # axial: x and -x are the same point
```

Streaming / out-of-core — feed chunks, finalize, predict; memory stays bounded by `max_leaves`:

```python
est = betula_cluster.Betula(method="gmm", memory_budget_mb=512)
for chunk in stream_of_arrays:        # each chunk is a 2-D float32/float64 array
    est.partial_fit(chunk)
est.partial_fit()                     # finalize the global clustering over everything seen
labels = est.predict(X_query)
```

Robustness — the CF-tree is insertion-order sensitive, so `consensus` clusters several random
permutations and votes, returning a consensus labelling **plus** a per-point stability score (any
partitional head — `kmeans` / `gmm` / `ward` / `spectral`):

```python
res = betula_cluster.consensus(X, n_clusters=10, n_runs=5, method="kmeans", n_jobs=-1)  # -1 = all cores
res.labels           # (n,) consensus label per point
res.confidence       # (n,) in [0, 1] — per-point agreement across runs (1.0 = every order agrees)
res.mean_confidence  # scalar robustness summary
```

Memory-aware hyperparameter tuning (`tune`, optional Optuna), Mapper topology (`mapper`),
semi-supervised constraints (COP-KMeans), mixed numeric+categorical+directional (`KPrototypes`),
streaming density (`DenStream` / `DbStream`), the `O(nnz)` sparse-native path (`fit_predict_sparse`),
CF-weighted NMF for nonnegative data (`projection="weighted-nmf"`, or `"weighted-nmf-kl"` for counts),
quantile
sketches, `scipy.sparse` input, `threshold="auto"`, soft assignment / coresets / diagnostics / drift
snapshots / active-learning batches, the Rust API, and the CLI — all in the
[**usage guide**](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).

## Capabilities

**Stable core** — production-ready:

- **Clustering heads** — weighted k-means (Hamerly), GMM (diagonal & full covariance, BIC auto-`k`),
  exact Ward HAC, **spectral** (non-convex / manifold), **Leiden** graph community detection
  (auto community count, `resolution` / CPM, optional covariance/manifold-aware affinity), and
  **directional** spherical k-means / von
  Mises–Fisher mixtures for L2-normalized embeddings (cosine geometry), all over the numerically
  stable BETULA CF-tree.
- **Streaming** — `partial_fit` at bounded memory (`max_leaves` / `memory_budget_mb`), EWMA `decay`.
- **scikit-learn API** — `fit` / `predict` / `fit_predict`, `get_params` / `set_params` (works with
  `Pipeline` / `clone` / `GridSearchCV`); typed abi3 wheel, `save` / `load` + pickle, reusable Rust core.
- **Inspection & robustness** — `predict_proba`, coresets, microcluster/cluster geometry, outliers,
  near-duplicates, representatives, diagnostics, and `consensus` (per-point stability across
  insertion-order permutations).
- **Tuning** — `tune`: memory-aware hyperparameter search with a **quality / memory / speed** Pareto
  mode; NumPy-only, optional Optuna backend (`pip install 'betula-cluster[tune]'`).

**Experimental / evolving** — useful today, API may still move:

- **Density & topology** — HDBSCAN-CF (density over microclusters), **scale-space** Morse-persistence
  density-mode clustering (`method="scale-space"` — no `k`, no bandwidth), and a Mapper topological
  skeleton (`mapper` / `mapper_stability`).
- **Naming `k` on the density hierarchy** (`method="dc-median"` / `"dc-center"`, Beer et al. 2023) —
  exact `k`-median / `k`-center in the density-connectivity ultrametric, cutting the same
  mutual-reachability tree `hdbscan` reads its count off. `dc-median` beats `spectral` by 0.198 on
  noisy moons at 20× less wall clock and ties `ward` on `digits`; `dc-center` is mass-blind by
  construction and is published with the measurement that shows it.
- **OPTICS reachability plot** (`reachability()` → `ReachabilityPlot`) — the density diagnostic, over
  the microclusters. Not a lookalike of `method="hdbscan"`: OPTICS with no ε cutoff *is* Prim on the
  mutual-reachability graph, so the sweep walks that head's own spanning tree and `labels_at(ε)` is
  its hierarchy cut at ε, exactly. Cost is set by the leaf count, not by `N` — 0.0028 s at both
  20 000 and 320 000 rows.
- **X-means** (`method="xmeans"`, Pelleg & Moore 2000) — recursive splitting instead of a `k` sweep:
  `n_clusters` is an upper bound rather than a target, and it is the only auto-`k` head that can
  return more than 20 clusters. Its split test needs a cut capturing `1 − 2^(−2/d)` of a region's
  scatter, so it lands on the true count at every `d ≥ 5` and under-splits only at `d = 2` — see
  [*Where `xmeans` refuses to split*](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).
- **k-medoids** (`method="kmedoids"`, eager FasterPAM — Schubert & Rousseeuw 2021) — the centre of a
  cluster is one of the summary's own micro-clusters rather than an average: an exemplar you can show,
  and a centre that stays on the data manifold. Exact on the summary, because
  `Σ_{x∈leaf} ‖x − μ‖² = S + n‖μ_leaf − μ‖²` makes the leaf-level objective the point-level one — note
  the *square*, since classical PAM's absolute distance has no closed form in a cluster feature. On
  `digits` at one leaf per point it reads ARI **0.554** [0.554–0.570] against `kmeans`'s 0.467
  [0.443–0.571] over seeds 0–4, and loses at a coarse summary (115 leaves: 0.219 vs 0.240) where there
  are too few candidate centres; the `O(m²)` swap pass costs 6× `kmeans` at 1797 leaves. `n_clusters=0`
  switches objective to the medoid silhouette, because total deviation is monotone in `k` —
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).
- **Fuzzy c-means** (`method="fuzzy-cmeans"`, Bezdek 1981) — the only soft head that fits no density:
  it publishes a partition of unity `u_j ∝ d_j^(−1/(m−1))` over the centres, exact on the summary by
  the same identity `kmedoids` uses, with Xie–Beni as its automatic `k`. Take it for the membership,
  not for the labels — the hard partition is a **loss** against `kmeans` on every fixture measured
  (`blobs` 0.847 vs 0.864, `aniso` 0.535 vs 0.545, `digits` 0.483 vs 0.467 at `m=1.3`), and the loss
  grows with cluster overlap and with `m`, because no membership is ever zero and every centre is
  pulled toward the grand mean —
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).
- **Structured-covariance GMM** — a three-rung Toeplitz ladder: **`method="gmm-toeplitz"`** (banded AR),
  **`"gmm-toeplitz-full"`** (general positive-definite Toeplitz covariance), and **`"gmm-toeplitz-gs"`**
  (full-order Gohberg–Semencul **MLE** precision): covariance-*shape* clustering for **ordered, stationary
  signals** (time-series windows, trajectories, sensor waveforms), well-posed where full covariance is
  singular (`N_k ≪ d`) and a diagonal model ignores neighbour correlation; the `-full` head captures
  structure beyond a low-order AR (e.g. a long-lag echo), the `-gs` head fits a likelihood-optimal
  precision with a cheaper E-step than full at large `d`.
- **Subspace GMM** — **`method="mppca"`**, a mixture of probabilistic PCA (Tipping & Bishop 1999):
  each component covariance is `W Wᵀ + σ²I` of rank `rank`, so it carries orientation at `O(d·rank)`
  per component instead of `O(d²)` and runs at `d = 784` where the full head cannot. At one leaf per
  point on `digits` it beats both the diagonal and the full head (ARI 0.600 vs 0.461 / 0.575); on a
  coarse summary it loses to the diagonal head, and
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md) measures why.
- **Factor-analyser GMM** — **`method="mfa"`** (Ghahramani & Hinton 1996), the same subspace model
  with `mppca`'s single `σ²` relaxed to a per-dimension `diag(ψ)`, for tables whose columns are in
  different units and cannot be standardised. The two heads dissociate in both directions on
  controlled fixtures — `mfa` reads a quiet axis that a loud nuisance pair drowns (ARI **1.00** vs
  `mppca`'s 0.04–0.34), `mppca` reads lines that differ only in orientation (**1.00** vs ≈ 0.00) —
  but on real tables already on a common scale `mppca` wins every row measured (`digits` 0.738 vs
  0.562, MNIST-20k 0.365 vs 0.277). What `mfa` buys is a floor, not a ceiling: it contains the
  diagonal Gaussian mixture at `rank=0` and falls back onto it, where `mppca` can land at less than
  half its score. Not rotation-equivariant, for the same reason `gmm` is not.
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md) has the tables.
- **Axial mixture** — **`method="watson"`** (Watson 1965), for directional data whose **sign is
  arbitrary**: eigenvectors, SVD/PCA axes, line orientations, any feature where `x` and `−x` are the
  same observation. `p(x) ∝ exp(κ (μᵀx)²)` is antipodally symmetric, so where `vmf` spends half its
  components on the antipodes of the other half, this reads one axis. Its sufficient statistic is the
  second moment `Σ_i + μ_i μ_iᵀ`, which the `full` leaf carries exactly — the E-step is closed form,
  not an approximation. On a 32-D four-axis fixture with half of each cluster at each pole it scores
  ARI **0.953** against `vmf`'s 0.218 and `gmm-full`'s 0.316, and its BIC finds `k = 4` where `vmf`
  answers 8; on `digits`-PCA20 with a random half of the rows sign-flipped it moves 0.498 → **0.503**
  while every other head loses 60–70 %. The trade is explicit: where the sign *does* carry information
  it loses to `vmf` (0.952 vs 0.976), and it costs 5–8× `vmf` in wall clock. `κ < 0` fits **girdle**
  (equatorial) components too, and `n_clusters=0` selects `k` by BIC —
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).
- **Hyperbolic embeddings** — **`method="hyperbolic"`** (Law et al., ICML 2019) for data that is
  already a point set of `H^d`: Poincaré or Lorentz coordinates of a taxonomy, ontology or scale-free
  graph. It clusters under the **squared Lorentzian distance** `d_L² = −2 − 2⟨x,y⟩_L`, whose centroid
  is the normalised sum `R/|R|_L` — and which is *affine*, so a leaf enters only through `(n_i, R_i)`
  and its covariance is not read at all. That makes `feature="spherical"` exactly as good as `"full"`
  (measured: identical to four digits), which is true of no other head here. The deliverable is
  **invariance, not ARI**: a Lorentz boost is an isometry of `H^d`, and on a 15 360-point tree
  embedding boosted by rapidity 3 the Poincaré-ball route with `gmm-full` falls 0.817 → **0.311**
  while this head holds 0.731 → **0.596**. On an *un-boosted* embedding that ball route wins outright,
  and the residual drift here is the Euclidean CF-tree's, not the head's — at one leaf per point the
  head is exactly boost-invariant (0.772 at every rapidity). The trade, and the `f64` working radius
  of ≈ 18, are in
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).
- **More heads & data** — `DenStream` / `DbStream` evolving-stream density, mergeable `KllSketch` /
  `DdSketch` quantiles, `scipy.sparse` (`O(nnz)`, never densified), mixed
  numeric+categorical+directional (`KPrototypes`), COP-KMeans constraints, robust (Huber) insertion,
  drift snapshots, dependency-free CLI.

Full reference: [**`docs/FEATURES.md`**](https://github.com/ilgrad/betula-cluster/blob/main/docs/FEATURES.md).

## Examples

**Seventeen** executed, plotted notebooks — one per capability — live in
[`examples/`](https://github.com/ilgrad/betula-cluster/blob/main/examples/README.md) (render on GitHub):

- **Core** — [quickstart](https://github.com/ilgrad/betula-cluster/blob/main/examples/01_quickstart.ipynb),
  [embeddings & inspection](https://github.com/ilgrad/betula-cluster/blob/main/examples/02_embeddings_and_inspection.ipynb),
  [streaming & persistence](https://github.com/ilgrad/betula-cluster/blob/main/examples/03_streaming_and_persistence.ipynb),
  [method comparison](https://github.com/ilgrad/betula-cluster/blob/main/examples/04_method_comparison.ipynb),
  [Mapper topology](https://github.com/ilgrad/betula-cluster/blob/main/examples/05_topology_mapper.ipynb).
- **Streaming density** — [`DenStream` & `DbStream`](https://github.com/ilgrad/betula-cluster/blob/main/examples/06_streaming_density.ipynb).
- **Mixed data** — [`KPrototypes`](https://github.com/ilgrad/betula-cluster/blob/main/examples/07_mixed_data_kprototypes.ipynb).
- **Sketches** — [`KllSketch` & `DdSketch`](https://github.com/ilgrad/betula-cluster/blob/main/examples/08_quantile_sketches.ipynb).
- **Semi-supervised** — [must-link / cannot-link](https://github.com/ilgrad/betula-cluster/blob/main/examples/09_semisupervised_constraints.ipynb).
- **Sparse / high-dim** — [`scipy.sparse` + `fit_predict_sparse`](https://github.com/ilgrad/betula-cluster/blob/main/examples/10_sparse_highdim.ipynb).
- **Soft assignment & coresets** —
  [`predict_proba`, coresets, diagnostics](https://github.com/ilgrad/betula-cluster/blob/main/examples/11_soft_assignment_coreset_diagnostics.ipynb).
- **Production ops** — [drift, active learning, robust, memory budgets](https://github.com/ilgrad/betula-cluster/blob/main/examples/12_drift_robust_memory.ipynb).
- **Graph & geometry** — [graph clustering (Leiden)](https://github.com/ilgrad/betula-cluster/blob/main/examples/13_graph_clustering.ipynb),
  [directional embeddings (vMF / spherical)](https://github.com/ilgrad/betula-cluster/blob/main/examples/14_directional_embeddings.ipynb),
  [geometry-aware clustering (covariance / manifold)](https://github.com/ilgrad/betula-cluster/blob/main/examples/15_geometry_aware_clustering.ipynb).
- **Time-series** — [`gmm-toeplitz` AR/Toeplitz covariance for stationary signals](https://github.com/ilgrad/betula-cluster/blob/main/examples/16_toeplitz_timeseries.ipynb).
- **Nonnegative data** — [`projection="weighted-nmf"` CF-weighted NMF on topic counts](https://github.com/ilgrad/betula-cluster/blob/main/examples/17_nmf_topics.ipynb).

And six **end-to-end use cases** (each scored against ground truth):

- 🧹 [**Embedding dedup**](https://github.com/ilgrad/betula-cluster/blob/main/examples/usecases/usecase_01_embedding_dedup.ipynb) — collapse a repost-heavy corpus to representatives.
- 🚨 [**Log anomaly detection**](https://github.com/ilgrad/betula-cluster/blob/main/examples/usecases/usecase_02_log_anomaly_detection.ipynb) — batch outlier scoring + streaming `DbStream` flags.
- 👥 [**Customer segmentation**](https://github.com/ilgrad/betula-cluster/blob/main/examples/usecases/usecase_03_customer_segmentation.ipynb) — mixed RFM + categorical personas with `KPrototypes`.
- 🧠 [**RAG corpus curation**](https://github.com/ilgrad/betula-cluster/blob/main/examples/usecases/usecase_04_rag_corpus_curation.ipynb) — junk removal, topic coherence, and topic-leakage detection via Mapper.
- 🔢 [**Real-data clustering**](https://github.com/ilgrad/betula-cluster/blob/main/examples/usecases/usecase_05_real_data_clustering.ipynb) — handwritten digits, ARI parity + centroid/exemplar inspection.
- 🌐 [**Graph communities**](https://github.com/ilgrad/betula-cluster/blob/main/examples/usecases/usecase_06_graph_communities.ipynb) — Leiden community detection on a network, scored against planted communities.

## Documentation

- [**Usage guide**](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md) — runnable snippets for every interface.
- [**Features**](https://github.com/ilgrad/betula-cluster/blob/main/docs/FEATURES.md) — full capability reference + crate architecture.
- [**Math**](https://github.com/ilgrad/betula-cluster/blob/main/docs/MATH.md) — stable CF, GMM E-step, distance derivations, relation to BIRCH/BETULA.
- [**Benchmarks**](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md) — methodology, every metric, all tables, honest wins & losses.
- [**Design**](https://github.com/ilgrad/betula-cluster/blob/main/DESIGN.md) — internal design, invariants, and testing strategy.

Verified: **713** Rust unit + 15 integration tests (plus 8 for the CLI binary) + a **457-case**
Python suite at **100%** wrapper coverage (Rust ≥95%, CI-enforced), `clippy -D warnings` + `fmt`
clean across all feature sets, on Python 3.11–3.14 (single abi3 wheel).

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
5. **The expected-log GMM optimizes a CF-level objective**, not pointwise EM — a deliberate choice,
   and the *exact* variational bound on the pointwise objective rather than an ad-hoc surrogate.
   Because `log N(x | μ, Σ)` is quadratic in `x`, the leaf expectation is computed exactly from the
   cluster feature `(n, μ, S)` with no error from the within-leaf shape; the E-step is therefore exact
   variational EM in which the responsibilities are **tied** within each leaf, and that tying is the
   whole of the approximation. The paper's convolution variant computes `log E[p] ≥ E[log p]`, a
   different quantity and not the EM bound at all — which is why it degrades on coarse summaries where
   this one does not
   ([`docs/MATH.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/MATH.md)).
6. **Frequent-Directions is an approximate low-rank covariance** (exact only up to its rank `ℓ`).
7. **A diagonal Gaussian is a weak model for raw image pixels.** Since 0.6.0 the mixture heads label a
   point by its own posterior under the fitted mixture, which is the model's own rule — but on raw
   pixels the model itself is the limit: `gmm` scores ARI 0.185 on MNIST-20k at the default leaf budget
   where a nearest-centre rule scores 0.378, because a diagonal covariance sums 784 independent
   per-dimension penalties for 784 correlated pixels. At a coarser budget (`max_leaves=300`) the
   posterior wins for both GMM heads. For raw images prefer `kmeans` or a `projection`; the full
   derivation is in [`docs/MATH.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/MATH.md).

## How to cite

If betula-cluster supports your research, please cite **the software** and **the underlying
algorithms** it implements. Machine-readable metadata (including the method references) lives in
[`CITATION.cff`](https://github.com/ilgrad/betula-cluster/blob/main/CITATION.cff) — GitHub's
*"Cite this repository"* renders it directly.

```bibtex
@software{gradina_betula_cluster,
  author  = {Gradina, Ilia},
  title   = {betula-cluster: numerically stable {BETULA} clustering with a {Rust} core},
  year    = {2026},
  version = {0.7.0},
  doi     = {10.5281/zenodo.21427331},
  license = {MIT},
  url     = {https://github.com/ilgrad/betula-cluster}
}
```

betula-cluster is an independent implementation (with extensions); the algorithms are due to
**BETULA** — Lang & Schubert, *Information Systems* (2022),
[doi:10.1016/j.is.2021.101918](https://doi.org/10.1016/j.is.2021.101918) — building on **BIRCH** —
Zhang, Ramakrishnan & Livny, *SIGMOD* (1996),
[doi:10.1145/233269.233324](https://doi.org/10.1145/233269.233324).

## License

MIT © Ilia Gradina
