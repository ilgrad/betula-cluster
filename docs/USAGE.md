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

Keyword args: `feature ∈ {spherical, diagonal, full, fd}`, `method ∈ {kmeans, gmm, gmm-full, mppca, ward, average, weighted, centroid, median, spectral, leiden, leiden-cpm, spherical-kmeans, vmf, gmm-toeplitz, gmm-toeplitz-full, gmm-toeplitz-gs, hdbscan, scale-space}`,
`distance ∈ {euclidean, manhattan, ward, average}` (routing measure),
`absorb ∈ {euclidean, manhattan, average, diameter, ward, radius, chi2, subspace}` (see *Absorption criteria*
below; `chi2` = mass-invariant Mahalanobis gate at level `chi2_p` with `chi2_scale` = within-cluster
variance; fixes the BIRCH size-imbalance bug), `decay` (EWMA factor
for streaming concept drift), `normalize` (L2-normalize rows → cluster by *direction*; on the unit
sphere squared-Euclidean is monotone in cosine, so the tree clusters by angle. It earns its keep on
`digits`-64 (k-means **0.467 → 0.569**, ward **0.643 → 0.699**, median of three seeds); on MNIST-784
it is now a wash — 0.307 → 0.346, inside the seed spread and sign-flipping between seeds, since the
tree-rebuild fix removed most of the Euclidean collapse it used to compensate for. Leave it off for
tabular data where magnitude is signal: it takes covtype ward to **−0.049**, worse than random),
`n_jobs` (parallel shard+merge tree build — `>1` gives ~4–5× on large
`N`), `threshold`, `branching`, `leaf_cap`, `max_leaves` (an integer is an absolute leaf cap; a
float in `(0, 1)` is a **fraction of the row count**, resolved as `ceil(frac·N)` at `fit` time —
ELKI's `-cftree.maxleaves` convention, whose own default is `0.05`. A fraction is undefined for
`partial_fit`, which never sees a final `N`, and raises there rather than guessing a batch size;
`memory_budget_mb` overrides either form, being the harder constraint), `max_iter`, `min_samples`
(for `method="hdbscan"`, the core-distance neighbourhood **counting the point itself** —
the convention of Campello's Def. 3.1, `sklearn.cluster.HDBSCAN` and ELKI, so `min_samples=1`
leaves every core distance at 0 and HDBSCAN\* degenerates to single linkage;
`scikit-learn-contrib/hdbscan` excludes it, where the same number means one neighbour more),
`min_cluster_size`, `graph_degree` (for `method="hdbscan"`, the out-degree of the proximity graph the
density head runs on; `0` = the exact complete graph, a positive value is a **floor** the head raises
to whatever `min_samples` needs — see *Graph-indexing the density head* below), `resolution` (Leiden γ — granularity for `method="leiden"` / `"leiden-cpm"`, higher
⇒ more communities), `covariance_weight` (Leiden β — a log-Euclidean covariance/shape term in the
affinity, `feature="full"`; `0` = off, the centroid-only default), `tangent_weight` / `tangent_rank`
(Leiden γ — a Grassmann tangent-subspace term of rank `tangent_rank` for manifold-aware communities,
`feature="full"`; `0` = off), `rank` (MPPCA subspace rank `q` for `method="mppca"`, clamped to at
most `dim - 1`; `0` makes every component spherical), `projection` / `projection_dim` / `projection_max_iter` (reduce the leaf centroids to
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
| `subspace` | — | the same gate read on the leaf's own low-rank basis | χ²`dim` quantile via `chi2_p` |

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

**`subspace` reads the same χ² gate on the leaf's own basis, and only `feature="fd"` has one.** Every
other feature model falls back to `chi2`, so the option changes nothing unless you asked for the
Frequent-Directions sketch. It takes the same `chi2_p` and `chi2_scale`, in the same units.

Use it when your clusters differ in *orientation* more than in *location*. On a fixture where six
rank-5 subspaces share a single centre — so centroid distance carries no information at all — leaf
purity goes **0.820 → 0.938** (median of seeds 0/1/2, `max_leaves=2000`, `chi2_scale=0.01`, ranges
disjoint), and on well-separated blobs it reaches the same ARI 1.0 with **6 leaves instead of 99**.

**On MNIST-20k it is a loss**, and that is the case to weigh it against: ARI 0.250–0.260 against
`chi2`'s 0.274–0.291 at every scale tried, with more leaves and ~20 % more time (the gate costs
`O(ℓ²d)` per decision against `O(d)`, which shows at `d=784` and not at `d=100`). Real image data at
leaf scale did not have the structure the gate is built to find.

One caveat worth stating because it bounds what the option can currently buy: on that concentric
fixture *both* gates score ARI ≈ 0.05 while purity is 0.82–0.96. Every head here assigns by centroid,
so a better-oriented summary has nothing to consume it — `subspace` improves the tree, not yet the
answer.

### Choosing a head

| your data / goal | `method` | needs `k`? |
|---|---|---|
| compact/spherical groups, fastest | `kmeans` | yes |
| elliptical / correlated / anisotropic, soft assignment | `gmm` (diag) or `gmm-full` | yes (or `0` = BIC) |
| clusters on **low-dimensional subspaces**, `d` too large for `gmm-full` | `mppca` + `feature="fd"`, `rank` = the intrinsic dimension — read *`rank`, and where `mppca` loses* first | yes (or `0` = BIC) |
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

### `rank`, and where `mppca` loses — `method="mppca"`

`mppca` constrains each component covariance to `W_c W_cᵀ + σ_c² I` with `W_c` of rank `rank`: a
`rank`-dimensional principal subspace plus isotropic noise. It buys `gmm-full`'s orientation at
`O(d·rank)` per component instead of `O(d²)`, which is what makes it usable at `d = 784` where the
full head's per-leaf dense scatters need ~38 GB and simply do not run. Pair it with `feature="fd"`,
whose leaf scatter is already low-rank, and the E-step never forms a `d×d` matrix either.

**`rank` is the intrinsic dimension of a cluster, and the fit finds it.** Six 5-dimensional
subspaces sharing one centre in 100-D — where every centroid coincides and orientation is the only
signal — `max_leaves=2000`, median of seeds 0/1/2:

| `rank` | 2 | 3 | **5** | 10 | 20 | `gmm` (diag) |
|---|---|---|---|---|---|---|
| ARI | 0.385 | 0.654 | **0.998** | 0.823 | 0.727 | 0.166 |

The peak is exactly at the true rank, and the band at `rank=5` is [0.9976, 0.9984] — this is not a
lucky seed. Overshooting costs less than undershooting *here*; on a compressed summary it costs much
more, which is the next paragraph. Where the centroids are far enough apart to separate the clusters
on their own, the extra parameters cost nothing: on the same six subspaces pulled apart, `gmm` and
`mppca` both score 1.0000 at every rank from 2 to 20.

**The trade is against compression, not against dimension.** The expected-log E-step folds each
leaf's own scatter into the component covariance as `−½ tr(Σ_c⁻¹ Σ_i)`. That within-leaf scatter is
locally oriented and adds up to a term that carries almost none of the *between-cluster*
orientation — so the more orientation a head models, the more the summary costs it. Measured on
`digits` (1797×64, `feature="fd"`, median of seeds 0/1/2), where `max_leaves=2000` gives one leaf per
point and the correction is exactly zero:

| `max_leaves` | leaves | `gmm` (diag) | `mppca` `rank=5` | `mppca` `rank=10` | `gmm-full` |
|---|---|---|---|---|---|
| 2000 | 1797 (= n) | 0.461 | **0.600** | 0.555 | 0.575 |
| 300 | 296 | **0.493** | 0.406 | 0.348 | 0.273 |
| 120 | 115 | **0.235** | 0.168 | 0.121 | 0.099 |

At full resolution `mppca` beats both the diagonal head *and* the full head at a fraction of the
parameters. At 6:1 compression the ordering inverts, and it inverts in exact order of how much
orientation each head carries. On MNIST-20k (784-D, `max_leaves=2000` ⇒ 1880 leaves, 10.6:1) that
puts `mppca` behind the diagonal head at every rank tried — ARI **0.159 / 0.069 / 0.024** for
`rank` 2 / 5 / 10 against `gmm`'s **0.274** — and the loss grows with rank, as the mechanism
predicts. Use `mppca` when the summary is fine relative to the clusters; use `gmm` when it is coarse.

### `min_samples` on a summary — `hdbscan`

`min_samples` and `min_cluster_size` are counted in **points**, not in leaves: a leaf contributes its
whole weight, so both arguments mean the same thing whether the head sees one feature per point or a
summary of a million of them. Transfer them from `sklearn.cluster.HDBSCAN` unchanged.

What does *not* transfer is a *small* `min_samples`. HDBSCAN\* separates overlapping densities
through the core distance — the radius enclosing `min_samples` points. A single leaf already holds
`N / max_leaves` points at one coordinate, so any `min_samples` below that leaf mass is enclosed at
radius zero, every core distance collapses, and mutual reachability degenerates to plain distance,
i.e. single linkage, which chains through overlaps. Measured on six overlapping 2-D Gaussians,
N = 100 000, `min_cluster_size = 250` (ARI, clusters found):

| `min_samples` | 10 | 100 | 1 000 |
|---|---:|---:|---:|
| `max_leaves = 2 000` (leaf mass 50) | 0.478 (3) | 0.566 (4) | 0.785 (5) |
| `max_leaves = 8 000` (leaf mass 12) | 0.566 (4) | 0.799 (5) | **0.843** (6) |

So set `min_samples` comfortably above `N / max_leaves`, or raise `max_leaves` until the leaf mass
falls below the `min_samples` you want. On well-separated clusters neither matters; on overlapping
ones it is the difference between finding three clusters and finding six.

### Graph-indexing the density head — `graph_degree`

The exact head is quadratic in the leaf count twice over: a full sort per leaf for the core
distances, then Prim over the complete mutual-reachability graph. That is what makes a large
`max_leaves` unaffordable exactly where the section above says a density head needs one.

`graph_degree > 0` replaces both with the **two-pass** construction of Okkels et al. (Inf. Syst. 142
(2026) 102768, Alg. 4): build a bounded-degree approximate k-NN graph over the leaf means, read the
core distances off *that* graph, take an exact MST of it. The graph is flat — no HNSW layer stack,
following Thordsen & Schubert (SISAP 2025), who find the hierarchy buys little in high dimension and
that a *capped* beam search is the part worth keeping — with three uniformly random out-edges per
vertex standing in for the long edges the upper layers would have contributed.

**The number is a floor, not a ceiling.** Core distances read off a graph saturate at the farthest
neighbour, so a degree below what `min_samples` needs *underestimates* every core distance with no
bound on the error. The head therefore raises the requested degree to `min_samples / mean leaf mass`
whenever that is larger; `graph_degree=1` is a request for the cheapest legal graph, not a broken one.

Median of seeds 0/1/2, one BLAS thread, `min_cluster_size = N/100`, `min_samples = 4N/max_leaves`
(the rule the section above argues for). "head" is the time after subtracting the identical tree
build, which is what the parameter changes:

| dataset | `max_leaves` | exact head | ARI | `graph_degree=16` | ARI | `graph_degree=32` | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| blobs 100 k, 2-D | 2 000 | 0.11 s | 0.5645 | 0.02 s | 0.5596 | 0.04 s | 0.5634 |
| | 8 000 | 2.04 s | 0.5668 | 0.11 s | 0.5530 | 0.21 s | 0.5640 |
| | 32 000 | 36.6 s | 0.5674 | 0.52 s | 0.5454 | **0.98 s** | **0.5608** |
| covtype 581 k, 54-D | 2 000 | 0.41 s | 0.0531 | 0.20 s | 0.0496 | 0.24 s | 0.0519 |
| | 8 000 | 4.61 s | 0.0490 | 0.28 s | 0.0374 | 0.45 s | 0.0457 |
| MNIST 70 k, 784-D | 2 000 | 3.00 s | 0.0298 | 0.13 s | **0.0298** | 0.28 s | **0.0298** |
| | 8 000 | 52.0 s | 0.0523 | < 0.5 s | **0.0523** | **0.45 s** | **0.0523** |

At 32 000 leaves on blobs the head goes from 36.6 s to 1.0 s — **37×** — for 1.2% of the ARI. On
MNIST at 8 000 leaves it goes from 52.0 s to 0.45 s — **116×** — for *no* ARI at all: the graph
reproduces the exact partition to four decimals with the same ten clusters. The trade is monotone in
the degree and it is the *degree*, not the graph, that costs the quality: doubling 16 to 32 recovers
most of the loss at half the saving. Below ~2 000 leaves the exact path is already cheap and there is
nothing to buy.

**Degree 8 is not enough in high dimension**, whatever the leaf budget: on MNIST it gives ARI 0.0190
with 4 clusters against the exact 0.0523 with 10, and swings across [0.0068, 0.0600] between seeds.
16 is the smallest degree measured to be lossless at `d = 784`.

**What it does not fix.** The approximation is in which edges the MST may choose from, never in the
criterion; where the head is weak on the exact graph (covtype: ARI 0.05 at any budget) it stays
weak on the approximate one. `graph_degree` buys leaves, not quality.

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

### When more leaves buy nothing — check where the budget went

A budget can be fully spent and still spent badly. The absorption radius is one global number, so a
region that is dense relative to it collapses into a single leaf while sparse regions keep splitting:
the tree fills 90–98 % of `max_leaves` and puts most of the *mass* in a handful of them. The symptom
is that raising `max_leaves` does not move the score at all.

The diagnostic costs one line, and the heaviest leaf's share of the mass is the number to read:

```python
w = np.asarray(est.microcluster_weights_)
print(w.max() / w.sum())          # ≈ 1/n_leaves is healthy; 0.5+ means one leaf holds half your data
```

If it is large, set **`balance`** — a per-leaf cap of that many times the `n / max_leaves` ideal:

```python
est = betula_cluster.Betula(n_clusters=7, max_leaves=1000, balance=4.0)
```

`max_leaves` stays a hard bound; the cap is best-effort and yields to it. On a fixture with 80 % of
the mass in one tight core this moves `kmeans` from ARI 0.4174 to **1.0000** at every budget from 250
to 4000 — but it is a lever, not a free win, so measure it against `balance=None` on your own data.

## Soft assignment, coresets, diagnostics, drift

All over the microclusters the tree already holds (no extra data passes):

```python
proba = est.predict_proba(X_query)            # (n, k): the point's own mixture posterior (argmax == predict); centroid-softmax heuristic for the non-generative heads
conf  = est.assignment_confidence(X_query)    # (n,) in [0, 1] — low flags boundary / ambiguous points
coreset = est.export_coreset()                # coreset.centers / .weights / .radii — fit any weighted model on these
coreset = est.export_coreset(size=500, k=8)   # …or a (k, eps)-coreset of 500 leaves; see below
report  = est.diagnostics()                   # compression_ratio, radius p50/p90/p99, cluster mass spread, n_rebuilds
reps    = est.representatives(X_query, cluster_id=0, method="medoid")   # or "boundary" / "outlier" / "diverse"
profile = est.cluster_profile(0)              # JSON-able geometry + nearest clusters (e.g. to LLM-name a cluster)
batch   = est.active_learning_batch(X_query, n=100, strategy="uncertain")  # rows to review/label

snap = est.snapshot()                         # cluster geometry now; later, detect drift:
drift = betula_cluster.Betula.compare_snapshots(snap, est_next.snapshot())  # matched clusters: centroid shifts / mass ratios
```

### Internal validity — `validity()`

```python
est.validity()
# {'calinski_harabasz': 8143.2, 'davies_bouldin': 0.41, 'medoid_silhouette': 0.93}
```

Three indices off the leaf summary, all in $O(\ell k d)$ — there is no second pass over the data
and no $O(N^2)$ term, because the sum of squared distances inside a leaf is
$S_i + n_i\lVert\mu_i - c\rVert^2$ exactly. On a fine tree (`threshold=0` with a leaf budget above
$N$) `calinski_harabasz` reproduces scikit-learn's point-level `calinski_harabasz_score` to
floating-point noise; the test suite asserts it.

Read the caveats before selecting `k` with any of them:

| index | direction | status on cluster features |
|---|---|---|
| `calinski_harabasz` | higher is better | **exact**; undefined at `k = 1` |
| `davies_bouldin` | lower is better | the **RMS**-dispersion variant, $\sigma_j=\sqrt{E\lVert x-c_j\rVert^2}$ — the classical mean-distance form is not a function of a cluster feature at all |
| `medoid_silhouette` | higher is better, ≤ 1 | the index **of the summary**: a per-leaf ratio weighted by leaf mass, which converges to the point-level value only as the leaves shrink |

**None of the three can say "there is no structure here."** Schubert, *Stop using the elbow
criterion for k-means* (SIGKDD Explorations 25(1), 2023), Table 1 shows the distance-based indices
reporting 3–22 clusters in pure noise where BIC correctly reports one. Calinski–Harabasz is
undefined at `k = 1`, which is the same limitation stated honestly. For the "is there anything here
at all" question, fit with `n_clusters=0` on a mixture head and let BIC answer — that path is
unchanged and is the authority.

`method="ward"` with `n_clusters=0` now cuts the dendrogram at the best Calinski–Harabasz score
rather than at the largest relative jump in merge height. The old rule was the elbow criterion in a
dendrogram's clothing, and it fails exactly where the paper says it does: on two far groups of two
nearby subclusters each, the tallest relative jump is the one that joins the far groups, so it
reported `k = 2` on every seed where the variance ratio reports the true 4.

### Why is my tree collapsing? — `tree_report()`

```python
est.tree_report()
# {'n_leaves': 241, 'max_leaves': 250, 'fill': 0.964, 'threshold': 1.681,
#  'heaviest_leaf_mass_fraction': 0.800, 'heaviest_leaf_width': 0.56,
#  'leaf_mass_quantiles': {50: 3.0, 90: 41.0, 99: 512.0, 100: 80000.0},
#  'diagnosis': ['the leaf budget is 96% spent and one leaf holds 80% of the mass: …']}
```

`fill` and `heaviest_leaf_mass_fraction` locate the size-imbalance pathology of scikit-learn's Birch
issue [#22854](https://github.com/scikit-learn/scikit-learn/issues/22854) — a spent budget with the
mass in one leaf means the tree resolved the sparse part of the data and merged the dense part.
`heaviest_leaf_width` (that leaf's RMS radius over the median leaf's) is what says whether it cost
anything: a dense region that really is point-like is summarized faithfully by one *tight* leaf,
while a heavy leaf as wide as a typical one is a merged region and whatever was inside it is gone.

Pass the data for an A-BIRCH threshold estimate beside the threshold in use:

```python
est.tree_report(X)["suggested_threshold"]     # gap-statistic estimate from a sample
betula_cluster.estimate_threshold(X)          # …or on its own, without a fitted tree
# ThresholdEstimate(threshold=1.74, n_clusters=4, radius_ratio=1.08, separation=11.8, assumptions=[])
```

**Advisory only.** `max_leaves` is the knob that binds — the threshold is what the rebuild derives
from it — and the estimate assumes well-separated, near-spherical clusters of comparable size.
`assumptions` names each of those the data breaks rather than leaving you to guess; a non-empty list
does not make the number useless, it makes it a hint. What it is genuinely good for is the
comparison: a tree that settled *above twice* the sampled estimate is absorbing points from more
than one cluster, and `tree_report(X)` says so in `diagnosis`.

## A coreset with a guarantee — `export_coreset(size=…)`

`export_coreset()` with no arguments is the streaming summary it always was: every leaf, at its own
mass, in one `O(n_leaves)` pass. Passing a `size` subsamples it by **sensitivity sampling**
(Feldman & Langberg, STOC 2011) and turns the word *coreset* into a claim: every candidate solution
scores within `(1 ± ε)` of its score on the full summary, not just the one this estimator fitted.

The error is two independent halves, and the API keeps them apart because they fail differently.

**Summarization** — present in both modes. With `Δ = coreset.offset = Σᵢ Sᵢ`, the summary's cost
`ĉost(C) = Σᵢ (Sᵢ + nᵢ‖μᵢ − C‖²)` is *exactly* the cost of sending every point of a leaf to the
centre nearest that leaf's centroid, so it can only over-charge, and by a bounded amount:

$$0 \le \hat{c}(C) - c(C) \le 4\sqrt{\Delta \cdot c(C)} + 4\Delta \qquad \text{for every } C, \text{ every } k$$

That is a relative error of `4√ρ + 4ρ` at `ρ = Δ / c(C)`, and `c(C) ≥ OPT_k` bounds it uniformly.
`Δ` is known exactly; `OPT_k` is not, so `coreset.summary_epsilon(alpha)` makes you name the
approximation factor you assume rather than picking one for you — `reference_cost` upper-bounds
`OPT_k`, so `summary_epsilon(1.0)` is optimistic, not a certificate.

**Sampling** — only when `size` is given. Since `ĉost(C) = Δ + Σᵢ nᵢ‖μᵢ − C‖²` and `Δ` does not
depend on `C`, the sample only has to be a coreset of the weighted set `{(μᵢ, nᵢ)}`; `offset`
carries the constant instead of losing it, and `coreset.cost(centers)` adds it back so you cannot
forget. Sensitivity sampling attains the optimal worst-case size `Õ(k·ε⁻²·min(√k, ε⁻²))` — matching
the STOC 2022 lower bound — and `Õ(k/ε²)` on stable instances (arXiv 2405.01339).

```python
cs = est.export_coreset(size=500, k=8)
cs.centers.shape          # (<= 500, d)
cs.cost(candidate_centers)  # weighted cost + offset
cs.summary_epsilon(1.0)   # optimistic; pass the alpha you can defend
cs.total_sensitivity      # 10 + 4k when the reference solution left no cluster empty
```

A `size` at or above the leaf count returns every leaf exactly, with no sampling error — not a
noisy redraw of something already held exactly.

## The other four linkages — `average` / `weighted` / `centroid` / `median`

`method="ward"` is the nearest-neighbour chain, which is only valid for a *reducible* linkage.
The other four run on Anderberg's algorithm and take the same `n_clusters` (and `n_clusters=0`
for a Calinski–Harabasz-scored cut) as every other partitional head. Names follow SciPy's
`scipy.cluster.hierarchy.linkage(method=…)`:

| `method` | classical name | what it measures between two clusters | children weighted by |
|---|---|---|---|
| `average` | UPGMA | mean squared distance over all cross-cluster point pairs | mass |
| `weighted` | WPGMA (McQuitty) | the same, with the two children counted equally | 1 each |
| `centroid` | UPGMC | squared distance between mass-weighted centroids | mass |
| `median` | WPGMC | squared distance between dyadic midpoints | 1 each |
| `ward` | Ward | `2·n_a n_b/(n_a+n_b)·‖Δμ‖²` | mass |

All five are on **squared** distances, and on single-point leaves all five reduce to the plain
squared distance between the two points — that is what the factor two on Ward is for.

Three of them are exactly the CF distances the tree already routes by: `average` is `D2²`,
`centroid` is `D0²`, `ward` is `2·D4²`. `weighted` and `median` are not, and cannot be: a cluster
feature merge is mass-weighted by construction, so nothing built out of cluster features can
represent a cluster whose children were combined equally regardless of size. They are driven by a
per-cluster `(mean, mean squared radius)` pair instead, updated by the König–Huygens recurrence in
its all-positive form — no `Σα‖μ‖² − ‖m‖²`, so no cancellation far from the origin.

**`centroid` and `median` invert.** They can merge at a height below one of their children's. This
is a property of the linkage, not a bug, and it is why cuts here are taken as a prefix of the
agglomeration order rather than by sorting on height. If you need a monotone dendrogram, use
`average`, `weighted` or `ward`.

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

## Windowed stream queries — `WindowStream`

`DenStream` has only a present: decay makes the past fade, so it cannot answer "what did the data
look like between `t₀` and `t₁`". `WindowStream` keeps a summary **per frame** and answers a window
by summing the frames it covers:

```python
from betula_cluster import WindowStream

ws = WindowStream(frame_width=3600.0, capacity=48, max_micros=256)  # 48 hourly frames retained
for chunk, times in stream_of_arrays_with_timestamps:
    ws.partial_fit(chunk, times)          # timestamps must be one per row, non-decreasing per call
ws.close_frame()                          # seal the frame still filling, so it can be queried

ws.window_moments(t0, t1)                 # {'weight', 'mean', 'ssd'} summed over that window
centers, weights, cost = ws.cluster_window(t0, t1, 5)   # k-means over just that window's summary
```

Two properties are worth stating because they are the whole design:

- **The window is never computed by subtraction.** CluStream (Aggarwal et al., VLDB 2003) stores
  cumulative snapshots and gets `[t₀, t₁]` as `CF(t₁) − CF(t₀)`. That inverse merge loses
  `log₁₀(S_AB/S_B)` digits of the scatter, and under drift `S_AB` is dominated by the displacement
  *between* the windows, so the ratio runs away while the point counts stay small — a mass-based
  guard sees nothing. On a two-half fixture measured here it costs a factor of **6155** in the
  recovered variance at a mass ratio of 2.0. Summation has no such term.
- **The price is resolution, and it is bounded.** A window resolves only to a frame boundary: a query
  ending 0.1 s into a frame gets that whole frame. The error is bounded by `frame_width`, where the
  subtraction's error is bounded by nothing. Pick `frame_width` as the coarsest resolution you will
  ever query at, and `capacity` as how far back you want to be able to look.

`Moments::checked_subtract` in the Rust core does implement the inverse merge, and refuses rather
than returning digits it does not have — it is there to be measured against, not to be relied on.

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
