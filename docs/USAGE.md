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

Keyword args: `feature ∈ {spherical, diagonal, full, fd}`, `method ∈ {kmeans, xmeans, kmedoids, fuzzy-cmeans, gmm, gmm-full, mppca, mfa, ward, average, weighted, centroid, median, spectral, leiden, leiden-cpm, spherical-kmeans, vmf, watson, hyperbolic, gmm-toeplitz, gmm-toeplitz-full, gmm-toeplitz-gs, hdbscan, dc-center, dc-median, scale-space}`,
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
`feature="full"`; `0` = off), `rank` (subspace rank `q` for `method="mppca"` and `method="mfa"`, clamped to at
most `dim - 1`; `0` makes every `mppca` component spherical and every `mfa` component diagonal), `fuzzifier` (the exponent `m > 1` of
`method="fuzzy-cmeans"`, default `2.0`; `m → 1⁺` is k-means, `m → ∞` sends every membership to
`1/k`), `projection` / `projection_dim` / `projection_max_iter` (reduce the leaf centroids to
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
| compact/spherical groups, fastest | `kmeans` | yes (or `0` = BIC sweep, capped at 20) |
| the same, but the count may exceed 20, or the sweep is too slow | `xmeans` | **no** — `n_clusters` is an upper bound; see *Where `xmeans` refuses to split* |
| the same, but the centre must be a real observation (an exemplar you can show) | `kmedoids` | yes — `0` switches objective to the medoid silhouette |
| a **graded membership** with no density behind it (each point partly in several clusters) | `fuzzy-cmeans` | yes (or `0` = Xie–Beni) — read *A soft head that fits no density* first: the hard partition is a **loss** against `kmeans` |
| elliptical / correlated / anisotropic, soft assignment | `gmm` (diag) or `gmm-full` | yes (or `0` = BIC) |
| clusters on **low-dimensional subspaces**, `d` too large for `gmm-full` | `mppca` + `feature="fd"`, `rank` = the intrinsic dimension — read *`rank`, and where `mppca` loses* first | yes (or `0` = BIC) |
| the same, but the **columns are in different units** and cannot be standardised | `mfa` — per-axis noise instead of one `σ²` — read *Per-axis noise, and the narrow case for it* first: on a common scale `mppca` wins every table measured | yes (or `0` = BIC) |
| **L2-normalized embeddings** (CLIP / face / sentence / speaker), cosine geometry | `vmf` (soft) or `spherical-kmeans` (hard) | yes (or `0` = BIC, `vmf`) |
| the same, but the **sign is arbitrary** — eigenvectors, SVD/PCA axes, line orientations, any feature where `x` and `−x` mean the same thing | `watson` — read *Directions without a sign* | yes (or `0` = BIC) |
| a **hyperbolic embedding** of a hierarchy (Poincaré / Lorentz coordinates of a tree, taxonomy, scale-free graph) | `hyperbolic` — read *Clustering in the hyperbolic plane* first: the win is **invariance**, not ARI, and a Poincaré-ball chart plus `gmm-full` scores higher on a centred embedding | yes — **no** auto-`k` |
| a cluster *hierarchy* / merge structure | `ward` | yes (or `0` = dendrogram cut) |
| **non-convex / manifold** shapes (moons, rings, spirals) | `spectral` | yes (pair with a **small** `threshold`) |
| **community / graph structure**, unknown count | `leiden` (or `leiden-cpm`) | **no** — count is discovered; tune `resolution` |
| variable-density clusters **+ noise**, unknown count | `hdbscan` | no |
| the same density structure, but you know `k` and want a partition rather than noise | `dc-median` | yes — read *Naming `k` on the density hierarchy*; `dc-center` is the other objective and is **mass-blind** |
| **density peaks**, arbitrary count, no `k` *or* bandwidth to pick | `scale-space` | **no** — scale chosen by mode persistence |
| **ordered / stationary signals** (time-series windows, trajectories, sensor waveforms), covariance *shape* | `gmm-toeplitz` | yes (or `0` = BIC) |
| ordered signals with structure **beyond a low-order AR** (long-lag echo, narrowband) | `gmm-toeplitz-full` (any lag) or `gmm-toeplitz-gs` (likelihood-optimal precision, ≤ order 16) | yes (or `0` = BIC) |
| topological skeleton / #components / loops | [`mapper()`](FEATURES.md) | no |

`n_clusters=0` auto-selects `k` for the parametric heads; `leiden` / `hdbscan` always discover it
(`leiden` reads the count off the graph — tune granularity with `resolution` γ, higher ⇒ more).
For a robustness score per point, wrap any partitional head in `consensus` (see below).

### `auto_k_max` — the ceiling `n_clusters=0` searches under

Two families of selector sit behind `n_clusters=0`, and they pay for a wide search differently.

A **sweep** — `kmeans`, `gmm`, `gmm-full`, `mppca`, `mfa`, `vmf`, `watson` and the three `gmm-toeplitz` rungs — refits
the whole head at every candidate `k` and keeps the best BIC. Its work is `Σ_{k≤K} k = O(K²)`, so the
ceiling is the only thing bounding it, and it defaults to **20**. A **cut** selector — `ward`,
`average`, `weighted`, `centroid`, `median` — builds one dendrogram and scores its cuts, and `xmeans`
stops on its own split test; a wider ceiling costs those a linear pass and nothing more, so they are
bounded only by the leaf count.

Measured on 480 leaves in 64 dimensions holding **120** true groups:

| head | ceiling 20 | ceiling 120 | ceiling = leaf count (480) |
|---|---|---|---|
| `ward` | 5.7 ms, `k` = 2, ARI 0.009 | 7.8 ms, `k` = 120, **ARI 1.000** | 23.8 ms, `k` = 120, **1.000** |
| `kmeans` | 45.5 ms, `k` = 20, ARI 0.109 | 1449 ms, `k` = 120, **1.000** | — |
| `gmm` | 502 ms, `k` = 20, ARI 0.109 | 4100 ms, `k` = 120, **1.000** | — |
| `xmeans` | — | — | 12.1 ms, `k` = 120, **1.000** |

(`the_cost_of_the_auto_k_ceiling` in `src/model.rs`, `cargo test --release --all-features --lib --
--ignored --nocapture the_cost_of_the_auto_k`. The leaf-count column is left unmeasured for the two
sweeps rather than run for minutes to restate what the 120 column already shows.)

Read three things off it. The ceiling is not a mild preference — at 20 the answer is wrong, not
merely coarse. Lifting it for a sweep costs 8–32×, which is why the default stays at 20 and
`auto_k_max` is an explicit opt-in (`0` = the default). And on k-means-shaped data `xmeans` gets the
same partition as the fully-swept `kmeans` **120× faster**, which is the head to reach for before
`auto_k_max`.

A selection that lands exactly on its ceiling raises a `UserWarning`: an argmax on the last candidate
is evidence the search stopped early, not evidence about the data.

### Where `xmeans` refuses to split — `method="xmeans"`

`kmeans` with `n_clusters=0` fits a full k-means at **every** `k` and keeps the best BIC. That costs
`O(k_max²)` passes over the leaves, which is why its cap is 20. `xmeans` (Pelleg & Moore, ICML 2000)
asks a different question: it tests each centre separately for a 2-way split and stops when none
wants one, so the cost is `O(k)` two-centre problems on shrinking subsets and there is nothing to
cap. `n_clusters` is an **upper bound** here, not a target; `0` bounds it only by the leaf count.

**The split test has a threshold, and it is a function of `d`.** A balanced binary split costs
`n·ln 2` of mixture-weight likelihood and buys `½·n·d·ln(S₁/S₂)`, so it is accepted only when the cut
captures more than `1 − 2^(−2/d)` of the region's sum of squares:

| `d` | 2 | 5 | 10 | 32 | 64 |
|---|---|---|---|---|---|
| fraction of the scatter one cut must capture | **0.50** | 0.24 | 0.13 | 0.043 | 0.022 |

A cut through a cloud that is round at every scale captures about `0.64/d`, always less than the
`1.39/d` the rule asks for. **The recursion therefore starts at `k = 2`, not at `k = 1`** — a greedy
splitter has no way back from a refused split, and at `k = 1` the entire answer rides on the one
comparison the threshold answers worst, since a layout of many well-separated groups is itself
close to round. Measured: from `k = 1` the head is exact at 10, 20 and 30 blobs and then collapses
to `k = 1` in five of twenty `(k*, seed)` cells, in every seed at `k* = 60`, and on a 3×3 grid of
nine equal 2-D blobs. From `k = 2` all of those return the true count. Pelleg & Moore, ELKI and
pyclustering all start at 2 for this reason.

What survives the fix is the threshold itself, at `d = 2` and a large `k`. Measured on random blob
layouts, 4 leaves per blob, 40 points per blob, median of seeds 0/1/2, with `kmeans`/`n_clusters=0`
at its shipped cap of 20 and `xmeans` bounded only by the leaf count:

| `d` | true `k` | `kmeans` \|Δk\| | `kmeans` s | `xmeans` \|Δk\| | `xmeans` s |
|---|---|---|---|---|---|
| 2 | 10 | **0.3** | 0.0012 | 0.7 | 0.0001 |
| 2 | 30 | 10.0 *(capped)* | 0.0030 | 6.7 | 0.0005 |
| 5 | 10 | **0.0** | 0.0012 | **0.0** | 0.0001 |
| 5 | 30 | 10.0 *(capped)* | 0.0034 | **0.0** | 0.0005 |
| 10 | 10 | **0.0** | 0.0016 | **0.0** | 0.0001 |
| 10 | 30 | 10.0 *(capped)* | 0.0048 | **0.0** | 0.0008 |
| 32 | 30 | 10.0 *(capped)* | 0.0067 | **0.0** | 0.0010 |
| 64 | 30 | 10.0 *(capped)* | 0.0090 | **0.0** | 0.0014 |

Read the rule off the table: at every `d ≥ 5` `xmeans` lands on the true count, including where the
sweep is stuck at its cap, for 6–16× less time. At `d = 2` it still under-splits once the count is
large — 6.7 short of 30 — because that is where one cut has to capture half the scatter; use
`kmeans` with `n_clusters=0` there, since a sweep compares `k = 1` against `k = 30` directly and
never has to pass through the cuts the split test refuses.

### A centre that exists in the data — `method="kmedoids"`

`kmedoids` runs eager FasterPAM (Schubert & Rousseeuw 2021) over the leaf centroids, weighting each
leaf by its mass. The centre of a cluster is one of the summary's own micro-clusters rather than an
average, which is the point: it is an exemplar you can show, and it stays on the data manifold where
a mean need not.

It is **exact on the summary** under the whole-leaf restriction every shipped head accepts. With the
medoid drawn from the leaf-centroid set,

$$\sum_{x \in \text{leaf } i} \lVert x - \mu_j \rVert^2 = S_i + n_i \lVert \mu_i - \mu_j \rVert^2,$$

so the leaf-level total the swap search minimises **is** the point-level sum of squares, up to the
constant $\sum_i S_i$ no medoid choice can move. Note the square: classical PAM minimises the sum of
*absolute* distances, and $\sum_{x \in \text{leaf}} \lVert x - \mu_j \rVert$ has no closed form in a
cluster feature. The squared objective is the one a summary can answer exactly, and it is what this
head reports as its loss.

`n_clusters=0` is **not** a free auto-`k` here: total deviation falls monotonically as `k` grows, so
this head's own objective cannot choose. The automatic arm runs `dyn_msc` — the medoid silhouette of
Lenssen & Schubert 2024, a different objective — and says so rather than pretending the sweep is free.

`refine` is a no-op. A Lloyd sweep is the k-means update; it would move each medoid to the mean of
its cluster, off the data and out of the head's own objective.

Measured on `digits` (1797 × 64, `feature="spherical"`, `threshold=0.0`), ARI as median [min–max]
over seeds 0–4:

| `max_leaves` | leaves | `kmeans` | `kmedoids` | `kmeans` s | `kmedoids` s |
|---|---|---|---|---|---|
| 2000 (one leaf per point) | 1797 | 0.467 [0.443–0.571] | **0.554** [0.554–0.570] | 0.025 | 0.157 |
| 300 | 296 | 0.487 [0.381–0.523] | **0.520** [0.468–0.520] | 0.007 | 0.008 |
| 120 | 115 | **0.240** [0.202–0.252] | 0.219 [0.219–0.219] | 0.005 | 0.004 |

Two things to read off it. The medoid restriction is a regulariser: at a fine summary it wins on the
median *and* on the spread, because the centre cannot drift into the gap between two digit classes.
And it inverts at a coarse summary — 115 candidate centres is not enough to place ten of them well,
and there `kmeans` is free to put a centre where no leaf sits. The `O(m²)` swap pass is what costs:
invisible at 296 leaves, 6× `kmeans` at 1797.

On the synthetic fixtures at `max_leaves=4000` the two are level (`blobs` 0.864/0.864, `aniso`
0.545/0.548, `varied` 0.540/0.548, `highdim` 1.000/1.000, median of seeds 0/1/2), which is the
expected result: where the mean is a good centre, so is the nearest micro-cluster to it.

`fit_predict_sparse` does not accept this head — see [Sparse input](#sparse-input).

### A soft head that fits no density — `method="fuzzy-cmeans"`

`fuzzy-cmeans` is weighted fuzzy c-means (Bezdek 1981) over the leaf centroids. It minimises

$$J_m = \sum_i \sum_j u_{ij}^m \, n_i \, d_{ij}, \qquad d_{ij} = \lVert \mu_i - c_j \rVert^2 + S_i / n_i,$$

with $\sum_j u_{ij} = 1$. Tying the membership within a leaf makes the leaf-level objective **equal**
to the point-level one — the same König–Huygens identity every centroid head here rests on — so
`d_ij` is exact and the head pays leaf cost for a point-level answer. The alternating minimiser is
$u_{ij} \propto d_{ij}^{-1/(m-1)}$ and $c_j = \sum_i u_{ij}^m n_i \mu_i / \sum_i u_{ij}^m n_i$; the
mass $n_i$ cancels out of the membership normalisation and survives only in the centre update.

It is the only soft head in the crate that fits no density. `predict_proba` returns the memberships
themselves — a partition of unity over the centres, **not** a posterior, and not comparable across
fits with different `m`. `n_clusters=0` selects by Xie–Beni, $J_m / (W \cdot \min_{i \ne j} \lVert c_i - c_j \rVert^2)$,
which is this family's own validity index rather than a borrowed likelihood criterion.

**Read this before choosing it: the hard partition is a loss.** ARI on the bench fixtures at
`max_leaves=4000`, `feature="spherical"`, `threshold=0.0`, median of seeds 0/1/2:

| fixture | `kmeans` | `fuzzy-cmeans` `m=1.3` | `fuzzy-cmeans` `m=2.0` |
|---|---|---|---|
| `blobs` | **0.864** | 0.847 | 0.843 |
| `aniso` | **0.545** | 0.535 | 0.531 |
| `varied` | **0.540** | 0.531 | 0.537 |
| `highdim` | 1.000 | 1.000 | 1.000 |
| `digits` | 0.467 | **0.483** | 0.356 |

The intuition that a soft head should win where clusters overlap is the wrong way round, and the
overlap sweep says so directly — 6 000 points, 8 dimensions, 5 centres, `max_leaves=2000`, median of
seeds 0/1/2. `AUC` is how well `assignment_confidence` separates correctly- from wrongly-labelled
points, so it scores the *membership* rather than the partition:

| `cluster_std` | `kmeans` ARI / AUC | `m=1.3` | `m=2.0` |
|---|---|---|---|
| 3.0 | 0.990 / **0.985** | 0.990 / 0.975 | 0.990 / 0.955 |
| 4.0 | **0.922** / **0.931** | 0.913 / 0.916 | 0.889 / 0.830 |
| 5.0 | **0.804** / **0.881** | 0.623 / 0.836 | 0.580 / 0.792 |

The gap widens with overlap, and it widens with `m`, because both push the same mechanism: no
membership is ever zero, so **every** point contributes to **every** centre and each centre is pulled
toward the grand mean. Measured on the `cluster_std=5.0` fixture (seed 0), the mean distance from a
cluster centre to their common centroid runs 13.75 for `kmeans` and 13.19 / 12.98 / 11.88 for
`m = 1.3 / 2.0 / 3.0`, against a true 13.67. `m` is a contraction knob as much as a softness knob.

So take this head for the membership, not for the labels: a per-point degree of belonging that comes
out of the objective the fit minimised, where every other centre-based head can only offer the
`softmax(−d²/2τ²)` proxy with a τ nobody chose. If you want the labels, `kmeans` is faster and
better; if you want a calibrated posterior, `gmm` fits one.

Xie–Beni on the same fixtures (median over seeds 0/1/2, showing all three): exact on `highdim`
(8/8/8 for a true 8) and on the separated overlap fixtures (5/5/5 at `cluster_std ≤ 3.0`); it
under-counts on `blobs` (2/4/4 for 6) and `aniso` (2/2/2 for 3), over-counts on `varied` (4/4/4 for
3), and on `digits` runs into the sweep ceiling (18/20/19 for 10, which raises the saturation
warning). Like every sweep selector here it costs a refit per candidate `k`, so `auto_k_max` bounds
it at 20 by default.

`refine` is a no-op, for the same reason it is under `kmedoids`: a Lloyd sweep replaces each
membership-weighted centre with the hard mean of its argmax cluster, which is a different objective
under the same name.

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

### Per-axis noise, and the narrow case for it — `method="mfa"`

`mfa` is `mppca` with one constraint removed: the residual a component cannot explain with its
`rank`-dimensional subspace is per dimension, `Σ_c = W_c W_cᵀ + diag(ψ_c)`, rather than a single
`σ_c² I`. That is the classical mixture of factor analysers (Ghahramani & Hinton 1996). It costs
`d − 1` extra parameters per component — against `d²/2` for `gmm-full` — and it removes the head's
rotation equivariance, exactly as `gmm` (diagonal) does: `diag(ψ)` is a statement about the
coordinate axes, so the answer depends on them. `rank=0` is a diagonal Gaussian mixture, bit for bit.

**The two heads dissociate in both directions.** Which one is right is a question about where the
signal sits relative to the axis scales, and neither answer is general:

| fixture (6-D, ARI over 5 seeds) | `mfa` | `mppca` | `gmm` |
|---|---|---|---|
| separation on a **quiet** axis, two loud nuisance axes | **1.00** | 0.04–0.34 | 1.00 |
| three lines differing only in **orientation** | ≈ 0.00 | **1.00** | ≈ 0.00 |

The first row is `mppca` paying for its isotropic residual: one `σ²` has to cover an axis of sd 1.6
and an axis of sd 0.12 at once, and the loud pair wins. The second is the mirror image — `ψ` can
absorb an elongation that belonged in `W`, so the factor never has to find the orientation, and the
head lands on the diagonal answer.

**On real tables it is behind `mppca` at full leaf resolution everywhere it was tried.** ARI /
seconds, `rank=2`, `threshold=0`, median of seeds 0/1/2:

| dataset | `feature` | `max_leaves` | leaves | `gmm` | `gmm-full` | `mppca` | `mfa` |
|---|---|---|---|---|---|---|---|
| `digits` (1797×64) | `full` | 2000 | 1797 | 0.507 | **0.754** | 0.738 | 0.562 |
| | | 300 | 270 | 0.576 | 0.623 | **0.657** | 0.523 |
| | | 120 | 114 | 0.623 | **0.676** | 0.675 | 0.660 |
| `digits` | `fd` | 300 | 270 | 0.562 | 0.467 | **0.652** | 0.502 |
| | | 120 | 114 | 0.611 | 0.278 | **0.671** | 0.631 |
| `covtype`-20k, raw units (54-D) | `full` | 2000 | 1815 | 0.045 | 0.080 | **0.087** | 0.062 |
| | | 300 | 299 | **0.063** | 0.047 | 0.046 | **0.063** |
| `covtype`-20k, standardised | `full` | 2000 | 1899 | **0.077** | **0.077** | 0.030 | **0.077** |
| | | 300 | 271 | **0.089** | 0.084 | 0.045 | **0.089** |

MNIST-20k (784-D, `feature="fd"`, `max_leaves=2000`) says the same by rank — `mppca`
**0.365 / 0.309 / 0.205** at `rank` 2 / 5 / 10 against `mfa`'s **0.277 / 0.299 / 0.130** — at
indistinguishable cost, 30.9 / 76.7 / 185.8 s against 37.6 / 80.6 / 197.5 s. The extra `d − 1`
parameters are not what is expensive; they are what is hard to estimate.

**What the head is actually for, then.** Not a higher ceiling — a *floor*. It contains the diagonal
Gaussian mixture as its `rank=0` limit and, where the factor finds nothing, converges onto it: on
standardised `covtype` it reproduces `gmm`'s partition exactly at both budgets, while `mppca` — whose
model cannot fall back on a per-axis noise — scores **less than half** the diagonal head there
(0.030 against 0.077). Reach for `mfa` when the columns are in different units and you cannot
standardise them (a mixed feature table, a physical measurement whose scales are meaningful), and for
`mppca` when they are already on a common scale, which is every image, embedding and whitened
design matrix.

One reading to avoid: the `covtype`-standardised row is *not* evidence that `W` collapsed to zero.
At `max_leaves=2000` in 54 dimensions the posterior is saturated — all 20 000 points sit at
`max_c P(c|x) = 1` to within `1e-12`, for `mfa`, `gmm` **and** `gmm-full` alike — so agreeing on
labels is what any mean-driven partition looks like there, and identical `predict_proba` output
restates the labels rather than the covariance. At `max_leaves=300`, where the posterior is no longer
saturated, the two partitions come apart — ARI 0.909 between them, not 1.

The `fd` rows start at 300 because at `max_leaves=2000` every leaf holds one point, where the sketch
has nothing to truncate and reproduces `full` to the last digit.

### Directions without a sign — `method="watson"`

`vmf` models a *direction*: `−μ` is as far from `μ` as its density goes. A great deal of directional
data is not like that. An eigenvector, an SVD or PCA axis, a line's orientation, a fibre direction,
the output of any routine that fixes a sign arbitrarily — for all of them `x` and `−x` are the same
observation. Handed that, `vmf` spends half its components on the antipodes of the other half, and
where the two poles are equally populated its resultant $\sum_i n_i \mu_i$ cancels to nothing.

The Watson distribution (Watson 1965; Mardia & Jupp 2000 §9.4.1) is the axial answer:

$$p(x \mid \mu, \kappa) = \frac{\Gamma(d/2)}{2\pi^{d/2} M(\tfrac12, \tfrac d2, \kappa)} \exp\!\big(\kappa (\mu^\top x)^2\big), \qquad x, \mu \in S^{d-1}$$

$(\mu^\top x)^2$ does not change when `x` flips sign, which is the whole point. $M$ is Kummer's
confluent hypergeometric $_1F_1$. `κ > 0` is **bipolar** — mass at both ends of the axis; `κ < 0` is
**girdle** — mass on the equator *orthogonal* to `μ`, which is how a co-planar structure is
described. Both signs are fitted, and which one a component takes is decided by likelihood, not by a
flag.

```python
labels = betula_cluster.fit_predict(X, n_clusters=4, method="watson", feature="full")
```

Input is auto-L2-normalized, exactly as for `vmf` / `spherical-kmeans`. `n_clusters=0` selects `k`
by BIC.

**Why the leaf summary is enough.** The sufficient statistic is the second moment about the origin,
and a cluster feature carries it exactly: $E[x x^\top] = \Sigma_i + \mu_i \mu_i^\top$. So the E-step
term $E_{x \in \text{leaf } i}[(\mu_c^\top x)^2] = \mu_c^\top (\Sigma_i + \mu_i \mu_i^\top) \mu_c$ is
closed-form and exact, and the within-leaf spread is integrated rather than dropped — the same status
as the Gaussian heads' E-step. The M-step needs $T_c = \sum_i r_{ic} n_i (\Sigma_i + \mu_i \mu_i^\top)$,
also exact, and takes an extreme eigenvector of it.

**Where it wins.** `N = 6 000`, `max_leaves=300`, `feature="full"`, `k` given, median of seeds 0/1/2 —
ARI / seconds. Each axial fixture puts **half of each cluster's points at each pole** of its axis:

| fixture | `watson` | `vmf` | `spherical-kmeans` | `gmm-full` | `kmeans` |
|---|---|---|---|---|---|
| axial, 8-D, 2 poles/axis | **0.870** / 0.114 | 0.208 / 0.009 | 0.269 / 0.007 | 0.309 / 0.013 | 0.233 / 0.008 |
| axial, 32-D | **0.953** / 0.176 | 0.218 / 0.041 | 0.202 / 0.039 | 0.316 / 0.093 | 0.244 / 0.038 |
| axial, 64-D | **0.961** / 0.653 | 0.221 / 0.131 | 0.234 / 0.129 | 0.200 / 0.384 | 0.232 / 0.130 |
| one pole per axis, 32-D | 0.952 / 0.186 | **0.976** / 0.041 | **0.976** / 0.034 | 0.966 / 0.082 | **0.976** / 0.034 |
| girdles, 8-D | 0.100 / 0.476 | 0.000 / 0.011 | −0.000 / 0.009 | 0.000 / 0.015 | −0.000 / 0.008 |

Three readings, in order of how much they should change what you do:

1. **Where the sign is genuinely arbitrary, the gap is not close** — 0.87–0.96 against 0.20–0.32 for
   every other head, which is the difference between recovering the structure and not.
2. **Where it is not arbitrary, `watson` costs you.** Row four is the same generator with one pole per
   axis: `vmf` reads 0.976 and `watson` 0.952, because identifying `μ` with `−μ` throws away
   information that was there. This is a head for a property of your data, not a strictly better `vmf`.
3. **A mixture of girdles is not recovered on this fixture.** `κ < 0` fits and every other head scores
   exactly zero, but 0.100 is not a result — three great circles through a common origin intersect, and
   near the intersections no axial model can separate them. Read `κ < 0` as *a component that can
   describe a flat cluster*, not as a girdle-mixture head.

**Real data, where the sign ambiguity is the whole story.** `digits`, PCA-20, L2-normalized;
`flipped` multiplies a random half of the rows by −1 and leaves the labels alone — the transformation
a sign-arbitrary pipeline applies for free:

| fixture | `watson` | `vmf` | `spherical-kmeans` | `gmm-full` | `kmeans` |
|---|---|---|---|---|---|
| `digits`-PCA20 | 0.498 / 1.407 | 0.631 / 0.036 | 0.625 / 0.010 | **0.685** / 0.120 | 0.665 / 0.009 |
| `digits`-PCA20, signs flipped | **0.503** / 1.282 | 0.173 / 0.020 | 0.204 / 0.009 | 0.284 / 0.341 | 0.221 / 0.010 |

`watson` moves by 0.005 across the flip and every other head loses 60–70 % of its score. That
invariance is the deliverable. It is also the whole trade: on the un-flipped data `watson` is the
worst of the five, because `digits` has no axial structure to find and the head has given up the sign
for nothing.

**Which leaf model.** The theory says the axial signal lives in the off-diagonal scatter that
`spherical` and `diagonal` leaves discard. The measurement says that is only true at a coarse budget.
Axial 32-D, `k = 4`, median of seeds 0/1/2; `leaves` is the realised count:

| `max_leaves` | leaves | `spherical` | `diagonal` | `full` | `fd` | median leaf-mean norm |
|---|---|---|---|---|---|---|
| 4 | 4 | 0.477 | 0.496 | **0.573** | 0.181 | 0.454 |
| 6 | 6 | 0.655 | 0.654 | **0.668** | 0.322 | 0.577 |
| 8 | 8 | 0.623 | 0.618 | **0.678** | 0.570 | 0.584 |
| 12 | 11 | 0.782 | 0.680 | **0.935** | 0.384 | 0.582 |
| 20 | 18 | 0.609 | 0.604 | **0.641** | 0.396 | 1.000 |
| 40 | 39 | 0.950 | 0.950 | 0.949 | 0.938 | 1.000 |
| 100 | 100 | 0.914 | 0.915 | **0.936** | 0.763 | 1.000 |
| 300 | 272 | 0.953 | 0.953 | 0.953 | 0.951 | 1.000 |

The last column is the mechanism. A leaf holding one pole has a mean of unit norm; a leaf straddling
both poles of its axis has a mean that cancels toward zero, and there the off-diagonal block is the
only thing left that names the axis. Above ~20 leaves every leaf sits on one pole, the mean alone is
enough, and the three dense leaf models tie. Below that `full` leads by 0.10–0.15 — on a budget whose
absolute quality is poor either way, and where the column is not monotone in the budget at all.
**Take `full`; do not expect it to buy much once the summary is fine.** `fd` is the one to avoid at a
coarse budget: the sketch's rank truncation costs it 0.2–0.55 there.

**Cost.** Axial fixture, `k = 4`, seed 0, seconds:

| `d` | `max_leaves` | `watson` | `vmf` | `gmm-full` |
|---|---|---|---|---|
| 8 | 120 / 300 / 1000 | 0.062 / 0.086 / 0.121 | 0.007 / 0.008 / 0.018 | 0.010 / 0.012 / 0.036 |
| 64 | 120 / 300 / 1000 | 0.751 / 0.656 / 1.954 | 0.119 / 0.132 / 0.243 | 0.193 / 0.380 / 1.491 |
| 256 | 120 / 300 / 1000 | 16.218 / 12.660 / 29.536 | 2.336 / 2.396 / 4.873 | 3.105 / 4.357 / 14.957 |

It is the most expensive of the directional heads: 5–8× `vmf`, and 2–6× `gmm-full`. The cost is the
special function, not the linear algebra — every E-step normalizer and every `κ` solve runs an
ascending $_1F_1$ series whose length grows with `κ`, which is why the head is capped at `κ = 10⁴`.

**Auto-`k`.** `n_clusters=0` runs the BIC sweep. On the axial fixture it is the *only* head that
finds the axis count: `vmf` answers 8 for 4 true axes, one component per pole, which is exactly the
failure the head exists to fix.

| fixture | true `k` | `watson` | `vmf` |
|---|---|---|---|
| axial, 32-D | 4 | **4** | 8 |
| one pole per axis, 32-D | 4 | **4** | **4** |
| girdles, 8-D | 3 | 1 | 1 |

**Numerics.** `log M(1/2, d/2, κ)` and the concentration equation `g(κ) = r̄` rest on two identities
verified symbolically in Maxima before any of it was written: `M'(a,b,z) = (a/b) M(a+1,b+1,z)`, and
Kummer's transformation `M(a,b,z) = e^z M(b−a,b,−z)`, which is what makes `κ < 0` computable at all —
the ascending series alternates for a negative argument and loses every digit to cancellation, while
for a positive one every term is positive. The leading asymptotic is **not** used: measured against
Maxima it is still 42 % low at `d = 50, κ = 50`. `g` is strictly increasing with `g(0) = 1/d`, so
`r̄ ≷ 1/d` fixes the sign of `κ` before any solving and the solver cannot pick the wrong branch.

### Clustering in the hyperbolic plane — `method="hyperbolic"`

Hierarchies embed into hyperbolic space with vanishing distortion where Euclidean space needs
exponentially many dimensions (Sarkar 2011; Nickel & Kiela 2017), so taxonomies, ontologies and
scale-free graphs increasingly arrive as points of `H^d`. This head clusters them there.

Rows are `(d+1)`-dimensional **Lorentz** coordinates, coordinate 0 time-like, and the boundary
recomputes it as `x₀ = √(1 + ‖s‖²)` before insertion — so a row that is slightly off the sheet is
projected rather than believed. A Poincaré-ball embedding `p` converts with
`x = (1 + ‖p‖², 2p) / (1 − ‖p‖²)`.

```python
labels = betula_cluster.fit_predict(X_lorentz, n_clusters=16, method="hyperbolic")
```

**Why a cluster feature can carry this at all.** The head does not use the geodesic distance
`d_H = arccosh(−⟨x,y⟩_L)`, which has no closed-form Fréchet mean. It uses the **squared Lorentzian
distance** of Law, Liao, Snavely & Dhillon (ICML 2019), `d_L²(x,y) = −2 − 2⟨x,y⟩_L`, which is a
strictly increasing function of `d_H` — so it orders pairs identically — and whose centroid is the
normalised sum `μ = R/|R|_L` with `R = Σ_i n_i x_i`.

`d_L²` is **affine in each argument**. Assigning a whole leaf costs `−2n_i − 2⟨R_i, c⟩_L`, which
reads the leaf only through `(n_i, R_i)`. There is **no scatter term at all** — a linear function has
no second order — so unlike every other head here the leaf's covariance is not merely cheap to use,
it is not used. Measured, tree layout below, `k = 16`, seed 0: `feature="spherical"` /
`"diagonal"` / `"full"` all read ARI **0.6911**, to four digits, because they are the same
computation. Take `spherical`; the others buy nothing.

**The fixture.** A 4-ary tree of depth 4 laid out in geodesic polar coordinates — depth sets the
hyperbolic radius (`τ = 1.6` per level), each node owns an angular interval its children subdivide —
with 60 points per tree leaf, 15 360 points, and the depth-2 subtree (16 groups) as the truth.
`threshold=0`, `max_leaves=2000`, median of seeds 0/1/2.

**Same points, three charts.** A hyperbolic embedding is handed over in one of three coordinate
systems, and all three are things people feed a clustering routine:

| chart | `hyperbolic` | `kmeans` | `gmm` | `gmm-full` | `ward` |
|---|---|---|---|---|---|
| Lorentz | **0.731** | 0.407 | 0.223 | 0.663 | 0.389 |
| Poincaré ball | — | 0.575 | 0.584 | **0.817** | 0.598 |
| tangent at the origin | — | 0.767 | 0.581 | 0.530 | 0.632 |

Read that honestly: **on a centred embedding, converting to the Poincaré ball and running `gmm-full`
scores higher than this head does.** The ball is a bounded chart that preserves angle and compresses
radius by `tanh(r/2)`, which is exactly what a Euclidean head needs, and on this fixture the class
structure is angular. What the head wins is the Lorentz-coordinate column, where a Euclidean head is
ranking `sinh r` above angle and loses half its score.

**Where that reading breaks.** A Lorentz boost is an *isometry* of `H^d` — the same points, the same
hyperbolic distances, a different origin. Nothing about the data changed; a clustering that moves
under it was answering about the chart. Boosting the fixture by rapidity `φ` along one axis:

| `φ` | `hyperbolic` | Poincaré + `kmeans` | Poincaré + `gmm-full` | tangent + `kmeans` |
|---|---|---|---|---|
| 0.0 | 0.731 | 0.575 | **0.817** | 0.767 |
| 0.5 | **0.653** | 0.618 | 0.637 | 0.647 |
| 1.0 | **0.675** | 0.577 | 0.584 | 0.590 |
| 2.0 | **0.614** | 0.478 | 0.513 | 0.523 |
| 3.0 | **0.596** | 0.348 | 0.311 | 0.375 |

The ball route loses 0.51 of its ARI and the head loses 0.14. **The head is the one whose answer is a
property of the data rather than of where the embedding put its origin** — and if you do not know
that your embedding is centred, which for a learned embedding you do not, the `φ = 0` column is the
one you cannot count on.

**And the residual 0.14 is not the head's.** Both Lloyd steps read only `(n_i, R_i)` and `Λ` is
linear with `|ΛR|_L = |R|_L`, so the partition over a *fixed* leaf set is exactly boost-invariant
(asserted in `tests/equivariance.rs`). What moves is the CF-tree, which routes and absorbs in the
ambient Euclidean coordinates. Raise the budget until every point is its own leaf and the drift
disappears entirely:

| `max_leaves` | `φ=0` | `φ=1` | `φ=3` |
|---|---|---|---|
| 500 | 0.634 | 0.569 | 0.576 |
| 2000 | 0.731 | 0.675 | 0.596 |
| 8000 | 0.809 | 0.583 | 0.696 |
| 20000 (one leaf per point) | **0.772** | **0.772** | **0.772** |

That is the honest boundary of the feature: **the head is hyperbolic, the tree is not.** A
Lorentz-aware routing distance would close it and is not built.

**No automatic `k`.** `Σ_c 2(|R_c|_L − W_c)` falls monotonically in `k`, exactly as total deviation
does for `kmedoids`, so this head's own objective cannot select the count. `n_clusters=0` yields a
single cluster; name the `k` you want.

**Constraints.** No `predict_proba` — there is no density, and the Euclidean softmax fallback the
other centroid heads use would disagree with this head's own labels, so it raises instead. No
`projection=` — a code vector has no time-like coordinate. No sparse (CSR) input — the sheet
projection writes a nonzero into column 0 of every row. No `refine` — a Lloyd sweep is the Euclidean
update.

**The working radius is finite and is not a detail.** `⟨x,y⟩_L` is a difference of two nearly equal
`Θ(cosh² r)` terms against an `Θ(1)` answer, so `f64` has a hard ceiling at `r ≈ ½ ln(2/ε) ≈ 18.4`.
Measured against 60-digit mpmath, relative error of `d_L²` on two points a unit apart: `1e−16` at
`r = 0`, `1e−8` at 10, `2e−3` at 15, and the **sign flips at 18**. `r ≲ 10` is exact for any purpose;
past 18 nothing is. This is a property of the representation — a Poincaré-ball implementation
saturates *earlier* — which is why the Rust API's `HyperbolicKMeans` reports the largest radius it
saw.

### How many leaves the spectral head can use — `method="spectral"`

Every leaf is a node of the graph. It did not used to be: above 256 microclusters the head reduced
them to 256 weighted k-means landmarks and let each leaf inherit its landmark's label, which is a
second lossy summarisation on top of the tree's. What forced that was the `O(M³)` dense eigensolver,
and it has been replaced by a Chebyshev-filtered subspace iteration that only ever multiplies by the
sparse graph. Two internal boundaries remain, both about cost rather than correctness: the exact
Jacobi solver runs to 256 nodes, the exact `O(M²)` affinity to 2048, and past each the approximate
route takes over.

What that buys, A/B against the landmark path on identical trees, median of seeds 0/1/2:

| | `max_leaves` | landmark | Chebyshev |
|---|---|---|---|
| two-moons / two-circles, N=20 000 | 500 → 5000 | ARI 1.000, 0.20–0.43 s | ARI 1.000, **0.02–0.36 s** |
| `digits`-PCA20 | 500 | 0.660, 0.36 s | **0.779**, 0.03 s |
| | 1000 | 0.786, 0.40 s | **0.801**, 0.06 s |
| | 1797 (= n) | **0.766**, 0.50 s | 0.735, 0.19 s |

**The head has a leaf ceiling anyway, and it is the graph's, not the solver's.** At 10 000 leaves the
non-convex fixtures still score ARI 1.000 in 0.7 s. At 20 000 — one leaf per point — they fall to
≈ 0.6, and forcing the *exact* affinity there gives the same answer for ten times the wall clock. The
cause is the fixed neighbour count: 10 neighbours is `1.0 · log n` at 20 000 nodes, at the
connectivity threshold below which a k-NN Laplacian's spectrum stops describing the manifold. So
spend the budget where the head reads it — a few hundred to a few thousand leaves — and note that
this is the same advice `max_leaves` gets everywhere else, now with the mechanism attached.

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

**What "exact" and "of the summary" mean, precisely.** A cluster feature is a sum-decomposition in
the sense of Deep Sets — $f(X)=\varphi(\sum_u \psi(x_u))$ with $\psi(x)=(1,x,xx^{\mathsf T})$ — so it
carries exactly the permutation-invariant polynomials of degree $\le 2$ and nothing else. Everything
built from $\sum w\lVert x-c\rVert^2$ therefore comes back exactly: WCSS, the Ward merge cost,
Calinski–Harabasz, the RMS Davies–Bouldin, the Bregman $D4_\varphi$ and the GMM sufficient
statistics. Anything built from a `min` or `max` over pairs does not. The two sets

$$A=\{-3,-1,2,2\},\qquad B=\{-3,0,0,3\}$$

have the same weight, mean and scatter *exactly*, so their spherical, diagonal, full and FD features
are the same object — yet their single-linkage merge heights are $0,2,3$ against $0,3,3$, and DBSCAN
at $\varepsilon=2$, `minPts` $=2$ gives two clusters and no noise on $A$ against one cluster and two
noise points on $B$. A density or single-linkage head reading the summary is answering a question
the summary does not contain, and the counterexample is four points rather than an asymptotic
argument.

For the exact indices the summary's error is not a bound but an identity: when every leaf lies
inside one cluster,

$$\mathrm{WCSS}_{\text{points}} = \mathrm{WCSS}_{\text{summary}} + \sum_i S_i,$$

so the total leaf SSD — the quantity `threshold` and `max_leaves` control — *is* the cost of
summarising, exactly. For any index Lipschitz in the Wasserstein-2 metric the same quantity bounds
the error, since sending each point to its own leaf centroid is a feasible transport plan:
$W_2(\text{data},\text{summary})^2 \le \sum_i S_i / W$.

### `k` with a `k = 1` answer available — `gap_statistic`

```python
curve = betula_cluster.gap_statistic(X, k_max=8, n_refs=10, max_leaves=200)
curve.k                       # the chosen k -- 1 is a possible answer
curve.ks, curve.gaps, curve.standard_errors    # the whole sweep, for plotting
```

The null is a uniform sample over the data's bounding box **re-summarized at the same leaf budget**,
so both sides of the gap pay the same quantization error and the statistic measures structure rather
than compression; the same reference draws are reused across `k`, which pairs the comparison. Cost is
`O(ℓ k d)` per fit off the leaf summary, with no second pass over the points.

Measured on the two null fixtures from that table (1000×5, medians of seeds 0/1/2): a **single
Gaussian** answers `k = 1` on every seed, **uniform noise** on two seeds of three — the third is a
near-tie, `gap(1) − gap(2) = −0.007` against `−2.83` for a genuine two-cluster fixture, so it is an
undecided answer rather than a confident wrong one. That matches the paper's "partly". Two-blob and
four-blob controls come back at 2 and 4 on every seed.

| selector | trust it when | cannot |
|---|---|---|
| BIC (`n_clusters=0`, mixture head) | you want the authority on "is there anything here" | — it is parametric, so it is answering about *Gaussians* |
| `calinski_harabasz` | you want a cheap `k` and already believe `k > 1` | say `k = 1`; it is undefined there |
| `gap_statistic` | you need `k = 1` on the table but not a distributional assumption | be decisive on pure noise — expect the near-tie |

Hartigan's dip test is the obvious fourth row, and it is deliberately absent. The statistic was
reproduced exactly here — the definition solved as a linear program agrees with the reference
implementation to `5.5e-12` over 648 samples spanning seven shapes, ties included — but that form
costs one linear program per candidate mode (`7.97 s` at `n = 200`), which cannot back the
Monte-Carlo null a p-value needs. The cheap closed form, one convex hull each side of the mode, is
exact on tied data and under-estimates continuous data by up to `9.97e-3` on a statistic of order
`0.06`; an under-estimated dip calls multimodal data unimodal, which is the one error the test
exists to prevent. Closing that gap needs the published iterative algorithm, whose available sources
are GPL-2 or non-commercial and so cannot enter an MIT crate.

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

## Naming `k` on the density hierarchy — `dc-center` / `dc-median`

`hdbscan` reads its cluster count off the persistence of the mutual-reachability spanning tree. These
two cut the *same tree* for a `k` you name instead, and both cuts are **provably optimal** — the
density-connectivity distance (Beer et al., KDD 2023) is an ultrametric, and `k`-center and
`k`-median are exactly solvable in one.

```python
# k-median: minimise the mass-weighted total dc-dist. The one to reach for.
labels = betula_cluster.fit_predict(X, n_clusters=6, method="dc-median", min_samples=5)
# k-center: minimise the largest dc-dist. Exact, and mass-blind — see below.
labels = betula_cluster.fit_predict(X, n_clusters=6, method="dc-center", min_samples=5)
```

`min_samples` and `graph_degree` are `hdbscan`'s and mean the same thing. **Neither head ever
answers `-1`**: they partition. If you want noise, the question is DBSCAN\*'s and `hdbscan` is the
head for it, over this very tree.

**Why they are exact.** `dc(a, b)` is the heaviest edge on the path between `a` and `b` in that tree,
so it is the height of their lowest common ancestor in the single-linkage dendrogram. `k`-center then
falls out of deleting the `k − 1` heaviest edges. `k`-median falls out of a second observation: a
leaf's service cost depends only on *which subtree* its nearest centre sits in, never on which centre
it is — which turns choosing `k` centres into an `O(m·k)` knapsack over the dendrogram. Draganov et
al. (NeurIPS 2025) run the same recursion for every `k` at once. Both are checked against brute force
over every `C(m, k)` centre set for `m ≤ 12`.

**What they score.** `N = 6 000`, `max_leaves=600`, `min_samples=5`, `k` given (except `hdbscan`,
which picks its own), median of seeds 0/1/2 — ARI / seconds:

| fixture | `dc-center` | `dc-median` | `hdbscan` | `ward` | `spectral` | `kmeans` |
|---|---|---|---|---|---|---|
| moons, noise 0.06 | **1.000** / 0.020 | **1.000** / 0.021 | **1.000** / 0.020 | 0.434 / 0.011 | **1.000** / 0.203 | 0.255 / 0.004 |
| moons, noise 0.10 | 0.000 / 0.014 | **0.889** / 0.014 | 0.015 / 0.016 | 0.420 / 0.007 | 0.691 / 0.383 | 0.256 / 0.003 |
| circles | **1.000** / 0.015 | **1.000** / 0.015 | **1.000** / 0.015 | 0.000 / 0.006 | **1.000** / 0.221 | −0.000 / 0.004 |
| blobs, 8-D | 1.000 / 0.019 | 1.000 / 0.019 | 1.000 / 0.020 | 1.000 / 0.015 | 1.000 / 0.095 | 1.000 / 0.008 |
| blobs + 5 % noise | 0.000 / 0.015 | 0.725 / 0.015 | 0.322 / 0.016 | 0.841 / 0.007 | 0.864 / 0.381 | **0.878** / 0.005 |
| digits, PCA-20 | −0.000 / 0.018 | **0.725** / 0.020 | 0.458 / 0.019 | 0.721 / 0.010 | 0.703 / 0.408 | 0.669 / 0.007 |

`dc-median` is the head to take from this: it matches `spectral` wherever `spectral` wins and beats
it by 0.198 on the harder moons, at **20×** less wall clock, and it ties `ward` on `digits` while
beating `hdbscan` there by 0.267. It loses the noise fixture to the three centroid heads, which is
the expected shape — an ultrametric has no notion of a point being *between* clusters.

**`dc-center` is mass-blind, and that is not a bug to fix.** A maximum cannot see a weight, so on a
summary — where an outlier *is* a low-mass leaf — it spends the whole budget isolating strays. On the
noise fixture at `k = 6` its clusters hold `[6297, 2, 1, 0]` rows against `dc-median`'s
`[1670, 1240, 1039, 1022]`. This is the mass-based answer the CURE probe went looking for and did not
find in shrinkage: not a repair of the geometric objective but a different objective.

Seen from the other side, the same property makes `dc-center` **insensitive to over-specifying `k`** —
on moons-0.06 it holds 1.000 / 1.000 / 0.999 / 0.999 / 0.997 at `k = 2/3/4/6/10`, because its extra
clusters are singletons, while `dc-median` falls 1.000 / 0.754 / 0.524 / 0.465 / 0.285 because it
genuinely splits mass. Read that as one fact stated twice, not as two findings.

**Cost is the spanning tree's**, which both heads share with `hdbscan`: at `N = 6 000`, `k = 8`,
0.014 / 0.047 / 0.350 / 1.528 s at `max_leaves` 300 / 1000 / 3000 / 6000, within 2 % of `hdbscan`'s
own 0.015 / 0.044 / 0.347 / 1.526 and about 3× `ward`'s. `graph_degree > 0` bounds it, with the same
caveat as everywhere else on this tree: it changes the tree, so the answer stays optimal for a
different problem.

Neither head takes `n_clusters=0`. Both costs fall monotonically in `k` to zero at `k = m`, so the
objective cannot select it — the same reason `kmedoids` has no automatic mode.

## Density structure — `reachability()`

An OPTICS reachability plot (Ankerst et al. 1999) over the microclusters — a **diagnostic**, not a
head. It answers "what does the density structure look like, and at what scale does it change"
rather than "which cluster is this row in".

```python
est = betula_cluster.Betula(method="hdbscan", min_samples=5, max_leaves=600).fit(X)
p = est.reachability(min_samples=5)     # pass the same min_samples / graph_degree the fit used

p.order            # leaf indices in sweep order
p.reachability     # aligned with p.order; p.reachability[0] is inf — nothing reached the first leaf
p.core_distances   # per leaf, in leaf indexing (not sweep order)
p.weights          # per leaf mass, in leaf indexing

import matplotlib.pyplot as plt
plt.bar(range(len(p.order)), p.reachability)   # valleys are clusters, peaks are the walks between

leaf_labels = p.labels_at(eps)                      # DBSCAN* at eps, per leaf; -1 is noise
row_labels = leaf_labels[est.assign_microclusters(X)]   # ...lifted back to rows
```

**It is the density head's own hierarchy, not a second opinion about it.** OPTICS with no ε cutoff is
Prim's algorithm on the mutual-reachability graph, so the sweep walks the *same* spanning tree
`method="hdbscan"` cuts. Every peak is a merge height in that hierarchy and `labels_at(ε)` reproduces
its cut at ε exactly — asserted by test, not measured as an approximation. That is also why the
reachability here is the mutual `max(core(p), core(q), d(p, q))` rather than Ankerst's asymmetric
`max(core(q), d(q, p))`: the asymmetric form draws a similar-looking picture of a *different* tree.

**One position per leaf.** A valley's width is a leaf count, not a point count — three leaves can
hold a hundred thousand rows. Read `weights` before reading a width. Core distances follow the head's
convention too: they are mass-weighted, so a leaf that already carries `min_samples` points has core
distance 0 and can never be noise.

**Cost.** The sweep runs over the leaves, so it does not see `N` at all. `blobs`, 6 centres,
`max_leaves=300`, median of seeds 0/1/2:

| N | fit s | `reachability()` s | plot / fit | leaves |
|---|---|---|---|---|
| 5 000 | 0.005 | 0.0024 | 0.52 | 280 |
| 20 000 | 0.007 | 0.0027 | 0.39 | 291 |
| 80 000 | 0.017 | 0.0027 | 0.16 | 290 |
| 320 000 | 0.054 | 0.0028 | 0.05 | 297 |

What it *does* see is the leaf budget, quadratically — the default neighbour pass is exact. Same
fixture at `N = 80 000`: 99 leaves 0.0003 s, 290 leaves 0.0028 s, 926 leaves 0.0284 s, 2883 leaves
0.2970 s. `graph_degree > 0` bounds it with the same approximate proximity graph the density head
uses, and pays for it in the same currency: at 2883 leaves, `graph_degree=16` runs in 0.0364 s but
its cut agrees with the exact plot's at only **ARI 0.518**, and 32 / 64 / 128 all sit at **0.923** —
so the approximate graph gives you a plot of a slightly different tree, not a cheaper plot of the
same one. Leave it at 0 unless the leaf count makes that impossible.

**Against `sklearn.cluster.OPTICS` on the raw points.** `N = 6 000`, `max_leaves=600`,
`min_samples=5`, best ε on each side's own reachability grid, median of seeds 0/1/2:

| fixture | `labels_at` ARI | sklearn ARI | betula s (fit + plot) | sklearn s |
|---|---|---|---|---|
| blobs, 6 centres | **0.452** | 0.448 | 0.024 | 5.38 |
| moons | **0.997** | 0.978 | 0.028 | 4.92 |
| circles | **0.997** | 0.978 | 0.026 | 5.86 |
| blobs + 5 % uniform noise | 0.687 | **0.753** | 0.041 | 6.26 |

Two hundred times faster on all four, and it loses the one that is about noise. **The compression is
not why.** Pushing the leaf budget up on that fixture drives the mass per leaf from 23.0 to 1.0 —
one leaf per point, no compression left — and the best ARI only moves 0.518 → 0.687 → 0.687 →
0.686 → 0.707 at 274 / 572 / 1461 / 3899 / 6300 leaves. The first step is real and is the
mass-weighted core distance: at 274 leaves, 41 % of the true noise points sit in a leaf whose core
distance is 0, which by definition can never be labelled noise; by 3899 leaves that is 2 %. The rest
of the gap to 0.753 survives at zero compression, so it is the convention, not the summary —
DBSCAN\* has no border points to hand back to a cluster, and the mutual reachability is a stricter
link than Ankerst's asymmetric one.

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

### Has the stream changed? — `drift_`

`decay` is a *schedule*: it forgets at a fixed rate whether or not anything changed, and a wrong λ is
silent in both directions — too small and the model never forgets, too large and it forgets structure
that was never stale. Both streaming heads therefore also carry an **ADWIN** change detector (Bifet &
Gavaldà, SDM 2007), which answers the other question — *did the stream change* — from the data:

```python
ds.partial_fit(chunk)
ds.drift_        # {'alarms': 3, 'last_alarm': 2016.0, 'distance': 1.12, 'window': 650}
```

- `alarms` — change reports since construction, `last_alarm` the stream time (points seen) of the
  most recent.
- `distance` — the statistic being watched: the distance from an incoming point to the nearest
  micro-cluster, **in units of the micro-cluster radius** (`eps` / `r`), averaged over the adaptive
  window. Stationary data sits near 1 by construction; a distribution moving into space the model
  does not cover sends it far higher. Reported in radii so the reading does not depend on the scale
  of your features.
- `window` — points the adaptive window holds. It collapses on a change and regrows while the stream
  is stationary, so it is itself a read on how settled the model is.

The false-positive rate is a stated δ = 0.002 per point, not a tuned threshold; measured at 0.00050
on stationary streams ([`bench/drift.py`](https://github.com/ilgrad/betula-cluster/blob/main/bench/drift.py)).
Two things to expect:

- **Early alarms are the model warming up**, not drift. Until the first micro-clusters exist, points
  really are landing far from everything, and the falling routing distance is a real change in the
  statistic.
- **A `decay` fast enough to prune micro-clusters as fast as they form has no baseline to depart
  from**, and the detector then reports nothing at all rather than reporting noise. Use `decay` to
  choose how fast the model follows; use `drift_` to find out when it had to.

**It reports; it does not act.** An alarm prunes nothing, promotes nothing and relabels nothing —
what to do about a change is your policy, and a detector that silently rebuilt the model would turn
its own false-positive rate into a correctness problem.

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

`kmedoids` is rejected on this entry point, and the reason is a property of the flat summary rather
than of the head. A medoid centre is one micro-cluster, and a row's squared distance to one is
dominated by that micro-cluster's own norm — which varies with how many terms its handful of rows
happened to carry — while the overlap term that knows the topic is a fraction of it. On the
four-block corpus at `max_leaves=300` the head reads ARI **0.017** by its own medoid rule and
**0.002** by the micro-cluster route, against **1.000** for `kmeans`, whose pooled cluster centroids
average the norms out. `Betula(method="kmedoids").fit(X)` on the same CSR builds a real CF-tree and
scores **1.000**; use that.

`fuzzy-cmeans` is absent for a duller reason: this entry point takes no `fuzzifier` keyword, and a
head with a knob nobody can set is worse than no head. Its centre is a weighted mean of many
micro-clusters, so the norm-domination argument above does not apply to it —
`Betula(method="fuzzy-cmeans").fit(X)` on a CSR is the supported route.

### Text: reduce and cluster in one call — `projection="svd"`

Clustering TF-IDF in its own geometry does not work, and the size of the failure is worth stating:
on 20-newsgroups the unprojected sparse path scores **ARI 0.053**, against 0.056 for scikit-learn's
k-means on the same rows. The standard fix is to reduce first, and `projection="svd"` does it inside
the same call — a CF-weighted PCA of the **leaf summary**, so the factorization runs over `M ≈ 10³`
micro-clusters rather than `N` documents.

```python
labels = fit_predict_sparse(
    X, n_clusters=20, method="spherical-kmeans",   # cosine geometry on the codes -- see below
    max_leaves=256, projection="svd", projection_dim=50,
)
```

20-newsgroups TF-IDF (18 846 × 2 000, `k`=20, rank 50, median of seeds 0/1/2, single-threaded,
re-measured 2026-08-30):

| | ARI | time |
|---|---|---|
| sparse path, no projection | 0.053 | 3.2 s |
| `projection="svd"`, `max_leaves=256` | 0.146 | 0.61 s |
| `projection="svd"`, `max_leaves=512` | 0.136 | 1.2 s |
| `projection="svd"`, `max_leaves=2048` | 0.195 | 4.7 s |
| `TruncatedSVD(50)` + `KMeans` on the raw rows | 0.143 | 0.78 s |

Two things decide whether this works for you.

**Use a cosine head on the codes.** `method="kmeans"` on the same codes scores **0.026** against
`spherical-kmeans`'s 0.195 — a seven-fold difference, because the leading principal direction of a
TF-IDF corpus is document length, and only an angular objective ignores it.

**The leaf budget is the cost, not the projection.** Sweeping the rank from 1 to 100 moves the total
by 1.2 s; sweeping `max_leaves` from 256 to 2048 moves it from 0.61 s to 4.7 s, because the sparse
summarizer compares each row against every micro-cluster it has so far. Buy resolution deliberately.

The basis is not a compromise for being built from a summary: labelling raw rows in it scored 0.159
against 0.143 for `TruncatedSVD`'s own basis on the same rows (2026-08-24, not re-measured since the
leader pass changed). Under the spherical cluster feature the discarded within-leaf scatter is
isotropic, so it shifts eigenvalues and leaves the directions alone.

Unlike `weighted-nmf`, a PCA is a linear map, so each row is labelled by **its own** code
(`(x − x̄)Vᵀ`, computed from its non-zeros) rather than by its micro-cluster's. That distinction was
worth 0.062 ARI here when it was measured (2026-08-24), and it is why the NMF projection cannot be
given the same treatment: its code is the solution of a per-row nonnegative least squares, not a
matrix product.

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
