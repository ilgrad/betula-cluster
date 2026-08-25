# Benchmark: betula-cluster vs scikit-learn — quality · speed · memory

> **Provenance — read before quoting a number.** Re-measured **2026-08-24** against the working tree
> after 0.7.0. Every quality table is the **median of seeds 0, 1, 2**, and
> each ships its own
> min/median/max sidecar (`results_*_spread.csv`); the speed, memory, scale and sparse tables are a
> single run, since `bench/_worker.py` pins `seed=0` for them and they are seed-invariant by
> construction. The previous edition of this page was a single run of 2026-07-18 against a **0.2.0**
> build; every number on it has been replaced, several of them downwards. This page is re-measured as
> a whole, not patched cell by cell.
>
> This edition re-ran every table after the CF k-means++ sampling weight gained the leaf's own scatter
> term (Lang Eq. 5.4, `S_i + n_i·D²_i`; see the changelog's `[Unreleased]`). Cell-by-cell against the
> previous run, **only the seeded heads moved** — `kmeans`, `gmm`, `gmm-full` and `spectral`, plus the
> two sparse rows that cluster with `method="kmeans"`. `ward`, `leiden`, `hdbscan` and every
> scikit-learn row reproduced to the last digit, which is what makes the deltas attributable to that
> one change. The synthetic tables did not move at all: at `N = 30 000` under `max_leaves=4000` the
> quality suite is not where the term bites, and `digits` at `max_leaves=1797` holds one point per leaf,
> so `S_i = 0` and the change is provably a no-op there.
>
> The previous edition re-ran the quality tables after the `min_samples` convention change (the head now
> counts the microcluster itself, matching `sklearn.cluster.HDBSCAN`). No table here
> exercises `method="vmf"` (verified by grep over every `results_*.csv`), so 0.7.0's vMF
> concentration-cap fix moves nothing on this page.

Reproduce: `python bench/comprehensive.py --quality-only --seed S --tag _sS` for S ∈ {1, 2}, then
`python bench/comprehensive.py --seed 0 --tag _s0` (the full run, which also writes
`results_{scaling,memory,real_scale,sparse}.csv`), then `python bench/median_of_seeds.py --seeds 0 1 2`
and finally `python bench/comprehensive.py --plots-only`. Order matters: `--plots-only` reads the
canonical CSVs, so run before the median step it draws one seed instead of the published medians.
Export `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` explicitly — `comprehensive.py` uses
`os.environ.setdefault`, so an exported `OMP_NUM_THREADS=8` wins and silently un-pins BLAS. The
harness needs `scikit-learn pandas matplotlib scipy seaborn`; `seaborn` is imported only inside
`make_plots`, which runs *last*, so without it every CSV is still written and only the plot step
raises — check `plots/*.png` mtimes rather than trusting a run that printed its tables. The
supplementary studies are separate scripts: `bench/spectral_nonconvex.py` (spectral timing),
`bench/toeplitz_ar_mixture.py` (the `gmm-toeplitz` showcase) and `bench/nmf_cf_weighted.py`
(CF-weighted NMF). Every cell is guarded and failures are recorded, not hidden; both **wins and
losses** are reported below.

## TL;DR (honest)

- **Always faster, always lighter — on every row below.** betula labels 1 M points in **0.26 s**
  (9× faster than scikit-learn KMeans, 15× vs GaussianMixture, 30× vs Birch) and streams 10 M in a
  flat **~60 MB** where an in-core KMeans needs **~5.0 GB** (**82× less**, and the gap grows without
  bound). This is the unconditional win: it holds for every method at every size, and it is what a
  bounded-memory compression engine is built to deliver.
- **Quality is at parity on the centroid heads and ahead on the structured ones.** betula's k-means is
  at parity with scikit-learn (blobs 0.793 vs 0.794, `digits` 0.467 vs 0.468); full-covariance GMM
  **beats** scikit-learn's on anisotropic data (**0.961 vs 0.902**) and on real 64-D `digits`
  (**0.575 vs 0.463**, via the high-dimensional covariance floor); the **spectral** and **HDBSCAN**
  heads hit ARI **1.00** on moons and circles.
- **Two losses stated plainly.** On `covtype` and MNIST, `sklearn-birch` beats **every** betula head —
  0.131 vs a best of 0.100, and 0.426 vs 0.377. On `covtype` that is a loss on the merits (not a
  leaf-budget artefact, measured both ways) and the mechanism is now known: the leaf budget produces
  cells of far more unequal mass than a radius threshold does. On MNIST most of the gap is the price
  of compression — Birch returns 20 000 subclusters for 20 000 points and compresses nothing, and at
  equal (non-)compression the 0.059 gap falls to 0.010. See *The `covtype` loss*. HDBSCAN-on-CF also
  trails raw HDBSCAN on overlapping density (blobs 0.142 vs 0.324, `varied` 0.479 vs 0.802).
- **A capability no mainstream library ships:** `method="gmm-toeplitz"` — an AR/Toeplitz-structured
  covariance GMM for ordered stationary signals — recovers a mixture of AR processes that differ *only*
  in autocovariance (ARI **1.00** at long windows) exactly where both diagonal (blind) and full
  (singular at `N_k ≪ d`) score ≈ 0.

## Seed dependence — read this before comparing any two cells

Clustering quality is seed-dependent and EM is non-convex. On the synthetic sets **every** row moves
by more than 0.05 ARI across three seeds; `digits` `betula-kmeans` spans 0.443–0.571, and
`covtype` `betula-spectral` spans 0.014–0.101 — a 7× ratio. The previous edition of this page
published single runs, and in more than one place happened to publish the top of that range as the
result.

Every quality table below is therefore a **median of three seeds**, and the accompanying
`results_*_spread.csv` carries min/median/max per cell. A difference smaller than the two cells'
spreads is not a result. Where a comparison in this page turns on such a margin, it says so.

## Environment (reproducibility)

Absolute times vary by machine; the *ratios* far less.

| | |
|---|---|
| CPU | AMD Ryzen 7 5800HS (8 cores / 16 threads) |
| RAM | 38 GiB |
| OS / kernel | Fedora Linux 44, kernel 7.1.9 |
| Python / NumPy / SciPy / scikit-learn | 3.14.7 / 2.5.2 / 1.18.1 / 1.9.0 |
| matplotlib / pandas | 3.11.1 / 3.0.5 |
| Rust | rustc 1.98.0 |
| betula-cluster | 0.7.0 + unreleased changes (working tree), `maturin --release` (LTO, `codegen-units=1`); **portable** wheel (no `target-cpu=native`) |
| BLAS threads | 1 (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`) — comparable single-thread timings |

## Methodology

- **Datasets** (fixed seed, `StandardScaler`-normalized so a single betula `threshold` is fair across
  all): `blobs` (6 Gaussians), `aniso` (sheared/anisotropic), `varied` (unequal variances), `moons`,
  `circles` (non-convex), `highdim` (20-D, 8 clusters).
- **Metrics** — external (vs ground truth): **ARI**, **AMI**, **V-measure**; internal: **silhouette**
  (on a 5 k sample), **Davies-Bouldin**, **Calinski-Harabasz** (all in `results_quality.csv`).
- **betula params:** `threshold=0`, `max_leaves=4000`, single-thread — the memory-bounded default —
  except `leiden` (`threshold=0.4`, `max_leaves=800`; a very fine graph over-splits), the speed and
  memory suites (`max_leaves=2000`) and the sparse suite (`max_leaves=2048`). **sklearn params:**
  library defaults (`KMeans n_init=10`, etc.), on the **full** `N` (the `O(N²)` Agglomerative /
  HDBSCAN baselines run at `N = 30 000` in the quality table; only the *scaling* table caps them,
  since they cannot reach 1 M).
- **Isolation:** every speed/memory point runs in its **own fresh `subprocess`**; peak RSS is sampled
  from `/proc/self/statm` (the post-`exec` process only — immune to the launcher's footprint), with a
  14 GiB `RLIMIT_AS` cap and a timeout, so a method that explodes fails gracefully.

## Quality — ARI vs ground truth (N = 30 000)

![ARI heatmap](plots/quality_ari.png)

Median of seeds 0/1/2 (`results_quality.csv`; spreads in `results_quality_spread.csv`).

| method | blobs | aniso | varied | moons | circles | highdim |
|---|---|---|---|---|---|---|
| **betula-kmeans** | 0.793 | 0.545 | 0.671 | 0.485 | −0.000 | 1.00 |
| **betula-gmm** (diag) | 0.812 | 0.540 | **0.907** | 0.514 | −0.000 | 1.00 |
| **betula-gmm-full** | 0.811 | **0.961** | **0.907** | 0.504 | −0.000 | 1.00 |
| **betula-ward** | 0.776 | 0.573 | 0.663 | **0.641** | 0.014 | 1.00 |
| **betula-spectral** | 0.766 | 0.413 | 0.650 | **1.00** | **1.00** | 1.00 |
| **betula-leiden** (auto-`k`) | 0.722 | 0.465 | 0.633 | 0.512 | 0.007 | 1.00 |
| **betula-hdbscan** | 0.142 | 0.568 | 0.479 | **1.00** | **1.00** | 1.00 |
| sklearn-kmeans | 0.794 | 0.545 | 0.670 | 0.484 | −0.000 | 1.00 |
| sklearn-minibatch | 0.694 | 0.547 | 0.665 | 0.484 | −0.000 | 1.00 |
| sklearn-birch | 0.748 | 0.554 | 0.460 | 0.616 | 0.005 | 1.00 |
| sklearn-gmm (full) | 0.807 | 0.902 | **0.907** | 0.504 | −0.000 | 1.00 |
| sklearn-ward | 0.770 | 0.565 | 0.673 | 0.507 | 0.001 | 1.00 |
| sklearn-hdbscan | 0.324 | 0.567 | 0.802 | **1.00** | **1.00** | 1.00 |

Reading it honestly:

- **betula-kmeans ≡ sklearn-kmeans.** 0.793 vs 0.794 on blobs, 0.545 vs 0.545 on aniso, 0.671 vs
  0.670 on varied — three ties inside the seed spread. CF compression costs no quality at this
  resolution. (The previous edition read this as "*exact* parity, 0.861 = 0.861"; 0.861 is the top of
  the three-seed range, and the two heads agree at every point in it.)
- **betula-gmm-full beats sklearn's full GMM on the anisotropic case**, 0.961 vs 0.902 — the case
  k-means cannot do at 0.545. On `varied` all three GMM variants tie at 0.907.
- **betula-ward** (bounded, 4 000 leaves) beats the **full-30 000** `sklearn-ward` on `varied`
  (0.663 vs 0.673 — a tie), `moons` (0.641 vs 0.507) and `aniso` (0.573 vs 0.565), and edges it on
  blobs (0.776 vs 0.770). Compression does not cost here; the CF microclusters denoise the linkage.
- **betula-spectral and betula-hdbscan own the non-convex cases** — moons and circles **1.00**, where
  every centroid head sits at 0.49–0.51 and ≈ 0.
- **betula-leiden** discovers the community count with **no `k`** — strong on separable community
  structure (highdim 1.00, blobs 0.722) but, being a modularity community-detector rather than a
  general partitioner, it over-splits elongated manifolds (moons 0.512). Use spectral for those.
- The honest weak spot: **HDBSCAN-on-CF on overlapping density.** blobs 0.142 vs raw HDBSCAN's 0.324,
  `varied` 0.479 vs 0.802. Both are the wrong tool for overlapping Gaussians, and the CF approximation
  widens the gap. Use a parametric head for blobs; HDBSCAN-CF / spectral for density / non-convex.
  These cells are also by far the least stable in the whole table — betula-hdbscan's three-seed range
  is 0.059–0.327 on blobs, 0.077–0.568 on `varied` and **0.016–0.993** on `aniso`, where the median
  0.568 sits between a near-total failure and a near-perfect recovery. Read the whole row as a regime
  without a stable answer rather than as six numbers.

### Non-convex: spectral clustering that scales (N = 30 000)

`method="spectral"` runs the Ng-Jordan-Weiss spectral pipeline on the ≤ `max_leaves` CF microclusters
rather than on all `N`, so its cost is bounded by the microcluster count and the same call scales to
`N = 1 M`, where scikit-learn's graph + eigensolve are `O(N)`+ in memory and effectively cap out around
30 k (`bench/spectral_nonconvex.py`).

| method | moons ARI | circles ARI | time (moons / circles) |
|---|---|---|---|
| **betula-spectral** | **1.00** | **1.00** | **0.373 s / 0.241 s** |
| sklearn-SpectralClustering (k-NN affinity) | 0.999 | **1.00** | 0.389 s / 0.372 s |
| betula-leiden | 0.116 | 0.127 | 0.145 s / 0.117 s |

**Correction to the previous edition, which claimed 3–5× the speed.** At equal quality betula is now
**1.04× on moons and 1.54× on circles** — parity, not a multiple. betula's own times barely moved
(0.39 → 0.373, 0.25 → 0.241); scikit-learn's fell from 1.23 s / 1.22 s to 0.389 s / 0.372 s between
sklearn versions. The durable claim is the **scaling** one — betula's cost is set by `max_leaves`, not
by `N` — not a constant-factor speedup at 30 k.

`method="leiden"` is included as an honest negative: it is built for community / blob structure, not
elongated manifolds — modularity chops each arc into ~19 segments (ARI ~0.12), exactly the
resolution-limit behaviour the docs warn about. Use spectral for manifolds, Leiden for communities.

## Speed — fit time at N = 1 000 000

Measured at `max_leaves = 2000` (the tight bound); the quality tables above use `4000`. The extra
leaves add only a small constant to Phase-3, which clusters the leaves rather than the `N` points, so
the scaling shape — `O(N)` build, flat memory — is unchanged.

![Fit time vs N](plots/scaling_time.png)

| method | time @ 1 M | vs betula-kmeans |
|---|---|---|
| **betula-kmeans** | **0.264 s** | 1× |
| betula-gmm | 0.292 s | 1.1× |
| betula-gmm-full | 0.360 s | 1.4× |
| betula-ward | 0.417 s | 1.6× |
| betula-hdbscan | 0.801 s | 3.0× |
| sklearn-kmeans | 2.45 s | 9.3× |
| sklearn-minibatch | 2.76 s | 10.5× |
| sklearn-gmm | 3.85 s | 14.5× |
| sklearn-birch | 8.01 s | **30×** |
| sklearn-ward, sklearn-hdbscan | (O(N²) — cannot reach 1 M) | — |

All five betula heads finish a million points in **under 0.9 s**; full-covariance GMM runs **4 EM
restarts** (for robustness against local optima) **in parallel**, finishing in **0.360 s** — 10.7×
faster than scikit-learn's GMM. betula-ward does the equivalent of `O(N²)` agglomerative at 1 M in
**0.417 s**, where scikit-learn's Agglomerative cannot run past ~10 k at all.

This table is a **single run** — `bench/_worker.py` pins `seed=0` for the speed phase, so it is
seed-invariant by construction, but it is one timing sample and the ratios move by a few percent
between editions (previous editions read 0.230 s and 0.242 s for the same cell, and the sklearn
baselines moved further than betula's did — `sklearn-kmeans` 3.10 s → 2.45 s). Quote the order of
magnitude, not the third digit.

Two betula rows moved for reasons that are **not** timing noise, and both are named rather than
smoothed. `betula-hdbscan` 0.681 s → 0.801 s is the price of counting `min_samples` in points instead
of leaves: the core distance is now the smallest radius enclosing that much *weight*, which is more
work and finds more clusters. `betula-kmeans` 0.242 s → 0.264 s is the AVX2 dispatch's residual cost
below `d = 16` — this fixture is `d = 2` blobs, the one shape in the whole suite where the packed
kernels cannot help and their length gate is pure overhead (ADR 003). The heads that run on real
dimensions gain instead: streaming 20-D is **1.29×** faster and the sparse SVD pipeline **1.18×**.

## Memory — streaming stays bounded

![Peak memory vs N](plots/memory_streaming.png)

Peak RSS (own process, `/proc/self/statm`), betula via chunked `partial_fit` (never materializing the
array) vs an in-core KMeans that must hold all of `X` (20-D):

| N | betula (streaming) | sklearn KMeans (one-shot) | ratio |
|---|---|---|---|
| 500 k | 60.1 MB | 407 MB | 6.8× |
| 1 M | 60.4 MB | 647 MB | 10.7× |
| 2 M | 60.1 MB | 1.13 GB | 18.7× |
| 5 M | 60.5 MB | 2.57 GB | 42.4× |
| 10 M | **60.3 MB** | **4.97 GB** | **82×** |

betula's footprint is **flat in N** — the CF-tree is bounded by `max_leaves`, so it clusters streams
larger than RAM. Any in-core method's memory grows linearly with `N` (it must hold `X`), and
Agglomerative's pairwise-distance matrix is O(N²) — **~1 GB at just 10 k points**, OOM beyond.

`results_memory.csv` also carries a time column. Two earlier editions recorded the sklearn 10 M cell
as *lower* than its 5 M cell and called it a reproducible allocator artefact of the one-shot path;
**this edition does not reproduce it** (7.89 s at 5 M, 8.85 s at 10 M — sublinear, but monotone), so
that claim is withdrawn rather than carried forward. The betula column is the one this section rests
on, and it is 1.28–1.31× faster than the previous edition across all five sizes, which is the AVX2
kernels doing their job on 20-dimensional data.

## Real datasets — bounded 4 000-leaf budget

Synthetic data can flatter a method, so the same comparison on real datasets loaded straight from
scikit-learn (`load_digits`, `fetch_openml("mnist_784")`, `fetch_covtype`), standardized. The large
ones are subsampled to 20 k for the all-methods table so the O(N²) baselines stay feasible
(full-covariance GMM is skipped past ~100 dims — it is O(d³) per component; `—` below). Downloads are
best-effort. **Here betula runs at the tight `max_leaves = 4000`**; the next section removes that
self-handicap.

![Real-dataset ARI heatmap](plots/quality_real_ari.png)

Median of seeds 0/1/2 (`results_real.csv`; spreads in `results_real_spread.csv`).

| method | digits (1797×64) | covtype (20k×54) | mnist (20k×784) |
|---|---|---|---|
| **betula-kmeans** | 0.467 | **0.074** | 0.307 |
| sklearn-kmeans | 0.468 | 0.054 | 0.324 |
| **betula-gmm** (diag) | 0.461 | 0.076 | 0.234 |
| **betula-gmm-full** | **0.575** | 0.076 | — |
| sklearn-gmm (full) | 0.463 | 0.080 | — |
| **betula-ward** | 0.643 | 0.091 | 0.377 |
| sklearn-ward | 0.664 | — | — |
| **betula-spectral** | 0.653 | 0.100 | 0.203 |
| **betula-leiden** | 0.781 | 0.056 | 0.005 |
| **betula-hdbscan** | **0.164** | 0.051 | 0.000 |
| sklearn-hdbscan | 0.149 | — | — |
| sklearn-birch | 0.664 | **0.131** | **0.426** |

Bold on a betula row = it beats its same-algorithm scikit-learn counterpart. `sklearn-birch` is a
CF-tree method with no direct betula counterpart; it is shown for context and **it leads the
all-methods table on both large real sets**.

Reading it honestly:

- **digits (64-D):** k-means is a **tie** — 0.467 vs 0.468, well inside betula's 0.443–0.571 seed
  spread. The previous edition claimed a 0.568-vs-0.468 lead; 0.568 is the top of that range under
  `normalize=True`, not the raw-Euclidean median. The real win here is the **full GMM: 0.575 vs
  scikit-learn's 0.463**, where the high-dimensional covariance floor keeps all 10 components
  populated and an unregularized full GMM collapses one. `betula-leiden` leads the whole table at
  0.781 with no `k` supplied. **HDBSCAN is now a win too — 0.164 against `sklearn-hdbscan`'s 0.149,
  both zero-spread across three seeds** — and it is the first edition where that comparison is
  like-for-like: betula used to exclude the object from its own `min_samples` neighbourhood, so
  `min_samples=5` acted like 6 and the published row compared an effective 11 against
  `sklearn.cluster.HDBSCAN`'s 10. Aligning the convention also cut the noise fraction from 0.620 to
  0.580.
- **covtype (54-D):** a genuinely hard set — every method scores low, and at `max_leaves=4000` the
  betula heads sit within one seed spread of each other and of scikit-learn. betula-kmeans beats
  sklearn-kmeans (0.074 vs 0.054); the diagonal GMM at 0.076 against scikit-learn's full GMM at
  0.080 is a **tie**, not the win the previous edition claimed — the two three-seed ranges are
  0.055–0.096 and 0.055–0.102, i.e. almost coincident, and the margin either way is a fifth of the
  spread. The head to quote here is `ward` (0.091, range 0.086–0.093, the tightest on the table);
  `spectral`'s 0.100 median is higher still but spans −0.015 to 0.128 and cannot be leaned on. At
  **16 000 leaves** the GMM does separate from scikit-learn — 0.104 vs 0.080, two sections down.
  `sklearn-birch` at **0.131** still beats every betula head. See below.
- **MNIST (784-D):** raw Euclidean k-means scores **0.307** against scikit-learn's 0.324 — in 784
  dimensions distances concentrate (concentration of measure). `normalize=True` closes it (two
  sections down). `sklearn-birch` leads here too, 0.426 against betula-ward's 0.377 — but at
  **20 000 subclusters for 20 000 points**, i.e. no compression at all against betula's 5.3×; give
  betula the same non-compression and it reaches 0.416. See below.

### The `covtype` loss to `sklearn-birch` is real, not a budget artefact

`sklearn-birch` beating every betula head on `covtype` invites the obvious defence — that betula is
handicapped by its leaf budget while Birch's `threshold=0.5` default produces far more subclusters. It
does not survive measurement, tested in both directions:

- Raise Birch's `threshold` to 1.0 and it yields **4 222 subclusters — a 4.74 : 1 compression, the
  same ratio betula gets at `max_leaves=4000`** — and its ARI is unchanged at **0.132** vs 0.131.
- Hand betula Birch's own **11 774 leaves** and every head gets *worse*: kmeans 0.086, gmm 0.077,
  ward 0.086.
- Across seeds Birch's minimum (0.119) exceeds every betula head's **median**, and all but one head's
  maximum (0.102 for kmeans, 0.093 for ward). The exception is `spectral`, whose three seeds span
  −0.015 to 0.128 and so straddle Birch's floor on its best seed — a spread that wide is not a
  counter-example, it is a reason not to quote `spectral` on this dataset at all.

So this is a loss on the merits, recorded as one. The mechanism is below — it *is* the absorption
criterion, and it is now measured rather than conjectured.

### The mechanism: cell **shape**, not weighting and not compression

A BIRCH-family pipeline has two parts, so a 2×2 isolates them (`local/scratch/birch_gap_mechanism.py`,
`max_leaves=4000`, seeds 0/1/2, both sides assigned by nearest centre so only one thing differs):

| | covtype | mnist |
|---|---|---|
| **A** betula leaves + mass-weighted (the shipped path) | 0.0861 | 0.3667 |
| **B** betula leaves + *unweighted* agglomerative | 0.0274 | 0.0125 |
| **C** Birch subclusters + unweighted (Birch, reproduced — ARI 1.0000 against its own labels) | 0.1306 | 0.4257 |

**Mass weighting is not our deficit, it is our largest asset.** Holding the leaf set fixed and only
removing the weights costs **−0.059** on covtype and **−0.354** on MNIST. The entire gap — and more —
sits in the *summary*: B → C moves +0.103 and +0.413.

Two further measurements say what is wrong with the summary (`local/scratch/birch_gap_followup.py`):

- **covtype — the leaf masses are far more skewed.** At matched cell counts (betula 3 602, Birch 4 222):

  | | median | p99 | max | singletons | Gini | mass in heaviest 1 % |
  |---|---|---|---|---|---|---|
  | Birch @ `threshold=1.0` | 3.0 | 24 | 63 | 23.4 % | 0.476 | **6.6 %** |
  | betula leaves | 2.0 | 70 | 295 | 45.3 % | 0.683 | **23.2 %** |

  A radius threshold produces cells of roughly equal *extent*; a leaf budget produces cells of wildly
  unequal *mass*. Nearly half of betula's budget goes to singleton leaves, which summarize nothing,
  while 36 cells hold a quarter of the data at a radius that blurs class boundaries. Cell purity
  differs by only 0.024 (0.7915 vs 0.8158) — far too little to explain 0.046 of outcome, so it is the
  shape and not the purity that the linkage is following.
- **MNIST — most of that "loss" is the cost of compression, and Birch pays none of it.** In 784
  standardized dimensions the typical pairwise distance is ≈ √(2·784) ≈ 39.6, so Birch's radius
  threshold absorbs nothing: it returns **20 000 subclusters for 20 000 points** at both
  `threshold=0.5` and `1.0`, purity 1.0000. It is not compressing MNIST at all. Give betula the same
  non-compression and the gap nearly closes:

  | betula-ward `max_leaves` | realised leaves | compression | ARI |
  |---|---|---|---|
  | 4 000 | 3 779 | 5.29× | 0.3667 |
  | 8 000 | 7 666 | 2.61× | 0.3591 |
  | 16 000 | 15 664 | 1.28× | 0.4069 |
  | 20 000 | 20 000 | 1.00× | **0.4159** |

  Against Birch's 0.4257 the published **0.059 gap becomes 0.010 at equal compression** — so ~83 % of
  the MNIST row is the price betula pays for a 5.3× summary that Birch never pays because it never
  builds one.

The actionable half of this is covtype's mass skew, which is a property of the leaf-budget criterion
and not of the CF or of the head. It is tracked as its own task (mass-balanced leaf budget); the
MNIST row is not a defect to fix but a comparison that has to be read with its compression ratios
attached.

### De-handicapped: betula at adequate leaf resolution

The 4 000-leaf cap exists for the **memory / scale** story — betula's footprint stays flat while
`N → 10 M`. On a **20 k-row eval set** that cap is a *self-handicap*, not a fair quality setting: there
is no memory pressure at 20 k. Re-running the same heads at `max_leaves = min(N, 16 000)` (still
bounded, still `O(N)`) — an honest de-handicap, not a tuned number (`results_real_hires.csv`):

| method | digits (64-D, `ml`=1797) | covtype (54-D, `ml`=16000) | mnist (784-D, `ml`=16000) |
|---|---|---|---|
| **betula-kmeans** | 0.467 | **0.067** | 0.325 |
| sklearn-kmeans | 0.468 | 0.054 | 0.324 |
| **betula-gmm** (diag) | 0.461 | **0.104** | 0.267 |
| **betula-gmm-full** | **0.575** | **0.103** | — |
| sklearn-gmm (full) | 0.463 | 0.080 | — |

- **covtype GMM improves with resolution**, 0.076 → 0.104, and only here does it clear
  scikit-learn's full GMM (0.080) by more than a seed spread; `gmm-full` likewise 0.076 → 0.103. It
  still does not reach `sklearn-birch`'s 0.131. covtype **k-means goes the other way**, 0.074 → 0.067
  — resolution is not monotone even within one dataset.
- **digits** is unchanged — its 1797 points already fit under 4 000 leaves, so at `max_leaves=1797`
  every leaf holds one point and the summary is lossless. That also means both digits columns are
  measured at **zero compression** and say nothing about summarization; the leaf-budget sweep below
  shows they are on the wrong side of the peak, since halving the leaves *raises* ward from 0.643 to
  0.682 and k-means from 0.467 to 0.560.
- **mnist k-means** closes to **0.325 vs 0.324**, a tie; the diagonal GMM gains far less
  (0.234 → 0.267), which is the resolution/over-fragmentation trade the head pays in 784 dimensions.

### `normalize=True` — a direction fix that no longer helps MNIST

`normalize=True` L2-normalizes rows so the heads cluster by *direction*. Median of three seeds
(`results_real_normalize.csv`; spread of the "on" column in `results_real_normalize_spread.csv`):

| `normalize` off → **on** | betula-kmeans | betula-gmm (diag) | betula-ward |
|---|---|---|---|
| digits (64-D) | 0.467 → **0.569** | 0.461 → 0.387 | 0.643 → **0.699** |
| mnist (784-D) | 0.307 → 0.346 | 0.234 → 0.258 | 0.377 → 0.380 |
| covtype (54-D) | 0.074 → 0.005 | 0.076 → 0.053 | 0.091 → **−0.049** |

**Correction to the previous edition, which reported MNIST k-means 0.203 → 0.334 as the flagship
result.** The 0.203 baseline is gone: on this tree raw MNIST k-means is already 0.307, and the
normalized median of 0.346 sits inside its own seed spread — and the off-vs-on sign
*flips between seeds*. Whatever the flag was
compensating for in 784-D, the `[Unreleased]` tree-rebuild fix removed most of it. Reported as a wash.

Where it still earns its place is `digits`: **k-means 0.467 → 0.569 and ward 0.643 → 0.699**, stable
across all three seeds. And it remains **off by default on purpose** — magnitude *is* signal on
ordinary tabular data, where unit-normalizing destroys the clustering (covtype ward 0.091 → −0.049,
i.e. worse than random).

### Real data at scale — full covtype (581 012 × 54)

Clustering a **real** half-million-row dataset, each run isolated in its own subprocess (peak RSS from
`/proc/self/statm`):

| method | time | peak RSS | ARI |
|---|---|---|---|
| **betula-kmeans** | **1.06 s** | 0.91 GB | **0.070** |
| sklearn-kmeans | 4.92 s | 0.93 GB | 0.049 |

betula-kmeans clusters the full 581 k-row covtype **4.7× faster** than scikit-learn KMeans — at the
same memory and, on this run, a higher ARI (0.070 vs 0.049), on real data rather than blobs. This is
a single seed, like every row in the speed suite, and the 20 k subsample's three-seed spread
(0.071–0.102 for the same head) is wide enough that the ARI column here should be read as "not
worse", not as a 43 % lead.

## Structured covariance — `gmm-toeplitz` / `gmm-toeplitz-full` on stationary signals

For **ordered, wide-sense-stationary** signals (fixed-length time-series windows, waveforms, sensor
traces) neither diagonal nor full covariance is the right prior: diagonal ignores the neighbour
correlation that *is* the signal, and full has `O(d²)` parameters and is singular exactly when
`N_k ≪ d`. `method="gmm-toeplitz"` models each component covariance as **AR(w) / Toeplitz-structured**
(Levinson-Durbin → exact Gohberg-Semencul precision, order by BIC), `O(w)` parameters; the companion
`method="gmm-toeplitz-full"` drops the AR order cap for a **general positive-definite Toeplitz**
covariance (dense, from the biased autocovariance); `method="gmm-toeplitz-gs"` is the rung between
them (GS-MLE precision at a capped order) — all positive-definite by construction, see
[ADR 001](../docs/adr/001-gmm-toeplitz.md).

The adversarial test (`bench/toeplitz_ar_mixture.py`): a 3-component mixture of AR processes that
differ **only** in autocovariance (each window rescaled to unit marginal variance, so the signal is
entirely in the covariance *structure*), 30 windows per component, ARI vs window length `d`:

| d (window) | N_k/d | **gmm-toeplitz** | **-toeplitz-full** | **-toeplitz-gs** | betula-diag | betula-full | sklearn-diag | sklearn-full |
|---|---|---|---|---|---|---|---|---|
| 32  | 0.94 | **0.521** | 0.487 | 0.511 |  0.013 | −0.001 |  0.075 | −0.009 |
| 64  | 0.47 | 0.699 | **0.721** | 0.637 | −0.015 | −0.007 |  0.014 |  0.019 |
| 128 | 0.23 | 0.966 | 0.966 | **1.000** | −0.008 | −0.005 | −0.001 |  0.028 |
| 256 | 0.12 | **1.000** | **1.000** | **1.000** | −0.015 |  0.023 | −0.014 | −0.002 |

All three Toeplitz rungs recover the components — **improving with `d`** to perfect separation,
precisely the regime (`N_k ≪ d`) where diagonal is blind and full is singular; every non-Toeplitz head
sits at chance.

**Where the general head is *required*.** AR(w) has a *banded* precision, so it cannot represent an
autocovariance whose support exceeds order `w`. A mixture whose components differ only by a **single
echo at lag `K ∈ {16, 28, 40}`** (all beyond the cap `w_max = 10`) is invisible to AR:

| d (window) | N_k/d | gmm-toeplitz (AR) | **-toeplitz-full** | -toeplitz-gs | betula-diag |
|---|---|---|---|---|---|
| 64  | 0.47 | −0.005 | **0.731** | 0.376 |  0.002 |
| 96  | 0.31 | −0.015 | **0.934** | 0.487 | −0.014 |
| 128 | 0.23 | −0.011 | **1.000** | 0.500 | −0.001 |
| 192 | 0.16 | −0.006 | **1.000** | 0.523 | −0.007 |

The AR head is at chance (a lag-`K > w` spike is unreachable by any order-`w` model); the general head
climbs to **1.00**; `-toeplitz-gs` captures only the lags inside its order cap (≤ 16), which is why it
plateaus near 0.5. All three are **experimental / off by default** and scoped to ordered stationary
signals — on generic embeddings (no coordinate order) the structure is meaningless; use `gmm` /
`gmm-full` there. Guidance: reach for `gmm-toeplitz` first (`O(d·w)`, fast); switch to
`gmm-toeplitz-full` (`O(d³)`) when the structure lives beyond a low AR order.

## CF-weighted NMF — quality that is stable, speed that is not a win

`projection="weighted-nmf"` factorizes the ≤ `max_leaves` leaf centroids with their masses as weights
rather than all `N` points (`bench/nmf_cf_weighted.py`, nonnegative topic mixtures, d = 60, k = 4,
NMF rank 8 → k-means, median over 8 seeds):

| N | sklearn NMF → k-means | CF-weighted NMF | speed |
|---|---|---|---|
| 8 000 | ARI 0.812 ± 0.372 · 0.05 s | **ARI 1.000 ± 0.000** · 0.22 s | 0.2× |
| 40 000 | ARI 0.991 ± 0.372 · 0.26 s | **ARI 1.000 ± 0.000** · 0.36 s | 0.7× |
| 160 000 | ARI 0.967 ± 0.378 · 0.92 s | **ARI 1.000 ± 0.000** · 1.00 s | 0.9× |

The quality result is the one that matters and it is about **determinism, not accuracy in the mean**:
NMF is invariant to `(W D, D⁻¹ H)`, so an unpinned split lets one component's arbitrary scale dominate
the Euclidean geometry the head then clusters — hence scikit-learn's ±0.37 seed spread against
betula's ±0.00. betula returns a canonical factorization (unit-L2 parts, energy-ordered).

**The speed column is a loss and is printed as one.** At every `N` measured, the CF-weighted path is
*slower* (0.2× / 0.7× / 0.9×); the gap closes as `N` grows but has not crossed by 160 k. Reach for
this for the reproducibility, or for bounded memory at `N` beyond what a dense NMF can hold — not for
throughput at these sizes.

## Sparse text — 20 newsgroups (TF-IDF)

18 846 documents × 2 000 TF-IDF features clustered into the 20 ground-truth topics, each method
isolated in its own subprocess (`bench/results_sparse.csv`):

| reduction | clusterer | time | ARI |
|---|---|---|---|
| raw 2 000-D (none) | betula `fit_predict_sparse` (O(nnz)) | 1.94 s | 0.004 |
| raw 2 000-D (none) | sklearn k-means | 0.61 s | 0.056 |
| **betula CF-weighted PCA(50)** | **betula** spherical k-means | 3.31 s | **0.164** |
| TruncatedSVD(50) | sklearn k-means | **0.41 s** | 0.130 |
| NMF(20) | **betula** k-means | 2.14 s | 0.130 |
| NMF(20) | sklearn k-means | 2.42 s | 0.124 |

The two betula times fell **3.4×** and **1.5×** against the previous edition of this table with the
labels bit-identical at every budget — task #71, the transposed micro-cluster storage described
below. Nothing about the partition changed; only the memory traffic it took to reach it.

The `betula-svd` row changed meaning. It used to call scikit-learn's `TruncatedSVD` and then
cluster with betula, so it measured scikit-learn's reducer; it now runs betula's own
`projection="svd"` — a CF-weighted PCA of the leaf summary — end to end in one call. The `betula-nmf`
row still borrows scikit-learn's NMF, and says so.

Quality against the leaf budget, since the harness pins `max_leaves=2048` for every sparse row and
that is the whole cost here. Unlike the table above — which `bench/comprehensive.py` runs at `seed=0`
only, like every other sparse row — this sweep is the **median of seeds 0/1/2**, one BLAS thread, so
its 2048 cell (0.152, spread [0.135, 0.164]) reads lower than the 0.164 above: seed 0 is the top of
that spread, not a different measurement.

| `max_leaves` | ARI | time, before #71 | time, after |
|---|---|---|---|
| 128 | 0.097 | 0.22 s | 0.20 s |
| 256 | 0.130 | 0.43 s | 0.37 s |
| 512 | 0.143 | 0.87 s | 0.74 s |
| 1024 | 0.150 | 2.62 s | 1.53 s |
| 2048 | 0.152 | 6.99 s | **3.44 s** |

All fifteen label digests (five budgets × three seeds) are identical on both sides, so the ARI column
is the *same* measurement re-timed. Two cells read 0.001 below the previous edition of this table
(128 and 512); that predates this change — the earlier numbers were taken on an older build, which is
also why its 2048 time (5.42 s) sits between the two columns above.

Read honestly:

- **Raw high-dimensional TF-IDF is the wrong input for any compression / fast clusterer.** At
  `d = 2 000` Euclidean distances concentrate, so the O(nnz) sparse-native path (0.004, ≈ random) and
  even raw sklearn k-means (0.056) barely beat chance. The standard fix for sparse text is
  **reduce-then-cluster**, and `projection="svd"` now does it inside the same call.
- **On quality the leaf-summary PCA wins; on time it does not.** 0.164 against `sklearn-svd`'s 0.130
  at this budget, but 3.31 s against 0.41 s. The basis is not the compromise — labelling raw rows in
  it scores 0.159 against 0.143 for `TruncatedSVD`'s own basis on the same rows, since the within-leaf
  scatter the summary discards is isotropic under the spherical cluster feature and so moves no
  direction. The time is the **leaf budget**: sweeping the rank from 1 to 100 moves the total by
  1.2 s, while `max_leaves` 256 → 2048 moves it from 0.37 s to 3.4 s, because both sparse passes are
  flat scans that score each row against every micro-cluster. At 256 leaves the same call is 0.130 ARI
  in 0.37 s — scikit-learn's quality at close to its speed.

### The 20news rows were memory-bound, not compute-bound (task #71)

The scan is linear in the leaf budget by construction, so the budget sweep is the cheap test of where
the time goes: if the leader pass dominates, wall time is linear in `max_leaves`. It was worse than
linear. Doubling 512 → 1024 cost **6.3×**, and 128 → 2048 (16× the budget) cost 76×.

That elbow is not an algorithm, it is a cache. Each micro-cluster held its coordinate sum as its own
dense `Vec<f64>` of length `n_features`, so scoring one row against `L` of them pulled `L · nnz`
scattered cache lines out of `L · n_features · 8` bytes of state — 32 MB at `L = 2048` on this
fixture, none of it reused before eviction. The jump sits exactly where that working set leaves L3.

Both sparse passes now hold the coordinates **feature-major** — `[c · L + i]` rather than a vector per
cluster — so a row walks `nnz` contiguous runs of `L` doubles instead. The arithmetic is untouched and
deliberately so: for a fixed cluster the row's terms are still accumulated in row order, which is why
every label digest above is unchanged. Measured on the raw `method="kmeans"` path, seed 0:

| `max_leaves` | before | after | |
|---|---|---|---|
| 128 | 0.169 s | 0.081 s | 2.09× |
| 512 | 0.789 s | 0.315 s | 2.50× |
| 1024 | 4.942 s | 0.840 s | **5.88×** |
| 2048 | 12.764 s | 2.185 s | **5.84×** |

The super-linear elbow is gone (1024 → 2048 now costs 2.60×, against 2.58× at the low end where the
old layout was still cache-resident). The speedup is largest exactly where the old code fell out of
cache, which is what makes the diagnosis a mechanism rather than a story.

A `perf` profile taken between the two halves of the change is what kept it honest: after fixing the
summarisation pass alone the fit was only 1.5× faster, and the profile put **65% of remaining samples
in the row-labelling pass** and 6% in summarisation. The task named summarisation; the profile named
labelling. Both have the same defect and both are fixed, and the 3.4× on the headline row is mostly
the half the task did not ask for.
- **Use a cosine head on the codes.** `method="kmeans"` on the same codes scores 0.014 against
  `spherical-kmeans`'s 0.152: the leading principal direction of a TF-IDF corpus is document length,
  and only an angular objective ignores it.
- Net: `fit_predict_sparse` alone is a **scale / bounded-memory** tool, not a quality lever on high-`d`
  text. With `projection="svd"` it becomes the quality tool too, and the knob that matters is
  `max_leaves`.

## Specialist baselines — FAISS and `fast_hdbscan`

Everything above measures betula against scikit-learn, which is the right *default* comparison and
the wrong *hardest* one. `bench/external_baselines.py` asks two questions of the strongest specialist
on each axis instead. Neither library is a project dependency and neither may become one; both are
pulled per invocation:

```bash
uv run --with faiss-cpu --with fast-hdbscan --with scikit-learn --with pandas \
    python bench/external_baselines.py
```

Single-threaded, seed 0, `max_leaves = 2000`, one subprocess per row so peak RSS is that method's own.
`min_samples = 10` and `min_cluster_size = n/400` are handed to every method identically, in points.

| contest | n | method | time | peak RSS | clusters found | ARI |
|---|---:|---|---:|---:|---:|---:|
| k-means, `highdim`, k=8 | 200 000 | **betula-kmeans** | 0.19 s | 219 MB | 8 | **1.000** |
| | 200 000 | faiss-kmeans, defaults | **0.07 s** | 208 MB | 8 | 0.630 |
| | 200 000 | faiss-kmeans, matched | 1.19 s | 208 MB | 8 | **1.000** |
| | 200 000 | sklearn-kmeans | 1.40 s | 266 MB | 8 | **1.000** |
| | 1 000 000 | **betula-kmeans** | **0.70 s** | 538 MB | 8 | **1.000** |
| | 1 000 000 | faiss-kmeans, defaults | **0.08 s** | 538 MB | 8 | 0.624 |
| | 1 000 000 | faiss-kmeans, matched | 5.38 s | 538 MB | 8 | 0.835 |
| | 1 000 000 | sklearn-kmeans | 5.00 s | 655 MB | 8 | **1.000** |
| HDBSCAN, `blobs`, k=6 | 100 000 | **betula-hdbscan** | **0.18 s** | **162 MB** | 3 | 0.478 |
| | 100 000 | fast-hdbscan | 1.65 s | 315 MB | 6 | **0.910** |
| | 500 000 | **betula-hdbscan** | **0.32 s** | **175 MB** | 3 | 0.478 |
| | 500 000 | fast-hdbscan | 3.02 s | 411 MB | 6 | **0.892** |

One contest is a win once both sides answer the same question; the other is a loss on quality and a
win on cost. Both are recorded as such rather than dropped.

### The FAISS row is two rows, because FAISS's defaults are not a like-for-like fit

**At its own defaults FAISS is 2.7×–8.8× faster and does not recover the partition** (0.62–0.63
against betula's 1.000). Task #73 asked whether that is a throughput gap worth closing. It is not,
and the reason is that two of the three things making the default row fast are quality-costing
shortcuts rather than engineering:

- **`faiss.Kmeans` defaults to `max_points_per_centroid = 256`.** At `k = 8` it trains on **2 048 of
  the 200 000 rows** and then assigns the rest. The 0.07 s is a fit on 1% of the data.
- **It seeds from a random subset, never k-means++**, and runs `nredo = 1` — a single random init.
- Only the third — a fixed 25 Lloyd iterations over float32 with hand-written SIMD kernels — is
  throughput.

The `matched` row hands FAISS every row and ten restarts, the cheapest setting whose *median* ARI
reaches betula's on this fixture. It costs **1.19 s at 200 000 (6.3× betula) and 5.38 s at 1 000 000
(7.7× betula), and at 1 M it still lands at 0.835.** So there is no budget at which FAISS answers
betula's question faster than betula does.

The ceiling is initialisation, and sklearn is the control that proves it. Same fixture, same `k`,
every row used, medians over seeds 0/1/2 (`local/scratch/faiss_mechanism.py`):

| method | median time | median ARI | across seeds |
|---|---:|---:|---|
| betula-kmeans | 0.16 s | **1.000** | — |
| faiss, 2 048 train rows, `nredo=1` | 0.008 s | 0.693 | [0.630, 0.718] |
| faiss, all rows, `nredo=1` | 0.114 s | 0.835 | [0.692, 0.835] |
| faiss, all rows, `nredo=3` | 0.401 s | 0.835 | [0.835, 0.835] |
| faiss, all rows, `nredo=10` | 1.208 s | **1.000** | [0.835, 1.000] |
| sklearn, `init="random"`, `n_init=1` | 0.077 s | 0.692 | [0.692, 0.717] |
| sklearn, `init="random"`, `n_init=10` | 0.320 s | 0.835 | [0.835, 1.000] |
| sklearn, `init="k-means++"`, `n_init=1` | 0.107 s | **1.000** | [1.000, 1.000] |

Read the last two rows against each other: one random init reaches 0.69, ten reach 0.835, and **one
k-means++ init reaches 1.000 on every seed**. ARI 0.835 is not an unlucky draw, it is a persistent
local optimum — two of the eight blobs merged and one split — that random restarts land in and cannot
climb out of. Adding data does not help (all-rows `nredo=1` is no better than the 1% subsample);
adding iterations does not help (ARI is flat from `niter = 25` to `niter = 300`); only the seeding
rule helps.

That is the whole finding. betula's advantage here is not a faster kernel — it is that it runs
k-means++ on the exact CF potential (`8cb3439`), on a leaf summary small enough that ten restarts are
free. **Task #73 is closed by measurement: the gap was never throughput.**

**`fast_hdbscan` finds all six clusters at every setting; betula needs a much larger `min_samples`
and a much larger leaf budget to get there.** The `blobs` fixture is not the easy case its name
suggests: six centres drawn uniformly from `[-10, 10]²` with unit spread, then standardized, leaves
the closest pair **0.37 apart at unit width** — heavily overlapping, which is the regime this page
already credits raw HDBSCAN with owning. Swept on n = 100 000 with `min_cluster_size = 250`:

| `min_samples` | 10 | 100 | 1 000 | 2 000 |
|---|---:|---:|---:|---:|
| fast-hdbscan | **0.910** (6) | 0.900 (6) | 0.845 (6) | 0.762 (5) |
| betula, `max_leaves` 2 000 | 0.478 (3) | 0.566 (4) | 0.785 (5) | 0.762 (5) |
| betula, `max_leaves` 8 000 | 0.566 (4) | 0.799 (5) | **0.843** (6) | 0.764 (5) |

So best-against-best the gap is 0.910 in ~1.4 s against 0.843 in 2.1 s, not the 0.910-against-0.478
the contest row shows — and the shape of the table is the mechanism. HDBSCAN\* separates overlapping
densities through the core distance, the radius enclosing `min_samples` points. Over raw points that
radius is small and varies with local density; over leaf centroids a single leaf already holds
n/`max_leaves` = 50 points, so any `min_samples` below that is enclosed at **distance zero**, every
core distance collapses, and mutual reachability degenerates to plain distance — single linkage,
which chains straight through the overlap. Raising `min_samples` past the leaf mass, or raising
`max_leaves` until the leaf mass drops below `min_samples`, restores the estimate; both columns of the
table move for that one reason.

That points at the fix rather than at a tuning note: the leaf is not a point, and its own mass should
be enclosed at its own radius, which the cluster feature already carries as `√(ssd/weight)`. Task #72
owns it.

The units trap found on the way is fixed as of this edition, and was the more serious half. On the
summary route `min_cluster_size` and `min_samples` used to be counted in **leaves**: `hdbscan.rs`
thresholded `node_size`, a leaf count, while stability used `node_mass`, a point count. The
point-level value a scikit-learn user passes — 1 250 at n = 500 000 — therefore asked for 1 250 of
2 000 leaves and returned **zero clusters, ARI 0.0000, with no warning**; it was also not scale-free,
since the threshold changed meaning whenever `max_leaves` did. Both arguments now count points. The
n = 500 000 row moved 0.000 → 0.478, which is exactly the n = 100 000 row: the same question now gets
the same answer at both scales.

One caveat on the timings: `fast_hdbscan` is numba-compiled, so its first call in a cold process pays
JIT — measured at 9.0 s against 0.3 s once the on-disk cache is warm. The table above is warm-cache.

### The scoreboard

`bench/scoreboard.py` reads every committed `results_*.csv`, pairs each betula row against its rivals
three ways — `vs-same` (like-for-like algorithm), `vs-best` (each side's champion on a slice, chosen
by its **worst** seed so a lucky median cannot absorb a real gap), `vs-external` (the table above) —
and prints one verdict per cell under the tie rule this page already states: a difference smaller than
the wider of the two cells' three-seed spreads is not a result. Tables with no spread sidecar are
single runs by construction and fall back to a per-axis tolerance.

```
## quality — 8 win · 54 tie · 6 loss
## speed   — 33 win · 0 tie · 5 loss
## memory  — 30 win · 4 tie · 0 loss
```

`bench/scoreboard.json` records those 140 verdicts; `--check` re-derives them and exits non-zero if
any cell got worse or vanished, and `--update` is the deliberate act of accepting a new board. The
six quality losses are the `covtype` and `digits` rows discussed above, the raw-TF-IDF `20news` row,
MNIST k-means, and the two `fast_hdbscan` cells; the five speed losses are the two FAISS cells and the
three `20news` rows the sparse scans own — 3.4× and 1.5× closer after task #71, but still losses.

**The ratchet fired on this edition, and it was right to.** Accepting the new board took one genuine
demotion and six vanished pairings, listed here because the whole point of the file is that a board is
accepted deliberately rather than overwritten:

- `results_sparse/speed/vs-same/20news/betula-nmf`: **win → tie.** The NMF pipeline went 2.30 s →
  2.08 s while scikit-learn's went 2.55 s → 2.19 s, which closes the margin below the single-run
  tolerance. Nothing got slower; the rival got faster by more than we did, on a row where both sides
  spend most of their time inside the same scikit-learn NMF call.
- Six `vs-best` pairings **vanished** rather than lost. `vs-best` names the rival in its key, and the
  non-betula champion changed identity: at 1 M `sklearn-kmeans` (2.45 s) is now faster than
  `sklearn-minibatch` (2.76 s), where the previous edition had them the other way round. The verdicts
  survive under the new rival's name, which is why the speed column moved *up* (31 → 32 wins) in the
  same run that reported six disappearances. A pairing key that embeds the opponent cannot tell
  "we lost" from "the opponent changed", so the tool reports both and leaves the call here.
- Task #71 moved the speed column to **33 win · 0 tie · 5 loss** and vanished two more pairings, both
  re-namings rather than regressions. `results_sparse/speed/vs-best/20news` names the *fastest betula*
  row in its key, and `betula-sparse` (6.57 s → 1.94 s) has overtaken `betula-nmf` (2.14 s) for that
  slot, so the cell survives as `betula-sparse-vs-sklearn-svd` and is still a loss.
  `results_sparse/memory/vs-best/20news` re-named for the opposite reason — the best non-betula RSS
  moved from `sklearn-nmf` to `sklearn-svd` by **0.004 MB**, which is the pairing key being more
  precise than the measurement. The one genuine promotion is `results_sparse/speed/vs-same/betula-nmf`
  **tie → win**, and it is not ours: `betula-nmf` went 2.08 s → 2.14 s while `sklearn-nmf` went 2.19 s
  → 2.42 s, so the margin re-opened because the rival got slower. Both sides sit inside the same
  scikit-learn NMF call; treat the cell as noise in both directions.
- Adding the `faiss-kmeans-matched` row **vanished** one more pairing the same way:
  `results_external/quality/vs-external/highdim/200000/…/betula-kmeans-vs-sklearn-kmeans`. The new
  row ties betula at ARI 1.000 and becomes the named rival for that cell, so the verdict survives as
  `…-vs-faiss-kmeans-matched`. The three column totals are unchanged (8/54/6, 32/1/5, 30/4/0): a
  configuration of FAISS that reaches betula's quality also costs 6.3× the time, so it converts a
  quality cell from tie-against-sklearn to tie-against-FAISS and adds nothing on speed.

## Where the leaf budget goes — geometry, not mass (tasks #70 and #77)

Two open questions about the budget, settled together because one measurement answers both.

**Is the budget under-used?** No — that claim was about ELKI and does not transfer. Realised leaves
over `max_leaves`, medians of seeds 0/1/2, `threshold=0`:

| dataset | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| covtype-20k | 0.96 | 0.95 | 0.96 | 0.98 | 0.90 |
| mnist-20k | 0.94 | 0.96 | 0.97 | 0.90 | 0.94 |
| blobs-100k | 0.97 | 0.95 | 0.95 | 0.93 | 0.95 |
| highdim-100k | 0.97 | 0.96 | 0.96 | 0.97 | 0.97 |

The tree fills **90–98%** of its budget everywhere. There is no unused-budget lever here.

**Is the budget well *spent*?** No, and that is the real defect. The same run, share of total mass in
the heaviest leaf (`top1`) and the heaviest 10% of leaves (`top10`):

| dataset | budget | Gini | top1 | top10 | heaviest leaf |
|---|---:|---:|---:|---:|---:|
| mnist-20k | 250 | 0.979 | **0.831** | 0.984 | 16 625 of 20 000 |
| mnist-20k | 1000 | 0.938 | 0.360 | 0.948 | 7 193 |
| covtype-20k | 500 | 0.886 | 0.149 | 0.851 | 2 976 |
| covtype-20k | 4000 | 0.683 | 0.015 | 0.613 | 295 |
| imbalanced-100k | 4000 | 0.949 | **0.800** | 0.952 | **80 000** |

The `imbalanced` fixture is the clean case: 80 000 points in a tight core, 20 000 spread across five
diffuse minorities ten times wider. **The entire core lands in one leaf at every budget from 250 to
4000.** With 3 792 leaves realised, 3 791 of them go to the 20% of the mass that happens to be spread
out. The budget is spent by geometry — how far apart points are — and not by mass.

**The mechanism is the single global threshold.** The rebuild heuristic raises one absorption radius
until the leaf count fits under `max_leaves`, and stops as soon as it does. One global radius cannot
serve two densities: at `max_leaves=4000` the threshold settles at 0.705, which is still wider than
the core's whole diameter, so the core cannot split — and lowering it far enough to split the core
would explode the minorities past any budget.

**What it costs.** `structured` gives the core internal structure — two true clusters inside it — so
collapsing it to one leaf makes them unrecoverable by *any* Phase-3 head. `flat` is the control with
the same mass profile and no internal structure. Medians of seeds 0/1/2:

| fixture | sklearn-kmeans (raw points) | betula @250 | @1000 | @4000 |
|---|---:|---:|---:|---:|
| structured (k=7) | **1.0000** | 0.4174 | 0.4174 | 0.4174 |
| flat (k=6) | 1.0000 | **1.0000** | 1.0000 | 1.0000 |

Sixteen times the budget buys **nothing** on `structured`, and `kmeans` / `ward` / `gmm` agree to
three decimals — this is not a head choosing badly, it is a summary that no longer contains the
answer. On `flat` the identical collapse is free. This reproduces the shape of scikit-learn's Birch
issue #22854 on our own tree, and it is the mechanism behind the `covtype` and MNIST rows above,
where `top10` reaches 0.61–0.98. Fixing it means a budget that is allocated by mass rather than by
radius — see `balance`, below.

### The fix, and the honest range over which it is one (`balance`)

`balance = b` caps a leaf at `b × (mass / max_leaves)`, refusing absorption into a full leaf and
skipping the same pairs at compaction; `max_leaves` stays a hard bound. On the fixture that motivated
it, `kmeans`, medians of seeds 0/1/2:

| budget | off | b=8 | b=4 | b=2 | b=1 |
|---|---:|---:|---:|---:|---:|
| structured @250 | 0.4174 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (one seed 0.4191) |
| structured @1000 | 0.4174 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| structured @4000 | 0.4174 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| flat @250–4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

`top1` falls from 0.800 to 0.001–0.03 and the realised leaf count barely moves, so this is the same
budget spent differently rather than a larger one. `b = 1` asks for perfect balance and is the only
setting that is ever unstable; 2–8 are indistinguishable here.

**On data that is not built to expose it, it is a lever and not a free win.** All three exact heads,
`feature="spherical"`, medians of seeds 0/1/2, `balance=4` against `balance=None` — 19 of 27 cells
improve, but read the `top1` column, not the average:

| dataset | budget | top1 off → on | kmeans | ward | gmm |
|---|---:|---|---:|---:|---:|
| mnist-10k | 250 | 0.594 → 0.016 | **+0.1676** | **+0.2514** | **+0.1873** |
| mnist-10k | 500 | 0.520 → 0.008 | **+0.1167** | **+0.0785** | **+0.2090** |
| mnist-10k | 2000 | 0.108 → 0.002 | +0.0219 | −0.0407 | +0.0346 |
| covtype-20k | 250 | 0.176 → 0.016 | −0.0078 | +0.0001 | −0.0247 |
| covtype-20k | 1000 | 0.069 → 0.004 | +0.0065 | +0.0007 | +0.0029 |
| covtype-20k | 4000 | 0.011 → 0.001 | +0.0247 | +0.0004 | −0.0448 |
| digits | 225 | 0.135 → 0.017 | −0.0424 | +0.1894 | +0.1261 |
| digits | 450 | 0.089 → 0.008 | +0.0379 | −0.0656 | +0.2444 |
| digits | 900 | 0.030 → 0.004 | −0.0684 | −0.0004 | +0.2164 |

The rule the table gives is **the heaviest leaf's share of the mass predicts the gain**. Every cell
where `top1 ≥ 0.5` improves, by +0.08 to +0.25 with all three heads agreeing; where `top1 < 0.1` the
change is within seed noise in both directions. That is the diagnostic to run before reaching for the
parameter — `max(microcluster_weights_) / sum(...)` — and it is why the default stays off: on a
well-spread summary the cap trades geometry for balance and buys nothing.

One cross-effect worth naming rather than claiming: the large `gmm` gains on `digits` (+0.216 at 900
leaves, +0.244 at 450) are in the *low*-`top1` regime, and they are the isotropic-scatter collapse
from the section below — a lighter leaf carries less `ssd`, so there is less isotropic variance to
inflate every dimension with. That is a side-effect of the cap, not the mass-balance argument for it.

### It is the CF-tree family, not this implementation (task #47)

scikit-learn's Birch issue #22854 reports the same shape from the other implementation, so the
comparison belongs in the benchmark rather than in a footnote. `bench/size_imbalance.py`, both
fixtures, medians of seeds 0/1/2 for the betula rows (Birch is deterministic and runs once):

| fixture | method | budget | leaves | ARI | top1 |
|---|---|---:|---:|---:|---:|
| structured | sklearn-kmeans (raw points) | — | — | **1.0000** | — |
| structured | sklearn-birch (defaults, `threshold=0.5`) | — | 7 140 | 0.3600 | 0.800 |
| structured | sklearn-birch, threshold matched to the budget | 250 | 210 | 0.4180 | 0.800 |
| structured | sklearn-birch, threshold matched to the budget | 1000 | 971 | 0.3535 | 0.800 |
| structured | sklearn-birch, threshold matched to the budget | 4000 | 4 183 | 0.3266 | 0.800 |
| structured | betula-kmeans | 250 / 1000 / 4000 | 241 / 914 / 3 767 | 0.4174 | 0.800 |
| flat | sklearn-birch (defaults) | — | 7 162 | 0.8519 | 0.800 |
| flat | sklearn-birch, matched | 1000 | 987 | 0.8521 | 0.800 |
| flat | betula-kmeans | 250 / 1000 / 4000 | 232 / 920 / 3 792 | **1.0000** | 0.800 |

Three things this settles. **`top1 = 0.800` in every row**: both trees put the entire core in one
leaf, so the mis-allocation is a property of the shared design — one global absorption radius — and
not of either implementation. **Birch is measured at its defaults *and* matched**, because its
default threshold produced ~7 000 subclusters against budgets of 250–4 000, and a rival compared only
at its defaults is a rival misreported (the same discipline as the FAISS row above). **More leaves
make Birch worse** on `structured` — 0.4180 → 0.3266 as its subcluster count goes 210 → 4 183 —
which is what mis-allocation looks like when the extra budget goes to the minorities.

betula is on the better side of the shared defect at every budget, and on `flat` it recovers the
answer exactly where Birch loses 15 points. That is worth saying plainly: this row is not a win
claimed over a fixed rival, it is the same known failure measured in both.

## Quality against the leaf budget — the knob the tables never varied

`bench/leaf_budget.py`, median of seeds 0/1/2, `feature="spherical"`, `threshold=0.0`, crossed over
the four routing distances; `bench/results_budget.csv` holds all 252 cells. Every table above this
one fixes `max_leaves` and varies the head, which answers the wrong question for a summarization
library: the user's knob is the budget.

**The `digits` rows above run at zero compression, and that is a defect in the record.** At
`max_leaves=4000` against `n=1797` the sweep measures 1797 leaves, ×1.0, maximum leaf weight **1**,
mean squared radius **0** — every leaf is a single point. Re-running at `max_leaves=1797` reproduces
those cells to the digit. That row is raw-point clustering behind a betula wrapper and cannot support
any claim about summarization.

It also understates the library. On `digits` the curve has an **interior optimum**, not a monotone
decline:

| `max_leaves` | leaves | compression | max leaf weight | mean sq radius | ward | k-means | gmm |
|---|---|---|---|---|---|---|---|
| 4000 / 1797 | 1797 | ×1.0 | 1 | 0.000 | 0.6428 | 0.4670 | 0.4613 |
| 900 | 898 | **×2.0** | 54 | 4.689 | **0.6819** | **0.5600** | 0.0088 |
| 450 | 427 | ×4.2 | 160 | 12.47 | 0.6197 | 0.5241 | 0.2139 |
| 225 | 218 | ×8.2 | 243 | 20.77 | 0.3909 | 0.5628 | 0.4105 |
| 112 | 111 | ×16.2 | 605 | 30.72 | 0.2870 | 0.1903 | 0.2897 |
| 90 | 89 | ×20.2 | 1102 | 34.64 | 0.1101 | 0.1786 | 0.2124 |
| 45 | 44 | ×40.8 | 1251 | 39.21 | 0.1778 | 0.1198 | 0.1666 |

Halving the leaf count **improves** ward (0.6428 → 0.6819) and k-means (0.4670 → 0.5600). The
summary is not a lossy approximation of the point-level answer there; it is a denoising step, and
the published zero-compression row is on the wrong side of the peak.

`covtype-20k` behaves differently and more usefully: the `ward` head holds 0.1412–0.1430 from ×11.1
all the way to **×202** (99 leaves, maximum leaf weight 5773), and the order study below reads
0.1416 for the same head at ×1.0 — two orders of magnitude of compression cost nothing measurable,
and the best `covtype` cell in the whole sweep (0.1430) is the *most* compressed one. `mnist-10k` degrades gently to ×5.5 (k-means 0.2900 → 0.2725, ward 0.3419 → 0.3228)
and then falls off a cliff between ×10 and ×22.

Two further readings the sweep settles:

- **The `gmm` head is the fragile one, and the cause is a feature/head mismatch (task #89).** It
  collapses to 0.0088 on `digits` at ×2.0 and to 0.0618 / 0.0512 on MNIST at ×5.5 / ×10, in cells
  where k-means and ward are still near their best. Nothing in the fixed-budget tables exposed this,
  because they never crossed the region where it happens. The mechanism is below.

### Why `gmm` collapses under compression, and why `k-means` does not

Three measurements, each ruling out the previous explanation.

**It is not a degenerate fit.** At the 0.0088 cell the fitted model has ten non-empty components, the
largest holding 30% of the points, and cluster radii in the same range as every healthy cell. No
variance spike, no merged blob, no empty component.

**It is not the summary.** Labelling the *same* points three ways off the *same* fitted model:

| leaf budget | maximum posterior | nearest fitted centre | via each point's leaf |
|---|---|---|---|
| 1797 (×1.0) | 0.4613 | 0.5132 | 0.5178 |
| 898 (×2.0) | **0.0088** | 0.5288 | 0.5296 |
| 484 (×3.7) | **0.0104** | 0.4988 | 0.5267 |
| 296 (×6.1) | 0.4650 | 0.5004 | 0.4849 |

The centres are healthy at every budget. Only the posterior collapses, and the only thing the
posterior uses that a nearest-centre rule does not is the fitted covariance.

**It is the covariance, and one line of code says why.** `Spherical::variance(_d)` ignores its
argument — it returns `ssd / (w · dim)`, one **isotropic** number for every dimension, because a
spherical cluster feature carries a scalar scatter and cannot carry more. The diagonal M-step adds
that number to all `dim` per-component variances. At ×1.0 every leaf is a singleton, `ssd = 0`, and
nothing is added. Under compression each component is inflated **equally in every dimension** by
however much leaf scatter it happens to cover, so a dimension with genuinely near-zero variance
(`digits` has constant border pixels) is lifted to the isotropic average. In 64 dimensions the
maximum-posterior argmax is dominated by `ln|Σ_c| = Σ_d ln σ²_cd`; a nearest-centre rule ignores `Σ`
entirely. That is the whole difference between the columns above.

The prediction that follows — features carrying *per-dimension* scatter must not show it — and its
test, medians of seeds 0/1/2 on `digits`:

| leaf budget | `spherical` | `fd` | `full` |
|---|---|---|---|
| 1797 (×1.0) | 0.4613 | 0.4613 | 0.4613 |
| 1200 | **0.0439** | 0.4843 | 0.4427 |
| 900 | **0.0088** | 0.3840 | 0.4403 |
| 500 | **0.0104** | 0.4562 | 0.5083 |
| 300 | 0.4650 | 0.4928 | 0.3943 |

The ×1.0 row is the control: with no scatter to add, all three agree to the digit. `gmm-full` on the
spherical feature collapses the same way (0.0096 at 1200 leaves, 0.0115 at 500) and never does on
`feature="full"`, since `cov_dense`'s default is the same isotropic diagonal.

So `method="gmm"` with `feature="spherical"` is a mismatch as soon as the tree compresses: the head
asks for a per-dimension covariance and the feature has none. That combination now **warns**, naming
the measured cost and the fix. The other heads read the same isotropic `variance(d)` and were not
measured, so the warning does not claim them.
- **The routing distance only exists under compression, mechanically.** At ×1.0 the spread across
  `euclidean` / `manhattan` / `ward` / `average` is exactly **0.0000** on all three datasets: the four
  distances build the identical singleton leaf set, so there is nothing left to differ. The spread
  grows with compression (`digits` gmm: 0.0000 → 0.2745 at ×2.2 → 0.6300 at ×8.2). Task 27 measured
  the routing lever at one budget and found it small; this says the budget it was measured at is the
  one regime where it provably cannot matter. Counting wins per cell, `ward` routing takes 13/23 on
  `digits` and 13/19 on MNIST — and only 2/21 on `covtype`, where `manhattan` (8) and `average` (7)
  lead. The default stays `euclidean`; the recommendation is now data-dependent and measured.

The `mean_sq_radius` column is Σᵢwᵢrᵢ²/n, the summary's mean squared quantization error, and it is
the input to task #60's Zador-form fit. It is monotone in the leaf count on all three datasets and is
label-free, so it can be reported for data with no ground truth at all.

## The FD sketch was reporting a third less scatter than it holds (task #75)

`FdSketch` approximates a leaf's scatter by a Frequent-Directions sketch, and the shrink step
subtracts the lower-median squared singular value from every direction. That subtracted mass was
discarded outright, so `ssd()` — the number the radius, the absorption gate, D2/D3 and the k-means++
potential all read — under-reported the leaf. How much is not a rounding question:

| `d` | `ℓ` | fraction of the true scatter still reported |
|---:|---:|---:|
| 64 | 16 | 0.670 |
| 64 | 32 | 0.820 |
| 784 | 16 | 0.660 |
| 784 | 64 | 0.897 |

(uniform-ish synthetic rows, `n = 5000`; the ratio is stable in `n` past a few hundred points.)

The fix banks the discarded trace, so `ssd()` is now exact — asserted against the `Full` feature over
the same points in both a unit test and a property test, both of which fail on revert.

**Giving the trace back to the *shape* is a separate question, and it was measured, not assumed.**
The sketch destroyed the directions along with the magnitudes, so two completions are available:
scale the retained directions up to carry the trace, or spread it isotropically over all `d`. Median
ARI of seeds 0/1/2, `feature="fd"`:

| dataset | head | budget | no fix | isotropic | proportional |
|---|---|---:|---:|---:|---:|
| digits | gmm | 900 | 0.3840 | 0.3828 | **0.4386** |
| digits | gmm | 450 | 0.4538 | 0.1346 | **0.5456** |
| digits | gmm | 225 | 0.4179 | 0.3081 | **0.4929** |
| mnist-10k | gmm | 2000 | **0.2736** | 0.1247 | 0.2363 |
| mnist-10k | gmm | 500 | **0.2512** | 0.1718 | 0.1580 |
| mnist-10k | gmm | 250 | **0.1841** | 0.1286 | 0.1393 |

Three things the table settles:

- **Isotropic is refuted.** It loses in all six cells, by up to 0.32 ARI. The mechanism is visible in
  the shape of the two datasets: on MNIST it spreads a third of the leaf's mass across 784
  directions when the sketch only ever saw 32, filling 752 directions that hold no data at all.
- **The shipped completion is proportional**, and its sign is dimension-dependent: +0.05…+0.09 on
  64-dimensional `digits`, −0.04…−0.09 on 784-dimensional MNIST. At `d ≫ ℓ` it concentrates the
  recovered mass in too few directions, which is the opposite error and a smaller one. There is no
  completion that is right for both, because the sketch no longer holds the information that would
  decide between them; the trace is the only part it can track exactly, and now does.
- **Only the `gmm` head can see any of this.** `ward` is byte-identical across all three variants
  (D4 is a pure centroid measure), and `kmeans` moves only within its own seed spread. The leaf
  counts are identical in every cell, so the tree itself — absorption and routing — is unchanged by
  the exact trace at these budgets.

## Insertion-order sensitivity — the property the whole BIRCH family inherits

`bench/insertion_order.py`, `bench/results_order.csv` (54 cells). A CF-tree routes each point against
the tree as it stands at that moment, so reordering the input changes the tree. Every BIRCH-class
library inherits this and none of them publish the size of it. Two arms of `P = 8` runs per cell:
`vary="order"` (8 permutations, estimator seed pinned at 0) and `vary="seed"` (identity order, seeds
0–7).

**The harness carries its own control.** The `ward` head is deterministic given a leaf set, so its
seed arm must read exactly zero. It does — spread `0.0000`, pairwise ARI `1.0000`, in all **9** of
its seed cells. That makes `ward` + `vary="order"` a pure measurement of insertion order with no
restart term in it.

| dataset | `max_leaves` | leaves | head | order spread | order pairwise | seed spread | seed pairwise |
|---|---|---|---|---|---|---|---|
| digits | 4000 | 1797 (×1.0) | ward | 0.0159 | 0.9261 | **0.0000** | **1.0000** |
| digits | 360 | 327–358 | ward | **0.2880** | 0.5454 | **0.0000** | **1.0000** |
| digits | 90 | 83–90 | ward | 0.1986 | 0.3425 | **0.0000** | **1.0000** |
| mnist-10k | 10000 | 10000 (×1.0) | ward | 0.0062 | 0.8417 | **0.0000** | **1.0000** |
| mnist-10k | 1000 | 909–1000 | ward | 0.1129 | 0.3536 | **0.0000** | **1.0000** |
| mnist-10k | 200 | 180–195 | ward | 0.1635 | 0.3087 | **0.0000** | **1.0000** |
| covtype-20k | 20000 | 20000 (×1.0) | ward | 0.0005 | 0.9960 | **0.0000** | **1.0000** |
| covtype-20k | 2000 | 1805–1983 | ward | 0.0005 | 0.9984 | **0.0000** | **1.0000** |
| covtype-20k | 300 | 278–299 | ward | 0.0013 | 0.9902 | **0.0000** | **1.0000** |

Three results:

- **The effect scales with compression, not with the dataset or the head.** `digits` goes 0.0159 →
  0.2880 → 0.1986 as the budget falls; MNIST goes 0.0062 → 0.1129 → 0.1635. `covtype` stays under
  0.0013 everywhere only because its `ward` ARI is pinned at ~0.1416 whatever the leaves are.
- **At real compression the input order is a bigger lever than the seed.** MNIST at
  `max_leaves=200`, k-means head: order pairwise ARI **0.2949** against seed pairwise **0.7026** —
  reordering the rows disagrees with itself 2.4× more than reseeding the head does. At
  `max_leaves=1000` it is 0.3574 against 0.4751. Every published table in this file, and every
  competitor's, fixes the order and varies the seed.
- **The realised leaf count is itself order-dependent** (327–358, 909–1000, 180–195) and is constant
  under reseeding — so the two arms are not even comparing summaries of the same size.

This closes the loose end from task 27: `ward` on `digits` varied 0.6224–0.6525 across the four
routing distances at an identical singleton leaf set, which had no explanation while the leaf set was
believed to be the only input. Here the same head varies 0.6428–0.6587 across permutations of that
same identical leaf set. Both are the tie-break order in which equal-distance leaves are visited, and
the budget sweep above shows why it was visible at all: `digits` at ×1.0 is exactly where nothing
else can differ.

The practical consequence is a caveat, not a fix — a single pass over an ordered stream is what the
algorithm is for. Shuffle before fitting when the input has structure in its row order, and read any
single-permutation ARI at high compression as a draw from a distribution roughly 0.15 wide.

## Conclusions

- **Use betula** when data is large or streaming, memory is bounded, or you want one numerically
  stable engine spanning k-means / GMM (diag, full & Toeplitz) / Ward / spectral / Leiden /
  HDBSCAN-style / Mapper with sklearn-style `predict` and inspection. Quality is at parity with
  scikit-learn on the centroid heads and ahead on the structured ones (anisotropic full GMM, 64-D
  `digits` GMM, covtype GMM); speed and memory are dramatically better at scale; `gmm-toeplitz` is a
  capability no mainstream library ships.
- **Use raw scikit-learn** when `N` is small enough to fit comfortably and you want the canonical
  point-level algorithm with no compression — at small `N` the two-phase overhead removes betula's
  speed edge, and raw HDBSCAN is stronger on overlapping density.
- **Use FAISS** if k-means throughput is the only criterion and an approximate partition is
  acceptable — at its defaults it fits 256·k sampled rows from one random init, which is 2.7×–8.8×
  faster than betula and lands at ARI 0.62. Configured to recover the same partition it is 6.3×–7.7×
  *slower* than betula and still misses at 1 M, so it is not the choice when the partition matters.
  Use **`fast_hdbscan`** for density clustering in low dimension when the data fits in memory and the
  densities overlap: it beats betula outright on that axis, where betula needs `min_samples` above the
  mass of a single leaf — the summary erases the small-radius density estimate HDBSCAN\* runs on.
- **Use `sklearn-birch`** if `covtype`-like all-methods quality at ~20 k rows is the only criterion and
  no summary is needed: it beats every betula head there, and the compression-ratio defence does not
  hold. Its MNIST lead comes with no compression at all (20 000 subclusters for 20 000 points), so it
  buys nothing on the axis a CF-tree exists for.
- **For sparse high-dimensional text**, reduce dimensionality first — raw TF-IDF concentrates and
  defeats every fast clusterer. `fit_predict_sparse(..., projection="svd", method="spherical-kmeans")`
  does the reduction and the clustering in one call and tops the 20-newsgroups table (0.164); tune
  `max_leaves` for the time you are willing to spend.
- **Tune `max_leaves`, and shuffle first.** The budget is the knob that matters and it is not
  monotone: on `digits` ×2 compression beats no compression on both ward and k-means, on `covtype`
  ×202 costs nothing, on MNIST the cliff is between ×10 and ×22. And at real compression the *row
  order* moves ARI more than the seed does (MNIST at 200 leaves: pairwise 0.295 across permutations
  against 0.703 across seeds), so a single-permutation number at high compression is a draw from a
  distribution roughly 0.15 wide — including every such number in the tables above.
- The numbers above are what the committed `bench/comprehensive.py` + `bench/median_of_seeds.py`
  (plus `bench/spectral_nonconvex.py`, `bench/toeplitz_ar_mixture.py`, `bench/nmf_cf_weighted.py`)
  produce; re-run them to regenerate every table and plot.
