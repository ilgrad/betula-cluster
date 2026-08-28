# What a cluster feature can and cannot answer (measured)

> The formal boundary of the summary every head in this library reads. Reproduce with
> `uv run --no-sync --with scikit-learn python research/cf_boundary.py` — the script recomputes
> each claim below from a second, independently written arithmetic rather than calling the crate.

Deep Sets (Zaheer et al., NeurIPS 2017) says a permutation-invariant set function has the form
`f(X) = φ(Σ_u ψ(x_u))`, and Wagstaff et al. (ICML 2019) study what such a representation can reach.
A cluster feature **is** that sum: `ψ(x) = (1, x, ‖x‖²)` for the spherical model, `(1, x, x∘x)` for
diagonal, `(1, x, xxᵀ)` for full, `(1, x, sketch)` for FD. `ClusterFeature::merge`
(`src/feature.rs`) is the `Σ`, `push` is `ψ` followed by it, and every Phase-3 head is a `φ`.

That framing replaces "the summary loses something" with three answerable questions. All three are
settled here, and none of the answers is asymptotic.

## (a) Exact — the degree-≤ 2 objectives

The moment map `X ↦ (W, Σ w x, Σ w xxᵀ)` is far from injective, but it is **sufficient** for every
objective that is a permutation-invariant polynomial of degree ≤ 2 in the points. One expansion
covers the whole shipped list, because each is built from `Σ w‖x − c‖²` for a `c` that is itself an
affine function of the moments:

    Σ_{x ∈ cl} w‖x − c‖²  =  Σ_{leaves l ⊆ cl} ( S_l + w_l‖μ_l − c‖² )

with `S_l = Σ w‖x − μ_l‖²` the leaf SSD (`ClusterFeature::ssd`). Given the leaf features and an
assignment of *whole leaves* to clusters, the right-hand side is exact.

| objective | exact from the summary? | why |
|---|---|---|
| WCSS / `k`-means potential | yes | the identity above with `c` the cluster centroid |
| Ward merge cost | yes | `(w_a w_b / (w_a + w_b))‖μ_a − μ_b‖²`, moments only |
| Calinski–Harabasz | yes | between and within are both WCSS-shaped |
| Davies–Bouldin, RMS radius | yes | radius `√(S_cl / w_cl)`, centroid distance from means |
| Bregman `D4_φ` | yes, **if `Σ w φ(x)` is carried** | `Σ w D_φ(x, μ) = Σ w φ(x) − W φ(μ)`; the gradient term vanishes at the mean |
| GMM log-likelihood | yes | the E-step's sufficient statistics *are* the moments |
| exact silhouette | **no** | needs pairwise distances; unbounded degree |
| single-linkage dendrogram | **no** | `min` over pairs is not a polynomial |
| DBSCAN labelling | **no** | same |

Checked rather than asserted — 400 points in 3-D, 20 leaves, 4 clusters, each index computed from
the raw points and from `(w, μ, S)` per leaf:

| index | from points | from summary | relative |
|---|---:|---:|---:|
| WCSS | 1198.2888719871 | 1198.2888719871 | 0 |
| Ward (first merge) | 0.2051955950 | 0.2051955950 | 1.8e-15 |
| Calinski–Harabasz | 1.1911439030 | 1.1911439030 | 1.9e-16 |
| Davies–Bouldin | 37.4571743863 | 37.4571743863 | 7.6e-16 |

`validity::medoid_silhouette` (`src/validity.rs`) is therefore an approximation **by construction**,
not by implementation choice — which is the honest way to state the caveat it carries, and the
reason it exists at all: the exact silhouette is not reachable from any feature in the family.

## (b) Impossible — an exact counterexample, not a coarseness argument

The strong form of the negative statement is not "the cheap feature is too coarse" but "**no**
feature in the family can tell these apart". Two four-point sets on the `x`-axis:

    A = { −3, −1, 2, 2 }        B = { −3, 0, 0, 3 }

Weight 4, mean 0, scatter `[[18, 0], [0, 0]]` — equal **exactly in binary floating point**, so the
spherical, diagonal and full features and the FD sketch of `A` and `B` are the same object. Integer
coordinates are deliberate: nothing here rests on a tolerance.

What differs:

| | A | B |
|---|---|---|
| single-linkage merge heights | 0, 2, 3 | 0, 3, 3 |
| distinct pairwise distances | 0, 2, 3, 5 | 0, 3, 6 |
| DBSCAN, `eps = 2`, `minPts = 2` | 2 clusters, 0 noise | 1 cluster, 2 noise |

The pair was found by exhaustive search over integer multisets, not by hand: at `n = 4` with
coordinates in `[−3, 3]` there are **24** such twin pairs, at `n = 5` in `[−2, 2]` **22**, at
`n = 6` **60**. Two more, for a test that wants a second shape: `(−2, −1, 1, 1, 1)` against
`(−2, 0, 0, 0, 2)`, and `(−2, 0, 0, 0, 1, 1)` against `(−1, −1, −1, 1, 1, 1)`.

So a density head or a single-linkage head reading the summary is answering a question the summary
does not contain the information for, and the failure is not a limit — it is four points. Pinned in
the suite as `feature::tests::two_point_sets_can_share_a_feature_and_not_a_geometry`, which asserts
both halves: that the features agree across all models, and that the geometry does not.

## (c) The cost — an identity for the exact objectives, a measured rate for the rest

For everything in the first table the usual deformation-stability *inequality* is not needed,
because the error has a closed form. When every leaf lies inside one cluster,

    WCSS_points = WCSS_summary + Σ_l S_l

exactly — residual `2.8e-14` on 60 random points in 3-D over six leaves and two clusters, and pinned
in the suite as `feature::tests::the_summary_costs_exactly_the_total_leaf_scatter`. `Σ_l S_l` is
precisely what BIRCH's absorption `threshold` and `max_leaves` control, so "quality vs `max_leaves`"
is not a heuristic curve but a **budget**: the summary costs exactly the total leaf scatter and
nothing else. (The shape of that curve is #60's Zador fit; this is why it has a shape at all.)

For an index that is *not* degree-≤ 2, mapping every point to its own leaf centroid is a feasible
transport plan, so

    W₂(data, summary)²  ≤  (1/W) Σ_l S_l  =  the weighted mean leaf variance

and any index that is `L`-Lipschitz in `W₂` inherits `|index(summary) − index(data)| ≤ L·rms`, with
`rms = √(Σ_l S_l / W)`. The medoid silhouette is not uniformly Lipschitz — it is a ratio — so `L`
is measured rather than derived. Same index, same partition, read once off the leaves and once off
the points underneath them:

| dataset | radius | leaves | rms leaf | sil(points) | sil(summary) | gap | gap/rms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blobs2d | 2.00 | 21 | 1.0473 | 0.7870 | 0.9016 | 0.1146 | 0.109 |
| blobs2d | 1.50 | 32 | 0.8239 | 0.8481 | 0.9114 | 0.0633 | 0.077 |
| blobs2d | 1.00 | 65 | 0.5858 | 0.9207 | 0.9295 | 0.0088 | 0.015 |
| blobs2d | 0.70 | 104 | 0.4026 | 0.9273 | 0.9252 | 0.0021 | 0.005 |
| blobs2d | 0.40 | 247 | 0.2102 | 0.9026 | 0.9013 | 0.0013 | 0.006 |
| digits64 | 2.20 | 72 | 1.3975 | 0.2485 | 0.3795 | 0.1310 | 0.094 |
| digits64 | 1.90 | 158 | 1.2020 | 0.4135 | 0.4587 | 0.0452 | 0.038 |
| digits64 | 1.60 | 344 | 0.9753 | 0.3049 | 0.3634 | 0.0585 | 0.060 |
| digits64 | 1.30 | 723 | 0.6902 | 0.4340 | 0.4625 | 0.0285 | 0.041 |
| digits64 | 1.00 | 1324 | 0.3465 | 0.4242 | 0.4329 | 0.0086 | 0.025 |

`gap/rms ≤ 0.11` on every row of both datasets, and it falls as the budget grows: the gap closes
**faster** than `rms`, not merely proportionally. That is the empirical content of the bound — the
envelope holds with a small constant, and the constant is not being paid in full.

Two things this table does *not* say, and it matters:

- **`sil(points)` is not a quality curve.** The partition changes with the leaf budget (Lloyd runs
  on a different leaf set at every row), so it moves non-monotonically — `blobs2d` peaks at 104
  leaves and dips at 247. Only the *gap* is a statement about summarisation; the level is a
  statement about which partition that row happened to find.
- **`rms` is not free of the data.** It is in the units of the space, so `gap/rms` compares rows
  within a dataset, not across the two. `digits64` reaches a smaller `gap/rms` at a larger `rms`
  precisely because 64 dimensions put more of the leaf scatter in directions the index does not
  read — the same concentration effect recorded in the high-D notes.

## What this settles

1. Reading `k`-means, Ward, CH, DB or a Gaussian likelihood off the summary is **not** an
   approximation, and the leaf budget is the exact error term when it is one.
2. Reading a single-linkage dendrogram, a DBSCAN labelling or an exact silhouette off the summary
   **cannot** be made exact by enriching the feature — the full model already fails on four points.
   The heads that do it (`hdbscan`, `dbstream`, `medoid_silhouette`) are approximations by
   construction, and their docs should say so rather than quantify a tolerance.
3. Between the two, the summarisation error is bounded by the mean leaf radius, with a constant
   measured at ≤ 0.11 for the medoid silhouette on the two datasets here.

Related, not duplicated: #55 (`mmd_fidelity`) measures the left-hand side of the `W₂` bound
directly; #56 (`mixture_w2`) measures it between two fitted models; #60 fits the Zador form to the
quality-vs-`max_leaves` curve whose *existence* is item 1 above.
