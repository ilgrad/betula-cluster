# Features — full reference

A capability-by-capability reference. For runnable code see [`USAGE.md`](USAGE.md) and the
[example notebooks](https://github.com/ilgrad/betula-cluster/blob/main/examples/README.md); for the math behind these, see [`MATH.md`](MATH.md).

- CF-tree (BIRCH/BETULA Phase 1) with auto-rebuild and covariance models — spherical, diagonal,
  full (PSD-by-construction via Cholesky), and a **Frequent-Directions sketch** for very
  high-dimensional data ($O(\ell d)$ memory per leaf instead of $O(d^2)$; trades speed for memory, for
  `d` so large the full covariance does not fit).
- hand-written AVX2/FMA distance kernels chosen by run-time feature detection, with the scalar
  fold as the fallback on every other target (the reductions do not autovectorize — `Iterator::sum`
  is an ordered fold, so LLVM may not reassociate; measured 1.38–1.59x, labels unchanged, ADR 003);
  rayon-parallel point
  labeling and rebuild-threshold estimation (deterministic — bit-identical labels to the serial
  path; `parallel` feature, on by default, `--no-default-features` for a serial build).
- Global clustering heads: weighted **k-means** (k-means++ + exact Lloyd), weighted **k-medoids**
  (`method="kmedoids"`, eager FasterPAM — Schubert & Rousseeuw 2021) whose centre is one of the
  summary's own micro-clusters rather than an average, exact on the summary because
  `Σ_{x ∈ leaf i} ‖x − μ_j‖² = S_i + n_i‖μ_i − μ_j‖²` makes the leaf-level total the point-level
  one (note the *square*: classical PAM's absolute distance has no closed form in a cluster
  feature), weighted **fuzzy c-means**
  (`method="fuzzy-cmeans"`, Bezdek 1981) — the only soft head that fits no density, publishing a
  partition of unity `u_ij ∝ d_ij^(−1/(m−1))` rather than a posterior, with Xie–Beni as its
  automatic `k`; **diagonal &
  full-covariance GMM-EM** (expected-log E-step + NIW/MAP regularization + a per-dimension covariance
  floor that keeps components well-conditioned in high dimensions — no starved-component collapse,
  full covariance captures rotated/correlated clusters, **BIC auto-selects the component count** when
  `n_clusters=0`), an **AR / Toeplitz-structured GMM** (`method="gmm-toeplitz"`) for **ordered,
  wide-sense-stationary signals** — fixed-length time-series windows, trajectories, sensor / audio
  waveforms — where each component covariance is an AR(w) process (Levinson-Durbin → a banded
  positive-definite precision `Γ = AᵀA/σ²`, `O(w)` parameters, order `w` by BIC), well-posed at
  `N_k ≪ d` where full covariance is singular and a diagonal model is blind to neighbour correlation
  (ordered coordinates only — not generic embeddings; based on the Gohberg-Semencul estimator of
  arXiv:2311.14995, see [`docs/adr/001-gmm-toeplitz.md`](adr/001-gmm-toeplitz.md)) plus a general
  (non-AR) **`gmm-toeplitz-full`** head — a dense positive-definite Toeplitz covariance from the biased
  autocovariance — for signals whose autocovariance a low-order AR cannot represent (e.g. a long-lag
  echo: it recovers such a mixture where the banded AR head sits at chance), and a **`gmm-toeplitz-gs`**
  head — the full-order **Gohberg-Semencul MLE** precision (Yule-Walker warm start + exact-likelihood
  coordinate ascent, PD by `|k| < 1`), the likelihood-optimal general precision with a cheaper `O(m·d·p)`
  E-step than the dense route,
  a **mixture of probabilistic PCA** (`method="mppca"`, Tipping & Bishop 1999) whose component
  covariance is `W Wᵀ + σ²I` of rank `rank` — orientation like the full-covariance head at
  `O(d·rank)` per component instead of `O(d²)`, with the Woodbury inverse and the `σ^(2(d−q))|M|`
  determinant keeping every step off the `d×d` matrix; pair it with `feature="fd"`, whose leaf
  scatter is already low-rank,
  a **mixture of factor analysers** (`method="mfa"`, Ghahramani & Hinton 1996) — the same subspace
  model with the isotropic residual relaxed to a per-dimension one, `W Wᵀ + diag(ψ)`, for feature
  tables whose columns are in different units; `rank=0` is a diagonal Gaussian mixture bit for bit,
  and that floor is what the head is for, since on every already-standardised table measured `mppca`
  scores higher,
  **Ward agglomerative HAC** (exact, via nearest-neighbour chain; dendrogram-cut auto-k) and the
  four non-Ward linkages **`average`** (UPGMA), **`weighted`** (WPGMA), **`centroid`** (UPGMC) and
  **`median`** (WPGMC) on an Anderberg driver — with a **height-bounded exact** variant
  (`clustering::dendrogram_below`, Rust API) that computes every merge below a chosen $h_{\max}$ over
  a candidate radius graph and reports how far it is certified, so a cut it cannot serve exactly is
  *refused* rather than approximated; average and weighted get radius $r^2 = h$, Ward
  $r^2 = h/w_{\min}$, and centroid/median get no radius at all because they invert,
  **spectral clustering** (self-tuning k-NN affinity + normalized Laplacian embedding — exact
  Jacobi to 256 microclusters, then **Chebyshev-filtered subspace iteration** on the sparse graph,
  so every leaf stays a node instead of being reduced to 256 k-means landmarks: ARI is a tie or a
  win wherever the two paths differ and the fit is 2–12× faster, `digits`-PCA20 0.660 → 0.779 at 500
  leaves for 0.36 s → 0.03 s — separates non-convex / manifold clusters the centroid heads cannot;
  pair it with a small `threshold` so the microclusters resolve the manifold),
  **Leiden community detection** (graph clustering, Traag et al. 2019) over the microcluster affinity
  graph — local moving → refinement (each community connected by construction) → seeded aggregation;
  **discovers the community count**, no `k` needed; a `resolution` γ knob with **modularity**
  (`"leiden"`) or resolution-limit-free **CPM** (`"leiden-cpm"`) objectives; pure Rust — pair it with
  a moderate `threshold`, a very fine graph over-splits per modularity's resolution limit;
  `covariance_weight > 0` makes the affinity **covariance-aware** via a log-Euclidean shape term
  (`feature="full"`), so communities agree in both centroid and covariance; `tangent_weight > 0` adds
  a **Grassmann** tangent-subspace term (GeoBETULA) for manifold-aware communities that separate
  crossing / adjacent structures),
  **directional clustering on the unit hypersphere** — hard **spherical k-means**
  (`"spherical-kmeans"`) and a soft **mixture of von Mises–Fisher** distributions (`"vmf"`, EM with a
  true posterior and BIC auto-`k`) for L2-normalized embeddings (CLIP / face / sentence / speaker),
  where cosine — not Euclidean — geometry matters; each leaf keeps its weighted mean so the resultant
  `R_c = Σ n_i μ_i` is exactly mergeable (BETULA on the sphere) and the concentration `κ`
  (Banerjee 2005) is estimated without a Bessel library, with input auto-L2-normalized;
  **axial** clustering for data whose sign carries no information — eigenvectors, SVD/PCA axes, line
  orientations — through a **mixture of Watson distributions** (`"watson"`, Watson 1965;
  `p(x) ∝ exp(κ (μᵀx)²)`, so `x` and `−x` are the same point), whose sufficient statistic is the
  second moment `Σ_i + μ_i μ_iᵀ` the `full` leaf already carries exactly, with `κ < 0` fitting
  **girdle** (equatorial) components as well as bipolar ones and BIC auto-`k`, and
  **HDBSCAN-style density clustering over the CF microclusters** (mass-aware mutual-reachability +
  mass-weighted stability → non-convex clusters and noise, automatic count; an *approximation* of
  raw-point HDBSCAN over the $M \ll N$ microclusters, not identical to it, with `graph_degree > 0`
  swapping the complete mutual-reachability graph for a bounded-degree approximate k-NN graph —
  Okkels et al.'s two-pass construction over a flat capped-beam index — so a large `max_leaves`
  becomes affordable for the one head that most wants it), and
  **scale-space (Morse-persistence) density-mode clustering** (`method="scale-space"` — mean-shift over
  the microcluster KDE, with the bandwidth *and* the cluster count chosen by **mode persistence** across
  scale, so no `k` or bandwidth is required; non-convex, arbitrary count. Each leaf enters the KDE as
  the Gaussian it summarises, $N(\mu_j, \Sigma_j + h^2 I)$, not as a point. The sweep is two-pass —
  truncated at the first single-mode scale, then narrowed onto the merge cascade — because a single
  grid spends most of its points on the trivial tail and answered `k = 1` in 42 of 52 measured
  `(dimension × leaf budget)` cells on `digits`; that is now 3 of 52. Still the weakest head on
  `covtype`, where every cell scores a negative ARI — read the `scale-space` section of
  [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md) before
  choosing this head).
- **Soft assignment & confidence**: `predict_proba` (the point's own posterior under the fitted
  mixture for the GMM, **vMF** and Toeplitz heads, so `predict_proba(X).argmax(1) == predict(X)`; a
  documented centroid-distance softmax *heuristic* for k-means / Ward / spectral / Leiden / HDBSCAN),
  `assignment_confidence`,
  `microcluster_proba_` (per-microcluster GMM responsibilities, GMM heads only), `export_coreset` (the leaves as weighted points, or with `size=` a
  sensitivity-sampled `(k, ε)`-coreset carrying the `4√ρ + 4ρ` summarization bound it satisfies),
  `diagnostics`, `representatives`, `cluster_profile`.
- **`DenStream`** — a separate streaming density clusterer (Cao et al., SDM 2006) over *fading*
  micro-clusters, for evolving streams where old data should decay out: `partial_fit` chunks, then
  `predict` (`-1` = noise). Reuses the same numerically stable CFs (decay is exact and leaves the
  centroid/radius untouched, only the weight).
- **`DbStream`** — a streaming **DBSTREAM** clusterer (Hahsler & Bolaños, 2016) that connects fading
  micro-clusters by **shared density** (the mass of points within radius `r` of *both*), not mere
  proximity: it recovers arbitrarily-shaped clusters as chains of overlapping micro-clusters and —
  unlike a distance-only rule — keeps two close-but-disconnected dense regions apart (an empty gap
  carries zero shared density). Same fading-CF core as `DenStream`; `partial_fit` / `predict`.
- **`drift_`** on both streaming heads — an **ADWIN** change detector (Bifet & Gavaldà, SDM 2007) over
  the routing distance in micro-cluster radii, at a stated false-positive ceiling δ = 0.002 rather
  than a tuned threshold. `decay` sets how fast the model forgets, on a fixed schedule; this says
  whether it had to. It reports and does not act: an alarm changes no label.
- **`WindowStream`** — windowed queries over a timestamped stream: "cluster what arrived between
  `t₀` and `t₁`". Summaries are kept **per frame** and a window is their *sum*, never a difference of
  two cumulative snapshots the way CluStream does it — an inverse merge loses `log₁₀(S_AB/S_B)`
  digits of the scatter, and under drift that ratio runs away while the point counts stay small (a
  measured 6155× error at a mass ratio of 2.0). The price is that a window resolves only to the frame
  boundary, which is an error bounded by `frame_width` rather than by nothing. Old frames merge
  pairwise as `capacity` fills, so resolution coarsens with age and never with recency.
- **Streaming quantile sketches** (`KllSketch`, `DdSketch`) — compact, mergeable summaries that
  answer quantile / rank queries over a stream in bounded memory: **KLL** with a rank-error guarantee
  (uniform across the distribution) and **DDSketch** with a relative-error guarantee (ideal for
  skewed / positive / long-tailed data such as latencies).
- **Sparse input** — `fit` / `fit_predict` / `partial_fit` / `predict` accept a `scipy.sparse`
  matrix directly; rows are expanded one at a time, so the dense `N × d` matrix is **never
  materialized** (cluster a million-row sparse matrix that would never fit dense). This dense-tree
  path keeps the cancellation-free guarantee; compute scales with the feature count (the CF centroid
  is dense, as in every CF-tree method — sklearn-Birch included).
- **$O(\mathrm{nnz})$ sparse-native** (`fit_predict_sparse`) — for very high-dimensional sparse data, a
  one-shot path that touches only the non-zeros: rows summarize into spherical micro-clusters keeping
  $(n, \Sigma X, \|\Sigma X\|^2, S)$ so updates and centroid distances are $O(\mathrm{nnz})$, then a parametric head
  (`kmeans` default) clusters them. It uses the *expanded* squared-distance form for speed and so does
  **not** carry the dense path's cancellation-free guarantee — accurate for sparse rows far from the
  dense centroid; use the dense `Betula` path when you need cancellation-free scatter.
- **CF-weighted NMF reduction** (`projection="weighted-nmf"`, `projection_dim`) — for **nonnegative**
  data (TF-IDF / bag-of-words / event counts / spectrogram magnitudes / histograms), a nonnegative
  low-rank projection applied over the $M \ll N$ leaf **centroids**, not the raw $N \times d$ matrix: by
  König-Huygens the weighted-centroid NMF equals the full-data NMF up to the within-leaf scatter
  constant, so it runs NMF at BETULA scale and bounded memory (something point-level NMF cannot), then
  any head clusters the nonnegative codes. Dependency-free weighted **HALS** (no BLAS — the matrices are
  small because $M \ll N$); `projection="weighted-nmf-kl"` switches to the **KL-divergence** variant
  (Lee-Seung multiplicative updates) — the Poisson maximum-likelihood objective for **count** data. The
  advantage is largest where counts are **sparse** and is now small: **+0.04 ARI** over Frobenius at
  mean rate 0.2 (0.892 against 0.850) and nil beyond it, converging to Frobenius as counts grow and
  Poisson → Gaussian. An earlier edition reported up to +0.5 ARI; that margin closed when 0.6.0 fixed
  the labelling downstream of the factorization, not because this objective changed.
  Both solvers start from a deterministic **NNDSVDar** initialization (a randomized range finder, so no
  LAPACK; rank-deficient triplets cut at the numerical-rank threshold rather than amplified) and return a
  **canonical** factorization — component rows unit-L2, ordered by descending
  energy. That last part is load-bearing, not cosmetic: NMF is invariant to `(W D, D⁻¹H)`, so an
  unpinned split lets one component's arbitrary scale dominate the Euclidean geometry the head then
  clusters (measured over 8 seeds at N = 8k/40k/160k: median ARI 0.81/0.99/0.97 → 1.00, seed spread
  ±0.37 → ±0.00 — the gain is determinism, not accuracy in the mean). `components_`
  and `reconstruction_err_` expose the parts and the fit; `projection_max_iter` is the solver's own
  budget, independent of the head's `max_iter`. Dense **and** sparse CSR input; signed input is
  rejected, not shifted. See [`MATH.md`](MATH.md).
- **Mass-balanced leaf budget** (`balance`) — optional per-leaf cap on how much of the total mass one
  micro-cluster may hold, as a multiple of the `n / max_leaves` ideal. The textbook budget is purely
  geometric: one global absorption radius, raised until the leaf count fits. That radius is a single
  number and real data has more than one density, so once it passes a dense region's diameter that
  region collapses into one leaf while sparse regions keep splitting — measured at 80 % of the mass
  in a single leaf, at every budget from 250 to 4000, with the budget itself 90–96 % filled. Setting
  `balance` (e.g. `4.0`) refuses absorption into a full leaf and skips the same pairs at compaction;
  `max_leaves` stays a **hard** bound, so a rebuild that cannot reach its target under the cap merges
  over it rather than leave the tree over budget. Off by default, because it is a lever and not a
  free win: on a size-imbalanced fixture it is worth **+0.58 ARI**, and on well-spread data it is
  roughly neutral. The diagnostic that tells you which case you are in is the heaviest leaf's share
  of the mass — `max(microcluster_weights_) / sum(...)`. See [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md).
- **Robust insertion** (`huber_k`) — optional Huber/winsorized point updates: each incoming point is
  clamped to within `huber_k` per-dimension standard deviations of its target microcluster *before*
  it is folded in, so a single extreme value cannot stretch a centroid or inflate a radius. Off by
  default; most valuable for streaming, where you cannot go back and re-fit on cleaned data. See the
  formula in [`MATH.md`](MATH.md).
- **Constrained clustering** (`must_link` / `cannot_link`) — semi-supervised **COP-KMeans** (Wagstaff
  et al., 2001): pass pairwise row-index constraints to `fit` / `fit_predict` and points that *must*
  share a cluster are kept together and points that *cannot* are kept apart. Constraints are honoured
  at the microcluster granularity (a cannot-link between two points the tree compressed into one leaf
  is reported as infeasible — lower `threshold` to separate them); contradictory or over-constrained
  inputs raise rather than silently violate. `method="kmeans"` only, dense input.
- **Mixed numeric + categorical** (`KPrototypes`) — **k-prototypes** (Huang, 1997) for data that is
  part numeric, part categorical. Each cluster is a *mixed CF*: the stable numeric $(n, \mu, S)$ plus a
  category-count histogram per categorical attribute (its mode is the categorical centroid). Distance
  is $\|\Delta_\text{numeric}\|^2 + \gamma \cdot (\text{categorical mismatch})$, with $\gamma$ auto-set to Huang's heuristic. Rows are
  leader-summarized into bounded mixed micro-clusters first, so it scales like the rest of the library.
- **Bregman geometry** (`BregmanBetula`) — the CF-tree and three heads over an arbitrary Bregman
  divergence instead of squared Euclidean: `divergence="kl"` for distributions, `"itakura-saito"` for
  spectra (scale-invariant), `"logistic"` for probabilities, `"euclidean"` for the case that reduces
  to the shipped estimator. The feature is $(n, \mu, S_\varphi)$ with $S_\varphi$ the Bregman
  *information*, and the merge is unchanged — the arithmetic mean is the right-sided Bregman centroid
  for **every** $\varphi$ (Banerjee et al. 2005), so the tree machinery carries over untouched.
  Heads: `method="kmeans"` (Bregman k-means), `"ward"` (Bregman-Ward HAC on an Anderberg driver,
  because reducibility fails from $d \ge 2$ — [`docs/adr/002`](adr/002-bregman-ward-anderberg.md)),
  `"mixture"` (soft mixture by variational EM, with `beta` the inverse dispersion, measured in nats).
  A **separate estimator** rather than a keyword on `Betula`, so the meaningless combinations cannot
  be written at all — [`docs/adr/004`](adr/004-bregman-public-api.md). The domain (`x > 0` for KL and
  Itakura–Saito, `x ∈ (0,1)` for logistic) is validated at the Python boundary.
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
  `_radii_` / `_sizes_`) and, on top of it, `summary()`, `validity()` (Calinski–Harabasz,
  Davies–Bouldin and the medoid silhouette in $O(\ell k d)$ off the leaf summary, no second pass
  over the points and no $O(N^2)$ term), `outlier_scores(X)` (distance to the
  assigned centroid ÷ cluster radius), `find_outliers`, `find_near_duplicates` (unscored groups),
  `near_duplicate_pairs(X, threshold)` (scored cosine pairs, exact within each leaf-block — the
  scalable counterpart to an $O(N^2)$ all-pairs scan), `sample_representatives`, and
  `assign_microclusters` — for embedding dataset cleaning, deduplication, and outlier discovery,
  reusing the CF-tree already built (no extra passes). `outlier_scores` / `find_outliers` take a
  `metric` — `"radius"` (scalar, `O(d)` per row) or `"mahalanobis"` (whitened by the cluster's pooled
  covariance).
    Measured on the axes Sanchez Vinces, Schubert, Zimek and Cordeiro set out in
    *Clustering-based outlier detection* (DAMI 2025) — detection quality, resilience to parameter
    variation, and auto-filtering by an internal index. `outlier_scores` is a **local** score, and it
    wins where that matters: on clusters of unequal density it reaches ROC-AUC 1.000 / average
    precision 0.998 against IsolationForest's 0.887 / **0.175**, at the smallest parameter spread of
    the three detectors (0.001 over a 12-cell sweep). Its default `metric="radius"` divides by a
    single scalar RMS radius, so a sheared cluster's short axis is judged by the length of its long
    one (ROC-AUC 0.596, and identical at `feature="diagonal"` and `feature="full"` because that
    radius is the covariance's *trace*); `metric="mahalanobis"` whitens by the cluster's pooled
    covariance instead and lifts the same case to 0.748, at `O(k·d³)` once plus `O(d²)` per row. The
    two are calibrated — on an isotropic cluster they return the same number — so the refinement
    moves a score only where the cluster has a shape. And the parameter cell can be chosen
    without labels: taking the largest `validity()["calinski_harabasz"]` over the sweep lands within
    0.033 ROC-AUC of the best cell in it. Tables in
    [`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md).
- **Mapper topological skeleton** (`mapper()` → `MapperGraph`) — TDA Mapper specialised to the
  microclusters: a lens (`density` / `radius` / `l2norm` / `coordinate` / `eccentricity`) is covered
  by overlapping bins, microclusters in each bin are single-linked at a data-adaptive scale, and the
  nerve graph exposes **branch points** and **bridges** (thin links between otherwise separate
  regions — topic leakage / merges in embeddings). Each edge also carries a **CF-aware Bhattacharyya
  overlap** (`edge_overlap ∈ (0, 1]`) from the two nodes' pooled diagonal-Gaussian summaries, so a
  bridge across a sparse neck scores lower than an edge inside one dense blob — distributional, not a
  bare shared-microcluster count. Runs over the $M \ll N$ microclusters, with an optional
  `to_networkx()` (edges carry `weight` / `overlap` / `bridge`) for plotting; `mapper_stability()`
  sweeps the resolution to find the topologically stable scale. An exploration tool (structure, RAG
  curation, dedup), not a partition — complementary to the HDBSCAN density head.
- **Exact `k`-center / `k`-median in the density-connectivity ultrametric**
  (`method="dc-center"` / `"dc-median"`, Beer, Draganov, Hohma, Jahn, Frey & Assent, KDD 2023) — the
  same mutual-reachability spanning tree `method="hdbscan"` takes its hierarchy from, cut for a `k`
  you name instead of a `min_cluster_size`. `dc(a, b)` is the heaviest edge on the path between them,
  which makes it an ultrametric and both objectives exactly solvable: `k`-center by deleting the
  `k − 1` heaviest edges, `k`-median by an `O(m·k)` knapsack over the dendrogram (a leaf's cost
  depends only on *which subtree* holds its nearest centre, never on which centre). Both are checked
  against brute force over every `C(m, k)` centre set below `m = 12`. Neither emits `-1` — they
  partition; `hdbscan` is the head for noise. Measured at `N = 6 000`, `max_leaves=600`, median of
  seeds 0/1/2: `dc-median` reaches **0.889** on moons at noise 0.10 against `spectral`'s 0.691 and
  `hdbscan`'s 0.015, at 0.014 s against 0.383 s, and **0.725** on `digits`-PCA20 against `ward`'s
  0.721 and `hdbscan`'s 0.458. `dc-center` is **mass-blind by construction** — a maximum cannot see a
  weight, so on a summary, where an outlier is a low-mass leaf, it spends its budget isolating strays
  (`[6297, 2, 1, 0]` rows at `k = 6` on the noise fixture). Tables in
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).
- **OPTICS reachability plot** (`reachability()` → `ReachabilityPlot`) — the density *diagnostic*
  over the microclusters: one sweep position per leaf, with `core_distances` and per-leaf `weights`
  alongside, and `labels_at(ε)` for the DBSCAN\* cut. It is not an approximation of the HDBSCAN head
  — OPTICS with no ε cutoff is Prim's algorithm on the mutual-reachability graph, so the sweep walks
  the *same* spanning tree that head takes its hierarchy from, and every peak is one of its merge
  heights. (That is also why the reachability is the mutual `max(core(p), core(q), d(p,q))` rather
  than Ankerst's asymmetric form, which would picture a different tree.) Cost is set by the leaf
  count, not `N`: 0.0028 s at both 20 000 and 320 000 rows, against a 0.007 s / 0.054 s fit. Read as
  a partition against `sklearn.cluster.OPTICS` on the raw points (N = 6 000, best ε each side,
  median of seeds 0/1/2) it wins three of four fixtures — blobs 0.452 vs 0.448, moons and circles
  0.997 vs 0.978 — at ~200× the speed, and **loses the noise fixture 0.687 to 0.753**. The
  compression is not the reason: at one leaf per point that fixture only reaches 0.707, so the
  residual is the convention (DBSCAN\* has no border points, and mutual reachability is a stricter
  link than the asymmetric one). Tables in
  [`docs/USAGE.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/USAGE.md).
- **Memory-aware hyperparameter tuning** (`tune` → `TuneResult`) — searches betula's CF-representation
  knobs (`max_leaves`, covariance model, `normalize`) for the best clustering into `n_clusters`,
  scored by an internal metric (Calinski-Harabasz / Davies-Bouldin) or ARI against ground-truth
  labels, with an optional **quality / memory (`n_leaves`) / speed (fit-time)** Pareto mode. NumPy-only
  by default (random search); an optional Optuna backend (TPE / NSGA-II) via
  `pip install 'betula-cluster[tune]'`. Because betula fits are cheap, a few hundred trials run in
  seconds — the search is over the compression, so cost is bounded by the microcluster count, not `N`.
- **Tree diagnostics** (`tree_report()`, `estimate_threshold`) — answers "why is my tree collapsing?"
  from the leaf summary: how much of the leaf budget was spent, how much of the *mass* one leaf ended
  up holding, and — the number that decides whether that cost anything — how **wide** that leaf is
  against a typical one. Mass concentration alone cannot tell a real loss from a faithful summary:
  the `structured` and `flat` fixtures of `bench/size_imbalance.py` both put 80% of the mass in one
  leaf, and only the first has anything inside it to lose. The width separates them 3× (0.53/0.75
  against 0.17/0.27), so the report distinguishes "the structure inside it is unrecoverable" from
  "the dense part is genuinely point-like". Pass `X` for an **advisory** A-BIRCH gap-statistic
  threshold estimate (Lorbeer et al. 2018) beside the threshold in use, which names each of the
  paper's assumptions — comparable cluster sizes, well-separated clusters — that the data breaks.
- **Consensus & stability** (`consensus` → `ConsensusResult`) — clusters `X` under several random
  insertion-order permutations and votes, converting the CF-tree's insertion-order sensitivity into a
  measurable signal: a consensus labelling plus a **per-point stability score** in `[0, 1]` (low on
  unstable boundaries, high where every order agrees). NumPy-only; partitional heads at a fixed
  `n_clusters`.

## Architecture (crate layout)

| module | role |
|--------|------|
| `types` | `Real` numeric trait (`f32` / `f64`) |
| `linalg` | Cholesky / triangular solve / logdet / Mahalanobis / Jacobi eigensolver (no LAPACK) |
| `stats` | χ² quantile (inverse regularized incomplete gamma) for Mahalanobis gates |
| `feature` | clustering features: `Spherical` / `Diagonal` / `Full` / `FdSketch` (high-d) |
| `kernels` | distance kernels: scalar fold + hand-written AVX2/FMA path |
| `distance` | D0–D4, radius, Mahalanobis (stable forms) |
| `tree` | arena CF-tree + budget-targeting auto-rebuild |
| `clustering` | `kmeans` / `kmeans_auto` / `cop_kmeans`, `xmeans` (recursive splitting), `kmedoids` / `dyn_msc` (FasterPAM), `fuzzy_cmeans` / `fuzzy_cmeans_auto`, `gmm_diagonal`, `gmm_full`, `gmm_toeplitz{,_full,_gs}`, `ward_hac`, `agglomerative` (UPGMA/WPGMA/UPGMC/WPGMC), `spectral`, `leiden`, `spherical_kmeans`, `movmf`, `scale_space`, `hdbscan`, `kprototypes`, `nmf` (the `projection` reducer) |
| `mixture` | fitted-mixture kernels (diagonal / full-Cholesky / stationary / vMF) that score a raw point — what `predict` / `predict_proba` label by |
| `stream` | `DenStream` + `DbStream` fading-microcluster density heads, with the `adwin` drift detector on their routing distance |
| `window` | frame-summed windowed summaries + `WindowStream`; the conditioned inverse merge |
| `adwin` | ADWIN2 adaptive windowing — the change detector behind `drift_` |
| `coreset` | sensitivity-sampled `(k, ε)`-coreset with its summarization bound |
| `sparse` | `O(nnz)` sparse-native summarisation (`fit_predict_sparse`) |
| `sketch` | KLL + DDSketch mergeable quantile sketches |
| `topology` | Mapper nerve + 0-D persistence |
| `model` | end-to-end `Model::fit` / `predict`; the `Method` enum and the per-head assignment rule |
| `python` | PyO3 bindings: one-shot `fit_predict` + streaming `Betula` estimator |

See [`DESIGN.md`](https://github.com/ilgrad/betula-cluster/blob/main/DESIGN.md) for the full design and the verified mathematical foundation.
