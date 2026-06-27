# betula-cluster — Design

A from-scratch, numerically-stable, fast clustering library for large data and embeddings,
built on the BIRCH/BETULA Clustering-Feature (CF) tree. Rust core + Python (PyO3) bindings.
Goal: best-in-class clustering — correct, numerically stable, and fast with minimal memory.

All formulas below are verified (symbolically in Maxima and/or numerically against an
mpmath/Julia ground truth) in `../../math_improove/` and `research/`. Key empirical result:
`research/RESULTS-estep.md` (the GMM E-step choice).

## Design principles
1. **Stability by construction.** Store `(n, μ, S)` (weight, mean, sum of squared deviations),
   never `(N, ΣX, ΣX²)`. Variances are sums of non-negatives → no catastrophic cancellation
   (BIRCH fails at coordinate ~1e7 in f64; BETULA does not — `math_improove/01`).
2. **Make illegal states unrepresentable.** Covariance is PSD by construction (matrix Welford);
   the full-covariance feature uses ONE tested upper-triangular index function (the reference
   impl had a silent `(j-1)·dim/2` vs `(j-1)·j/2` bug that corrupts dim≥4 — `math_improove/09`).
3. **Best formula, tested.** Where a formula choice affects quality we pick by measured ARI, not
   by tradition (e.g. GMM E-step: expected-log beats the paper's convolution — see below).
4. **Functional core, imperative shell.** Pure CF math + clustering kernels in the core; effects
   (I/O, Python) at the edges.
5. **Speed via the right algorithm first, then SIMD/parallel.** Accelerated *exact* k-means
   (Hamerly) before micro-optimizing; SIMD distance kernels; rayon.

## Mathematical foundation (locked, verified)

### Clustering Feature `CF = (n, μ, S)`
- `n` weight (allows fractional weights for decay), `μ ∈ ℝ^d` mean, `S` sum of squared
  deviations from the mean — scalar (spherical), per-dim vector (diagonal), or full scatter
  matrix `M = Σ w (x-μ)(x-μ)ᵀ` (full).
- **Weighted add** (Welford/West): `W' = W + w; μ += (w/W')(x-μ); S += w·(1 - w/W')·δ⊙δ`
  (matrix: `M += w(1-w/W')·δδᵀ`), `δ = x - μ_old`.
- **Merge** (Chan): `W = W_A+W_B; μ = μ_A + (W_B/W)Δ; S = S_A + S_B + (W_A W_B/W)·Δ⊙Δ`
  (matrix `+ (W_A W_B/W)ΔΔᵀ`), `Δ = μ_B - μ_A`. Algebraically exact (Maxima, `math_improove/02`).
- Full covariance via on-demand Cholesky `Σ = M/n = L Lᵀ`; `logdet = 2Σ ln L_ii`;
  `mahalanobis²(x) = ‖L⁻¹(x-μ)‖²`. PSD verified; naive scatter goes indefinite at large offset
  (`math_improove/03`). No cross-type CF merge (homogeneous trees) — avoids the isotropic
  approximation and the index-bug class.

### Distances (stable forms, Maxima-verified ≡ BIRCH forms)
D0 `‖μ_A-μ_B‖`, D1 `‖μ_A-μ_B‖₁`, D2 `S_A/n_A + S_B/n_B + ‖Δμ‖²`,
D3/diameter, D4 (variance-increase = Ward) `(n_A n_B/n_AB)‖Δμ‖²` (no S term — König–Huygens),
radius `R = √(S/n)`. Mahalanobis-χ² as an absorption option (`distance::MahalanobisChi2` +
`stats::chi2_quantile`, DLMF 8.2.4) — mass-invariant, fixes the size-imbalance failure
(`math_improove/05`, sklearn #22854). All squared internally (no sqrt in hot path).

### Phase-3 global clustering on leaf CFs
- **k-means**: weighted k-means++ init; accelerated **exact** Lloyd (Hamerly bounds; Yinyang
  later). Cost includes within-CF spread: `SSE = Σ_i [S_i + n_i‖μ_i - c‖²]`.
- **GMM-EM** (spherical/diagonal/full): each leaf CF is a mini-Gaussian `N(μ_i, Σ_i)`.
  **E-step = expected-log (variant C)** — *measured best* (`research/RESULTS-estep.md`):
  `log r_ik = log π_k + log N(μ_i|μ_k, Σ_k) − ½ tr(Σ_k⁻¹ Σ_i)`, log-sum-exp normalized.
  (The paper's convolution `N(μ_i|μ_k, Σ_k+Σ_i)` is *worse* on coarse CFs — washes out
  components.) M-step folds within-CF spread:
  `Σ_k = Σ_i w_ik (Σ_i + (μ_i-μ_k)(μ_i-μ_k)ᵀ)/N_k`, `w_ik = n_i r_ik`; NIW/MAP regularization
  `Σ_k = (Ψ + ·)/(ν + N_k + d + 1)` to avoid singular covariance; Cholesky for logdet/solve.
- **Ward-HAC**: agglomerative merging of leaf CFs by minimum variance increase (D4/Ward linkage),
  via the exact nearest-neighbour-chain algorithm — O(m²) time, O(m) space. Merges are exact CF
  merges (no Lance-Williams update approximation).

### Phase-3b density/topological head (`../../plans/topology-tda.md`)
**HDBSCAN-on-CF** (done): mutual-reachability (`min_samples` core distance) + mass-weighted
stability → non-convex / variable-density clusters, noise, automatic count. Its 0D persistence *is*
single-linkage-with-persistence-pruning, so this is the density-topological head.

**`topology::mapper`** (done): TDA Mapper over the microclusters — a lens (density / radius /
`‖μ‖` / coordinate / eccentricity) covered by overlapping bins, per-bin single-linkage at the
median-NN scale `link_scale × median nn-gap`, nerve graph with branch points and **bridges** (Tarjan).
Exposed as `Betula.mapper() -> MapperGraph` (+ `to_networkx()`). Exploration of structure / RAG
leakage / dedup, `O(M²)` over the `M ≪ N` microclusters — a tool, not a partition. Complementary to
parametric Phase-3a. Roadmap: CF-aware edges (Bhattacharyya/Hellinger) and persistence diagrams.

## Architecture (crate layout)
```
src/
  types.rs       Real (f32/f64) numeric trait
  linalg.rs      dense Cholesky/solve/logdet/mahalanobis + Jacobi eigensolver (no LAPACK)
  stats.rs       χ² quantile (inverse regularized incomplete gamma) for Mahalanobis gates
  feature.rs     ClusterFeature trait + Spherical / Diagonal / Full / FdSketch (FD high-d)
  distance.rs    CFDistance trait + measures; uses simd kernels
  kernels.rs     auto-vectorized distance kernels (inline reductions)
  tree.rs        arena CF-tree (insert/split/rebuild)
  clustering/    kmeans.rs, gmm.rs, ward.rs, hdbscan.rs (Phase 3)
  stream.rs      DenStream + DbStream streaming density heads (fading micro-clusters)
  sparse.rs      O(nnz) sparse-native spherical summarisation (fit_predict_sparse)
  sketch/        KLL + DDSketch streaming quantile sketches (betula-sketch)
  topology.rs    Mapper / persistence (exploration)
  bin/betula.rs  command-line interface (feature = "cli", std-only)
  python.rs      PyO3 bindings (feature = "python")
```
Core abstractions: `Real` (numeric), `ClusterFeature<R>` (Spherical/Diagonal/Full),
`CFDistance<R, C>`, `CFTree<R, C, D, A>`, `GlobalClustering<R, C>`. Generics monomorphize →
zero dispatch cost; Python/CLI pick variants via enums.

## Numeric & performance strategy
- **SIMD**: inline auto-vectorized kernels (sq-euclidean, dot, manhattan) — the compiler vectorizes
  the tight reductions at each call site. Measured faster than a `multiversion` runtime SIMD
  dispatcher on the small-`d` CF-tree hot path (its indirect call cannot inline, and the per-call
  dispatch dominates the few arithmetic ops; no high-`d` win either, where the GMM `O(d³)` Cholesky
  dominates, not these reductions).
- **Parallel**: `rayon` for assignment/E-step over points/CFs; thread-local accumulators merged
  (CF is a commutative monoid → exact reduction).
- **Linalg**: hand-rolled tiny Cholesky in `linalg.rs` for per-CF d×d; `faer` (pure-Rust SIMD,
  no LAPACK) for any larger/batched ops.
- **Python**: PyO3 0.29 `abi3-py311` wheels via maturin; `rust-numpy` zero-copy; `Python::detach`
  around compute (release GIL); sklearn-compatible `fit/partial_fit/predict`.

## Testing
- Unit (per module) + golden (fixed seed) + property (proptest: merge assoc/commut, "tree CF =
  Σ points", upper-tri index round-trip incl. dim≥4) + numerical-stability (offset 1e7–1e8 vs
  mpmath) + end-to-end ARI vs ground truth (mirrors `research/gmm_cf_estep.py`).
- `cargo test`, `clippy -D warnings`, `fmt --check`; Python `pytest` + sklearn cross-check.

## Status

**Done & verified** — 122 Rust unit + 4 integration tests (`--features python,persistence`; more
behind `cli`) + a 121-case `pytest` suite (Python wrapper at 100 % line coverage, Rust ≥95 %
CI-enforced via `cargo llvm-cov`); `clippy -D warnings` + `fmt` clean (across `parallel`, serial,
`persistence`, `cli`, and `python` feature sets); GitHub Actions CI (Rust gate
+ Python build/pytest on 3.11–3.14) and a multi-platform wheel-release workflow (`.github/workflows/`);
Python end-to-end + scikit-learn benchmark (`README.md`, `bench/RESULTS.md`):
- `types`, `linalg` (+ Jacobi symmetric eigensolver), `feature` — Spherical/Diagonal/Full/FdSketch.
  `FdSketch` is a Frequent-Directions scatter sketch (Liberty 2013; ℓ×d, `M ≈ BᵀB`) for very high `d`
  where a full `d×d` per leaf does not fit: `O(ℓ·d)` memory, exact mean/weight, lossless on rank ≤ ℓ
  data, and it trades speed for memory (an eigendecomposition per shrink). The full-cov GMM consumes
  it in **low-rank form** (`SecondMoment::LowRank`): `tr(Σ_k⁻¹ Σ_i) = Σ_r ‖L_k⁻¹ f_r‖²` and the
  M-step accumulates `Σ_r f_r f_rᵀ`, so the GMM never materialises a `d×d` matrix per leaf and FD's
  `O(ℓ·d)` advantage carries through clustering (otherwise it would be lost). Identical math to the
  dense path. All PSD; tested incl. dim≥4 merge, FD-vs-full agreement on low-rank data, and
  FD-vs-full GMM clustering (ARI 1.0, peak RSS bounded by ℓ·d).
- `distance` + `kernels` (inline auto-vectorized distance reductions).
- `tree` (insert/split/rebuild — `estimate_threshold` is the within-leaf median nearest-sibling gap,
  ELKI/BETULA-standard, `O(M·capacity)`, threshold raised monotonically; reverse-DFS-order reinsert
  matches the reference tree shape; per-feature EWMA `decay`; runtime-selectable routing; **parallel
  shard+merge build**
  `build_parallel` / `n_jobs` — each shard summarises to `max_leaves/shards` leaves so the merge
  stays ~`max_leaves` CFs, giving ~4–5× on large `N` at equal granularity; opt-in, default serial;
  optional **robust insertion** `set_huber_k(k)` — winsorize a point to ±`k·σ` of its target
  microcluster before the Welford fold so stream outliers cannot stretch a centroid/radius, gated by
  a 5-point warm-up and `σ_j = 0` pass-through, leaving a valid `(n, μ, S)`; point inserts only,
  rebuild reinserts unaffected).
- Phase-3a `clustering::{kmeans, xmeans, gmm_diagonal, gmm_full, *_auto, ward_hac, ward_hac_auto}`
  (**Hamerly-accelerated exact Lloyd**, tested ≡ brute; variant-C E-step + NIW/MAP; full-covariance
  GMM; auto-`k` at `n_clusters = 0` for every parametric head — BIC for k-means (X-means) / GMM,
  dendrogram cut for Ward-HAC).
- `clustering::cop_kmeans` — **constrained** (semi-supervised) k-means (COP-KMeans, Wagstaff et al.
  2001): must-link transitive closure into chunklets, cannot-link lifted to chunklets, greedy
  nearest-feasible assignment, `n_init` restarts kept by SSE. Point constraints are translated to
  leaf-index constraints at the Python boundary (`fit(X, must_link, cannot_link)`); a within-leaf
  cannot-link and over-/contradictory constraints return a typed `ConstraintError` rather than
  silently violating. Greedy ⇒ infeasibility is conservative (documented).
- `clustering::kprototypes` — **mixed numeric + categorical** clustering (k-prototypes, Huang 1997).
  `MixedCf` = numeric `(n, μ, S)` (a reused `Diagonal` CF) + one category-count histogram per
  categorical attribute (mode = categorical centroid), an exact mergeable monoid. Distance is
  `‖Δnum‖² + γ·Σ[x_cat ≠ mode]`. A flat leader pass (`summarize_mixed`, bounded `max_leaves`,
  absorb-to-nearest at cap) summarises rows into mixed micro-clusters; k-prototypes then clusters
  those. Standalone `KPrototypes` head (like `DenStream`): the generic CF-tree can't carry a
  categorical schema through `ClusterFeature::new(dim)`, and GMM/Ward are meaningless over categories,
  so it is a separate module, not a tree feature model.
- Phase-3b `clustering::hdbscan` (mutual-reachability + mass-weighted stability + noise).
- `model` (end-to-end fit/predict); `python` (PyO3 abi3 wheel: one-shot `fit_predict`, float32 or
  float64 with no upcast, + streaming `Betula` estimator (f64 *or* f32 tree, picked at first fit)
  with `partial_fit` / `fit` / `predict` / `fit_predict`, plus `save` / `load` + pickle persistence
  via serde + CBOR (`ciborium`), schema-versioned). The compiled module is the private `_core`; the public
  `betula_cluster.Betula` is a thin Python estimator over it so the scikit-learn parameter protocol
  (`get_params` / `set_params`) returns the identity-stable objects `clone` / `Pipeline` /
  `GridSearchCV` require — a compiled getter rebuilds Python objects each call and fails that check.
  The package is typed (PEP 561 `py.typed` + `__init__.pyi`). Inputs are validated at the boundary:
  a non-finite value (`NaN` / `Inf`) raises `ValueError` rather than silently poisoning the tree.
- Inspection (the dataset-structure layer, not just labels): the estimator exposes microcluster and
  cluster geometry computed in `f64` from the CF-tree it already holds — `microcluster_centers_` /
  `_weights_` / `_radii_` (RMS `sqrt(ssd/w)`), `cluster_centers_` / `_radii_` / `_sizes_` (mass-pooled
  per label, radius via König–Huygens), plus `outlier_scores(X)` (distance to the assigned centroid ÷
  cluster radius, `+inf` for HDBSCAN noise), `assign_microclusters(X)`, and the Python-side `summary`
  / `find_outliers` / `find_near_duplicates` / `sample_representatives` for embedding dataset cleaning
  and deduplication. No extra data passes; the row-mapping helper (`map_rows`) runs them in parallel.
- Soft assignment & summaries: `predict_proba` (true GMM posterior via the per-leaf responsibility
  matrix `microcluster_proba_`, threaded out of the E-step; a documented centroid-distance softmax
  *heuristic* for k-means / Ward / HDBSCAN), `assignment_confidence`, `export_coreset` (leaves as
  weighted points — a streaming coreset), `diagnostics`, `representatives(method=…)`,
  `cluster_profile`; plus the **`topology::mapper`** Mapper graph (`Betula.mapper() → MapperGraph`).
- Parallelism (`parallel` feature, default-on): rayon over point labeling and `estimate_threshold`
  (index-ordered serial reduction → bit-identical to serial), plus opt-in parallel Phase-1 build
  (shard+merge, `n_jobs > 1`) — the latter changes the leaf structure like a different insertion
  order, so it is off by default. `--no-default-features` = fully serial.
- `stats::chi2_quantile` (inverse regularized incomplete gamma, NR `invgammp`; tested vs χ² tables
  to 1e-3) + `distance::MahalanobisChi2` — a Phase-1-safe Mahalanobis-χ² absorption gate with a
  Normal-Inverse-Gamma variance prior `var_eff = (S_j + κ·s₀)/(n + κ)`, so single-point entries fall
  back to the prior scale `s₀` instead of a singular covariance (the guard that lets χ² be used
  during tree growth). Tested mass-invariant (12 vs 10⁴ points → same decision, sklearn #22854) and
  finite at `n = 1`. Exposed to Python on both the one-shot `fit_predict` and the streaming `Betula`
  estimator as `absorb="chi2", chi2_p=…, chi2_scale=…`:
  a runtime `AbsorbKind` enum keeps a single tree type, and the threshold is
  `chi2_quantile(dim, chi2_p)`. The prior scale `s₀` is the user-supplied within-cluster variance
  `chi2_scale` — auto-estimating it from the data picks up between-cluster spread and makes the gate
  too loose (the Python test suite caught exactly this, so the scale is now explicit). Default stays
  euclidean (byte-for-byte identical to before — verified ARI unchanged).

**Embeddings / cosine (decided, measured in `bench/cosine_spike.py`).** `normalize=True` L2-normalizes
every input row (at the Rust boundary, so fit/predict/inspection stay in the same space; persisted as a
constructor param) — on varying-norm direction clusters it lifts ARI from ~0 to ~1. A *native* cosine
CF was evaluated and **rejected**: on the unit sphere squared-Euclidean is monotone in cosine
(`d² = 2 − 2cosθ`), so absorption/routing decisions are identical and `normalize=True` already *is*
spherical clustering; only the centroid representation would differ. The earlier "non-monotonic
`max_leaves`" symptom resolved into two distinct causes — (1) the rebuild threshold: `estimate_threshold`
is now the within-leaf median nearest-sibling gap (an earlier global, stride-sampled scan could
over-grow the threshold and collapse the tree far below `max_leaves`); and (2) an
**inherent BIRCH-family limit**: when clusters overlap at the compression scale (within-≈between-cluster
distance, e.g. loose high-dim spherical clusters), greedy absorption merges across them, sensitive to
insertion order. sklearn-Birch shares this (it only "escapes" by not compressing). Mitigation is more
leaves; the `n_rebuilds_` / `threshold_` getters expose when the tree is thrashing.

**Experimental:** the `topology::mapper` Mapper graph (works + tested, but the linkage scale / lens
choice need tuning per dataset; treat as exploration, not a stable partition API). The FD-sketch
high-d head is correct but slow (eigendecomposition per shrink) — use it only when a `d×d`-per-leaf
tree will not fit.

**Planned / not yet done:**
- Performance: Yinyang k-means, rayon over the GMM E-step. (GPU kernels are **out of scope** — the
  CF compression makes Phase-3 cheap enough on CPU that a GPU path is not worth its complexity.)
- API/eng: a hierarchical sparse CF-tree (`O(log L · nnz)` routing) refining the flat sparse-native
  pass; workspace split into `betula-{sketch,stream,index}` crates. See the deferred-index note below.

(`memory_budget_mb` → `max_leaves` sizing, `snapshot` / `compare_snapshots` drift monitoring,
`active_learning_batch` curation, the streaming density heads **DenStream** (`stream::DenStream`) and
**DbStream** (`stream::DbStream` → `betula_cluster.DbStream`; DBSTREAM with shared-density
connectivity — micro-clusters linked by overlap mass `≥ α·min_weight`, so arbitrarily-shaped clusters
chain and close-but-disconnected regions stay apart), and the **betula-sketch** quantile sketches
(`sketch::{KllSketch, DdSketch}` → `betula_cluster.KllSketch` / `DdSketch`) are **done** — the latter
live in `src/sketch/` as a module today; promoting them to a standalone `betula-sketch` crate is a
mechanical follow-up.)

**Sparse CSR input** (done): `fit` / `fit_predict` / `partial_fit` / `predict` accept a `scipy.sparse`
matrix; the wrapper hands its CSR arrays to f64 `*_csr` core methods (`stream_chunk_csr` /
`route_csr`) that expand one row at a time into a reused dense buffer — the dense `N × d` matrix is
never materialized (only the tree plus one row). f64-only. This dense-tree path costs `O(N·d)` (the
CF centroid is dense and the Welford mean update touches every dimension — the same cost as any
CF-tree method, sklearn-Birch included) and keeps the cancellation-free guarantee.

**`O(nnz)` sparse-native** (done, `sparse.rs` → `betula_cluster.fit_predict_sparse`): a one-shot path
for very high-`d` sparse data. A micro-cluster keeps `(n, ΣX, ‖ΣX‖², S)`, so the mean `μ = ΣX/n`,
cached `‖μ‖² = ‖ΣX‖²/n²`, and the centroid distance `‖x−μ‖² = ‖x‖² − 2⟨x,μ⟩ + ‖μ‖²` update/evaluate
in `O(nnz)`. Rows are summarised by a flat leader pass (bounded by `max_leaves`), then materialised to
dense `Spherical` and clustered by the ordinary Phase-3 heads (`kmeans` default). **Trade-off:** an
`O(nnz)` scatter update is only possible via this *expanded* squared-distance form, which is **not**
cancellation-free (accurate when sparse rows sit far from the dense centroid; near-duplicate dense
points lose precision). The dense path remains the stable default; this is a documented opt-in. A
hierarchical sparse CF-tree (`O(log L · nnz)` routing vs the flat pass's `O(L · nnz)`) is a possible
future refinement.

## Known limitations
Honest scope — inherent to CF-compression + streaming, not bugs:
1. **Insertion-order sensitive**, like all BIRCH-family streaming (the parallel build differs from
   serial, as a different order would).
2. **`threshold` / `max_leaves` are real hyperparameters** (compression vs resolution; watch
   `n_rebuilds_` / `threshold_`).
3. **CF-level heads approximate raw-data clustering** — quality degrades when clusters overlap at the
   compression scale; mitigation is more leaves.
4. **HDBSCAN-on-CF ≠ raw-point HDBSCAN** — mass-aware HDBSCAN over the microclusters (approximate).
5. **Expected-log GMM optimizes a CF-level objective**, not pointwise EM (a measured, deliberate
   objective choice).
6. **FD is an approximate low-rank covariance** (exact only up to rank `ℓ`).

## Non-goals / Deferred: string indexing (`betula-index`)
String/key/path indexing is **intentionally out of the numeric core**. The core is numeric — `float`
vectors, arena-addressed CF-tree (integer node ids), numeric microcluster/label arrays — so a trie
does not improve the math or the algorithm. It would only help a *layer on top of results*: huge
catalogs / semantic-hierarchy browsing (e.g. `movies/action/sci-fi/...`), external string IDs →
cluster ids, prefix/autocomplete navigation. That belongs in a future optional crate `betula-index`,
**pure-Rust first** — `ptr_hash` (minimal perfect hash) for exact `string → id`, `fst` for
prefix/ordered/automaton lookup; a custom succinct trie only if benchmarks show those insufficient.
We deliberately do **not** add `marisa-trie` or any C++ trie dependency (it would break the portable
abi3 wheel). Add a dependency only when a concrete string-indexing API exists to support.

## References
BETULA: Lang & Schubert, SISAP 2020 / Information Systems 2022. BIRCH: Zhang et al., SIGMOD 1996.
k-means accel: Elkan 2003, Hamerly 2010, Ding 2015 (Yinyang). Init: Arthur–Vassilvitskii 2007.
HDBSCAN: Campello–Moulavi–Sander 2013. Stable moments: Schubert–Gertz 2018; Chan–Golub–LeVeque.
Frequent Directions: Liberty 2013. Full proofs/tests: `../../math_improove/`, `research/`.
