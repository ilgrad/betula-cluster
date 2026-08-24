# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`projection="svd"`: a CF-weighted PCA of the leaf summary, and the one-call text pipeline.**
  Clustering TF-IDF in its own geometry does not work — the sparse path scores ARI **0.003** on
  20-newsgroups — and the standard fix is to reduce first. This does the reduction inside the same
  call, factorizing the `M ≈ 10³` leaf centroids rather than the `N` documents: on 20-newsgroups
  (18 846 × 2 000, k=20, rank 50, median of seeds 0/1/2) `fit_predict_sparse(..., projection="svd",
  max_leaves=256)` scores **0.130 in 0.30 s**, against `TruncatedSVD(50)` + `KMeans` at 0.143 in
  0.54 s; at `max_leaves=2048` it reaches **0.152 in 5.4 s**. Available on `fit_predict`, the
  `Betula` estimator and `fit_predict_sparse`, and unlike the NMF projection it accepts signed data.

  **The projection is only half of it — the other half is that a PCA is a linear map.** A raw row
  encodes as `(x − x̄)Vᵀ`, so a projected fit keeps the head's own point rule instead of falling back
  to "answer with the row's leaf's label". Measured on the same basis and the same rows, that
  distinction alone is worth **0.062 ARI** (0.097 → 0.159). The NMF projection cannot be given the
  same treatment: its code is the solution of a per-row nonnegative least squares, not a matrix
  product, so it keeps the microcluster route and the docs now say why.

  Two measured facts decide whether it works for a given corpus, both in `docs/USAGE.md`. Use a
  **cosine head**: `method="kmeans"` on the same codes scores 0.014 against `spherical-kmeans`'s
  0.152, because the leading principal direction of a TF-IDF corpus is document length. And **the
  leaf budget is the cost, not the projection**: sweeping rank 1 → 100 moves the total by 1.2 s,
  while `max_leaves` 256 → 2048 moves it from 0.30 s to 5.4 s, the sparse summarizer being quadratic
  in the micro-cluster count.

  The basis is not a compromise for being built from a summary. Labelling raw rows in it scores 0.159
  against 0.143 for `TruncatedSVD`'s own basis on the same rows — under the spherical cluster feature
  the discarded within-leaf scatter is isotropic, so it shifts eigenvalues and leaves the directions
  alone. `bench/_worker.py`'s `betula-svd` row now runs betula's own reduction; it previously
  borrowed scikit-learn's `TruncatedSVD` and so measured scikit-learn's reducer.
- **`refine=n`: BIRCH's Phase 4, which this crate never had.** `n` Lloyd sweeps over the raw rows,
  warm-started from the Phase-3 centres. Off by default, centroid heads only (`kmeans`,
  `spherical-kmeans` — a mixture assigns by maximum posterior and Ward/Spectral/Leiden by
  microcluster, so sweeping centres would substitute a partition they never fit), and in-memory
  `fit` / `fit_predict` only: `partial_fit` keeps a tree rather than the data, and the CSR path would
  have to densify the matrix it exists to avoid. **Label-changing where enabled** — which is why it
  is opt-in rather than a new default.

  It moves the objective exactly where the theory says — a summary coarse relative to the data — and
  **on this benchmark that is not the same as moving the answer.** On MNIST (first 20 000 rows,
  784-D, `StandardScaler`, k=10, spherical CF, `max_leaves=4000`, median of seeds 0/1/2) twenty
  sweeps take the k-means objective 11 750 563 → 11 710 630 for 4.2 s → 6.4 s, and take ARI
  0.315 → 0.309 with it. Five sweeps: 11 720 402 and 0.311. The descent is monotone in the objective
  and monotone *downwards* in ARI.

  **The acceptance target was not met, and the misses are the interesting part.** Against
  `sklearn.cluster.KMeans(n_init=10)` (ARI 0.324, objective 11 671 351, 19.3 s) refinement lands at
  3.0× the speed but 0.34 % *above* the objective: one warm start from a lossy summary settles in a
  worse basin than the best of ten k-means++ restarts, and more sweeps cannot leave a basin.
  `digits` and `covtype` gain nothing at all, structurally — `digits` at `max_leaves=4000` realises
  1 797 leaves for 1 797 rows, so Phase 3 already *is* exact k-means on the raw points and Phase 4
  begins at its fixed point, while `covtype` moves 0.1993 → 0.1998. MNIST at `max_leaves=16000`
  behaves like `digits`: 0.3237 at objective 11 671 813 — scikit-learn's own answer in 15.8 s
  against its 19.3 s — and twenty sweeps then move the ARI by 0.0001.

  The measurement also sharpens the caveat the feature ships with: on this benchmark a **lower
  k-means objective goes with a worse partition**, twice. `covtype` — `n_init=10` objective 827 314 /
  ARI 0.174, `n_init=1` objective 832 081 / ARI 0.277. `digits` — `n_init=10` 69 405 / 0.468,
  `n_init=1` 69 749 / 0.559. `refine` optimizes the objective faithfully; `docs/USAGE.md` says
  plainly that this is not the same as optimizing the answer.
- **The full BIRCH absorption grid, and D3 implemented for the first time.** `absorb` accepted only
  `euclidean` and `chi2`; it now takes `manhattan` (D1), `average` (D2), `diameter` (D3), `ward` (D4)
  and `radius` (R) as well. D3 — the mean squared distance *within* the cell that results from the
  merge — did not exist in `src/distance.rs` at all, so a third of the criteria a BIRCH paper refers
  to were unreachable.

  D3 closes over `(n, μ, S)` because the double sum telescopes: `Σᵢⱼ‖xᵢ−xⱼ‖² = 2·n·S`, leaving
  `D3²(A,B) = 2·(S_A + S_B + n_A·n_B/(n_A+n_B)·‖Δμ‖²)/(n_A+n_B−1)` — D4's merge term and the two
  scatters over one fewer than the merged mass. Checked against brute-force enumeration of the
  underlying points, both in a Rust fixture and over random point sets to 5.7e-14.

  **The default does not move**, and `docs/USAGE.md` now says why rather than leaving it implicit:
  the thesis tunes absorption for minimum variance and prefers D4/D2, while this crate chose
  mass-invariance, and the two objectives genuinely conflict — the variance-minimising criteria
  inherit the BIRCH size-imbalance bug that `chi2` exists to fix. `threshold` is read in each
  criterion's own units (L1 for `manhattan`, squared for the rest, a χ² quantile for `chi2`), so it
  does not transfer between them; the table in `docs/USAGE.md` states the units per criterion.

  The one-shot path had carried a verbatim copy of the estimator's gate resolution, which would have
  meant adding every criterion twice; it now calls the same `resolve_gate`, and the accepted-value
  list exists once so the parser and the error messages cannot drift apart.
- **A `UserWarning` when the CF summary is too coarse to carry `n_clusters`.** Asking for `k`
  clusters from fewer than `2k` leaves silently produced a worse partition with no signal; the
  warning names the realised leaf count, `k`, their ratio and the current `max_leaves`, and points
  at the three parameters that fix it. Fires from every entry point that partitions a summary into a
  caller-supplied `k` — the `fit_predict` function, the `Betula` estimator (dense and CSR),
  `fit_predict_sparse` and `KPrototypes` — and stays silent for auto-`k` and for the heads that
  discover their own count (`hdbscan`, `scale-space`, `leiden`).

  The threshold is measured, not assumed: over three seeds on the `ward` head, well-separated data
  is already at its plateau ARI at ≈2 leaves per cluster and loses 29 % (k=50) / 55 % (k=200) of it
  at ≈1, while `digits` and `covtype` score 0.000 and 0.003 at ≈1. Lang's thesis reports the same
  floor from the other end (Sec. 5.5.4). It is a *floor*, not a recommendation — `covtype` peaks at
  ≈8 leaves per cluster and declines after — and the check reads the **realised** leaf count, since
  the tree routinely settles below its cap and at `n < max_leaves` the cap never binds at all.

### Changed
- **The CF k-means++ sampling weight now carries the leaf's own scatter.** Seeding drew each
  candidate centre proportional to `n_i·D²_i` — the point-level k-means++ weight, with the leaf
  treated as a point at its mean. The exact CF-adapted potential is Lang's Eq. 5.4,
  **`S_i + n_i·D²_i`**, and the missing `S_i` is not a rounding detail: a wide leaf sitting on an
  already-chosen centre has `D²_i = 0` and so was **unsamplable however much scatter it held**. The
  greedy candidate score inside the trial loop is unchanged and correct as it was — it differs from
  the exact potential by `Σ_i S_i`, which is constant in the candidate. `cop_kmeans` gets the same
  weight, over a chunklet scatter assembled by König-Huygens (`Σ S_i + Σ n_i‖μ_i − c‖²`), so with no
  must-links it seeds identically to `kmeans`.

  **This is on correctness grounds and claims no quality gain — it relabels output and the
  benchmark moved in both directions.** It reaches `kmeans`, `gmm`, `gmm-full`, `spectral`, `xmeans`
  and `cop_kmeans`; `ward`, `leiden`, `hdbscan`, `vmf` and `spherical-kmeans` (which has its own
  angular seeder) are untouched, and every scikit-learn row reproduced to the last digit. It is a
  **no-op wherever the summary is lossless**: at `max_leaves ≥ N` every leaf holds one point, `S_i`
  is 0, and the weight is bit-identical to the old one — which is why the whole synthetic quality
  table and every `digits` row are unchanged, and it is a useful self-check that a benchmark budget
  actually compresses.

  Where it does bite, over three seeds (`bench/RESULTS.md` re-measured whole): covtype `spectral`
  0.037 → **0.100** and MNIST `spectral` 0.155 → **0.203**; MNIST `kmeans` 0.302 → 0.307; covtype
  `kmeans` 0.088 → 0.074, `gmm` 0.088 → 0.076. At 16 000 leaves covtype `gmm` goes 0.094 → **0.104**
  and `kmeans` 0.078 → 0.067. The covtype GMM result the README quoted at the *default* budget was a
  win by 0.008 and is now a loss by 0.004 — both sit inside two nearly coincident seed ranges
  (0.055–0.096 against 0.055–0.102), so the honest reading is that it was a tie before and is a tie
  now, and the page says so. The claim that survives is the one at adequate leaf resolution, which
  got stronger.

  It also inverted the `refine=` MNIST result above: with the new seeding Phase 3 already lands at
  ARI 0.315 rather than 0.275, and twenty Lloyd sweeps now lower the objective while lowering the
  ARI. That entry was re-measured rather than left standing.

  A prior sweep of ELKI's three `CFInitWeight` options at 18×/6×/2× compression over five seeds
  found no consistent winner and every difference inside the seed spread
  (`local/scratch/elki/init_weight_ab.py`); Lang's thesis separates them only at RMSD ≈ 0.005 with
  ±0.004 confidence intervals. This change is made because the formula is the right one, not because
  it scores better.
- **`min_samples` now counts the microcluster itself** in `method="hdbscan"`, so `min_samples=1`
  leaves every core distance at 0 and HDBSCAN\* degenerates to single linkage. The convention is a
  genuine split in the field — Campello, Moulavi & Sander's Def. 3.1, `sklearn.cluster.HDBSCAN` and
  ELKI (whose parameter reads "including this point") all include the object, while
  `scikit-learn-contrib/hdbscan` excludes it — and it was previously stated nowhere: not in the Rust
  docs, the `.pyi` or `docs/USAGE.md`. The shipped behaviour was the *exclusive* one, so
  `min_samples=5` acted like the standard `min_samples=6`, and the published benchmark compared our
  effective 11 against `sklearn-hdbscan`'s 10.

  Aligning with the majority — and with the library this project mirrors and benchmarks against —
  makes the comparison like-for-like. **This relabels `method="hdbscan"` output**: to keep the old
  behaviour, add one to `min_samples`. The convention is now pinned by a test on the core distances
  themselves rather than inferred from a downstream partition.

  The four quality tables were re-measured over three seeds after the change. Cell by cell against
  the previous run, **only `betula-hdbscan` rows moved** — every other method reproduced to the last
  digit on every metric — so the deltas are attributable to this and nothing else. `digits` goes from
  a loss to a win (ARI 0.146 → **0.164** against `sklearn-hdbscan`'s 0.149, both zero-spread, with
  the noise fraction down from 0.620 to 0.580); `blobs` 0.142 → 0.154 and `varied` 0.519 → 0.536;
  `aniso` 0.569 → 0.568 and `covtype` +0.0001 are ties; MNIST stays all-noise.

### Fixed
- **`distance="average"` panicked with an index-out-of-bounds instead of clustering.** A node split
  picks the farthest pair of children as seeds and then assigns every child to the nearer seed —
  which silently assumes `between(cf, cf) == 0`. That holds for `euclidean`, `manhattan` and `ward`,
  but not for `average` (D2): the average inter-point distance between a cluster and *itself* is
  `2·S/n`, not zero. A seed with enough of its own scatter therefore measured further from itself
  than from the other seed and joined the wrong group; once every child did that, the sibling node
  was created with no children, and the next `descend` indexed `children[0]` on it. Reproduced on
  `load_digits` at `max_leaves ≥ 800` with any `feature`.

  The seeds now anchor their own groups by construction, so both sides are non-empty whatever the
  routing measure does — the same root cause the seed scan already guards against by skipping
  `i == j`, one step later in the same function. This is the second half of that fix, and it is what
  the non-metric CF distances of the planned linkage and Bregman work need in order to be safe.
  **Labels are unchanged on every metric route** (48/48 `(dataset × feature × head × distance ×
  seed)` label arrays hash identically across the fix), so nothing published moves.

## [0.7.0] — 2026-08-23

### Changed
- **Rust 2024 edition; minimum supported Rust version 1.82 → 1.85.** The migration needed two source
  changes in total (a binding mode made explicit in `gmm_toeplitz.rs`); the rest is `rustfmt`'s 2024
  style edition collapsing short `if`/`else` onto one line. 1.85 is the edition's own floor.
- **A rebuild no longer rebalances the whole tree every time it compacts.** 0.6.0 set the leaf count
  exactly by merging the `k` closest sibling pairs, which is also the policy that grows the
  absorption threshold by the smallest amount that reaches the count — so the 10 % headroom refilled
  almost at once and the rebuild fired again. Each one then called the rebalance pass, which clears
  the node and entry arenas and re-routes every surviving entry: on 1 M blobs at `max_leaves=4000`
  that is 385 rebuilds × ~3600 entries ≈ 1.4 M descents *on top of* the 1 M point inserts, and it
  made `fit` ~3.2× slower than 0.5.0 on the tree-dominated path. A profile hides it, because the
  rebuild's cost *is* `descend` and `try_absorb`.

  Compaction merges strictly inside a leaf, so mass is conserved per node and the tree is already
  valid when it finishes — the rebalance is a re-partitioning of leaves that mix two clusters, not a
  repair, and the drift it corrects accumulates with *merges* rather than with compactions. It now
  runs once the merges since the last rebalance reach `max_leaves`. Measured on 1 M blobs at
  `max_leaves=4000`, median of 5 fits, arms alternated on an idle machine: **0.743 s → 0.230 s**,
  385 rebuilds → 31, leaf utilization 92.4 % → 92.6 %, ARI 0.8597 → 0.8598. Labels move wherever a
  rebuild actually fires — three seeds each, median [min–max]:

  | | 0.6.0 | this release |
  |---|---|---|
  | mnist-20k `kmeans` | 0.2275 [0.1719–0.2510] | 0.2571 [0.2275–0.3263] |
  | mnist-20k `gmm` | 0.2842 [0.2547–0.2842] | 0.2788 [0.2668–0.2788] |
  | mnist-20k `ward` | 0.3272 | 0.3666 |
  | covtype-20k `kmeans` | 0.0528 [0.0506–0.1061] | 0.0878 [0.0755–0.0999] |
  | covtype-20k `gmm` | 0.0843 [0.0782–0.0962] | 0.0882 [0.0820–0.1129] |
  | covtype-20k `ward` | 0.0910 | 0.0910 |

  digits does not move: at n = 1797 against a 4000-leaf budget it never rebuilds. Widening the
  compaction margin to `max_leaves / 6` was measured as the alternative and rejected — it reaches
  only 0.369 s and takes covtype utilization down to 83.7 %, which is the 90–99 % band the exact-count
  compaction was written to provide. `CFTree` gains one `usize` field, `#[serde(default)]`, so
  `persistence` snapshots written by 0.6.0 still load and `SCHEMA_VERSION` stays at 2 — verified by
  a snapshot from the released 0.6.0 wheel, committed at `tests/data/v2_0.6.0.betula` alongside the
  attributes it should reproduce. `test_save_load_roundtrip` writes and reads with the same build
  and cannot see a schema break at all; dropping the `serde(default)` makes the new test fail with
  `missing field merged_since_rebalance`, which is the claim itself.

### Fixed
- **The `vmf` head reported the same concentration for every tight cluster in high dimension.** `κ`
  was clamped at `10⁴`, which in 784 dimensions binds from `R̄ ≈ 0.9616` — ordinary for embedding
  data. At `R̄ = 0.99` the clamp understated the Banerjee estimate 3.9×, and at `R̄ = 0.999` 39×
  (true `κ ≈ 3.91·10⁵`, reported `10⁴`). Every saturated component therefore carried an identical,
  wrong `κ`, which biases `movmf_auto`'s BIC uniformly and can move the `k` it selects.

  The comment blamed numerical stability; the cause was cost. `log I_ν(κ)` came from an ascending
  power series whose peak term sits at `m ≈ κ/2`, so it is `O(κ)` inside the EM loop, and its own
  `m > 200_000` stop truncates *before* the peak above `κ ≈ 4·10⁵` — raising the cap alone would have
  traded a wrong answer for a slow one and then for a wrong one again. `log_iv` now branches at
  `κ = 10⁴` to DLMF 10.41.3, the uniform asymptotic expansion for large order, which is `O(1)` and
  correctly rounded there — measured against 50-digit arithmetic over `ν ∈ [1, 2047]`,
  `κ ∈ [10⁴, 10⁶]`, the f64 result is within **0.8 ulp**. The cap rises to `10⁶`, set by
  representability rather than by the normalizer. The expansion divides by `ν = d/2 − 1` and so does
  not exist below `d = 4`; there the series remains the only evaluator and the cap stays at `10⁴`.

  **Labels move for `vmf` in high dimension**, and `n_clusters=0` can select a different component
  count. No published benchmark table uses the head, so no committed number changes.
- **`predict` could not return a cluster whose radius is zero.** The Voronoi rule that `predict`
  applies for the centroid heads (`kmeans`, `spherical-kmeans`, `ward`, …) filters out clusters with
  no weight, but the stats helper it reads yields radii *before* weights for clusters — the opposite
  of the order it uses for leaves — and the rule destructured it in leaf order. Every degenerate
  cluster (a singleton, or one whose members coincide) was therefore dropped from the rule and became
  unreachable: the point sitting exactly on such a centre was labelled with some other cluster, at a
  distance three orders of magnitude larger. Only the filter was wrong; the centres themselves,
  `cluster_sizes_` and `cluster_radii_` were always correct.
- **Leiden was not reproducible: the same graph and the same `seed` gave different communities on
  successive calls in one process.** `refine` (and, on a tied graph, `one_level`) picked its target
  sub-community by scanning a `HashMap`, and kept whichever tied candidate the iterator produced
  first. `std::collections::HashMap` draws a fresh `RandomState` seed per instance, so the candidate
  order — and with it the tie-break — changed between calls; `seed` only ever controlled the node
  visit order. `aggregate` had the same problem one level down: it emitted each super-node's
  adjacency row in `HashMap` order, and since floating-point addition is not associative, the
  weighted degree summed from that row differed run to run, moving every gain computed from it.
  Both now scan in community / neighbour index order. Labels change for graphs that carry tied
  gains — symmetric ones above all — and are now stable. Measured: reverting the three hunks in place
  and re-running every `betula-leiden` row of `bench/results_quality.csv` and `bench/results_real.csv`
  against the same build reproduces all nine to the last digit, so no published benchmark number moves
  with this fix.
- **Documentation shipped with 0.6.0 still described 0.5.0.** The verified-suite line in `README.md`
  and `DESIGN.md` claimed 185 Rust + a 213-case Python suite; the tree now carries **404 Rust unit +
  4 integration tests** (402 + 4 with `--no-default-features`, 8 more for the `cli` binary) and a
  **230-case** Python suite (229 passed / 1 skipped without the optional `scipy` / `networkx` test
  dependencies). `DESIGN.md` also still described `predict_proba` as the per-leaf responsibility
  matrix, which 0.6.0 replaced with scoring the row itself under the fitted mixture, and its crate
  layout predated `src/mixture.rs`. `docs/api.md` omitted `consensus` / `ConsensusResult` although
  both are exported from `__all__`; `SECURITY.md` still named `0.1.x` as the supported line.
- **Every published benchmark number is re-measured, and several move down.** `bench/RESULTS.md`, all
  eight `results_*.csv`, the four plots and the README headline were a single run of 2026-07-18
  against a 0.2.0 build. They are now the **median of seeds 0, 1, 2** against this tree, with per-cell
  min/median/max in new `results_*_spread.csv` sidecars and the seed list in `results_seeds.json` —
  because on the synthetic sets every row moves by more than 0.05 ARI across three seeds, and the old
  single-run page had in several places published the top of that range. Corrections worth naming:
  `digits` k-means is a **tie** (0.467 vs scikit-learn's 0.468, not 0.568 vs 0.468); the spectral head
  matches `SpectralClustering` at **1.0–1.5×** its speed, not 3–5× (betula's time barely moved,
  scikit-learn's fell threefold between versions); `normalize=True` on MNIST is now a **wash** — its
  0.203 baseline is gone and the off-vs-on sign flips between seeds — though it still earns its place
  on `digits` (k-means 0.467 → 0.569); and CF-weighted NMF is **slower** than scikit-learn's at every
  `N` measured, its win being the ±0.00 seed spread against ±0.37. The `covtype` loss to
  `sklearn-birch` is confirmed as a loss on the merits: matching Birch's compression ratio leaves its
  ARI unchanged (0.132 vs 0.131) and handing betula Birch's 11 774 leaves makes every head worse. The
  23 notebooks still carry 0.2.0-era outputs and their provenance banners still say so.
- References to `../../math_improove/` in `DESIGN.md`, `src/distance.rs`, `src/feature.rs` and
  `src/stats.rs` pointed outside the repository; they now say the working notes are local-only.
- **CI on macOS and Windows broke the moment `.python-version` was added.** That job builds a
  deliberate 3.12 environment (`uv venv --python 3.12`) and installs its tools into it; with a
  repo-wide 3.14 pin present, a bare `uv run` no longer accepts that environment, recreates it, and
  discards the tools — `error: Failed to spawn: maturin`. Both `uv run` calls there, and the two in
  the `mutmut` job that would have hit the same wall on its next schedule, now pass `--no-sync`.
- **Mutation testing had never once produced a result, and covered a third of the crate.** All seven
  scheduled runs since 2026-07-06 were killed by the 90-minute cap: measured on the last of them, 90
  minutes at `-j2` completed **98 of 1011 mutants (9.7%)**. Sharding alone does not fix it — a
  12-way split still leaves ~115 mutants per job against a suite where a single
  method-dispatch test costs 25 of its 27 seconds, and every mutant pays it in full.
  `--test-tool nextest` does, because it stops at the first failing test: measured A/B over the 13
  mutants of `src/kernels.rs` at `-j4`, `cargo test` took **525 s** and reported 8 caught with 2
  timeouts, `nextest` took **144 s** and reported 10 caught with none — **3.6× faster and more
  accurate**, since both former timeouts resolve to caught. That makes full coverage affordable, so
  `examine_globs` is gone: the scope is now the **whole crate, 4310 mutants** (still excluding
  `python.rs` and `bin/`, which have no Rust-level tests and would only yield false survivors),
  split 24 ways at ~180 mutants per job. A summary job collates every shard and fails loudly on a
  survivor. `.cargo/mutants.toml` records the per-module mutant counts and the measured throughput
  so the matrix can be resized from numbers rather than guesses.
- **Four mutants had been surviving in `src/distance.rs` unread inside a cancelled run's artifact.**
  `Radius::between` was pinned only by a case where `na·nb` and `na/nb` coincide (2·1) and where one
  scatter was zero, so `*`→`/` and `+`→`-` both went unnoticed; `MahalanobisChi2::maha_sq` was only
  ever exercised at mean zero, where `x − μ` and `x + μ` agree, so the sign of the residual was
  unconstrained. Two tests now pin both. The fourth (`||`→`&&` in `VarianceIncrease::between`) was
  an equivalent mutant — with exactly one weight zero the formula already returns zero — so the
  guard is now written on the sum, `nab = na + nb`, matching `Radius::between` directly below it.
  Same result for every input; the operator is simply no longer free.
- **The mutation job failed whenever any mutant survived, which meant it never carried a signal.**
  That sounds like the strict setting, and it is the useless one: the crate has never had an empty
  survivor set — some survivors are provably equivalent mutants and no test can kill those — so the
  job was red every single week and a new hole was indistinguishable from the standing debt, which
  is exactly how the four `src/distance.rs` survivors above went unnoticed from July to August. It
  now fails on survivors that are *not* in a committed `mutants-baseline.txt`. That file records the
  set keyed `file:line:col`, grouped by module, each group carrying either the argument for why no
  test can close it or the constraint a killing fixture would have to satisfy — a claim of
  equivalence without an argument does not belong in it. The summary job reports both `comm`
  directions, so an entry that has since been killed surfaces as "trim this" rather than quietly
  padding the debt. Two failure modes of the job itself are closed with it: a shard that uploads no
  artifact is now an error rather than a smaller total that reads as good news (run 32347677860 lost
  shard 39 after 88 minutes against a ~20-minute median, and 95 of 96 collated into a number that
  looked complete), and the matrix is capped at `max-parallel: 12`, below the account's 20-slot
  limit, after the uncapped 96-shard run left ordinary CI of the same commit queued for two hours.
- **Reference and boundary fixtures close 48 of the recorded survivors** — `mutants-baseline.txt`
  goes from 361 entries to 313, with `src/tree.rs` at 22 → 2, `src/clustering/vmf.rs` at 38 → 19,
  `src/clustering/gmm.rs` at 19 → 13, `src/clustering/nmf.rs` at 15 → 13 and
  `src/clustering/kprototypes.rs` at 4 → 3. Each fixture is verified the only way that means
  anything: the mutation is applied at its own `line:col`, with an assertion that the line still
  holds what the entry says it does, and the test is confirmed to fail. Four of them closed
  mechanisms nothing had constrained. `randomized_svd` had no reference check at all — the range finder is deliberately
  insensitive to its own sketch, so an end-to-end NMF assertion cannot see whether the sketch or the
  power iterations are computed correctly — and is now compared against a dense symmetric
  eigendecomposition of `XᵀX` and the Eckart–Young optimum it yields. A rebuild's grown absorption
  threshold, `widest · (1 + 4ε)`, decides absorption for the whole remainder of a run and was pinned
  by nothing. And `split` seeded both its sides from a scan that, one `+`→`*` away, compares a child
  with *itself*: harmless under `CentroidEuclidean`, where a self-distance is zero, but `Radius` and
  `AverageIntercluster` both carry a child's own scatter into `between`, so a high-scatter child is
  further from itself than any two children are from each other and the split degenerates into a
  size tie-break that separates coincident entries. `spherical_lloyd` had the same shape of hole from
  the other side: its independent reference carries the same `if !changed && it > 0 { break; }` line,
  so no comparison against it can see that guard at all — what can is the Lloyd fixed-point condition
  itself, asserted on a fixture slow enough to still be moving at the second assignment. The Rust
  suite is 399 lib + 4 integration tests as a result.

## [0.6.0] — 2026-08-19

### Fixed
- **`predict` returned the label of an approximately-found leaf, not the label of the model's own
  partition.** k-means assigns a point to its nearest centre; the labels came instead from routing the
  point down the CF-tree to a leaf and reading that leaf's cluster. The descent is greedy, so it is an
  *approximate* nearest-microcluster search, and in high dimension it is wrong often: measured
  disagreement with the model's partition of **2.7% (digits), 14.9% (MNIST-20k) and 18.1%
  (20-newsgroups TF-IDF)** of points, and on TF-IDF only **14 of the 20 clusters the head found** were
  reachable by descent at all — the estimator reported `n_clusters_ = 20` while `predict` could never
  emit six of them. The argmin over `k` centres is also the cheaper of the two at `k ≪ n_leaves`.
  Measured ARI, `max_leaves=2000`, median of 3 seeds: MNIST-20k `kmeans` **0.397 → 0.417**,
  20-newsgroups `kmeans` **0.050 → 0.060** and `spherical-kmeans` **0.071 → 0.099**, digits `kmeans`
  0.667 → 0.663. Applies to `method="kmeans"` and `"spherical-kmeans"`, whose objective *is* "assign
  to the nearest centre" (the spherical head compares unit-normalized centres, where the Euclidean
  argmin and the cosine argmax agree). Every other head keeps the microcluster route: the mixture
  heads (GMM, Toeplitz-GMM, movMF) assign by maximum posterior — which weighs each component by its
  own covariance and mixing weight, a different partition rather than a slower route to the same one —
  and Ward / Spectral / Leiden / HDBSCAN / scale-space clusters need not be convex at all. Constrained
  runs (`fit_constrained`) also keep it, since a Voronoi rule is free to violate the pairwise
  constraints COP-KMeans satisfied. Measured separately: a nearest-centre rule *would* raise the GMM
  heads too (digits 0.504 → 0.576, MNIST-20k 0.343 → 0.418, 20news 0.053 → 0.073), which makes a true
  max-posterior `predict` for them the obvious follow-up — see the next entry, which ships it.
- **The mixture heads now label a point by maximum posterior instead of by tree descent.** `gmm`,
  `gmm-full`, `vmf` and `gmm-toeplitz{,-full,-gs}` score the point itself under the mixture they
  converged to — `argmax_c [ln π_c + ln p(x | θ_c)]` — rather than routing it to a leaf and copying that
  leaf's label. Each head hands its converged parameters to a `Mixture` (diagonal / full-Cholesky /
  stationary / vMF kernels), so the density that labels a point is the density that was fitted: the
  floored diagonal variances, the ridge-regularized Cholesky, the AR predictor bank. `predict_proba`
  normalizes the same scores, so `predict_proba(X).argmax(1) == predict(X)` by construction, and it now
  scores the row rather than the row's microcluster. A component that no leaf claims is silenced, so
  `predict` can only emit a label the fitted partition uses. Measured ARI, median of 3 seeds, defaults
  (`max_leaves=2000`), old rule → new rule: 20-newsgroups TF-IDF `gmm` **0.027 → 0.054** (72.1%
  relabelled) and `vmf` 0.021 → 0.026, digits `gmm` 0.489 → 0.507, `gmm-full` 0.722 → 0.754, `vmf`
  0.643 → 0.631, blobs `gmm` 1.000 → 1.000.
  **It costs accuracy on raw image pixels at a fine leaf budget, and that is not hidden here:**
  MNIST-20k `gmm` **0.340 → 0.185** (52.1% relabelled). The cause is the model, not the rule: on the
  same fit a *nearest-centre* rule, which ignores the covariance entirely, scores **0.378** — above
  both — so it is the fitted covariance that costs the ARI. A diagonal covariance treats 784 correlated
  pixels as independent, so scoring a raw point sums 784 separate penalties and the per-dimension
  modelling error accumulates; the leaf-level score damps this through the `−½ tr(Σ_c⁻¹ Σ_i)` term,
  which a single observation genuinely does not have, and the coarser the leaves the more that term
  smooths. The loss is accordingly budget-specific rather than head-specific: at `max_leaves=300` the
  new rule wins for `gmm` (0.206 → **0.239**) and for `gmm-full` (0.207 → **0.212**) alike. For raw
  images prefer `kmeans` or a `projection`.
- **The diagonal GMM could drive `1/σ²` to `1e12` in a dimension with no spread.** The variance floor
  was expressed purely relative to the spread of the dimension itself (`1e-3 · gvar_d + 1e-12`), so a
  constant or near-constant dimension — an always-zero border pixel, a term absent from every document —
  got a floor of essentially zero and its precision was bounded only by `1e-12`. Compounding it,
  `global_variance` summed only the between-leaf term while its own doc comment claimed between + within,
  so the scale feeding both the floor and the NIG prior was under-estimated in proportion to how hard the
  tree had compressed. Neither showed while the density was only ever evaluated at leaf means, which sit
  close to any component mean; a raw observation does not. The floor now also carries a global term
  (`VAR_FLOOR_REL · mean_d gvar_d`, mirroring the full head's `ridge = 1e-6 · tr(gcov)/d`) and
  `global_variance` adds the per-leaf term, as `global_cov` always did. Measured ARI, median of 3 seeds,
  before → after, as leaf-descent / max-posterior: MNIST-20k `gmm` 0.317 → 0.340 / **0.135 → 0.176**,
  20-newsgroups `gmm` 0.029 → 0.027 / **0.051 → 0.059**, digits `gmm` 0.471 → 0.479 / 0.484 → 0.496.
  `gmm-full`, `vmf` and blobs are byte-identical, as they must be — they never used this floor. The fix
  is real but partial: it lifts the MNIST point-level ARI by 30% relative and still leaves it below leaf
  descent, for the modelling reason given in the entry above.
- **Both GMM heads seeded every component with the *global* covariance, so a junk dimension could
  decide the first E-step.** `gmm` started each component's variance at `max(gvar_d, floor_d)` and
  `gmm-full` at `gcov` — the spread of the whole dataset. In a dimension that separates the clusters
  that spread is dominated by the *between*-cluster distance and says little about the within-component
  scatter: on four blobs eight units apart it overstates it **34×**. A dimension whose spread is already
  purely within-component — a sparse binary feature, a near-constant pixel — gets no such inflation, so
  a uniform-looking seed is in fact a skewed metric that weighs the junk dimension up to **889× more per
  dimension**, and one spike there costs more (51.7) than the entire blob separation (3.9). Both heads
  then converge with every component mean sitting on the global mean. Reproduced with four trivially
  separable blobs plus twelve binary columns that are 1 with probability 0.02: ARI **0.000 → 1.000**
  (`gmm`) and **0.001 → 0.985** (`gmm-full`), while `kmeans`, `ward` and `hdbscan` scored 1.000 on the
  very same CF-tree throughout — as did scikit-learn's `GaussianMixture(covariance_type="diag")`, which
  seeds from its k-means partition. Each component is now seeded from its own k-means cluster (the same
  within-cluster accumulation the M-step performs, the same floor and ridge, the global value as the
  fallback for an empty cluster). On real data this lands EM in a different local optimum and the effect
  is small in both directions (max-posterior ARI, median of 3 seeds): digits `gmm` 0.496 → **0.507** and
  MNIST-20k `gmm` 0.176 → **0.185**, against digits `gmm-full` 0.777 → 0.754 and 20-newsgroups `gmm`
  0.059 → 0.054. `vmf` never had the defect — it seeds κ from the hard assignment — and `gmm-toeplitz`
  starts from random responsibilities; both are unchanged to the last digit, as are all non-mixture heads.
- **CF-tree: the rebuild spent a third of the leaf budget, and collapsed the tree outright in high
  dimension.** `max_leaves` is a resolution budget — the summary handed to the global clustering is
  only as fine as the leaves actually kept — but the rebuild grew the threshold to the *mean*
  within-leaf nearest-sibling gap and then merged whatever fell under it, which is a prediction, not a
  target. Measured utilization of the budget: **65.6–96.2%** (mean 80.6%) across digits, blobs at
  n = 20 000/100 000 and d = 10/50, and uniform d = 20. On 3000-dimensional TF-IDF it failed
  catastrophically instead — **3 leaves against a 2000 budget**, one of them holding 85.7% of the mass,
  ARI 0.0 for every method and every seed. The cause is concentration of measure: the achievable leaf
  count is near-discontinuous in the threshold (7755 leaves at `threshold=1.0`, **12** at `1.3`), so no
  threshold-first policy has a safe value there. The rebuild now merges the `k` closest sibling pairs
  with `k` set by the budget and reads the grown threshold off the widest gap it took — `k` is exact,
  and merging is capped at one pair per entry, so the cliff cannot be stepped over. Utilization is now
  **90.0–99.0%** (mean 93.5%) on the same datasets and **90.0/94.1%** on TF-IDF (was 0.6/0.2%), where
  ARI goes 0.0 → 0.071 with the `spherical-kmeans` head.
- **CF-tree: the rebuild conflated reducing the entry count with rebalancing the node structure.**
  Merging two entries *inside their own leaf node* leaves every node CF in the tree exactly unchanged —
  a node's CF is the merge of its subtree, and merging two of its children does not change that
  multiset union — so mass is conserved per node, no ancestor needs touching, and no leaf can be
  emptied. The count reduction is therefore done in place, in one `O(Σ_leaf child_count²)` scan plus an
  `O(m log m)` sort, and the reinsertion pass that follows now merges **nothing**: it only re-routes
  entries so that leaves re-partition around the geometry the data actually has, which compaction
  alone cannot do (it can shrink a leaf that mixes two clusters but never split it). Keeping absorption
  on during that pass is what walks off the concentration cliff — measured, it collapses a d = 50 blob
  mixture to **9 leaves against a 500 budget**. Rebuilds became more frequent (each shaves 10% rather
  than overshooting) and individually cheaper; end-to-end build time moved **−6% to +98%** across the
  eight configurations, with the two largest (blobs n = 100 000, d = 50) essentially unchanged at
  +1% and +3%. `n_rebuilds_` counts a different, cheaper unit of work than before and is not comparable
  across the change.

  Quality follows the head, not the leaf count, and the finer summary is not uniformly better: on
  digits (5 seeds, median) the `kmeans` head goes **0.646 → 0.667** while the `gmm` head goes
  **0.593 → 0.458** at `max_leaves=500` — the diagonal-covariance head degrades as leaves get smaller
  and is best on that dataset at ~230 leaves, which is now a `max_leaves` choice rather than something
  the rebuild does behind the caller's back. Blob mixtures stay at ARI 1.000 throughout.
- **CF-weighted NMF: the solver stopped after 5 sweeps regardless of `max_iter`.** The convergence test
  compared `|prev − err|` against `tol · prev` with `prev` seeded at `+inf`; IEEE-754 says `inf <= inf`,
  so the first check always fired and the iteration budget was dead code. Every other EM loop in the
  crate guards this with `it > 0`; `nmf.rs` was the only one that did not. The budget is now honoured,
  and a regression test asserts the residual keeps falling as the budget grows.
- **CF-weighted NMF: most components collapsed to zero and never recovered.** Two independent causes,
  both fatal because zero is an absorbing state for HALS and for the multiplicative updates. (1) The
  randomized SVD recovers the right vector as `v = Bᵀu/σ`; below the noise floor that division amplified
  round-off into a vector of arbitrary magnitude, so a rank-deficient triplet seeded a component wildly
  out of scale. Such triplets are now cut at the LAPACK numerical-rank threshold and reported as exact
  zeros. (2) NNDSVD's zero-fill used `mean(X)`, but a filled component is a rank-1 block of constant
  magnitude and `r` of them add up, so the fill swamped the data as the rank grew. The init now uses the
  `ar` fill, `mean(X)·U(0,1)/100` — small enough to stay a perturbation, random enough to break the
  degeneracy a constant fill creates. Measured on a rank-12 matrix at rank 32: initial relative residual
  **13.5 → 1.2**, dead components **28/32 → 3/32** (3 is the honest count — the data is rank 12), and the
  converged residual **270×** lower. On `digits` at rank 24: **15/24 → 0** dead, reconstruction error
  **0.33 → 0.20**, downstream ARI **0.54 → 0.61**. The reconstruction error now matches `scikit-learn`'s
  own NMF to three decimals at every rank tested (10, 16, 24, 32), against being **2.3× worse** before.
- **CF-weighted NMF: an arbitrary per-component scale reweighted the downstream clustering.** NMF is
  invariant to `(W D, D⁻¹H)`, so the optimizer left whatever split it landed on — measured spreads of
  70× between component scales on a converged fit. Because the codes are consumed as a Euclidean
  feature vector by the Phase-3 head, that scale acts as a per-dimension weight and the head clustered
  along whichever component drew the largest number. The factorization is now canonical (component rows
  unit-L2, scale absorbed into the codes, components ordered by descending energy). Measured on a
  4-topic nonnegative mixture over 8 seeds: median ARI **0.63 → 1.00**, seed spread **0.37 → 0.00**;
  against the shipped 0.5.0 behaviour, median **0.92–1.00 → 1.00** with spread **0.31 → 0.00** at
  N = 160 000. `scikit-learn`'s own `NMF` shows the same 0.37 spread on this data, unfixed.

### Added
- **NNDSVDar initialization** for both NMF solvers (Boutsidis & Gallopoulos, 2008), built on a
  self-contained randomized range finder (Halko-Martinsson-Tropp) — deterministic given the seed and far
  better conditioned than the previous random start, which decided the basin a non-convex coordinate
  descent lands in.
- `Betula.components_` — the NMF parts `H` as `(projection_dim, dim)`, unit-L2 rows ordered by
  descending energy, so a row reads directly as a topic over the input features.
- `Betula.reconstruction_err_` — relative reconstruction error `‖X̃ − W H‖_F / ‖X̃‖_F` of the projection
  over the leaf centroid matrix.
- `projection_max_iter` (default 100) — the factorizer's own sweep budget, independent of the head's
  `max_iter`. Previously the two shared one number, so raising the clustering budget silently paid for
  NMF sweeps too.
- `projection="weighted-nmf"` / `"weighted-nmf-kl"` now accept **sparse CSR** input; only the stored
  values are checked for negativity (implicit zeros are already nonnegative). Measured ARI 1.00 on a
  sparse topic mixture, matching the dense path.
- A convergence check for the KL solver (it previously always burned the full `max_iter`). The Frobenius
  solver's check now follows the **size of the update** rather than the size of the objective, compared
  against the first sweep's — sklearn's rule. A relative test on the residual provably never fired: HALS
  converges sublinearly and keeps buying more than `tol` of relative improvement for hundreds of sweeps,
  so `max_iter` was the only brake (measured: a 4× budget cost 6× the time). The movement falls out of
  the sweep for free, so the check is also cheaper than the residual it replaced.

### Changed
- **`WᵀX`, the NMF sweep's hot loop, restructured for locality: measured 2.4-3.4× faster.** Evaluating
  it per output cell (`Σ_j w[j][k]·x[j][c]`) walks a whole column of `X` for each of the `r·d` cells,
  striding `d` floats per step through a matrix far larger than L2. Accumulating into the small,
  cache-resident `r×d` output while reading each row of `X` once, sequentially, computes the same
  product with the access pattern the hardware wants (verified equal to 1e-9 by a unit test).
- The NMF rank is capped by both centroid-matrix dimensions (`min(rank, d, M)`): `rank > M` makes the
  factorization rank-deficient by construction and leaves whole components with nothing to fit.

### Notes
- **A-HALS extrapolation (Ang & Gillis, 2019) was implemented, measured and rejected.** It lost to plain
  HALS at every sweep budget on two planted problems (up to 445× worse residual at 200 sweeps): the
  accept/reject test for the extrapolation parameter has to judge the objective at the *feasible*
  iterate, but the Gram factors on hand belong to the extrapolated point, so making it honest costs the
  extra `O(Mdr)` product the acceleration was meant to save. The cheap exact convergence check that came
  out of the attempt was kept.

## [0.5.0] — 2026-07-18

### Added
- `method="gmm-toeplitz-gs"` — **full-order Gohberg-Semencul MLE** Toeplitz-precision GMM for ordered
  stationary signals: a Yule-Walker (Levinson) warm start at order ≤ 16, then coordinate ascent of the
  exact log-likelihood over the reflection coefficients (positive-definite by the `|k| < 1` constraint) —
  the likelihood-optimal general precision (arXiv:2311.14995), completing the three-rung Toeplitz ladder
  of [`docs/adr/001-gmm-toeplitz.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/adr/001-gmm-toeplitz.md). Reuses the exact GS precision E-step;
  BIC auto-`k`; a true `predict_proba`. Measured: competitive with the AR head on AR signals and recovers
  mid-lag echo structure (lags 11–16, beyond the banded cap `w_max = 10`) the banded head is blind to,
  while the dense-covariance `gmm-toeplitz-full` covers arbitrarily long lags (`bench/toeplitz_ar_mixture.py`).
- `projection="weighted-nmf-kl"` — **KL-divergence (I-divergence) CF-weighted NMF** for **count** data
  (word counts / event tallies / Poisson observations), where the Frobenius objective (Gaussian noise) is
  mis-specified. Lee-Seung multiplicative updates with the shared components weighted by leaf mass — the
  Poisson maximum-likelihood fit. The advantage is largest where counts are **sparse**: measured **up to
  +0.5 ARI over Frobenius** on a Poisson-count mixture at mean rate < 0.5 (0.83 vs 0.24), narrowing to a
  few points as counts grow and Poisson → Gaussian (`examples/17_nmf_topics.ipynb`).

### Notes
- NMF Phase-2 warm-start / randomized range-finder were assessed and **deferred** (see
  `plans/nmf-cf-weighted.md`): in the CF-compressed regime the factorization already runs over the
  `M ≪ N` leaves, so the compression — not these — is the speedup; they would add API/state for a marginal
  gain and are not shipped.

## [0.4.0] — 2026-07-18

### Added
- `projection="weighted-nmf"` (+ `projection_dim`) — **CF-weighted nonnegative matrix factorization** as
  a Phase-3 reducer for **nonnegative** data (TF-IDF / bag-of-words / event counts / spectrogram
  magnitudes / histograms). Rather than factorizing the raw `N×d` matrix (which defeats the compression),
  it factorizes the `M ≪ N` leaf **centroids** weighted by their mass, `X̃_j = √n_j·μ_j`: by
  König-Huygens the full-data NMF objective equals the weighted-centroid one up to the within-leaf
  scatter constant, so the expensive factorization runs over the microclusters — memory-bounded, `O(M·d·r)`
  — and any head (k-means / GMM / Leiden) then clusters the nonnegative codes. The solver is a
  dependency-free weighted **HALS** (coordinate descent, Gram-reuse across sweeps); the compression, not a
  fast NMF, is the speedup. Available on the one-shot `fit_predict` and the streaming `Betula` estimator.
  Signed input is rejected (no silent shifting — use `vmf` / `spherical-kmeans` or PCA / TruncatedSVD for
  embeddings); dense input in this release. See [`docs/MATH.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/MATH.md).

## [0.3.0] — 2026-07-18

### Added
- `method="gmm-toeplitz-full"` — **general (non-AR) positive-definite Toeplitz-covariance GMM** for
  ordered wide-sense-stationary signals whose autocovariance a low-order AR cannot capture (e.g. a
  long-lag echo / narrowband structure beyond the AR order). Each component covariance is the dense
  Toeplitz matrix built from the **biased (periodogram-consistent) autocovariance** — positive-
  semidefinite by construction, made strictly positive-definite by the within-leaf variance plus a
  small ridge — factored by Cholesky for an exact multivariate-Gaussian E-step. `O(d²)` parameters,
  `O(d³)` per component; BIC auto-`k` at `n_clusters=0`; a true posterior via `predict_proba`. This is
  the general (non-AR) rung of the Toeplitz ladder recorded in
  [`docs/adr/001-gmm-toeplitz.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/adr/001-gmm-toeplitz.md). On a long-lag-echo mixture (echo lag
  `K ∈ {16, 28, 40}`, all beyond the AR order) it recovers the components (ARI 0.70 → 0.97 as the window
  grows) where the banded `gmm-toeplitz` sits at chance; on AR-generated signals the two match
  (`bench/toeplitz_ar_mixture.py`).

### Changed
- `gmm-toeplitz`: raised the internal AR-order cap `w_max` 6 → 10. BIC still selects the smallest
  sufficient order (easy signals are bit-for-bit unchanged); higher-order / MA-like signals gain
  headroom before the general `gmm-toeplitz-full` head is needed.

## [0.2.0] — 2026-07-18

### Added
- `method="spherical-kmeans"` / `method="vmf"` — **directional clustering on the unit hypersphere**
  for L2-normalized embeddings (CLIP / face / sentence / speaker vectors), where cosine — not
  Euclidean — geometry is what matters. `spherical-kmeans` is hard cosine assignment
  (`argmax_c μ̂·μ_c`, centers re-normalized to the sphere); `vmf` is a soft **mixture of
  von Mises–Fisher** distributions (EM, a true posterior for `predict_proba`, and BIC auto-`k` when
  `n_clusters=0`). Both reduce each leaf to its weighted mean `(n_i, μ_i)`, so the cluster resultant
  `R_c = Σ n_i μ_i` stays **exactly mergeable** — the BETULA property carries through to the sphere —
  and the within-leaf spread `‖μ_i‖` feeds the concentration `κ` (Banerjee et al. 2005), so
  microcluster compression does not over-estimate it. The engine L2-normalizes input automatically
  for these methods (`get_params` stays verbatim). The `κ` normalizer uses a dependency-free,
  numerically stable `log I_ν(κ)` (log-space series) — no Bessel library, the crate stays NumPy-only.
  Available on the dense one-shot / streaming estimator, the sparse (`O(nnz)`) path, and as the
  `spherical_kmeans` / `movmf` / `movmf_auto` Rust functions.
- `covariance_weight` (`method="leiden"` / `"leiden-cpm"`, `feature="full"`) — **covariance-aware
  community detection**. `β > 0` adds a **log-Euclidean** shape term `β·‖logΣ_i − logΣ_j‖²_F` to the
  microcluster affinity graph, so two microclusters must be close in **both** centroid *and*
  covariance to be neighbours — useful when clusters differ by orientation / shape (covariance
  descriptors, motion / time-series windows, anisotropic blobs). `logΣ` is computed with the in-house
  Jacobi eigensolver (new `linalg::matrix_log`) — no new dependency; `β = 0` (default) is the
  existing centroid-only affinity, bit-for-bit unchanged.
- `tangent_weight` / `tangent_rank` (`method="leiden"` / `"leiden-cpm"`, `feature="full"`) —
  **GeoBETULA manifold-aware community detection**. `γ > 0` adds a **Grassmann** term
  `γ · d²_Gr(U_i, U_j)` (projection distance between each microcluster's rank-`tangent_rank` principal
  subspace) to the affinity, so communities must agree in centroid, covariance **and** manifold
  orientation — separating crossing / adjacent manifolds that share a centroid neighbourhood. Reuses
  the in-house Jacobi eigensolver (no new dependency); `γ = 0` (default) leaves the graph unchanged.
- `method="scale-space"` — **scale-space (Morse-persistence) density-mode clustering**. Treats the
  microclusters as a weighted point set and clusters the modes of the KDE
  `ρ_h(x) = Σ_j n_j exp(−‖x−μ_j‖²/2h²)`; it **sweeps the bandwidth `h` and keeps the labelling at the
  most persistent mode count** (the widest plateau of the modes-vs-`log h` curve), so it needs **no
  `k` and no bandwidth** and finds non-convex, arbitrary-count structure. A **prominence**-based mode
  merge (collapse peaks separated by only a shallow density valley) cleans the mode-count curve, so it
  is robust from 2 to ~8+ well-separated clusters and on unequal densities. Pure-Rust mean-shift over
  the `M ≪ N` leaves — cost bounded by the leaf budget, not `N`.
- `method="gmm-toeplitz"` — **AR / Toeplitz-structured GMM for ordered, wide-sense-stationary
  signals** (fixed-length time-series windows, trajectories, sensor / audio / vibration waveforms).
  Each component's covariance is an **AR(w)** process: the pooled **unbiased (covariance-method)**
  autocovariance is mapped by
  **Levinson-Durbin** to the exact **Gohberg-Semencul** precision `Γ = (1/σ²)(BBᵀ − ZZᵀ)`, evaluated by
  the prediction-error decomposition so the `w` boundary positions are modelled exactly — **positive-
  definite by construction** (the reflection-coefficient clamp is the GS box constraint), `O(w)`
  parameters, order `w` chosen by BIC — so it stays well-posed in the
  `N_k ≪ d` regime where full covariance is singular and a diagonal model is blind to neighbour
  correlation. Reuses the CF scatter (no new tree machinery); a scalar stationary mean; BIC auto-`k`
  at `n_clusters=0`; a true posterior via `predict_proba`; parallel EM restarts (`parallel` feature).
  **For ordered coordinates only** — on generic embeddings the Toeplitz prior is
  wrong (use `gmm` / `gmm-full`). Based on the Gohberg-Semencul Toeplitz-precision estimator of
  arXiv:2311.14995; design and validation in [`docs/adr/001-gmm-toeplitz.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/adr/001-gmm-toeplitz.md).

### Fixed
- **High-dimensional GMM regularization** — the expected-log E-step adds a within-leaf correction
  (`−½ Σ_d (Σ_i)_dd/σ²_kd` for diagonal, `−½ tr(Σ_k⁻¹ Σ_i)` for full covariance) that turns
  *over-confident* when a component's own covariance goes near-singular along a low-variance
  direction — which is the norm in high dimensions with few effective microclusters per component.
  Two floors now keep the component covariances well-conditioned:
  - `method="gmm"` (diagonal): per-dimension variance floor raised from `1e-6·gvar_d` to
    `1e-3·gvar_d`. `digits` (64-D) ARI 0.372 → 0.396, now ahead of scikit-learn's
    `GaussianMixture(covariance_type="diag")` (0.324).
  - `method="gmm-full"` (full covariance): added a per-dimension floor on each component's covariance
    **diagonal** at `1e-3·gcov_dd` (off-diagonals — orientation — untouched). Previously a component
    could be starved to zero responsibility and the recovered count dropped below `k`; on `digits`
    the fit collapsed to 9 clusters at ARI 0.391, and now holds all 10 at ARI 0.511 — ahead of
    scikit-learn's `GaussianMixture(covariance_type="full")` (0.402).

  The floors are relative to the **per-dimension** global variance (not the global mean scale, which
  is inflated by between-cluster separation and would over-regularize tight clusters), so
  low-dimensional and anisotropic fits are unchanged (well-separated blobs still ARI 1.00; the
  rotated-anisotropic 2-D case still ties `GaussianMixture` at 0.887). No API change.

## [0.1.5] — 2026-07-04

### Added
- `method="leiden"` / `method="leiden-cpm"` — **graph clustering / community detection** over the
  microcluster affinity graph via the full **Leiden** algorithm (Traag, Waltman & van Eck 2019):
  local moving → refinement (sub-communities grown from singletons *along edges*, so each is
  connected by construction — Leiden's guarantee over Louvain) → aggregation seeded from the
  pre-refinement partition. It **discovers the community count** — no `k` (like the density head).
  A `resolution` (`γ`) knob trades community count against size; the **modularity** objective
  (`"leiden"`, γ = 1 default) has a resolution limit, the **CPM** objective (`"leiden-cpm"`) is
  resolution-limit-free (γ on a smaller, density scale). Pure Rust — no eigensolver, NumPy-only.
  Best for community/blob structure at a moderate `threshold`; use `method="spectral"` for elongated
  manifolds. The self-tuning k-NN affinity graph is shared between the spectral and Leiden heads.
- `betula_cluster.consensus(X, n_clusters, n_runs=…)` — clusters `X` under several random
  insertion-order permutations and votes, turning the CF-tree's **insertion-order sensitivity**
  (Known Limitation #1) into a measurable quantity: a consensus labelling plus a **per-point
  stability score** in `[0, 1]` (`ConsensusResult.confidence` — low on unstable boundaries, high
  where every order agrees). NumPy-only; for the partitional heads at a fixed `n_clusters`. The
  independent runs parallelize across threads with `n_jobs` (the Rust core releases the GIL).
- `method="spectral"` — spectral clustering over the CF-tree leaf microclusters for **non-convex /
  manifold** clusters (moons, rings, spirals) that the centroid heads cannot separate. Self-tuning
  symmetric k-NN affinity (Zelnik-Manor & Perona local scaling), the normalized Laplacian embedding
  (Ng-Jordan-Weiss) via the in-house Jacobi eigensolver — no LAPACK/ARPACK, the crate stays
  NumPy-only — with a k-means landmark reduction above 256 microclusters so the `O(m³)` solve stays
  bounded. Dense input only; pair it with a small `threshold` (many leaves) so the microclusters
  resolve the manifold. No built-in cluster-count selection: `n_clusters=0` defaults to 2.
- `threshold="auto"` for the `Betula` estimator — removes the one hyperparameter users most often
  have to guess. A bounded-subsample pilot fits a `threshold=0` tree at the same `max_leaves` and
  reads the threshold it converges to, warm-starting the full fit near-converged instead of growing
  it from zero (fewer rebuild passes, lower peak leaf count on large `n`). Cached across refits /
  streaming batches; below the pilot cap it is a no-op (growing from zero is already cheap), and it
  is dense-only (raises on sparse input).

### Changed
- Benchmarks now cover every head (spectral, Leiden added to `bench/comprehensive.py`) and the
  compression heads run at `max_leaves = 4000`: betula-kmeans is at *exact* parity with scikit-learn
  (blobs 0.861 = 0.861) and Ward beats raw Ward while running the full `N`. Docs / README / docs site
  surface the spectral, Leiden and consensus additions; test counts reconciled (190 Python, 158
  Rust). The docs site now renders the CHANGELOG and redeploys on every published release.

## [0.1.4] — 2026-07-04

### Added
- `MapperGraph.persistence_diagram` / `MapperGraph.persistence(filtration=…)` — 0-D persistent homology
  of the Mapper nerve by single-linkage union-find (elder rule, `O(E log E)`, pure Rust). Two
  filtrations: `"overlap"` (the `1 − edge_overlap` Bhattacharyya gap — a finite bar's death is the depth
  of a bottleneck, ranking the boolean `bridges`) and `"lens"` (the lens sublevel diagram). Essential
  connected-component classes carry `inf` death.
- Greedy weighted k-means++ init (scikit-learn's default): lower-inertia, lower-variance seeds at
  ~`ln k`× the negligible init cost over the leaves.
- `objective="dbcv"` for `tune` — Density-Based Clustering Validation (Moulavi et al. 2014, in
  `[-1, 1]`). Unlike the convex Calinski-Harabasz / Davies-Bouldin metrics (which *penalise* correct
  non-convex partitions), DBCV validates variable-density / non-convex clusters, so it is the right
  selection metric for the HDBSCAN-CF and DbStream density heads. NumPy-only, computed over a
  subsample.

### Changed
- `fit_predict_sparse` / the `_core` CSR entry points now cap `n_features` (`MAX_SPARSE_FEATURES`) and
  validate CSR arrays through the pure-Rust `sparse::validate_csr`, closing an unbounded-allocation DoS
  where a hostile caller could force an ~8 EB allocation with a single-nonzero row.
- Docs reconciled to the current suite: **172**-case Python suite, **147** Rust tests (143 unit + 4
  integration under default features; the `python` / `persistence` / `cli` surfaces add more, 155 total).

### Tests
- Mutation-testing infrastructure (`cargo-mutants` scoped to the CF math core, `mutmut` for the Python
  wrapper, a weekly non-blocking workflow) plus a CSR-fuzzing proptest and the two coverage gaps it
  surfaced (the CF-tree absorption boundary, exact tune-metric values).

## [0.1.3] — 2026-07-04

### Added
- `betula_cluster.tune` — memory-aware hyperparameter search over the CF knobs, scored by an internal
  metric (Calinski-Harabasz / Davies-Bouldin) or ARI, with a multi-objective **quality / memory /
  speed** Pareto mode. NumPy-only by default; an optional Optuna backend (TPE / NSGA-II) via
  `pip install 'betula-cluster[tune]'`.
- Property-based tests (`proptest`, dev-only) for the CF-tree invariants: the clustering feature is a
  commutative monoid (`merge` is associative/commutative and equals a sequential build), folding a
  tree's leaf features reconstructs the whole-dataset feature, the full-covariance upper-triangular
  index is a bijection (incl. `dim ≥ 4`), and the Frequent-Directions sketch is lossless on low-rank
  data and never overshoots the exact scatter.
- Sparse-text benchmark (20 newsgroups, TF-IDF): the `O(nnz)` `fit_predict_sparse` path and the
  standard reduce-then-cluster pipeline (TruncatedSVD / NMF → k-means) vs scikit-learn, written up
  honestly in `bench/RESULTS.md` (raw high-`d` TF-IDF concentrates for every fast clusterer; on NMF
  topics betula matches sklearn).
- `MapperGraph.edge_overlap` — a Bhattacharyya coefficient in `(0, 1]` per Mapper edge, from the pooled
  diagonal-Gaussian summaries of the two nodes' member microclusters. Surfaced on `to_networkx()` edges
  as `overlap=…`, so a bridge between well-separated regions reads as a lower-weight edge than one
  inside a dense blob.
- Documentation site (MkDocs Material + `mkdocstrings` API autodoc, MathJax-rendered math) built from
  `docs/`, with a GitHub Pages deploy workflow; `pip install 'betula-cluster[docs]'` for the toolchain.

### Changed
- Coverage floor (`cargo llvm-cov`, ≥95 % lines) now also measures the `persistence` and `cli` feature
  sets, not just the default core.
- Declared `rust-version = "1.82"` (MSRV) and lowered the real floor to it — the streaming heads had an
  implicit 1.87 dependency (`u64::is_multiple_of`), now rewritten. Added `Documentation` / `Changelog`
  project URLs.
- Docs reconciled to the current suite: **167**-case Python suite, **141** Rust tests (137 unit + 4
  integration), and **five** end-to-end use cases (README, DESIGN.md).
- Repository hardening: `macOS` / `Windows` CI test legs, an sdist install smoke test, a nightly
  `cargo audit` cron, Dependabot, and `SECURITY.md` / `CONTRIBUTING.md` / issue templates.

## [0.1.2] — 2026-06-28

### Added
- `betula_cluster.__version__`, resolved from the installed package metadata.

### Changed
- README repositioned: compress-then-cluster framing, the test/coverage story surfaced at the top, a
  "When to use it" section, and a **stable-core / experimental** capability split. HDBSCAN is labelled
  **HDBSCAN-CF** consistently in prose (the `method="hdbscan"` API string is unchanged).

### Fixed
- Stale docs: the Python suite is **153** cases (was written as 123); `betula-index` references now
  point to `lexindex` (the indexing companion's published name).

## [0.1.1] — 2026-06-28

### Fixed
- PyPI project description: README links to the docs, benchmarks, and examples are now absolute GitHub
  URLs so they resolve on the PyPI page (relative links only worked in the GitHub-rendered README).

## [0.1.0] — 2026-06-28

First public release.

### Added
- Numerically stable BETULA clustering features `(n, μ, S)` (Welford/Chan updates) with four
  covariance models: spherical, diagonal, full (PSD via Cholesky), and a Frequent-Directions sketch
  (`O(ℓ·d)` per leaf) for very high-dimensional data.
- Memory-bounded CF-tree (Phase 1) with auto-rebuild under a `max_leaves` cap; optional parallel
  shard+merge build (`n_jobs`); EWMA `decay` for streaming concept drift.
- Global clustering heads: Hamerly-accelerated exact k-means, diagonal & full-covariance GMM-EM
  (expected-log E-step + NIW/MAP), Ward-HAC (nearest-neighbour chain), and HDBSCAN-on-CF; automatic
  cluster count at `n_clusters=0` (BIC / X-means / dendrogram cut).
- χ² / Mahalanobis mass-invariant absorption gate (`absorb="chi2"`).
- `normalize=True` for cosine/direction clustering of embeddings (L2-normalized rows on the unit
  sphere; squared-Euclidean is monotone in cosine). Doubles as the **high-dimensional fix**: at d≫100
  raw Euclidean distances concentrate and the CF-tree collapses, but direction stays discriminative —
  on MNIST-784 it lifts ARI 0.04 → 0.44, beating scikit-learn (benchmarked in
  `bench/results_real_normalize.csv`). Off by default (magnitude is signal on tabular data).
- Inline auto-vectorized distance kernels (the compiler vectorizes the tight reductions per call
  site; `target-cpu=native` opts into AVX2 / AVX-512 — see `.cargo/config.toml`); rayon-parallel
  labeling.
- Python bindings: abi3 wheel (CPython 3.11+), zero-copy NumPy, `float32`/`float64` (no upcast), GIL
  released during compute; one-shot `fit_predict` and a scikit-learn-style streaming `Betula`
  estimator (`partial_fit` / `fit` / `predict` / `fit_predict`).
- Full scikit-learn parameter protocol (`get_params` / `set_params`) — works with `clone`,
  `Pipeline`, and `GridSearchCV`. PEP 561 typed (`py.typed` + stubs).
- Dataset-structure inspection: `microcluster_centers_`/`_weights_`/`_radii_`,
  `cluster_centers_`/`_radii_`/`_sizes_`, `outlier_scores`, `find_outliers`, `find_near_duplicates`,
  `near_duplicate_pairs` (scored cosine pairs, exact within each leaf-block — the scalable
  counterpart to an O(N²) all-pairs scan), `sample_representatives`, `assign_microclusters`,
  `summary`, and `n_rebuilds_` / `threshold_` diagnostics.
- **Mapper topological skeleton** (`topology::mapper` → `Betula.mapper()` → `MapperGraph`): a lens
  (`density` / `radius` / `l2norm` / `coordinate` / `eccentricity`) over the microclusters, an
  overlapping cover, per-bin single-linkage at a data-adaptive (median-NN) scale, and a nerve graph with branch
  points and bridges (Tarjan); optional `to_networkx()`. Exploration of structure / RAG leakage /
  dedup, not a partition. `mapper_stability()` sweeps the resolution and reports the topology's
  persistence across scale (β₀ components, β₁ loops, branch points, bridges per resolution).
- **Soft assignment & confidence**: `predict_proba` (true posterior for the GMM heads via the
  per-leaf responsibility matrix `microcluster_proba_`; a documented centroid-distance softmax
  *heuristic* for k-means / Ward / HDBSCAN) and `assignment_confidence`.
- **Coreset / diagnostics**: `export_coreset()` → `Coreset` (leaves as weighted points — a streaming
  coreset), `diagnostics()` (compression ratio, radius percentiles, cluster mass spread),
  `representatives(method=medoid|boundary|outlier|diverse)`, and `cluster_profile()` (JSON-able
  geometry for LLM cluster naming).
- **`memory_budget_mb`**: size `max_leaves` from a target tree-resident memory (MiB) at fit time
  instead of tuning it by hand; the resolved value is exposed as `effective_max_leaves_`.
- **Drift monitoring & curation**: `snapshot()` + `Betula.compare_snapshots(before, after)`
  (nearest-centroid match → centroid shifts / mass ratios) and `active_learning_batch(strategy=
  "uncertain"|"outlier")` (rows to review/label).
- **`DenStream`** streaming density clusterer (Cao et al., SDM 2006) over fading spherical
  micro-clusters built on the stable CFs (decay is centroid/radius-invariant); `partial_fit` /
  `cluster` / `fit` / `fit_predict` / `predict` (`-1` = noise) + microcluster getters, sklearn-style.
- **`DbStream`** streaming DBSTREAM clusterer (Hahsler & Bolaños, 2016): fading micro-clusters
  connected by **shared density** (faded overlap mass) rather than distance, so it recovers
  arbitrarily-shaped clusters and keeps close-but-disconnected dense regions apart. Fixed-radius
  multi-assignment online; offline connects a pair when their overlap mass is `≥ alpha·min_weight`.
  Same fading-CF core and sklearn-style API as `DenStream`; `core::stream::DbStream` in Rust.
- **Streaming quantile sketches** (`betula-sketch`, in `src/sketch/`): `KllSketch` (Karnin–Lang–
  Liberty, rank-error) and `DdSketch` (Masson et al., relative-error) — `update` / `update_many` /
  `merge` / `quantile` / `quantiles`, mergeable, bounded memory.
- **Sparse input**: `fit` / `fit_predict` / `partial_fit` / `predict` accept a `scipy.sparse` matrix
  (CSR-routed, rows expanded one at a time — the dense `N × d` matrix is never materialized). f64;
  this dense-tree path keeps the cancellation-free guarantee, compute `O(N·d)`.
- **`O(nnz)` sparse-native** (`fit_predict_sparse`): one-shot clustering of a `scipy.sparse` matrix
  that touches only the non-zeros. Rows summarize into spherical micro-clusters keeping
  `(n, ΣX, ‖ΣX‖², S)` (so the mean, cached `‖μ‖²`, and centroid distance are `O(nnz)`) via a flat
  leader pass bounded by `max_leaves`, then a parametric head (`kmeans` default — robust for
  high-`d` sparse) labels each row. Uses the *expanded* squared-distance form, so unlike the dense
  path it is not cancellation-free (accurate for sparse rows far from the dense centroid);
  `core::sparse::{summarize_sparse, nearest_sparse}` is the Rust API.
- **Robust insertion** (`huber_k`): optional Huber/winsorized point updates on the streaming
  estimator — each point is clamped to within `huber_k` per-dimension standard deviations of its
  target microcluster before the Welford fold-in, bounding any single point's pull on the centroid
  (`O(k·σ/n)`) so stream outliers cannot stretch a centroid or inflate a radius. Off by default;
  zero-variance dimensions pass through and a 5-point warm-up gates the clip. The result is still a
  valid `(n, μ, S)` triple, so every downstream head is unchanged.
- **Constrained clustering** (`must_link` / `cannot_link`): semi-supervised COP-KMeans (Wagstaff et
  al., 2001) over the leaf microclusters — `fit(X, must_link=..., cannot_link=...)` /
  `fit_predict(...)` take `(m, 2)` row-index pairs. Must-link is transitively closed; cannot-link is
  enforced per assignment. Constraints are honoured at the microcluster granularity, so a cannot-link
  inside one leaf (or contradictory / over-constrained inputs) raises `ValueError` rather than being
  silently dropped. `method="kmeans"`, dense input; `core::clustering::cop_kmeans` exposes the Rust
  API with a typed `ConstraintError`.
- **Mixed numeric + categorical clustering** (`KPrototypes`): k-prototypes (Huang, 1997) for mixed
  data. A *mixed CF* (`MixedCf`) pairs the stable numeric `(n, μ, S)` with a per-attribute category
  histogram (mode = categorical centroid); distance is `‖Δnumeric‖² + γ·(categorical mismatch)`, with
  `γ` defaulting to Huang's heuristic. Rows are leader-summarized into bounded mixed micro-clusters,
  then clustered. Standalone scikit-learn-style estimator (`categorical` column indices,
  `fit`/`fit_predict`/`predict`, `cluster_centroids_`/`cluster_modes_`); `core::clustering::{MixedCf,
  kprototypes, summarize_mixed}` is the Rust API.
- **Command-line interface** (`betula`, behind the `cli` feature): a dependency-free binary that
  clusters a delimited numeric file or stdin and writes one label per row; flags mirror the library
  (`--clusters` / `--method` / `--feature` / `--threshold` / … ; `--clusters 0` auto-selects `k`).
- `save` / `load` + pickle (`joblib`-compatible) persistence (serde + CBOR via ciborium,
  schema-versioned).
- NaN/Inf input validation at the boundary.

### Fixed
- `estimate_threshold` now measures the mean nearest-sibling distance **within each leaf node**
  (ELKI/BETULA-standard, `O(M·capacity)`) instead of a global all-pairs scan; the rebuild threshold
  rises monotonically (no multiplicative bump that compounded across rebuilds and collapsed the tree
  far below `max_leaves`), and rebuilds reinsert in reverse-DFS leaf order. The CF-tree build is now
  byte-for-byte the reference (`betulars`) tree shape and at speed parity with matched build flags.

[Unreleased]: https://github.com/ilgrad/betula-cluster/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/ilgrad/betula-cluster/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ilgrad/betula-cluster/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ilgrad/betula-cluster/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ilgrad/betula-cluster/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ilgrad/betula-cluster/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ilgrad/betula-cluster/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.5
[0.1.4]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.4
[0.1.3]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.3
[0.1.2]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.2
[0.1.1]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.1
[0.1.0]: https://github.com/ilgrad/betula-cluster/releases/tag/v0.1.0
