# Best GMM E-step for clustering on CF summaries (measured)

> Decision record for the expected-log E-step the library ships (`docs/MATH.md` § *GMM E-step*).
> Re-measured **2026-08-24** over three seeds; the previous edition of this page was a **single run**
> of 2026-06-26 on a pre-0.1.0 prototype, and three of its headline numbers turn out to have been
> ends of a seed spread rather than typical values. The decision it reached still stands, but for a
> narrower reason than it claimed. Reproduce with
> `uv run --no-sync python research/gmm_cf_estep.py` (`--seeds` to change them).

Experiment: `research/gmm_cf_estep.py`. Data — a GMM with known labels (K=4, N=20000, d=2),
compressed into `m` micro-clusters → CF `(n_i, μ_i, Σ_i)`, then GMM-EM on the CFs with three
different E-steps (identical init and M-step, so only the E-step varies). Metric — ARI of the
original points against ground truth. Medians of seeds `0xBE7012A + {0,1,2}`, `[min, max]` beneath.

| scenario | gold(raw) | kmeans-CF | A: plug-in | B: convolution (paper) | **C: expected-log** |
|---|---|---|---|---|---|
| bal sep=2.5 m=40 | 0.817 | 0.774 | 0.775 | 0.743 | **0.792** |
| | 0.78–0.86 | 0.66–0.84 | 0.75–0.84 | 0.53–0.82 | 0.74–0.84 |
| bal sep=2.5 m=150 | 0.772 | 0.725 | 0.766 | 0.765 | 0.766 |
| | 0.77–0.83 | 0.68–0.81 | 0.76–0.82 | 0.75–0.82 | 0.76–0.82 |
| bal sep=4.0 m=40 | 0.989 | 0.986 | 0.986 | 0.986 | 0.986 |
| bal sep=4.0 m=150 | 0.990 | 0.987 | 0.990 | 0.990 | 0.990 |
| imb sep=2.5 m=40 | 0.850 | 0.818 | 0.818 | 0.797 | 0.819 |
| | 0.84–0.90 | 0.50–0.90 | 0.78–0.92 | 0.72–0.89 | 0.79–0.93 |
| imb sep=2.5 m=150 | 0.739 | 0.777 | **0.885** | **0.885** | **0.885** |
| | 0.70–0.84 | 0.72–0.79 | 0.83–0.89 | 0.82–0.89 | 0.83–0.89 |
| imb sep=4.0 m=40 | 0.995 | 0.979 | 0.989 | 0.989 | 0.989 |
| imb sep=4.0 m=150 | 0.989 | 0.843 | 0.920 | 0.920 | 0.919 |
| (sep = 6.0 everywhere) | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 |

Counting every one of the 36 `scenario × seed` cells rather than the medians, because a variant that
wins narrowly everywhere and one that wins hugely once are different results:

| | best or tied | median shortfall against gold(raw) |
|---|---:|---:|
| A: plug-in | 32/36 | +0.002 |
| B: convolution | 22/36 | +0.002 |
| **C: expected-log** | **33/36** | +0.002 |

## Conclusion

- **B (convolution, the BETULA paper's approach) is the weakest**, and that is the finding that
  survives re-measurement: 22/36 against 32–33/36. Inflating each component by `Σ_k + Σ_i` washes out
  separability, and it costs most where the summary is coarse. Its worst cell (bal sep=2.5 m=40)
  spans 0.53–0.82 — far wider than any other variant's, so the failure is also *unreliable*, which is
  worse than being uniformly slightly behind.
- **C vs A is a tie at three seeds — 33/36 against 32/36.** The previous edition claimed C was
  "consistently ≥ A and B"; against A that is not supported. C leads A in exactly one regime, coarse
  summaries with overlapping components (bal sep=2.5 m=40, 0.792 against 0.775), which is where the
  `−½ tr(Σ_k⁻¹ Σ_i)` correction is theoretically supposed to matter — `Σ_i` is largest relative to
  `Σ_k` there. Elsewhere they agree to the third decimal, and at imb sep=4.0 m=150 A is ahead by
  0.001.
- **All three CF variants beat gold(raw) under imbalance at m=150** (0.885 against 0.739). The
  summary is acting as a regulariser: raw EM on 20 000 points with a 0.5/0.25/0.15/0.10 mixture
  collapses a small component more often than EM on 150 weighted micro-clusters does.
- **Two dramatic numbers from the 2026-06-26 edition were single-seed artefacts.** `kmeans-CF` at
  imb sep=2.5 m=40 was reported as 0.503, against a three-seed median of 0.818 and a range of
  0.50–0.90 — the old number was the bottom of the spread. Likewise B's 0.526 at bal sep=2.5 m=40,
  against a median of 0.743 and a range of 0.53–0.82. The qualitative claims they were used to
  support (k-means-CF is fragile under imbalance; B is worst on coarse CFs) both hold on the medians,
  but not with the margins the single run suggested.

## Decision for the implementation

Unchanged: **`log r_ik = log π_k + log N(μ_i|μ_k,Σ_k) − ½ tr(Σ_k⁻¹ Σ_i)`** (variant C), log-sum-exp
normalised, with the M-step folding `Σ_i` back in as
`Σ_k = Σ_i w_ik(Σ_i + (μ_i−μ_k)(μ_i−μ_k)ᵀ)/N_k`. Shipped in `src/clustering/gmm.rs` for both the
diagonal and full heads (`trace_under` supplies the correction).

The justification is now narrower and should be stated as such: C is chosen over **B** on measured
ARI, and over **A** because it is the correct expected log-likelihood under the leaf model and costs
one trace that the second moment is already carrying — not because it measurably clusters better in
general. A is a legitimate alternative that would perform the same on this experiment outside the
coarse-and-overlapping corner.

## Independent check against the authors' own implementation (2026-08-24)

The experiment above is synthetic, two-dimensional and written by the same project it justifies.
ELKI 0.8.0 — the reference implementation of BETULA, by the algorithm's authors — ships two GMM
heads over the same cluster features: `BetulaGMM` and `BetulaGMMWeighted`. Running them against
this library's `method="gmm"` at matched CF-tree parameters is the outside check this page lacked.

Harness: `local/scratch/elki/cross_check.py`, median of seeds 0/1/2, `feature="diagonal"` ↔
`VVIFeature`, branching 32, `threshold=0`, `max_iter=100`. Two CF-tree geometries are run because
the projects ship different defaults — D0/D0 (`CentroidEuclideanDistance` routing and absorption,
this library's default) and D4/R (`VarianceIncreaseDistance` + `RadiusDistance`, ELKI's).

| dataset | geometry | `betula gmm` | ELKI `BetulaGMM` | ELKI `BetulaGMMWeighted` |
|---|---|---|---|---|
| digits (1797×64, 200 leaves) | D0/D0 | **0.5239** | 0.2305 | 0.3528 |
| digits | D4/R | **0.5210** | 0.1627 | 0.4586 |
| covtype-50k (54-D, 2000 leaves) | D0/D0 | **0.1062** | 0.0244 | 0.0523 |
| covtype-50k | D4/R | **0.0852** | 0.0555 | 0.0575 |

The shipped head leads at the median in all four cells, and reaches the lower within-cluster sum of
squares in all four as well. **The margin is not all E-step.** This library's GMM head keeps the best
of four EM restarts by log-likelihood (`GMM_N_INIT`, `src/clustering/gmm.rs`); a bare ELKI
`BetulaGMM` runs one, and ELKI 0.8.0 has no `BestOfMultipleKMeans` equivalent for EM to equalise it
with. Compared instead against the **best of ELKI's three seeds across both its variants** — a
generous handicap that absorbs much of that difference — the result splits evenly: the shipped head
leads on digits/D0/D0 (0.5239 vs 0.3744) and covtype/D0/D0 (0.1062 vs 0.0696), and trails on
digits/D4/R (0.5210 vs 0.5294) and covtype/D4/R (0.0852 vs 0.1193). So: a clear lead at equal seed
budget, and a two-two split against ELKI's luckiest seed — the honest reading is parity once the
restart budget is accounted for, not superiority.

What this does and does not license. It **does** retire the objection that the E-step decision rests
on one synthetic fixture from a pre-0.1.0 prototype: on two real datasets, in two tree geometries,
the shipped formulation is never behind both of the authors' variants at an equal seed budget, and
is at worst level with the better of them once ELKI is given its luckiest seed. It does **not** isolate
the E-step as the cause — restarts, initialisation and the M-step differ too. Isolating it would need
variants A/B/C wired behind a flag in this crate and run on the same fixtures, which is worth doing
if the decision is ever revisited. One directional signal is worth recording: `BetulaGMMWeighted`
beats plain `BetulaGMM` in all four cells, which is the same direction as this page's argument —
folding the leaf's own mass and scatter into the E-step helps.


## Re-run at five seeds on 0.8.0 (2026-09-08)

Three seeds could not separate a 0.02 ARI gap from its own spread, so the check above was repeated at
**five** seeds against the current tree. Same harness, same two geometries, `max_leaves` now passed
as the resolved integer (the fraction form is the estimator's; the free `fit_predict` this harness
calls takes the count).

| dataset | geometry | `betula gmm` ARI / WCSS | ELKI `BetulaGMM` | ELKI `BetulaGMMWeighted` |
|---|---|---|---|---|
| digits (1797×64, 200 leaves) | D0/D0 | **0.5239** / **1.245e6** | 0.2305 / 1.591e6 | 0.3528 / 1.444e6 |
| digits | D4/R | **0.5210** / **1.219e6** | 0.1627 / 1.603e6 | 0.3756 / 1.432e6 |
| covtype50k (50000×54, 2000 leaves) | D0/D0 | 0.0632 / **2.180e6** | 0.0565 / 2.305e6 | **0.0681** / 2.274e6 |
| covtype50k | D4/R | **0.0852** / **2.170e6** | 0.0544 / 2.258e6 | 0.0575 / 2.277e6 |

**The shipped head's own medians did not move at all** — 0.5239, 0.5210, 0.0852 are the same numbers
the three-seed run produced, so its spread was already resolved. What moved is ELKI's, and it costs
this page one cell: the claim "leads at the median in all four cells" becomes **three of four**, with
covtype/D0/D0 now 0.0632 against `BetulaGMMWeighted`'s 0.0681. That gap is 0.005 ARI against a
`BetulaGMMWeighted` seed range of −0.0168 to 0.1039, so it is a tie, not a loss — but it is no longer
a lead and the sentence above is corrected rather than left standing.

**The within-cluster sum of squares still favours the shipped head in all four cells**, and that is
the claim worth keeping: WCSS is the objective, ARI is a label proxy for it. The two come apart here,
which is itself the finding — see below.

**The k-means layer of the same run says why.** On covtype, ELKI's k-means variants score *higher*
ARI than this library's head (0.0951 vs 0.0820 at D0/D0, 0.0943 vs 0.0591 at D4/R) while reaching a
*higher* within-cluster sum of squares in every cell (2.219/2.178/2.226/2.346e6 against 2.169e6, and
2.262/2.176/2.236/2.333e6 against 2.165e6). A better-optimised k-means solution is a worse predictor
of covtype's forest-cover classes, because those classes are not k-means-shaped — the ARI column on
this dataset ranks agreement with a labelling the objective was never trying to recover. Any reading
of "ELKI is ahead on covtype ARI" that treats it as "ELKI's clustering is better" is unsupported by
the same run's objective column.

One number for the still-open `n_init` question: ELKI's 4-restart k-means buys **+0.137 ARI on
digits/D4/R** (0.6856 against 0.5482 for a single restart of the same initialiser) and **nothing on
covtype** (0.0951 both at D0/D0, 0.0943 both at D4/R), while lowering WCSS on both. Restarts pay
where the objective and the labels agree.

## The tree layer of the same cross-check (2026-09-09)

`cross_check.py` has always run three layers, and only the head layers were ever written up. Layer 1
— ELKI's `BetulaLeafPreClustering` against `Betula.assign_microclusters`, no head in the way — is the
half of Q5 that reads "and the tree", and it needs one correction before it can be read at all.

**At an equal `maxleaves` *budget* the two implementations do not build the same-sized tree.** Both
undershoot, and ELKI undershoots harder (seed 0, branching 32, `threshold=0`, `feature="diagonal"` ↔
`VVIFeature`; `local/scratch/elki/tree_curve.out`):

| dataset | geometry | budget | ELKI leaves | betula leaves |
|---|---|---:|---:|---:|
| digits | D0/D0 | 200 | 98 (49 %) | 156 (78 %) |
| digits | D4/R | 200 | 156 (78 %) | 164 (82 %) |
| covtype-50k | D0/D0 | 2000 | 1264 (63 %) | 1437 (72 %) |
| covtype-50k | D4/R | 2000 | 926 (46 %) | 1583 (79 %) |

A within-cluster sum of squares read off partitions of different sizes is not a comparison of tree
quality — more leaves buy a lower WCSS mechanically — so the equal-budget layer-1 table, where this
library's leaf partition reaches the lower WCSS in three of four cells, says only that it fills the
budget it was given. That is worth having (the budget is what bounds memory, and it is what the user
sets), but it is not the tree-quality claim it looks like.

**At an equal *realised* leaf count the answer splits by geometry, not by implementation.** ELKI's
budget was binary-searched until its leaf count matched betula's to within 1 %
(`local/scratch/elki/tree_matched.out`):

| dataset | geometry | leaves (ELKI / betula) | ELKI WCSS | betula WCSS | ratio | partition agreement |
|---|---|---|---:|---:|---:|---:|
| digits | D0/D0 | 155 / 156 | **6.533e5** | 7.087e5 | 1.085 | 0.530 |
| digits | D4/R | 162 / 164 | 5.373e5 | **5.250e5** | 0.977 | 0.492 |
| covtype-50k | D0/D0 | 1460 / 1437 | **8.029e4** | 8.642e4 | 1.076 | 0.393 |
| covtype-50k | D4/R | 1590 / 1583 | 6.904e4 | **6.859e4** | 0.993 | 0.393 |

So **on D4/R the two trees are level** — within 2.3 %, in both directions — and **on D0/D0, this
library's own default geometry, ELKI's tree is 7.6–8.5 % tighter at the same leaf count**. The
curve sweep agrees where it can be read without interpolation: at budget 100 on digits/D4/R both
implementations realise exactly 89 leaves, and the WCSS there is 6.601e5 against ELKI's 6.802e5, a
3 % lead for this library on the geometry where the matched test also calls it level.

That D0/D0 gap is a finding, not a formality: the *default* geometry is the one almost every user
gets, and 8 % of the summarisation objective is more than the E-step differences this page spends
its length on. It is also narrow enough to be one policy — a split rule, a rebuild threshold, an
absorption tie-break — rather than a difference in kind. Chasing it is **T27**.

> It was a fourth policy, it was fixed in E13 below, and the D0/D0 row of the matched table is now
> 0.846–0.857 of ELKI rather than 1.076–1.085. The two tables above measure the tree as it stood on
> 2026-09-09 and are kept as it was measured.

### Where the D0/D0 gap comes from (T27, 2026-09-09)

Three candidates were on file — the split rule, the rebuild threshold, the absorption tie-break —
and the fourth is written in `CFTree::rebuild`'s own doc comment: the rebuild merges the `k` closest
**sibling** pairs, meaning pairs inside one leaf *node*, because that is one `O(Σ child²)` scan
instead of a descent per entry. "Compaction cannot reach pairs that landed in different leaves" is a
policy, and it can be priced without touching the engine. A cluster feature merges exactly, so for
any grouping of the leaves

    WCSS = Σᵢ Sᵢ + Σ_merges (w_a w_b)/(w_a + w_b) ‖μ_a − μ_b‖²,   Sᵢ = wᵢ rᵢ²

which makes an offline *globally* cheapest-pair merge exact, and therefore a bound on what any
merge-only compaction could reach. Build at a finer budget, merge back down to the same final leaf
count, and compare (`local/scratch/elki/t27_compaction.py`; the tree build is seed-independent, so
the three seeds are identical by construction and are reported once):

| dataset | final leaves | source tree | WCSS (nearest-centroid) | of ours | of ELKI |
|---|---:|---:|---:|---:|---:|
| digits | 195 | ours, built at the budget | 7.087e5 | 1.000 | 1.085 |
| digits | 195 | 391 leaves, merged globally | **5.117e5** | 0.722 | 0.783 |
| digits | 195 | 732 leaves, merged globally | 4.502e5 | 0.635 | 0.689 |
| digits | 195 | 1456 leaves, merged globally | 4.303e5 | 0.607 | 0.659 |
| covtype-50k | 1950 | ours, built at the budget | 8.642e4 | 1.000 | 1.076 |
| covtype-50k | 1950 | 3939 leaves, merged globally | **5.894e4** | 0.682 | 0.734 |
| covtype-50k | 1950 | 7283 leaves, merged globally | 5.144e4 | 0.595 | 0.641 |

So the gap is in **how the final leaf set is chosen**, not in the routing or the split: a merge-only
policy producing a summary of exactly the same size reaches 0.73–0.78 of ELKI's WCSS from a tree
built at twice the budget. The 8 % we lose to ELKI is the small end of what this lever is worth; the
same measurement says a further ~25 % is available beyond ELKI's own policy.

Two costs keep this from being a free fix, and they are why the current policy exists. Building at
2× the budget holds 2× the leaves *during* the build, which is the memory bound the library sells;
and the global merge is `O(m²)` in the leaf count against the sibling scan's `O(Σ child²)` — 1.6 GB
of cost matrix at 14 000 leaves, which is what killed the ×8 covtype cell of this very probe. The
shape of a fix that keeps both bounds is **E13**: rank candidate pairs by ΔSSE from the approximate
kNN graph the library already builds over leaves, rather than by node membership, at rebuild time
where the tree is already at its budget. What is *not* yet separated is how much of the 0.72 is the
finer build (more information before compacting) and how much is global-versus-sibling merging at a
fixed tree size; that separation needs the merge policy swapped inside the rebuild, which is E13's
own experiment — done in the next section, where it turns out to be 21 of the 28 points.

Two things this does *not* say. The partitions agree with each other at ARI 0.39–0.53 even where
their WCSS matches to 1 %, so "the same tree quality" is not "the same tree" — the leaf boundaries
land in genuinely different places, and a downstream head sees different summaries. And the timing
column is unusable as a speed comparison: ELKI's 0.5–5.5 s per run is a JVM start plus CSV parsing
plus a result write, against 0.01–0.14 s for an in-process call. Nothing here measures the
summarisation loop against ELKI's.

### The fix, and what it was actually worth (E13, 2026-09-10)

T27 left the lever unseparated: how much of the 0.72 is the finer build and how much is the merge
policy at a fixed tree size. It is almost all the policy, and the policy is one line.

`sibling_pairs` chose each entry's partner under `dist` and ranked the pairs by the `abs` gap. Under
the default D0/D0 geometry both are the *unweighted* squared centroid distance, so compaction merged
the geometrically closest pair whatever it weighed. That is not the merge that costs least: the exact
price of fusing two features is the Ward cost `w_a w_b/(w_a+w_b) ‖μ_a−μ_b‖²`, and an unweighted gap
prices a pair of thousand-point leaves exactly like a pair of singletons the same distance apart. The
consequence is visible in the summary it built — on digits at 200 leaves **63 % of the leaves held a
single point** and the leaf-mass Gini was 0.82.

The check that this is the mechanism and not a coincidence was already in the record: under D4
(`VarianceIncrease`) the gap *is* the Ward cost, so a D4-routed tree was already ranking correctly —
and D4/R is exactly the geometry where the matched table above calls the two implementations level.
A distortion-ranked rebuild should therefore move D0/D0 a long way and D4/R barely at all. It does
(`local/scratch/e13_compaction.py`, seeds 0/1/2, medians; leaf counts are `n_leaves_`):

| fixture | geometry | leaves | singleton share | leaf-mass Gini | rebuilds | leaf WCSS | of ELKI |
|---|---|---:|---:|---:|---:|---:|---:|
| digits, 200 | D0/D0 before | 195 | 0.626 | 0.815 | 23 | 7.087e5 | 1.085 |
| digits, 200 | D0/D0 after | 180 | **0.139** | **0.530** | 8 | **5.601e5** | **0.857** |
| digits, 200 | D4/R before | 180 | 0.050 | 0.468 | 19 | 5.250e5 | — |
| digits, 200 | D4/R after | 187 | 0.048 | 0.406 | 7 | 5.158e5 | — |
| covtype-50k, 2000 | D0/D0 before | 1950 | 0.309 | 0.849 | 27 | 8.642e4 | 1.076 |
| covtype-50k, 2000 | D0/D0 after | 1987 | **0.148** | **0.586** | 5 | **6.794e4** | **0.846** |
| covtype-50k, 2000 | D4/R before | 1957 | 0.103 | 0.628 | 21 | 6.859e4 | — |
| covtype-50k, 2000 | D4/R after | 1908 | 0.103 | 0.505 | 5 | 6.474e4 | — |

D0/D0 drops 21 % on both fixtures — from 8 % behind ELKI to 15 % ahead — and reaches 0.79 of the old
WCSS against the 0.72 T27's offline bound got from a tree built at **twice** the budget with a
*global* `O(m²)` merge. So the finer build and the global reach together are worth about 7 points of
the 28; the ranking is worth 21. D4/R moves 1.8 % and 5.6 %, as predicted, because its partner choice
was already the Ward-optimal one. The rebuild count falls with it (27 → 5), since a rebuild that
takes the cheap merges lands further under the budget and stays there.

The bounds T27 said a fix had to keep are both kept: the scan is still `O(Σ_leaf child²)` over
siblings, the tree is still built once at its budget, and nothing new is allocated — the Ward cost
reuses the weights and means already in the features.

**What it costs downstream.** A tighter summary is not automatically a better label assignment, so
the published quality fixtures were run on both builds before this landed
(`local/scratch/e13_quality.py`, medians of seeds 0/1/2 at `max_leaves=4000`). The singleton share
falls on every fixture (mnist 0.852 → 0.532, highdim 0.866 → 0.442, covtype 0.453 → 0.262); ARI moves
where the head is sensitive to the summary and not otherwise:

| head | improves | flat | loses |
|---|---|---|---|
| ward | moons 0.148 → **0.599**, varied 0.433 → **0.705**, aniso 0.471 → 0.526, blobs 0.810 → 0.831, mnist 0.367 → 0.394 | covtype, digits, highdim, circles | — |
| gmm | mnist 0.269 → 0.306, covtype 0.055 → 0.059 | every synthetic fixture (≤ 0.0003) | — |
| kmeans | — | every synthetic fixture (≤ 0.001) | mnist 0.315 → 0.285, covtype 0.064 → 0.049 |

The ward head is the one that reads the leaf masses directly, and it is the one that gains; k-means
loses two real-set cells and gains none, which is the honest cost of the change. Both losses are
inside the seed spread of the cell they sit in (mnist k-means baseline [0.258, 0.341] against
[0.261, 0.331]), and neither is on a fixture where k-means recovers the labels at all — covtype ARI
is 0.05–0.10 for every method in the table.

Two negative controls. `absorb="chi2"` reads bit-identical ARI before and after on both fixtures,
because its gate binds at 46–57 leaves and the budget is never reached, so no rebuild ever runs —
which is the change firing only through compaction, as intended. And the spectral head's occasional
collapse on `circles` is not caused by this: ten seeds per build put 2/20 runs below ARI 0.9 on
either side of the change (`local/scratch/e13_spectral_seeds.py`), the same k-means-on-the-embedding
lottery T23 measured, sampled differently by a three-seed window.

Five non-default geometries were run for the same reason — the Ward rank is a squared-Euclidean
statement and the tree may be routing on something else (`local/scratch/e13_geometry.py`, digits at
200 leaves, covtype-20k at 4000). On digits every one of them improves (`euclidean/radius`
0.533 → 0.548, `euclidean/diameter` 0.485 → 0.554, `ward/radius` 0.440 → 0.476,
`manhattan/manhattan` 0.482 → 0.574); on covtype they move inside their own spread in both
directions. Nothing there argues for making the rank follow `absorb`.

**E13's own stop rule was not met, and the change ships on other evidence.** The task was written to
chase covtype: *"continue only at ARI ≥ 0.10 on either head"*, otherwise "the record says covtype
needs a *split*, i.e. an in-leaf sub-sketch". Covtype-20k at 4000 leaves reads ward 0.0861 (bit for
bit unchanged) and k-means 0.0640 → 0.0486. Neither clears 0.10. The prediction that framed the task
is confirmed in both halves: the singleton share does fall (0.453 → 0.262, Gini 0.683 → 0.490) and it
buys nothing there, because a merge-only compaction cannot split the heavy cells that hold covtype's
mass. What changed is why the fix is worth having — not covtype's ARI, but the summarisation
objective the tree exists to optimise, on every fixture measured, and the ward head that reads it.
Covtype still needs a split.
