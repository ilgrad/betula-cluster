# ADR 006 — one exact kernel for nearest-centre assignment, and the pruned plan only behind a hint

**Status:** Accepted · records how `predict` finds a row's nearest centre or leaf in the release
after **1.5.0**. Through 1.5.0 the estimator's `predict` labelled a row by a serial scalar `f64`
loop over the centres. The pruned plan in `src/assign.rs` shipped in **1.0.0** with no caller; the
rows head (`leaf_graph="rows"`) is its first.

## Context

The crate has two exact ways to find the nearest of `k` points to a row.

1. **The full scan**, `kernels::nearest_sq_euclidean`. It takes the CPU dispatch once per scan and
   runs the packed AVX2 + FMA kernel from 16 coordinates, the serial loop below that. It keeps a
   candidate only on a strict `<`, in index order.
2. **The plan**, `AssignPlan`. It sums each candidate's squared distance in stages, in descending
   order of between-centroid variance, and stops a candidate as soon as its partial sum is strictly
   greater than the best full distance so far. From `d = 128` a stage is 64 coordinates
   (`assign::default_block`). A hint names a first candidate, whose distance becomes the starting
   threshold. The plan decides on the pair (distance, index), so a hint changes the work and never
   the answer.

The bound in the plan is exact: a running sum of non-negative terms does not decrease under
round-to-nearest, so a partial sum above the threshold proves the full sum is above it too.

Two callers need a nearest point.

- **A centre rule** labels a row by its nearest centre: `kmeans`, `xmeans`, `kmedoids` and
  `spherical-kmeans`, with or without a projection, and `fuzzy-cmeans`, whose memberships come
  from the same distances. It has no previous label to use as a hint.
- **The nearest-leaf rule** of the rows head labels a row by its nearest of about 4 000 leaves. The
  tree descent gives every row a leaf, which is a hint.

The scalar loop was the most expensive part of `predict`. At 256 centres it retired 32.5 G
instructions on MNIST-20k (`d = 784`) and 14.3 G on the 17 901 rows of 20 newsgroups (MiniLM,
`d = 384`), float32.

The plan's own benchmark (`src/assign.rs`) has it losing to a scan below `d ≈ 128` even with a
hint, and winning 2–8× from `d ≈ 784` with one. Without a hint it had not been measured on real
data. It was measured here on a `kmeans` rule, float32 rows, `max_leaves=2000`, `threshold=0.0`,
seed 0, one thread, retired instructions of one `predict` of the training rows: MNIST-20k, 20
newsgroups embedded by MiniLM (17 901 rows) and by MPNet (18 119), and 20 000 rows drawn from
CIFAR-100 embedded by CLIP:

| set | `d` | `k` = 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| MNIST | 784 | 2.22 → 1.63 G | 4.18 → 2.51 G | 8.10 → 4.03 G | 15.95 → 6.79 G |
| 20 newsgroups, MiniLM | 384 | 1.01 → 1.65 G | 1.91 → 2.77 G | 3.70 → 4.77 G | 7.29 → 8.34 G |
| 20 newsgroups, MPNet | 768 | 1.97 → 2.79 G | 3.71 → 4.50 G | — | 14.17 → 12.58 G |
| CIFAR-100, CLIP | 1280 | 3.57 → 3.81 G | 6.72 → 5.72 G | — | 25.63 → 13.31 G |

Each cell is full scan → plan. Every cell returned the same labels both ways. The plan wins on
MNIST from `k = 128` and on CLIP from `k = 256`. It loses on MPNet up to `k = 256` and on MiniLM at
every `k` measured. Why it loses there was not measured. Two causes are plausible: a flat variance
spectrum, which leaves little of the distance in the first stages, and the dispatch. Each stage
calls `kernels::sq_euclidean` through its run-time dispatch, which the full scan pays once per scan.
Building the plan costs about 30 full scans, so a one-row call with a plan retired 6.9 M
instructions against 0.23 M for the scan (MNIST, `k = 256`).

## Decision

**1. The distance is exact, with no rotation and no statistical bound.** A prefix bound prunes
without an error budget, and pruning cannot change a label: every comparison is between two sums
accumulated the same way. SuperKMeans's ADSampling bound (arXiv 2603.20009) was rejected in 1.0.0
and stays rejected: its Haar rotation costs `d²` per row, which at the `k` of a CF summary is more
than the assignment it speeds up.

**2. Every centre rule takes the full scan.** For `Centers` and `Fuzzy` rules, dense and CSR rows,
one row or many, `predict` widens each row to `f64` and scans it with `nearest_sq_euclidean`, the
kernel the one-shot function already labelled its rows with, in the tree's dtype. Fuzzy
memberships use `sq_euclidean` over the same rows, so `predict_proba(X).argmax(1)` names the centre
`predict` names. A centre rule builds no plan, at any width, any centre count or any row count.

**3. The plan runs only where a hint exists.** The nearest-leaf rule has one in the descent's leaf.
The graph pass and `predict` run the same hinted search in the tree's dtype, so a training row's
label is its leaf's community by construction, and `fit_predict` reads each row's leaf off the
graph pass. The plan is cached per tree dtype and built on first use.

**4. `d ≥ 128` is a necessary condition for the plan, not a sufficient one.** Below `d ≈ 128` it
loses even with a hint. From `d = 128` the stage is 64 coordinates. Without a hint, `d ≥ 128` and
`k ≥ 256` still lose on two embedding sets, so no fixed gate on `d` and `k` is adopted.

**5. Of two candidates at equal distance, the lower index wins, on every path.** The scan keeps
only a strictly smaller distance, in index order. The plan prunes only on a strictly greater
partial sum, so a tied candidate survives, and it decides on (distance, index).

**6. The order of the sum is not preserved, and the labels are a measurement.** From 16
coordinates the packed kernel sums in a different order from the serial loop, and the plan sums in
its own stages. A distance moves by a few ulp, about `1e-13` relative in `f64`, so an argmin decided
by less than that can flip. On MNIST-20k and MiniLM, over `kmeans` at their own `k` and at 300,
`xmeans`, `kmedoids` and `fuzzy-cmeans` in both dtypes, the smallest relative gap between a row's
two nearest centres was 2.5e-7, and no row sat on an exact tie.

## Consequences

**`predict` on a centre rule retires 3.5–8.1× fewer instructions, more with more centres.** On
the same rows: MNIST 1.49 → 0.41 G at 10 centres, 32.5 → 4.18 G at 256 and 129.2 → 15.9 G at
1 024; MiniLM 1.21 → 0.26 G at 20, 14.3 → 1.91 G at 256 and 56.9 → 7.29 G at 1 024.
The label digests equal those of the build before it on both sets, both dtypes and every
`predict` route. Fuzzy memberships move by at most 6.7e-16.

**A row is labelled alike alone and in a batch.** One kernel answers every call size, so
`predict(X)[i]` equals `predict(X[i:i+1])` by construction. A plan gated on the row count would
lose that at a near-tie.

**Nothing new is saved.** The rule is the same list of centres, and models saved by 1.5.0 load and
predict the same labels.

**The nearest-leaf rule pays the plan at every width.** Gating it on `d` would keep the label
identity only if the graph pass and `predict` moved together. Below `d = 128` the hinted plan costs
more than a scan by the plan's benchmark; this was not measured on the rows head.

**A plan without a hint stays open, with a bar to clear.** A plan that takes the dispatch once per
scan, or a gate on a statistic of the centres, could move the crossover. Any such proposal is
measured on MiniLM and MPNet, the sets that lose here, as well as on MNIST.

## Alternatives rejected

- **Plan the centre rules at `d ≥ 128` and `k ≥ 256`.** MiniLM `k = 256` 1.91 → 2.77 G, MPNet
  3.71 → 4.50 G: a regression on both embedding sets.
- **Plan them only at `k ≥ 1 024`.** MiniLM still loses, 7.29 → 8.34 G.
- **Seed the threshold from every candidate's first stage when there is no hint.** No coordinate is
  read twice and the answer does not change. Measured: MNIST `k = 256` 2.51 → 2.38 G, MiniLM 2.77 →
  2.79 G, MPNet `k = 1 024` 12.58 → 12.31 G. It does not change which side of the scan the plan
  falls on.
- **Keep the scalar loop.** It costs 3.5–8.1× the instructions of the full scan and gives the same
  labels on every set measured.
