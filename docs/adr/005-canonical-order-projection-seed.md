# ADR 005 — the canonical order's projection constant is fixed, and is not a tuning target

**Status:** Accepted · records a decision **not** to change a shipped constant, taken after a
measurement said changing it was justified and a held-out re-measure said it was not. The constant is
`order::PROJECTION_SEED = 0x0BE7_014A_C0DE_0117`, shipped since **0.7.0**.

## Context

`canonical_order=True` sorts the rows by a Morton interleave of eight random projections, and those
projections are drawn from one fixed constant. The guarantee the flag sells — the same rows give the
same labels however they arrived — is a guarantee *relative to that constant*: change it and every
`canonical_order=True` user gets different labels. It is therefore a type-1 door, and cheap to walk
through by accident, because it is one hexadecimal literal.

It looked badly chosen. `benches/canonical_order.rs` reports the canonical build rebuilding the tree
3 → 30 times at `50k × 784, max_leaves = 8000`, which is what makes the insert 1.6× the arrival-order
build at that shape — the single worst cell on the page. A sweep of seven candidate constants over
six shapes found four that read 3 → 3..5 there, and 138–152 total rebuilds against the shipped
constant's 172. A 25 % margin on a magic number nobody had ever measured is normally enough to change
it.

Three measurements decided otherwise.

**1. Quality does not move with the constant, so the choice is a cost choice.** Eighteen cells
(`digits` / `covtype-20k` / `mnist-10k` × two leaf budgets × `kmeans` / `ward` / `gmm`), seven
constants, three head seeds per cell, each scored as a delta against the arrival order's *median*
draw over eight permutations. Friedman over the seven columns gives **p = 0.53**; no pairwise test
against the shipped constant reaches p = 0.16; and the largest gap between any two constants' mean
delta is **0.028**, against the arrival order's own median min–max spread of **0.077**. The projection
draw decides *which* prototypes exist, so it could have moved quality. It does not.

**2. The cost margin does not survive shapes the candidates were not chosen on.** Three synthetic
shapes outside the screen's grid and two real datasets, shipped constant against the screen's winner
(`0xDEAD_BEEF_CAFE_0001`) and runner-up (`0xA5A5_5A5A_A5A5_5A5A`): total canonical rebuilds **107 /
109 / 99**. On `blobs 60k × 384, max_leaves = 6000` the roles invert outright — the shipped constant
rebuilds **3** times and the screen's winner **15**. A margin that disappears off the shapes it was
measured on is a fit to those shapes.

**3. The rebuild count is a right-tailed lottery over the draw, not a property of the constant.**
Twenty-four projection draws per shape, everything else held fixed:

| shape | arrival | min | median | max | shipped constant |
|---|---|---|---|---|---|
| `blobs 50k × 784`, `ml = 8000` | 17 | 3 | 4 | **31** | 4 (rank 10/24) |
| `blobs 60k × 384`, `ml = 6000` | 4 | 3 | 4 | 18 | 3 (rank 1/24) |
| `blobs 30k × 256`, `ml = 1000` | 22 | 3 | 7 | 19 | 3 (rank 1/24) |
| `mnist-20k × 784`, `ml = 2000` | 27 | 24 | 28 | 33 | 29 (rank 15/24) |

A median of 4 with a maximum of 31 on identical data is the whole finding: roughly one draw in
twenty-four is pathological at a given shape, and *which* draw is pathological changes with the
shape. The shipped constant is ordinary to best on all four and is never the tail. The published
3 → 30 cell is one such coincidence between a draw and a dataset, not a defect in the constant.

## Decision

**`PROJECTION_SEED` stays at `0x0BE7_014A_C0DE_0117`, and the constant is recorded as not tunable.**

Re-rolling it on a screen of a few shapes will always produce a winner — the distribution guarantees
one — and that winner will not generalise. Any future proposal to change it needs a held-out set
measured after the candidate was picked, and this record exists so that the six-shape screen is not
run a second time and mistaken for evidence.

## Consequences

**A published number is corrected rather than defended.** "3 → 30 rebuilds, insert 1.56×" was
presented as what the scheme costs at high dimension. It is what *this draw* costs on *that dataset*,
at the 24-draw maximum. Every place that quotes it now says so.

**The cost of the flag is a distribution, and the documentation quotes its shape.** A reader choosing
`canonical_order=True` should expect the median, know the tail exists, and know that it is not
avoidable by picking a better constant.

**Choosing the draw at fit time is foreclosed on evidence, not on principle.** Selecting among *k*
draws by a statistic of the data would preserve the guarantee — the choice would still be a function
of the multiset. It fails on measurement: no cheap statistic of the codes predicts the rebuild count.
Over the same 24 draws, Spearman ρ against the rebuild count is at most 0.48 in magnitude and mostly
under 0.27; the sign of the entropy correlation flips between cases; and the distinct-code fraction is
1.000 for every draw, carrying no signal at all. With nothing to select on, selection would mean
building a tree per candidate — *k* times the insert cost to dodge a tail in a rebuild count that is
itself a fraction of the fit.

**What is not foreclosed is a different key.** This record is about re-rolling the seed of the
existing eight-random-projection key. A key with a different construction is a different decision,
and one candidate was already measured and rejected on its own terms: a PCA-8 key improves the
locality proxies substantially (on MNIST the adjacent-row step falls 0.808 → 0.651 and the fraction
of long jumps 0.205 → 0.026) and buys no ARI and no rebuild-count advantage (16–17 rebuilds against
the random draw's 15–20).

## Alternatives rejected

- **Adopt the screen's winner (`0xDEAD_BEEF_CAFE_0001`).** Rejected by measurement 2: it is the worse
  constant on a held-out shape by the same margin that made it the winner on the screen, and the two
  are tied over the held-out set as a whole.
- **Adopt the runner-up (`0xA5A5_5A5A_A5A5_5A5A`), which did edge ahead held-out (99 vs 107).**
  Rejected because the difference is six cells wide, is inside the per-cell spread, and would relabel
  every `canonical_order=True` user to buy it.
- **Select the draw from the data at fit time.** Rejected on the ρ ≤ 0.48 result above.
- **Draw more projections, or quantise more finely.** Measured separately and rejected: at `d = 784`
  the eight projections already carry ~1 % of the variance, and neither more bits nor a rank
  quantiser moved the locality statistics in a direction that showed up in the fit.
