# ADR 002 — a Bregman-Ward HAC must use Anderberg, not the nearest-neighbour chain

**Status:** Accepted · decided **before** the head was written, on measurement. The feature and the
`D4_φ` measure (`bregman::BregmanCf`, `bregman::BregmanIncrease`) ship in **0.7.0**; the
agglomerative head over them is not written yet, and this record is what constrains it.

## Context

`clustering::ward_hac` runs the **nearest-neighbour-chain** algorithm. NN-chain is not a heuristic —
it produces the exact dendrogram in `O(m²)` time and `O(m)` space, but only for a **reducible**
linkage:

```text
D(A,B) ≤ D(A,C)  and  D(A,B) ≤ D(B,C)   ⟹   D(A∪B, C) ≥ D(A,B)
```

Ward's linkage over squared Euclidean is reducible (Murtagh 1983; Müllner 2011), which is why the
shipped head is correct. Generalising the cluster feature to an arbitrary Bregman divergence
generalises Ward's criterion with it — `D_φ(A,B) = n_A d_φ(μ_A, μ_AB) + n_B d_φ(μ_B, μ_AB)` — and
nothing about that construction preserves reducibility. The question is whether it survives anyway,
because if it does, the Bregman head inherits `ward_hac`'s driver for free.

## Decision

**It does not survive, and the failure is not academic.** The Bregman-Ward HAC uses **Anderberg** —
one global minimum per step over a lazily repaired nearest-neighbour cache, `O(m²·d)` expected and
`O(m·d)` space. The **coordinate-wise NN-chain** proposed as a cheaper rescue is rejected.

## Evidence

`local/scratch/bregman_reducibility.py` (counterexample search) and
`local/scratch/bregman_nnchain_cost.py` (dendrogram-level cost). A violation counts only when its
relative margin clears `1e-7`, four orders above the stable closed forms' own error, so f64 noise
cannot manufacture one.

**Reducibility holds in one dimension and fails from two.** 40 M sampled triples per divergence,
13.3 M of them admissible, over 16 different `φ` — **zero** violations at `d = 1`, for every single
one. At `d ≥ 2` every `φ` except squared Euclidean violates:

| divergence | d=1 | d=2 | d=5 | d=20 | worst relative violation |
|---|---|---|---|---|---|
| squared Euclidean | 0 | **0** | **0** | **0** | 0 |
| KL | 0 | 189 | 260 | 22 | 0.147 |
| Itakura–Saito | 0 | 1 076 | 2 224 | 483 | 0.335 |
| logistic | 0 | 205 | 566 | 945 | 0.092 |
| exponential | 0 | 337 | 1 662 | 2 942 | 0.245 |

(out of ~667 000 admissible triples per cell).

**Squared Euclidean is the only clean member of the power family**, and the degradation is
continuous rather than a threshold — `φ(t) = tᵖ` at `p = 1.99` and `p = 2.0` give zero violations
everywhere, `p = 2.01` gives 2 at `d = 20`, and the rate climbs monotonically with `|p − 2|` to
17 078 / 666 169 (2.6 %) at `p = 4`, `d = 20`. Nearness to squared Euclidean therefore buys graceful
degradation, not safety.

**The failure is purely additive.** Inside every pooled counterexample, each coordinate was checked
against its own one-dimensional reducibility: across roughly 250 000 coordinate-admissible cases,
**not one coordinate ever violated it**. For a separable `φ` the divergence is a sum of 1-D Bregman
divergences and each summand's inequality holds; it is the sum that fails. Squared Euclidean escapes
because its Ward distance `(n_A n_B / n_AB)‖Δμ‖²` factorises into a coordinate-independent scalar
times a sum of squares, and KL, Itakura–Saito, logistic and `exp` do not factorise that way.

**It costs a wrong dendrogram, not a wrong triple.** Exact Anderberg against NN-chain on the same
weighted set, comparing sorted merge-height spectra (NN-chain provably does not *emit* merges in
height order even when it is correct — comparing emission order measures bookkeeping, which the
euclidean control caught on the first run of the harness):

| divergence | d | m | instances | differing trees | worst ARI at k=4 |
|---|---|---|---|---|---|
| squared Euclidean | 2 / 5 / 20 | 5 / 8 / 12 | 3 000 each | **0 in all 9 cells** | 1.0000 |
| KL | 5 | 12 | 3 000 | 2 | 0.7673 |
| KL | 20 | 12 | 3 000 | 2 | 0.9009 |
| Itakura–Saito | 5 | 12 | 3 000 | 23 | 0.3663 |
| Itakura–Saito | 20 | 8 | 3 000 | 4 | **0.1000** |
| Itakura–Saito | 20 | 12 | 3 000 | **30 (1.0 %)** | 0.1720 |
| exponential | 20 | 8 | 3 000 | 17 | 0.2097 |
| exponential | 20 | 12 | 3 000 | **36 (1.2 %)** | 0.2143 |

Two things in that table decide it. The rate **grows with `m`** — Itakura–Saito at `d = 5` goes
2 → 15 → 23 as `m` goes 5 → 8 → 12 — and a real head runs at `m` in the thousands, where a 1 % rate
at `m = 12` extrapolates to a chain that essentially never builds the right tree. And when it does
diverge the answer is **destroyed, not perturbed**: ARI at `k = 4` of 0.1000, and one exponential
cell at `−0.1111`, worse than a random partition.

**The coordinate-wise rescue is refuted.** Since every coordinate *is* reducible, the natural idea is
to run per-coordinate chains and recover the linear space bound. That needs the coordinates to agree
about which pair is nearest. Measured over 400 rounds of 12 clusters: a single coordinate's nearest
pair matches the pooled nearest pair **2.4 %–6.5 %** of the time against a chance rate of **1.52 %**.
The coordinates carry almost no information about the pooled minimum, so per-coordinate chains do not
define a single merge sequence, let alone the right one.

## Consequences

- The Bregman HAC pays `O(m²·d)` expected time and gives up NN-chain's `O(m·d)` space. At the leaf
  counts a CF-tree produces this is the same cost class the shipped `agglomerative` driver already
  pays for centroid and median linkage, which are non-reducible for the same structural reason.
- `clustering::ward_hac` is **unaffected**: it is Euclidean, the control row is clean at every
  dimension and every size, and nothing here argues for changing it.
- A user asking for Ward-like agglomeration under a near-quadratic `φ` gets a warning-free correct
  answer from Anderberg rather than a silently wrong one from a chain, which is the point.
- The 1-D result is a **conjecture supported by 213 million admissible triples across 16 `φ`**, not a
  proof. It is not load-bearing — nothing ships that depends on 1-D reducibility — but it is the
  natural next theory question, and a proof would characterise exactly which `φ` the pooled
  inequality can survive.

## Alternatives considered

- **Use NN-chain anyway and document the risk.** Rejected: a 1 % failure rate at `m = 12` that grows
  with `m`, producing ARI 0.10 when it fires, is not a documentable caveat. The failure is silent —
  there is no signal in the output that the tree is wrong.
- **Detect violations at run time and fall back.** Rejected: the check is per-triple, and checking
  every triple is exactly the `O(m²)` work Anderberg already does. It would buy the space bound only
  when the data happens to be benign, at the cost of a second code path with its own failure mode.
- **Restrict the head to `φ` empirically near `t²`.** Rejected: `p = 2.01` already violates at
  `d = 20`. There is no safe neighbourhood, only a small one, and a library cannot ship a head whose
  correctness depends on how far the user's `φ` is from squared Euclidean.
