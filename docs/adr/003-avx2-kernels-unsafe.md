# ADR 003 — the distance kernels get a hand-written AVX2 path, and the crate's first `unsafe`

**Status:** Accepted, on measurement, in **0.7.0**. Scope: `src/kernels.rs` only. Amended after the
first version shipped a 1.55× regression at `d = 2` — see *The length gate* below.

## Context

`src/kernels.rs` holds three reductions — `sq_euclidean`, `dot`, `manhattan` — and its module doc
claimed they were "plain inlinable loops the compiler vectorizes at each call site". A profile taken
for task #40 showed how much rides on that claim and that the claim was false.

Single-threaded `perf record -F 997 --call-graph fp`, extension built with frame pointers:

| shape | fit | share of samples inside these kernels |
|---|---|---|
| 1 000 000 × 20 | `kmeans` | `RouteKind::point` 31.5% + `CFTree::descend` 24.7% = **56%** |
| 20 000 × 784 | `gmm` | `RouteKind::point` 10.6% + `descend` 12.6% = 23% |

`perf annotate` on the 31.5% symbol returns **only `subsd`, `mulsd`, `addsd`** — not one packed
instruction.

The reason is a language rule, not a missing build flag. `a.iter().zip(b).map(…).sum()` is
`Iterator::sum`, a strictly-ordered left fold; IEEE addition is not associative; so LLVM may not
reassociate the reduction, and without reassociation there is nothing legal to pack. Adding
`-C target-cpu=native` cannot change this, and the crate does not set it anyway — the shipped wheels
target baseline `x86-64`, where even a legal vectorization would only get SSE2's two `f64` lanes.

A microbenchmark separates the two blockers. Against the shipped scalar fold, on this machine:

| `d` | 4 independent accumulators (baseline ISA) | explicit AVX2 + FMA |
|---|---|---|
| 20 | **0.93×** (slower) | **2.35×** |
| 64 | 1.13× | **3.89×** |
| 784 | 1.46× | **2.43×** |

So breaking the dependency chain alone is not the win — at `d = 20` it is a small loss. The win is
the register width, and at baseline ISA the only way to reach it is to write the intrinsics out and
detect the feature at run time. That requires `unsafe`, which this crate had none of.

## Decision

**Write the AVX2 + FMA kernels by hand in `src/kernels.rs`, dispatched at run time, for `f64` and
`f32`. Accept `unsafe` in that one module and nowhere else.**

Shape of the implementation, and why each part is the way it is:

- **Two packed accumulators, then a scalar tail in the original order.** A vector shorter than the
  step is byte-for-byte the old code.
- **A length gate: the packed path is taken only at `d ≥ 16`.** See below; this was not in the first
  version and its absence was a regression, not a missed optimisation.
- **`n = a.len().min(b.len())`, passed explicitly into every kernel.** This is the soundness guard.
  `zip` truncates to the shorter operand and `CFTree::insert` documents that it accepts a point
  longer than the tree's dimension, so an intrinsic loop bounded by `a.len()` would be an
  out-of-bounds `loadu`, not merely a wrong number. It has its own test, which calls the kernel
  directly — the public wrapper's `debug_assert_eq!` makes the case unreachable through the API in a
  debug build, which is exactly why the guard needs testing from underneath.
- **Dispatch through a macro, not a function pointer.** An earlier attempt with the `multiversion`
  crate was measured *slower* for small `d`; this is why. An indirect call cannot inline, and at
  `d = 20` the call overhead is a fifth of the total work. A macro expands the branch at each call
  site and keeps the intrinsic body inlinable.
- **`TypeId` to reach a concrete type from `R: Real`.** `Real` is a blanket impl over
  `num_traits::Float`, so there is no `f64` to specialize on. `TypeId` is injective over `'static`
  types, so a match *proves* `R` is `f64` and the slice reinterpretation is a layout no-op. The
  result comes back through `FromPrimitive`, which is exact and needs no `unsafe` at all.
- **`#[allow(unused_unsafe)]` on the module.** Rust 1.87 made the arithmetic intrinsics safe to call
  inside a matching `#[target_feature]` function; the declared MSRV is 1.85, where the blocks are
  mandatory. The lint is silenced rather than the floor raised, with a note to remove it when
  `rust-version` passes 1.87.

## Evidence

End-to-end, A-B-A-B on the same build with only this module differing, `RAYON_NUM_THREADS=1`,
best of three per cell:

| shape | scalar | AVX2 | speedup | labels |
|---|---|---|---|---|
| 300 000 × 20, `kmeans`, `max_leaves=4000` | 0.390 s | 0.282 s | **1.38×** | `dc646b4dc9b9a7be` both |
| 20 000 × 784, `gmm`, `max_leaves=2000` | 2.111 s | 1.326 s | **1.59×** | `e887117b6aaa25c2` both |

The task's acceptance gate was "no label changes". Eight partial sums and an FMA are a different
rounding from one serial chain, so an argmin decided by less than an ulp could in principle flip, and
the honest claim is a measurement rather than a theorem. What was measured: the label digests above;
the 518-test Rust suite, which includes the SciPy dendrogram cross-checks, the ELKI cross-check and
several exact-value pins; and a re-run of the whole quality benchmark at seed 1.

That last one is the interesting evidence, because it did **not** come back clean at first. Three of
78 quality rows and two of the real-dataset rows moved, all of them `betula-hdbscan`, one of them by
a lot (`aniso` ARI 0.571 → 0.993). A direct A/B on exactly those cells — same fixture, same
arguments, the two builds differing only in `src/kernels.rs` — returned **identical label digests on
both sides**, so the kernels are not the cause: the committed CSVs simply predate `835d05f`, the
commit that made `min_samples` and `min_cluster_size` count points rather than leaves and whose own
message says "on a summary the labels change". The other 75 rows reproduce cell for cell. Re-running
the benchmark and diffing against the record is what turned a scary-looking diff into two separate,
correctly-attributed facts.

## The length gate, and the regression that forced it

The first version dispatched on feature detection alone. That was wrong, and the measurement that
found it was the one that should have been taken before committing: sweep `d`, not just the two
profiled shapes.

A `#[target_feature]` function cannot be inlined into a caller that does not carry the same features.
So taking the AVX2 path replaces an inlined loop with a real call plus a horizontal sum — a fixed
cost of maybe twenty instructions. At `d = 20` that is noise against 20 fused multiply-adds. At
`d = 2` the scalar body *is* two multiply-adds, and the fixed cost is the entire function.

Measured end-to-end, `kmeans`, `RAYON_NUM_THREADS=1`, best of three, ungated against the pre-AVX2
build: **0.65× at `d = 2`** (500 000 rows, 0.136 s → 0.210 s) and **0.89× at `d = 8`**. Two-dimensional
input is not a corner case — the crate's own published scaling table is measured on `d = 2` blobs, and
geospatial input is a documented use — so this would have shipped a 1.55× slowdown on the shape the
benchmark advertises.

The fix is a constant, `AVX2_MIN = 16`, plus `#[inline(never)]` on the dispatch helper so the public
wrapper stays small enough to inline and the scalar fold keeps the codegen it had before this module
grew. Both halves were needed: with the gate but the dispatch inlined, `d = 8` measured 1.26× *slower*
than the ungated version it was meant to fix, because the bulk made LLVM decline to inline the wrapper
at all.

Where the constant comes from: at `d = 12` the packed path measured 0.95× and at `d = 16` it measured
1.08×, so 16 is the first width that pays. `f32` crosses over at the same element count despite its
packed step being twice as wide.

**A residual remains and is not explained.** With the gate in place, `d ≤ 8` still measures ≈0.92× —
about 8% — even though those inputs execute the same scalar fold plus one compare against a constant.
It is real and not build noise: three rebuilds of the unmodified base from identical source measured
0.135 / 0.137 / 0.138 s, a ±1% spread. Gating on the loop-invariant operand so LLVM could hoist the
compare, and `#[inline(always)]` on the wrappers, each moved it by under 2%. The likely cause is
second-order codegen in the inlined callers rather than the branch itself. It is recorded rather than
argued away.

Final position, against the pre-AVX2 build:

| shape | before | after | |
|---|---|---|---|
| 500 000 × 2, `kmeans` | 0.136 s | 0.146 s | **0.93×** |
| 300 000 × 8, `kmeans` | 0.230 s | 0.251 s | **0.92×** |
| 300 000 × 16, `kmeans` | 0.342 s | 0.318 s | **1.08×** |
| 300 000 × 20, `kmeans` | 0.397 s | 0.321 s | **1.24×** |
| 300 000 × 16, `kmeans`, `float32` | 0.303 s | 0.256 s | **1.18×** |
| 300 000 × 24, `kmeans`, `float32` | 0.374 s | 0.237 s | **1.58×** |
| 20 000 × 784, `gmm` | 2.12 s | 1.35 s | **1.57×** |

That is the trade being accepted: about 8% below `d = 16`, 1.1–1.6× above it. The library's own real
datasets sit at `d` = 54, 64, 100 and 784.

## `float32` relabels, and why that is not a behaviour change

On `float32` input the SHA-256 label digest **does** differ between the two builds at `d = 24` and
`d = 32` — 87% and 25% of rows carry a different integer. The **ARI between the two label vectors is
exactly 1.000000**: the partition is identical and only the cluster numbering is permuted, because
k-means++ draws its seeds in a different order when a sampling weight moves by an `f32` ulp. `f64`
digests are identical everywhere measured.

Reported because a digest comparison alone would have called this a behaviour change, and an ARI
alone would have hidden that the ids move. Both are true and they mean different things.

## Consequences

- **The crate is no longer `unsafe`-free.** Roughly 130 lines in one leaf module with no callers'
  invariants to uphold: two preconditions (features detected, `n` bounds both slices), both
  established one call frame above and both tested.
- **Non-`x86_64` targets are unaffected** — the module is `#[cfg]`-gated and the scalar fold is the
  fallback, so aarch64 and everything else compile and behave exactly as before, just without the
  speedup. A NEON path is the obvious follow-up and is deliberately not in this change.
- **Pre-AVX2 `x86_64` is unaffected**, by run-time detection rather than by build configuration, so
  one wheel keeps serving both.
- **The published speed tables are now stale by 1.4–1.6× on the two profiled shapes** and are
  re-measured alongside this change; that is a benefit, but it is also the maintenance cost of
  every future edit to this module.
- **`f32` gets the same treatment as `f64`** (8 lanes instead of 4). Shipping the fast path for only
  one of the two scalar types the crate is generic over would have been an arbitrary seam.

## Alternatives considered

- **Do nothing and correct the module doc.** Rejected once the ceiling was measured: 56% of a
  `kmeans` fit sitting in a kernel running at 40% of the machine's width is not a documentation
  problem. The doc is corrected as well, either way.
- **`-C target-cpu=native` in `.cargo/config.toml`.** Rejected twice over: it cannot legalise the
  reassociation, so it would not vectorize these loops at all; and it produces binaries that
  `SIGILL` on any older CPU, which is not a property to give a published wheel.
- **The `multiversion` crate.** A new dependency, and the thing it does — dispatch through a function
  pointer — is the thing already measured slower at the `d` that matters most here.
- **Four independent accumulators, no intrinsics, no `unsafe`.** The honest safe alternative, and the
  one that would have been taken if the numbers supported it. They do not: 0.93× at `d = 20`, and it
  still changes the summation order, so it pays the same label-identity price for a fraction of the
  gain.
- **`core::simd`.** Portable, safe, and nightly-only. Revisit when it stabilises; it would let the
  `unsafe` and the `#[cfg]` fork both go away.
- **Narrow `Real` to a sealed trait over `f32`/`f64` and specialize properly.** Cleaner than `TypeId`,
  but it is a breaking change to a public trait in service of an internal dispatch detail. Worth
  doing at a major version if `Real` is narrowed for other reasons; not worth doing for this.
