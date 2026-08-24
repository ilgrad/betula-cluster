# ADR 003 — the distance kernels get a hand-written AVX2 path, and the crate's first `unsafe`

**Status:** Accepted, on measurement, in **0.7.0**. Scope: `src/kernels.rs` only.

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
