//! Vector distance kernels — small reductions with an explicit AVX2 path on `x86_64`.
//!
//! These run in the CF-tree's hot path: a profile of `method="kmeans"` on 1 000 000 × 20
//! (`perf -F 997 --call-graph fp`, single-threaded) put `RouteKind::point` at 31.5% and
//! `CFTree::descend` at 24.7% — over half the fit inside these three functions.
//!
//! **They do not autovectorize, and no build flag makes them.** `perf annotate` on that 31.5%
//! returns only `subsd`, `mulsd` and `addsd`: not one packed instruction. The reason is a language
//! rule rather than a missing target feature — `a.iter().zip(b).map(…).sum()` is `Iterator::sum`, a
//! strictly-ordered left fold, and IEEE addition is not associative, so LLVM may not reassociate the
//! reduction and without reassociation there is nothing legal to pack. `-C target-cpu=native` cannot
//! change that.
//!
//! So the packing is written out, guarded by runtime feature detection, for the two scalar types the
//! crate is ever instantiated at. Measured end-to-end, A-B-A-B on the same build with only this
//! module differing, `RAYON_NUM_THREADS=1`: 300 000 × 20 `kmeans` 0.391 s → 0.277 s (**1.41×**),
//! 20 000 × 784 `gmm` 2.112 s → 1.327 s (**1.59×**), with the SHA-256 label digest identical in
//! every run of both shapes.
//!
//! Identical labels are a measurement, not a guarantee: eight partial sums and an FMA are a
//! different rounding from one serial chain, so an argmin decided by less than an ulp could in
//! principle flip. Nothing downstream stores these values — they are compared, never accumulated —
//! and the whole benchmark suite reproduces bit-for-bit, which is the strongest statement available.
//!
//! The dispatch is a macro rather than a function taking a kernel pointer on purpose. An earlier
//! attempt with the `multiversion` crate was measured *slower* for small `d`, and this is why: the
//! indirect call cannot inline, and at `d = 20` the call overhead is a fifth of the work. Expanding
//! the branch at each call site keeps the intrinsic body inlinable.

use crate::types::Real;

/// The AVX2 kernels, one per (operation, scalar type).
///
/// Each is `#[target_feature(enable = "avx2,fma")]`, so each is `unsafe` to call: the caller must
/// have checked `is_x86_feature_detected!` for both features. Every load is `loadu` — no alignment
/// precondition — and every index is bounded by the loop condition against `n`, which the callers
/// below set to `a.len().min(b.len())` so that the contract matches `zip`'s truncation exactly.
/// That `min` is load-bearing rather than defensive: `CFTree::insert` accepts a point *longer* than
/// the tree's dimension and relies on the truncation.
// `unused_unsafe` fires on the arithmetic intrinsics from Rust 1.87, which made them safe to call
// inside a matching `#[target_feature]` function. The crate's declared MSRV is 1.85, where they are
// still `unsafe fn` and the blocks are mandatory, so they stay and the lint is silenced here rather
// than the floor being raised for a warning. Remove this the day `rust-version` passes 1.87.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of a `__m256d`.
    ///
    /// # Safety
    /// Requires `avx`.
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_pd(v: __m256d) -> f64 {
        unsafe {
            let s = _mm_add_pd(_mm256_extractf128_pd(v, 1), _mm256_castpd256_pd128(v));
            _mm_cvtsd_f64(_mm_add_pd(s, _mm_unpackhi_pd(s, s)))
        }
    }

    /// Horizontal sum of a `__m256`.
    ///
    /// # Safety
    /// Requires `avx`.
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        unsafe {
            let s = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55)))
        }
    }

    /// Two-accumulator packed reduction, shared by all six kernels.
    ///
    /// `$lanes` elements per vector, `$step` = `2 * $lanes`. The tail after the packed loop is a
    /// scalar fold in the original order, so a short vector is exactly the old code.
    macro_rules! packed {
        ($name:ident, $ty:ty, $reg:ty, $lanes:expr,
         $zero:ident, $load:ident, $add:ident, $hsum:ident, $body:expr, $tail:expr) => {
            /// # Safety
            /// Requires `avx2` and `fma`; `n <= a.len()` and `n <= b.len()`.
            #[target_feature(enable = "avx2,fma")]
            pub(super) unsafe fn $name(a: &[$ty], b: &[$ty], n: usize) -> $ty {
                let (pa, pb) = (a.as_ptr(), b.as_ptr());
                let step = 2 * $lanes;
                let f: fn($reg, $reg, $reg) -> $reg = $body;
                unsafe {
                    let mut acc0 = $zero();
                    let mut acc1 = $zero();
                    let mut i = 0usize;
                    while i + step <= n {
                        acc0 = f($load(pa.add(i)), $load(pb.add(i)), acc0);
                        acc1 = f($load(pa.add(i + $lanes)), $load(pb.add(i + $lanes)), acc1);
                        i += step;
                    }
                    while i + $lanes <= n {
                        acc0 = f($load(pa.add(i)), $load(pb.add(i)), acc0);
                        i += $lanes;
                    }
                    let mut t = $hsum($add(acc0, acc1));
                    let g: fn($ty, $ty) -> $ty = $tail;
                    while i < n {
                        t += g(a[i], b[i]);
                        i += 1;
                    }
                    t
                }
            }
        };
    }

    packed!(
        sq_euclidean_f64,
        f64,
        __m256d,
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_add_pd,
        hsum_pd,
        |x, y, acc| unsafe {
            let d = _mm256_sub_pd(x, y);
            _mm256_fmadd_pd(d, d, acc)
        },
        |x, y| (x - y) * (x - y)
    );
    packed!(
        sq_euclidean_f32,
        f32,
        __m256,
        8,
        _mm256_setzero_ps,
        _mm256_loadu_ps,
        _mm256_add_ps,
        hsum_ps,
        |x, y, acc| unsafe {
            let d = _mm256_sub_ps(x, y);
            _mm256_fmadd_ps(d, d, acc)
        },
        |x, y| (x - y) * (x - y)
    );
    packed!(
        dot_f64,
        f64,
        __m256d,
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_add_pd,
        hsum_pd,
        |x, y, acc| unsafe { _mm256_fmadd_pd(x, y, acc) },
        |x, y| x * y
    );
    packed!(
        dot_f32,
        f32,
        __m256,
        8,
        _mm256_setzero_ps,
        _mm256_loadu_ps,
        _mm256_add_ps,
        hsum_ps,
        |x, y, acc| unsafe { _mm256_fmadd_ps(x, y, acc) },
        |x, y| x * y
    );
    packed!(
        manhattan_f64,
        f64,
        __m256d,
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_add_pd,
        hsum_pd,
        |x, y, acc| unsafe {
            // `andnot(-0.0, v)` clears the sign bit: `|v|` without a branch or a compare.
            _mm256_add_pd(
                acc,
                _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_sub_pd(x, y)),
            )
        },
        |x, y| (x - y).abs()
    );
    packed!(
        manhattan_f32,
        f32,
        __m256,
        8,
        _mm256_setzero_ps,
        _mm256_loadu_ps,
        _mm256_add_ps,
        hsum_ps,
        |x, y, acc| unsafe {
            _mm256_add_ps(
                acc,
                _mm256_andnot_ps(_mm256_set1_ps(-0.0), _mm256_sub_ps(x, y)),
            )
        },
        |x, y| (x - y).abs()
    );
}

/// The NEON kernels, the same six reductions for `aarch64`.
///
/// Two differences from the AVX2 module above, both from the architecture rather than from taste.
/// **There is no feature detection**: NEON (ASIMD) is architecturally mandatory on AArch64 and the
/// Rust target enables it, so there is no `is_*_feature_detected!` and no `#[target_feature]` — the
/// intrinsics are usable from an ordinary `#[inline]` function, which means these bodies can inline
/// into the caller instead of costing the out-of-line call the x86 path has to pay for. **The
/// vectors are half as wide**: `float64x2_t` is 2 lanes and `float32x4_t` is 4, against AVX2's 4 and
/// 8, so the same two-accumulator shape unrolls to 4 and 8 elements per iteration.
///
/// `vfmaq_*` is a fused multiply-add, so the rounding matches the x86 path's `_mm256_fmadd_*` in
/// kind though not in reduction order — as there, values from these kernels are compared and never
/// accumulated.
// The intrinsics are `unsafe fn` on the declared MSRV (1.85) and safe from 1.87; the blocks stay
// and the lint is silenced, exactly as in the `avx2` module.
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
mod neon {
    use std::arch::aarch64::*;

    macro_rules! packed {
        ($name:ident, $ty:ty, $reg:ty, $lanes:expr,
         $zero:expr, $load:ident, $add:ident, $hsum:ident, $body:expr, $tail:expr) => {
            /// # Safety
            /// `n <= a.len()` and `n <= b.len()`.
            #[inline]
            pub(super) unsafe fn $name(a: &[$ty], b: &[$ty], n: usize) -> $ty {
                let (pa, pb) = (a.as_ptr(), b.as_ptr());
                let step = 2 * $lanes;
                let f: fn($reg, $reg, $reg) -> $reg = $body;
                unsafe {
                    let mut acc0 = $zero;
                    let mut acc1 = $zero;
                    let mut i = 0usize;
                    while i + step <= n {
                        acc0 = f($load(pa.add(i)), $load(pb.add(i)), acc0);
                        acc1 = f($load(pa.add(i + $lanes)), $load(pb.add(i + $lanes)), acc1);
                        i += step;
                    }
                    while i + $lanes <= n {
                        acc0 = f($load(pa.add(i)), $load(pb.add(i)), acc0);
                        i += $lanes;
                    }
                    let mut t = $hsum($add(acc0, acc1));
                    let g: fn($ty, $ty) -> $ty = $tail;
                    while i < n {
                        t += g(a[i], b[i]);
                        i += 1;
                    }
                    t
                }
            }
        };
    }

    packed!(
        sq_euclidean_f64,
        f64,
        float64x2_t,
        2,
        vdupq_n_f64(0.0),
        vld1q_f64,
        vaddq_f64,
        vaddvq_f64,
        |x, y, acc| unsafe {
            let d = vsubq_f64(x, y);
            vfmaq_f64(acc, d, d)
        },
        |x, y| (x - y) * (x - y)
    );
    packed!(
        sq_euclidean_f32,
        f32,
        float32x4_t,
        4,
        vdupq_n_f32(0.0),
        vld1q_f32,
        vaddq_f32,
        vaddvq_f32,
        |x, y, acc| unsafe {
            let d = vsubq_f32(x, y);
            vfmaq_f32(acc, d, d)
        },
        |x, y| (x - y) * (x - y)
    );
    packed!(
        dot_f64,
        f64,
        float64x2_t,
        2,
        vdupq_n_f64(0.0),
        vld1q_f64,
        vaddq_f64,
        vaddvq_f64,
        |x, y, acc| unsafe { vfmaq_f64(acc, x, y) },
        |x, y| x * y
    );
    packed!(
        dot_f32,
        f32,
        float32x4_t,
        4,
        vdupq_n_f32(0.0),
        vld1q_f32,
        vaddq_f32,
        vaddvq_f32,
        |x, y, acc| unsafe { vfmaq_f32(acc, x, y) },
        |x, y| x * y
    );
    packed!(
        manhattan_f64,
        f64,
        float64x2_t,
        2,
        vdupq_n_f64(0.0),
        vld1q_f64,
        vaddq_f64,
        vaddvq_f64,
        // `vabsq` is a single instruction here, where x86 needs the `andnot` sign-bit trick.
        |x, y, acc| unsafe { vaddq_f64(acc, vabsq_f64(vsubq_f64(x, y))) },
        |x, y| (x - y).abs()
    );
    packed!(
        manhattan_f32,
        f32,
        float32x4_t,
        4,
        vdupq_n_f32(0.0),
        vld1q_f32,
        vaddq_f32,
        vaddvq_f32,
        |x, y, acc| unsafe { vaddq_f32(acc, vabsq_f32(vsubq_f32(x, y))) },
        |x, y| (x - y).abs()
    );
}

/// Smallest `d` at which the packed kernel is worth calling, per scalar type.
///
/// A `#[target_feature]` function cannot be inlined into a caller that does not carry the same
/// features, so taking the AVX2 path replaces an inlined loop with an out-of-line call plus a
/// horizontal sum. Below these widths that fixed cost exceeds everything the packing saves, and the
/// loss is not small: measured end-to-end on `kmeans`, `RAYON_NUM_THREADS=1`, an ungated dispatch
/// runs **1.55× slower** on 500 000 × 2 and 1.15× slower on 300 000 × 8, while winning 1.08× at
/// `d = 12`, 1.20× at 16 and 1.36× at 20. `f32` crosses over later in elements — its packed step is
/// twice as wide, so it needs twice the work to amortise the same fixed cost: 1.12× slower at
/// `d = 8`, 1.36× faster at 16.
///
/// Two-dimensional data is not a corner case here — the published scaling table is measured on
/// `d = 2` blobs, and geospatial input is a documented use.
///
/// **On `aarch64` this number is inherited, not measured.** NEON needs no `#[target_feature]`, so
/// the packed body inlines and the fixed cost the threshold exists to amortise is smaller there;
/// 16 is therefore a conservative floor rather than a crossover. Lowering it is a change to make
/// against a benchmark on the hardware (`cargo bench --bench kernels` prints the `d` sweep), not
/// from this side of a cross-compiler.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SIMD_MIN: usize = 16;

/// The out-of-line half of the dispatch: feature detection, the `TypeId` narrowing and the packed
/// call, for one operation.
///
/// **`#[inline(never)]` is the point of this function, not an afterthought.** Inlined, its bulk made
/// LLVM decline to inline the tiny public wrapper at all, and every distance call in the CF-tree
/// became a real call — measured 1.26× *slower* at `d = 8` than the ungated version it was supposed
/// to fix. Out of line, the wrapper stays small enough to inline, the scalar fold keeps the codegen
/// it had before this module grew, and the branch is one predictable compare.
///
/// `TypeId` equality is what makes the reinterpretation sound: `TypeId` is injective over `'static`
/// types, so a match proves `R` *is* that type and `&[R]` and `&[f64]` are the same layout. The
/// result comes back through `FromPrimitive`, which is exact and needs no `unsafe` at all.
#[cfg(target_arch = "x86_64")]
macro_rules! simd_path {
    ($name:ident, $k64:path, $k32:path) => {
        #[inline(never)]
        fn $name<R: Real>(a: &[R], b: &[R]) -> Option<R> {
            let n = a.len().min(b.len());
            if n < SIMD_MIN
                || !(std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma"))
            {
                return None;
            }
            if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f64>() {
                // SAFETY: `TypeId` proved `R == f64`, so the slices are `&[f64]`; `n` bounds both;
                // AVX2 and FMA were just detected.
                let v = unsafe {
                    $k64(
                        std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), a.len()),
                        std::slice::from_raw_parts(b.as_ptr().cast::<f64>(), b.len()),
                        n,
                    )
                };
                return R::from_f64(v);
            }
            if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f32>() {
                // SAFETY: as above, with `R == f32`.
                let v = unsafe {
                    $k32(
                        std::slice::from_raw_parts(a.as_ptr().cast::<f32>(), a.len()),
                        std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len()),
                        n,
                    )
                };
                return R::from_f32(v);
            }
            None
        }
    };
}

#[cfg(target_arch = "x86_64")]
simd_path!(
    simd_sq_euclidean,
    avx2::sq_euclidean_f64,
    avx2::sq_euclidean_f32
);
#[cfg(target_arch = "x86_64")]
simd_path!(simd_dot, avx2::dot_f64, avx2::dot_f32);
#[cfg(target_arch = "x86_64")]
simd_path!(simd_manhattan, avx2::manhattan_f64, avx2::manhattan_f32);

/// The `aarch64` half of the dispatch: the same `TypeId` narrowing, with no feature detection to do
/// and nothing to keep out of line, since NEON needs no `#[target_feature]` and can therefore inline.
#[cfg(target_arch = "aarch64")]
macro_rules! simd_path {
    ($name:ident, $k64:path, $k32:path) => {
        #[inline]
        fn $name<R: Real>(a: &[R], b: &[R]) -> Option<R> {
            let n = a.len().min(b.len());
            if n < SIMD_MIN {
                return None;
            }
            if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f64>() {
                // SAFETY: `TypeId` proved `R == f64`, so the slices are `&[f64]`; `n` bounds both.
                let v = unsafe {
                    $k64(
                        std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), a.len()),
                        std::slice::from_raw_parts(b.as_ptr().cast::<f64>(), b.len()),
                        n,
                    )
                };
                return R::from_f64(v);
            }
            if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f32>() {
                // SAFETY: as above, with `R == f32`.
                let v = unsafe {
                    $k32(
                        std::slice::from_raw_parts(a.as_ptr().cast::<f32>(), a.len()),
                        std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len()),
                        n,
                    )
                };
                return R::from_f32(v);
            }
            None
        }
    };
}

#[cfg(target_arch = "aarch64")]
simd_path!(
    simd_sq_euclidean,
    neon::sq_euclidean_f64,
    neon::sq_euclidean_f32
);
#[cfg(target_arch = "aarch64")]
simd_path!(simd_dot, neon::dot_f64, neon::dot_f32);
#[cfg(target_arch = "aarch64")]
simd_path!(simd_manhattan, neon::manhattan_f64, neon::manhattan_f32);

/// Take the packed path only when the vector is long enough to pay for the out-of-line call.
macro_rules! dispatch {
    ($a:expr, $b:expr, $simd:ident) => {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        // One compare against a constant, and nothing else, on the path a short vector takes: at
        // `d = 2` the scalar body is two multiply-adds, so even computing `min` here was measurable.
        if $b.len() >= SIMD_MIN {
            if let Some(v) = $simd($a, $b) {
                return v;
            }
        }
    };
}

/// Squared Euclidean distance `Σ (a_i − b_i)²`.
#[inline(always)]
pub fn sq_euclidean<R: Real>(a: &[R], b: &[R]) -> R {
    debug_assert_eq!(a.len(), b.len());
    dispatch!(a, b, simd_sq_euclidean);
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// Dot product `Σ a_i b_i`.
#[inline(always)]
pub fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    debug_assert_eq!(a.len(), b.len());
    dispatch!(a, b, simd_dot);
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Manhattan (L1) distance `Σ |a_i − b_i|`.
#[inline(always)]
pub fn manhattan<R: Real>(a: &[R], b: &[R]) -> R {
    debug_assert_eq!(a.len(), b.len());
    dispatch!(a, b, simd_manhattan);
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn kernels_match_naive_across_dims() {
        for d in [1usize, 2, 7, 8, 9, 16, 33, 64] {
            let a: Vec<f64> = (0..d).map(|i| (i as f64 * 0.3).sin()).collect();
            let b: Vec<f64> = (0..d).map(|i| (i as f64 * 0.7).cos()).collect();
            let sqe: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            let dt: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let mh: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
            assert!(close(sq_euclidean(&a, &b), sqe), "sq_euclidean d={d}");
            assert!(close(dot(&a, &b), dt), "dot d={d}");
            assert!(close(manhattan(&a, &b), mh), "manhattan d={d}");
        }
    }

    /// The packed loop consumes 8 (`f64`) or 16 (`f32`) elements at a time and leaves a scalar tail,
    /// so every residue class mod the step has to be exercised — a kernel that agrees at `d = 64`
    /// and drops the last three elements at `d = 67` is exactly the bug this catches. `f32` is
    /// checked at its own tolerance because its packed loop is 8 wide, not 4.
    #[test]
    fn the_packed_path_agrees_with_the_scalar_fold_at_every_tail_length() {
        for d in 0..70usize {
            let a64: Vec<f64> = (0..d).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
            let b64: Vec<f64> = (0..d).map(|i| (i as f64 * 0.91).cos() * 3.0).collect();
            let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
            let b32: Vec<f32> = b64.iter().map(|&v| v as f32).collect();

            let sqe: f64 = a64.iter().zip(&b64).map(|(x, y)| (x - y) * (x - y)).sum();
            let dt: f64 = a64.iter().zip(&b64).map(|(x, y)| x * y).sum();
            let mh: f64 = a64.iter().zip(&b64).map(|(x, y)| (x - y).abs()).sum();
            assert!(
                (sq_euclidean(&a64, &b64) - sqe).abs() < 1e-12 * (1.0 + sqe.abs()),
                "sqe d={d}"
            );
            assert!(
                (dot(&a64, &b64) - dt).abs() < 1e-12 * (1.0 + dt.abs()),
                "dot d={d}"
            );
            assert!(
                (manhattan(&a64, &b64) - mh).abs() < 1e-12 * (1.0 + mh.abs()),
                "l1 d={d}"
            );

            let s32 = sq_euclidean(&a32, &b32) as f64;
            let d32 = dot(&a32, &b32) as f64;
            let m32 = manhattan(&a32, &b32) as f64;
            assert!(
                (s32 - sqe).abs() < 1e-4 * (1.0 + sqe.abs()),
                "sqe f32 d={d}"
            );
            assert!((d32 - dt).abs() < 1e-4 * (1.0 + dt.abs()), "dot f32 d={d}");
            assert!((m32 - mh).abs() < 1e-4 * (1.0 + mh.abs()), "l1 f32 d={d}");
        }
    }

    /// The length gate is a performance switch, so it must be invisible in the answer. Below
    /// `SIMD_MIN` the public function takes the scalar fold, which the sweep above already checks;
    /// what needs its own test is that the packed kernel the gate *skips* would have agreed anyway —
    /// otherwise a future change to the constant would silently move results.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_kernel_the_length_gate_skips_would_have_agreed_with_the_scalar_fold() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        for d in 0..SIMD_MIN {
            let a: Vec<f64> = (0..d).map(|i| (i as f64 * 0.41).sin() * 2.0).collect();
            let b: Vec<f64> = (0..d).map(|i| (i as f64 * 0.83).cos() * 2.0).collect();
            let want: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            // SAFETY: features detected above; `n = d` bounds both slices.
            let got = unsafe { avx2::sq_euclidean_f64(&a, &b, d) };
            assert!(close(got, want), "d={d}: packed {got}, scalar {want}");
            assert!(close(sq_euclidean(&a, &b), want), "d={d} through the gate");
        }
    }

    /// `n` is the soundness-critical argument of the packed kernels: it is the only thing standing
    /// between an unequal-length call and an out-of-bounds `loadu`. The public wrappers set it to
    /// `a.len().min(b.len())`, which is what `zip` does; this pins that the kernel then reads
    /// exactly `n` and stops, rather than running to the length of the longer slice. A debug build
    /// cannot reach this through the public API — `debug_assert_eq!` rejects unequal lengths there —
    /// which is precisely why the guard needs a test of its own.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_packed_kernel_reads_exactly_n_elements_and_not_the_longer_slice() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let a: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let long: Vec<f64> = (0..40).map(|i| (i as f64) * 0.5).collect();
        let want: f64 = a.iter().zip(&long).map(|(x, y)| (x - y) * (x - y)).sum();
        // SAFETY: features detected above; `n = 12` bounds both slices.
        let got = unsafe { avx2::sq_euclidean_f64(&a, &long, a.len()) };
        assert!(close(got, want), "got {got}, want {want}");
        // SAFETY: as above, with the long slice first.
        let swapped = unsafe { avx2::sq_euclidean_f64(&long, &a, a.len()) };
        assert!(close(swapped, want), "got {swapped}, want {want}");
    }
}
