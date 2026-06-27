//! Vector distance kernels — small, `#[inline]` autovectorizable reductions.
//!
//! These run in the CF-tree's hot path (millions of small-`d` distances per build), so they are
//! plain inlinable loops the compiler vectorizes at each call site. A `multiversion` runtime-SIMD
//! dispatcher was measured (release, AVX2) to be slower for small `d` — its indirect call cannot
//! inline and the per-call dispatch dominates — and no faster for high `d`, where the bottleneck is
//! the GMM linear algebra (`O(d³)` Cholesky), not these reductions. So they stay inline.

use crate::types::Real;

/// Squared Euclidean distance `Σ (a_i − b_i)²`.
#[inline]
pub fn sq_euclidean<R: Real>(a: &[R], b: &[R]) -> R {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// Dot product `Σ a_i b_i`.
#[inline]
pub fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Manhattan (L1) distance `Σ |a_i − b_i|`.
#[inline]
pub fn manhattan<R: Real>(a: &[R], b: &[R]) -> R {
    debug_assert_eq!(a.len(), b.len());
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
}
