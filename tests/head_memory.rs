//! Peak heap of the mixture heads that read a per-leaf covariance, counted by a global allocator.
//!
//! The allocator is per binary, which is why this is a test binary of its own, with one test: a
//! second test running in parallel would add its allocations to the same counter.
//!
//! On one-row `Spherical` or `Diagonal` leaves every within-leaf covariance is a diagonal, and the
//! heads used to ask each leaf for it as a dense `d × d` matrix, once per parallel restart: 67 MB
//! at one thread and 257 MB at eight for `gmm_full` on the shape below, and as much for `mppca` and
//! `mfa`. The bound is a quarter of one dense copy, so it fails at any thread count unless no
//! leaf's covariance is dense.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use betula_cluster::clustering::gmm::gmm_full;
use betula_cluster::clustering::mfa::mfa;
use betula_cluster::clustering::mppca::mppca;
use betula_cluster::feature::{ClusterFeature, Diagonal, Spherical};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(n: usize) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's layout and only adds bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grow(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grow(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                grow(new - l.size());
            } else {
                LIVE.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Bytes requested above the live heap at the call, at the call's peak.
fn peak_of(f: &dyn Fn()) -> usize {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    f();
    PEAK.load(Relaxed) - base
}

/// `m` one-row leaves of `d` coordinates around ten centres: the shape of `digits` at one row per
/// leaf, which is what a threshold-0 tree gives on it.
fn one_row_leaves<C: ClusterFeature<f64>>(m: usize, d: usize) -> Vec<C> {
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let centres: Vec<Vec<f64>> = (0..10)
        .map(|_| (0..d).map(|_| 8.0 * next()).collect())
        .collect();
    (0..m)
        .map(|i| {
            let x: Vec<f64> = centres[i % 10].iter().map(|&c| c + next()).collect();
            let mut f = C::new(d);
            f.push(&x, 1.0);
            f
        })
        .collect()
}

#[test]
fn the_covariance_heads_hold_no_dense_matrix_for_a_diagonal_leaf() {
    let (m, d, k) = (1_800, 64, 10);
    let bound = m * d * d * 8 / 4;
    let check = |name: &str, fit: &dyn Fn()| {
        let peak = peak_of(fit);
        assert!(
            peak < bound,
            "{name}: {peak} bytes at the peak, against a bound of {bound} (one dense copy is {})",
            m * d * d * 8
        );
    };
    let spherical: Vec<Spherical<f64>> = one_row_leaves(m, d);
    let diagonal: Vec<Diagonal<f64>> = one_row_leaves(m, d);
    check("gmm_full, spherical", &|| {
        drop(gmm_full(&spherical, k, 3, 0))
    });
    check("gmm_full, diagonal", &|| drop(gmm_full(&diagonal, k, 3, 0)));
    check("mppca, spherical", &|| drop(mppca(&spherical, k, 2, 3, 0)));
    check("mppca, diagonal", &|| drop(mppca(&diagonal, k, 2, 3, 0)));
    check("mfa, spherical", &|| drop(mfa(&spherical, k, 2, 3, 0)));
    check("mfa, diagonal", &|| drop(mfa(&diagonal, k, 2, 3, 0)));
}
