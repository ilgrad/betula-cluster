//! What a canonical insertion order costs, split into the three things it actually pays for.
//!
//! `canonical_order=True` measures 0.83–1.33× an arrival-order fit through the Python binding, which
//! is one number covering three different mechanisms with three different fixes:
//!
//! 1. **The key** — one `n × d × PROJECTIONS` projection pass. A dense GEMM written as a triple loop
//!    in `order::morton_codes`, not routed through the SIMD kernels in `kernels.rs`. If this
//!    dominates, the fix is those kernels.
//! 2. **The sort** — `O(n log n)` over `u64` codes, with a row comparison only on a collision. If
//!    *this* dominates the collision rate is too high and the fix is more bits, not more speed.
//! 3. **The insert** — the same work either way, but walking `X` through a permutation instead of
//!    front to back. Insertion is memory-bound, so this is the term that can only get worse; the
//!    offset is that spatially coherent inserts split less, which shows up as fewer rebuilds.
//!
//! `cargo bench --bench canonical_order`. Each row is the median of [`ROUNDS`] repetitions, and the
//! arrival-order build is re-timed in the same loop rather than quoted from `tree_insert`, so the
//! ratio is not a comparison across two runs of the machine.

use betula_cluster::distance::{CentroidEuclidean, Radius};
use betula_cluster::feature::Spherical;
use betula_cluster::order::canonical_permutation;
use betula_cluster::tree::CFTree;
use std::hint::black_box;
use std::time::{Duration, Instant};

const ROUNDS: usize = 5;

/// `k` well-separated Gaussian blobs, flat row-major — deterministic, so every arm sees one dataset.
fn blobs(n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let centres: Vec<f64> = (0..k * d).map(|_| (next() - 0.5) * 12.0).collect();
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = (i * 2_654_435_761) % k;
        for j in 0..d {
            let (u1, u2) = (next().max(1e-12), next());
            let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            out.push(centres[c * d + j] + g);
        }
    }
    out
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

fn build(
    data: &[f64],
    n: usize,
    d: usize,
    max_leaves: usize,
    order: Option<&[u32]>,
) -> (usize, usize) {
    let mut tree: CFTree<f64, Spherical<f64>, CentroidEuclidean, Radius> =
        CFTree::new(d, 50, 50, 0.0, max_leaves, CentroidEuclidean, Radius);
    for rank in 0..n {
        let i = order.map_or(rank, |o| o[rank] as usize);
        tree.insert(black_box(&data[i * d..(i + 1) * d]));
    }
    (tree.leaf_features().len(), tree.rebuilds())
}

fn bench(n: usize, d: usize, max_leaves: usize) {
    let data = blobs(n, d, 10, 0x243F_6A88);
    let (mut arrival, mut key, mut ordered) = (vec![], vec![], vec![]);
    let (mut rb_a, mut rb_c) = (0, 0);

    for _ in 0..ROUNDS {
        // A-B-A-B within the round: an arrival build and a canonical build see the same thermal
        // state, which quoting a number from another benchmark run cannot promise.
        let t0 = Instant::now();
        let (_, r) = build(&data, n, d, max_leaves, None);
        arrival.push(t0.elapsed());
        rb_a = r;

        let t0 = Instant::now();
        let perm = canonical_permutation(&data, n, d);
        key.push(t0.elapsed());

        let t0 = Instant::now();
        let (_, r) = build(&data, n, d, max_leaves, Some(black_box(&perm)));
        ordered.push(t0.elapsed());
        rb_c = r;
        black_box(&perm);
    }

    let (a, k, o) = (
        median(arrival).as_secs_f64(),
        median(key).as_secs_f64(),
        median(ordered).as_secs_f64(),
    );
    println!(
        "n={n:<8} d={d:<5} ml={max_leaves:<6} arrival {a:>7.3} s | key {k:>7.3} s ({:>5.1} %) \
         insert {o:>7.3} s ({:>5.2}x) | total {:>5.2}x | rebuilds {rb_a} -> {rb_c}",
        100.0 * k / a,
        o / a,
        (k + o) / a,
    );
}

fn main() {
    println!("# median of {ROUNDS} A-B repetitions, single-threaded");
    println!("# 'key' is order::canonical_permutation; 'insert' is the build walking that order");
    for &(n, d) in &[(200_000usize, 20usize), (200_000, 128), (50_000, 784)] {
        for &ml in &[2_000usize, 8_000] {
            bench(n, d, ml);
        }
    }
}
