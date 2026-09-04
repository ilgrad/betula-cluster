//! The insertion path, which is 90–99 % of a fit at scale.
//!
//! Measured through the Rust API rather than the Python binding so a change to the tree layout can
//! be timed without a `maturin develop` in between, and with no interpreter in the sample. Run with
//! `cargo bench --bench tree_insert`; each row is the median of [`ROUNDS`] builds.
//!
//! Rows/s is the number to move. `descend` cost is `depth × branching × d` and every step chases a
//! pointer into a separately allocated `Vec`, so this is a memory-latency benchmark that happens to
//! be written in floating point.

use betula_cluster::distance::{CentroidEuclidean, Radius};
use betula_cluster::feature::Spherical;
use betula_cluster::tree::CFTree;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Timed builds per row; the report is their median.
const ROUNDS: usize = 5;

/// `k` well-separated Gaussian blobs, flat row-major — deterministic, so two builds see one dataset.
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
            // Box-Muller on the same stream, so the noise is Gaussian rather than uniform: the
            // absorption gate is a radius test and uniform noise fills a box, not a ball.
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

fn bench(n: usize, d: usize, max_leaves: usize) {
    let data = blobs(n, d, 10, 0x243F_6A88);
    let mut rounds = Vec::with_capacity(ROUNDS);
    let mut leaves = 0;
    let mut rebuilds = 0;
    for _ in 0..ROUNDS {
        let mut tree: CFTree<f64, Spherical<f64>, CentroidEuclidean, Radius> =
            CFTree::new(d, 50, 50, 0.0, max_leaves, CentroidEuclidean, Radius);
        let t0 = Instant::now();
        for i in 0..n {
            tree.insert(black_box(&data[i * d..(i + 1) * d]));
        }
        rounds.push(t0.elapsed());
        leaves = tree.leaf_features().len();
        rebuilds = tree.rebuilds();
        black_box(&tree);
    }
    let dt = median(rounds).as_secs_f64();
    println!(
        "insert n={n:<8} d={d:<5} max_leaves={max_leaves:<6} {dt:>8.3} s  \
         {:>10.0} rows/s  leaves={leaves} rebuilds={rebuilds}",
        n as f64 / dt,
    );
}

fn main() {
    println!("# median of {ROUNDS} builds, single-threaded");
    for &(n, d) in &[(200_000usize, 20usize), (200_000, 128), (50_000, 784)] {
        for &ml in &[2_000usize, 8_000] {
            bench(n, d, ml);
        }
    }
}
