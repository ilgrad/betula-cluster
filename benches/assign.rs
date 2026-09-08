//! Pruned assignment against a full scan, across the `(d, k)` grid the library is run at.
//!
//! `cargo bench --bench assign`. Same conventions as `benches/kernels.rs`: no harness crate,
//! `Instant` and `black_box`, the median of [`ROUNDS`] timed batches after one warm-up.
//!
//! Two numbers per row and they answer different questions. **Dimensions read** is the algorithmic
//! quantity — how much of the `n × k × d` scan the bound removed, hardware-independent, the one to
//! compare against `local/scratch/skm_prefix.py`. **Seconds** is what a caller feels, and it lags
//! the read count because a pruned scan trades sequential reads for a comparison per stage per
//! surviving candidate, and because the full scan runs the AVX2 kernel over the whole row with no
//! branch in it at all. A row where the reads fall and the time does not is the honest outcome to
//! publish, not a failure to hide.
//!
//! The data is a Gaussian mixture at a separation that puts the distance contrast in the range the
//! bench datasets sit in; clusters that are too far apart make any bound look good.

use betula_cluster::assign::AssignPlan;
use betula_cluster::kernels::sq_euclidean;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Timed batches per row; the report is their median.
const ROUNDS: usize = 5;
/// Points assigned per batch.
const POINTS: usize = 4096;

/// Deterministic pseudo-random stream — a benchmark that reads the OS entropy pool measures the OS.
struct Xorshift(u64);

impl Xorshift {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// `k` centroids and `n` points from a mixture around them, `spread` between the centres.
fn mixture(n: usize, dim: usize, k: usize, spread: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Xorshift(seed | 1);
    let centroids: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..dim).map(|_| rng.gauss() * spread).collect())
        .collect();
    let mut flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centroids[i % k];
        flat.extend(c.iter().map(|&v| v + rng.gauss()));
    }
    (centroids, flat)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Full scan over every centroid — the reference the plan must agree with.
fn brute(flat: &[f64], n: usize, dim: usize, centroids: &[Vec<f64>], out: &mut [u32]) {
    for i in 0..n {
        let x = &flat[i * dim..(i + 1) * dim];
        let mut best = 0usize;
        let mut bd = sq_euclidean(x, &centroids[0]);
        for (j, c) in centroids.iter().enumerate().skip(1) {
            let d = sq_euclidean(x, c);
            if d < bd {
                bd = d;
                best = j;
            }
        }
        out[i] = best as u32;
    }
}

fn timed(mut body: impl FnMut()) -> Duration {
    let mut rounds = Vec::with_capacity(ROUNDS);
    for r in 0..ROUNDS + 1 {
        let t0 = Instant::now();
        body();
        let dt = t0.elapsed();
        if r > 0 {
            rounds.push(dt); // round 0 warms the buffers
        }
    }
    median(rounds)
}

fn row(dim: usize, k: usize, spread: f64) {
    let n = POINTS;
    let (centroids, flat) = mixture(n, dim, k, spread, 0x9E37_79B9);
    let plan = AssignPlan::new(&centroids, None).expect("centroids are well formed");

    let mut want = vec![0u32; n];
    let t_brute = timed(|| brute(black_box(&flat), n, dim, &centroids, &mut want));

    let mut got = vec![0u32; n];
    let mut stats = Default::default();
    let t_plan = timed(|| stats = plan.assign_many(black_box(&flat), n, None, &mut got));
    assert_eq!(got, want, "pruned assignment disagreed with the full scan");

    // A second pass with the first pass's labels as hints — the Lloyd case, where the threshold
    // starts at the previous distance instead of infinity.
    let mut got2 = vec![0u32; n];
    let mut stats_h = Default::default();
    let t_hint = timed(|| stats_h = plan.assign_many(black_box(&flat), n, Some(&want), &mut got2));
    assert_eq!(got2, want, "hinted assignment disagreed with the full scan");

    // The same bound with the dimensions left alone: the difference against the rows above is
    // exactly what the per-point gather into the variance order costs and buys.
    let flat_plan = AssignPlan::identity_order(&centroids).expect("centroids are well formed");
    let mut got3 = vec![0u32; n];
    let mut stats_i = Default::default();
    let t_ident =
        timed(|| stats_i = flat_plan.assign_many(black_box(&flat), n, Some(&want), &mut got3));
    assert_eq!(
        got3, want,
        "identity-order assignment disagreed with the full scan"
    );

    let us = |d: Duration| d.as_secs_f64() * 1e6 / n as f64;
    println!(
        "d={dim:<5} k={k:<5} | read {:>5.1} % hint {:>5.1} % ident {:>5.1} % \
         | us/pt brute {:>8.2} plan {:>8.2} hint {:>8.2} ident {:>8.2} \
         | speedup hint {:>5.2}x ident {:>5.2}x",
        100.0 * stats.read_fraction(),
        100.0 * stats_h.read_fraction(),
        100.0 * stats_i.read_fraction(),
        us(t_brute),
        us(t_plan),
        us(t_hint),
        us(t_ident),
        t_brute.as_secs_f64() / t_hint.as_secs_f64(),
        t_brute.as_secs_f64() / t_ident.as_secs_f64(),
    );
}

fn main() {
    println!("# median of {ROUNDS} batches of {POINTS} points, warm buffers, single-threaded");
    println!("# spread 1.5 = interleaved clusters, the regime the bench datasets sit in");
    for &dim in &[20usize, 64, 128, 784, 1024] {
        for &k in &[10usize, 100, 1000] {
            row(dim, k, 1.5);
        }
    }
    println!("# spread 4.0 = well-separated clusters, the best case for any bound");
    for &dim in &[64usize, 784] {
        for &k in &[10usize, 100] {
            row(dim, k, 4.0);
        }
    }
}
