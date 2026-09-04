//! Distance kernels across the dimensions the library is actually run at.
//!
//! No harness crate: `std::time::Instant` and `std::hint::black_box` measure this accurately enough
//! to compare two builds of the same kernel, which is all this is for, and a benchmark framework in
//! `[dev-dependencies]` is a dependency the release build has to justify. Run with
//! `cargo bench --bench kernels`; each row is the median of [`ROUNDS`] timed batches.
//!
//! The `d` sweep is the point. A kernel that wins at `d = 784` and loses at `d = 20` has not been
//! measured until both are on the table — the AVX2 path starts at 16 lanes and the tree spends most
//! of its life on small vectors.

use betula_cluster::kernels::{dot, manhattan, sq_euclidean};
use betula_cluster::types::Real;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Timed batches per row; the report is their median, so one scheduler hiccup cannot set the number.
const ROUNDS: usize = 9;
/// Vector pairs per batch, sized so a batch is milliseconds rather than nanoseconds.
const PAIRS: usize = 1 << 12;

/// A deterministic pseudo-random buffer — a benchmark that reads from the OS entropy pool is a
/// benchmark of the OS entropy pool.
fn buffer<R: Real>(len: usize, seed: u64) -> Vec<R> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            R::from_f64((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5).unwrap()
        })
        .collect()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// One kernel at one width: `PAIRS` calls per batch, `ROUNDS` batches, median reported.
fn bench<R: Real>(name: &str, d: usize, kernel: fn(&[R], &[R]) -> R) {
    let a: Vec<R> = buffer(d * PAIRS, 0x9E37_79B9);
    let b: Vec<R> = buffer(d * PAIRS, 0xBF58_476D);
    let mut rounds = Vec::with_capacity(ROUNDS);
    for r in 0..ROUNDS + 1 {
        let t0 = Instant::now();
        let mut acc = R::zero();
        for i in 0..PAIRS {
            let (x, y) = (&a[i * d..(i + 1) * d], &b[i * d..(i + 1) * d]);
            acc = acc + kernel(black_box(x), black_box(y));
        }
        let dt = t0.elapsed();
        black_box(acc);
        if r > 0 {
            rounds.push(dt); // round 0 is the warm-up: first touch of both buffers
        }
    }
    let per_call = median(rounds).as_secs_f64() / PAIRS as f64;
    let width = std::mem::size_of::<R>();
    println!(
        "{name:<16} d={d:<5} f{:<3} {:>9.2} ns/call {:>8.2} GB/s",
        width * 8,
        per_call * 1e9,
        (2 * d * width) as f64 / per_call / 1e9,
    );
}

fn main() {
    println!("# median of {ROUNDS} batches of {PAIRS} calls, warm buffers");
    for &d in &[8usize, 16, 20, 32, 64, 128, 784, 1024] {
        bench::<f64>("sq_euclidean", d, sq_euclidean::<f64>);
        bench::<f32>("sq_euclidean", d, sq_euclidean::<f32>);
        bench::<f64>("dot", d, dot::<f64>);
        bench::<f64>("manhattan", d, manhattan::<f64>);
    }
}
