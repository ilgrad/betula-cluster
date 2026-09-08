//! Pruned **exact** nearest-centroid assignment.
//!
//! Assignment is `n × k × d` scalar reads and it is where the centroid heads spend their time on
//! the point side: `predict` on the mixture and centroid rules, and the per-node child scan in the
//! tree descent. This module cuts the reads without moving a single label.
//!
//! # The bound
//!
//! A squared Euclidean distance is a sum of non-negative terms, so any prefix of it is a lower
//! bound on the whole:
//!
//! ```text
//! Σ_{i<p} (x_i − c_i)²  ≤  Σ_{i<d} (x_i − c_i)²   for every p ≤ d
//! ```
//!
//! and a candidate whose prefix already exceeds the best distance found so far cannot win. The
//! bound holds in floating point as well as in exact arithmetic, because a running sum of
//! non-negative terms is non-decreasing under round-to-nearest. It does **not** hold for the
//! expanded form `‖x‖² + ‖c‖² − 2x·c` accumulated blockwise, which is a difference of large
//! numbers and can move either way; that form is what makes a stage a matrix multiply, and it is
//! why this module does not use one.
//!
//! Two things decide how much the bound prunes: the threshold it is compared against, and how much
//! of the distance the early dimensions carry.
//!
//! * **Threshold.** The tighter the incumbent, the more it prunes. Callers that have a previous
//!   label (a Lloyd iteration, a re-`predict` of the same rows) pass it as a hint and start from
//!   its distance instead of infinity.
//! * **Dimension order.** Free at the algebra level, because the bound is order-independent.
//!   Ordering by descending between-centroid variance maximises the expected prefix, which is the
//!   rearrangement inequality applied to the per-dimension contributions. Measured on 2 000 rows
//!   against the exact argmin (`local/scratch/skm_prefix.py`, 2026-09-08), mean dimensions read
//!   per (point, centroid) pair:
//!
//! ```text
//!                        natural order   variance order
//!   mnist   d=784 k=10          66.6 %           42.7 %
//!   mnist   d=784 k=100         43.7 %           29.6 %
//!   covtype d=54  k=10          47.8 %           33.2 %
//!   digits  d=64  k=10          52.2 %           41.6 %
//! ```
//!
//! # What it is worth in seconds
//!
//! Fewer reads is not automatically less time, and on this machine it often is not. Reordering the
//! dimensions means gathering each point into the plan's order — a copy of `dim` scalars per point,
//! amortised over `k` candidates — and a staged scan replaces one straight-line AVX2 pass over the
//! row with a comparison per stage per survivor. `cargo bench --bench assign`, Ryzen 7 5800HS,
//! single-threaded, 4096 points, clusters interleaved (spread 1.5), assignment with hints against a
//! full scan:
//!
//! ```text
//!                 k=10    k=100   k=1000        k=10    k=100   k=1000
//!   d=20         0.51x    0.32x    0.32x       0.40x    0.28x    0.32x
//!   d=64         0.76x    0.53x    0.64x       0.70x    0.51x    0.65x
//!   d=128        0.60x    1.05x    1.10x       1.09x    1.12x    0.95x
//!   d=784        0.55x    2.11x    2.04x       1.17x    1.77x    2.08x
//!   d=1024       0.60x    1.98x    8.20x       1.27x    2.01x    7.91x
//!               |------ variance order ------|------ identity order ------|
//! ```
//!
//! Three things to take from it, and they are what a caller should decide on:
//!
//! * **Below `d ≈ 128` the bound loses**, whatever the ordering. The full-row kernel has no branch
//!   in it and the rows are short enough that pruning cannot pay for the control flow. Use a plain
//!   scan there.
//! * **From `d ≈ 784` with a hint it wins 2–8×**, and the win grows with `k` because the per-point
//!   gather is amortised while the pruning saving scales with `k × dim`.
//! * **The ordering is a `k` decision, not a `d` one.** At `k = 10` the gather costs more than the
//!   extra pruning buys and [`AssignPlan::identity_order`] is faster; from `k = 100` the variance
//!   order pays for itself. Neither changes a label, so this is a pure cost choice.
//!
//! Without a hint the first candidate faces an infinite threshold and the pass reads 45–60 % of the
//! dimensions rather than 20 %; the module is built for the case where a previous label exists — a
//! Lloyd iteration, a re-`predict` of stored rows — which is the case that matters.
//!
//! # Why not ADSampling
//!
//! SuperKMeans (arXiv 2603.20009) prunes harder by rotating the data and testing a *statistical*
//! bound: after a Haar rotation the fraction of the squared norm in the first `p` coordinates is
//! `Beta(p/2, (d−p)/2)` (verified symbolically in `local/scratch/skm_math.mac`), so a threshold
//! scaled by `(p/d)(1 + ε₀/√p)²` prunes at a known error rate. On mnist at `k = 10` that reads
//! 34.1 % of the dimensions at a zero measured mislabel rate against this module's 42.7 %.
//!
//! It is not used here, for one reason: the rotation costs `d²` per point, and at the `k` a CF
//! summary is clustered at that is more than the assignment it accelerates — `k ≳ 0.6 d` before it
//! pays back over ten iterations. SuperKMeans's own code agrees, disabling the path below
//! `d = 128` or `k ≤ 256` (`common.h:99-100`). The exact bound needs no rotation, no error budget
//! and no parameter, and it cannot change a label.

use crate::types::Real;

/// Dimensions added between two checks, and where the first check falls.
///
/// A check costs a comparison per surviving candidate, so blocks that are too small spend more on
/// checking than they save; 64 matches the block the PDX layout is built around, and below
/// `2 × 64` it scales down so that low-dimensional data gets more than one stage.
fn default_block(dim: usize) -> usize {
    if dim >= 128 { 64 } else { (dim / 8).max(1) }
}

/// A set of centroids arranged for pruned assignment.
///
/// Built once per centroid set and reused across every point assigned to it. The centroids are
/// stored permuted into the plan's dimension order, so the inner loop reads each candidate's
/// stage contiguously.
#[derive(Debug, Clone)]
pub struct AssignPlan<R: Real> {
    dim: usize,
    k: usize,
    /// Dimension permutation: `order[j]` is the original index of the plan's `j`-th dimension.
    order: Vec<u32>,
    /// Cumulative dimension counts at which a check happens; the last entry is always `dim`.
    stages: Vec<usize>,
    /// `order` is `0..dim`, so a point can be read where it lies instead of being gathered.
    identity: bool,
    /// Row-major `k × dim`, permuted into `order`.
    centroids: Vec<R>,
}

/// What an assignment pass touched — for benchmarks and for the `d′` tuning a caller may want.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AssignStats {
    /// Scalar coordinates read across all (point, candidate) pairs.
    pub dims_read: u64,
    /// Coordinates a full scan would have read, `n × k × dim`.
    pub dims_total: u64,
}

impl AssignStats {
    /// Share of the full scan actually read, in `[0, 1]`; `1.0` when nothing was assigned.
    pub fn read_fraction(&self) -> f64 {
        if self.dims_total == 0 {
            1.0
        } else {
            self.dims_read as f64 / self.dims_total as f64
        }
    }
}

/// The centroid set was empty, or its rows disagreed on the dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    /// No centroids were given; there is nothing to assign to.
    Empty,
    /// Row `row` has length `len` where the first row has length `dim`.
    Ragged { row: usize, len: usize, dim: usize },
    /// The centroids have zero dimensions.
    ZeroDim,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "no centroids to assign to"),
            Self::Ragged { row, len, dim } => {
                write!(f, "centroid {row} has {len} dimensions, expected {dim}")
            }
            Self::ZeroDim => write!(f, "centroids have zero dimensions"),
        }
    }
}

impl std::error::Error for PlanError {}

impl<R: Real> AssignPlan<R> {
    /// Build a plan over `centroids`, ordering dimensions by descending between-centroid variance.
    ///
    /// `weights` are the cluster masses, used to weight that variance; `None` weights them equally.
    /// A weight that is not finite, or a set summing to zero, falls back to equal weighting rather
    /// than producing a meaningless order.
    pub fn new(centroids: &[Vec<R>], weights: Option<&[R]>) -> Result<Self, PlanError> {
        let dim = centroids.first().ok_or(PlanError::Empty)?.len();
        if dim == 0 {
            return Err(PlanError::ZeroDim);
        }
        for (row, c) in centroids.iter().enumerate() {
            if c.len() != dim {
                return Err(PlanError::Ragged {
                    row,
                    len: c.len(),
                    dim,
                });
            }
        }
        let block = default_block(dim);
        Ok(Self::with_order(
            centroids,
            variance_order(centroids, weights, dim),
            block,
            dim,
        ))
    }

    /// Build a plan with the dimensions left in their original order.
    ///
    /// It prunes less than [`AssignPlan::new`] on every dataset measured, and it is nonetheless
    /// the faster of the two at small `k`, because a point needs no gathering: see the timing
    /// table in the module documentation for the crossover, which sits near `k = 100`.
    pub fn identity_order(centroids: &[Vec<R>]) -> Result<Self, PlanError> {
        let dim = centroids.first().ok_or(PlanError::Empty)?.len();
        if dim == 0 {
            return Err(PlanError::ZeroDim);
        }
        for (row, c) in centroids.iter().enumerate() {
            if c.len() != dim {
                return Err(PlanError::Ragged {
                    row,
                    len: c.len(),
                    dim,
                });
            }
        }
        let block = default_block(dim);
        Ok(Self::with_order(
            centroids,
            (0..dim as u32).collect(),
            block,
            dim,
        ))
    }

    fn with_order(centroids: &[Vec<R>], order: Vec<u32>, block: usize, dim: usize) -> Self {
        let k = centroids.len();
        let mut flat = Vec::with_capacity(k * dim);
        for c in centroids {
            flat.extend(order.iter().map(|&j| c[j as usize]));
        }
        let mut stages = Vec::new();
        let mut at = block.min(dim);
        while at < dim {
            stages.push(at);
            at += block;
        }
        stages.push(dim);
        let identity = order.iter().enumerate().all(|(i, &j)| i as u32 == j);
        Self {
            dim,
            k,
            order,
            stages,
            identity,
            centroids: flat,
        }
    }

    /// Number of centroids.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Dimension of the space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The dimension permutation the plan reads in, most discriminating first.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Nearest centroid to `x`, and its squared distance.
    ///
    /// `hint` is a candidate to seed the pruning threshold with — the row's label from a previous
    /// iteration, typically. A wrong hint costs one extra full distance and changes nothing else;
    /// a hint outside `0..k` is ignored.
    ///
    /// The result is the argmin over the plan's own distances, ties broken by the lower index —
    /// the same rule an ordered full scan applies. Those distances are summed in the plan's
    /// dimension order and in stages, which is a different rounding from one serial pass over the
    /// original order, so a value can differ from `kernels::sq_euclidean` by an ulp and an argmin
    /// decided by less than that could in principle flip. Pruning itself cannot cause it: every
    /// comparison is between two numbers accumulated the same way. This is the caveat
    /// [`crate::kernels`] already carries for its SIMD paths, for the same reason.
    ///
    /// # Panics
    /// If `x.len() != self.dim()`.
    pub fn nearest(&self, x: &[R], hint: Option<usize>) -> (usize, R) {
        assert_eq!(
            x.len(),
            self.dim,
            "point has {} dimensions, plan has {}",
            x.len(),
            self.dim
        );
        let (label, dist, _) = if self.identity {
            self.nearest_permuted(x, hint)
        } else {
            let mut point = Vec::with_capacity(self.dim);
            point.extend(self.order.iter().map(|&j| x[j as usize]));
            self.nearest_permuted(&point, hint)
        };
        (label, dist)
    }

    /// `nearest` for a point already permuted into [`AssignPlan::order`]; also returns the reads.
    fn nearest_permuted(&self, p: &[R], hint: Option<usize>) -> (usize, R, u64) {
        debug_assert_eq!(p.len(), self.dim);
        let mut reads: u64 = 0;
        let mut best = usize::MAX;
        let mut best_dist = R::infinity();

        // The hint is evaluated first and in full, so its distance becomes the threshold every
        // other candidate is pruned against. It has to be accumulated stage by stage like the
        // rest: a full-length `sq_euclidean` sums in a different order and can land an ulp below
        // the staged sum, which would make the hint prune *itself* and, when it was the true
        // nearest, leave the point with no label at all.
        if let Some(h) = hint.filter(|&h| h < self.k) {
            best_dist = self.staged_distance(p, h);
            best = h;
            reads += self.dim as u64;
        }

        for j in 0..self.k {
            if j == best {
                continue; // already evaluated as the hint
            }
            let row = &self.centroids[j * self.dim..(j + 1) * self.dim];
            let mut acc = R::zero();
            let mut lo = 0usize;
            let mut pruned = false;
            for &hi in &self.stages {
                acc = acc + crate::kernels::sq_euclidean(&p[lo..hi], &row[lo..hi]);
                reads += (hi - lo) as u64;
                lo = hi;
                // Strictly greater: a candidate tied with the incumbent survives, which is what
                // makes the lowest-index tie-break match a full scan.
                if acc > best_dist {
                    pruned = true;
                    break;
                }
            }
            if pruned {
                continue;
            }
            // Lexicographic on (distance, index): among ties the lower index wins, exactly as an
            // ordered full scan would decide it, whether or not a hint was given.
            if acc < best_dist || (acc == best_dist && j < best) {
                best_dist = acc;
                best = j;
            }
        }
        // With k >= 1 a best is always found: the incumbent is never pruned against itself, and
        // with no hint the first candidate faces an infinite threshold.
        debug_assert!(
            best < self.k,
            "no candidate survived over {} centroids",
            self.k
        );
        (best, best_dist, reads)
    }

    /// The full distance to centroid `j`, accumulated over the plan's stages.
    ///
    /// Same summation order as the pruning loop, so the two are exactly comparable.
    fn staged_distance(&self, p: &[R], j: usize) -> R {
        let row = &self.centroids[j * self.dim..(j + 1) * self.dim];
        let mut acc = R::zero();
        let mut lo = 0usize;
        for &hi in &self.stages {
            acc = acc + crate::kernels::sq_euclidean(&p[lo..hi], &row[lo..hi]);
            lo = hi;
        }
        acc
    }

    /// Assign every row of the row-major `n × dim` matrix `flat`, writing labels into `out`.
    ///
    /// `hints` are per-row previous labels; `None` starts every row from an infinite threshold.
    ///
    /// # Panics
    /// If `flat.len() != n * self.dim()`, or `out.len() != n`, or `hints` is `Some` with a length
    /// other than `n`.
    pub fn assign_many(
        &self,
        flat: &[R],
        n: usize,
        hints: Option<&[u32]>,
        out: &mut [u32],
    ) -> AssignStats {
        assert_eq!(flat.len(), n * self.dim, "matrix is not {n} × {}", self.dim);
        assert_eq!(out.len(), n, "label buffer is not {n} long");
        if let Some(h) = hints {
            assert_eq!(h.len(), n, "hint buffer is not {n} long");
        }
        let mut point = Vec::with_capacity(self.dim);
        let mut reads = 0u64;
        for i in 0..n {
            let row = &flat[i * self.dim..(i + 1) * self.dim];
            // Gathering the row into the plan's order costs a copy of `dim` scalars per point,
            // amortised over `k` candidates. When the order is the identity there is nothing to
            // gather and the row is read where it lies.
            let p = if self.identity {
                row
            } else {
                point.clear();
                point.extend(self.order.iter().map(|&j| row[j as usize]));
                &point[..]
            };
            let hint = hints.map(|h| h[i] as usize);
            let (label, _, r) = self.nearest_permuted(p, hint);
            out[i] = label as u32;
            reads += r;
        }
        AssignStats {
            dims_read: reads,
            dims_total: (n as u64) * (self.k as u64) * self.dim as u64,
        }
    }
}

/// Dimensions by descending mass-weighted between-centroid variance.
///
/// Any order gives an exact bound; this one maximises the expected prefix, so it prunes soonest.
fn variance_order<R: Real>(centroids: &[Vec<R>], weights: Option<&[R]>, dim: usize) -> Vec<u32> {
    let k = centroids.len();
    let usable = weights
        .filter(|w| w.len() == k && w.iter().all(|x| x.is_finite() && *x >= R::zero()))
        .filter(|w| w.iter().fold(R::zero(), |a, &b| a + b) > R::zero());
    let total = match usable {
        Some(w) => w.iter().fold(R::zero(), |a, &b| a + b),
        None => R::from_usize(k).unwrap_or_else(R::one),
    };
    let weight = |i: usize| match usable {
        Some(w) => w[i] / total,
        None => R::one() / total,
    };

    let mut var = vec![0f64; dim];
    for d in 0..dim {
        let mean = (0..k).fold(R::zero(), |a, i| a + weight(i) * centroids[i][d]);
        let v = (0..k).fold(R::zero(), |a, i| {
            let e = centroids[i][d] - mean;
            a + weight(i) * e * e
        });
        // A non-finite coordinate makes the variance meaningless; sort those dimensions last
        // rather than letting NaN decide the comparison.
        var[d] = v
            .to_f64()
            .filter(|x| x.is_finite())
            .unwrap_or(f64::NEG_INFINITY);
    }
    let mut order: Vec<u32> = (0..dim as u32).collect();
    // Descending variance, ties by original index, so the order is a deterministic function of the
    // centroid set and not of the sort's internal state.
    order.sort_by(|&a, &b| {
        var[b as usize]
            .partial_cmp(&var[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;

    fn brute(x: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
        let mut best = 0;
        let mut bd = crate::kernels::sq_euclidean(x, &centroids[0]);
        for (i, c) in centroids.iter().enumerate().skip(1) {
            let d = crate::kernels::sq_euclidean(x, c);
            if d < bd {
                bd = d;
                best = i;
            }
        }
        (best, bd)
    }

    fn sample(rng: &mut SplitMix64, n: usize, dim: usize, spread: f64) -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| (0..dim).map(|_| (rng.next_f64() - 0.5) * spread).collect())
            .collect()
    }

    #[test]
    fn the_plan_returns_the_same_label_as_a_full_scan() {
        let mut rng = SplitMix64::new(7);
        for &(k, dim) in &[
            (1usize, 1usize),
            (2, 3),
            (5, 17),
            (10, 64),
            (33, 129),
            (7, 800),
        ] {
            let centroids = sample(&mut rng, k, dim, 4.0);
            let plan = AssignPlan::new(&centroids, None).unwrap();
            for _ in 0..40 {
                let x: Vec<f64> = (0..dim).map(|_| (rng.next_f64() - 0.5) * 6.0).collect();
                let (want, wd) = brute(&x, &centroids);
                let (got, gd) = plan.nearest(&x, None);
                assert_eq!(got, want, "k={k} dim={dim}");
                // the plan sums in its own order and in stages, a different rounding
                assert!(
                    (gd - wd).abs() <= 1e-12 * wd.max(1.0),
                    "k={k} dim={dim}: {gd} vs {wd}"
                );
            }
        }
    }

    #[test]
    fn a_hint_changes_the_work_but_never_the_answer() {
        let mut rng = SplitMix64::new(11);
        let centroids = sample(&mut rng, 12, 40, 3.0);
        let plan = AssignPlan::new(&centroids, None).unwrap();
        for _ in 0..50 {
            let x: Vec<f64> = (0..40).map(|_| (rng.next_f64() - 0.5) * 5.0).collect();
            let (want, wd) = brute(&x, &centroids);
            let unhinted = plan.nearest(&x, None);
            assert_eq!(unhinted.0, want);
            assert!(
                (unhinted.1 - wd).abs() <= 1e-12 * wd.max(1.0),
                "{} vs {wd}",
                unhinted.1
            );
            for hint in [Some(0), Some(5), Some(11), Some(999)] {
                // Exact equality, including the distance: the hint is only a threshold, and if it
                // were measured with a different summation than the loop uses it could both
                // report a differently-rounded distance and prune a candidate it should not.
                assert_eq!(plan.nearest(&x, hint), unhinted, "hint {hint:?}");
            }
        }
    }

    #[test]
    fn the_dimension_order_does_not_move_a_label() {
        let mut rng = SplitMix64::new(13);
        let centroids = sample(&mut rng, 9, 50, 2.0);
        let ordered = AssignPlan::new(&centroids, None).unwrap();
        let natural = AssignPlan::identity_order(&centroids).unwrap();
        assert_ne!(
            ordered.order(),
            natural.order(),
            "the fixture must exercise a real permutation"
        );
        for _ in 0..60 {
            let x: Vec<f64> = (0..50).map(|_| (rng.next_f64() - 0.5) * 4.0).collect();
            assert_eq!(ordered.nearest(&x, None).0, natural.nearest(&x, None).0);
        }
    }

    #[test]
    fn ties_go_to_the_lower_index_as_in_a_full_scan() {
        // Three centroids, the first and last equidistant from the query.
        let centroids = vec![vec![-1.0, 0.0], vec![0.0, 9.0], vec![1.0, 0.0]];
        let plan = AssignPlan::new(&centroids, None).unwrap();
        let x = [0.0, 0.0];
        assert_eq!(plan.nearest(&x, None).0, brute(&x, &centroids).0);
        assert_eq!(plan.nearest(&x, None).0, 0);
        // ... and a hint on the losing side of the tie must not win it
        assert_eq!(plan.nearest(&x, Some(2)).0, 0);
    }

    #[test]
    fn pruning_actually_prunes_on_separated_clusters() {
        let mut rng = SplitMix64::new(17);
        let dim = 256;
        let k = 20;
        let signal = 16; // the cluster identity lives in 16 of the 256 dimensions
        // The other 240 dimensions carry a shared constant and no cluster information, so the
        // variance order has to put the 16 informative ones first for the bound to bite early.
        let centroids: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                (0..dim)
                    .map(|d| {
                        if d < signal {
                            (((i + d) % 5) as f64) * 8.0
                        } else {
                            1.0
                        }
                    })
                    .collect()
            })
            .collect();
        let plan = AssignPlan::new(&centroids, None).unwrap();
        let n = 200;
        let mut flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            let c = &centroids[i % k];
            flat.extend(c.iter().map(|&v| v + (rng.next_f64() - 0.5) * 0.2));
        }
        let mut out = vec![0u32; n];
        let stats = plan.assign_many(&flat, n, None, &mut out);
        assert!(
            stats.read_fraction() < 0.5,
            "read {:.3}",
            stats.read_fraction()
        );
        // and the labels still agree with a full scan, whatever the fixture's geometry
        for i in 0..n {
            let want = brute(&flat[i * dim..(i + 1) * dim], &centroids).0;
            assert_eq!(out[i] as usize, want, "row {i}");
        }
    }

    #[test]
    fn assign_many_matches_a_full_scan_with_and_without_hints() {
        let mut rng = SplitMix64::new(19);
        let dim = 33;
        let centroids = sample(&mut rng, 6, dim, 3.0);
        let plan = AssignPlan::new(&centroids, None).unwrap();
        let n = 120;
        let mut flat = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            flat.push((rng.next_f64() - 0.5) * 5.0);
        }
        let want: Vec<u32> = (0..n)
            .map(|i| brute(&flat[i * dim..(i + 1) * dim], &centroids).0 as u32)
            .collect();
        let mut out = vec![u32::MAX; n];
        let stats = plan.assign_many(&flat, n, None, &mut out);
        assert_eq!(out, want);
        assert_eq!(stats.dims_total, (n * 6 * dim) as u64);
        // hints from the previous answer: same labels, and the threshold starts tight
        let mut out2 = vec![u32::MAX; n];
        let hinted = plan.assign_many(&flat, n, Some(&want), &mut out2);
        assert_eq!(out2, want);
        assert!(hinted.dims_read <= stats.dims_read + (n * dim) as u64);
    }

    #[test]
    fn weights_steer_the_order_without_steering_the_answer() {
        let mut rng = SplitMix64::new(23);
        let centroids = sample(&mut rng, 8, 24, 2.0);
        let heavy = vec![100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let a = AssignPlan::new(&centroids, Some(&heavy)).unwrap();
        let b = AssignPlan::new(&centroids, None).unwrap();
        for _ in 0..40 {
            let x: Vec<f64> = (0..24).map(|_| (rng.next_f64() - 0.5) * 4.0).collect();
            assert_eq!(a.nearest(&x, None).0, b.nearest(&x, None).0);
        }
    }

    #[test]
    fn degenerate_weights_fall_back_to_equal_weighting() {
        let centroids = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let equal = AssignPlan::new(&centroids, None).unwrap();
        for bad in [
            vec![0.0, 0.0],
            vec![f64::NAN, 1.0],
            vec![-1.0, 2.0],
            vec![1.0],
        ] {
            let plan = AssignPlan::new(&centroids, Some(&bad)).unwrap();
            assert_eq!(plan.order(), equal.order(), "weights {bad:?}");
        }
    }

    #[test]
    fn a_constant_dimension_sorts_last() {
        let centroids = vec![
            vec![0.0, 5.0, 1.0],
            vec![0.0, 5.0, 9.0],
            vec![0.0, 5.0, -4.0],
        ];
        let plan = AssignPlan::new(&centroids, None).unwrap();
        assert_eq!(
            plan.order()[0],
            2,
            "the only varying dimension must be read first"
        );
    }

    #[test]
    fn a_bad_centroid_set_is_rejected_rather_than_guessed_at() {
        let empty: Vec<Vec<f64>> = vec![];
        assert_eq!(AssignPlan::new(&empty, None).unwrap_err(), PlanError::Empty);
        assert_eq!(
            AssignPlan::<f64>::new(&[vec![]], None).unwrap_err(),
            PlanError::ZeroDim
        );
        let ragged = vec![vec![1.0, 2.0], vec![3.0]];
        assert_eq!(
            AssignPlan::new(&ragged, None).unwrap_err(),
            PlanError::Ragged {
                row: 1,
                len: 1,
                dim: 2
            }
        );
        assert_eq!(
            AssignPlan::identity_order(&empty).unwrap_err(),
            PlanError::Empty
        );
        assert_eq!(
            AssignPlan::identity_order(&ragged).unwrap_err(),
            PlanError::Ragged {
                row: 1,
                len: 1,
                dim: 2
            }
        );
    }

    #[test]
    fn the_prefix_bound_never_exceeds_the_full_distance() {
        // The property the whole module rests on, checked directly: a running sum of squared
        // differences is non-decreasing and ends at the full distance.
        let mut rng = SplitMix64::new(29);
        for _ in 0..200 {
            let dim = 1 + (rng.next_u64() % 300) as usize;
            let a: Vec<f64> = (0..dim).map(|_| (rng.next_f64() - 0.5) * 1e6).collect();
            let b: Vec<f64> = (0..dim).map(|_| (rng.next_f64() - 0.5) * 1e6).collect();
            let full = crate::kernels::sq_euclidean(&a, &b);
            let mut acc = 0.0;
            for i in 0..dim {
                acc += (a[i] - b[i]) * (a[i] - b[i]);
                assert!(
                    acc <= full * (1.0 + 1e-12) + 1e-9,
                    "prefix {acc} exceeded full {full}"
                );
            }
        }
    }

    #[test]
    fn read_fraction_is_one_when_nothing_was_assigned() {
        assert_eq!(AssignStats::default().read_fraction(), 1.0);
    }

    #[test]
    fn plan_errors_render_a_message_naming_the_offending_row() {
        assert!(PlanError::Empty.to_string().contains("no centroids"));
        assert!(PlanError::ZeroDim.to_string().contains("zero dimensions"));
        let m = PlanError::Ragged {
            row: 3,
            len: 2,
            dim: 5,
        }
        .to_string();
        assert!(m.contains('3') && m.contains('2') && m.contains('5'), "{m}");
    }

    #[test]
    #[should_panic(expected = "point has 3 dimensions")]
    fn a_point_of_the_wrong_length_is_rejected() {
        let plan = AssignPlan::new(&[vec![0.0, 0.0]], None).unwrap();
        plan.nearest(&[1.0, 2.0, 3.0], None);
    }

    #[test]
    #[should_panic(expected = "is not 2 ×")]
    fn a_matrix_of_the_wrong_shape_is_rejected() {
        let plan = AssignPlan::new(&[vec![0.0, 0.0]], None).unwrap();
        let mut out = vec![0u32; 2];
        plan.assign_many(&[1.0, 2.0, 3.0], 2, None, &mut out);
    }

    #[test]
    #[should_panic(expected = "hint buffer")]
    fn a_hint_buffer_of_the_wrong_length_is_rejected() {
        let plan = AssignPlan::new(&[vec![0.0, 0.0]], None).unwrap();
        let mut out = vec![0u32; 2];
        plan.assign_many(&[1.0, 2.0, 3.0, 4.0], 2, Some(&[0]), &mut out);
    }

    #[test]
    fn f32_plans_work_the_same_way() {
        let centroids: Vec<Vec<f32>> = vec![vec![0.0, 0.0, 1.0], vec![3.0, 0.0, 1.0]];
        let plan = AssignPlan::new(&centroids, None).unwrap();
        assert_eq!(plan.nearest(&[0.1f32, 0.0, 1.0], None).0, 0);
        assert_eq!(plan.nearest(&[2.9f32, 0.0, 1.0], None).0, 1);
        assert_eq!(plan.dim(), 3);
        assert_eq!(plan.k(), 2);
    }
}
