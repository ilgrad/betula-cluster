//! KLL — a compact, mergeable quantile sketch with **rank-error** guarantees.
//!
//! Karnin, Lang & Liberty, *FOCS 2016* ("Optimal Quantile Approximation in Streams"). A hierarchy of
//! *compactors*: compactor `h` holds items each standing for `2^h` stream points. When a compactor
//! fills, it is sorted and halved — every other item is promoted to compactor `h+1` (doubling its
//! weight), the rest discarded — so the total weight is invariant (= `n`) while space stays `O(k)`.
//! Rank error is `≈ ε·n` with `ε = O(1/k)`, uniform across the distribution (unlike DDSketch's
//! relative error). Exact min/max are tracked so `quantile(0)` / `quantile(1)` are exact.
//!
//! This follows the reference implementation by the paper's authors
//! (`edoliberty/streaming-quantiles`); the randomized compaction is seeded for reproducibility.

/// A KLL quantile sketch over `f64`.
#[derive(Clone)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct KllSketch {
    k: usize,
    compactors: Vec<Vec<f64>>,
    size: usize,     // items currently stored across all compactors
    max_size: usize, // sum of per-compactor capacities at the current height
    n: u64,          // total items added (== total stored weight)
    min: f64,
    max: f64,
    rng: u64, // xorshift64 state for the compaction coin
}

impl KllSketch {
    /// New sketch; larger `k` ⇒ smaller error (`≈ 1/k`) and more memory (`O(k)`). `seed` makes the
    /// randomized compaction reproducible.
    pub fn new(k: usize, seed: u64) -> Self {
        let mut s = Self {
            k: k.max(2),
            compactors: Vec::new(),
            size: 0,
            max_size: 0,
            n: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            rng: seed | 1, // nonzero state
        };
        s.grow();
        s
    }

    fn grow(&mut self) {
        self.compactors.push(Vec::new());
        let h = self.compactors.len();
        self.max_size = (0..h).map(|height| self.capacity(height, h)).sum();
    }

    /// Capacity of compactor `height` when the sketch has `levels` compactors (taller compactors are
    /// larger; the shortest floor at ~2), summing to `O(k)`.
    fn capacity(&self, height: usize, levels: usize) -> usize {
        let depth = levels - height - 1;
        ((2.0_f64 / 3.0).powi(depth as i32) * self.k as f64).ceil() as usize + 1
    }

    /// A fair coin (low bit of an xorshift64 step) for choosing which half to keep on compaction.
    fn coin(&mut self) -> usize {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng & 1) as usize
    }

    /// Add one value.
    pub fn update(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.compactors[0].push(x);
        self.size += 1;
        self.n += 1;
        self.min = self.min.min(x);
        self.max = self.max.max(x);
        if self.size >= self.max_size {
            self.compress();
        }
    }

    fn compress(&mut self) {
        let levels = self.compactors.len();
        for h in 0..levels {
            if self.compactors[h].len() >= self.capacity(h, self.compactors.len()) {
                if h + 1 >= self.compactors.len() {
                    self.grow();
                }
                self.compactors[h].sort_by(|a, b| a.partial_cmp(b).unwrap());
                let offset = self.coin();
                let promoted: Vec<f64> = self.compactors[h]
                    .iter()
                    .skip(offset)
                    .step_by(2)
                    .copied()
                    .collect();
                self.compactors[h + 1].extend(promoted);
                self.compactors[h].clear();
                self.size = self.compactors.iter().map(|c| c.len()).sum();
                break;
            }
        }
    }

    /// Total number of values added.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Smallest / largest value seen (`NaN` if empty).
    pub fn min(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.min
        }
    }
    pub fn max(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.max
        }
    }

    /// Sorted `(value, cumulative_weight)` pairs over all compactors (cumulative weight reaches `n`).
    fn cdf(&self) -> Vec<(f64, u64)> {
        let mut items: Vec<(f64, u64)> = Vec::with_capacity(self.size);
        for (h, c) in self.compactors.iter().enumerate() {
            let w = 1u64 << h;
            for &v in c {
                items.push((v, w));
            }
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut acc = 0u64;
        for it in &mut items {
            acc += it.1;
            it.1 = acc;
        }
        items
    }

    /// Estimated number of stored values `≤ value`.
    pub fn rank(&self, value: f64) -> u64 {
        let mut r = 0u64;
        for (h, c) in self.compactors.iter().enumerate() {
            let w = 1u64 << h;
            r += w * c.iter().filter(|&&v| v <= value).count() as u64;
        }
        r
    }

    /// Estimated `q`-quantile (`q ∈ [0, 1]`); exact at the endpoints, `NaN` if empty.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let q = q.clamp(0.0, 1.0);
        if q <= 0.0 {
            return self.min;
        }
        if q >= 1.0 {
            return self.max;
        }
        let cdf = self.cdf();
        let target = (q * self.n as f64).ceil() as u64;
        for (v, cum) in &cdf {
            if *cum >= target {
                return *v;
            }
        }
        self.max
    }

    /// Estimated quantiles for many `q` in one pass over the (built-once) CDF.
    pub fn quantiles(&self, qs: &[f64]) -> Vec<f64> {
        if self.n == 0 {
            return vec![f64::NAN; qs.len()];
        }
        let cdf = self.cdf();
        let n = self.n as f64;
        qs.iter()
            .map(|&q| {
                let q = q.clamp(0.0, 1.0);
                if q <= 0.0 {
                    return self.min;
                }
                if q >= 1.0 {
                    return self.max;
                }
                let target = (q * n).ceil() as u64;
                cdf.iter()
                    .find(|(_, cum)| *cum >= target)
                    .map_or(self.max, |(v, _)| *v)
            })
            .collect()
    }

    /// Merge `other` into `self` (both keep `k`); the result summarizes the union of both streams.
    pub fn merge(&mut self, other: &KllSketch) {
        if other.n == 0 {
            return;
        }
        while self.compactors.len() < other.compactors.len() {
            self.grow();
        }
        for (h, c) in other.compactors.iter().enumerate() {
            self.compactors[h].extend_from_slice(c);
        }
        self.n += other.n;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.size = self.compactors.iter().map(|c| c.len()).sum();
        // Restore the size invariant (a merge can overfill several levels at once).
        while self.size >= self.max_size {
            let before = self.size;
            self.compress();
            if self.size == before {
                break; // nothing was over capacity (guards against a stuck loop)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;

    /// Largest |estimated_rank − true_rank| / n over a grid of quantiles.
    fn max_rank_error(s: &KllSketch, sorted: &[f64]) -> f64 {
        let n = sorted.len() as f64;
        let mut worst = 0.0f64;
        for i in 1..100 {
            let q = i as f64 / 100.0;
            let est = s.quantile(q);
            // true rank of `est` in the exact data
            let true_rank = sorted.partition_point(|&v| v <= est) as f64;
            worst = worst.max((true_rank / n - q).abs());
        }
        worst
    }

    #[test]
    fn kll_rank_error_within_bound() {
        let mut rng = SplitMix64::new(42);
        let n = 200_000usize;
        let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        // shuffle (Fisher–Yates) so insertion order is not sorted
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            data.swap(i, j);
        }
        let mut s = KllSketch::new(256, 1);
        for &x in &data {
            s.update(x);
        }
        assert_eq!(s.count(), n as u64);
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // ε ≈ O(1/k); k = 256 ⇒ comfortably under 3 %.
        assert!(max_rank_error(&s, &sorted) < 0.03, "rank error too large");
        assert_eq!(s.min(), 0.0);
        assert_eq!(s.max(), (n - 1) as f64);
    }

    #[test]
    fn kll_merge_matches_combined_stream() {
        let mut a = KllSketch::new(256, 1);
        let mut b = KllSketch::new(256, 2);
        let n = 100_000usize;
        for i in 0..n {
            a.update(i as f64);
            b.update((i + n) as f64);
        }
        a.merge(&b);
        assert_eq!(a.count(), 2 * n as u64);
        let total = 2 * n;
        let sorted: Vec<f64> = (0..total).map(|i| i as f64).collect();
        assert!(
            max_rank_error(&a, &sorted) < 0.03,
            "merged rank error too large"
        );
        assert_eq!(a.max(), (total - 1) as f64);
    }

    #[test]
    fn kll_empty_and_single() {
        let mut s = KllSketch::new(200, 1);
        assert!(s.quantile(0.5).is_nan());
        assert!(s.min().is_nan());
        assert!(s.quantiles(&[0.1, 0.9]).iter().all(|v| v.is_nan()));
        s.update(3.0);
        assert_eq!(s.count(), 1);
        assert_eq!(s.quantile(0.0), 3.0);
        assert_eq!(s.quantile(0.5), 3.0);
        assert_eq!(s.quantile(1.0), 3.0);
    }

    #[test]
    fn kll_ignores_nan() {
        let mut s = KllSketch::new(200, 1);
        s.update(f64::NAN);
        s.update(1.0);
        assert_eq!(s.count(), 1);
    }
}
