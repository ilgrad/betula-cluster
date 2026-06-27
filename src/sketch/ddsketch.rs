//! DDSketch — a fully-mergeable quantile sketch with **relative-error** guarantees.
//!
//! Masson, Rim & Lee, *VLDB 2019*. A value `x > 0` maps to bucket `⌈log_γ(x)⌉` with
//! `γ = (1+α)/(1−α)`, so the bucket's representative is within relative error `α` of `x`. Counts per
//! bucket are kept in a map; a quantile walks buckets in value order until the cumulative count
//! passes the target rank. Negatives use a mirror store, exact zeros a separate counter. Unlike
//! KLL's uniform rank error, the error here is *relative* — ideal for skewed, positive,
//! long-tailed data (latencies, sizes). `max_bins` bounds memory by collapsing the lowest buckets.

use std::collections::HashMap;

/// A DDSketch quantile sketch over `f64`.
#[derive(Clone)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct DdSketch {
    gamma: f64,
    log_gamma: f64,
    alpha: f64,
    max_bins: usize,
    positive: HashMap<i32, u64>,
    negative: HashMap<i32, u64>, // keyed by |x|
    zero: u64,
    n: u64,
    min: f64,
    max: f64,
}

impl DdSketch {
    /// New sketch with relative accuracy `alpha ∈ (0, 1)`; `max_bins` caps the buckets per sign
    /// (lowest collapse beyond it). Smaller `alpha` ⇒ tighter quantiles and more buckets.
    pub fn new(alpha: f64, max_bins: usize) -> Result<Self, &'static str> {
        if alpha.is_nan() || alpha <= 0.0 || alpha >= 1.0 {
            return Err("alpha must be in (0, 1)");
        }
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        Ok(Self {
            gamma,
            log_gamma: gamma.ln(),
            alpha,
            max_bins: max_bins.max(1),
            positive: HashMap::new(),
            negative: HashMap::new(),
            zero: 0,
            n: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        })
    }

    /// Relative accuracy `α`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Bucket index for a positive magnitude.
    fn key(&self, mag: f64) -> i32 {
        (mag.ln() / self.log_gamma).ceil() as i32
    }

    /// Representative value of a positive bucket (within `α` relative error of any value in it).
    fn value(&self, key: i32) -> f64 {
        2.0 * self.gamma.powi(key) / (self.gamma + 1.0)
    }

    /// Collapse the lowest buckets of `store` until at most `max_bins` remain (merge `min → min+1`).
    fn collapse(store: &mut HashMap<i32, u64>, max_bins: usize) {
        while store.len() > max_bins {
            let lo = *store.keys().min().unwrap();
            let c = store.remove(&lo).unwrap();
            *store.entry(lo + 1).or_insert(0) += c;
        }
    }

    /// Add one value.
    pub fn update(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.n += 1;
        self.min = self.min.min(x);
        self.max = self.max.max(x);
        if x > 0.0 {
            *self.positive.entry(self.key(x)).or_insert(0) += 1;
            Self::collapse(&mut self.positive, self.max_bins);
        } else if x < 0.0 {
            *self.negative.entry(self.key(-x)).or_insert(0) += 1;
            Self::collapse(&mut self.negative, self.max_bins);
        } else {
            self.zero += 1;
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
        let rank = (q * (self.n - 1) as f64).floor() as u64;
        let mut cum = 0u64;
        // ascending value order: negatives (largest |x| first), then zeros, then positives.
        let mut neg: Vec<i32> = self.negative.keys().copied().collect();
        neg.sort_unstable_by(|a, b| b.cmp(a));
        for k in neg {
            cum += self.negative[&k];
            if cum > rank {
                return -self.value(k);
            }
        }
        cum += self.zero;
        if cum > rank {
            return 0.0;
        }
        let mut pos: Vec<i32> = self.positive.keys().copied().collect();
        pos.sort_unstable();
        for k in pos {
            cum += self.positive[&k];
            if cum > rank {
                return self.value(k);
            }
        }
        self.max
    }

    /// Estimated quantiles for many `q`.
    pub fn quantiles(&self, qs: &[f64]) -> Vec<f64> {
        qs.iter().map(|&q| self.quantile(q)).collect()
    }

    /// Merge `other` into `self` (relative accuracies must match).
    pub fn merge(&mut self, other: &DdSketch) -> Result<(), &'static str> {
        if (self.gamma - other.gamma).abs() > 1e-12 {
            return Err("cannot merge DDSketches with different alpha");
        }
        if other.n == 0 {
            return Ok(());
        }
        for (&k, &c) in &other.positive {
            *self.positive.entry(k).or_insert(0) += c;
        }
        for (&k, &c) in &other.negative {
            *self.negative.entry(k).or_insert(0) += c;
        }
        Self::collapse(&mut self.positive, self.max_bins);
        Self::collapse(&mut self.negative, self.max_bins);
        self.zero += other.zero;
        self.n += other.n;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddsketch_relative_error_on_positive_data() {
        let alpha = 0.01;
        let mut s = DdSketch::new(alpha, 2048).unwrap();
        let n = 100_000usize;
        for i in 1..=n {
            s.update(i as f64);
        }
        assert_eq!(s.count(), n as u64);
        for &q in &[0.1, 0.5, 0.9, 0.99] {
            let est = s.quantile(q);
            let truth = (q * (n as f64 - 1.0)).floor() + 1.0; // value at that rank in 1..=n
            assert!(
                (est - truth).abs() / truth <= alpha * 1.5,
                "q={q}: est={est} truth={truth} rel_err too large"
            );
        }
        assert_eq!(s.min(), 1.0);
        assert_eq!(s.max(), n as f64);
    }

    #[test]
    fn ddsketch_handles_negatives_and_zeros() {
        let mut s = DdSketch::new(0.01, 2048).unwrap();
        for i in -1000..=1000 {
            s.update(i as f64);
        }
        assert_eq!(s.count(), 2001);
        assert!(s.quantile(0.5).abs() <= 1.0); // median ≈ 0
        assert!(s.quantile(0.1) < 0.0);
        assert!(s.quantile(0.9) > 0.0);
    }

    #[test]
    fn ddsketch_merge_matches_combined() {
        let mut a = DdSketch::new(0.01, 2048).unwrap();
        let mut b = DdSketch::new(0.01, 2048).unwrap();
        for i in 1..=50_000 {
            a.update(i as f64);
        }
        for i in 50_001..=100_000 {
            b.update(i as f64);
        }
        a.merge(&b).unwrap();
        assert_eq!(a.count(), 100_000);
        let est = a.quantile(0.5);
        assert!((est - 50_000.0).abs() / 50_000.0 <= 0.02);
    }

    #[test]
    fn ddsketch_max_bins_bounds_memory() {
        let mut s = DdSketch::new(0.001, 16).unwrap(); // tiny cap, wide range
        for i in 1..=100_000 {
            s.update(i as f64);
        }
        assert!(s.positive.len() <= 16, "bucket count exceeded max_bins");
        // high quantiles (the kept, high end) stay reasonable despite low-end collapse
        assert!(s.quantile(0.99) > s.quantile(0.5));
    }

    #[test]
    fn ddsketch_merge_alpha_mismatch_errors() {
        let mut a = DdSketch::new(0.01, 2048).unwrap();
        let b = DdSketch::new(0.02, 2048).unwrap();
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn ddsketch_empty_edge_and_bad_alpha() {
        let s = DdSketch::new(0.01, 2048).unwrap();
        assert!(s.quantile(0.5).is_nan());
        assert!(s.min().is_nan());
        assert!(DdSketch::new(0.0, 16).is_err());
        assert!(DdSketch::new(1.0, 16).is_err());
        let mut one = DdSketch::new(0.01, 16).unwrap();
        one.update(0.0);
        one.update(f64::NAN); // ignored
        assert_eq!(one.count(), 1);
        assert_eq!(one.quantile(0.5), 0.0);
    }
}
