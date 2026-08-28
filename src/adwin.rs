//! ADWIN: adaptive windowing over a scalar stream, as a change detector.
//!
//! Bifet & Gavaldà, *Learning from Time-Changing Data with Adaptive Windowing* (SDM 2007). The
//! window holds the recent values and keeps exactly as much of them as still looks like one
//! distribution: if any split of it into an older `W₀` and a newer `W₁` has means far enough apart
//! that chance does not explain the gap, the older part is dropped and a change is reported. The
//! bound it tests against is Hoeffding with the observed variance,
//!
//! ```text
//! ε_cut = √(2·m⁻¹·σ̂²_W·ln(2/δ')) + (2/3)·m⁻¹·ln(2/δ'),   m⁻¹ = 1/n₀' + 1/n₁',  δ' = δ/ln|W|
//! ```
//!
//! so the false-positive rate is a stated parameter rather than a tuned threshold — which is the
//! whole reason to prefer it here. This crate already decays microclusters by `2^(−λ·Δt)`, but λ has
//! to be chosen in advance and a wrong λ is silent: too small and the model never forgets, too large
//! and it forgets structure that never changed. ADWIN answers the other question — *did the stream
//! change* — from the data, with δ as the only knob and a meaning attached to it.
//!
//! **What it is not.** It reports; it does not act. Firing the detector does not rebuild a tree,
//! reset a decay rate or relabel anything: what to do about a detected change is a policy that
//! belongs to the caller, and a detector that silently rebuilt the model would make its own
//! false-positive rate a correctness problem rather than a reported number.
//!
//! Memory is `O(log |W|)` and time per update `O(log |W|)` — the ADWIN2 exponential histogram, which
//! keeps at most `M + 1` buckets of each size `2^i` rather than the raw window. The crate feeds it
//! one value per streamed *point*, so `update` sits on an insert path and allocates nothing.

/// Buckets kept per size class before the two oldest are merged into the next size up. Bifet &
/// Gavaldà's `M`: the memory bound is `(M+1)·log|W|` buckets and the relative error of any
/// sub-window mean is `O(1/M)`, so this trades precision of the cut point against space.
const BUCKETS_PER_LEVEL: usize = 5;

/// Shortest either side of a split may be. Two points can be arbitrarily far apart by chance, and
/// the variance estimate a bound this tight is built on is worthless below a handful of samples.
const MIN_SIDE: f64 = 5.0;

/// Updates between change tests. Bifet & Gavaldà's own implementation (MOA's `ADWIN.mintClock`) does
/// the same: examining *fewer* cuts can only make the union bound behind δ more conservative, so the
/// guarantee survives and detection is merely delayed by at most this many values. It is not an
/// optimisation to take lightly — the test walks `O(log|W|)` buckets with a square root each, and
/// measured on the insert path of [`crate::stream::DenStream`] the unclocked detector cost 35% of an
/// insert against 2% clocked.
pub(crate) const CHECK_EVERY: u64 = 32;

/// One bucket of an exponential histogram: `size` values summing to `total`, with `variance` their
/// sum of squared deviations from their own mean.
#[derive(Clone, Copy, Debug)]
struct Bucket {
    size: f64,
    total: f64,
    variance: f64,
}

impl Bucket {
    /// Merge `self` (older) with `newer`. The cross term is the price of pooling two means:
    /// `Σ(x−μ)²` over the union is the two inner sums plus `n₀n₁/(n₀+n₁)·(μ₀−μ₁)²`.
    fn merged(self, newer: Bucket) -> Bucket {
        let size = self.size + newer.size;
        let gap = self.total / self.size - newer.total / newer.size;
        Bucket {
            size,
            total: self.total + newer.total,
            variance: self.variance + newer.variance + self.size * newer.size / size * gap * gap,
        }
    }
}

/// Adaptive window over a scalar stream.
///
/// Feed it one number per observation with [`Adwin::update`]; it returns `true` on the update that
/// dropped a prefix, which is the change signal. [`Adwin::mean`] and [`Adwin::width`] describe the
/// window that survived — which at the moment of firing still holds some of the old regime, because
/// the bound is built on the variance of what is left and a window spanning two regimes has an
/// inflated one. The window converges on the new regime over the updates that follow.
pub struct Adwin {
    /// Confidence: the bound is built so that a false positive has probability at most `delta`.
    delta: f64,
    /// Size classes, index `i` holding buckets of `2^i` values. Within a class the front is the
    /// newest bucket; across classes, a higher index is older than every bucket below it.
    levels: Vec<Vec<Bucket>>,
    width: f64,
    total: f64,
    /// `Σ(x − μ_W)²` over the whole window, maintained incrementally.
    variance: f64,
    /// Values admitted since the last change test.
    since_check: u64,
}

impl Adwin {
    /// A detector at confidence `delta` — the ceiling on the false-positive rate per update.
    /// Clamped into `(0, 1)` because the bound divides by `ln(2/δ')` and `δ ≥ 1` makes it negative,
    /// which would report a change on every update rather than erroring out somewhere useful.
    pub fn new(delta: f64) -> Self {
        Self {
            delta: delta.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON),
            levels: Vec::new(),
            width: 0.0,
            total: 0.0,
            variance: 0.0,
            since_check: 0,
        }
    }

    /// Values currently in the window.
    pub fn width(&self) -> usize {
        self.width as usize
    }

    /// Mean of the window; `0.0` while it is empty.
    pub fn mean(&self) -> f64 {
        if self.width > 0.0 {
            self.total / self.width
        } else {
            0.0
        }
    }

    /// Sample variance of the window; `0.0` below two values.
    pub fn variance(&self) -> f64 {
        if self.width > 1.0 {
            self.variance / self.width
        } else {
            0.0
        }
    }

    /// Add `value` and test for a change. Returns `true` if a prefix of the window was dropped,
    /// i.e. the stream is reported to have changed. A non-finite value is ignored rather than
    /// admitted: one NaN in the window makes every subsequent comparison false and silently
    /// disables the detector.
    pub fn update(&mut self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        // Incremental `Σ(x − μ)²`: the new value's deviation is measured against the mean *before*
        // it was added, scaled by `n/(n+1)`.
        if self.width > 0.0 {
            let dev = value - self.total / self.width;
            self.variance += self.width * dev * dev / (self.width + 1.0);
        }
        self.width += 1.0;
        self.total += value;
        self.push_bucket(Bucket {
            size: 1.0,
            total: value,
            variance: 0.0,
        });
        self.since_check += 1;
        if self.since_check < CHECK_EVERY {
            return false;
        }
        self.since_check = 0;
        self.shrink()
    }

    /// Insert a size-1 bucket and carry the overflow up the size classes.
    fn push_bucket(&mut self, b: Bucket) {
        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }
        self.levels[0].insert(0, b);
        let mut i = 0;
        while self.levels[i].len() > BUCKETS_PER_LEVEL + 1 {
            // The two oldest of this class become one bucket of the next class up, and it is the
            // newest there: everything already at `i + 1` is older than everything at `i`.
            let old = self.levels[i].pop().expect("the class is over its cap");
            let newer = self.levels[i].pop().expect("the class is over its cap");
            let merged = old.merged(newer);
            if i + 1 == self.levels.len() {
                self.levels.push(Vec::new());
            }
            self.levels[i + 1].insert(0, merged);
            i += 1;
        }
    }

    /// Buckets oldest-first, which is the order a split point walks: a higher size class is older
    /// than every bucket below it, and within a class the front is the newest.
    fn oldest_first(&self) -> impl Iterator<Item = &Bucket> {
        self.levels.iter().rev().flat_map(|lvl| lvl.iter().rev())
    }

    /// Drop the oldest bucket, keeping `width`, `total` and `variance` consistent with what is left.
    fn drop_oldest(&mut self) {
        let Some(level) = self.levels.iter().rposition(|l| !l.is_empty()) else {
            return;
        };
        let gone = self.levels[level].pop().expect("the class is non-empty");
        while self.levels.last().is_some_and(|l| l.is_empty()) {
            self.levels.pop();
        }
        let rest = self.width - gone.size;
        self.width = rest;
        self.total -= gone.total;
        if rest > 0.0 {
            // Removing a group splits `Σ(x−μ)²` the same way merging pooled it, in reverse.
            let gap = gone.total / gone.size - self.total / rest;
            self.variance -= gone.variance + gone.size * rest / (gone.size + rest) * gap * gap;
            self.variance = self.variance.max(0.0);
        } else {
            self.variance = 0.0;
        }
    }

    /// Drop the oldest bucket for as long as some split of the window fails the bound. Returns
    /// whether anything was dropped.
    fn shrink(&mut self) -> bool {
        let mut changed = false;
        // Each pass drops at most one bucket, and there are `O(log|W|)` of them, so the loop is
        // bounded by the histogram rather than by the test that decides it.
        loop {
            if self.width < 2.0 * MIN_SIDE || !self.has_split_that_fails() {
                return changed;
            }
            self.drop_oldest();
            changed = true;
        }
    }

    /// Is there a split of the window whose two halves are further apart than `ε_cut` allows?
    ///
    /// Only bucket boundaries are candidates — that is the approximation the histogram buys, and
    /// with `BUCKETS_PER_LEVEL = 5` the boundary is within `O(1/M)` of the true cut point.
    fn has_split_that_fails(&self) -> bool {
        let (mut n0, mut t0) = (0.0f64, 0.0f64);
        // `ln|W|` rather than `|W|` in the union bound: the number of *cuts* examined over the life
        // of a window grows logarithmically, not linearly, once the histogram is what holds it.
        let log_w = self.width.ln().max(1.0);
        let dd = (2.0 * log_w / self.delta).ln();
        let var = self.variance / self.width;
        for b in self.oldest_first() {
            n0 += b.size;
            t0 += b.total;
            // The newest bucket is not a split point — `W₁` would be empty — and `n1 < MIN_SIDE`
            // is what rules it out, so the walk needs no separate end case.
            let n1 = self.width - n0;
            if n0 < MIN_SIDE || n1 < MIN_SIDE {
                continue;
            }
            let m_inv = 1.0 / (n0 - MIN_SIDE + 1.0) + 1.0 / (n1 - MIN_SIDE + 1.0);
            let eps = (2.0 * m_inv * var * dd).sqrt() + 2.0 / 3.0 * m_inv * dd;
            if (t0 / n0 - (self.total - t0) / n1).abs() > eps {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;

    /// Standard normal, so the fixtures state a mean shift in units the bound is written in.
    fn gauss(rng: &mut SplitMix64) -> f64 {
        rng.gauss()
    }

    #[test]
    fn a_stationary_stream_does_not_fire_at_anything_like_the_rate_delta_allows() {
        // The claim `delta` makes is a ceiling on the false-positive rate, so the measurement is a
        // rate and not a yes/no: 20 independent stationary streams of 2000 draws each.
        let mut fires = 0usize;
        let mut updates = 0usize;
        for seed in 0..20u64 {
            let mut rng = SplitMix64::new(seed);
            let mut adwin = Adwin::new(0.002);
            for _ in 0..2000 {
                updates += 1;
                fires += usize::from(adwin.update(gauss(&mut rng)));
            }
        }
        let rate = fires as f64 / updates as f64;
        assert!(
            rate <= 0.002,
            "{fires} false positives in {updates} updates ({rate:.5}) is above the delta claimed"
        );
    }

    #[test]
    fn an_abrupt_shift_is_reported_and_the_prefix_that_predates_it_is_dropped() {
        for seed in 0..5u64 {
            let mut rng = SplitMix64::new(seed);
            let mut adwin = Adwin::new(0.002);
            for _ in 0..500 {
                assert!(
                    !adwin.update(gauss(&mut rng)),
                    "seed {seed} fired before the change"
                );
            }
            let before = adwin.width();
            let mut fired_at = None;
            for i in 0..500 {
                if adwin.update(4.0 + gauss(&mut rng)) {
                    fired_at = Some(i);
                    break;
                }
            }
            let at =
                fired_at.unwrap_or_else(|| panic!("seed {seed}: no detection after the shift"));
            assert!(at < 60, "seed {seed}: {at} samples to notice a 4σ shift");
            // Detection *is* the drop — the window is cut, not merely flagged. It cannot be cut all
            // the way to the new regime at the moment of firing, because the bound is built on the
            // variance of what is left and a window still holding both regimes has an inflated one;
            // what must hold now is that the old data is going.
            assert!(
                adwin.width() < before,
                "seed {seed}: the window did not shrink on the update that reported a change"
            );
            // Fed nothing but the new regime, it converges there rather than holding a blend.
            for _ in 0..300 {
                adwin.update(4.0 + gauss(&mut rng));
            }
            assert!(
                adwin.mean() > 3.5,
                "seed {seed}: window mean {:.3} still carries the old regime",
                adwin.mean()
            );
        }
    }

    #[test]
    fn the_change_test_runs_on_a_clock_so_an_alarm_can_only_land_on_a_multiple_of_it() {
        // `since_check` counts admitted values and resets only when a test runs, so the k-th value
        // is tested exactly when `k % CHECK_EVERY == 0`. Skipping tests is what keeps the detector
        // affordable on a per-point insert path; this pins that it is a clock and not a filter that
        // could drop the test that would have fired.
        let mut rng = SplitMix64::new(13);
        let mut adwin = Adwin::new(0.002);
        let mut seen = 0usize;
        let mut at = None;
        for i in 0..2000 {
            seen += 1;
            let v = if i < 400 {
                gauss(&mut rng)
            } else {
                4.0 + gauss(&mut rng)
            };
            if adwin.update(v) {
                at = Some(seen);
                break;
            }
        }
        let at = at.expect("a 4σ shift went unreported");
        assert_eq!(
            at % CHECK_EVERY as usize,
            0,
            "value {at} raised an alarm off the clock"
        );
        assert!(
            at < 400 + 2 * CHECK_EVERY as usize,
            "{at} is more than one clock late"
        );
    }

    #[test]
    fn a_smaller_shift_needs_more_evidence_and_still_arrives() {
        // Monotone in the obvious direction: the bound is a function of the gap, so halving the
        // shift must not make detection *faster*. Pinned because a sign slip in `eps` would fire
        // sooner on less evidence, which no other test here would notice.
        let delay = |shift: f64| -> usize {
            let mut rng = SplitMix64::new(11);
            let mut adwin = Adwin::new(0.002);
            for _ in 0..400 {
                adwin.update(gauss(&mut rng));
            }
            (0..4000)
                .find(|_| adwin.update(shift + gauss(&mut rng)))
                .unwrap_or(usize::MAX)
        };
        let (big, small) = (delay(4.0), delay(1.0));
        assert!(big < 60 && small < 4000, "big {big}, small {small}");
        assert!(big < small, "a 4σ shift took {big} and a 1σ shift {small}");
    }

    #[test]
    fn the_window_is_logarithmic_in_what_it_has_seen() {
        // The histogram is the reason this can run on an unbounded stream: 100 000 stationary
        // values must not become 100 000 buckets.
        let mut rng = SplitMix64::new(3);
        let mut adwin = Adwin::new(0.002);
        for _ in 0..100_000 {
            adwin.update(gauss(&mut rng));
        }
        let buckets: usize = adwin.levels.iter().map(Vec::len).sum();
        assert!(
            buckets <= (BUCKETS_PER_LEVEL + 1) * (adwin.width().ilog2() as usize + 2),
            "{buckets} buckets for a window of {}",
            adwin.width()
        );
        assert!(
            adwin.width() > 1000,
            "the window collapsed on stationary data"
        );
    }

    #[test]
    fn the_window_statistics_match_a_direct_pass_over_what_it_holds() {
        // The incremental variance and the merge/drop corrections are three different pieces of
        // arithmetic for one quantity; on a stream that never triggers a drop they must agree with
        // the textbook two-pass form over the same values.
        let mut rng = SplitMix64::new(5);
        let mut adwin = Adwin::new(1e-12);
        let mut seen = Vec::new();
        for _ in 0..200 {
            let v = gauss(&mut rng);
            seen.push(v);
            adwin.update(v);
        }
        assert_eq!(
            adwin.width(),
            seen.len(),
            "a run at delta = 1e-12 dropped nothing"
        );
        let mean = seen.iter().sum::<f64>() / seen.len() as f64;
        let var = seen.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / seen.len() as f64;
        assert!(
            (adwin.mean() - mean).abs() < 1e-9,
            "{} vs {mean}",
            adwin.mean()
        );
        assert!(
            (adwin.variance() - var).abs() < 1e-9,
            "{} vs {var}",
            adwin.variance()
        );
    }

    #[test]
    fn a_non_finite_value_is_refused_rather_than_admitted() {
        // One NaN inside the window makes every `>` comparison false and disables the detector
        // silently, which is worse than dropping the value.
        let mut rng = SplitMix64::new(7);
        let mut adwin = Adwin::new(0.002);
        for _ in 0..100 {
            adwin.update(gauss(&mut rng));
        }
        let (w, m) = (adwin.width(), adwin.mean());
        assert!(!adwin.update(f64::NAN));
        assert!(!adwin.update(f64::INFINITY));
        assert_eq!(adwin.width(), w);
        assert_eq!(adwin.mean(), m);
        for _ in 0..200 {
            adwin.update(6.0 + gauss(&mut rng));
        }
        assert!(
            adwin.mean() > 4.0,
            "the detector stopped working after a NaN"
        );
    }

    #[test]
    fn delta_is_clamped_into_the_range_the_bound_is_defined_on() {
        // `delta >= 1` makes `ln(2·ln|W|/δ)` negative and `eps` imaginary-then-NaN, at which point
        // every comparison is false and the detector never fires. Clamping is what keeps a caller's
        // `delta = 1.0` merely useless rather than silently broken in the opposite direction.
        let mut rng = SplitMix64::new(9);
        let mut adwin = Adwin::new(2.0);
        for _ in 0..200 {
            adwin.update(gauss(&mut rng));
        }
        for _ in 0..200 {
            adwin.update(9.0 + gauss(&mut rng));
        }
        assert!(
            adwin.mean() > 6.0,
            "a clamped delta must still detect a 9σ shift"
        );
        assert!(Adwin::new(0.0).delta > 0.0);
    }

    #[test]
    fn an_empty_window_reports_zeros_rather_than_dividing_by_its_width() {
        let adwin = Adwin::new(0.002);
        assert_eq!(adwin.width(), 0);
        assert_eq!(adwin.mean(), 0.0);
        assert_eq!(adwin.variance(), 0.0);
        let mut one = Adwin::new(0.002);
        assert!(!one.update(3.0));
        assert_eq!(one.mean(), 3.0);
        assert_eq!(one.variance(), 0.0);
    }
}
