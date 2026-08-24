//! Windowed queries over a stream: what subtracting two cluster features actually costs, and the
//! frame index that avoids paying it.
//!
//! CluStream (Aggarwal et al., VLDB 2003) answers "cluster the window `[t₀, t₁]`" by keeping
//! snapshots of the whole micro-cluster set at pyramidally spaced times and **subtracting** the
//! snapshot at `t₀` from the one at `t₁`. Cluster-feature additivity makes that exact in real
//! arithmetic. It is not exact in floating point, and subtraction is the one CF operation BETULA's
//! stable form does not protect — the whole point of carrying `(n, μ, S)` instead of
//! `(n, Σx, Σx²)` is to never form a difference of two nearly equal large quantities, and an
//! inverse merge forms exactly that, one level up.
//!
//! ## What the subtraction costs, precisely
//!
//! Inverting the Chan merge `S_AB = S_A + S_B + (n_A n_B/n_AB)‖μ_A − μ_B‖²` gives
//!
//! ```text
//! n_B = n_AB − n_A
//! μ_B = (n_AB·μ_AB − n_A·μ_A) / n_B
//! S_B = S_AB − S_A − (n_A n_B / n_AB)·‖μ_A − μ_B‖²
//! ```
//!
//! and both lines are cancellations. The numerator of `μ_B` has absolute error about
//! `u·n_AB·‖μ‖`, so `μ_B` carries relative error about `u·(n_AB/n_B)`. The `S_B` line has absolute
//! error about `u·S_AB`, so its relative error is about `u·(S_AB/S_B)`. Writing `u` for the unit
//! roundoff, a query loses roughly
//!
//! ```text
//! log₁₀( max( n_AB/n_B ,  S_AB/S_B ) )
//! ```
//!
//! decimal digits. **These two ratios are not interchangeable, and the second is the one that
//! bites.** On a stationary stream `S` grows with `n` and they agree. Under drift `S_AB` picks up
//! the between-window displacement term, which has nothing to do with either window's internal
//! spread, and `S_AB/S_B` runs away while `n_AB/n_B` stays small — so a guard written on the point
//! counts passes cleanly on exactly the streams a window query was asked about in the first place.
//! [`Moments::checked_subtract`] therefore conditions on both, a posteriori, and returns a
//! [`SubtractError`] rather than a number with no digits in it.
//!
//! A Cholesky downdate — the usual answer for a sliding-window covariance — is worth naming so it
//! can be ruled out here: it restores the *definiteness* the cancellation destroys, which for the
//! full-covariance feature is a real and separate problem, but it cannot restore digits that were
//! never stored. It fixes a symptom of this, not this.
//!
//! ## The index that does not subtract
//!
//! [`WindowIndex`] stores micro-clusters **per frame** rather than cumulatively, so a window is a
//! *sum* of frames and every combination is the stable Chan merge. The trade is explicit and goes
//! the other way from CluStream's: exact in real arithmetic is given up (a window can only be
//! resolved to the frame boundary) in exchange for an answer that is sound in floating point at
//! every ratio. Older frames are merged pairwise as capacity is reached, so resolution coarsens
//! with age — the pyramidal property, obtained by merging rather than by differencing.

use crate::clustering::{KMeans, kmeans};
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Time support of a summary: the same `(weight, mean, ssd)` contract the spatial feature uses,
/// applied to timestamps, so a windowed summary can say *when* it is from.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeSpan {
    /// Total mass of the timestamps folded in.
    pub weight: f64,
    /// Mass-weighted mean timestamp.
    pub mean: f64,
    /// `Σ w (t − mean)²`.
    pub ssd: f64,
    /// Earliest timestamp seen; `+∞` while empty.
    pub min: f64,
    /// Latest timestamp seen; `−∞` while empty.
    pub max: f64,
}

impl Default for TimeSpan {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSpan {
    /// Empty span.
    pub fn new() -> Self {
        TimeSpan {
            weight: 0.0,
            mean: 0.0,
            ssd: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Fold in one timestamp at weight `w` (weighted Welford).
    pub fn push(&mut self, t: f64, w: f64) {
        if w <= 0.0 {
            return;
        }
        let total = self.weight + w;
        let delta = t - self.mean;
        self.mean += delta * w / total;
        self.ssd += delta * (t - self.mean) * w;
        self.weight = total;
        self.min = self.min.min(t);
        self.max = self.max.max(t);
    }

    /// Absorb another span (Chan parallel update).
    pub fn merge(&mut self, other: &Self) {
        if other.weight <= 0.0 {
            return;
        }
        if self.weight <= 0.0 {
            *self = *other;
            return;
        }
        let total = self.weight + other.weight;
        let delta = other.mean - self.mean;
        self.ssd += other.ssd + delta * delta * self.weight * other.weight / total;
        self.mean += delta * other.weight / total;
        self.weight = total;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Does this span reach into `[t0, t1]` at all?
    pub fn overlaps(&self, t0: f64, t1: f64) -> bool {
        self.weight > 0.0 && self.max >= t0 && self.min <= t1
    }
}

/// Why an inverse merge declined to produce a number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SubtractError {
    /// The part is at least as heavy as the whole, so the remainder has no mass.
    EmptyRemainder,
    /// The recovered scatter came out negative — cancellation caught in the act, no estimate
    /// needed: the answer is not merely inaccurate, it is outside the range of the quantity.
    NegativeScatter,
    /// The remainder is representable but has fewer than the requested significant decimal digits.
    IllConditioned {
        /// `max(n_AB/n_B, S_AB/S_B)`, evaluated on the computed remainder.
        condition: f64,
        /// `−log₁₀(u · condition)` — decimal digits expected to survive.
        digits: f64,
    },
}

/// The stable triple a windowed query manipulates: mass, centroid, total scatter.
///
/// Deliberately not a [`ClusterFeature`]. An inverse merge is only well defined for the scalar
/// scatter — recovering a *matrix* second moment needs the matching matrix downdate, and pretending
/// the two are the same operation is how a diagonal or full feature would silently lose its
/// off-diagonal structure through a window query.
#[derive(Clone, Debug, PartialEq)]
pub struct Moments<R> {
    /// Total mass `n`.
    pub weight: R,
    /// Mass-weighted centroid `μ`.
    pub mean: Vec<R>,
    /// Total scatter `S = Σ w‖x − μ‖²`.
    pub ssd: R,
}

impl<R: Real> Moments<R> {
    /// Read the triple off any cluster feature.
    pub fn from_feature<C: ClusterFeature<R>>(cf: &C) -> Self {
        Moments {
            weight: cf.weight(),
            mean: cf.mean().to_vec(),
            ssd: cf.ssd(),
        }
    }

    /// Absorb another summary (Chan parallel update; the stable direction).
    pub fn merge(&mut self, other: &Self) {
        if other.weight <= R::zero() {
            return;
        }
        if self.weight <= R::zero() {
            *self = other.clone();
            return;
        }
        let total = self.weight + other.weight;
        let d2 = sq_euclidean(&self.mean, &other.mean);
        self.ssd = self.ssd + other.ssd + d2 * self.weight * other.weight / total;
        let share = other.weight / total;
        for (m, &o) in self.mean.iter_mut().zip(&other.mean) {
            *m = *m + (o - *m) * share;
        }
        self.weight = total;
    }

    /// Recover `B` from `AB` and `A`, or say why the result would carry no information.
    ///
    /// `min_digits` is the number of significant decimal digits the caller needs in the answer;
    /// the conditioning estimate is a posteriori, computed on the remainder that came out, because
    /// `S_AB/S_B` cannot be known before `S_B` is. See the module docs for why the point-count
    /// ratio alone is not a sufficient guard.
    pub fn checked_subtract(&self, part: &Self, min_digits: f64) -> Result<Self, SubtractError> {
        let w = self.weight - part.weight;
        if w <= R::zero() {
            return Err(SubtractError::EmptyRemainder);
        }
        let mut mean = vec![R::zero(); self.mean.len()];
        for ((m, &ab), &a) in mean.iter_mut().zip(&self.mean).zip(&part.mean) {
            *m = (ab * self.weight - a * part.weight) / w;
        }
        let d2 = sq_euclidean(&part.mean, &mean);
        let ssd = self.ssd - part.ssd - d2 * part.weight * w / self.weight;
        if ssd < R::zero() {
            return Err(SubtractError::NegativeScatter);
        }
        let mass_ratio = (self.weight / w).to_f64().unwrap_or(f64::INFINITY);
        let scatter_ratio = if ssd > R::zero() {
            (self.ssd / ssd).to_f64().unwrap_or(f64::INFINITY)
        } else {
            f64::INFINITY
        };
        let condition = mass_ratio.max(scatter_ratio).max(1.0);
        let unit = R::epsilon().to_f64().unwrap_or(f64::EPSILON) * 0.5;
        let digits = -(unit * condition).log10();
        if digits < min_digits {
            return Err(SubtractError::IllConditioned { condition, digits });
        }
        Ok(Moments {
            weight: w,
            mean,
            ssd,
        })
    }
}

/// One closed frame: the micro-clusters summarising it, and when it was.
#[derive(Clone, Debug)]
pub struct Frame<C> {
    /// Time support of the points folded into `micros`.
    pub span: TimeSpan,
    /// Micro-clusters over this frame alone — never cumulative, so combining frames is a merge.
    pub micros: Vec<C>,
}

/// A frame index answering window queries by summation.
///
/// Newest frames sit at the end. When the frame count exceeds `capacity` the two *oldest* adjacent
/// frames are merged, so time resolution degrades with age and never with recency — the pyramidal
/// property, reached by merging instead of by differencing.
pub struct WindowIndex<R: Real, C: ClusterFeature<R>> {
    frames: Vec<Frame<C>>,
    capacity: usize,
    max_micros: usize,
    seed: u64,
    _marker: std::marker::PhantomData<R>,
}

impl<R: Real, C: ClusterFeature<R>> WindowIndex<R, C> {
    /// `capacity` frames retained; a frame carrying more than `max_micros` micro-clusters after a
    /// merge is compacted back down to that many. Both are clamped to at least 1.
    pub fn new(capacity: usize, max_micros: usize, seed: u64) -> Self {
        WindowIndex {
            frames: Vec::new(),
            capacity: capacity.max(1),
            max_micros: max_micros.max(1),
            seed,
            _marker: std::marker::PhantomData,
        }
    }

    /// Frames currently retained, oldest first.
    pub fn frames(&self) -> &[Frame<C>] {
        &self.frames
    }

    /// Close a frame. `micros` is that frame's own summary — typically a CF-tree's leaves,
    /// harvested and the tree reset.
    pub fn push_frame(&mut self, span: TimeSpan, micros: Vec<C>) {
        self.frames.push(Frame { span, micros });
        while self.frames.len() > self.capacity && self.frames.len() >= 2 {
            let older = self.frames.remove(0);
            let newer = self.frames.remove(0);
            let mut span = older.span;
            span.merge(&newer.span);
            let mut micros = older.micros;
            micros.extend(newer.micros);
            let micros = compact(micros, self.max_micros, self.seed);
            self.frames.insert(0, Frame { span, micros });
        }
    }

    /// Every micro-cluster from a frame reaching into `[t0, t1]`.
    ///
    /// Frame-granular, and says so: a frame straddling `t0` contributes whole. That is the price of
    /// never subtracting, and it is bounded by the frame width rather than by a condition number.
    pub fn window(&self, t0: f64, t1: f64) -> Vec<C> {
        let mut out = Vec::new();
        for f in &self.frames {
            if f.span.overlaps(t0, t1) {
                out.extend(f.micros.iter().cloned());
            }
        }
        out
    }

    /// Combined `(weight, mean, ssd)` of the window — the quantity CluStream would have reached by
    /// subtracting two snapshots, reached here by merging frames instead.
    pub fn window_moments(&self, t0: f64, t1: f64, dim: usize) -> Moments<R> {
        let mut acc = Moments {
            weight: R::zero(),
            mean: vec![R::zero(); dim],
            ssd: R::zero(),
        };
        for f in &self.frames {
            if !f.span.overlaps(t0, t1) {
                continue;
            }
            for cf in &f.micros {
                acc.merge(&Moments::from_feature(cf));
            }
        }
        acc
    }

    /// Cluster the window into `k` groups. `None` when the window holds fewer than `k`
    /// micro-clusters, which is a question the summary cannot answer rather than one to guess at.
    pub fn cluster_window(&self, t0: f64, t1: f64, k: usize, max_iter: usize) -> Option<KMeans<R>> {
        let micros = self.window(t0, t1);
        if k == 0 || micros.len() < k {
            return None;
        }
        Some(kmeans(&micros, k, max_iter, 3, self.seed))
    }
}

/// Reduce a micro-cluster list to at most `target` entries by clustering it and merging within
/// group. Every merge is the exact CF update, so total mass and total scatter are preserved to
/// floating-point noise — this coarsens resolution, it does not discard mass.
fn compact<R: Real, C: ClusterFeature<R>>(micros: Vec<C>, target: usize, seed: u64) -> Vec<C> {
    if micros.len() <= target {
        return micros;
    }
    let km = kmeans(&micros, target, 50, 1, seed);
    let mut groups: Vec<Option<C>> = vec![None; target];
    for (cf, &l) in micros.iter().zip(&km.labels) {
        match &mut groups[l] {
            Some(acc) => acc.merge(cf),
            slot => *slot = Some(cf.clone()),
        }
    }
    groups.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::feature::Spherical;

    fn cf<R: Real>(points: &[[f64; 2]]) -> Spherical<R> {
        let mut c = Spherical::new(2);
        for p in points {
            let row: Vec<R> = p.iter().map(|&v| R::from_f64(v).unwrap()).collect();
            c.push(&row, R::one());
        }
        c
    }

    fn cloud<R: Real>(rng: &mut SplitMix64, n: usize, cx: f64, cy: f64, s: f64) -> Spherical<R> {
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|_| [cx + s * rng.gauss(), cy + s * rng.gauss()])
            .collect();
        cf(&pts)
    }

    // ───────────────────────────── the subtraction and its conditioning ─────────────────────────

    #[test]
    fn merge_then_subtract_round_trips_where_the_parts_are_comparable() {
        let mut rng = SplitMix64::new(1);
        let a = Moments::from_feature(&cloud::<f64>(&mut rng, 400, 0.0, 0.0, 1.0));
        let b = Moments::from_feature(&cloud::<f64>(&mut rng, 400, 1.0, 0.5, 1.0));
        let mut ab = a.clone();
        ab.merge(&b);
        let got = ab.checked_subtract(&a, 8.0).expect("well conditioned");
        assert!((got.weight - b.weight).abs() < 1e-9);
        assert!((got.ssd - b.ssd).abs() < 1e-9 * b.ssd);
        for (g, w) in got.mean.iter().zip(&b.mean) {
            assert!((g - w).abs() < 1e-9);
        }
    }

    #[test]
    fn the_scatter_ratio_not_the_mass_ratio_is_the_condition_number() {
        // A drifting stream in miniature: two halves of equal mass, one tight, sitting far apart.
        // The mass ratio is 2 -- a guard written on point counts sees nothing at all -- while the
        // displacement term makes S_AB enormous next to the tight half's own S_B.
        let mut rng = SplitMix64::new(2);
        let a = Moments::from_feature(&cloud::<f64>(&mut rng, 500, 0.0, 0.0, 1.0));
        let b = Moments::from_feature(&cloud::<f64>(&mut rng, 500, 1.0e6, 0.0, 1.0e-4));
        let mut ab = a.clone();
        ab.merge(&b);

        let mass_ratio = ab.weight / (ab.weight - a.weight);
        assert!((mass_ratio - 2.0).abs() < 1e-9, "mass ratio {mass_ratio}");

        let err = ab.checked_subtract(&a, 15.0).unwrap_err();
        let SubtractError::IllConditioned { condition, digits } = err else {
            panic!("expected an ill-conditioned verdict, got {err:?}");
        };
        assert!(
            condition > 1e8 * mass_ratio,
            "condition {condition} tracked the mass ratio, not the scatter"
        );
        // And the verdict is not merely pessimistic: the answer really has lost those digits.
        let got = ab.checked_subtract(&a, 0.0).expect("representable");
        let rel = (got.ssd - b.ssd).abs() / b.ssd;
        assert!(
            rel > 1e-6,
            "relative error {rel} — the fixture is not ill-conditioned after all"
        );
        assert!(digits < 8.0, "{digits} digits claimed to survive");
    }

    #[test]
    fn the_same_window_survives_in_f64_and_does_not_in_f32() {
        // The point-count ratio is identical in both; only the unit roundoff differs, so this
        // isolates precision from conditioning.
        let mut rng = SplitMix64::new(3);
        let big: Spherical<f64> = cloud(&mut rng, 20_000, 0.0, 0.0, 1.0);
        let small: Spherical<f64> = cloud(&mut rng, 20, 0.3, -0.2, 1.0);
        let to32 = |c: &Spherical<f64>| -> Moments<f32> {
            let mut out: Spherical<f32> = Spherical::new(2);
            let mean: Vec<f32> = c.mean().iter().map(|&v| v as f32).collect();
            out.push(&mean, c.weight() as f32);
            Moments {
                weight: c.weight() as f32,
                mean,
                ssd: c.ssd() as f32,
            }
        };
        let (a64, b64) = (Moments::from_feature(&big), Moments::from_feature(&small));
        let mut ab64 = a64.clone();
        ab64.merge(&b64);
        let (a32, b32) = (to32(&big), to32(&small));
        let mut ab32 = a32.clone();
        ab32.merge(&b32);

        assert!(ab64.checked_subtract(&a64, 6.0).is_ok());
        assert!(
            ab32.checked_subtract(&a32, 6.0).is_err(),
            "f32 claimed six digits at a condition number f64 barely covers"
        );
    }

    #[test]
    fn an_impossible_remainder_is_named_rather_than_returned() {
        let mut rng = SplitMix64::new(4);
        let a = Moments::from_feature(&cloud::<f64>(&mut rng, 100, 0.0, 0.0, 1.0));
        assert_eq!(
            a.checked_subtract(&a, 0.0).unwrap_err(),
            SubtractError::EmptyRemainder
        );
        // A "part" that was never inside the whole: the scatter identity has no solution, and the
        // arithmetic says so by going negative rather than by returning a plausible small number.
        let tight = Moments::from_feature(&cloud::<f64>(&mut rng, 50, 0.0, 0.0, 1e-9));
        let mut whole = tight.clone();
        whole.merge(&Moments::from_feature(&cloud::<f64>(
            &mut rng, 60, 0.0, 0.0, 1e-9,
        )));
        let wide = Moments::from_feature(&cloud::<f64>(&mut rng, 50, 0.0, 0.0, 5.0));
        assert_eq!(
            whole.checked_subtract(&wide, 0.0).unwrap_err(),
            SubtractError::NegativeScatter
        );
    }

    #[test]
    fn summing_frames_answers_the_window_the_subtraction_could_not() {
        // Same drifting fixture as the conditioning test, asked of the index instead. Summation has
        // no condition number to report because it never forms the difference.
        let mut rng = SplitMix64::new(2);
        let old: Spherical<f64> = cloud(&mut rng, 500, 0.0, 0.0, 1.0);
        let recent: Spherical<f64> = cloud(&mut rng, 500, 1.0e6, 0.0, 1.0e-4);
        let want = Moments::from_feature(&recent);

        let mut idx: WindowIndex<f64, Spherical<f64>> = WindowIndex::new(8, 64, 0);
        let mut s0 = TimeSpan::new();
        s0.push(0.0, 500.0);
        let mut s1 = TimeSpan::new();
        s1.push(10.0, 500.0);
        idx.push_frame(s0, vec![old]);
        idx.push_frame(s1, vec![recent]);

        let got = idx.window_moments(5.0, 20.0, 2);
        assert!((got.weight - want.weight).abs() < 1e-9);
        let rel = (got.ssd - want.ssd).abs() / want.ssd;
        assert!(rel < 1e-12, "summation lost digits too: {rel}");
    }

    // ───────────────────────────────────── the index ────────────────────────────────────────────

    #[test]
    fn a_window_takes_the_frames_that_reach_into_it_and_no_others() {
        let mut idx: WindowIndex<f64, Spherical<f64>> = WindowIndex::new(16, 32, 0);
        for t in 0..5 {
            let mut span = TimeSpan::new();
            span.push(t as f64, 10.0);
            idx.push_frame(span, vec![cf::<f64>(&[[t as f64, 0.0]])]);
        }
        assert_eq!(idx.window(1.0, 3.0).len(), 3);
        assert_eq!(idx.window(-5.0, -1.0).len(), 0);
        assert_eq!(idx.window(0.0, 100.0).len(), 5);
    }

    #[test]
    fn compaction_coarsens_the_oldest_frames_and_conserves_their_mass() {
        let mut rng = SplitMix64::new(5);
        let mut idx: WindowIndex<f64, Spherical<f64>> = WindowIndex::new(3, 4, 7);
        let mut total = 0.0;
        for t in 0..8 {
            let micros: Vec<Spherical<f64>> = (0..6)
                .map(|_| cloud(&mut rng, 5, t as f64, 0.0, 0.5))
                .collect();
            total += micros.iter().map(|c| c.weight()).sum::<f64>();
            let mut span = TimeSpan::new();
            span.push(t as f64, 30.0);
            idx.push_frame(span, micros);
        }
        assert!(idx.frames().len() <= 3);
        let kept: f64 = idx
            .frames()
            .iter()
            .flat_map(|f| f.micros.iter())
            .map(|c| c.weight())
            .sum();
        assert!((kept - total).abs() < 1e-9, "{kept} vs {total}");
        assert!(
            idx.frames()[0].micros.len() <= 4,
            "oldest frame not compacted"
        );
        // The newest frame is untouched: resolution degrades with age, never with recency.
        assert_eq!(idx.frames().last().unwrap().micros.len(), 6);
    }

    #[test]
    fn clustering_a_window_sees_only_what_is_inside_it() {
        let mut rng = SplitMix64::new(6);
        let mut idx: WindowIndex<f64, Spherical<f64>> = WindowIndex::new(16, 64, 3);
        // Frames 0-1 hold two groups; frames 2-3 hold two different ones, far away.
        for (t, centers) in [
            (0.0, [(0.0, 0.0), (10.0, 0.0)]),
            (1.0, [(0.0, 0.0), (10.0, 0.0)]),
            (2.0, [(0.0, 60.0), (10.0, 60.0)]),
            (3.0, [(0.0, 60.0), (10.0, 60.0)]),
        ] {
            let micros: Vec<Spherical<f64>> = centers
                .iter()
                .flat_map(|&(cx, cy)| (0..4).map(move |_| (cx, cy)))
                .map(|(cx, cy)| cloud(&mut rng, 20, cx, cy, 0.3))
                .collect();
            let mut span = TimeSpan::new();
            span.push(t, 160.0);
            idx.push_frame(span, micros);
        }
        let early = idx.cluster_window(0.0, 1.5, 2, 50).expect("two groups");
        for c in &early.centers {
            assert!(c[1].abs() < 5.0, "early window reached forward: {c:?}");
        }
        let late = idx.cluster_window(2.0, 3.5, 2, 50).expect("two groups");
        for c in &late.centers {
            assert!((c[1] - 60.0).abs() < 5.0, "late window reached back: {c:?}");
        }
        assert!(idx.cluster_window(0.0, 3.5, 999, 50).is_none());
        assert!(idx.cluster_window(-10.0, -5.0, 2, 50).is_none());
    }

    #[test]
    fn the_time_span_is_the_same_contract_as_the_spatial_one() {
        let mut a = TimeSpan::new();
        let mut b = TimeSpan::new();
        let mut whole = TimeSpan::new();
        for t in [1.0, 2.0, 3.0, 4.0] {
            a.push(t, 1.0);
            whole.push(t, 1.0);
        }
        for t in [10.0, 12.0] {
            b.push(t, 1.0);
            whole.push(t, 1.0);
        }
        a.merge(&b);
        assert!((a.mean - whole.mean).abs() < 1e-12);
        assert!((a.ssd - whole.ssd).abs() < 1e-12);
        assert_eq!((a.min, a.max), (1.0, 12.0));
        assert!(a.overlaps(5.0, 11.0) && !a.overlaps(20.0, 30.0));
        assert!(!TimeSpan::new().overlaps(0.0, 1.0));
    }
}
