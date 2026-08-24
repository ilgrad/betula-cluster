//! The cluster feature, generalised from squared Euclidean to any Bregman divergence.
//!
//! For `d_φ(x, y) = φ(x) − φ(y) − ⟨∇φ(y), x − y⟩` the within-cluster **Bregman information**
//! `S_φ(A) = Σ_{i∈A} w_i d_φ(x_i, μ_A)` (Banerjee et al. 2005) obeys exactly the identities the
//! Euclidean scatter does — Maxima residual 0 for generic differentiable `φ`
//! (`math_improove` Lane A):
//!
//! ```text
//! bias–variance   Σ w_i d_φ(x_i, y) = S_φ + W·d_φ(μ, y)
//! mean merge      μ_AB = μ_A + (n_B/n_AB)(μ_B − μ_A)          — unchanged from Euclidean
//! merge           S_AB = S_A + S_B + n_A d_φ(μ_A, μ_AB) + n_B d_φ(μ_B, μ_AB)
//! Welford add     S′   = S + w·d_φ(x, μ′) + W·d_φ(μ, μ′)
//! ```
//!
//! The mean merge being unchanged is Banerjee's characterisation, not a coincidence: the arithmetic
//! mean is the right-sided Bregman centroid for *every* `φ`. So `(n, μ, S_φ)` is a commutative
//! monoid for every `φ`, the tree machinery is untouched, and only the `S` update and the distance
//! implementations change.
//!
//! # The trap the trait design exists to avoid
//!
//! The closed form `S_φ = Σ w_i φ(x_i) − W φ(μ)` is the exact analogue of BIRCH's `SS − n μ²` and
//! cancels for the same reason. **The recurrence alone does not save you**: both of its new terms
//! evaluate `d_φ` at arguments that nearly coincide (`μ` against `μ′` differ by `O(w/W)`), and
//! evaluating `d_φ` from its definition there subtracts two nearly equal values of `φ` — the same
//! cancellation one level down. Measured against a 60-digit reference at f64, tight cluster,
//! scales `1e0 … 1e8` (`local/scratch/bregman_stability.py`):
//!
//! ```text
//! divergence      naive Σφ − Wφ(μ)   recurrence, d_φ expanded   recurrence, d_φ closed form
//! euclidean       1.5e-6 … 2.5e-4    1.8e-5 … 1.2e-4            2.3e-11 … 7.9e-11
//! KL              9.7e-5 … 5.6e-3    2.8e-11 … 3.9e-3           1.8e-12 … 6.6e-11
//! Itakura–Saito   4.7e-4 … 5.2e-3    4.7e-11 … 5.3e-3           1.7e-11 … 9.5e-11
//! logistic        1.2e-4 … 1.00      2.5e-4 … 9.3e+10           3.7e-11 … 1.4e-2
//! ```
//!
//! The logistic row is the one that settles the design: the naive form loses *everything*
//! (relative error 1), and the expanded recurrence does not merely lose precision, it **diverges**
//! — the computed information comes out 10¹¹ times too large.
//!
//! Hence [`BregmanDivergence`] cannot be `fn phi(t)` with a derived default `d_φ`. It requires a
//! hand-written [`BregmanDivergence::divergence`] whose contract is *accuracy when `x ≈ y`*, and
//! keeps [`BregmanDivergence::phi`] as a test oracle to be used only at well-separated arguments.
//! That contract is the whole content of the BETULA paper, one level up.

use crate::distance::CFDistance;
use crate::feature::ClusterFeature;
use crate::types::Real;
use std::marker::PhantomData;

/// A Bregman divergence, supplied in a cancellation-free closed form.
///
/// Implementors are zero-sized: the divergence is a type, not a value, so a feature parameterised
/// by one costs the same as the Euclidean feature it generalises.
pub trait BregmanDivergence<R: Real>: Copy + Default + Send + Sync + 'static {
    /// The scalar divergence `d_φ(x, y)`.
    ///
    /// **Contract: accurate when `x ≈ y`.** This is the whole reason the trait exists. Write the
    /// `log1p` / `expm1` form; never `phi(x) - phi(y) - grad(y)*(x - y)`.
    fn divergence(&self, x: R, y: R) -> R;

    /// `φ′(t)`. Used by the naive oracle and by callers that need the dual coordinate.
    fn grad(&self, t: R) -> R;

    /// Is `t` inside the domain of `φ`? KL and Itakura–Saito need `t > 0`, logistic `t ∈ (0,1)`.
    fn is_valid(&self, t: R) -> bool;

    /// `φ(t)` itself. **Test oracle only.** Every use of it in library code reintroduces the
    /// cancellation the closed forms above exist to avoid.
    fn phi(&self, t: R) -> R;

    /// Separable extension to vectors: `Σ_j d_φ(x_j, y_j)`.
    #[inline]
    fn vector(&self, x: &[R], y: &[R]) -> R {
        x.iter()
            .zip(y)
            .map(|(&xi, &yi)| self.divergence(xi, yi))
            .sum()
    }
}

/// `φ(t) = t²` — squared Euclidean, `d = (x − y)²`. The control: a [`BregmanCf`] over this must
/// reproduce [`crate::feature::Spherical`] exactly.
#[derive(Clone, Copy, Default, Debug)]
pub struct SquaredEuclidean;

impl<R: Real> BregmanDivergence<R> for SquaredEuclidean {
    #[inline]
    fn divergence(&self, x: R, y: R) -> R {
        let d = x - y;
        d * d
    }
    #[inline]
    fn grad(&self, t: R) -> R {
        (R::one() + R::one()) * t
    }
    #[inline]
    fn is_valid(&self, t: R) -> bool {
        t.is_finite()
    }
    #[inline]
    fn phi(&self, t: R) -> R {
        t * t
    }
}

/// `φ(t) = t·log t` on `t > 0` — the generalised I-divergence `x·log(x/y) − (x − y)`, for counts,
/// histograms and topic weights. Reduces to Kullback–Leibler on the simplex, where the linear term
/// sums to zero.
#[derive(Clone, Copy, Default, Debug)]
pub struct KullbackLeibler;

impl<R: Real> BregmanDivergence<R> for KullbackLeibler {
    #[inline]
    fn divergence(&self, x: R, y: R) -> R {
        let u = (x - y) / y;
        x * u.ln_1p() - (x - y)
    }
    #[inline]
    fn grad(&self, t: R) -> R {
        t.ln() + R::one()
    }
    #[inline]
    fn is_valid(&self, t: R) -> bool {
        t.is_finite() && t > R::zero()
    }
    #[inline]
    fn phi(&self, t: R) -> R {
        t * t.ln()
    }
}

/// `φ(t) = −log t` on `t > 0` — Itakura–Saito `x/y − log(x/y) − 1`, for power spectra and other
/// non-negative signals where relative error is what matters.
#[derive(Clone, Copy, Default, Debug)]
pub struct ItakuraSaito;

impl<R: Real> BregmanDivergence<R> for ItakuraSaito {
    #[inline]
    fn divergence(&self, x: R, y: R) -> R {
        let u = (x - y) / y;
        u - u.ln_1p()
    }
    #[inline]
    fn grad(&self, t: R) -> R {
        -R::one() / t
    }
    #[inline]
    fn is_valid(&self, t: R) -> bool {
        t.is_finite() && t > R::zero()
    }
    #[inline]
    fn phi(&self, t: R) -> R {
        -t.ln()
    }
}

/// `φ(t) = t·log t + (1−t)·log(1−t)` on `(0, 1)` — the logistic loss, for binary features and
/// probabilities. The divergence that made the closed-form contract non-negotiable: expanding it
/// from the definition overstates the information by a factor of 10¹¹ on a tight cluster.
#[derive(Clone, Copy, Default, Debug)]
pub struct Logistic;

impl<R: Real> BregmanDivergence<R> for Logistic {
    #[inline]
    fn divergence(&self, x: R, y: R) -> R {
        let one = R::one();
        let u = (x - y) / y;
        let v = (y - x) / (one - y);
        x * u.ln_1p() + (one - x) * v.ln_1p()
    }
    #[inline]
    fn grad(&self, t: R) -> R {
        (t / (R::one() - t)).ln()
    }
    #[inline]
    fn is_valid(&self, t: R) -> bool {
        t.is_finite() && t > R::zero() && t < R::one()
    }
    #[inline]
    fn phi(&self, t: R) -> R {
        let one = R::one();
        t * t.ln() + (one - t) * (one - t).ln()
    }
}

/// A cluster feature carrying `(n, μ, S_φ)` for an arbitrary Bregman divergence.
///
/// [`ClusterFeature::ssd`] returns the Bregman information `S_φ`, which coincides with the ordinary
/// sum of squared deviations exactly when `B = SquaredEuclidean`.
#[derive(Clone, Debug)]
pub struct BregmanCf<R: Real, B: BregmanDivergence<R>> {
    w: R,
    mean: Vec<R>,
    info: R,
    div: PhantomData<B>,
}

impl<R: Real, B: BregmanDivergence<R>> BregmanCf<R, B> {
    /// Mean Bregman information `S_φ / n` — the analogue of the leaf's mean squared radius, and
    /// what an absorption threshold on this feature compares against. Zero when empty.
    pub fn mean_information(&self) -> R {
        if self.w <= R::zero() {
            R::zero()
        } else {
            self.info / self.w
        }
    }

    /// `Σ w_i d_φ(x_i, y)` for an arbitrary `y`, from the stored moments alone — the bias–variance
    /// identity `S_φ + W·d_φ(μ, y)`. Exact for every `φ`, and the reason a Bregman k-means can
    /// score a candidate centre without revisiting a single point.
    pub fn information_about(&self, y: &[R]) -> R {
        if self.w <= R::zero() {
            return R::zero();
        }
        self.info + self.w * B::default().vector(&self.mean, y)
    }
}

impl<R: Real, B: BregmanDivergence<R>> ClusterFeature<R> for BregmanCf<R, B> {
    fn new(dim: usize) -> Self {
        Self {
            w: R::zero(),
            mean: vec![R::zero(); dim],
            info: R::zero(),
            div: PhantomData,
        }
    }
    fn dim(&self) -> usize {
        self.mean.len()
    }
    fn weight(&self) -> R {
        self.w
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        self.info
    }
    /// Isotropic proxy `S_φ / (n·d)`. For a non-Euclidean `φ` this is mean Bregman information per
    /// coordinate, not a variance — the Gaussian heads that read it are asking the wrong question
    /// of this feature.
    fn variance(&self, _d: usize) -> R {
        if self.w <= R::zero() {
            R::zero()
        } else {
            self.info / self.w / R::from_usize(self.mean.len()).unwrap()
        }
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        // The empty case is not just an optimisation. A zero mean is outside the domain of KL,
        // Itakura-Saito and logistic, so evaluating d_φ(0, x) there yields NaN — which no
        // multiplication by a zero weight would clean up afterwards.
        if self.w <= R::zero() {
            self.mean.copy_from_slice(x);
            self.info = R::zero();
            self.w = w;
            return;
        }
        let div = B::default();
        debug_assert!(x.iter().all(|&t| div.is_valid(t)), "point outside dom φ");
        let w_new = self.w + w;
        let factor = w / w_new;
        let mut delta = R::zero();
        for (m, &xi) in self.mean.iter_mut().zip(x) {
            let old = *m;
            let new = old + factor * (xi - old);
            delta = delta + w * div.divergence(xi, new) + self.w * div.divergence(old, new);
            *m = new;
        }
        self.info = self.info + delta;
        self.w = w_new;
    }
    fn merge(&mut self, other: &Self) {
        if other.w <= R::zero() {
            return;
        }
        if self.w <= R::zero() {
            *self = other.clone();
            return;
        }
        let div = B::default();
        let w_new = self.w + other.w;
        let factor = other.w / w_new;
        let mut delta = R::zero();
        for (m, &om) in self.mean.iter_mut().zip(&other.mean) {
            let old = *m;
            let new = old + factor * (om - old);
            delta = delta + self.w * div.divergence(old, new) + other.w * div.divergence(om, new);
            *m = new;
        }
        self.info = self.info + other.info + delta;
        self.w = w_new;
    }
    fn decay(&mut self, factor: R) {
        self.w = self.w * factor;
        self.info = self.info * factor;
    }
}

/// D0_φ — the divergence between a feature's centroid and a point, the routing measure.
///
/// Lives here rather than in [`crate::distance`] because it is not feature-agnostic: it has to know
/// which `φ` the feature was built with, so it is parameterised by the same `B`. Everything in
/// `distance.rs` works for any [`ClusterFeature`]; these two work for exactly one.
///
/// **Asymmetric on purpose.** `point` evaluates `d_φ(x, μ)` — the point first, the centroid second
/// — which is the orientation under which the arithmetic mean is the optimal representative
/// (Banerjee et al.). Reversing it would minimise a different objective.
pub struct BregmanCentroid<B> {
    div: PhantomData<B>,
}

impl<B> BregmanCentroid<B> {
    /// The measure over divergence `B`.
    pub fn new() -> Self {
        Self { div: PhantomData }
    }
}

impl<B> Default for BregmanCentroid<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Real, B: BregmanDivergence<R>> CFDistance<R, BregmanCf<R, B>> for BregmanCentroid<B> {
    fn point(&self, cf: &BregmanCf<R, B>, x: &[R]) -> R {
        B::default().vector(x, cf.mean())
    }
    fn between(&self, a: &BregmanCf<R, B>, b: &BregmanCf<R, B>) -> R {
        B::default().vector(a.mean(), b.mean())
    }
}

/// D4_φ / Bregman-Ward — the exact increase in Bregman information from a merge.
///
/// `D(A,B) = n_A d_φ(μ_A, μ_AB) + n_B d_φ(μ_B, μ_AB) = S_AB − S_A − S_B`, straight from the merge
/// identity. Both stored informations cancel, so this is a pure centroid measure and inherits the
/// stability of the closed form rather than of the difference. At `φ(t) = t²` it collapses to
/// `(n_A n_B / n_AB)‖Δμ‖²`, which is [`crate::distance::VarianceIncrease`].
///
/// **Not reducible for `d ≥ 2` outside squared Euclidean**, so an agglomerative head over this
/// needs Anderberg, not the nearest-neighbour chain `ward.rs` uses. That is measured rather than
/// feared: at `d = 20, m = 12` a chain builds a different dendrogram in 1.0 % of Itakura–Saito
/// instances and 1.2 % of exponential ones, the rate grows with `m`, and when it fires the answer
/// is destroyed (ARI 0.10 at `k = 4`, one cell at −0.11). `docs/adr/002-bregman-ward-anderberg.md`.
pub struct BregmanIncrease<B> {
    div: PhantomData<B>,
}

impl<B> BregmanIncrease<B> {
    /// The measure over divergence `B`.
    pub fn new() -> Self {
        Self { div: PhantomData }
    }
}

impl<B> Default for BregmanIncrease<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Real, B: BregmanDivergence<R>> CFDistance<R, BregmanCf<R, B>> for BregmanIncrease<B> {
    fn point(&self, cf: &BregmanCf<R, B>, x: &[R]) -> R {
        let n = cf.weight();
        if n <= R::zero() {
            return R::zero();
        }
        let div = B::default();
        let factor = R::one() / (n + R::one());
        cf.mean()
            .iter()
            .zip(x)
            .map(|(&m, &xi)| {
                let merged = m + factor * (xi - m);
                n * div.divergence(m, merged) + div.divergence(xi, merged)
            })
            .sum()
    }
    fn between(&self, a: &BregmanCf<R, B>, b: &BregmanCf<R, B>) -> R {
        let (na, nb) = (a.weight(), b.weight());
        let nab = na + nb;
        if na <= R::zero() || nb <= R::zero() {
            return R::zero();
        }
        let div = B::default();
        let factor = nb / nab;
        a.mean()
            .iter()
            .zip(b.mean())
            .map(|(&ma, &mb)| {
                let merged = ma + factor * (mb - ma);
                na * div.divergence(ma, merged) + nb * div.divergence(mb, merged)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::Spherical;

    /// A deterministic LCG, so the fixtures are reproducible without pulling the RNG in.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn span(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
    }

    fn points(rng: &mut Lcg, n: usize, dim: usize, lo: f64, hi: f64) -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| (0..dim).map(|_| rng.span(lo, hi)).collect())
            .collect()
    }

    /// `Σ w_i d_φ(x_i, μ)` computed directly from the points — the definition the recurrence claims
    /// to reproduce, and the only reference that does not go through the recurrence itself.
    fn brute_information<B: BregmanDivergence<f64>>(
        pts: &[Vec<f64>],
        w: &[f64],
        mu: &[f64],
    ) -> f64 {
        let div = B::default();
        pts.iter()
            .zip(w)
            .map(|(p, &wi)| wi * div.vector(p, mu))
            .sum()
    }

    fn weighted_mean(pts: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
        let total: f64 = w.iter().sum();
        let dim = pts[0].len();
        (0..dim)
            .map(|j| pts.iter().zip(w).map(|(p, &wi)| wi * p[j]).sum::<f64>() / total)
            .collect()
    }

    fn build<B: BregmanDivergence<f64>>(pts: &[Vec<f64>], w: &[f64]) -> BregmanCf<f64, B> {
        let mut cf = BregmanCf::<f64, B>::new(pts[0].len());
        for (p, &wi) in pts.iter().zip(w) {
            cf.push(p, wi);
        }
        cf
    }

    #[test]
    fn the_squared_euclidean_instantiation_reproduces_the_spherical_feature() {
        // The regression test that matters: φ(t) = t² must give back the feature that already
        // ships, so nothing about the Euclidean path is a new implementation.
        let mut rng = Lcg(1);
        let pts = points(&mut rng, 200, 7, -3.0, 3.0);
        let w: Vec<f64> = (0..200).map(|_| rng.span(0.5, 4.0)).collect();

        let bregman = build::<SquaredEuclidean>(&pts, &w);
        let mut spherical = Spherical::<f64>::new(7);
        for (p, &wi) in pts.iter().zip(&w) {
            spherical.push(p, wi);
        }

        assert!((bregman.weight() - spherical.weight()).abs() < 1e-12);
        assert!((bregman.ssd() - spherical.ssd()).abs() < 1e-9 * spherical.ssd().abs());
        for (a, b) in bregman.mean().iter().zip(spherical.mean()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn the_welford_recurrence_reproduces_the_definition_for_every_divergence() {
        let mut rng = Lcg(2);
        let pts = points(&mut rng, 150, 5, 0.3, 6.0);
        let w: Vec<f64> = (0..150).map(|_| rng.span(0.5, 3.0)).collect();
        let mu = weighted_mean(&pts, &w);

        macro_rules! check {
            ($b:ty) => {{
                let cf = build::<$b>(&pts, &w);
                let want = brute_information::<$b>(&pts, &w, &mu);
                let got = cf.ssd();
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs(),
                    "{}: {got} vs {want}",
                    stringify!($b)
                );
            }};
        }
        check!(SquaredEuclidean);
        check!(KullbackLeibler);
        check!(ItakuraSaito);

        // The logistic domain is (0, 1), so it gets its own fixture rather than a rescaled one.
        let pts = points(&mut rng, 150, 5, 0.05, 0.95);
        let mu = weighted_mean(&pts, &w);
        let cf = build::<Logistic>(&pts, &w);
        let want = brute_information::<Logistic>(&pts, &w, &mu);
        assert!((cf.ssd() - want).abs() <= 1e-9 * want.abs());
    }

    #[test]
    fn merge_reproduces_the_definition_and_is_commutative() {
        let mut rng = Lcg(3);
        let left = points(&mut rng, 60, 4, 0.4, 5.0);
        let right = points(&mut rng, 90, 4, 2.0, 9.0);
        let wl: Vec<f64> = (0..60).map(|_| rng.span(0.5, 3.0)).collect();
        let wr: Vec<f64> = (0..90).map(|_| rng.span(0.5, 3.0)).collect();

        macro_rules! check {
            ($b:ty) => {{
                let mut ab = build::<$b>(&left, &wl);
                let b = build::<$b>(&right, &wr);
                let mut ba = b.clone();
                ab.merge(&b);
                ba.merge(&build::<$b>(&left, &wl));

                let all: Vec<Vec<f64>> = left.iter().chain(&right).cloned().collect();
                let aw: Vec<f64> = wl.iter().chain(&wr).copied().collect();
                let mu = weighted_mean(&all, &aw);
                let want = brute_information::<$b>(&all, &aw, &mu);

                assert!(
                    (ab.ssd() - want).abs() <= 1e-9 * want.abs(),
                    "{}: merge {} vs definition {want}",
                    stringify!($b),
                    ab.ssd()
                );
                assert!(
                    (ab.ssd() - ba.ssd()).abs() <= 1e-9 * want.abs(),
                    "{}: merge is not commutative",
                    stringify!($b)
                );
            }};
        }
        check!(SquaredEuclidean);
        check!(KullbackLeibler);
        check!(ItakuraSaito);
    }

    #[test]
    fn the_bias_variance_identity_scores_a_centre_without_the_points() {
        let mut rng = Lcg(4);
        let pts = points(&mut rng, 120, 6, 0.5, 7.0);
        let w: Vec<f64> = (0..120).map(|_| rng.span(0.5, 2.0)).collect();
        let y: Vec<f64> = (0..6).map(|_| rng.span(0.5, 7.0)).collect();

        macro_rules! check {
            ($b:ty) => {{
                let cf = build::<$b>(&pts, &w);
                let want = brute_information::<$b>(&pts, &w, &y);
                let got = cf.information_about(&y);
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs(),
                    "{}: {got} vs {want}",
                    stringify!($b)
                );
            }};
        }
        check!(SquaredEuclidean);
        check!(KullbackLeibler);
        check!(ItakuraSaito);
    }

    #[test]
    fn every_divergence_is_non_negative_and_vanishes_only_on_the_diagonal() {
        let mut rng = Lcg(5);
        macro_rules! check {
            ($b:ty, $lo:expr, $hi:expr) => {{
                let div = <$b>::default();
                for _ in 0..20_000 {
                    let x = rng.span($lo, $hi);
                    let y = rng.span($lo, $hi);
                    let d = BregmanDivergence::<f64>::divergence(&div, x, y);
                    assert!(d >= 0.0, "{}: d({x},{y}) = {d}", stringify!($b));
                    assert_eq!(BregmanDivergence::<f64>::divergence(&div, x, x), 0.0);
                    if (x - y).abs() > 1e-3 {
                        assert!(d > 0.0, "{}: d({x},{y}) vanished", stringify!($b));
                    }
                }
            }};
        }
        check!(SquaredEuclidean, -4.0, 4.0);
        check!(KullbackLeibler, 0.1, 8.0);
        check!(ItakuraSaito, 0.1, 8.0);
        check!(Logistic, 0.05, 0.95);
    }

    #[test]
    fn the_closed_form_beats_the_naive_one_where_the_naive_one_cancels() {
        // The measurement that dictated the trait's shape, as a test: on a tight cluster far from
        // the origin, `Σ w φ(x) − W φ(μ)` loses the answer while the recurrence keeps it.
        let mut rng = Lcg(6);
        let base = 1.0e6;
        let pts: Vec<Vec<f64>> = (0..64)
            .map(|_| {
                (0..3)
                    .map(|_| base * (1.0 + 1e-6 * rng.span(-1.0, 1.0)))
                    .collect()
            })
            .collect();
        let w = vec![1.0; 64];
        let mu = weighted_mean(&pts, &w);

        let div = KullbackLeibler;
        let total: f64 = w.iter().sum();
        let naive: f64 = pts
            .iter()
            .zip(&w)
            .map(|(p, &wi)| wi * p.iter().map(|&t| div.phi(t)).sum::<f64>())
            .sum::<f64>()
            - total * mu.iter().map(|&t| div.phi(t)).sum::<f64>();

        let cf = build::<KullbackLeibler>(&pts, &w);
        let want = brute_information::<KullbackLeibler>(&pts, &w, &mu);

        let naive_err = (naive - want).abs() / want.abs();
        let recurrence_err = (cf.ssd() - want).abs() / want.abs();
        assert!(
            recurrence_err < 1e-9,
            "recurrence lost it: {recurrence_err:e}"
        );
        assert!(
            naive_err > 1e4 * recurrence_err.max(f64::MIN_POSITIVE),
            "the naive form was supposed to cancel here: naive {naive_err:e} vs \
             recurrence {recurrence_err:e}"
        );
    }

    #[test]
    fn the_bregman_increase_is_exactly_the_information_a_merge_adds() {
        let mut rng = Lcg(8);
        let left = points(&mut rng, 45, 4, 0.4, 5.0);
        let right = points(&mut rng, 70, 4, 1.5, 8.0);
        let wl: Vec<f64> = (0..45).map(|_| rng.span(0.5, 3.0)).collect();
        let wr: Vec<f64> = (0..70).map(|_| rng.span(0.5, 3.0)).collect();

        macro_rules! check {
            ($b:ty) => {{
                let a = build::<$b>(&left, &wl);
                let b = build::<$b>(&right, &wr);
                let mut ab = a.clone();
                ab.merge(&b);
                let want = ab.ssd() - a.ssd() - b.ssd();
                let got = BregmanIncrease::<$b>::new().between(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs(),
                    "{}: D4 {got} vs S_AB - S_A - S_B {want}",
                    stringify!($b)
                );
            }};
        }
        check!(SquaredEuclidean);
        check!(KullbackLeibler);
        check!(ItakuraSaito);
    }

    #[test]
    fn absorbing_one_point_costs_what_the_increase_says_it_will() {
        let mut rng = Lcg(9);
        let pts = points(&mut rng, 30, 3, 0.5, 6.0);
        let w = vec![1.0; 30];
        let x: Vec<f64> = (0..3).map(|_| rng.span(0.5, 6.0)).collect();

        macro_rules! check {
            ($b:ty) => {{
                let cf = build::<$b>(&pts, &w);
                let mut after = cf.clone();
                after.push(&x, 1.0);
                let want = after.ssd() - cf.ssd();
                let got = BregmanIncrease::<$b>::new().point(&cf, &x);
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs(),
                    "{}: {got} vs {want}",
                    stringify!($b)
                );
            }};
        }
        check!(SquaredEuclidean);
        check!(KullbackLeibler);
        check!(ItakuraSaito);
    }

    #[test]
    fn at_squared_euclidean_the_two_measures_are_the_ones_that_already_ship() {
        // D4_φ must collapse to (n_A n_B / n_AB)‖Δμ‖² and D0_φ to the squared Euclidean distance,
        // or the generalisation has quietly changed the Euclidean answer.
        use crate::distance::{CentroidEuclidean, VarianceIncrease};

        let mut rng = Lcg(10);
        let left = points(&mut rng, 25, 5, -3.0, 3.0);
        let right = points(&mut rng, 40, 5, -1.0, 6.0);
        let wl: Vec<f64> = (0..25).map(|_| rng.span(0.5, 3.0)).collect();
        let wr: Vec<f64> = (0..40).map(|_| rng.span(0.5, 3.0)).collect();

        let a = build::<SquaredEuclidean>(&left, &wl);
        let b = build::<SquaredEuclidean>(&right, &wr);
        let mut sa = Spherical::<f64>::new(5);
        for (p, &wi) in left.iter().zip(&wl) {
            sa.push(p, wi);
        }
        let mut sb = Spherical::<f64>::new(5);
        for (p, &wi) in right.iter().zip(&wr) {
            sb.push(p, wi);
        }

        let ward_b = BregmanIncrease::<SquaredEuclidean>::new().between(&a, &b);
        let ward_e = VarianceIncrease.between(&sa, &sb);
        assert!(
            (ward_b - ward_e).abs() <= 1e-9 * ward_e,
            "{ward_b} vs {ward_e}"
        );

        let d0_b = BregmanCentroid::<SquaredEuclidean>::new().between(&a, &b);
        let d0_e = CentroidEuclidean.between(&sa, &sb);
        assert!((d0_b - d0_e).abs() <= 1e-9 * d0_e, "{d0_b} vs {d0_e}");
    }

    #[test]
    fn a_cf_tree_over_a_bregman_feature_conserves_the_information_it_summarises() {
        // The point of implementing `ClusterFeature` rather than a parallel type: the shipped tree
        // takes this feature unmodified. What has to hold is the monoid law one level up -- merging
        // every leaf back together must reproduce the single feature built from all the points.
        use crate::tree::CFTree;

        let mut rng = Lcg(11);
        let pts = points(&mut rng, 400, 6, 0.5, 9.0);

        let mut tree: CFTree<
            f64,
            BregmanCf<f64, ItakuraSaito>,
            BregmanCentroid<ItakuraSaito>,
            BregmanIncrease<ItakuraSaito>,
        > = CFTree::new(
            6,
            16,
            16,
            0.05,
            64,
            BregmanCentroid::new(),
            BregmanIncrease::new(),
        );
        for p in &pts {
            tree.insert(p);
        }
        let leaves = tree.leaf_features();
        assert!(leaves.len() > 1, "the fixture never split");

        let mut pooled = BregmanCf::<f64, ItakuraSaito>::new(6);
        for leaf in leaves {
            pooled.merge(leaf);
        }
        let w = vec![1.0; pts.len()];
        let direct = build::<ItakuraSaito>(&pts, &w);

        assert!((pooled.weight() - direct.weight()).abs() < 1e-9);
        assert!(
            (pooled.ssd() - direct.ssd()).abs() <= 1e-9 * direct.ssd(),
            "pooled {} vs direct {}",
            pooled.ssd(),
            direct.ssd()
        );
        for (a, b) in pooled.mean().iter().zip(direct.mean()) {
            assert!((a - b).abs() <= 1e-9 * b.abs(), "{a} vs {b}");
        }
    }

    #[test]
    fn decay_scales_the_mass_and_leaves_the_geometry_alone() {
        let mut rng = Lcg(7);
        let pts = points(&mut rng, 40, 3, 0.5, 4.0);
        let w = vec![1.0; 40];
        let mut cf = build::<ItakuraSaito>(&pts, &w);
        let (w0, s0, mu0) = (cf.weight(), cf.ssd(), cf.mean().to_vec());
        cf.decay(0.25);
        assert!((cf.weight() - 0.25 * w0).abs() < 1e-12);
        assert!((cf.ssd() - 0.25 * s0).abs() < 1e-12 * s0);
        assert_eq!(cf.mean(), &mu0[..]);
    }

    #[test]
    fn an_empty_feature_absorbs_its_first_point_without_leaving_the_domain() {
        // A zero mean is outside dom φ for KL, IS and logistic. Without the empty-case branch the
        // first push evaluates d_φ(0, x) and the whole feature is NaN from then on.
        let mut cf = BregmanCf::<f64, ItakuraSaito>::new(2);
        cf.push(&[3.0, 5.0], 2.0);
        assert_eq!(cf.ssd(), 0.0);
        assert_eq!(cf.mean(), &[3.0, 5.0]);
        cf.push(&[4.0, 6.0], 1.0);
        assert!(cf.ssd().is_finite() && cf.ssd() > 0.0);
    }
}
