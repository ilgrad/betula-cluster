//! Distances between clustering features (and to raw points), in numerically stable BETULA forms.
//!
//! Each measure exposes `point(cf, x)` (feature vs raw point — the absorption criterion) and
//! `between(a, b)` (feature vs feature — tree routing). Values are squared (no sqrt in the hot
//! path). All forms are derived from `(n, μ, S)` and were verified algebraically against the
//! classic BIRCH forms in `../../math_improove/02-distance-equivalence`.

use crate::feature::{ClusterFeature, Full};
use crate::kernels;
use crate::types::Real;

/// A distance / absorption criterion over clustering features of model `C`.
///
/// `Send + Sync` lets a distance be shared across rayon worker threads.
pub trait CFDistance<R: Real, C>: Send + Sync {
    /// Squared distance from feature `cf` to a raw point `x`.
    fn point(&self, cf: &C, x: &[R]) -> R;
    /// Squared distance between two features.
    fn between(&self, a: &C, b: &C) -> R;
}

/// D0 — squared Euclidean distance between centroids.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct CentroidEuclidean;
impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for CentroidEuclidean {
    #[inline]
    fn point(&self, cf: &C, x: &[R]) -> R {
        kernels::sq_euclidean(cf.mean(), x)
    }
    #[inline]
    fn between(&self, a: &C, b: &C) -> R {
        kernels::sq_euclidean(a.mean(), b.mean())
    }
}

/// D1 — Manhattan (L1) distance between centroids (note: L1, not squared).
pub struct CentroidManhattan;
impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for CentroidManhattan {
    fn point(&self, cf: &C, x: &[R]) -> R {
        kernels::manhattan(cf.mean(), x)
    }
    fn between(&self, a: &C, b: &C) -> R {
        kernels::manhattan(a.mean(), b.mean())
    }
}

/// D2 — average inter-cluster squared distance `‖Δμ‖² + Var(a) + Var(b)`.
pub struct AverageIntercluster;
impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for AverageIntercluster {
    fn point(&self, cf: &C, x: &[R]) -> R {
        let n = cf.weight();
        if n <= R::zero() {
            return R::zero();
        }
        kernels::sq_euclidean(cf.mean(), x) + cf.ssd() / n
    }
    fn between(&self, a: &C, b: &C) -> R {
        let (na, nb) = (a.weight(), b.weight());
        if na <= R::zero() || nb <= R::zero() {
            return R::zero();
        }
        kernels::sq_euclidean(a.mean(), b.mean()) + a.ssd() / na + b.ssd() / nb
    }
}

/// D4 / Ward — variance increase from absorbing a point / merging two features.
/// `S` terms cancel (König–Huygens): purely a centroid measure, hence perfectly stable.
pub struct VarianceIncrease;
impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for VarianceIncrease {
    fn point(&self, cf: &C, x: &[R]) -> R {
        let n = cf.weight();
        if n <= R::zero() {
            return R::zero();
        }
        kernels::sq_euclidean(cf.mean(), x) * n / (n + R::one())
    }
    fn between(&self, a: &C, b: &C) -> R {
        let (na, nb) = (a.weight(), b.weight());
        if na <= R::zero() || nb <= R::zero() {
            return R::zero();
        }
        kernels::sq_euclidean(a.mean(), b.mean()) * na * nb / (na + nb)
    }
}

/// BIRCH "R" — average squared radius of the cluster that results from absorbing/merging.
pub struct Radius;
impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for Radius {
    fn point(&self, cf: &C, x: &[R]) -> R {
        let n = cf.weight();
        if n <= R::zero() {
            return R::zero();
        }
        let np1 = n + R::one();
        (n * kernels::sq_euclidean(cf.mean(), x) + np1 * cf.ssd()) / (np1 * np1)
    }
    fn between(&self, a: &C, b: &C) -> R {
        let (na, nb) = (a.weight(), b.weight());
        let nab = na + nb;
        if nab <= R::zero() {
            return R::zero();
        }
        (na * nb * kernels::sq_euclidean(a.mean(), b.mean()) + nab * (a.ssd() + b.ssd()))
            / (nab * nab)
    }
}

/// Squared Mahalanobis distance using the feature's own (full) covariance — mass-invariant,
/// scale-aware. Falls back to squared Euclidean when the covariance is not positive-definite
/// (e.g. a feature with fewer points than dimensions).
pub struct Mahalanobis;
impl<R: Real> CFDistance<R, Full<R>> for Mahalanobis {
    fn point(&self, cf: &Full<R>, x: &[R]) -> R {
        cf.mahalanobis_sq(x)
            .unwrap_or_else(|| kernels::sq_euclidean(cf.mean(), x))
    }
    fn between(&self, a: &Full<R>, b: &Full<R>) -> R {
        a.mahalanobis_sq(b.mean())
            .unwrap_or_else(|| kernels::sq_euclidean(a.mean(), b.mean()))
    }
}

/// Mahalanobis-χ² absorption gate with a Normal-Inverse-Gamma variance prior — mass- and
/// scale-invariant, which fixes the BIRCH size-imbalance bug (scikit-learn #22854: a huge cluster
/// swallows a far point because its average radius barely moves). Per dimension the effective
/// variance is `(S_j + κ·s₀) / (n + κ)` — the posterior mean under an inverse-gamma prior
/// `(κ, κ·s₀)` — so a fresh single-point entry (`S_j = 0`) falls back to the prior scale `s₀`
/// instead of a singular covariance, and the gate never diverges during tree growth (this is the
/// guard that lets χ² absorption be used in Phase-1, unlike the raw full-covariance `Mahalanobis`).
///
/// Use it as the tree's absorption criterion with `threshold = stats::chi2_quantile(dim, p)`: a
/// point is absorbed into a leaf iff its squared Mahalanobis distance is below the `p`-quantile of
/// χ²_dim. Diagonal/isotropic by construction (uses per-dimension variance), so it is well-defined
/// for any feature model including single-point and low-mass entries.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct MahalanobisChi2<R> {
    prior_scale: R,
    prior_count: R,
}

impl<R: Real> MahalanobisChi2<R> {
    /// `prior_scale` = `s₀`, the fallback per-dimension variance (≈ the data's feature scale);
    /// `prior_count` = `κ`, the prior strength in pseudo-points (e.g. `dim + 2`).
    pub fn new(prior_scale: R, prior_count: R) -> Self {
        Self {
            prior_scale,
            prior_count,
        }
    }

    fn maha_sq<C: ClusterFeature<R>>(&self, cf: &C, x: &[R]) -> R {
        let n = cf.weight();
        let denom = n + self.prior_count;
        let prior = self.prior_count * self.prior_scale;
        let mu = cf.mean();
        let mut s = R::zero();
        for (j, (&xj, &mj)) in x.iter().zip(mu).enumerate() {
            let scatter = cf.variance(j) * n; // S_j = Var_j · n
            let var_eff = (scatter + prior) / denom;
            let diff = xj - mj;
            s = s + diff * diff / var_eff;
        }
        s
    }
}

impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for MahalanobisChi2<R> {
    fn point(&self, cf: &C, x: &[R]) -> R {
        self.maha_sq(cf, x)
    }
    fn between(&self, a: &C, b: &C) -> R {
        self.maha_sq(a, b.mean())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{Diagonal, Full, Spherical};
    use crate::stats::chi2_quantile;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn build<C: ClusterFeature<f64>>(dim: usize, pts: &[&[f64]]) -> C {
        let mut c = C::new(dim);
        for p in pts {
            c.push(p, 1.0);
        }
        c
    }

    #[test]
    fn centroid_euclidean() {
        let a: Spherical<f64> = build(2, &[&[0., 0.], &[2., 0.]]); // mean (1,0)
        let b: Spherical<f64> = build(2, &[&[0., 4.], &[0., 6.]]); // mean (0,5)
        let d = CentroidEuclidean;
        assert!(close(d.between(&a, &b), 26.0)); // ‖(1,-5)‖² = 1 + 25
        assert!(close(d.point(&a, &[4., 0.]), 9.0)); // (1-4)²
    }

    #[test]
    fn variance_increase_two_points() {
        let a: Spherical<f64> = build(1, &[&[0.]]);
        let b: Spherical<f64> = build(1, &[&[3.]]);
        assert!(close(VarianceIncrease.between(&a, &b), 4.5)); // (1·1/2)·9
    }

    #[test]
    fn radius_matches_formula() {
        let c: Spherical<f64> = build(1, &[&[1.], &[3.], &[5.]]); // mean 3, ssd 8
                                                                  // point to 0: (n·cd + (n+1)·S)/(n+1)² = (3·9 + 4·8)/16
        assert!(close(Radius.point(&c, &[0.]), 59.0 / 16.0));
    }

    #[test]
    fn average_intercluster() {
        let a: Diagonal<f64> = build(1, &[&[0.], &[2.]]); // mean 1, var 1
        let b: Diagonal<f64> = build(1, &[&[10.], &[12.]]); // mean 11, var 1
        assert!(close(AverageIntercluster.between(&a, &b), 102.0)); // 100 + 1 + 1
    }

    #[test]
    fn mahalanobis_full() {
        let c: Full<f64> = build(2, &[&[-1., -2.], &[1., 2.], &[-1., 2.], &[1., -2.]]); // cov diag(1,4)
        assert!(close(Mahalanobis.point(&c, &[2., 2.]), 5.0)); // 4/1 + 4/4
    }

    /// A tight cluster of `4·k` points at the origin with per-dim variance `σ²` (mean 0).
    fn tight_cluster(k: usize, sigma: f64) -> Diagonal<f64> {
        let mut c = Diagonal::<f64>::new(2);
        for _ in 0..k {
            for p in [
                [sigma, sigma],
                [-sigma, -sigma],
                [sigma, -sigma],
                [-sigma, sigma],
            ] {
                c.push(&p, 1.0);
            }
        }
        c
    }

    #[test]
    fn mahalanobis_chi2_gate_is_mass_invariant() {
        // sklearn #22854: the absorption decision must depend on shape, not mass. Two clusters of
        // identical spread but 833× different mass must give (nearly) the same χ² gate value.
        let sigma = 0.01;
        let big = tight_cluster(2500, sigma); // 10000 points
        let small = tight_cluster(3, sigma); //    12 points
        let gate = MahalanobisChi2::new(sigma * sigma, 4.0); // s₀ = σ², κ = d + 2
        let thr = chi2_quantile(2, 0.99); // ≈ 9.21

        let far = [1.0, 1.0];
        let (mb, ms) = (gate.point(&big, &far), gate.point(&small, &far));
        assert!(
            (mb - ms).abs() / mb < 0.05,
            "gate not mass-invariant: big={mb}, small={ms}"
        );
        // Far point rejected by both regardless of mass; near point (~1σ) absorbed.
        assert!(mb > thr && ms > thr, "far point should be rejected");
        let near = [sigma, sigma];
        assert!(
            gate.point(&big, &near) < thr,
            "near point should be absorbed"
        );
    }

    #[test]
    fn centroid_manhattan() {
        let a: Spherical<f64> = build(2, &[&[0., 0.], &[2., 0.]]); // mean (1,0)
        let b: Spherical<f64> = build(2, &[&[0., 4.], &[0., 6.]]); // mean (0,5)
        assert!(close(CentroidManhattan.between(&a, &b), 6.0)); // |1| + |−5|
        assert!(close(CentroidManhattan.point(&a, &[4., 2.]), 5.0)); // |1−4| + |0−2|
    }

    #[test]
    fn average_intercluster_point_and_empty_guard() {
        let a: Diagonal<f64> = build(1, &[&[0.], &[2.]]); // mean 1, ssd 2, n 2
        assert!(close(AverageIntercluster.point(&a, &[4.]), 10.0)); // (1−4)² + 2/2
        let empty: Diagonal<f64> = Diagonal::new(1);
        assert!(close(AverageIntercluster.point(&empty, &[1.]), 0.0));
        assert!(close(AverageIntercluster.between(&empty, &a), 0.0));
    }

    #[test]
    fn variance_increase_point_and_empty_guard() {
        let c: Spherical<f64> = build(1, &[&[0.], &[2.]]); // mean 1, n 2
        assert!(close(VarianceIncrease.point(&c, &[4.]), 6.0)); // 9·2/3
        let empty: Spherical<f64> = Spherical::new(1);
        assert!(close(VarianceIncrease.point(&empty, &[1.]), 0.0));
        assert!(close(VarianceIncrease.between(&empty, &c), 0.0));
    }

    #[test]
    fn radius_between_and_empty_guard() {
        let a: Spherical<f64> = build(1, &[&[0.], &[2.]]); // mean 1, ssd 2, n 2
        let b: Spherical<f64> = build(1, &[&[10.]]); // mean 10, ssd 0, n 1
        assert!(close(Radius.between(&a, &b), 168.0 / 9.0)); // (2·1·81 + 3·2)/9
        let empty: Spherical<f64> = Spherical::new(1);
        assert!(close(Radius.between(&empty, &empty), 0.0));
    }

    #[test]
    fn mahalanobis_between_and_euclidean_fallback() {
        let c: Full<f64> = build(2, &[&[-1., -2.], &[1., 2.], &[-1., 2.], &[1., -2.]]); // cov diag(1,4)
        let other: Full<f64> = build(2, &[&[2., 2.], &[2., 2.]]); // mean (2,2)
        assert!(close(Mahalanobis.between(&c, &other), 5.0)); // 4/1 + 4/4
        let one: Full<f64> = build(2, &[&[0., 0.]]); // non-PD ⇒ Euclidean fallback
        assert!(close(Mahalanobis.point(&one, &[3., 4.]), 25.0));
    }

    #[test]
    fn mahalanobis_chi2_between_uses_other_mean() {
        let a: Diagonal<f64> = build(2, &[&[0.0, 0.0]]); // single point, scatter 0
        let b: Diagonal<f64> = build(2, &[&[1.0, 1.0], &[1.0, 1.0]]); // mean (1,1)
        let gate = MahalanobisChi2::new(1.0, 2.0);
        assert!(close(gate.between(&a, &b), 3.0)); // (1+1)/((0+2)/(1+2))
    }

    #[test]
    fn mahalanobis_chi2_single_point_falls_back_to_prior() {
        // A one-point entry has zero scatter; the gate must use the prior scale, never diverge.
        let one: Diagonal<f64> = build(2, &[&[0.0, 0.0]]);
        let gate = MahalanobisChi2::new(1.0, 2.0);
        let m = gate.point(&one, &[1.0, 1.0]);
        assert!(
            m.is_finite() && m > 0.0,
            "expected finite fallback, got {m}"
        );
        // var_eff = (0 + 2·1)/(1 + 2) = 2/3 per dim ⇒ maha² = (1+1)/(2/3) = 3.
        assert!(close(m, 3.0), "maha² = {m}");
    }

    #[test]
    fn radius_point_empty_guard() {
        let empty: Spherical<f64> = Spherical::new(1);
        assert!(close(Radius.point(&empty, &[1.]), 0.0));
    }
}
