//! Distances between clustering features (and to raw points), in numerically stable BETULA forms.
//!
//! Each measure exposes `point(cf, x)` (feature vs raw point — the absorption criterion) and
//! `between(a, b)` (feature vs feature — tree routing). Values are squared (no sqrt in the hot
//! path). All forms are derived from `(n, μ, S)` and were verified algebraically against the
//! classic BIRCH forms in `math_improove/02-distance-equivalence` (local-only working notes).

use crate::feature::{ClusterFeature, Full, SecondMoment};
use crate::kernels;
use crate::linalg;
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

/// D3 — average intra-cluster squared distance of the cluster that *results* from absorbing a
/// point / merging two features (BIRCH's "diameter" criterion), i.e. the mean of `‖xᵢ − xⱼ‖²` over
/// ordered pairs `i ≠ j` of the union.
///
/// From `(n, μ, S)` alone: the double sum telescopes to `Σᵢⱼ‖xᵢ − xⱼ‖² = 2·n·S`, so `D3² = 2S/(n−1)`
/// for one cluster, and the union's `S` follows König–Huygens. That leaves
/// `D3²(A, B) = 2·(S_A + S_B + n_A·n_B/(n_A+n_B)·‖Δμ‖²) / (n_A + n_B − 1)` — the merge term is
/// exactly [`VarianceIncrease`], so D3 is D4 and the two scatters over one fewer than the merged
/// mass. Verified against brute-force enumeration over random point sets to 5.7e-14.
///
/// Returned squared, like every other measure here; BIRCH defines D3 as the root, which is
/// monotone in this and so ranks identically. Zero when the union holds at most one point, where
/// the mean over pairs is undefined rather than large.
pub struct AverageIntracluster;
impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for AverageIntracluster {
    fn point(&self, cf: &C, x: &[R]) -> R {
        let n = cf.weight();
        if n <= R::zero() {
            return R::zero();
        }
        let two = R::one() + R::one();
        two * (cf.ssd() + kernels::sq_euclidean(cf.mean(), x) * n / (n + R::one())) / n
    }
    fn between(&self, a: &C, b: &C) -> R {
        let (na, nb) = (a.weight(), b.weight());
        let nab = na + nb;
        if nab <= R::one() {
            return R::zero();
        }
        let two = R::one() + R::one();
        two * (a.ssd() + b.ssd() + kernels::sq_euclidean(a.mean(), b.mean()) * na * nb / nab)
            / (nab - R::one())
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
        let nab = na + nb;
        if nab <= R::zero() {
            return R::zero();
        }
        kernels::sq_euclidean(a.mean(), b.mean()) * na * nb / nab
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

/// Absorption gate on the leaf's **own low-rank subspace** — the χ² gate measured on `ℓ + 1`
/// effective dimensions instead of `d`.
///
/// Motivation is measured, not assumed: on MNIST-20k the relative contrast of distances to a class
/// model triples (0.446 → 1.423) and argmin accuracy goes 0.805 → 0.968 when the distance is taken
/// off a rank-40 class subspace instead of to the class mean. Concentration in high `d` is a
/// property of the *leaf model*, not of the data, so a leaf that already carries a basis should use
/// it to decide absorption.
///
/// Only [`FdSketch`](crate::feature::FdSketch) carries one — it reports
/// [`SecondMoment::LowRank`], whose rows `f_r` satisfy `Σ = Σ_r f_r f_rᵀ`. For every other feature
/// model this falls back to [`MahalanobisChi2`], so the gate is well defined for all of them and
/// `absorb="subspace"` only *means* anything with `feature="fd"`.
///
/// Under the same Normal-Inverse-Gamma prior the diagonal gate uses, the effective covariance is
/// the posterior mean `Σ_eff = (S + κ·s₀·I)/(n + κ) = a·FᵀF + b·I` with `a = n/(n+κ)` and
/// `b = κ·s₀/(n+κ)`. Its inverse never forms a `d×d` matrix — Woodbury reduces it to one `ℓ×ℓ`
/// solve:
///
/// ```text
/// d_L(y) = (1/b)·[ ‖y‖² − (F y)ᵀ G⁻¹ (F y) ],   G = (b/a)·I_ℓ + F Fᵀ,   y = x − μ
/// ```
///
/// At rank 0 the correction vanishes and `d_L(y) = ‖y‖²(n+κ)/(κ·s₀)`, which is exactly what
/// [`MahalanobisChi2`] returns on a leaf with no scatter — so the two agree by construction where
/// they overlap, and this one adds the off-diagonal structure the diagonal gate cannot see.
///
/// Threshold is read in the same units: `stats::chi2_quantile(dim, p)`.
///
/// Cost is `O(ℓ²d)` per decision, against `O(d)` for the diagonal gate. That is deliberate for now —
/// the question this answers is whether leaf-discovered subspaces are accurate enough to be worth
/// anything at all, and `G` is recomputed per call rather than cached on the leaf.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct SubspaceChi2<R> {
    diagonal: MahalanobisChi2<R>,
}

impl<R: Real> SubspaceChi2<R> {
    /// Same arguments as [`MahalanobisChi2::new`]: `s₀` the fallback per-dimension variance and
    /// `κ` the prior strength in pseudo-points.
    pub fn new(prior_scale: R, prior_count: R) -> Self {
        Self {
            diagonal: MahalanobisChi2::new(prior_scale, prior_count),
        }
    }

    fn subspace_sq<C: ClusterFeature<R>>(&self, cf: &C, x: &[R], rows: &[Vec<R>], iso: R) -> R {
        let n = cf.weight();
        let denom = n + self.diagonal.prior_count;
        // Σ_eff = a·(FᵀF + iso·I) + b·I, so the sketch's isotropic residual joins the prior's own
        // isotropic term and Woodbury is unchanged.
        let b = (self.diagonal.prior_count * self.diagonal.prior_scale + n * iso) / denom;
        if b <= R::zero() {
            return self.diagonal.maha_sq(cf, x);
        }
        let y: Vec<R> = x.iter().zip(cf.mean()).map(|(&xj, &mj)| xj - mj).collect();
        let iso = y.iter().map(|&v| v * v).fold(R::zero(), |s, v| s + v) / b;

        let a = n / denom;
        if rows.is_empty() || a <= R::zero() {
            return iso; // no basis, or no mass behind it — the prior is all there is
        }

        // G = (b/a)·I + F Fᵀ, and F y, both ℓ-sized.
        let ridge = b / a;
        let l = rows.len();
        let mut g = vec![vec![R::zero(); l]; l];
        let mut fy = vec![R::zero(); l];
        for i in 0..l {
            fy[i] = rows[i]
                .iter()
                .zip(&y)
                .map(|(&f, &v)| f * v)
                .fold(R::zero(), |s, v| s + v);
            for j in 0..=i {
                let dot = rows[i]
                    .iter()
                    .zip(&rows[j])
                    .map(|(&p, &q)| p * q)
                    .fold(R::zero(), |s, v| s + v);
                g[i][j] = dot;
                g[j][i] = dot;
            }
            g[i][i] = g[i][i] + ridge;
        }

        // `G` is symmetric positive definite (`F Fᵀ` is PSD and `ridge > 0`), so a failed Cholesky
        // means rounding, not a modelling error — fall back to the isotropic term rather than
        // returning something arbitrary.
        let Some(chol) = linalg::cholesky_lower(&g) else {
            return iso;
        };
        let correction = linalg::mahalanobis_sq_from_chol(&chol, &fy);
        (iso - correction / b).max(R::zero())
    }
}

impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for SubspaceChi2<R> {
    fn point(&self, cf: &C, x: &[R]) -> R {
        match cf.second_moment() {
            SecondMoment::LowRank { rows, iso, .. } => self.subspace_sq(cf, x, &rows, iso),
            SecondMoment::Dense(_) => self.diagonal.maha_sq(cf, x),
        }
    }
    fn between(&self, a: &C, b: &C) -> R {
        self.point(a, b.mean())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{Diagonal, FdSketch, Full, Spherical};
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

    /// Mean squared distance over ordered pairs `i != j` -- the definition D3 closes over,
    /// enumerated directly so the closed form is checked against something that shares no algebra
    /// with it.
    fn brute_mean_sq_pairwise(pts: &[&[f64]]) -> f64 {
        let n = pts.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for a in pts {
            for b in pts {
                total += a
                    .iter()
                    .zip(*b)
                    .map(|(u, v)| (u - v) * (u - v))
                    .sum::<f64>();
            }
        }
        total / (n * (n - 1)) as f64
    }

    #[test]
    fn average_intracluster_matches_brute_force_enumeration() {
        // Unequal masses and non-zero scatter on both sides, so the merge term and the asymmetry
        // both have to be right -- a balanced or zero-scatter fixture would pass on a wrong formula.
        let pa: [&[f64]; 3] = [&[0., 0.], &[2., 0.], &[1., 3.]];
        let pb: [&[f64]; 2] = [&[7., 1.], &[9., 5.]];
        let a: Diagonal<f64> = build(2, &pa);
        let b: Diagonal<f64> = build(2, &pb);
        assert!(a.ssd() > 0.0 && b.ssd() > 0.0 && a.weight() != b.weight());

        let x: &[f64] = &[4., 8.];
        let mut union: Vec<&[f64]> = pa.to_vec();
        union.push(x);
        assert!(close(
            AverageIntracluster.point(&a, x),
            brute_mean_sq_pairwise(&union)
        ));

        let mut merged: Vec<&[f64]> = pa.to_vec();
        merged.extend_from_slice(&pb);
        assert!(close(
            AverageIntracluster.between(&a, &b),
            brute_mean_sq_pairwise(&merged)
        ));
    }

    #[test]
    fn average_intracluster_is_zero_where_the_mean_over_pairs_is_undefined() {
        // A union of at most one point has no ordered pair to average over. Reporting 0 rather than
        // dividing by `n - 1 == 0` is the whole reason the guard is on the merged mass and not on
        // either input's.
        let empty: Diagonal<f64> = Diagonal::new(1);
        let one: Diagonal<f64> = build(1, &[&[5.]]);
        assert!(close(AverageIntracluster.point(&empty, &[1.]), 0.0));
        assert!(close(AverageIntracluster.between(&empty, &one), 0.0));
        assert!(close(AverageIntracluster.between(&empty, &empty), 0.0));
        // Two singletons is the first defined case, and it is exactly the squared distance.
        let two: Diagonal<f64> = build(1, &[&[9.]]);
        assert!(close(AverageIntracluster.between(&one, &two), 16.0));
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
    fn radius_between_pins_the_product_and_the_pooled_scatter() {
        // Deliberately na·nb != na/nb and both scatters non-zero, so the weight product and the
        // ssd sum are each pinned to a value rather than to a coincidence.
        // a: n 2, mean 1, ssd 2. b: n 4, mean 11, ssd 20. (2·4·100 + 6·(2 + 20)) / 6² = 932/36.
        let a: Spherical<f64> = build(1, &[&[0.], &[2.]]);
        let b: Spherical<f64> = build(1, &[&[8.], &[10.], &[12.], &[14.]]);
        assert!(close(Radius.between(&a, &b), 233.0 / 9.0));
    }

    #[test]
    fn mahalanobis_chi2_uses_the_signed_difference() {
        // Away from the origin, so x - mean and x + mean disagree.
        let c: Diagonal<f64> = build(1, &[&[4.], &[6.]]); // n 2, mean 5, scatter 2
        let gate = MahalanobisChi2::new(1.0, 2.0);
        // var_eff = (2 + 2·1)/(2 + 2) = 1 ⇒ maha² = (1 − 5)² / 1 = 16.
        assert!(close(gate.point(&c, &[1.0]), 16.0));
    }

    #[test]
    fn radius_point_empty_guard() {
        let empty: Spherical<f64> = Spherical::new(1);
        assert!(close(Radius.point(&empty, &[1.]), 0.0));
    }

    /// `yᵀ(a·FᵀF + b·I)⁻¹y` the slow way: materialise the `d×d` matrix and invert it. Independent of
    /// the Woodbury path under test, which never forms a `d×d` matrix at all.
    fn dense_maha_sq(cf: &FdSketch<f64>, x: &[f64], s0: f64, kappa: f64) -> f64 {
        let SecondMoment::LowRank { rows, iso, .. } = cf.second_moment() else {
            unreachable!("FdSketch reports LowRank")
        };
        let (n, d) = (cf.weight(), x.len());
        let (a, b) = (n / (n + kappa), kappa * s0 / (n + kappa));
        let mut m = vec![vec![0.0; d]; d];
        for f in &rows {
            for i in 0..d {
                for j in 0..d {
                    m[i][j] += a * f[i] * f[j];
                }
            }
        }
        for (i, mi) in m.iter_mut().enumerate() {
            mi[i] += b + a * iso;
        }
        let y: Vec<f64> = x.iter().zip(cf.mean()).map(|(&xj, &mj)| xj - mj).collect();
        let chol = linalg::cholesky_lower(&m).expect("Σ_eff is positive definite for b > 0");
        linalg::mahalanobis_sq_from_chol(&chol, &y)
    }

    #[test]
    fn subspace_chi2_matches_the_dense_inverse() {
        let (s0, kappa) = (0.5, 4.0);
        let mut fd: FdSketch<f64> = FdSketch::with_ell(3, 3);
        for p in [
            [1.0, 2.0, 0.5],
            [-1.0, -2.0, 0.5],
            [2.0, 1.0, -1.5],
            [0.0, 3.0, 1.0],
        ] {
            fd.push(&p, 1.0);
        }
        let gate = SubspaceChi2::new(s0, kappa);
        for probe in [[0.0, 0.0, 0.0], [3.0, -1.0, 2.0], [-2.5, 0.5, 4.0]] {
            let want = dense_maha_sq(&fd, &probe, s0, kappa);
            assert!(
                (gate.point(&fd, &probe) - want).abs() < 1e-9,
                "probe {probe:?}: Woodbury {} vs dense {want}",
                gate.point(&fd, &probe)
            );
        }
    }

    #[test]
    fn subspace_chi2_falls_back_to_the_diagonal_gate() {
        // A feature with a `Dense` second moment carries no basis, so the two gates must agree
        // exactly — `absorb="subspace"` is only a different gate for `feature="fd"`.
        let c: Diagonal<f64> = build(2, &[&[0., 0.], &[2., 6.]]);
        let (s0, kappa) = (0.25, 3.0);
        assert!(close(
            SubspaceChi2::new(s0, kappa).point(&c, &[1.5, -2.]),
            MahalanobisChi2::new(s0, kappa).point(&c, &[1.5, -2.]),
        ));
    }

    #[test]
    fn subspace_chi2_sees_a_direction_the_diagonal_gate_cannot() {
        // Mass strung along (1,1)/√2, so per-dimension variance is equal in both coordinates and the
        // diagonal gate is blind to the orientation. Two probes at the same Euclidean distance —
        // one along the leaf's own direction, one across it — must therefore tie under the diagonal
        // gate and separate under the subspace one. This is the entire claim being tested.
        let mut fd: FdSketch<f64> = FdSketch::with_ell(2, 2);
        for t in [-3.0, -1.0, 1.0, 3.0] {
            fd.push(&[t, t], 1.0);
        }
        let (along, across) = ([2.0, 2.0], [2.0, -2.0]);
        let diag = MahalanobisChi2::new(0.1, 2.0);
        assert!(close(diag.point(&fd, &along), diag.point(&fd, &across)));

        let sub = SubspaceChi2::new(0.1, 2.0);
        assert!(
            sub.point(&fd, &along) < 0.5 * sub.point(&fd, &across),
            "along {} should be far cheaper than across {}",
            sub.point(&fd, &along),
            sub.point(&fd, &across),
        );
    }
}
