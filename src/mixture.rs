//! What a fitted mixture head says about a **raw point**.
//!
//! A Phase-3 head fits its parameters to the CF-tree's *leaf features*, whose within-leaf scatter
//! enters the E-step as the expected-log correction `−½ tr(Σ_c⁻¹ Σ_i)`. A single observation has no
//! scatter, so scoring one is the plain component log-density — and the head's own assignment rule
//! is the argmax of `ln π_c + ln p(x | θ_c)`, not a walk down the tree to the nearest microcluster.
//!
//! Keeping that density here, built by the EM that defined it, is what stops the fit and the point
//! rule from drifting apart: a change to a variance floor or a covariance ridge travels into
//! prediction automatically, because the same numbers are stored.
//!
//! Parameters are held in `f64` regardless of the tree's element type. `k · d` values cost nothing
//! next to the tree, an `f32` fit loses no accuracy by being *scored* in `f64`, and one concrete
//! type keeps the estimator's persisted state simple.

use crate::linalg::mahalanobis_sq_from_chol;
use crate::types::Real;

/// A fitted per-component covariance for the ordered / stationary (Toeplitz) heads: either the
/// AR(`w`) predictor bank or a dense Toeplitz Cholesky.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum StationaryCov<R: Real> {
    /// `phi[m]` / `v[m]` are the order-`m` predictors and error variances (`m = 0..=w`).
    Ar { phi: Vec<Vec<R>>, v: Vec<R> },
    /// Lower-Cholesky factor of a positive-definite Toeplitz covariance and its `log|Σ|`.
    Toeplitz { chol: Vec<Vec<R>>, logdet: R },
}

impl<R: Real> StationaryCov<R> {
    /// Log-density of a length-`d` mean-deviation vector under this component covariance.
    pub(crate) fn loglik(&self, delta: &[R]) -> R {
        match self {
            StationaryCov::Ar { phi, v } => ar_loglik_exact(delta, phi, v, phi.len() - 1),
            StationaryCov::Toeplitz { chol, logdet } => {
                let half = R::from_f64(0.5).unwrap();
                let log_two_pi = R::from_f64(std::f64::consts::TAU).unwrap().ln();
                let d = R::from_usize(delta.len()).unwrap();
                let quad = mahalanobis_sq_from_chol(chol, delta);
                -half * (d * log_two_pi + *logdet + quad)
            }
        }
    }

    /// AR coefficients for reporting (empty for the general Toeplitz model).
    pub(crate) fn ar_coeffs(&self) -> Vec<R> {
        match self {
            StationaryCov::Ar { phi, .. } => phi.last().cloned().unwrap_or_default(),
            StationaryCov::Toeplitz { .. } => Vec::new(),
        }
    }

    /// Innovation variance for reporting (`Σ_{00}` for the Toeplitz model).
    pub(crate) fn innov(&self) -> R {
        match self {
            StationaryCov::Ar { v, .. } => *v.last().unwrap(),
            StationaryCov::Toeplitz { chol, .. } => chol[0][0] * chol[0][0],
        }
    }

    fn widen(&self) -> StationaryCov<f64> {
        match self {
            StationaryCov::Ar { phi, v } => StationaryCov::Ar {
                phi: phi.iter().map(|r| widen_row(r)).collect(),
                v: widen_row(v),
            },
            StationaryCov::Toeplitz { chol, logdet } => StationaryCov::Toeplitz {
                chol: chol.iter().map(|r| widen_row(r)).collect(),
                logdet: as_f64(*logdet),
            },
        }
    }
}

/// **Exact** finite-sample AR (Gohberg-Semencul) log-density of `delta` via the prediction-error
/// decomposition. Position `t` is predicted by the order-`min(t, w)` predictor with its own error
/// variance `v[min(t,w)]`, so the first `w` boundary positions are modelled *exactly* — this is the
/// GS `Γ = (1/σ²)(BBᵀ − ZZᵀ)` precision (the `−ZZᵀ` term is the corner/edge correction) made
/// computational, rather than the conditional likelihood that simply drops those positions.
pub(crate) fn ar_loglik_exact<R: Real>(delta: &[R], phi: &[Vec<R>], v: &[R], w: usize) -> R {
    let half = R::from_f64(0.5).unwrap();
    let two_pi = R::from_f64(std::f64::consts::TAU).unwrap();
    let mut ll = R::zero();
    for (t, &dt) in delta.iter().enumerate() {
        let m = t.min(w);
        let mut pred = R::zero();
        for (j, &pj) in phi[m].iter().enumerate() {
            pred = pred + pj * delta[t - 1 - j];
        }
        let e = dt - pred;
        ll = ll - half * ((two_pi * v[m]).ln() + e * e / v[m]);
    }
    ll
}

fn as_f64<R: Real>(v: R) -> f64 {
    v.to_f64().unwrap_or(f64::NAN)
}

fn widen_row<R: Real>(v: &[R]) -> Vec<f64> {
    v.iter().map(|&x| as_f64(x)).collect()
}

fn widen_rows<R: Real>(v: &[Vec<R>]) -> Vec<Vec<f64>> {
    v.iter().map(|r| widen_row(r)).collect()
}

/// The component densities of a fitted head, in the form the scoring loop wants: every constant that
/// does not depend on the point is folded into `logw`.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
enum Kernel {
    /// Diagonal Gaussian; `inv_var[c][d] = 1/σ²_cd`.
    Diagonal {
        means: Vec<Vec<f64>>,
        inv_var: Vec<Vec<f64>>,
    },
    /// Full-covariance Gaussian, held as the lower-Cholesky factor of `Σ_c`.
    Full {
        means: Vec<Vec<f64>>,
        chol: Vec<Vec<Vec<f64>>>,
    },
    /// Toeplitz-structured Gaussian over an ordered signal; the mean is one scalar level per
    /// component, broadcast across positions (wide-sense stationarity).
    Stationary {
        means: Vec<f64>,
        covs: Vec<StationaryCov<f64>>,
    },
    /// von Mises-Fisher on the unit sphere; the point is normalized before the dot product.
    Vmf {
        means: Vec<Vec<f64>>,
        kappas: Vec<f64>,
    },
}

/// The point-level density of a fitted mixture head: `ln π_c + ln p(x | θ_c)` per component, and the
/// maximum-posterior assignment that follows from it.
///
/// The representation is deliberately opaque. Callers score points; they do not read parameters
/// (`weights_` / `means_` and friends stay on the head's own result type), so the set of component
/// families here can grow without becoming API.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Mixture {
    /// `ln π_c` plus every point-independent normalizer of component `c`. `-∞` marks a component no
    /// leaf claims — see [`Mixture::restrict_to`].
    logw: Vec<f64>,
    kernel: Kernel,
}

impl Mixture {
    /// Diagonal Gaussian mixture (`method="gmm"`).
    pub(crate) fn diagonal<R: Real>(weights: &[R], means: &[Vec<R>], vars: &[Vec<R>]) -> Self {
        let log_two_pi = std::f64::consts::TAU.ln();
        let logw = vars
            .iter()
            .zip(weights)
            .map(|(v, &w)| {
                let norm: f64 = v.iter().map(|&s2| log_two_pi + as_f64(s2).ln()).sum();
                ln_weight(w) - 0.5 * norm
            })
            .collect();
        let inv_var = vars
            .iter()
            .map(|v| v.iter().map(|&s2| 1.0 / as_f64(s2)).collect())
            .collect();
        Self {
            logw,
            kernel: Kernel::Diagonal {
                means: widen_rows(means),
                inv_var,
            },
        }
    }

    /// Full-covariance Gaussian mixture (`method="gmm-full"`), given the Cholesky factor of each
    /// `Σ_c` and its `log|Σ_c|` — the same factorization the E-step used.
    pub(crate) fn full<R: Real>(
        weights: &[R],
        means: &[Vec<R>],
        chol: &[Vec<Vec<R>>],
        logdet: &[R],
    ) -> Self {
        let dim = means.first().map_or(0, |m| m.len()) as f64;
        let log_two_pi = std::f64::consts::TAU.ln();
        let logw = weights
            .iter()
            .zip(logdet)
            .map(|(&w, &ld)| ln_weight(w) - 0.5 * (dim * log_two_pi + as_f64(ld)))
            .collect();
        Self {
            logw,
            kernel: Kernel::Full {
                means: widen_rows(means),
                chol: chol.iter().map(|m| widen_rows(m)).collect(),
            },
        }
    }

    /// Toeplitz-structured mixture (`method="gmm-toeplitz"` and its `-full` / `-gs` rungs). The
    /// covariance normalizer lives inside [`StationaryCov::loglik`], so only `ln π_c` is folded here.
    pub(crate) fn stationary<R: Real>(
        weights: &[R],
        means: &[R],
        covs: &[StationaryCov<R>],
    ) -> Self {
        Self {
            logw: weights.iter().map(|&w| ln_weight(w)).collect(),
            kernel: Kernel::Stationary {
                means: widen_row(means),
                covs: covs.iter().map(|c| c.widen()).collect(),
            },
        }
    }

    /// von Mises-Fisher mixture (`method="vmf"`), given each component's log-normalizer
    /// `ln C_d(κ_c)` (the Bessel numerics stay with the head that fits `κ`).
    pub(crate) fn vmf<R: Real>(
        weights: &[R],
        means: &[Vec<R>],
        kappas: &[R],
        logc: &[f64],
    ) -> Self {
        Self {
            logw: weights
                .iter()
                .zip(logc)
                .map(|(&w, &lc)| ln_weight(w) + lc)
                .collect(),
            kernel: Kernel::Vmf {
                means: widen_rows(means),
                kappas: widen_row(kappas),
            },
        }
    }

    /// Silence every component that no leaf hard-assigns, so a prediction can only name a label the
    /// fitted partition actually uses. An EM component can end up with responsibility everywhere and
    /// the argmax nowhere; without this it would be reachable from `predict` but absent from
    /// `labels_`, and `cluster_centers_[label]` would be a zero row or out of range.
    pub(crate) fn restrict_to(&mut self, labels: &[usize]) {
        let mut claimed = vec![false; self.logw.len()];
        for &l in labels {
            if let Some(c) = claimed.get_mut(l) {
                *c = true;
            }
        }
        for (w, keep) in self.logw.iter_mut().zip(claimed) {
            if !keep {
                *w = f64::NEG_INFINITY;
            }
        }
    }

    /// Number of components, including any silenced by [`Mixture::restrict_to`].
    pub fn n_components(&self) -> usize {
        self.logw.len()
    }

    /// Unnormalized log posterior `ln π_c + ln p(x | θ_c)` per component, written into `out`.
    pub fn log_joint<R: Real>(&self, x: &[R], out: &mut Vec<f64>) {
        out.clear();
        match &self.kernel {
            Kernel::Diagonal { means, inv_var } => {
                for ((lw, mu), iv) in self.logw.iter().zip(means).zip(inv_var) {
                    let mut quad = 0.0;
                    for ((&xd, &md), &ivd) in x.iter().zip(mu).zip(iv) {
                        let diff = as_f64(xd) - md;
                        quad += diff * diff * ivd;
                    }
                    out.push(lw - 0.5 * quad);
                }
            }
            Kernel::Full { means, chol } => {
                let mut delta = vec![0.0; x.len()];
                for ((lw, mu), l) in self.logw.iter().zip(means).zip(chol) {
                    for ((dv, &xd), &md) in delta.iter_mut().zip(x).zip(mu) {
                        *dv = as_f64(xd) - md;
                    }
                    out.push(lw - 0.5 * mahalanobis_sq_from_chol(l, &delta));
                }
            }
            Kernel::Stationary { means, covs } => {
                let mut delta = vec![0.0; x.len()];
                for ((&lw, &mu), cov) in self.logw.iter().zip(means).zip(covs) {
                    for (dv, &xd) in delta.iter_mut().zip(x) {
                        *dv = as_f64(xd) - mu;
                    }
                    out.push(lw + cov.loglik(&delta));
                }
            }
            Kernel::Vmf { means, kappas } => {
                let norm = x.iter().map(|&v| as_f64(v) * as_f64(v)).sum::<f64>().sqrt();
                let scale = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                for ((&lw, mu), &kap) in self.logw.iter().zip(means).zip(kappas) {
                    let dot: f64 = x.iter().zip(mu).map(|(&xd, &md)| as_f64(xd) * md).sum();
                    out.push(lw + kap * dot * scale);
                }
            }
        }
    }

    /// Maximum-posterior component for `x` — the head's own assignment rule. `scratch` is a
    /// reusable score buffer, so labelling a matrix allocates once rather than once per row.
    pub fn assign_into<R: Real>(&self, x: &[R], scratch: &mut Vec<f64>) -> usize {
        self.log_joint(x, scratch);
        self.best_of(scratch)
    }

    /// [`Mixture::assign_into`] with its own buffer.
    pub fn assign<R: Real>(&self, x: &[R]) -> usize {
        let mut scratch = Vec::with_capacity(self.logw.len());
        self.assign_into(x, &mut scratch)
    }

    /// Highest-scoring live component. A component silenced by [`Mixture::restrict_to`] can never
    /// win — including when every density has underflowed to `-∞`, where the argmax alone would
    /// return index 0 whether or not the partition uses it.
    fn best_of(&self, scores: &[f64]) -> usize {
        let best = argmax(scores);
        if scores[best].is_finite() {
            best
        } else {
            self.logw.iter().position(|w| w.is_finite()).unwrap_or(0)
        }
    }

    /// Posterior responsibilities `p(c | x)`, written into `out`. Components silenced by
    /// [`Mixture::restrict_to`] get exactly zero, so `argmax(out)` is [`Mixture::assign`].
    pub fn responsibilities<R: Real>(&self, x: &[R], out: &mut Vec<f64>) {
        self.log_joint(x, out);
        let mx = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !mx.is_finite() {
            // Every component is either silenced or numerically impossible here; fall back to a
            // point mass rather than emitting NaN from 0/0.
            let best = self.best_of(out);
            out.iter_mut().for_each(|v| *v = 0.0);
            out[best] = 1.0;
            return;
        }
        let mut total = 0.0;
        for v in out.iter_mut() {
            *v = (*v - mx).exp();
            total += *v;
        }
        for v in out.iter_mut() {
            *v /= total;
        }
    }
}

/// `ln π` with a floor, so an emptied component contributes `-∞`-ish rather than `NaN`.
fn ln_weight<R: Real>(w: R) -> f64 {
    as_f64(w).max(f64::MIN_POSITIVE).ln()
}

/// Index of the largest value; first index on ties (and for an all-`-∞` vector).
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ln N(x | μ, σ²I)` written out longhand, as the reference for the folded `logw`.
    fn ln_gauss_diag(x: &[f64], mu: &[f64], var: &[f64]) -> f64 {
        let log_two_pi = std::f64::consts::TAU.ln();
        x.iter()
            .zip(mu)
            .zip(var)
            .map(|((&xd, &md), &s2)| {
                let diff = xd - md;
                -0.5 * (log_two_pi + s2.ln() + diff * diff / s2)
            })
            .sum()
    }

    #[test]
    fn diagonal_log_joint_matches_the_longhand_density() {
        let weights = [0.3_f64, 0.7];
        let means = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let vars = vec![vec![1.0, 4.0], vec![0.25, 2.0]];
        let m = Mixture::diagonal(&weights, &means, &vars);
        let x = [1.0_f64, 0.5];
        let mut got = Vec::new();
        m.log_joint(&x, &mut got);
        for c in 0..2 {
            let want = weights[c].ln() + ln_gauss_diag(&x, &means[c], &vars[c]);
            assert!((got[c] - want).abs() < 1e-12, "c={c}: {} vs {want}", got[c]);
        }
    }

    #[test]
    fn a_tight_component_wins_a_point_a_nearer_centre_would_take() {
        // The point sits closer to component 1's centre, but component 0 is wide enough (and heavy
        // enough) that its posterior is larger — exactly the disagreement a nearest-centre rule has
        // with a mixture, and the reason `predict` cannot be an argmin over centroids here.
        let weights = [0.9_f64, 0.1];
        let means = vec![vec![0.0], vec![2.0]];
        let vars = vec![vec![4.0], vec![0.01]];
        let m = Mixture::diagonal(&weights, &means, &vars);
        let x = [1.4_f64];
        assert!((x[0] - means[1][0]).abs() < (x[0] - means[0][0]).abs());
        assert_eq!(m.assign(&x), 0);
    }

    #[test]
    fn responsibilities_are_a_normalized_softmax_of_the_log_joint() {
        let m = Mixture::diagonal(
            &[0.5_f64, 0.5],
            &[vec![0.0, 0.0], vec![1.0, 1.0]],
            &[vec![1.0, 1.0], vec![1.0, 1.0]],
        );
        let x = [0.25_f64, 0.75];
        let (mut lj, mut r) = (Vec::new(), Vec::new());
        m.log_joint(&x, &mut lj);
        m.responsibilities(&x, &mut r);
        let sum: f64 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        let ratio = (lj[0] - lj[1]).exp();
        assert!((r[0] / r[1] - ratio).abs() < 1e-10);
        assert_eq!(m.assign(&x), argmax(&r));
    }

    #[test]
    fn restrict_to_makes_unclaimed_components_unreachable() {
        let mut m = Mixture::diagonal(
            &[0.5_f64, 0.5],
            &[vec![0.0], vec![10.0]],
            &[vec![1.0], vec![1.0]],
        );
        let x = [9.9_f64];
        assert_eq!(m.assign(&x), 1);
        m.restrict_to(&[0, 0, 0]); // no leaf ever lands in component 1
        assert_eq!(m.assign(&x), 0);
        let mut r = Vec::new();
        m.responsibilities(&x, &mut r);
        assert_eq!(r[1], 0.0);
        assert!((r[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vmf_assignment_follows_direction_not_magnitude() {
        let means = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0]];
        let m = Mixture::vmf(&[0.5_f64, 0.5], &means, &[8.0_f64, 8.0], &[0.0, 0.0]);
        for scale in [0.01_f64, 1.0, 1000.0] {
            assert_eq!(m.assign(&[0.9 * scale, 0.4 * scale]), 0);
            assert_eq!(m.assign(&[0.4 * scale, 0.9 * scale]), 1);
        }
    }

    #[test]
    fn full_covariance_reduces_to_the_diagonal_case_when_sigma_is_diagonal() {
        let vars = vec![vec![2.0_f64, 0.5], vec![1.0, 3.0]];
        let means = vec![vec![0.0_f64, 1.0], vec![2.0, -1.0]];
        let weights = [0.4_f64, 0.6];
        let chol: Vec<Vec<Vec<f64>>> = vars
            .iter()
            .map(|v| vec![vec![v[0].sqrt(), 0.0], vec![0.0, v[1].sqrt()]])
            .collect();
        let logdet: Vec<f64> = vars.iter().map(|v| (v[0] * v[1]).ln()).collect();
        let diag = Mixture::diagonal(&weights, &means, &vars);
        let full = Mixture::full(&weights, &means, &chol, &logdet);
        let x = [0.7_f64, 0.3];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        diag.log_joint(&x, &mut a);
        full.log_joint(&x, &mut b);
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-12, "{u} vs {v}");
        }
    }

    #[test]
    fn stationary_ar0_is_an_isotropic_gaussian_about_a_scalar_level() {
        let covs = vec![StationaryCov::Ar {
            phi: vec![Vec::new()],
            v: vec![2.0_f64],
        }];
        let m = Mixture::stationary(&[1.0_f64], &[0.5_f64], &covs);
        let x = [1.5_f64, -0.5, 0.5];
        let mut got = Vec::new();
        m.log_joint(&x, &mut got);
        let want = ln_gauss_diag(&x, &[0.5, 0.5, 0.5], &[2.0, 2.0, 2.0]);
        assert!((got[0] - want).abs() < 1e-12, "{} vs {want}", got[0]);
    }
}
