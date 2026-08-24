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
    /// Probabilistic-PCA component: `Σ_c = W_c W_cᵀ + σ_c² I`, held as the `q` loading rows, the
    /// Cholesky of `M_c = σ_c² I_q + W_cᵀ W_c` and `1/σ_c²`. Scoring goes through Woodbury, so no
    /// `d×d` matrix exists here either — the reason this head can hold `k` components at `d = 784`.
    LowRank {
        means: Vec<Vec<f64>>,
        loads: Vec<Vec<Vec<f64>>>,
        m_chol: Vec<Vec<Vec<f64>>>,
        inv_noise: Vec<f64>,
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

    /// Probabilistic-PCA mixture (`method="mppca"`), given each component's `q` loading rows
    /// (`loads[c][r]` is a length-`d` column of `W_c`) and its isotropic noise `σ_c²`.
    ///
    /// `M_c = σ_c² I_q + W_cᵀ W_c` is built and factorized once here, so scoring costs `O(q·d)` per
    /// component and `log|Σ_c| = (d−q) ln σ_c² + log|M_c|` never touches a `d×d` determinant.
    pub(crate) fn low_rank<R: Real>(
        weights: &[R],
        means: &[Vec<R>],
        loads: &[Vec<Vec<R>>],
        noise: &[R],
    ) -> Self {
        let dim = means.first().map_or(0, |m| m.len()) as f64;
        let log_two_pi = std::f64::consts::TAU.ln();
        let mut logw = Vec::with_capacity(weights.len());
        let mut kept = Vec::with_capacity(weights.len());
        let mut m_chol = Vec::with_capacity(weights.len());
        let mut inv_noise = Vec::with_capacity(weights.len());
        for ((&w, rows), &s2) in weights.iter().zip(loads).zip(noise) {
            let s2 = as_f64(s2).max(f64::MIN_POSITIVE);
            let rows: Vec<Vec<f64>> = widen_rows(rows);
            let q = rows.len();
            let mut m = vec![vec![0.0; q]; q];
            for i in 0..q {
                for j in 0..=i {
                    let dot: f64 = rows[i].iter().zip(&rows[j]).map(|(&a, &b)| a * b).sum();
                    m[i][j] = dot;
                    m[j][i] = dot;
                }
                m[i][i] += s2;
            }
            // `M ⪰ σ² I ≻ 0`, so this factors unless a loading row has overflowed. Dropping the
            // loadings then leaves a well-defined isotropic component rather than a NaN density.
            let chol = crate::linalg::cholesky_lower(&m);
            let (rows, chol, logdet) = match chol {
                Some(l) => {
                    let ld = crate::linalg::logdet_from_chol(&l);
                    let ld = (dim - q as f64) * s2.ln() + ld;
                    (rows, l, ld)
                }
                None => (Vec::new(), Vec::new(), dim * s2.ln()),
            };
            logw.push(ln_weight(w) - 0.5 * (dim * log_two_pi + logdet));
            kept.push(rows);
            m_chol.push(chol);
            inv_noise.push(1.0 / s2);
        }
        Self {
            logw,
            kernel: Kernel::LowRank {
                means: widen_rows(means),
                loads: kept,
                m_chol,
                inv_noise,
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
            Kernel::LowRank {
                means,
                loads,
                m_chol,
                inv_noise,
            } => {
                let mut delta = vec![0.0; x.len()];
                let mut proj = Vec::new();
                for (((&lw, mu), rows), (l, &iv)) in self
                    .logw
                    .iter()
                    .zip(means)
                    .zip(loads)
                    .zip(m_chol.iter().zip(inv_noise))
                {
                    let mut iso = 0.0;
                    for ((dv, &xd), &md) in delta.iter_mut().zip(x).zip(mu) {
                        *dv = as_f64(xd) - md;
                        iso += *dv * *dv;
                    }
                    proj.clear();
                    proj.extend(
                        rows.iter()
                            .map(|r| r.iter().zip(&delta).map(|(&f, &d)| f * d).sum::<f64>()),
                    );
                    let corr = if proj.is_empty() {
                        0.0
                    } else {
                        mahalanobis_sq_from_chol(l, &proj)
                    };
                    out.push(lw - 0.5 * (iso - corr) * iv);
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

    /// Woodbury against the dense inverse it exists to avoid: the `LowRank` arm must agree with a
    /// `Full` component built by materializing `Σ = W Wᵀ + σ² I` and factorizing it, on both the
    /// quadratic form and the normalizer.
    #[test]
    fn low_rank_matches_the_dense_covariance_it_never_forms() {
        let dim = 4;
        let loads = vec![
            vec![vec![1.0_f64, -2.0, 0.5, 3.0], vec![0.0, 1.0, 2.0, -1.0]],
            vec![vec![2.0_f64, 0.0, -1.0, 1.0]],
        ];
        let noise = [0.75_f64, 1.5];
        let means = vec![vec![0.0_f64, 1.0, -1.0, 2.0], vec![1.0, 1.0, 1.0, 1.0]];
        let weights = [0.35_f64, 0.65];

        let mut chol = Vec::new();
        let mut logdet = Vec::new();
        for (rows, &s2) in loads.iter().zip(&noise) {
            let mut cov = vec![vec![0.0_f64; dim]; dim];
            for r in rows {
                for i in 0..dim {
                    for j in 0..dim {
                        cov[i][j] += r[i] * r[j];
                    }
                }
            }
            for (i, row) in cov.iter_mut().enumerate() {
                row[i] += s2;
            }
            let l = crate::linalg::cholesky_lower(&cov).expect("Sigma is positive definite");
            logdet.push(crate::linalg::logdet_from_chol(&l));
            chol.push(l);
        }

        let dense = Mixture::full(&weights, &means, &chol, &logdet);
        let lowrank = Mixture::low_rank(&weights, &means, &loads, &noise);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for x in [
            [0.0_f64, 0.0, 0.0, 0.0],
            [1.3, -0.4, 2.2, 0.1],
            [-5.0, 4.0, -3.0, 6.0],
        ] {
            dense.log_joint(&x, &mut a);
            lowrank.log_joint(&x, &mut b);
            for (c, (u, v)) in a.iter().zip(&b).enumerate() {
                assert!((u - v).abs() < 1e-10, "c={c}: {u} vs {v}");
            }
        }
    }

    /// `q = 0` is the degenerate rung the head falls back to when a component holds no direction
    /// worth keeping: it must be a spherical Gaussian exactly, not approximately.
    #[test]
    fn low_rank_without_loadings_is_a_spherical_gaussian() {
        let means = vec![vec![0.0_f64, 1.0], vec![2.0, -1.0]];
        let weights = [0.4_f64, 0.6];
        let noise = [2.0_f64, 0.5];
        let vars: Vec<Vec<f64>> = noise.iter().map(|&s2| vec![s2, s2]).collect();
        let sphere = Mixture::diagonal(&weights, &means, &vars);
        let lowrank = Mixture::low_rank(&weights, &means, &[Vec::new(), Vec::new()], &noise);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let x = [0.7_f64, 0.3];
        sphere.log_joint(&x, &mut a);
        lowrank.log_joint(&x, &mut b);
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

    /// The `Toeplitz` arm of [`StationaryCov`] had no direct test: every existing case builds an
    /// `Ar` covariance, so the general-Toeplitz log-density, its innovation variance and its empty
    /// coefficient list were all reachable only through a head that never distinguished them.
    #[test]
    fn the_toeplitz_arm_scores_a_gaussian_and_reports_no_ar_coefficients() {
        // L = [[2, 0], [1, 3]] ⇒ Σ = L Lᵀ = [[4, 2], [2, 10]], |Σ| = 36, ln|Σ| = ln 36.
        let chol = vec![vec![2.0, 0.0], vec![1.0, 3.0]];
        let logdet = 36.0_f64.ln();
        let cov: StationaryCov<f64> = StationaryCov::Toeplitz {
            chol: chol.clone(),
            logdet,
        };

        // Σ⁻¹ = (1/36)·[[10, −2], [−2, 4]]; for δ = (1, 1): δᵀΣ⁻¹δ = (10 − 2 − 2 + 4)/36 = 10/36.
        let delta = [1.0, 1.0];
        let quad = 10.0 / 36.0;
        let want = -0.5 * (2.0 * std::f64::consts::TAU.ln() + logdet + quad);
        let got = cov.loglik(&delta);
        assert!((got - want).abs() < 1e-12, "loglik = {got}, want {want}");

        // The innovation variance of the general model is Σ₀₀ = L₀₀², not L₀₀ and not 2·L₀₀.
        assert!((cov.innov() - 4.0).abs() < 1e-12, "innov = {}", cov.innov());
        assert!(
            cov.ar_coeffs().is_empty(),
            "the Toeplitz arm reported AR coefficients: {:?}",
            cov.ar_coeffs()
        );

        // The zero deviation is the density's mode, so no other δ may score higher.
        let mode = cov.loglik(&[0.0, 0.0]);
        assert!(mode > got, "the mode did not dominate: {mode} vs {got}");
        assert!((mode - -0.5 * (2.0 * std::f64::consts::TAU.ln() + logdet)).abs() < 1e-12);
    }

    #[test]
    fn the_ar_arm_reports_its_highest_order_predictor_and_last_innovation() {
        // `phi[m]` is the order-`m` predictor; reporting must take the highest order, not the first.
        let cov: StationaryCov<f64> = StationaryCov::Ar {
            phi: vec![vec![], vec![0.5], vec![0.65, -0.3]],
            v: vec![2.0, 1.5, 1.365],
        };
        assert_eq!(cov.ar_coeffs(), vec![0.65, -0.3]);
        assert!(
            (cov.innov() - 1.365).abs() < 1e-12,
            "innov = {}",
            cov.innov()
        );
    }
}
