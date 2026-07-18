//! AR / Toeplitz-structured Gaussian Mixture EM for clustering **ordered, wide-sense-stationary
//! signals** — fixed-length time-series windows, trajectories, sensor / audio / vibration waveforms.
//!
//! For a stationary signal the covariance is (approximately) Toeplitz, `Σ_{ts} = c(|t − s|)`, so it
//! is determined by an autocovariance sequence rather than a dense `d × d` matrix. Each component's
//! covariance is modelled as an **AR(w)** process: the biased pooled autocovariance `r(0..w)` is
//! mapped by **Levinson-Durbin** to the order-`w` predictor and innovation variance `σ²`. The precision
//! is the **exact Gohberg-Semencul** form `Γ = (1/σ²)(BBᵀ − ZZᵀ)`, evaluated via the prediction-error
//! decomposition so the `w` boundary positions are modelled *exactly* (the `−ZZᵀ` corner term), not
//! dropped as in a conditional likelihood. `Γ` is **positive-definite by construction** — Levinson's
//! reflection-coefficient clamp *is* the GS box constraint `|αᵢ/α₀| ≤ Kᵢ` — and has `O(w)` parameters
//! instead of `O(d²)`, well-posed exactly in the `N_k ≪ d` regime where full covariance is singular
//! and a diagonal model is blind to neighbour correlation. Estimator + PD constraint follow
//! arXiv:2311.14995.
//!
//! The autocovariance is pooled from the leaf **mean deviations** `δ_i = μ_i − μ_c` (the between-leaf
//! structure) plus the within-leaf per-dimension variance folded into the zero lag `r(0)` (so the
//! component variance accounts for CF compression spread). Off-diagonal within-leaf covariance is
//! treated as negligible — exact for spherical / diagonal leaves, the intended features here.
//! Cost is `O(d·w)` per (leaf, component). Zero-mean stationary data carries no centroid signal, so
//! the EM uses **random-responsibility restarts** (k-means warm-start would be uninformative).
//!
//! This head is for *ordered* coordinates only. On generic embeddings (permutation-invariant
//! semantics) the Toeplitz prior is wrong; use `gmm` / `gmm-full` there. See
//! `docs/adr/001-gmm-toeplitz.md`.

use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::types::Real;

/// Random EM restarts kept by data log-likelihood (EM is non-convex; covariance-only clustering is
/// init-sensitive, so this is higher than the centroid heads' — still deterministic for a `seed`).
const TOEPLITZ_N_INIT: u64 = 8;
/// Upper bound on the AR order searched by BIC per component. AR(w) approaches any wide-sense-stationary
/// precision as `w` grows (Wold), and BIC keeps the smallest sufficient order, so a generous cap adds
/// headroom for higher-order / MA-like signals at no cost on the easy ones (it self-limits at `d − 1`).
const TOEPLITZ_W_MAX: usize = 10;

/// Which per-component covariance model the EM fits: the banded **AR(w)** precision (few parameters,
/// well-posed at `N_k ≪ d`) or a **general positive-definite Toeplitz** covariance (the full
/// autocovariance sequence, for signals whose structure a low-order AR cannot capture).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CovKind {
    /// Banded AR(`w`) precision via Levinson-Durbin + the exact Gohberg-Semencul decomposition.
    Ar,
    /// Dense general Toeplitz covariance from the biased (periodogram-consistent) autocovariance.
    ToeplitzFull,
}

/// A fitted per-component covariance: either the AR(`w`) predictor bank or a dense Toeplitz Cholesky.
enum CompCov<R: Real> {
    /// `phi[m]` / `v[m]` are the order-`m` predictors and error variances (`m = 0..=w`).
    Ar { phi: Vec<Vec<R>>, v: Vec<R> },
    /// Lower-Cholesky factor of a positive-definite Toeplitz covariance and its `log|Σ|`.
    Toeplitz { chol: Vec<Vec<R>>, logdet: R },
}

impl<R: Real> CompCov<R> {
    /// Log-density of a length-`d` mean-deviation vector under this component covariance.
    fn loglik(&self, delta: &[R]) -> R {
        match self {
            CompCov::Ar { phi, v } => ar_loglik_exact(delta, phi, v, phi.len() - 1),
            CompCov::Toeplitz { chol, logdet } => {
                let half = R::from_f64(0.5).unwrap();
                let log_two_pi = R::from_f64(std::f64::consts::TAU).unwrap().ln();
                let d = R::from_usize(delta.len()).unwrap();
                let quad = crate::linalg::mahalanobis_sq_from_chol(chol, delta);
                -half * (d * log_two_pi + *logdet + quad)
            }
        }
    }
    /// AR coefficients for reporting (empty for the general Toeplitz model).
    fn ar_coeffs(&self) -> Vec<R> {
        match self {
            CompCov::Ar { phi, .. } => phi.last().cloned().unwrap_or_default(),
            CompCov::Toeplitz { .. } => Vec::new(),
        }
    }
    /// Innovation variance for reporting (`Σ_{00}` for the Toeplitz model).
    fn innov(&self) -> R {
        match self {
            CompCov::Ar { v, .. } => *v.last().unwrap(),
            CompCov::Toeplitz { chol, .. } => chol[0][0] * chol[0][0],
        }
    }
}

/// Result of an AR/Toeplitz GMM-EM run.
pub struct GmmToeplitz<R: Real> {
    /// Hard label (argmax responsibility) per feature.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[feature][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_k`.
    pub weights: Vec<R>,
    /// Per-component constant (stationary) mean level `μ_k` — a scalar broadcast across positions,
    /// as befits a wide-sense-stationary signal (one parameter, not `d`).
    pub means: Vec<R>,
    /// Per-component AR coefficients `[a_1 .. a_{w_k}]`.
    pub ar: Vec<Vec<R>>,
    /// Per-component innovation variance `σ²_k`.
    pub innov: Vec<R>,
    /// Weighted data log-likelihood at convergence.
    pub loglik: R,
}

/// Levinson-Durbin producing **all** intermediate order-`m` predictors and prediction-error
/// variances (`m = 0..=w`): `phi[m]` are the order-`m` AR coefficients (length `m`) and `v[m]` the
/// step-`m` prediction-error variance. The final `(phi[w], v[w])` are the AR(w) coefficients and
/// innovation variance; the lower orders give the **exact** boundary likelihood below. Reflection
/// coefficients are clamped to `|k| ≤ 0.999`, keeping every predictor stable (⇒ each `v[m] > 0`),
/// which is exactly the positive-definiteness constraint the Gohberg-Semencul parameterization needs.
fn levinson_full<R: Real>(r: &[R], w: usize) -> (Vec<Vec<R>>, Vec<R>) {
    let tiny = R::from_f64(1e-12).unwrap();
    let lim = R::from_f64(0.999).unwrap();
    let mut phi: Vec<Vec<R>> = Vec::with_capacity(w + 1);
    phi.push(Vec::new()); // order 0: no predictor
    let mut v = vec![R::zero(); w + 1];
    v[0] = r[0].max(tiny);
    let mut a = vec![R::zero(); w];
    for m in 1..=w {
        let mut acc = r[m];
        for i in 0..m - 1 {
            acc = acc - a[i] * r[m - 1 - i];
        }
        let mut k = acc / v[m - 1];
        if k > lim {
            k = lim;
        } else if k < -lim {
            k = -lim;
        }
        let old = a.clone();
        a[m - 1] = k;
        for i in 0..m - 1 {
            a[i] = old[i] - k * old[m - 2 - i];
        }
        v[m] = (v[m - 1] * (R::one() - k * k)).max(tiny);
        phi.push(a[..m].to_vec());
    }
    (phi, v)
}

/// **Exact** finite-sample AR (Gohberg-Semencul) log-density of `delta` via the prediction-error
/// decomposition. Position `t` is predicted by the order-`min(t, w)` predictor with its own error
/// variance `v[min(t,w)]`, so the first `w` boundary positions are modelled *exactly* — this is the
/// GS `Γ = (1/σ²)(BBᵀ − ZZᵀ)` precision (the `−ZZᵀ` term is the corner/edge correction) made
/// computational, rather than the conditional likelihood that simply drops those positions.
fn ar_loglik_exact<R: Real>(delta: &[R], phi: &[Vec<R>], v: &[R], w: usize) -> R {
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

/// Pooled unbiased (covariance-method) weighted autocovariance `r[0..=w]` of a component with mean
/// `mu_c`. `wt[i]` is the leaf's responsibility-weighted mass `n_i · r_{ic}`; the within-leaf
/// per-dimension variance is folded into the zero lag. Each lag is normalized by its own count
/// `d − τ` (see the note in the body).
fn component_autocov<R, C>(features: &[C], wt: &[R], mu_c: R, dim: usize, w: usize) -> Vec<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    let mut r = vec![R::zero(); w + 1];
    let mut nsum = R::zero();
    for (f, &wi) in features.iter().zip(wt) {
        if wi <= R::zero() {
            continue;
        }
        let mu_i = f.mean();
        let delta: Vec<R> = (0..dim).map(|t| mu_i[t] - mu_c).collect();
        for (tau, rt) in r.iter_mut().enumerate() {
            let mut s = R::zero();
            for t in 0..dim - tau {
                s = s + delta[t] * delta[t + tau];
            }
            *rt = *rt + wi * s;
        }
        let mut trv = R::zero();
        for t in 0..dim {
            trv = trv + f.variance(t);
        }
        r[0] = r[0] + wi * trv;
        nsum = nsum + wi;
    }
    // Unbiased normalization (divide lag τ by `d − τ`, not `d`) — the covariance-method estimator:
    // less biased than the autocorrelation method, sharper on few samples, measured to cluster better
    // (mean and worst-case ARI over seeds). It is not guaranteed PSD, but `levinson_full`'s
    // reflection-coefficient clamp projects it back to a stable (positive-definite) AR — the
    // CF-compatible form of the paper's covariance-method + PD-projection (arXiv:2311.14995).
    let tiny = R::from_f64(1e-12).unwrap();
    for (tau, rt) in r.iter_mut().enumerate() {
        let denom = (nsum * R::from_usize(dim - tau).unwrap()).max(tiny);
        *rt = *rt / denom;
    }
    r
}

/// Pooled **biased** (÷`d`) weighted autocovariance `r_b[0..d]` with mean `mu_c`. Unlike the
/// covariance-method estimator used for AR, the biased (periodogram-consistent) sequence yields a
/// Toeplitz matrix that is **positive-semidefinite by construction** — each leaf contributes the
/// autocorrelation of its zero-padded deviation (a nonnegative spectrum), and a nonnegative-weighted
/// sum stays PSD. The `÷d` shrinks high-lag terms (few products), a free regularization exactly where
/// `N_k ≪ d`. The within-leaf per-dimension variance folded into the zero lag makes it strictly PD.
fn component_autocov_biased<R, C>(features: &[C], wt: &[R], mu_c: R, dim: usize) -> Vec<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    let mut r = vec![R::zero(); dim];
    let mut nsum = R::zero();
    for (f, &wi) in features.iter().zip(wt) {
        if wi <= R::zero() {
            continue;
        }
        let mu_i = f.mean();
        let delta: Vec<R> = (0..dim).map(|t| mu_i[t] - mu_c).collect();
        for (tau, rt) in r.iter_mut().enumerate() {
            let mut s = R::zero();
            for t in 0..dim - tau {
                s = s + delta[t] * delta[t + tau];
            }
            *rt = *rt + wi * s;
        }
        let mut trv = R::zero();
        for t in 0..dim {
            trv = trv + f.variance(t);
        }
        r[0] = r[0] + wi * trv;
        nsum = nsum + wi;
    }
    let tiny = R::from_f64(1e-12).unwrap();
    let denom = (nsum * R::from_usize(dim).unwrap()).max(tiny);
    for rt in r.iter_mut() {
        *rt = *rt / denom;
    }
    r
}

/// Fit a **general positive-definite Toeplitz** covariance for a component: build the dense Toeplitz
/// matrix `Σ_{ij} = r_b(|i − j|)` from the biased autocovariance and take its ridge-regularized
/// Cholesky. Captures autocovariance structure a low-order AR cannot (broadband / high-order signals),
/// at `O(d²)` parameters and an `O(d³)` factorization per component — the general (non-AR) rung of the
/// Toeplitz ladder (`docs/adr/001-gmm-toeplitz.md`).
fn fit_toeplitz_full<R, C>(features: &[C], wt: &[R], mu_c: R, dim: usize) -> CompCov<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    let rb = component_autocov_biased(features, wt, mu_c, dim);
    let mut cov = vec![vec![R::zero(); dim]; dim];
    for (i, row) in cov.iter_mut().enumerate() {
        for (j, cij) in row.iter_mut().enumerate() {
            *cij = rb[i.abs_diff(j)];
        }
    }
    let scale = rb[0].max(R::from_f64(1e-12).unwrap());
    let (chol, logdet) =
        crate::clustering::gmm::chol_regularized(&cov, scale, R::from_f64(1e-6).unwrap());
    CompCov::Toeplitz { chol, logdet }
}

/// Fit the AR order `w ∈ [1, w_max]` by BIC for a component with weights `wt` and mean `mu_c`,
/// returning the intermediate predictors `phi[0..=w]` and error variances `v[0..=w]` for the selected
/// order (consumed by the exact-likelihood E-step).
fn fit_component<R, C>(
    features: &[C],
    wt: &[R],
    mu_c: R,
    dim: usize,
    w_max: usize,
) -> (Vec<Vec<R>>, Vec<R>)
where
    R: Real,
    C: ClusterFeature<R>,
{
    let w_hi = w_max.min(dim.saturating_sub(1)).max(1);
    let r = component_autocov(features, wt, mu_c, dim, w_hi);
    let (phi_all, v_all) = levinson_full(&r, w_hi);
    let nsum: R = wt
        .iter()
        .copied()
        .filter(|&x| x > R::zero())
        .fold(R::zero(), |a, x| a + x);
    let n_eff = (nsum * R::from_usize(dim).unwrap()).max(R::one());
    let two = R::from_f64(2.0).unwrap();
    let mut best_w = 1;
    let mut best_bic = R::infinity();
    for w in 1..=w_hi {
        let mut ll = R::zero();
        for (f, &wi) in features.iter().zip(wt) {
            if wi <= R::zero() {
                continue;
            }
            let delta: Vec<R> = (0..dim).map(|t| f.mean()[t] - mu_c).collect();
            ll = ll + wi * ar_loglik_exact(&delta, &phi_all[..=w], &v_all[..=w], w);
        }
        let bic = -two * ll + R::from_usize(w).unwrap() * n_eff.ln();
        if bic < best_bic {
            best_bic = bic;
            best_w = w;
        }
    }
    (phi_all[..=best_w].to_vec(), v_all[..=best_w].to_vec())
}

fn argmax<R: Real>(v: &[R]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// One EM run from a random-responsibility init, fitting the requested per-component covariance `kind`.
fn gmm_toeplitz_once<R, C>(
    features: &[C],
    k: usize,
    w_max: usize,
    kind: CovKind,
    max_iter: usize,
    seed: u64,
) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let m = features.len();
    let dim = features[0].dim();
    let n: Vec<R> = features.iter().map(|f| f.weight()).collect();
    let mu: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();

    let mut rng = SplitMix64::new(seed);
    let mut resp = vec![vec![R::zero(); k]; m];
    for row in resp.iter_mut() {
        let mut s = R::zero();
        for x in row.iter_mut() {
            let u = R::from_f64(rng.next_f64() + 1e-3).unwrap();
            *x = u;
            s = s + u;
        }
        for x in row.iter_mut() {
            *x = *x / s;
        }
    }

    let mut weights = vec![R::one() / R::from_usize(k).unwrap(); k];
    let mut means = vec![R::zero(); k];
    let mut covs: Vec<CompCov<R>> = (0..k)
        .map(|_| CompCov::Ar {
            phi: vec![Vec::new()],
            v: vec![R::one()],
        })
        .collect();
    let mut loglik = R::neg_infinity();
    let tol = R::from_f64(1e-6).unwrap();

    for it in 0..max_iter {
        // ── M-step ──
        let mut nk = vec![R::zero(); k];
        for c in 0..k {
            let wt: Vec<R> = (0..m).map(|i| n[i] * resp[i][c]).collect();
            let nkc: R = wt.iter().copied().sum();
            nk[c] = nkc;
            let mut mc = R::zero();
            if nkc > R::zero() {
                for (i, &wi) in wt.iter().enumerate() {
                    let si = mu[i].iter().copied().fold(R::zero(), |acc, x| acc + x);
                    mc = mc + wi * si;
                }
                mc = mc / (nkc * R::from_usize(dim).unwrap());
            }
            means[c] = mc;
            covs[c] = match kind {
                CovKind::Ar => {
                    let (phi, v) = fit_component(features, &wt, mc, dim, w_max);
                    CompCov::Ar { phi, v }
                }
                CovKind::ToeplitzFull => fit_toeplitz_full(features, &wt, mc, dim),
            };
        }
        let ntot: R = nk.iter().copied().sum();
        for c in 0..k {
            weights[c] = nk[c] / ntot;
        }

        // ── E-step ──
        let mut new_ll = R::zero();
        for i in 0..m {
            let mut logr = vec![R::zero(); k];
            for c in 0..k {
                let delta: Vec<R> = (0..dim).map(|t| mu[i][t] - means[c]).collect();
                logr[c] = weights[c].ln() + covs[c].loglik(&delta);
            }
            let mx = logr.iter().copied().fold(R::neg_infinity(), R::max);
            let mut s = R::zero();
            for &lr in &logr {
                s = s + (lr - mx).exp();
            }
            let lse = mx + s.ln();
            new_ll = new_ll + n[i] * lse;
            for c in 0..k {
                resp[i][c] = (logr[c] - lse).exp();
            }
        }
        if it > 0 && (new_ll - loglik).abs() <= tol * loglik.abs().max(R::one()) {
            loglik = new_ll;
            break;
        }
        loglik = new_ll;
    }

    let labels = resp.iter().map(|r| argmax(r)).collect();
    let ar: Vec<Vec<R>> = covs.iter().map(|c| c.ar_coeffs()).collect();
    let innov: Vec<R> = covs.iter().map(|c| c.innov()).collect();
    GmmToeplitz {
        labels,
        resp,
        weights,
        means,
        ar,
        innov,
        loglik,
    }
}

/// Fit a `k`-component Toeplitz GMM of the given covariance `kind`, keeping the best of
/// [`TOEPLITZ_N_INIT`] random restarts by data log-likelihood (deterministic for a given `seed`).
fn gmm_toeplitz_kind<R, C>(
    features: &[C],
    k: usize,
    kind: CovKind,
    max_iter: usize,
    seed: u64,
) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    crate::clustering::gmm::best_of_restarts(
        TOEPLITZ_N_INIT,
        seed,
        |g: &GmmToeplitz<R>| g.loglik,
        |s| gmm_toeplitz_once(features, k, TOEPLITZ_W_MAX, kind, max_iter, s),
    )
}

/// Toeplitz GMM of the given covariance `kind` with automatic component count by BIC over
/// `k ∈ [k_min, k_max]`.
fn gmm_toeplitz_auto_kind<R, C>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    kind: CovKind,
    max_iter: usize,
    seed: u64,
) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    let ntot: R = features
        .iter()
        .map(|f| f.weight())
        .fold(R::zero(), |a, x| a + x);
    let two = R::from_f64(2.0).unwrap();
    let dim = features[0].dim();
    let k_hi = k_max.min(features.len()).max(1);
    let mut best: Option<GmmToeplitz<R>> = None;
    let mut best_bic = R::infinity();
    for k in k_min.max(1)..=k_hi {
        let g = gmm_toeplitz_kind(features, k, kind, max_iter, seed);
        // per component: a scalar mean + the covariance model (AR order ≤ w_max + innovation, or the
        // `d` general-Toeplitz autocovariances), plus the mixing weights.
        let cov_p = match kind {
            CovKind::Ar => 1 + TOEPLITZ_W_MAX,
            CovKind::ToeplitzFull => dim,
        };
        let p = k * (1 + cov_p) + (k - 1);
        let bic = -two * g.loglik + R::from_usize(p).unwrap() * ntot.ln();
        if bic < best_bic {
            best_bic = bic;
            best = Some(g);
        }
    }
    best.unwrap()
}

/// Fit a `k`-component **AR(w)** Toeplitz GMM — banded Gohberg-Semencul precision, `O(w)` parameters
/// per component, well-posed at `N_k ≪ d`. Best of [`TOEPLITZ_N_INIT`] restarts (deterministic per `seed`).
pub fn gmm_toeplitz<R, C>(features: &[C], k: usize, max_iter: usize, seed: u64) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    gmm_toeplitz_kind(features, k, CovKind::Ar, max_iter, seed)
}

/// AR(w) Toeplitz GMM with automatic component count by BIC over `k ∈ [k_min, k_max]`.
pub fn gmm_toeplitz_auto<R, C>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    gmm_toeplitz_auto_kind(features, k_min, k_max, CovKind::Ar, max_iter, seed)
}

/// Fit a `k`-component **general Toeplitz** GMM — a dense positive-definite Toeplitz covariance from
/// the biased autocovariance, capturing structure a low-order AR cannot. `O(d²)` parameters, `O(d³)`
/// per component; for signals where AR(w) is genuinely too restrictive (`docs/adr/001-gmm-toeplitz.md`).
pub fn gmm_toeplitz_full<R, C>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    gmm_toeplitz_kind(features, k, CovKind::ToeplitzFull, max_iter, seed)
}

/// General Toeplitz GMM with automatic component count by BIC over `k ∈ [k_min, k_max]`.
pub fn gmm_toeplitz_full_auto<R, C>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    gmm_toeplitz_auto_kind(
        features,
        k_min,
        k_max,
        CovKind::ToeplitzFull,
        max_iter,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::ari;
    use crate::feature::{ClusterFeature, Spherical};

    /// One length-`d` window from a zero-mean AR(len(a)) process (burn-in discarded).
    fn ar_window(rng: &mut SplitMix64, d: usize, a: &[f64]) -> Vec<f64> {
        let w = a.len();
        let burn = 256;
        let mut buf = vec![0.0; d + burn];
        for t in w..d + burn {
            let mut x = rng.gauss();
            for (j, &aj) in a.iter().enumerate() {
                x += aj * buf[t - 1 - j];
            }
            buf[t] = x;
        }
        let mut win: Vec<f64> = buf[burn..].to_vec();
        // standardize so the marginal variance carries no clustering signal
        let mean = win.iter().sum::<f64>() / d as f64;
        let var = win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
        let sd = var.sqrt().max(1e-9);
        for v in &mut win {
            *v = (*v - mean) / sd;
        }
        win
    }

    /// Build one single-point spherical leaf per window (mirrors the Python prototype).
    fn ar_mixture(
        d: usize,
        per: usize,
        specs: &[&[f64]],
        seed: u64,
    ) -> (Vec<Spherical<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let mut feats = Vec::new();
        let mut truth = Vec::new();
        for (c, a) in specs.iter().enumerate() {
            for _ in 0..per {
                let win = ar_window(&mut rng, d, a);
                let mut f = Spherical::new(d);
                f.push(&win, 1.0);
                feats.push(f);
                truth.push(c);
            }
        }
        (feats, truth)
    }

    #[test]
    fn levinson_recovers_ar1() {
        // Autocovariance of AR(1) with a=0.8: r(τ) = a^|τ| / (1 − a²). Levinson must return a≈0.8.
        let a: f64 = 0.8;
        let r: Vec<f64> = (0i32..=4).map(|t| a.powi(t) / (1.0 - a * a)).collect();
        let (phi, _v) = levinson_full(&r, 1);
        assert!((phi[1][0] - a).abs() < 1e-6, "recovered {}", phi[1][0]);
    }

    #[test]
    fn toeplitz_separates_ar_mixture() {
        // Three components differing ONLY in autocovariance (unit marginal variance). At d ≫ N_k the
        // AR/Toeplitz head separates them; a diagonal GMM (blind to neighbour correlation) cannot.
        let specs: &[&[f64]] = &[&[0.8], &[1.1, -0.4], &[]];
        let (feats, truth) = ar_mixture(128, 30, specs, 1);
        let toe = gmm_toeplitz(&feats, 3, 200, 1);
        let a_toe = ari(&toe.labels, &truth);
        let diag = crate::clustering::gmm::gmm_diagonal(&feats, 3, 200, 1);
        let a_diag = ari(&diag.labels, &truth);
        assert!(a_toe > 0.8, "toeplitz ARI = {a_toe} (diagonal = {a_diag})");
        assert!(
            a_toe > a_diag + 0.3,
            "toeplitz {a_toe} should clearly beat diagonal {a_diag}"
        );
    }

    #[test]
    fn toeplitz_auto_k_recovers_count() {
        let specs: &[&[f64]] = &[&[0.85], &[1.2, -0.5], &[]];
        let (feats, truth) = ar_mixture(96, 25, specs, 7);
        let g = gmm_toeplitz_auto(&feats, 1, 6, 200, 7);
        assert_eq!(g.means.len(), 3, "selected k = {}", g.means.len());
        assert!(ari(&g.labels, &truth) > 0.8);
    }

    #[test]
    fn toeplitz_full_clusters_ar_mixture() {
        // The general (non-AR) Toeplitz covariance is a superset model; it also recovers the
        // autocovariance-only mixture and clearly beats a diagonal GMM (blind to correlation).
        let specs: &[&[f64]] = &[&[0.8], &[1.1, -0.4], &[]];
        let (feats, truth) = ar_mixture(64, 40, specs, 1);
        let full = gmm_toeplitz_full(&feats, 3, 200, 1);
        let a_full = ari(&full.labels, &truth);
        let diag = crate::clustering::gmm::gmm_diagonal(&feats, 3, 200, 1);
        let a_diag = ari(&diag.labels, &truth);
        assert!(
            a_full > 0.6,
            "toeplitz-full ARI = {a_full} (diagonal = {a_diag})"
        );
        assert!(
            a_full > a_diag + 0.3,
            "toeplitz-full {a_full} should clearly beat diagonal {a_diag}"
        );
    }

    #[test]
    fn toeplitz_full_auto_k_produces_pd_covariance() {
        // A PD Toeplitz covariance ⇒ every leaf gets a finite log-density (no singular / NaN E-step);
        // also exercises the general-Toeplitz auto-`k` (BIC) path.
        let specs: &[&[f64]] = &[&[0.85], &[1.2, -0.5], &[]];
        let (feats, _truth) = ar_mixture(48, 20, specs, 3);
        let g = gmm_toeplitz_full_auto(&feats, 1, 4, 200, 3);
        assert!(g.loglik.is_finite(), "loglik = {}", g.loglik);
        assert!(g.resp.iter().flatten().all(|r| r.is_finite()));
        assert!(!g.means.is_empty());
    }
}
