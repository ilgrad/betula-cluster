//! AR / Toeplitz-structured Gaussian Mixture EM for clustering **ordered, wide-sense-stationary
//! signals** — fixed-length time-series windows, trajectories, sensor / audio / vibration waveforms.
//!
//! For a stationary signal the covariance is (approximately) Toeplitz, `Σ_{ts} = c(|t − s|)`, so it
//! is determined by an autocovariance sequence rather than a dense `d × d` matrix. Each component's
//! covariance is modelled as an **AR(w)** process: the biased pooled autocovariance `r(0..w)` is
//! mapped by **Levinson-Durbin** to AR coefficients `a` and innovation variance `σ²`, and the
//! precision is the banded whitening filter `Γ = AᵀA / σ²` (unit diagonal, `−a_j` on the j-th
//! sub-diagonal). `Γ` is **positive-definite by construction** (`σ² > 0`) and has `O(w)` parameters
//! instead of `O(d²)` — well-posed exactly in the `N_k ≪ d` regime where full covariance is singular
//! and a diagonal model is blind to neighbour correlation.
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
/// Upper bound on the AR order searched by BIC per component.
const TOEPLITZ_W_MAX: usize = 6;

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

/// Levinson-Durbin recursion: autocovariance `r[0..=w]` → AR coefficients `[a_1 .. a_w]` and
/// innovation variance. Reflection coefficients are clamped to keep the whitening filter stable.
fn levinson<R: Real>(r: &[R], w: usize) -> (Vec<R>, R) {
    let tiny = R::from_f64(1e-12).unwrap();
    let lim = R::from_f64(0.999).unwrap();
    let mut a = vec![R::zero(); w];
    let mut e = r[0].max(tiny);
    for m in 1..=w {
        let mut acc = r[m];
        for i in 0..m - 1 {
            acc = acc - a[i] * r[m - 1 - i];
        }
        let mut k = acc / e;
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
        e = (e * (R::one() - k * k)).max(tiny);
    }
    (a, e)
}

/// Whitening-residual energy `Σ_{t≥w} (δ_t − Σ_j a_j δ_{t−j})²` — the conditional AR quadratic form.
fn ar_energy<R: Real>(delta: &[R], a: &[R]) -> R {
    let w = a.len();
    let mut e = R::zero();
    for t in w..delta.len() {
        let mut resid = delta[t];
        for (j, &aj) in a.iter().enumerate() {
            resid = resid - aj * delta[t - 1 - j];
        }
        e = e + resid * resid;
    }
    e
}

/// Conditional AR log-density of `delta` under `(a, σ²)` (drops the first `w` boundary positions).
fn ar_loglik<R: Real>(delta: &[R], a: &[R], sigma2: R) -> R {
    let half = R::from_f64(0.5).unwrap();
    let two_pi = R::from_f64(std::f64::consts::TAU).unwrap();
    let n_eff = R::from_usize(delta.len() - a.len()).unwrap();
    -half * (n_eff * (two_pi * sigma2).ln() + ar_energy(delta, a) / sigma2)
}

/// Pooled biased weighted autocovariance `r[0..=w]` of a component with mean `mu_c`. `wt[i]` is the
/// leaf's responsibility-weighted mass `n_i · r_{ic}`; the within-leaf per-dimension variance is
/// folded into the zero lag.
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
    let denom = (nsum * R::from_usize(dim).unwrap()).max(R::from_f64(1e-12).unwrap());
    for rt in r.iter_mut() {
        *rt = *rt / denom;
    }
    r
}

/// Fit the AR order `w ∈ [1, w_max]` by BIC for a component with weights `wt` and mean `mu_c`.
fn fit_component<R, C>(features: &[C], wt: &[R], mu_c: R, dim: usize, w_max: usize) -> (Vec<R>, R)
where
    R: Real,
    C: ClusterFeature<R>,
{
    let w_hi = w_max.min(dim.saturating_sub(1)).max(1);
    let r = component_autocov(features, wt, mu_c, dim, w_hi);
    let nsum: R = wt
        .iter()
        .copied()
        .filter(|&x| x > R::zero())
        .fold(R::zero(), |a, x| a + x);
    let n_eff = (nsum * R::from_usize(dim).unwrap()).max(R::one());
    let two = R::from_f64(2.0).unwrap();
    let mut best = (vec![R::zero(); 1], r[0].max(R::from_f64(1e-12).unwrap()));
    let mut best_bic = R::infinity();
    for w in 1..=w_hi {
        let (a, e) = levinson(&r[..=w], w);
        let mut ll = R::zero();
        for (f, &wi) in features.iter().zip(wt) {
            if wi <= R::zero() {
                continue;
            }
            let delta: Vec<R> = (0..dim).map(|t| f.mean()[t] - mu_c).collect();
            ll = ll + wi * ar_loglik(&delta, &a, e);
        }
        let bic = -two * ll + R::from_usize(w).unwrap() * n_eff.ln();
        if bic < best_bic {
            best_bic = bic;
            best = (a, e);
        }
    }
    best
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

/// One EM run from a random-responsibility init.
fn gmm_toeplitz_once<R, C>(
    features: &[C],
    k: usize,
    w_max: usize,
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
    let mut ar: Vec<Vec<R>> = vec![vec![R::zero(); 1]; k];
    let mut innov = vec![R::one(); k];
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
            let (a, e) = fit_component(features, &wt, mc, dim, w_max);
            means[c] = mc;
            ar[c] = a;
            innov[c] = e;
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
                logr[c] = weights[c].ln() + ar_loglik(&delta, &ar[c], innov[c]);
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

/// Fit a `k`-component AR/Toeplitz GMM, keeping the best of [`TOEPLITZ_N_INIT`] random restarts by
/// data log-likelihood (deterministic for a given `seed`).
pub fn gmm_toeplitz<R, C>(features: &[C], k: usize, max_iter: usize, seed: u64) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    let mut best: Option<GmmToeplitz<R>> = None;
    for r in 0..TOEPLITZ_N_INIT {
        let cand = gmm_toeplitz_once(features, k, TOEPLITZ_W_MAX, max_iter, seed.wrapping_add(r));
        if best.as_ref().is_none_or(|b| cand.loglik > b.loglik) {
            best = Some(cand);
        }
    }
    best.unwrap()
}

/// AR/Toeplitz GMM with automatic component count by BIC over `k ∈ [k_min, k_max]`.
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
    let ntot: R = features
        .iter()
        .map(|f| f.weight())
        .fold(R::zero(), |a, x| a + x);
    let two = R::from_f64(2.0).unwrap();
    let k_hi = k_max.min(features.len()).max(1);
    let mut best: Option<GmmToeplitz<R>> = None;
    let mut best_bic = R::infinity();
    for k in k_min.max(1)..=k_hi {
        let g = gmm_toeplitz(features, k, max_iter, seed);
        // parameters: per component a scalar mean, an AR order (≤ w_max) + innovation, plus mixing.
        let p = k * (2 + TOEPLITZ_W_MAX) + (k - 1);
        let bic = -two * g.loglik + R::from_usize(p).unwrap() * ntot.ln();
        if bic < best_bic {
            best_bic = bic;
            best = Some(g);
        }
    }
    best.unwrap()
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
        let (coeff, _e) = levinson(&r, 1);
        assert!((coeff[0] - a).abs() < 1e-6, "recovered {}", coeff[0]);
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
}
