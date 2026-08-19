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
use crate::mixture::{ar_loglik_exact, Mixture, StationaryCov};
use crate::types::Real;

/// Random EM restarts kept by data log-likelihood (EM is non-convex; covariance-only clustering is
/// init-sensitive, so this is higher than the centroid heads' — still deterministic for a `seed`).
const TOEPLITZ_N_INIT: u64 = 8;
/// Upper bound on the AR order searched by BIC per component. AR(w) approaches any wide-sense-stationary
/// precision as `w` grows (Wold), and BIC keeps the smallest sufficient order, so a generous cap adds
/// headroom for higher-order / MA-like signals at no cost on the easy ones (it self-limits at `d − 1`).
const TOEPLITZ_W_MAX: usize = 10;
/// Order cap for the full-order Gohberg-Semencul MLE head (`gmm-toeplitz-gs`); it fits up to this order
/// (self-limited at `d − 1`), a general (non-banded) precision that captures autocovariance structure
/// beyond the banded AR head. `O(m·d·p)` per likelihood eval, so kept moderate.
const GS_ORDER_MAX: usize = 16;
/// Coordinate-ascent sweeps refining the reflection coefficients toward the exact-likelihood optimum —
/// the MLE step on top of the Yule-Walker (Levinson) warm start.
const GS_REFINE_SWEEPS: usize = 1;
/// EM restarts for the GS-MLE head — fewer than the cheaper heads because each fit is `O(m·d·p)` per
/// M-step (the full-order likelihood refinement); still deterministic for a `seed`.
const GS_N_INIT: u64 = 4;

/// Which per-component covariance model the EM fits: the banded **AR(w)** precision (few parameters,
/// well-posed at `N_k ≪ d`), a **general positive-definite Toeplitz** covariance (the full
/// autocovariance sequence), or the full-order **Gohberg-Semencul MLE** precision.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CovKind {
    /// Banded AR(`w`) precision via Levinson-Durbin + the exact Gohberg-Semencul decomposition.
    Ar,
    /// Dense general Toeplitz covariance from the biased (periodogram-consistent) autocovariance.
    ToeplitzFull,
    /// Full-order **Gohberg-Semencul MLE** precision: Yule-Walker warm start refined by coordinate
    /// ascent of the exact log-likelihood over the reflection coefficients (positive-definite by the
    /// `|k| < 1` constraint). The likelihood-optimal general precision; see arXiv:2311.14995.
    GsMle,
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
    /// The fitted density, for scoring raw points (`ar` / `innov` are its reporting projection).
    pub mixture: Mixture,
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
fn fit_toeplitz_full<R, C>(features: &[C], wt: &[R], mu_c: R, dim: usize) -> StationaryCov<R>
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
    StationaryCov::Toeplitz { chol, logdet }
}

/// Levinson step-up: build the order-`m` predictors `phi[0..=p]` and error variances `v[0..=p]` from
/// reflection coefficients `refl[1..=p]` and the zero-lag variance `r0` — the inverse of the reflection
/// extraction. `|refl_m| < 1` keeps every `v[m] > 0`, i.e. the Gohberg-Semencul precision PD.
fn step_up<R: Real>(refl: &[R], r0: R, p: usize) -> (Vec<Vec<R>>, Vec<R>) {
    let tiny = R::from_f64(1e-12).unwrap();
    let mut phi: Vec<Vec<R>> = Vec::with_capacity(p + 1);
    phi.push(Vec::new());
    let mut v = vec![R::zero(); p + 1];
    v[0] = r0.max(tiny);
    let mut a = vec![R::zero(); p];
    for m in 1..=p {
        let km = refl[m];
        let old = a.clone();
        a[m - 1] = km;
        for i in 0..m - 1 {
            a[i] = old[i] - km * old[m - 2 - i];
        }
        v[m] = (v[m - 1] * (R::one() - km * km)).max(tiny);
        phi.push(a[..m].to_vec());
    }
    (phi, v)
}

/// Fit the full-order **Gohberg-Semencul MLE** precision for a component: a Yule-Walker (Levinson) warm
/// start at order `min(d−1, GS_ORDER_MAX)`, then coordinate ascent of the exact weighted log-likelihood
/// over the reflection coefficients (the MLE refinement; `|k| < 1` keeps it positive-definite). Returns
/// the refined predictor bank as an `Ar` covariance — the E-step consumes the same exact GS precision.
fn fit_component_gs<R, C>(features: &[C], wt: &[R], mu_c: R, dim: usize) -> StationaryCov<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    let p = GS_ORDER_MAX.min(dim.saturating_sub(1)).max(1);
    let r = component_autocov(features, wt, mu_c, dim, p);
    let (phi0, _v0) = levinson_full(&r, p);
    // reflection coefficient at order m is the last coefficient of the order-m predictor.
    let mut refl = vec![R::zero(); p + 1];
    for (m, item) in refl.iter_mut().enumerate().take(p + 1).skip(1) {
        *item = *phi0[m].last().unwrap();
    }
    let lim = R::from_f64(0.999).unwrap();

    let deltas: Vec<(R, Vec<R>)> = features
        .iter()
        .zip(wt)
        .filter(|(_, &w)| w > R::zero())
        .map(|(f, &w)| (w, (0..dim).map(|t| f.mean()[t] - mu_c).collect()))
        .collect();
    let r0 = r[0];
    let eval = |refl: &[R]| -> R {
        let (phi, v) = step_up(refl, r0, p);
        deltas
            .iter()
            .map(|(w, d)| *w * ar_loglik_exact(d, &phi, &v, p))
            .fold(R::zero(), |a, b| a + b)
    };

    // Coordinate ascent: a local pattern search per reflection coefficient (best of a few step sizes),
    // warm-started from Yule-Walker so a couple of sweeps suffice.
    let steps = [R::from_f64(0.15).unwrap(), R::from_f64(0.05).unwrap()];
    let mut cur = eval(&refl);
    for _ in 0..GS_REFINE_SWEEPS {
        for m in 1..=p {
            let (mut best_k, mut best_l) = (refl[m], cur);
            for &s in &steps {
                for &dir in &[R::one(), -R::one()] {
                    let cand = (refl[m] + dir * s).max(-lim).min(lim);
                    let saved = refl[m];
                    refl[m] = cand;
                    let l = eval(&refl);
                    refl[m] = saved;
                    if l > best_l {
                        best_l = l;
                        best_k = cand;
                    }
                }
            }
            refl[m] = best_k;
            cur = best_l;
        }
    }

    let (phi, v) = step_up(&refl, r0, p);
    StationaryCov::Ar { phi, v }
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
    let mut covs: Vec<StationaryCov<R>> = (0..k)
        .map(|_| StationaryCov::Ar {
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
                    StationaryCov::Ar { phi, v }
                }
                CovKind::ToeplitzFull => fit_toeplitz_full(features, &wt, mc, dim),
                CovKind::GsMle => fit_component_gs(features, &wt, mc, dim),
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
    let mixture = Mixture::stationary(&weights, &means, &covs);
    GmmToeplitz {
        labels,
        resp,
        weights,
        means,
        ar,
        innov,
        loglik,
        mixture,
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
    let n_init = match kind {
        CovKind::GsMle => GS_N_INIT,
        _ => TOEPLITZ_N_INIT,
    };
    crate::clustering::gmm::best_of_restarts(
        n_init,
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
            CovKind::GsMle => 1 + GS_ORDER_MAX,
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

/// Fit a `k`-component **Gohberg-Semencul MLE** Toeplitz GMM — a full-order (`≤ GS_ORDER_MAX`) precision,
/// Yule-Walker-warm-started and refined by exact-likelihood coordinate ascent (PD by `|k| < 1`). The
/// likelihood-optimal general precision; see [ADR 001](../docs/adr/001-gmm-toeplitz.md).
pub fn gmm_toeplitz_gs<R, C>(features: &[C], k: usize, max_iter: usize, seed: u64) -> GmmToeplitz<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    gmm_toeplitz_kind(features, k, CovKind::GsMle, max_iter, seed)
}

/// Gohberg-Semencul MLE Toeplitz GMM with automatic component count by BIC over `k ∈ [k_min, k_max]`.
pub fn gmm_toeplitz_gs_auto<R, C>(
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
    gmm_toeplitz_auto_kind(features, k_min, k_max, CovKind::GsMle, max_iter, seed)
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

    /// One length-`d` window of a single-echo MA process `x_t = e_t + 0.7·e_{t−lag}`, unit variance.
    fn echo_window(rng: &mut SplitMix64, d: usize, lag: usize) -> Vec<f64> {
        let mut e = vec![0.0; d + lag];
        for v in e.iter_mut() {
            *v = rng.gauss();
        }
        let mut win: Vec<f64> = (0..d).map(|t| e[t + lag] + 0.7 * e[t]).collect();
        let mean = win.iter().sum::<f64>() / d as f64;
        let var = win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
        let sd = var.sqrt().max(1e-9);
        for v in &mut win {
            *v = (*v - mean) / sd;
        }
        win
    }

    fn echo_mixture(
        d: usize,
        per: usize,
        lags: &[usize],
        seed: u64,
    ) -> (Vec<Spherical<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let mut feats = Vec::new();
        let mut truth = Vec::new();
        for (c, &lag) in lags.iter().enumerate() {
            for _ in 0..per {
                let win = echo_window(&mut rng, d, lag);
                let mut f = Spherical::new(d);
                f.push(&win, 1.0);
                feats.push(f);
                truth.push(c);
            }
        }
        (feats, truth)
    }

    #[test]
    fn gs_mle_clusters_ar_mixture() {
        let specs: &[&[f64]] = &[&[0.8], &[1.1, -0.4], &[]];
        let (feats, truth) = ar_mixture(64, 40, specs, 1);
        let g = gmm_toeplitz_gs(&feats, 3, 200, 1);
        assert!(g.loglik.is_finite());
        assert!(
            ari(&g.labels, &truth) > 0.6,
            "gs ARI {}",
            ari(&g.labels, &truth)
        );
    }

    #[test]
    fn gs_mle_recovers_long_lag_echo() {
        // Echoes at lags {11,13,15}, all beyond the banded AR cap w_max=10 but within GS_ORDER_MAX: the
        // full-order GS precision captures them where the banded AR head is structurally blind.
        let (feats, truth) = echo_mixture(64, 40, &[11, 13, 15], 1);
        let gs = gmm_toeplitz_gs(&feats, 3, 200, 1);
        let ar = gmm_toeplitz(&feats, 3, 200, 1);
        let (a_gs, a_ar) = (ari(&gs.labels, &truth), ari(&ar.labels, &truth));
        assert!(a_gs > 0.5, "gs echo ARI {a_gs} (banded AR {a_ar})");
        assert!(
            a_gs > a_ar + 0.2,
            "gs {a_gs} should beat banded AR {a_ar} on a long-lag echo"
        );
    }

    /// Two leaves whose autocovariance is small enough to work out by hand: leaf A is a single point
    /// (no within-leaf spread) and leaf B is a heavier, flat leaf that contributes only through the
    /// zero lag. Fixes the weighting, the trace fold and the per-lag denominator at once.
    fn autocov_fixture() -> (Vec<Spherical<f64>>, Vec<f64>) {
        let a = Spherical::from_moments(1.0, vec![1.0, 2.0, 3.0, 4.0], 0.0);
        let b = Spherical::from_moments(2.0, vec![2.0, 2.0, 2.0, 2.0], 8.0);
        (vec![a, b], vec![1.0, 2.0])
    }

    #[test]
    fn unbiased_autocov_divides_each_lag_by_its_own_product_count() {
        // δ_A = [−1.5, −0.5, 0.5, 1.5], δ_B = [−0.5; 4], tr Σ_B = 4, weights 1 and 2.
        // raw r = [1·5 + 2·1 + 2·4, 1·1.25 + 2·0.75, 1·(−1.5) + 2·0.5] = [15, 2.75, −0.5],
        // divided by nsum·(d − τ) = [12, 9, 6].
        let (feats, wt) = autocov_fixture();
        let r = component_autocov(&feats, &wt, 2.5, 4, 2);
        let want = [15.0 / 12.0, 2.75 / 9.0, -0.5 / 6.0];
        assert_eq!(r.len(), 3);
        for (t, (&got, &w)) in r.iter().zip(&want).enumerate() {
            assert!((got - w).abs() < 1e-12, "lag {t}: got {got}, want {w}");
        }
    }

    #[test]
    fn biased_autocov_divides_every_lag_by_d() {
        // Same raw sums, one shared denominator nsum·d = 12, and a lag-3 term δ_A[0]·δ_A[3] +
        // 2·δ_B[0]·δ_B[3] = −2.25 + 0.5 = −1.75.
        let (feats, wt) = autocov_fixture();
        let r = component_autocov_biased(&feats, &wt, 2.5, 4);
        let want = [15.0 / 12.0, 2.75 / 12.0, -0.5 / 12.0, -1.75 / 12.0];
        assert_eq!(r.len(), 4);
        for (t, (&got, &w)) in r.iter().zip(&want).enumerate() {
            assert!((got - w).abs() < 1e-12, "lag {t}: got {got}, want {w}");
        }
    }

    #[test]
    fn step_up_runs_the_levinson_recursion_in_reverse_index_order() {
        // φ_m[i] = φ_{m−1}[i] − k_m·φ_{m−1}[m−2−i] — the *reversed* index is the whole content of the
        // recursion, and it is invisible below order 3 where m−2−i collapses to i.
        let (phi, v): (Vec<Vec<f64>>, Vec<f64>) = step_up(&[0.0, 0.5, -0.3, 0.4], 2.0, 3);
        assert_eq!(phi[1], vec![0.5]);
        for (got, want) in phi[2].iter().zip(&[0.65, -0.3]) {
            assert!((got - want).abs() < 1e-12, "order 2: {phi:?}");
        }
        for (got, want) in phi[3].iter().zip(&[0.77, -0.56, 0.4]) {
            assert!((got - want).abs() < 1e-12, "order 3: {phi:?}");
        }
        for (got, want) in v.iter().zip(&[2.0, 1.5, 1.365, 1.1466]) {
            assert!((got - want).abs() < 1e-12, "variances: {v:?}");
        }
    }

    /// Independent re-derivation of the coordinate ascent in [`fit_component_gs`]: pattern search on
    /// each reflection coefficient, best of ±0.15 and ±0.05, clamped to ±0.999.
    fn reference_gs_reflections(
        feats: &[Spherical<f64>],
        wt: &[f64],
        mu_c: f64,
        dim: usize,
    ) -> Vec<f64> {
        let p = GS_ORDER_MAX.min(dim.saturating_sub(1)).max(1);
        let r = component_autocov(feats, wt, mu_c, dim, p);
        let (phi0, _) = levinson_full(&r, p);
        let mut refl = vec![0.0; p + 1];
        for m in 1..=p {
            refl[m] = *phi0[m].last().unwrap();
        }
        let deltas: Vec<(f64, Vec<f64>)> = feats
            .iter()
            .zip(wt)
            .filter(|(_, &w)| w > 0.0)
            .map(|(f, &w)| (w, (0..dim).map(|t| f.mean()[t] - mu_c).collect()))
            .collect();
        let r0 = r[0];
        let eval = |refl: &[f64]| -> f64 {
            let (phi, v) = step_up(refl, r0, p);
            deltas
                .iter()
                .map(|(w, d)| w * ar_loglik_exact(d, &phi, &v, p))
                .sum()
        };
        let mut cur = eval(&refl);
        for _ in 0..GS_REFINE_SWEEPS {
            for m in 1..=p {
                let (mut best_k, mut best_l) = (refl[m], cur);
                for s in [0.15_f64, 0.05] {
                    for dir in [1.0_f64, -1.0] {
                        let cand = (refl[m] + dir * s).clamp(-0.999, 0.999);
                        let saved = refl[m];
                        refl[m] = cand;
                        let l = eval(&refl);
                        refl[m] = saved;
                        if l > best_l {
                            best_l = l;
                            best_k = cand;
                        }
                    }
                }
                refl[m] = best_k;
                cur = best_l;
            }
        }
        refl
    }

    #[test]
    fn gs_refinement_matches_an_independent_ascent_and_never_loses_likelihood() {
        let mut rng = SplitMix64::new(17);
        let dim = 24;
        let feats: Vec<Spherical<f64>> = (0..30)
            .map(|_| {
                let row = ar_window(&mut rng, dim, &[0.6, -0.25]);
                Spherical::from_moments(1.0, row, 0.0)
            })
            .collect();
        let wt = vec![1.0; feats.len()];
        let mu_c: f64 =
            feats.iter().flat_map(|f| f.mean().to_vec()).sum::<f64>() / (30 * dim) as f64;

        let got = fit_component_gs(&feats, &wt, mu_c, dim);
        let StationaryCov::Ar { phi, v } = got else {
            panic!("GS fit must return an AR covariance");
        };

        let p = GS_ORDER_MAX.min(dim - 1).max(1);
        let refl = reference_gs_reflections(&feats, &wt, mu_c, dim);
        let (rphi, rv) = step_up(&refl, component_autocov(&feats, &wt, mu_c, dim, p)[0], p);
        for m in 1..=p {
            for (i, (&a, &b)) in phi[m].iter().zip(&rphi[m]).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0),
                    "order {m} coefficient {i}: {a} vs {b}"
                );
            }
        }
        for (m, (&a, &b)) in v.iter().zip(&rv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9 * a.abs().max(1.0),
                "variance {m}: {a} vs {b}"
            );
        }

        // Ascent, not just agreement: the refined coefficients must not score below the Yule-Walker
        // warm start they were seeded from.
        let deltas: Vec<Vec<f64>> = feats
            .iter()
            .map(|f| (0..dim).map(|t| f.mean()[t] - mu_c).collect())
            .collect();
        let score = |ph: &[Vec<f64>], vv: &[f64]| -> f64 {
            deltas.iter().map(|d| ar_loglik_exact(d, ph, vv, p)).sum()
        };
        let r = component_autocov(&feats, &wt, mu_c, dim, p);
        let (wphi, wv) = levinson_full(&r, p);
        assert!(
            score(&phi, &v) >= score(&wphi, &wv) - 1e-9,
            "refinement lost likelihood against the warm start"
        );
    }

    #[test]
    fn auto_kind_selects_the_argmin_of_an_independently_scored_bic() {
        let mut rng = SplitMix64::new(4);
        let dim = 16;
        let mut feats: Vec<Spherical<f64>> = Vec::new();
        for a in [[0.6, -0.25], [-0.5, 0.2]] {
            for _ in 0..15 {
                let row = ar_window(&mut rng, dim, &a);
                feats.push(Spherical::from_moments(1.0, row, 0.0));
            }
        }
        let ntot: f64 = feats.iter().map(|f| f.weight()).sum();
        for (kind, name) in [
            (CovKind::Ar, "Ar"),
            (CovKind::ToeplitzFull, "ToeplitzFull"),
            (CovKind::GsMle, "GsMle"),
        ] {
            let cov_p = match kind {
                CovKind::Ar => 1 + TOEPLITZ_W_MAX,
                CovKind::ToeplitzFull => dim,
                CovKind::GsMle => 1 + GS_ORDER_MAX,
            };
            let (lo, hi) = (1usize, 2usize);
            let mut want = lo;
            let mut best = f64::INFINITY;
            for k in lo..=hi {
                let g: GmmToeplitz<f64> = gmm_toeplitz_kind(&feats, k, kind, 40, 3);
                let p = k * (1 + cov_p) + (k - 1);
                let bic = -2.0 * g.loglik + p as f64 * ntot.ln();
                if bic < best {
                    best = bic;
                    want = k;
                }
            }
            let got: GmmToeplitz<f64> = gmm_toeplitz_auto_kind(&feats, lo, hi, kind, 40, 3);
            assert_eq!(got.means.len(), want, "{name}");
        }
    }
}
