//! Diagonal-covariance Gaussian Mixture EM on leaf clustering features.
//!
//! Each feature is treated as a mini-Gaussian `N(μ_i, Σ_i)` of weight `n_i` (here `Σ_i` is the
//! diagonal of the feature's covariance). The E-step uses the **expected-log responsibility**
//! with the exact within-feature correction (measured best in `research/RESULTS-estep.md`):
//!
//! ```text
//! log r_ik = log π_k + log N(μ_i | μ_k, σ²_k) − ½ Σ_d (Σ_i)_dd / σ²_kd
//! ```
//!
//! normalised with log-sum-exp. The M-step folds the within-feature variance into the component
//! variance and applies a weak Normal-Inverse-Gamma (MAP) prior to avoid degenerate components.
//! The diagonal model is `O(d)` per (feature, component) and scales to high-dimensional embeddings.

use crate::clustering::kmeans::kmeans;
use crate::feature::{ClusterFeature, SecondMoment};
use crate::types::Real;

/// Result of a GMM-EM run over features.
pub struct Gmm<R: Real> {
    /// Hard label (argmax responsibility) per input feature.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[feature][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_k`.
    pub weights: Vec<R>,
    /// Component means `μ_k`.
    pub means: Vec<Vec<R>>,
    /// Component per-dimension variances `σ²_kd`.
    pub vars: Vec<Vec<R>>,
    /// Weighted data log-likelihood at convergence.
    pub loglik: R,
}

/// Fit a `k`-component diagonal GMM to `features`, warm-started from k-means.
fn gmm_diagonal_once<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Gmm<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let dim = features[0].dim();
    let m = features.len();
    let mu: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let n: Vec<R> = features.iter().map(|f| f.weight()).collect();
    let var: Vec<Vec<R>> = features
        .iter()
        .map(|f| (0..dim).map(|d| f.variance(d)).collect())
        .collect();

    let half = R::from_f64(0.5).unwrap();
    let two_pi = R::from_f64(std::f64::consts::TAU).unwrap();
    let reg = R::from_f64(1e-3).unwrap();
    let tiny = R::from_f64(1e-12).unwrap();
    let gvar = global_variance(&mu, &n, dim);
    let floor: Vec<R> = gvar
        .iter()
        .map(|&g| g * R::from_f64(1e-6).unwrap() + tiny)
        .collect();

    // warm start from k-means
    let km = kmeans(features, k, 50, 1, seed);
    let mut means = km.centers;
    // Floor the warm-start variances. Real data often has constant dimensions (e.g. always-zero
    // border pixels in images) where `gvar` is 0; without flooring, the first E-step divides by that
    // zero and every responsibility becomes NaN, collapsing the model to a single cluster. The M-step
    // already floors each variance — the initial value must too.
    let var0: Vec<R> = gvar
        .iter()
        .zip(&floor)
        .map(|(&g, &f)| if g > f { g } else { f })
        .collect();
    let mut vars = vec![var0; k];
    let mut weights = vec![R::one() / R::from_usize(k).unwrap(); k];

    let mut resp = vec![vec![R::zero(); k]; m];
    let mut loglik = R::neg_infinity();
    let tol = R::from_f64(1e-7).unwrap();

    for it in 0..max_iter {
        // ── E-step ──
        let mut new_ll = R::zero();
        for i in 0..m {
            let mut logr = vec![R::zero(); k];
            for c in 0..k {
                let mut acc = weights[c].ln();
                for d in 0..dim {
                    let s2 = vars[c][d];
                    let diff = mu[i][d] - means[c][d];
                    acc = acc
                        - half * (two_pi * s2).ln()
                        - half * diff * diff / s2
                        - half * var[i][d] / s2;
                }
                logr[c] = acc;
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

        // ── M-step ──
        let mut nk = vec![R::zero(); k];
        let mut new_means = vec![vec![R::zero(); dim]; k];
        for i in 0..m {
            for c in 0..k {
                let wik = n[i] * resp[i][c];
                nk[c] = nk[c] + wik;
                for d in 0..dim {
                    new_means[c][d] = new_means[c][d] + wik * mu[i][d];
                }
            }
        }
        let ntot: R = nk.iter().copied().sum();
        for c in 0..k {
            weights[c] = nk[c] / ntot;
            if nk[c] > R::zero() {
                for v in new_means[c].iter_mut() {
                    *v = *v / nk[c];
                }
            }
        }
        let mut new_vars = vec![vec![R::zero(); dim]; k];
        for i in 0..m {
            for c in 0..k {
                let wik = n[i] * resp[i][c];
                for d in 0..dim {
                    let diff = mu[i][d] - new_means[c][d];
                    new_vars[c][d] = new_vars[c][d] + wik * (var[i][d] + diff * diff);
                }
            }
        }
        for c in 0..k {
            for d in 0..dim {
                let raw = (new_vars[c][d] + reg * gvar[d]) / (nk[c] + reg);
                new_vars[c][d] = if raw > floor[d] { raw } else { floor[d] };
            }
        }
        means = new_means;
        vars = new_vars;

        if it > 0 && (new_ll - loglik).abs() <= tol * loglik.abs().max(R::one()) {
            loglik = new_ll;
            break;
        }
        loglik = new_ll;
    }

    let labels = resp.iter().map(|r| argmax(r)).collect();
    Gmm {
        labels,
        resp,
        weights,
        means,
        vars,
        loglik,
    }
}

/// Number of k-means-seeded EM restarts for the fixed-`k` GMM heads; the fit with the highest data
/// log-likelihood is kept. EM is non-convex, so a single init occasionally lands in a poor local
/// optimum (most visible for full covariance); a few seed-derived restarts make the result robust and
/// still fully deterministic for a given `seed`.
const GMM_N_INIT: u64 = 4;

/// Best of [`GMM_N_INIT`] EM restarts (seeds `seed, seed+1, …`) by data log-likelihood. The restarts
/// are independent, so they run in parallel when the `parallel` feature is on; ties are broken by the
/// lowest seed offset, so the result is deterministic for a given `seed` on either path.
fn best_of_restarts<R, T>(
    seed: u64,
    loglik: impl Fn(&T) -> R + Sync,
    run: impl Fn(u64) -> T + Sync,
) -> T
where
    R: Real,
    T: Send,
{
    #[cfg(feature = "parallel")]
    let cands: Vec<(u64, T)> = {
        use rayon::prelude::*;
        (0..GMM_N_INIT)
            .into_par_iter()
            .map(|r| (r, run(seed.wrapping_add(r))))
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let cands: Vec<(u64, T)> = (0..GMM_N_INIT).map(|r| (r, run(seed.wrapping_add(r)))).collect();
    cands
        .into_iter()
        .max_by(|(ri, a), (rj, b)| {
            loglik(a)
                .partial_cmp(&loglik(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(rj.cmp(ri)) // tie → lower seed offset wins (deterministic)
        })
        .map(|(_, t)| t)
        .expect("GMM_N_INIT >= 1")
}

/// Fit a `k`-component diagonal GMM, keeping the best of [`GMM_N_INIT`] EM restarts by log-likelihood.
pub fn gmm_diagonal<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Gmm<R> {
    best_of_restarts(seed, |g: &Gmm<R>| g.loglik, |s| gmm_diagonal_once(features, k, max_iter, s))
}

/// Total per-dimension variance of the underlying points (between-feature + within-feature),
/// used as a prior scale and variance floor.
fn global_variance<R: Real>(mu: &[Vec<R>], n: &[R], dim: usize) -> Vec<R> {
    let wtot: R = n.iter().copied().sum();
    if wtot <= R::zero() {
        return vec![R::one(); dim];
    }
    let mut mean = vec![R::zero(); dim];
    for (mi, &ni) in mu.iter().zip(n) {
        for (m, &v) in mean.iter_mut().zip(mi) {
            *m = *m + ni * v;
        }
    }
    for m in &mut mean {
        *m = *m / wtot;
    }
    let mut g = vec![R::zero(); dim];
    for (mi, &ni) in mu.iter().zip(n) {
        for d in 0..dim {
            let diff = mi[d] - mean[d];
            g[d] = g[d] + ni * diff * diff;
        }
    }
    for v in &mut g {
        *v = *v / wtot;
    }
    g
}

fn argmax<R: Real>(v: &[R]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Result of a full-covariance GMM-EM run over features.
pub struct GmmFull<R: Real> {
    /// Hard label (argmax responsibility) per input feature.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[feature][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_k`.
    pub weights: Vec<R>,
    /// Component means `μ_k`.
    pub means: Vec<Vec<R>>,
    /// Component covariances `Σ_k` (`k × d × d`).
    pub covs: Vec<Vec<Vec<R>>>,
    /// Weighted data log-likelihood at convergence.
    pub loglik: R,
}

/// Fit a `k`-component full-covariance GMM, warm-started from k-means. Captures rotated /
/// correlated clusters that the diagonal model cannot. Same expected-log E-step with the full
/// within-feature correction `−½ tr(Σ_k⁻¹ Σ_i)`; component covariances are factored (Cholesky)
/// for a stable log-determinant, quadratic form and inverse.
fn gmm_full_once<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> GmmFull<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let dim = features[0].dim();
    let m = features.len();
    let mu: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let n: Vec<R> = features.iter().map(|f| f.weight()).collect();
    // Per-leaf covariance in GMM-ready form: FD features stay low-rank (ℓ×d) instead of dense d×d,
    // so the E/M-steps never materialise a d×d matrix per leaf (preserving FD's O(ℓ·d) memory).
    let sig: Vec<SecondMoment<R>> = features.iter().map(|f| f.second_moment()).collect();

    let half = R::from_f64(0.5).unwrap();
    let log_two_pi = R::from_f64(std::f64::consts::TAU).unwrap().ln();
    let dimr = R::from_usize(dim).unwrap();
    let gcov = global_cov(&mu, &n, &sig, dim);
    let scale = {
        let mut t = R::zero();
        for (d, row) in gcov.iter().enumerate() {
            t = t + row[d];
        }
        (t / dimr).max(R::from_f64(1e-12).unwrap())
    };
    let ridge = R::from_f64(1e-6).unwrap();
    let reg = R::from_f64(1e-3).unwrap();

    let km = kmeans(features, k, 50, 1, seed);
    let mut means = km.centers;
    let mut covs = vec![gcov.clone(); k];
    let mut weights = vec![R::one() / R::from_usize(k).unwrap(); k];

    let mut resp = vec![vec![R::zero(); k]; m];
    let mut loglik = R::neg_infinity();
    let tol = R::from_f64(1e-7).unwrap();

    for it in 0..max_iter {
        let mut chol = Vec::with_capacity(k);
        let mut inv = Vec::with_capacity(k);
        let mut logdet = vec![R::zero(); k];
        for (c, cov) in covs.iter().enumerate() {
            let (l, ld) = chol_regularized(cov, scale, ridge);
            logdet[c] = ld;
            inv.push(crate::linalg::inv_from_chol(&l));
            chol.push(l);
        }

        let mut new_ll = R::zero();
        for i in 0..m {
            let mut logr = vec![R::zero(); k];
            for c in 0..k {
                let delta: Vec<R> = (0..dim).map(|d| mu[i][d] - means[c][d]).collect();
                let quad = crate::linalg::mahalanobis_sq_from_chol(&chol[c], &delta);
                let trace = sig[i].trace_under(&chol[c], &inv[c]);
                logr[c] =
                    weights[c].ln() - half * (dimr * log_two_pi + logdet[c] + quad) - half * trace;
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

        let mut nk = vec![R::zero(); k];
        let mut new_means = vec![vec![R::zero(); dim]; k];
        for i in 0..m {
            for c in 0..k {
                let w = n[i] * resp[i][c];
                nk[c] = nk[c] + w;
                for d in 0..dim {
                    new_means[c][d] = new_means[c][d] + w * mu[i][d];
                }
            }
        }
        let ntot: R = nk.iter().copied().sum();
        for c in 0..k {
            weights[c] = nk[c] / ntot;
            if nk[c] > R::zero() {
                for v in new_means[c].iter_mut() {
                    *v = *v / nk[c];
                }
            }
        }
        let mut new_covs = vec![vec![vec![R::zero(); dim]; dim]; k];
        for i in 0..m {
            for c in 0..k {
                let w = n[i] * resp[i][c];
                let delta: Vec<R> = (0..dim).map(|d| mu[i][d] - new_means[c][d]).collect();
                sig[i].add_scaled(&mut new_covs[c], w); // w · Σ_i (within-leaf scatter)
                for a in 0..dim {
                    for b in 0..dim {
                        new_covs[c][a][b] = new_covs[c][a][b] + w * delta[a] * delta[b];
                    }
                }
            }
        }
        for c in 0..k {
            let denom = nk[c] + reg;
            for a in 0..dim {
                for b in 0..dim {
                    new_covs[c][a][b] = (new_covs[c][a][b] + reg * gcov[a][b]) / denom;
                }
            }
        }
        means = new_means;
        covs = new_covs;

        if it > 0 && (new_ll - loglik).abs() <= tol * loglik.abs().max(R::one()) {
            loglik = new_ll;
            break;
        }
        loglik = new_ll;
    }

    let labels = resp.iter().map(|r| argmax(r)).collect();
    GmmFull {
        labels,
        resp,
        weights,
        means,
        covs,
        loglik,
    }
}

/// Fit a `k`-component full-covariance GMM, keeping the best of [`GMM_N_INIT`] EM restarts by
/// log-likelihood. Full covariance has the most local optima, so the restarts matter most here.
pub fn gmm_full<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> GmmFull<R> {
    best_of_restarts(seed, |g: &GmmFull<R>| g.loglik, |s| gmm_full_once(features, k, max_iter, s))
}

/// Total per-pair covariance of the underlying points (between-feature + within-feature).
fn global_cov<R: Real>(mu: &[Vec<R>], n: &[R], sig: &[SecondMoment<R>], dim: usize) -> Vec<Vec<R>> {
    let wtot: R = n.iter().copied().sum();
    let mut g = vec![vec![R::zero(); dim]; dim];
    if wtot <= R::zero() {
        for (d, row) in g.iter_mut().enumerate() {
            row[d] = R::one();
        }
        return g;
    }
    let mut mean = vec![R::zero(); dim];
    for (mi, &ni) in mu.iter().zip(n) {
        for (mv, &v) in mean.iter_mut().zip(mi) {
            *mv = *mv + ni * v;
        }
    }
    for mv in &mut mean {
        *mv = *mv / wtot;
    }
    for (i, mi) in mu.iter().enumerate() {
        let delta: Vec<R> = (0..dim).map(|d| mi[d] - mean[d]).collect();
        sig[i].add_scaled(&mut g, n[i]); // n_i · Σ_i (within)
        for a in 0..dim {
            for b in 0..dim {
                g[a][b] = g[a][b] + n[i] * delta[a] * delta[b]; // n_i · δδᵀ (between)
            }
        }
    }
    for row in &mut g {
        for v in row.iter_mut() {
            *v = *v / wtot;
        }
    }
    g
}

/// Cholesky of `cov + r·I`, growing the ridge `r` until positive-definite; returns `(L, log|·|)`.
fn chol_regularized<R: Real>(cov: &[Vec<R>], scale: R, ridge0: R) -> (Vec<Vec<R>>, R) {
    let dim = cov.len();
    let mut r = ridge0 * scale;
    for _ in 0..10 {
        let mut a = cov.to_vec();
        for (d, row) in a.iter_mut().enumerate() {
            row[d] = row[d] + r;
        }
        if let Some(l) = crate::linalg::cholesky_lower(&a) {
            let ld = crate::linalg::logdet_from_chol(&l);
            return (l, ld);
        }
        r = r * R::from_f64(10.0).unwrap();
    }
    let mut a = vec![vec![R::zero(); dim]; dim];
    for (d, row) in a.iter_mut().enumerate() {
        row[d] = R::one();
    }
    let l = crate::linalg::cholesky_lower(&a).unwrap();
    let ld = crate::linalg::logdet_from_chol(&l);
    (l, ld)
}

/// Bayesian Information Criterion `−2·loglik + p·ln N` (lower is better); `p` = free parameters,
/// `N` = total point weight. Lets us pick the component count `k` without a user-supplied value.
fn bic<R: Real>(loglik: R, n_params: usize, n_total: R) -> R {
    let two = R::from_f64(2.0).unwrap();
    -two * loglik + R::from_usize(n_params).unwrap() * n_total.ln()
}

fn total_weight<R: Real, C: ClusterFeature<R>>(features: &[C]) -> R {
    features
        .iter()
        .map(|f| f.weight())
        .fold(R::zero(), |a, b| a + b)
}

/// Diagonal GMM with automatic component count: fit every `k ∈ [k_min, k_max]` and keep the
/// lowest-BIC model (`k_max` clamped to the feature count). The chosen `k` is `result.means.len()`.
pub fn gmm_diagonal_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> Gmm<R> {
    let d = features[0].dim();
    let ntot = total_weight(features);
    let k_hi = k_max.min(features.len()).max(1);
    let k_lo = k_min.max(1).min(k_hi);
    let mut best_score = R::infinity();
    let mut best: Option<Gmm<R>> = None;
    for k in k_lo..=k_hi {
        let g = gmm_diagonal_once(features, k, max_iter, seed);
        let p = 2 * k * d + (k - 1); // means + diagonal vars + mixing weights
        let score = bic(g.loglik, p, ntot);
        if score < best_score {
            best_score = score;
            best = Some(g);
        }
    }
    best.unwrap()
}

/// Full-covariance GMM with automatic component count (BIC over `k ∈ [k_min, k_max]`). Each extra
/// component costs `d(d+1)/2` covariance parameters, so BIC favours diagonal-like solutions unless
/// the orientation genuinely pays for itself.
pub fn gmm_full_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> GmmFull<R> {
    let d = features[0].dim();
    let ntot = total_weight(features);
    let k_hi = k_max.min(features.len()).max(1);
    let k_lo = k_min.max(1).min(k_hi);
    let mut best_score = R::infinity();
    let mut best: Option<GmmFull<R>> = None;
    for k in k_lo..=k_hi {
        let g = gmm_full_once(features, k, max_iter, seed);
        let p = k * d + k * d * (d + 1) / 2 + (k - 1); // means + lower-tri cov + mixing
        let score = bic(g.loglik, p, ntot);
        if score < best_score {
            best_score = score;
            best = Some(g);
        }
    }
    best.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros};

    #[test]
    fn gmm_recovers_separated_blobs() {
        let mut rng = SplitMix64::new(11);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.7);
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let g = gmm_diagonal(&micros, 4, 200, 7);
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| g.labels[m]).collect();
        let score = ari(&labels, &truth);
        assert!(score > 0.95, "ARI = {score}");
    }

    #[test]
    fn gmm_full_restarts_keep_highest_loglik() {
        // EM is non-convex; the fixed-k wrapper must return the best of its GMM_N_INIT restarts,
        // never a worse one (guards the local-optimum "dip" fix).
        let mut rng = SplitMix64::new(5);
        let centers = [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]];
        let (pts, _) = blobs(&mut rng, 300, &centers, 1.4); // overlap → multiple local optima
        let (micros, _) = grid_micros(&pts, 0.4);
        let multi = gmm_full(&micros, 4, 200, 0);
        for r in 0..GMM_N_INIT {
            let once = gmm_full_once(&micros, 4, 200, r);
            assert!(
                multi.loglik + 1e-6 >= once.loglik,
                "wrapper loglik {} < single-init seed {} loglik {}",
                multi.loglik,
                r,
                once.loglik
            );
        }
    }

    #[test]
    fn gmm_handles_anisotropic_overlap() {
        // Anisotropic but x-separable clusters: elongated along y, separated along x.
        // The diagonal model must learn a large σ_y and still split on x.
        let mut rng = SplitMix64::new(3);
        let centers = [[0.0, 0.0], [6.0, 0.0]];
        let (mut pts, truth) = blobs(&mut rng, 600, &centers, 0.5);
        for p in &mut pts {
            p[1] *= 2.0; // elongate along y (still dominated by the x separation)
        }
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let g = gmm_diagonal(&micros, 2, 200, 1);
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| g.labels[m]).collect();
        let score = ari(&labels, &truth);
        assert!(score > 0.9, "ARI = {score}");
    }

    #[test]
    fn gmm_full_beats_diagonal_on_crossed_clusters() {
        // Two perpendicular elongated clusters crossing at the origin (an "X"): an axis-aligned
        // model (diagonal GMM / k-means) cannot separate them; full covariance can (orientation).
        let mut rng = SplitMix64::new(13);
        let mut pts: Vec<Vec<f64>> = Vec::new();
        let mut truth: Vec<usize> = Vec::new();
        let r = std::f64::consts::FRAC_1_SQRT_2;
        for (c, sign) in [(0usize, 1.0f64), (1usize, -1.0f64)] {
            for _ in 0..1200 {
                let long = 3.0 * rng.gauss();
                let short = 0.3 * rng.gauss();
                let (ux, uy) = (r, r * sign);
                pts.push(vec![long * ux - short * uy, long * uy + short * ux]);
                truth.push(c);
            }
        }
        let (micros, point_to_micro) = grid_micros(&pts, 0.4);
        let full = gmm_full(&micros, 2, 200, 7);
        let diag = gmm_diagonal(&micros, 2, 200, 7);
        let lf: Vec<usize> = point_to_micro.iter().map(|&m| full.labels[m]).collect();
        let ld: Vec<usize> = point_to_micro.iter().map(|&m| diag.labels[m]).collect();
        let (af, ad) = (ari(&lf, &truth), ari(&ld, &truth));
        assert!(af > 0.6, "full-cov ARI = {af} (diagonal = {ad})");
        assert!(af > ad, "full-cov {af} should beat diagonal {ad}");
    }

    #[test]
    fn auto_k_recovers_cluster_count() {
        // Four well-separated blobs: BIC should select exactly k = 4 with no k supplied.
        let mut rng = SplitMix64::new(21);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.7);
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let g = gmm_diagonal_auto(&micros, 1, 8, 200, 7);
        assert_eq!(g.means.len(), 4, "selected k = {}", g.means.len());
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| g.labels[m]).collect();
        assert!(ari(&labels, &truth) > 0.95);
    }

    #[test]
    fn gmm_full_on_fd_sketch_low_rank_and_auto() {
        // Full-cov GMM over Frequent-Directions leaves exercises the low-rank second-moment path
        // (`FdSketch::second_moment` ⇒ `SecondMoment::LowRank` ⇒ `trace_under` / `add_scaled`), plus
        // the auto-k variant `gmm_full_auto`.
        use crate::feature::{ClusterFeature, FdSketch};
        let mut rng = SplitMix64::new(31);
        let centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 240, &centers, 0.6);
        let mut micros: Vec<FdSketch<f64>> = (0..6).map(|_| FdSketch::new(2)).collect();
        for (i, (p, &t)) in pts.iter().zip(&truth).enumerate() {
            micros[t * 2 + (i % 2)].push(p, 1.0); // 2 FD leaves per blob
        }
        let g = gmm_full(&micros, 3, 100, 7);
        assert_eq!(g.means.len(), 3);
        let ga = gmm_full_auto(&micros, 1, 5, 100, 7);
        assert!(!ga.means.is_empty() && ga.means.len() <= 5);
    }

    #[test]
    fn gmm_diagonal_survives_constant_dimension() {
        // Real data routinely has constant columns (e.g. always-zero image-border pixels) where the
        // global variance is 0. Without flooring the warm-start variance, the first E-step divides by
        // that zero, every responsibility becomes NaN, and the model collapses to a single cluster.
        use crate::feature::{ClusterFeature, Diagonal};
        let mut rng = SplitMix64::new(2);
        let (pts, truth) = blobs(&mut rng, 200, &[[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]], 0.5);
        let feats: Vec<Diagonal<f64>> = pts
            .iter()
            .map(|p| {
                let mut f = <Diagonal<f64> as ClusterFeature<f64>>::new(3);
                f.push(&[p[0], p[1], 0.0], 1.0); // 3rd dimension is constant 0 → gvar = 0
                f
            })
            .collect();
        let g = gmm_diagonal(&feats, 3, 200, 1);
        let score = ari(&g.labels, &truth);
        assert!(score > 0.9, "constant-dim collapse: ARI = {score}");
    }
}
