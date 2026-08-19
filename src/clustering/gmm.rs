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
use crate::mixture::Mixture;
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
    /// The fitted density, for scoring raw points.
    pub mixture: Mixture,
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
    let gvar = global_variance(&mu, &n, &var, dim);
    // Variance floor, in two parts. The per-dimension part (`1e-3·gvar_d`) keeps a variance from
    // collapsing relative to the spread that dimension actually has — a collapsed diagonal variance
    // makes the E-step wildly over-confident on one dimension and the mixture over-fits away from the
    // (good) k-means warm start (e.g. `digits`, 64-D). On its own it cannot floor a dimension that has
    // *no* spread, since `1e-3 · 0 = 0`; the global part (`VAR_FLOOR_REL · mean_d gvar_d`) is what
    // bounds `1/σ²_cd` there, and mirrors the full head's `ridge = 1e-6 · tr(gcov)/d`.
    let scale = gvar.iter().copied().sum::<R>() / R::from_usize(dim.max(1)).unwrap();
    let abs_floor = scale * R::from_f64(VAR_FLOOR_REL).unwrap();
    let floor: Vec<R> = gvar
        .iter()
        .map(|&g| (g * R::from_f64(1e-3).unwrap()).max(abs_floor) + tiny)
        .collect();

    // warm start from k-means
    let km = kmeans(features, k, 50, 1, seed);
    let mut means = km.centers;
    // Seed each component's variance from its own k-means cluster, not from `gvar`. `gvar` is the
    // spread of the *whole* dataset, so in a well-separated dimension it is dominated by the
    // between-cluster distance and overstates the within-component spread — by 34x on four blobs
    // eight units apart. Inflating every dimension by the same factor would be harmless: it cancels
    // in the argmax. It is the *ratio across* dimensions that decides responsibilities, and a
    // dimension whose spread is already purely within-component — a sparse binary feature, a
    // near-constant pixel — gets no such inflation, so it silently gains that factor of relative
    // weight and can dominate the first E-step. Four blobs plus twelve 2%-density binary columns
    // collapsed every component onto the global mean (ARI 1.000 -> 0.000) until the seed came from
    // the k-means partition. Still floored: real data has constant dimensions where the variance is
    // 0, and without a floor the first E-step divides by zero and every responsibility becomes NaN.
    let mut vars = vec![vec![R::zero(); dim]; k];
    {
        let mut nk = vec![R::zero(); k];
        for (i, &c) in km.labels.iter().enumerate() {
            nk[c] = nk[c] + n[i];
            for d in 0..dim {
                let diff = mu[i][d] - means[c][d];
                vars[c][d] = vars[c][d] + n[i] * (diff * diff + var[i][d]);
            }
        }
        for c in 0..k {
            for d in 0..dim {
                let raw = if nk[c] > R::zero() {
                    vars[c][d] / nk[c]
                } else {
                    gvar[d]
                };
                vars[c][d] = if raw > floor[d] { raw } else { floor[d] };
            }
        }
    }
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
    let mixture = Mixture::diagonal(&weights, &means, &vars);
    Gmm {
        labels,
        resp,
        weights,
        means,
        vars,
        loglik,
        mixture,
    }
}

/// Number of k-means-seeded EM restarts for the fixed-`k` GMM heads; the fit with the highest data
/// log-likelihood is kept. EM is non-convex, so a single init occasionally lands in a poor local
/// optimum (most visible for full covariance); a few seed-derived restarts make the result robust and
/// still fully deterministic for a given `seed`.
const GMM_N_INIT: u64 = 4;

/// Floor on every diagonal variance as a fraction of the mean per-dimension variance — the global
/// counterpart of the per-dimension `1e-3·gvar_d` floor, and the same value the full-covariance head
/// uses for its Cholesky ridge. It bounds `1/σ²_cd` at `1/(VAR_FLOOR_REL·scale)` in a dimension whose
/// own variance is (near) zero, which is where an unbounded precision does its damage: a leaf mean is
/// a local average and sits close to any component mean there, but a raw point need not, so the
/// unfloored term can dominate the whole log-density when such a point is scored.
const VAR_FLOOR_REL: f64 = 1e-6;

/// Best of `n_init` EM restarts (seeds `seed, seed+1, …`) by data log-likelihood. The restarts are
/// independent, so they run in parallel when the `parallel` feature is on; ties are broken by the
/// lowest seed offset, so the result is deterministic for a given `seed` on either path. Shared by the
/// GMM heads (`GMM_N_INIT`) and the AR/Toeplitz head (`TOEPLITZ_N_INIT`).
pub(crate) fn best_of_restarts<R, T>(
    n_init: u64,
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
        (0..n_init)
            .into_par_iter()
            .map(|r| (r, run(seed.wrapping_add(r))))
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let cands: Vec<(u64, T)> = (0..n_init)
        .map(|r| (r, run(seed.wrapping_add(r))))
        .collect();
    cands
        .into_iter()
        .max_by(|(ri, a), (rj, b)| {
            loglik(a)
                .partial_cmp(&loglik(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(rj.cmp(ri)) // tie → lower seed offset wins (deterministic)
        })
        .map(|(_, t)| t)
        .expect("n_init >= 1")
}

/// Fit a `k`-component diagonal GMM, keeping the best of [`GMM_N_INIT`] EM restarts by log-likelihood.
pub fn gmm_diagonal<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Gmm<R> {
    best_of_restarts(
        GMM_N_INIT,
        seed,
        |g: &Gmm<R>| g.loglik,
        |s| gmm_diagonal_once(features, k, max_iter, s),
    )
}

/// Total per-dimension variance of the underlying points (between-feature + within-feature), used as
/// a prior scale and variance floor. The within-feature term is what makes this a variance of
/// *points* rather than of leaf means — [`global_cov`] carries the same term for the full-covariance
/// head, and without it the scale shrinks with every unit of tree compression.
fn global_variance<R: Real>(mu: &[Vec<R>], n: &[R], var: &[Vec<R>], dim: usize) -> Vec<R> {
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
    for ((mi, vi), &ni) in mu.iter().zip(var).zip(n) {
        for d in 0..dim {
            let diff = mi[d] - mean[d];
            g[d] = g[d] + ni * (diff * diff + vi[d]);
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
    /// The fitted density, for scoring raw points.
    pub mixture: Mixture,
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
    // Seed each component from its own k-means cluster rather than from `gcov`, for the reason given
    // at the same point in `gmm_diagonal_once`: the global covariance carries the between-cluster
    // separation, which inflates well-separated dimensions but not those whose spread is already
    // within-component, and that uneven inflation alone can decide the first E-step.
    let mut covs = vec![vec![vec![R::zero(); dim]; dim]; k];
    {
        let dfloor = R::from_f64(1e-3).unwrap();
        let mut nk = vec![R::zero(); k];
        for (i, &c) in km.labels.iter().enumerate() {
            nk[c] = nk[c] + n[i];
            let delta: Vec<R> = (0..dim).map(|d| mu[i][d] - means[c][d]).collect();
            sig[i].add_scaled(&mut covs[c], n[i]);
            for a in 0..dim {
                for b in 0..dim {
                    covs[c][a][b] = covs[c][a][b] + n[i] * delta[a] * delta[b];
                }
            }
        }
        for c in 0..k {
            if nk[c] > R::zero() {
                for row in covs[c].iter_mut() {
                    for v in row.iter_mut() {
                        *v = *v / nk[c];
                    }
                }
            } else {
                covs[c] = gcov.clone();
            }
            for d in 0..dim {
                let f = dfloor * gcov[d][d];
                if covs[c][d][d] < f {
                    covs[c][d][d] = f;
                }
            }
        }
    }
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
        let dfloor = R::from_f64(1e-3).unwrap();
        for c in 0..k {
            let denom = nk[c] + reg;
            for a in 0..dim {
                for b in 0..dim {
                    new_covs[c][a][b] = (new_covs[c][a][b] + reg * gcov[a][b]) / denom;
                }
            }
            // Per-dimension diagonal floor, relative to the global variance `gcov[d][d]`. In high
            // dimensions a component's covariance can go near-singular along low-variance directions;
            // the expected-log trace correction −½ tr(Σ_k⁻¹ Σ_i) then explodes and starves the
            // component, emptying it and collapsing the recovered count. Flooring the diagonal keeps
            // Σ_k well-conditioned. Relative to `gcov[d][d]` (not the global mean scale, which is
            // inflated by between-cluster separation and would over-regularize tight clusters);
            // off-diagonals (orientation) are left untouched, so anisotropic fits are preserved.
            for d in 0..dim {
                let floor = dfloor * gcov[d][d];
                if new_covs[c][d][d] < floor {
                    new_covs[c][d][d] = floor;
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
    // Factor the converged covariances once more, with the ridge policy the E-step used, so the
    // point density and the last E-step speak of the same `Σ_k`.
    let (chols, logdets): (Vec<Vec<Vec<R>>>, Vec<R>) = covs
        .iter()
        .map(|cov| chol_regularized(cov, scale, ridge))
        .unzip();
    let mixture = Mixture::full(&weights, &means, &chols, &logdets);
    GmmFull {
        labels,
        resp,
        weights,
        means,
        covs,
        loglik,
        mixture,
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
    best_of_restarts(
        GMM_N_INIT,
        seed,
        |g: &GmmFull<R>| g.loglik,
        |s| gmm_full_once(features, k, max_iter, s),
    )
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
pub(crate) fn chol_regularized<R: Real>(cov: &[Vec<R>], scale: R, ridge0: R) -> (Vec<Vec<R>>, R) {
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
    use crate::feature::{Diagonal, Spherical};
    use std::collections::HashMap;

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
    fn sparse_binary_columns_do_not_dilute_a_separable_partition() {
        // Four blobs eight units apart, plus twelve binary columns that are 1 with probability 0.02.
        // The columns carry no signal and each has variance ~0.02, while the two informative ones have
        // a *global* variance of ~16.5 — dominated by the between-blob distance, not by the 0.49 spread
        // within a blob. Seeding every component with that global variance therefore weights a junk
        // column ~900x more per dimension than an informative one, and a single spike outweighs the
        // whole blob separation: every component collapses onto the global mean (ARI 1.00 -> 0.00).
        // Seeding from the k-means partition instead measures the within-component spread directly.
        let mut rng = SplitMix64::new(4);
        let centers = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]];
        let (core, truth) = blobs(&mut rng, 400, &centers, 0.7);
        let dim = 2 + 12;
        let pts: Vec<Vec<f64>> = core
            .iter()
            .map(|p| {
                let mut row = Vec::with_capacity(dim);
                row.extend_from_slice(p);
                for _ in 2..dim {
                    row.push(if rng.next_f64() < 0.02 { 1.0 } else { 0.0 });
                }
                row
            })
            .collect();

        // Micro-cluster over *all* dimensions, as a CF-tree does: points that share a spike pattern
        // land in the same leaf, so the leaf means carry real spread in the junk columns. Bucketing on
        // the informative plane alone hides the bug — every leaf mean is then ~0.02 in every junk
        // column, the terms cancel in the argmax, and even the unfixed seed recovers the partition.
        let mut map: HashMap<Vec<i64>, usize> = HashMap::new();
        let mut micros: Vec<Spherical<f64>> = Vec::new();
        let mut point_to_micro = vec![0usize; pts.len()];
        for (i, p) in pts.iter().enumerate() {
            let key: Vec<i64> = p.iter().map(|&v| (v / 0.5).round() as i64).collect();
            let idx = *map.entry(key).or_insert_with(|| {
                micros.push(Spherical::new(dim));
                micros.len() - 1
            });
            micros[idx].push(p, 1.0);
            point_to_micro[i] = idx;
        }

        for (head, labels) in [
            ("gmm", gmm_diagonal(&micros, 4, 200, 3).labels),
            ("gmm-full", gmm_full(&micros, 4, 200, 3).labels),
        ] {
            let assigned: Vec<usize> = point_to_micro.iter().map(|&m| labels[m]).collect();
            let score = ari(&assigned, &truth);
            assert!(score > 0.95, "{head} ARI = {score}");
        }
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
    fn gmm_full_floors_covariance_in_high_dim() {
        // Regression for the high-dimensional covariance collapse. When a full-covariance component's
        // own covariance goes near-singular along the directions in which its blob is tight (the norm
        // in high dimensions), the expected-log trace correction −½ tr(Σ_k⁻¹ Σ_i) turns over-confident
        // and a component can be starved to zero responsibility, dropping the recovered count below k
        // (observed on `digits`: 10 → 9). The per-dimension floor on each component's covariance
        // diagonal, `1e-3·gcov_dd`, keeps every Σ_k well-conditioned. Tight, well-separated blobs in
        // 10-D: within each blob the variance is ~0 in every dimension, so the raw component covariance
        // is ~0 while the global variance is large along the axes that separate the blobs.
        use crate::feature::{ClusterFeature, SecondMoment, Spherical};
        let dim = 10;
        let k = 5;
        let mut rng = SplitMix64::new(101);
        let mut micros: Vec<Spherical<f64>> = Vec::new();
        let mut truth: Vec<usize> = Vec::new();
        for c in 0..k {
            for _ in 0..4 {
                let mut f = Spherical::new(dim);
                for _ in 0..25 {
                    let mut p = vec![0.0; dim];
                    p[c] = 6.0;
                    for pd in &mut p {
                        *pd += 1e-4 * rng.gauss(); // negligible within-blob spread ⇒ near-singular Σ_k
                    }
                    f.push(&p, 1.0);
                }
                micros.push(f);
                truth.push(c);
            }
        }
        let g = gmm_full(&micros, k, 200, 7);

        // (a) every component stays populated — no starvation collapse.
        let distinct: std::collections::HashSet<usize> = g.labels.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            k,
            "recovered {} of {k} components",
            distinct.len()
        );
        assert!(
            ari(&g.labels, &truth) > 0.99,
            "ARI = {}",
            ari(&g.labels, &truth)
        );

        // (b) the per-dimension covariance-diagonal floor is applied (fails if the floor is removed:
        // the raw component covariances here are ~1e-8, far below `1e-3·gcov_dd` on the separating axes).
        let mu: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let n: Vec<f64> = micros.iter().map(|f| f.weight()).collect();
        let sig: Vec<SecondMoment<f64>> = micros.iter().map(|f| f.second_moment()).collect();
        let gcov = global_cov(&mu, &n, &sig, dim);
        let mut floor_bound = false;
        for cov in &g.covs {
            for d in 0..dim {
                let floor = 1e-3 * gcov[d][d];
                assert!(
                    cov[d][d] + 1e-12 >= floor,
                    "cov diag {} below floor {floor} in dim {d}",
                    cov[d][d]
                );
                if floor > 1e-6 {
                    floor_bound = true; // at least one dimension where the floor genuinely binds
                }
            }
        }
        assert!(floor_bound, "test did not exercise a binding floor");
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

    #[test]
    fn no_dimension_gets_an_unbounded_precision() {
        // A dimension with (near) no spread of its own gets no floor from the `1e-3·gvar_d` term, so
        // `1/σ²_cd` is free to reach 1e12. That never showed while the density only ever scored leaf
        // means — a local average sits close to any component mean — and dominates the whole
        // log-density the moment a raw point is scored (MNIST: ~70 always-zero pixels).
        use crate::feature::{ClusterFeature, Diagonal};
        let mut rng = SplitMix64::new(9);
        let (pts, _) = blobs(&mut rng, 300, &[[0.0, 0.0], [7.0, 0.0], [0.0, 7.0]], 0.6);
        let dead = 10;
        let feats: Vec<Diagonal<f64>> = pts
            .iter()
            .map(|p| {
                let mut row = vec![0.0; 2 + dead];
                row[0] = p[0];
                row[1] = p[1];
                let mut f = <Diagonal<f64> as ClusterFeature<f64>>::new(2 + dead);
                f.push(&row, 1.0);
                f
            })
            .collect();
        let g = gmm_diagonal(&feats, 3, 200, 1);
        let scale: f64 = {
            let gv = global_variance(
                &feats.iter().map(|f| f.mean().to_vec()).collect::<Vec<_>>(),
                &feats.iter().map(|f| f.weight()).collect::<Vec<_>>(),
                &feats
                    .iter()
                    .map(|f| (0..2 + dead).map(|d| f.variance(d)).collect())
                    .collect::<Vec<Vec<f64>>>(),
                2 + dead,
            );
            gv.iter().sum::<f64>() / (2 + dead) as f64
        };
        let cap = 1.0 / (VAR_FLOOR_REL * scale);
        for (c, v) in g.vars.iter().enumerate() {
            for (d, &s2) in v.iter().enumerate() {
                assert!(
                    1.0 / s2 <= cap,
                    "component {c} dim {d}: 1/σ² = {:e} exceeds the floor's cap {cap:e}",
                    1.0 / s2
                );
            }
        }
    }

    /// Independent re-derivation of [`gmm_diagonal_once`], written from the EM equations rather than
    /// from the source, so that an operator swap anywhere in the fit shows up as a disagreement.
    /// The end-to-end tests above all assert an ARI, which four well-separated blobs make insensitive
    /// to the arithmetic: they hold even when the E-step is corrupted.
    ///
    /// Returns `(resp, weights, means, vars, loglik, iterations)`.
    #[allow(clippy::type_complexity)]
    fn reference_diagonal_em<C: ClusterFeature<f64>>(
        features: &[C],
        k: usize,
        max_iter: usize,
        seed: u64,
    ) -> (
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        f64,
        usize,
    ) {
        let dim = features[0].dim();
        let m = features.len();
        let mu: Vec<Vec<f64>> = features.iter().map(|f| f.mean().to_vec()).collect();
        let n: Vec<f64> = features.iter().map(|f| f.weight()).collect();
        let var: Vec<Vec<f64>> = features
            .iter()
            .map(|f| (0..dim).map(|d| f.variance(d)).collect())
            .collect();

        let wtot: f64 = n.iter().sum();
        let mut gmean = vec![0.0; dim];
        for i in 0..m {
            for d in 0..dim {
                gmean[d] += n[i] * mu[i][d];
            }
        }
        for g in gmean.iter_mut() {
            *g /= wtot;
        }
        let mut gvar = vec![0.0; dim];
        for i in 0..m {
            for d in 0..dim {
                gvar[d] += n[i] * ((mu[i][d] - gmean[d]).powi(2) + var[i][d]);
            }
        }
        for g in gvar.iter_mut() {
            *g /= wtot;
        }

        let scale = gvar.iter().sum::<f64>() / dim.max(1) as f64;
        let abs_floor = scale * VAR_FLOOR_REL;
        let floor: Vec<f64> = gvar
            .iter()
            .map(|&g| (g * 1e-3).max(abs_floor) + 1e-12)
            .collect();

        let km = kmeans(features, k, 50, 1, seed);
        let mut means = km.centers.clone();
        let mut vars = vec![vec![0.0; dim]; k];
        let mut seed_nk = vec![0.0; k];
        for (i, &c) in km.labels.iter().enumerate() {
            seed_nk[c] += n[i];
            for d in 0..dim {
                vars[c][d] += n[i] * ((mu[i][d] - means[c][d]).powi(2) + var[i][d]);
            }
        }
        for c in 0..k {
            for d in 0..dim {
                let raw = if seed_nk[c] > 0.0 {
                    vars[c][d] / seed_nk[c]
                } else {
                    gvar[d]
                };
                vars[c][d] = raw.max(floor[d]);
            }
        }

        let mut weights = vec![1.0 / k as f64; k];
        let mut resp = vec![vec![0.0; k]; m];
        let mut loglik = f64::NEG_INFINITY;
        let mut iters = 0;

        for it in 0..max_iter {
            iters = it + 1;
            let mut new_ll = 0.0;
            for i in 0..m {
                let logr: Vec<f64> = (0..k)
                    .map(|c| {
                        let mut acc = weights[c].ln();
                        for d in 0..dim {
                            let s2 = vars[c][d];
                            let diff = mu[i][d] - means[c][d];
                            acc -= 0.5 * (std::f64::consts::TAU * s2).ln();
                            acc -= 0.5 * (diff * diff) / s2;
                            acc -= 0.5 * var[i][d] / s2;
                        }
                        acc
                    })
                    .collect();
                let mx = logr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = mx + logr.iter().map(|&l| (l - mx).exp()).sum::<f64>().ln();
                new_ll += n[i] * lse;
                for c in 0..k {
                    resp[i][c] = (logr[c] - lse).exp();
                }
            }

            let nk: Vec<f64> = (0..k)
                .map(|c| (0..m).map(|i| n[i] * resp[i][c]).sum())
                .collect();
            let ntot: f64 = nk.iter().sum();
            let mut new_means = vec![vec![0.0; dim]; k];
            for c in 0..k {
                weights[c] = nk[c] / ntot;
                for d in 0..dim {
                    let acc: f64 = (0..m).map(|i| n[i] * resp[i][c] * mu[i][d]).sum();
                    new_means[c][d] = if nk[c] > 0.0 { acc / nk[c] } else { acc };
                }
            }
            let mut new_vars = vec![vec![0.0; dim]; k];
            for c in 0..k {
                for d in 0..dim {
                    let acc: f64 = (0..m)
                        .map(|i| {
                            n[i] * resp[i][c] * (var[i][d] + (mu[i][d] - new_means[c][d]).powi(2))
                        })
                        .sum();
                    new_vars[c][d] = ((acc + 1e-3 * gvar[d]) / (nk[c] + 1e-3)).max(floor[d]);
                }
            }
            means = new_means;
            vars = new_vars;

            if it > 0 && (new_ll - loglik).abs() <= 1e-7 * loglik.abs().max(1.0) {
                loglik = new_ll;
                break;
            }
            loglik = new_ll;
        }
        (resp, weights, means, vars, loglik, iters)
    }

    #[track_caller]
    fn assert_close_rel(got: f64, want: f64, tol: f64, what: &str) {
        let scale = got.abs().max(want.abs()).max(1.0);
        assert!(
            (got - want).abs() <= tol * scale,
            "{what}: got {got}, want {want}"
        );
    }

    /// Overlapping blobs, so every responsibility sits strictly between 0 and 1 and each term of the
    /// E-step actually moves the answer. Well-separated blobs saturate the posterior to 0/1 and hide
    /// arithmetic errors.
    fn soft_fixture() -> Vec<Diagonal<f64>> {
        let mut rng = SplitMix64::new(2024);
        let centers = [[0.0, 0.0], [2.5, 0.4], [1.1, 2.6]];
        let (pts, _) = blobs(&mut rng, 90, &centers, 1.3);
        // `Diagonal`, not `Spherical`, and a third dimension that never varies: `Spherical` carries
        // one pooled scalar variance for *every* dimension, so no dimension of a spherical fixture
        // can have zero spread. Here `gvar_2 = 0`, the relative floor `1e-3·gvar_d` vanishes with it,
        // and the absolute floor `VAR_FLOOR_REL · mean_d gvar_d` is what bounds `1/σ²_c2` — the only
        // path on which `scale` reaches the answer.
        let mut map: HashMap<(i64, i64), usize> = HashMap::new();
        let mut cfs: Vec<Diagonal<f64>> = Vec::new();
        for p in &pts {
            let key = ((p[0] / 0.45).round() as i64, (p[1] / 0.45).round() as i64);
            let idx = *map.entry(key).or_insert_with(|| {
                cfs.push(<Diagonal<f64> as ClusterFeature<f64>>::new(3));
                cfs.len() - 1
            });
            cfs[idx].push(&[p[0], p[1], 0.0], 1.0);
        }
        cfs
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index form mirrors the EM equations
    fn diagonal_em_matches_an_independent_reference_iteration_for_iteration() {
        let micros = soft_fixture();
        let (k, iters, seed) = (3, 5, 17);
        let g = gmm_diagonal_once(&micros, k, iters, seed);
        let (rresp, rweights, rmeans, rvars, rloglik, ran) =
            reference_diagonal_em(&micros, k, iters, seed);
        assert_eq!(
            ran, iters,
            "fixture converged early, so the path is untested"
        );

        assert_close_rel(g.loglik, rloglik, 1e-9, "loglik");
        for c in 0..k {
            assert_close_rel(g.weights[c], rweights[c], 1e-9, "weight");
            for d in 0..3 {
                assert_close_rel(g.means[c][d], rmeans[c][d], 1e-9, "mean");
                assert_close_rel(g.vars[c][d], rvars[c][d], 1e-9, "var");
            }
        }
        let mut soft = 0;
        for i in 0..micros.len() {
            for c in 0..k {
                assert_close_rel(g.resp[i][c], rresp[i][c], 1e-9, "resp");
                if g.resp[i][c] > 1e-3 && g.resp[i][c] < 1.0 - 1e-3 {
                    soft += 1;
                }
            }
            assert_eq!(
                g.labels[i],
                argmax(&rresp[i]),
                "label disagrees with the reference posterior"
            );
        }
        assert!(soft > 20, "fixture is not soft enough: {soft} soft entries");
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index form mirrors the EM equations
    fn diagonal_em_stops_on_the_relative_loglik_test() {
        // Same fixture run to convergence. The tolerance is loose because the two implementations
        // sum in a different order and may stop an iteration apart; it is still far tighter than the
        // O(1) shift any swapped operator in the stopping test produces (`<=` inverted stops after
        // one iteration, `&&` widened stops before the first M-step is scored).
        let micros = soft_fixture();
        let (k, seed) = (3, 17);
        let g = gmm_diagonal_once(&micros, k, 1000, seed);
        let (_, _, rmeans, _, rloglik, ran) = reference_diagonal_em(&micros, k, 1000, seed);
        assert!(ran > 5 && ran < 1000, "expected convergence, ran {ran}");
        assert_close_rel(g.loglik, rloglik, 1e-6, "converged loglik");
        for c in 0..k {
            for d in 0..3 {
                assert_close_rel(g.means[c][d], rmeans[c][d], 1e-5, "converged mean");
            }
        }
        // Monotone ascent is the defining property of EM: no iteration may lower the objective.
        let mut prev = f64::NEG_INFINITY;
        for t in 1..=12 {
            let ll = gmm_diagonal_once(&micros, k, t, seed).loglik;
            assert!(
                ll >= prev - 1e-9,
                "loglik fell at iteration {t}: {prev} -> {ll}"
            );
            prev = ll;
        }
    }

    #[test]
    fn argmax_keeps_the_first_of_equal_scores() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 3.0, 1.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    /// Independent re-derivation of [`gmm_full_once`], the full-covariance twin of
    /// [`reference_diagonal_em`]. Returns `(resp, weights, means, covs, loglik, iterations)`.
    #[allow(clippy::type_complexity, clippy::needless_range_loop)]
    fn reference_full_em<C: ClusterFeature<f64>>(
        features: &[C],
        k: usize,
        max_iter: usize,
        seed: u64,
    ) -> (
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<Vec<Vec<f64>>>,
        f64,
        usize,
    ) {
        let dim = features[0].dim();
        let m = features.len();
        let mu: Vec<Vec<f64>> = features.iter().map(|f| f.mean().to_vec()).collect();
        let n: Vec<f64> = features.iter().map(|f| f.weight()).collect();
        let sig: Vec<SecondMoment<f64>> = features.iter().map(|f| f.second_moment()).collect();
        let gcov = global_cov(&mu, &n, &sig, dim);
        let scale = ((0..dim).map(|d| gcov[d][d]).sum::<f64>() / dim as f64).max(1e-12);
        let (ridge, reg, dfloor) = (1e-6, 1e-3, 1e-3);

        let km = kmeans(features, k, 50, 1, seed);
        let mut means = km.centers.clone();
        let mut covs = vec![vec![vec![0.0; dim]; dim]; k];
        let mut seed_nk = vec![0.0; k];
        for (i, &c) in km.labels.iter().enumerate() {
            seed_nk[c] += n[i];
            sig[i].add_scaled(&mut covs[c], n[i]);
            for a in 0..dim {
                for b in 0..dim {
                    covs[c][a][b] += n[i] * (mu[i][a] - means[c][a]) * (mu[i][b] - means[c][b]);
                }
            }
        }
        for c in 0..k {
            if seed_nk[c] > 0.0 {
                for a in 0..dim {
                    for b in 0..dim {
                        covs[c][a][b] /= seed_nk[c];
                    }
                }
            } else {
                covs[c] = gcov.clone();
            }
            for d in 0..dim {
                covs[c][d][d] = covs[c][d][d].max(dfloor * gcov[d][d]);
            }
        }

        let mut weights = vec![1.0 / k as f64; k];
        let mut resp = vec![vec![0.0; k]; m];
        let mut loglik = f64::NEG_INFINITY;
        let mut iters = 0;

        for it in 0..max_iter {
            iters = it + 1;
            let mut chol = Vec::with_capacity(k);
            let mut inv = Vec::with_capacity(k);
            let mut logdet = vec![0.0; k];
            for c in 0..k {
                let (l, ld) = chol_regularized(&covs[c], scale, ridge);
                logdet[c] = ld;
                inv.push(crate::linalg::inv_from_chol(&l));
                chol.push(l);
            }

            let mut new_ll = 0.0;
            for i in 0..m {
                let logr: Vec<f64> = (0..k)
                    .map(|c| {
                        let delta: Vec<f64> = (0..dim).map(|d| mu[i][d] - means[c][d]).collect();
                        let quad = crate::linalg::mahalanobis_sq_from_chol(&chol[c], &delta);
                        let trace = sig[i].trace_under(&chol[c], &inv[c]);
                        weights[c].ln()
                            - 0.5 * (dim as f64 * std::f64::consts::TAU.ln() + logdet[c] + quad)
                            - 0.5 * trace
                    })
                    .collect();
                let mx = logr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = mx + logr.iter().map(|&l| (l - mx).exp()).sum::<f64>().ln();
                new_ll += n[i] * lse;
                for c in 0..k {
                    resp[i][c] = (logr[c] - lse).exp();
                }
            }

            let nk: Vec<f64> = (0..k)
                .map(|c| (0..m).map(|i| n[i] * resp[i][c]).sum())
                .collect();
            let ntot: f64 = nk.iter().sum();
            let mut new_means = vec![vec![0.0; dim]; k];
            for c in 0..k {
                weights[c] = nk[c] / ntot;
                for d in 0..dim {
                    let acc: f64 = (0..m).map(|i| n[i] * resp[i][c] * mu[i][d]).sum();
                    new_means[c][d] = if nk[c] > 0.0 { acc / nk[c] } else { acc };
                }
            }
            let mut new_covs = vec![vec![vec![0.0; dim]; dim]; k];
            for i in 0..m {
                for c in 0..k {
                    let w = n[i] * resp[i][c];
                    sig[i].add_scaled(&mut new_covs[c], w);
                    for a in 0..dim {
                        for b in 0..dim {
                            new_covs[c][a][b] +=
                                w * (mu[i][a] - new_means[c][a]) * (mu[i][b] - new_means[c][b]);
                        }
                    }
                }
            }
            for c in 0..k {
                for a in 0..dim {
                    for b in 0..dim {
                        new_covs[c][a][b] = (new_covs[c][a][b] + reg * gcov[a][b]) / (nk[c] + reg);
                    }
                }
                for d in 0..dim {
                    new_covs[c][d][d] = new_covs[c][d][d].max(dfloor * gcov[d][d]);
                }
            }
            means = new_means;
            covs = new_covs;

            if it > 0 && (new_ll - loglik).abs() <= 1e-7 * loglik.abs().max(1.0) {
                loglik = new_ll;
                break;
            }
            loglik = new_ll;
        }
        (resp, weights, means, covs, loglik, iters)
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index form mirrors the EM equations
    fn full_em_matches_an_independent_reference_iteration_for_iteration() {
        let micros = soft_fixture();
        let (k, iters, seed) = (3, 4, 17);
        let g: GmmFull<f64> = gmm_full_once(&micros, k, iters, seed);
        let (rresp, rweights, rmeans, rcovs, rloglik, ran) =
            reference_full_em(&micros, k, iters, seed);
        assert_eq!(
            ran, iters,
            "fixture converged early, so the path is untested"
        );

        assert_close_rel(g.loglik, rloglik, 1e-9, "loglik");
        for c in 0..k {
            assert_close_rel(g.weights[c], rweights[c], 1e-9, "weight");
            for a in 0..3 {
                assert_close_rel(g.means[c][a], rmeans[c][a], 1e-9, "mean");
                for b in 0..3 {
                    assert_close_rel(g.covs[c][a][b], rcovs[c][a][b], 1e-9, "cov");
                }
            }
        }
        let mut soft = 0;
        for i in 0..micros.len() {
            for c in 0..k {
                assert_close_rel(g.resp[i][c], rresp[i][c], 1e-9, "resp");
                if g.resp[i][c] > 1e-3 && g.resp[i][c] < 1.0 - 1e-3 {
                    soft += 1;
                }
            }
            assert_eq!(g.labels[i], argmax(&rresp[i]), "label disagrees");
        }
        assert!(soft > 20, "fixture is not soft enough: {soft} soft entries");
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index form mirrors the EM equations
    fn full_em_stops_on_the_relative_loglik_test() {
        let micros = soft_fixture();
        let (k, seed) = (3, 17);
        let g: GmmFull<f64> = gmm_full_once(&micros, k, 600, seed);
        let (_, _, rmeans, _, rloglik, ran) = reference_full_em(&micros, k, 600, seed);
        assert!(ran > 4 && ran < 600, "expected convergence, ran {ran}");
        assert_close_rel(g.loglik, rloglik, 1e-6, "converged loglik");
        for c in 0..k {
            for d in 0..3 {
                assert_close_rel(g.means[c][d], rmeans[c][d], 1e-5, "converged mean");
            }
        }
        let mut prev = f64::NEG_INFINITY;
        for t in 1..=10 {
            let ll = gmm_full_once::<f64, _>(&micros, k, t, seed).loglik;
            assert!(
                ll >= prev - 1e-9,
                "loglik fell at iteration {t}: {prev} -> {ll}"
            );
            prev = ll;
        }
    }

    /// Both automatic-`k` wrappers minimise BIC over the same sweep the test can reproduce exactly.
    /// The end-to-end tests only assert that the recovered `k` equals the planted one, which stays
    /// right for a wide range of wrong penalties.
    #[test]
    fn auto_k_selects_the_argmin_of_an_independently_scored_bic() {
        // Separated blobs, so the likelihood stops improving once `k` reaches the planted count and
        // the *penalty* is the only thing deciding the rest of the sweep. The soft fixture is useless
        // here: its BIC rises monotonically, so `k = 1` wins whatever the parameter count says.
        let mut rng = SplitMix64::new(88);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, _) = blobs(&mut rng, 150, &centers, 0.6);
        let micros = grid_micros(&pts, 0.5).0;
        let d = micros[0].dim();
        let ntot: f64 = micros.iter().map(|f| f.weight()).sum();
        let (lo, hi) = (1usize, 5usize);

        let mut want_diag = lo;
        let mut want_full = lo;
        let (mut best_diag, mut best_full) = (f64::INFINITY, f64::INFINITY);
        for k in lo..=hi {
            let g: Gmm<f64> = gmm_diagonal_once(&micros, k, 80, 3);
            let s = -2.0 * g.loglik + (2 * k * d + (k - 1)) as f64 * ntot.ln();
            if s < best_diag {
                best_diag = s;
                want_diag = k;
            }
            let gf: GmmFull<f64> = gmm_full_once(&micros, k, 80, 3);
            let sf = -2.0 * gf.loglik + (k * d + k * d * (d + 1) / 2 + (k - 1)) as f64 * ntot.ln();
            if sf < best_full {
                best_full = sf;
                want_full = k;
            }
        }
        assert_eq!(
            gmm_diagonal_auto::<f64, _>(&micros, lo, hi, 80, 3)
                .means
                .len(),
            want_diag,
            "diagonal BIC"
        );
        assert_eq!(
            gmm_full_auto::<f64, _>(&micros, lo, hi, 80, 3).means.len(),
            want_full,
            "full BIC"
        );
    }
}
