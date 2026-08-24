//! Mixture of probabilistic PCA (Tipping & Bishop 1999) on leaf clustering features.
//!
//! Each component is a Gaussian whose covariance is constrained to `Σ_c = W_c W_cᵀ + σ_c² I` with
//! `W_c` of rank `q ≪ d`: a `q`-dimensional principal subspace plus isotropic noise. It sits between
//! the diagonal head (which cannot represent a rotation at all) and the full head (which pays `d²`
//! per component for the privilege).
//!
//! Nothing here forms a `d×d` matrix. Both identities the head rests on are exact and were verified
//! symbolically in `local/scratch/mppca_identities.mac` before a line of this was written, with
//! `M_c = σ_c² I_q + W_cᵀ W_c` (`q×q`):
//!
//! ```text
//! Σ_c⁻¹ = (1/σ_c²) [ I − W_c M_c⁻¹ W_cᵀ ]        |Σ_c| = σ_c^(2(d−q)) |M_c|
//! ```
//!
//! The expected-log E-step of the other mixture heads carries a within-leaf correction
//! `−½ tr(Σ_c⁻¹ Σ_i)`, and the same Woodbury form turns it into
//! `(tr Σ_i − tr(M_c⁻¹ W_cᵀ Σ_i W_c)) / σ_c²` — which an `FdSketch` leaf answers in `O(ℓ·d·q)` from
//! its sketch rows, against `O(ℓ·d²)` for the full-covariance head. That gap is the point of the
//! head: at `d = 784` and `max_leaves = 2000` the full head's per-leaf dense scatters are the
//! measured ~35 GB hazard, and here they never exist.
//!
//! With a leaf model whose `second_moment` is `Dense` the per-leaf storage is the dense one again —
//! only the *component* parameters shrink (`k·d·q` against `k·d²`). The head is built for
//! `feature="fd"`.
//!
//! What the head costs is measured, and it is a trade against *compression*, not against dimension.
//! The `−½ tr(Σ_c⁻¹ Σ_i)` correction is a sum of within-leaf scatters that are locally oriented and
//! carry almost none of the between-cluster orientation, so a coarser summary penalises a head in
//! proportion to how much orientation it models. On `digits` at one leaf per point this head beats
//! both the diagonal and the full one (ARI 0.600 / 0.461 / 0.575); at 6:1 compression the ordering
//! inverts exactly (0.406 / 0.493 / 0.273). `docs/USAGE.md` carries the table.
//!
//! One property of MPPCA-EM is worth stating because it is counter-intuitive: within an already
//! correct subspace the loading scale converges at rate `1 − 2σ²/λ_r + 2(σ²/λ_r)²`, so a component
//! whose isotropic noise is *small* next to its retained eigenvalues converges *slowly* — the rate
//! approaches 1 as `σ²/λ → 0`. `max_iter` therefore earns more here than in the diagonal or full
//! head, where the M-step is a closed-form re-estimate rather than a fixed-point iteration.

use crate::clustering::gmm::{best_of_restarts, bic, chol_regularized, total_weight};
use crate::clustering::kmeans::kmeans;
use crate::clustering::rng::SplitMix64;
use crate::feature::{ClusterFeature, SecondMoment};
use crate::mixture::Mixture;
use crate::types::Real;

/// EM restarts kept for the best log-likelihood, matching the other mixture heads.
const MPPCA_N_INIT: u64 = 4;

/// Subspace (orthogonal) iterations used to seed `W_c` from its k-means cluster's scatter. Two
/// passes plus the Rayleigh pass that follows are enough to start EM inside the principal subspace
/// rather than at a random one; more only buys accuracy EM is about to refine anyway.
const INIT_SUBSPACE_ITERS: usize = 2;

/// Keeps the subspace-init random stream apart from the k-means one drawn at the same seed.
const INIT_STREAM_OFFSET: u64 = 0x9E37_79B9_7F4A_7C15;

/// Result of an MPPCA-EM run over features.
pub struct Mppca<R: Real> {
    /// Hard label (argmax responsibility) per input feature.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[feature][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_c`.
    pub weights: Vec<R>,
    /// Component means `μ_c`.
    pub means: Vec<Vec<R>>,
    /// Loadings `[component][q][d]`: row `r` is column `r` of `W_c`, so `Σ_c = W_c W_cᵀ + σ_c² I`.
    pub loads: Vec<Vec<Vec<R>>>,
    /// Isotropic noise variance `σ_c²` per component.
    pub noise: Vec<R>,
    /// Weighted data log-likelihood at convergence.
    pub loglik: R,
    /// The fitted density, for scoring raw points.
    pub mixture: Mixture,
}

/// `out[r] += w · (Σ + δδᵀ) v_r` for every row of `v` — one leaf's contribution to `S V`, the only
/// product the subspace init and the M-step take of a scatter matrix.
fn accumulate_scatter_rows<R: Real>(
    v: &[Vec<R>],
    sig: &SecondMoment<R>,
    delta: &[R],
    w: R,
    out: &mut [Vec<R>],
) {
    sig.apply_rows(v, out, w);
    for (vr, o) in v.iter().zip(out.iter_mut()) {
        let dot = dot(vr, delta);
        let c = w * dot;
        if c != R::zero() {
            for (ov, &dv) in o.iter_mut().zip(delta) {
                *ov = *ov + c * dv;
            }
        }
    }
}

/// Cholesky of a `q×q` matrix this head builds to be positive definite (`M ⪰ σ² I`, and
/// `σ²M + G ⪰ σ⁴ I`), ridged only as a numerical fallback. The full head's `chol_regularized`
/// always adds its ridge, which is right for a sample covariance and wrong here: the M-step's fixed
/// point would be biased by it, and with `q = d − 1` the ML solution is supposed to be exact.
fn chol_spd<R: Real>(a: &[Vec<R>], scale: R) -> (Vec<Vec<R>>, R) {
    match crate::linalg::cholesky_lower(a) {
        Some(l) => {
            let ld = crate::linalg::logdet_from_chol(&l);
            (l, ld)
        }
        None => chol_regularized(a, scale, R::from_f64(1e-9).unwrap()),
    }
}

fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y)
        .fold(R::zero(), |p, q| p + q)
}

/// Gram-Schmidt with one re-orthogonalisation pass, in place. A row that collapses is left at zero:
/// it then carries no loading and no direction, which is the honest answer when the cluster's
/// scatter has lower rank than `q`.
fn orthonormalize<R: Real>(rows: &mut [Vec<R>]) {
    let tiny = R::from_f64(1e-150).unwrap();
    for i in 0..rows.len() {
        for _ in 0..2 {
            for j in 0..i {
                let p = dot(&rows[i], &rows[j]);
                if p != R::zero() {
                    for d in 0..rows[i].len() {
                        rows[i][d] = rows[i][d] - p * rows[j][d];
                    }
                }
            }
        }
        let norm = dot(&rows[i], &rows[i]).sqrt();
        if norm > tiny {
            for v in rows[i].iter_mut() {
                *v = *v / norm;
            }
        } else {
            rows[i].iter_mut().for_each(|v| *v = R::zero());
        }
    }
}

/// Fit a `k`-component MPPCA with subspace rank `q`, warm-started from k-means and a per-cluster
/// subspace iteration.
#[allow(clippy::needless_range_loop)] // component/rank/dimension indices read clearest explicitly
fn mppca_once<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> Mppca<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let dim = features[0].dim();
    let m = features.len();
    // `q = d` would leave no isotropic residual for σ² to explain, so the rank is capped one below
    // the dimension; `q = 0` is the spherical rung and stays reachable.
    let q = rank.min(dim.saturating_sub(1));
    let mu: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let n: Vec<R> = features.iter().map(|f| f.weight()).collect();
    let sig: Vec<SecondMoment<R>> = features.iter().map(|f| f.second_moment()).collect();
    let tr_sig: Vec<R> = sig.iter().map(|s| s.trace()).collect();

    let half = R::from_f64(0.5).unwrap();
    let log_two_pi = R::from_f64(std::f64::consts::TAU).unwrap().ln();
    let dimr = R::from_usize(dim).unwrap();

    // Mean global variance, matrix-free: the scale every floor in this head is expressed against.
    let ntot: R = n.iter().copied().fold(R::zero(), |a, b| a + b);
    let scale = {
        let mut centre = vec![R::zero(); dim];
        if ntot > R::zero() {
            for (mi, &ni) in mu.iter().zip(&n) {
                for (cv, &v) in centre.iter_mut().zip(mi) {
                    *cv = *cv + ni * v;
                }
            }
            for cv in &mut centre {
                *cv = *cv / ntot;
            }
        }
        let mut total = R::zero();
        for i in 0..m {
            let spread = mu[i]
                .iter()
                .zip(&centre)
                .map(|(&a, &b)| (a - b) * (a - b))
                .fold(R::zero(), |p, s| p + s);
            total = total + n[i] * (tr_sig[i] + spread);
        }
        if ntot > R::zero() {
            (total / (ntot * dimr)).max(R::from_f64(1e-12).unwrap())
        } else {
            R::one()
        }
    };
    let noise_floor = R::from_f64(1e-6).unwrap() * scale;
    // Without a floor on the loading scale, a cluster whose leading eigenvalue does not exceed σ²
    // gets `W_c = 0` — and `W = 0` is a fixed point of the M-step (`SW = 0 ⇒ W_new = 0`), so the
    // component would be frozen spherical for the rest of the run.
    let load_floor = R::from_f64(1e-6).unwrap() * scale;

    let km = kmeans(features, k, 50, 1, seed);
    let mut means = km.centers;
    let mut loads = vec![vec![vec![R::zero(); dim]; q]; k];
    let mut noise = vec![scale; k];
    {
        let mut rng = SplitMix64::new(seed ^ INIT_STREAM_OFFSET);
        for c in 0..k {
            let mut v: Vec<Vec<R>> = (0..q)
                .map(|_| {
                    (0..dim)
                        .map(|_| R::from_f64(rng.gauss()).unwrap())
                        .collect()
                })
                .collect();
            orthonormalize(&mut v);
            let mut nk = R::zero();
            let mut tr_s = R::zero();
            for i in 0..m {
                if km.labels[i] != c {
                    continue;
                }
                nk = nk + n[i];
                let spread = mu[i]
                    .iter()
                    .zip(&means[c])
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(R::zero(), |p, s| p + s);
                tr_s = tr_s + n[i] * (tr_sig[i] + spread);
            }
            if nk <= R::zero() {
                for (r, row) in loads[c].iter_mut().enumerate() {
                    let s = load_floor.sqrt();
                    for (lv, &vv) in row.iter_mut().zip(&v[r]) {
                        *lv = s * vv;
                    }
                }
                continue;
            }
            tr_s = tr_s / nk;
            let mut y = vec![vec![R::zero(); dim]; q];
            for pass in 0..=INIT_SUBSPACE_ITERS {
                y.iter_mut().for_each(|row| row.fill(R::zero()));
                for i in 0..m {
                    if km.labels[i] != c {
                        continue;
                    }
                    let delta: Vec<R> = mu[i].iter().zip(&means[c]).map(|(&a, &b)| a - b).collect();
                    accumulate_scatter_rows(&v, &sig[i], &delta, n[i] / nk, &mut y);
                }
                if pass < INIT_SUBSPACE_ITERS {
                    orthonormalize(&mut y);
                    v = std::mem::replace(&mut y, vec![vec![R::zero(); dim]; q]);
                }
            }
            // `y[r] = S_c v_r` for the converged `v`, so the Rayleigh quotient is one more dot.
            let lam: Vec<R> = (0..q).map(|r| dot(&v[r], &y[r]).max(R::zero())).collect();
            let kept: R = lam.iter().copied().fold(R::zero(), |a, b| a + b);
            let rest = dimr - R::from_usize(q).unwrap();
            noise[c] = if rest > R::zero() {
                ((tr_s - kept) / rest).max(noise_floor)
            } else {
                noise_floor
            };
            for r in 0..q {
                let s = (lam[r] - noise[c]).max(load_floor).sqrt();
                for (lv, &vv) in loads[c][r].iter_mut().zip(&v[r]) {
                    *lv = s * vv;
                }
            }
        }
    }
    let mut weights = vec![R::one() / R::from_usize(k).unwrap(); k];

    let mut resp = vec![vec![R::zero(); k]; m];
    let mut loglik = R::neg_infinity();
    let tol = R::from_f64(1e-7).unwrap();

    for it in 0..max_iter {
        // Per-component constants: the `q×q` factor of `M_c`, its inverse, and `log|Σ_c|`.
        let mut m_chol = Vec::with_capacity(k);
        let mut m_inv = Vec::with_capacity(k);
        let mut logdet = vec![R::zero(); k];
        let mut inv_noise = vec![R::zero(); k];
        for c in 0..k {
            let mut mm = vec![vec![R::zero(); q]; q];
            for a in 0..q {
                for b in 0..=a {
                    let v = dot(&loads[c][a], &loads[c][b]);
                    mm[a][b] = v;
                    mm[b][a] = v;
                }
                mm[a][a] = mm[a][a] + noise[c];
            }
            let (l, ld) = chol_spd(&mm, scale);
            logdet[c] = (dimr - R::from_usize(q).unwrap()) * noise[c].ln() + ld;
            m_inv.push(crate::linalg::inv_from_chol(&l));
            m_chol.push(l);
            inv_noise[c] = R::one() / noise[c];
        }

        // E-step, fused with everything in the M-step that does not depend on the new means:
        // `Σ_i w_ic Σ_i W_c` is mean-free, and it is the only term that costs more than `O(q·d)`.
        let mut nk = vec![R::zero(); k];
        let mut mean_sum = vec![vec![R::zero(); dim]; k];
        let mut sw = vec![vec![vec![R::zero(); dim]; q]; k];
        let mut tr_s = vec![R::zero(); k];
        let mut new_ll = R::zero();
        let mut logr = vec![R::zero(); k];
        let mut sig_w = vec![vec![R::zero(); dim]; q];
        let mut per_leaf = vec![vec![vec![R::zero(); dim]; q]; k];
        for i in 0..m {
            for c in 0..k {
                let delta: Vec<R> = mu[i].iter().zip(&means[c]).map(|(&a, &b)| a - b).collect();
                let iso = dot(&delta, &delta);
                let p: Vec<R> = loads[c].iter().map(|w| dot(w, &delta)).collect();
                let quad = (iso - crate::linalg::mahalanobis_sq_from_chol(&m_chol[c], &p))
                    .max(R::zero())
                    * inv_noise[c];
                sig_w.iter_mut().for_each(|row| row.fill(R::zero()));
                sig[i].apply_rows(&loads[c], &mut sig_w, R::one());
                // `tr(M⁻¹ WᵀΣ_i W)`, with `(WᵀΣ_i W)[a][b] = w_a · (Σ_i w_b)`.
                let mut folded = R::zero();
                for a in 0..q {
                    for b in 0..q {
                        folded = folded + m_inv[c][a][b] * dot(&loads[c][b], &sig_w[a]);
                    }
                }
                let trace = (tr_sig[i] - folded).max(R::zero()) * inv_noise[c];
                logr[c] =
                    weights[c].ln() - half * (dimr * log_two_pi + logdet[c] + quad) - half * trace;
                per_leaf[c].clone_from(&sig_w);
            }
            let mx = logr.iter().copied().fold(R::neg_infinity(), R::max);
            let mut s = R::zero();
            for &lr in &logr {
                s = s + (lr - mx).exp();
            }
            let lse = mx + s.ln();
            new_ll = new_ll + n[i] * lse;
            for c in 0..k {
                let r = (logr[c] - lse).exp();
                resp[i][c] = r;
                let w = n[i] * r;
                nk[c] = nk[c] + w;
                for (ms, &v) in mean_sum[c].iter_mut().zip(&mu[i]) {
                    *ms = *ms + w * v;
                }
                tr_s[c] = tr_s[c] + w * tr_sig[i];
                for (dst, src) in sw[c].iter_mut().zip(&per_leaf[c]) {
                    for (a, &b) in dst.iter_mut().zip(src) {
                        *a = *a + w * b;
                    }
                }
            }
        }

        let wtot: R = nk.iter().copied().fold(R::zero(), |a, b| a + b);
        let mut new_means = means.clone();
        for c in 0..k {
            if nk[c] > R::zero() {
                weights[c] = nk[c] / wtot;
                for (nm, &s) in new_means[c].iter_mut().zip(&mean_sum[c]) {
                    *nm = s / nk[c];
                }
            } else {
                // An emptied component keeps its parameters rather than collapsing onto the origin;
                // `ln π_c` floors it out of the posterior either way.
                weights[c] = R::zero();
            }
        }
        // The between-leaf term of `S_c`, which needs the new means. `δ (δᵀW)` is `O(q·d)`, so this
        // second pass costs a fraction of the E-step rather than repeating it.
        for i in 0..m {
            for c in 0..k {
                let w = n[i] * resp[i][c];
                if w <= R::zero() {
                    continue;
                }
                let delta: Vec<R> = mu[i]
                    .iter()
                    .zip(&new_means[c])
                    .map(|(&a, &b)| a - b)
                    .collect();
                tr_s[c] = tr_s[c] + w * dot(&delta, &delta);
                for (row, wr) in sw[c].iter_mut().zip(&loads[c]) {
                    let coef = w * dot(wr, &delta);
                    if coef != R::zero() {
                        for (a, &b) in row.iter_mut().zip(&delta) {
                            *a = *a + coef * b;
                        }
                    }
                }
            }
        }

        let mut new_loads = loads.clone();
        let mut new_noise = noise.clone();
        for c in 0..k {
            if nk[c] <= R::zero() {
                continue;
            }
            for row in sw[c].iter_mut() {
                for v in row.iter_mut() {
                    *v = *v / nk[c];
                }
            }
            let trace_s = tr_s[c] / nk[c];
            if q > 0 {
                // `W_new = SW (σ²I + M⁻¹ G)⁻¹ = SW (σ²M + G)⁻¹ M` with `G = Wᵀ SW`. The right-hand
                // form is symmetric positive definite, so it factors — the left-hand one does not.
                let mut kmat = vec![vec![R::zero(); q]; q];
                for a in 0..q {
                    for b in 0..=a {
                        let g = dot(&loads[c][a], &sw[c][b]);
                        let mab = if a == b {
                            noise[c] + dot(&loads[c][a], &loads[c][b])
                        } else {
                            dot(&loads[c][a], &loads[c][b])
                        };
                        let v = noise[c] * mab + g;
                        kmat[a][b] = v;
                        kmat[b][a] = v;
                    }
                }
                let (kl, _) = chol_spd(&kmat, scale);
                // `z = K⁻¹ M`, solved column by column through the factor.
                let mut z = vec![vec![R::zero(); q]; q];
                for j in 0..q {
                    let col: Vec<R> = (0..q)
                        .map(|a| {
                            let base = dot(&loads[c][a], &loads[c][j]);
                            if a == j { base + noise[c] } else { base }
                        })
                        .collect();
                    let y = crate::linalg::solve_lower(&kl, &col);
                    let x = crate::linalg::solve_upper_t(&kl, &y);
                    for a in 0..q {
                        z[a][j] = x[a];
                    }
                }
                for r in 0..q {
                    for j in 0..dim {
                        let mut acc = R::zero();
                        for a in 0..q {
                            acc = acc + sw[c][a][j] * z[a][r];
                        }
                        new_loads[c][r][j] = acc;
                    }
                }
                // `σ²_new = (tr S − tr(M⁻¹ W_newᵀ S W)) / d`, with the *old* `M`.
                let mut folded = R::zero();
                for a in 0..q {
                    for b in 0..q {
                        folded = folded + m_inv[c][a][b] * dot(&new_loads[c][b], &sw[c][a]);
                    }
                }
                new_noise[c] = ((trace_s - folded) / dimr).max(noise_floor);
            } else {
                new_noise[c] = (trace_s / dimr).max(noise_floor);
            }
        }
        means = new_means;
        loads = new_loads;
        noise = new_noise;

        if it > 0 && (new_ll - loglik).abs() <= tol * loglik.abs().max(R::one()) {
            loglik = new_ll;
            break;
        }
        loglik = new_ll;
    }

    let labels = resp.iter().map(|r| argmax(r)).collect();
    let mixture = Mixture::low_rank(&weights, &means, &loads, &noise);
    Mppca {
        labels,
        resp,
        weights,
        means,
        loads,
        noise,
        loglik,
        mixture,
    }
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

/// Fit a `k`-component MPPCA of subspace rank `rank`, keeping the best of [`MPPCA_N_INIT`] EM
/// restarts by log-likelihood.
pub fn mppca<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> Mppca<R> {
    best_of_restarts(
        MPPCA_N_INIT,
        seed,
        |g: &Mppca<R>| g.loglik,
        |s| mppca_once(features, k, rank, max_iter, s),
    )
}

/// MPPCA with automatic component count (BIC over `k ∈ [k_min, k_max]`). A component costs
/// `d` mean parameters, one noise variance and `d·q − q(q−1)/2` free loadings — the Stiefel
/// rotation of `W_c` is unidentifiable, so the `q(q−1)/2` of it are not counted.
pub fn mppca_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> Mppca<R> {
    let d = features[0].dim();
    let q = rank.min(d.saturating_sub(1));
    let ntot = total_weight(features);
    let k_hi = k_max.min(features.len()).max(1);
    let k_lo = k_min.max(1).min(k_hi);
    let mut best_score = R::infinity();
    let mut best: Option<Mppca<R>> = None;
    for k in k_lo..=k_hi {
        let g = mppca_once(features, k, rank, max_iter, seed);
        let p = k * (d + 1 + d * q - q * q.saturating_sub(1) / 2) + (k - 1);
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
    use crate::clustering::gmm::gmm_diagonal;
    use crate::clustering::testutil::ari;
    use crate::feature::{FdSketch, Full, Spherical};

    /// Pooled scatter of a set of features, the dense way — the reference the matrix-free M-step is
    /// checked against.
    fn pooled_scatter<C: ClusterFeature<f64>>(features: &[C], dim: usize) -> Vec<Vec<f64>> {
        let wtot: f64 = features.iter().map(|f| f.weight()).sum();
        let mut centre = vec![0.0; dim];
        for f in features {
            for (c, &v) in centre.iter_mut().zip(f.mean()) {
                *c += f.weight() * v;
            }
        }
        for c in &mut centre {
            *c /= wtot;
        }
        let mut s = vec![vec![0.0; dim]; dim];
        for f in features {
            let w = f.weight();
            let cov = f.cov_dense();
            let delta: Vec<f64> = f.mean().iter().zip(&centre).map(|(&a, &b)| a - b).collect();
            for a in 0..dim {
                for b in 0..dim {
                    s[a][b] += w * (cov[a][b] + delta[a] * delta[b]);
                }
            }
        }
        for row in &mut s {
            for v in row.iter_mut() {
                *v /= wtot;
            }
        }
        s
    }

    /// Points from a Gaussian whose covariance is `A Aᵀ`, chopped into `leaves` features.
    fn correlated_leaves(
        seed: u64,
        a: &[[f64; 3]; 3],
        leaves: usize,
        per: usize,
    ) -> Vec<Full<f64>> {
        let mut rng = SplitMix64::new(seed);
        (0..leaves)
            .map(|_| {
                let mut f = Full::new(3);
                for _ in 0..per {
                    let z = [rng.gauss(), rng.gauss(), rng.gauss()];
                    let p: Vec<f64> = (0..3)
                        .map(|i| (0..3).map(|j| a[i][j] * z[j]).sum())
                        .collect();
                    f.push(&p, 1.0);
                }
                f
            })
            .collect()
    }

    /// With `q = d − 1` the PPCA maximum-likelihood solution is exact: `σ²` is the smallest
    /// eigenvalue of the scatter and `W Wᵀ + σ² I` reproduces the scatter itself. That makes the
    /// whole M-step chain — `SW`, `G`, the `(σ²M + G)⁻¹M` rearrangement, and the `σ²` update —
    /// checkable against a number computed a completely different way.
    #[test]
    fn full_rank_loadings_reproduce_the_scatter_they_were_fitted_to() {
        // `H diag(√2, 1, √0.5)` for the Householder `H = I − 2vvᵀ`, `v = (1,1,1)/√3`, so the scatter
        // has spectrum {2, 1, 0.5} in a basis aligned with no axis. The spectrum is chosen for the
        // EM *rate*: the within-subspace scale converges at `1 − 2σ²/λ_r + 2(σ²/λ_r)²`, which is
        // slowest — arbitrarily close to 1 — exactly when the noise is small next to the retained
        // eigenvalues. Here `σ²/λ` is 1/4 and 1/2, and 20 iterations suffice.
        let r2 = std::f64::consts::SQRT_2;
        let a = [
            [r2 / 3.0, -2.0 / 3.0, -r2 / 3.0],
            [-2.0 * r2 / 3.0, 1.0 / 3.0, -r2 / 3.0],
            [-2.0 * r2 / 3.0, -2.0 / 3.0, r2 / 6.0],
        ];
        let feats = correlated_leaves(5, &a, 40, 60);
        let fit = mppca_once::<f64, _>(&feats, 1, 2, 400, 3);
        let want = pooled_scatter(&feats, 3);
        let mut got = vec![vec![0.0; 3]; 3];
        for row in &fit.loads[0] {
            for i in 0..3 {
                for j in 0..3 {
                    got[i][j] += row[i] * row[j];
                }
            }
        }
        for (i, row) in got.iter_mut().enumerate() {
            row[i] += fit.noise[0];
        }
        for i in 0..3 {
            for j in 0..3 {
                // The residual is the log-likelihood stopping rule, not the identity: EM halts at a
                // relative `Δll` of 1e-7, which leaves ~2e-4 here. Anything at 1e-3 or above is an
                // algebra error rather than an unconverged run.
                assert!(
                    (got[i][j] - want[i][j]).abs() < 1e-3,
                    "({i},{j}): {} vs {}",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }

    /// Two clusters that share a centre and differ only in the *direction* they are stretched along.
    /// Every per-dimension variance is identical between them, so a diagonal covariance carries no
    /// signal at all; a rank-1 subspace carries all of it.
    fn crossed_leaves(
        seed: u64,
        per_cluster: usize,
        per_leaf: usize,
    ) -> (Vec<Full<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let axes = [
            [
                1.0 / std::f64::consts::SQRT_2,
                1.0 / std::f64::consts::SQRT_2,
            ],
            [
                1.0 / std::f64::consts::SQRT_2,
                -1.0 / std::f64::consts::SQRT_2,
            ],
        ];
        let mut feats = Vec::new();
        let mut truth = Vec::new();
        for (c, ax) in axes.iter().enumerate() {
            let perp = [-ax[1], ax[0]];
            for _ in 0..per_cluster {
                let mut f = Full::new(2);
                for _ in 0..per_leaf {
                    let t = 3.0 * rng.gauss();
                    let s = 0.08 * rng.gauss();
                    f.push(&[t * ax[0] + s * perp[0], t * ax[1] + s * perp[1]], 1.0);
                }
                feats.push(f);
                truth.push(c);
            }
        }
        (feats, truth)
    }

    #[test]
    fn a_rank_one_subspace_separates_what_a_diagonal_covariance_cannot() {
        let (feats, truth) = crossed_leaves(11, 30, 40);
        let sub = ari(&mppca(&feats, 2, 1, 200, 7).labels, &truth);
        let diag = ari(&gmm_diagonal(&feats, 2, 200, 7).labels, &truth);
        assert!(sub > 0.9, "mppca ARI = {sub}");
        assert!(
            diag < 0.5,
            "diagonal ARI = {diag}, fixture is not discriminating"
        );
    }

    /// The same fixture through `FdSketch` leaves, which is the leaf model the head exists for: its
    /// `second_moment` is `LowRank`, so `trace` and `apply_rows` take their sketch-row paths.
    #[test]
    fn the_low_rank_leaf_path_recovers_the_same_partition() {
        let (dense, truth) = crossed_leaves(13, 30, 40);
        let sketched: Vec<FdSketch<f64>> = dense
            .iter()
            .map(|f| {
                let mut s = FdSketch::with_ell(2, 2);
                let cov = f.cov_dense();
                // Reconstruct the leaf's points is not possible from a CF, so feed the sketch a
                // two-point cloud with the same mean and second moment along each axis.
                let sd = [cov[0][0].sqrt(), cov[1][1].sqrt()];
                let mu = f.mean().to_vec();
                let w = f.weight() / 2.0;
                s.push(&[mu[0] + sd[0], mu[1] + sd[1]], w);
                s.push(&[mu[0] - sd[0], mu[1] - sd[1]], w);
                s
            })
            .collect();
        let got = ari(&mppca(&sketched, 2, 1, 200, 7).labels, &truth);
        assert!(got > 0.9, "FD-leaf ARI = {got}");
    }

    /// `rank = 0` drops the subspace entirely: every component is `σ_c² I`, and the fitted `σ_c²`
    /// must be the component's own mean variance rather than an arbitrary floor.
    #[test]
    fn rank_zero_is_a_spherical_mixture() {
        let mut rng = SplitMix64::new(17);
        let centres = [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 8.0]];
        let sigma = 0.6;
        let mut feats: Vec<Spherical<f64>> = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in centres.iter().enumerate() {
            for _ in 0..12 {
                let mut f = Spherical::new(3);
                for _ in 0..40 {
                    let p: Vec<f64> = ctr.iter().map(|&v| v + sigma * rng.gauss()).collect();
                    f.push(&p, 1.0);
                }
                feats.push(f);
                truth.push(c);
            }
        }
        let fit = mppca(&feats, 3, 0, 200, 5);
        assert!(ari(&fit.labels, &truth) > 0.99);
        for rows in &fit.loads {
            assert!(rows.is_empty(), "rank 0 must carry no loadings");
        }
        for &s2 in &fit.noise {
            assert!(
                (s2 - sigma * sigma).abs() < 0.15 * sigma * sigma,
                "sigma^2 = {s2}, true {}",
                sigma * sigma
            );
        }
    }

    #[test]
    fn auto_k_selects_the_component_count() {
        let mut rng = SplitMix64::new(23);
        let centres = [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]];
        let mut feats: Vec<Spherical<f64>> = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in centres.iter().enumerate() {
            for _ in 0..10 {
                let mut f = Spherical::new(3);
                for _ in 0..40 {
                    let p: Vec<f64> = ctr.iter().map(|&v| v + 0.7 * rng.gauss()).collect();
                    f.push(&p, 1.0);
                }
                feats.push(f);
                truth.push(c);
            }
        }
        let fit = mppca_auto(&feats, 1, 6, 1, 200, 5);
        assert_eq!(fit.means.len(), 3, "selected k = {}", fit.means.len());
        assert!(ari(&fit.labels, &truth) > 0.95);
    }

    /// The head's own point rule must be the mixture it fitted: a raw point drawn inside a
    /// component's subspace has to score highest under that component.
    #[test]
    fn the_fitted_mixture_scores_raw_points_by_the_subspace_it_learned() {
        let (feats, truth) = crossed_leaves(29, 30, 40);
        let fit = mppca(&feats, 2, 1, 200, 7);
        assert!(ari(&fit.labels, &truth) > 0.9);
        let along_first = fit.labels[0];
        let along_second = *fit.labels.last().unwrap();
        assert_ne!(along_first, along_second);
        let s = 1.0 / std::f64::consts::SQRT_2;
        assert_eq!(fit.mixture.assign(&[5.0 * s, 5.0 * s]), along_first);
        assert_eq!(fit.mixture.assign(&[5.0 * s, -5.0 * s]), along_second);
    }

    /// EM's defining guarantee, and the sharpest check on the M-step that does not depend on it
    /// converging: the observed-data log-likelihood may never decrease. A transposed factor or a
    /// stale `M` in the `σ²` update breaks this within a handful of iterations.
    #[test]
    fn every_em_iteration_increases_the_log_likelihood() {
        let (feats, _) = crossed_leaves(37, 20, 30);
        let mut prev = f64::NEG_INFINITY;
        for iters in 1..=25 {
            let ll = mppca_once::<f64, _>(&feats, 3, 1, iters, 9).loglik;
            assert!(
                ll >= prev - 1e-9 * prev.abs().max(1.0),
                "iteration {iters}: {ll} < {prev}"
            );
            prev = ll;
        }
    }
}
