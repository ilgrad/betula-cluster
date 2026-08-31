//! Mixture of factor analysers (Ghahramani & Hinton 1996) on leaf clustering features.
//!
//! One constraint away from [`mppca`](super::mppca): the noise a component cannot explain with its
//! `q`-dimensional subspace is **per dimension**, `Σ_c = W_c W_cᵀ + Ψ_c` with `Ψ_c = diag(ψ_c)`,
//! rather than the single `σ_c² I` MPPCA allows. That one relaxation is what makes the head
//! scale-aware: MPPCA's isotropic residual is only the right model when every coordinate has been
//! standardised to the same units, and a feature table where one column is a price and another a
//! count is exactly the case where it is not. The cost is `d − 1` extra parameters per component,
//! against `d²/2` for the full head.
//!
//! It is also the one difference that removes the head's rotation equivariance. MPPCA's `σ² I` is
//! invariant under `x ↦ Qx`, so its labels are; `diag(ψ)` singles out the coordinate axes, exactly
//! as the diagonal Gaussian head does. `tests/equivariance.rs` states it as a property.
//!
//! No `d×d` matrix is formed. Every step routes through the `q×q`
//!
//! ```text
//! M_c = I_q + W_cᵀ Ψ_c⁻¹ W_c,     U_c = Ψ_c⁻¹ W_c   (d×q)
//! ```
//!
//! and the identities below, all verified symbolically with the `ψ_j` left as free symbols in
//! `local/scratch/mfa_identities.mac` and `local/scratch/mfa_mstep_collapse.mac` before any of this
//! was written — residual exactly zero at both shapes checked:
//!
//! ```text
//! Σ_c⁻¹     = Ψ_c⁻¹ − U_c M_c⁻¹ U_cᵀ          log|Σ_c| = Σ_j ln ψ_cj + log|M_c|
//! β_c       = W_cᵀ Σ_c⁻¹ = M_c⁻¹ U_cᵀ
//! W_new     = (S U) (M + G)⁻¹ M               G = Uᵀ S U
//! ψ_new[j]  = S[j][j] − Σ_a W_new[a][j] (M⁻¹ (S U)ᵀ)[a][j]
//! ```
//!
//! The last two are Ghahramani & Hinton's M-step rewritten so that the weighted scatter `S_c` enters
//! **only** through the `d×q` product `S_c U_c` and its diagonal — both of which a leaf answers
//! directly, the first by [`SecondMoment::apply_rows`] and the second by
//! [`SecondMoment::add_diagonal_scaled`]. Their published form needs `S βᵀ (I − βW + βSβᵀ)⁻¹`; the
//! inner matrix collapses to `M⁻¹(M + G)M⁻¹`, and `M + G` is symmetric positive definite (`M ⪰ I`,
//! `G ⪰ 0`), so it factors by Cholesky where the assembled form would need a general inverse.
//!
//! The expected-log E-step carries the same within-leaf correction the other mixture heads do,
//! `−½ tr(Σ_c⁻¹ Σ_i)`, and Woodbury turns it into
//! `Σ_j Σ_i[j][j]/ψ_cj − tr(M_c⁻¹ U_cᵀ Σ_i U_c)` — `O(ℓ·d·q)` from an `FdSketch` leaf's rows,
//! against `O(ℓ·d²)` for the full-covariance head.
//!
//! **Where it wins over `mppca`, and where it does not — measured, and the answer is narrow.** The
//! extra parameters buy nothing when the residual really is isotropic, and they cost variance:
//! `d − 1` more numbers per component estimated from the same data. The two heads dissociate in both
//! directions on fixtures built to separate them — `mfa` reads a quiet axis a loud nuisance pair
//! drowns (ARI 1.00 against `mppca`'s 0.04–0.34), `mppca` reads three lines that differ only in
//! *orientation* (1.00 against `mfa`'s ≈ 0.00), because there `ψ` absorbs an elongation that
//! belonged in `W`. On real tables, median of seeds 0/1/2 at `rank=2`, `mfa` is **behind `mppca` at
//! full leaf resolution everywhere it was tried**: `digits` 0.562 vs 0.738, `MNIST`-20k 0.277 vs
//! 0.365, `covtype`-20k in raw units 0.062 vs 0.087. The one real row where the relaxation pays is
//! the one it was built for and in the direction of *safety* rather than of a win — on standardised
//! `covtype` `mppca`'s isotropic residual costs it **less than half** the diagonal head's score
//! (0.030 against 0.077) while `mfa` degrades onto the diagonal head it contains at `rank = 0`
//! rather than below it. Cost is a wash: 30.9 s against `mppca`'s 37.6 s on MNIST at `rank=2`.
//! `docs/USAGE.md` carries the tables. This is not a strictly better `mppca`, and on an already
//! standardised table it is usually not a better `gmm` either.

use crate::clustering::gmm::{best_of_restarts, bic, chol_regularized, total_weight};
use crate::clustering::kmeans::kmeans;
use crate::clustering::rng::SplitMix64;
use crate::feature::{ClusterFeature, SecondMoment};
use crate::mixture::Mixture;
use crate::types::Real;

/// EM restarts kept for the best log-likelihood, matching the other mixture heads.
const MFA_N_INIT: u64 = 4;

/// Subspace iterations used to seed `W_c` from its k-means cluster's scatter, as in `mppca`.
const INIT_SUBSPACE_ITERS: usize = 2;

/// Keeps the subspace-init random stream apart from the k-means one drawn at the same seed.
const INIT_STREAM_OFFSET: u64 = 0x2545_F491_4F6C_DD1D;

/// Result of an MFA-EM run over features.
pub struct Mfa<R: Real> {
    /// Hard label (argmax responsibility) per input feature.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[feature][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_c`.
    pub weights: Vec<R>,
    /// Component means `μ_c`.
    pub means: Vec<Vec<R>>,
    /// Loadings `[component][q][d]`: row `r` is column `r` of `W_c`, so `Σ_c = W_c W_cᵀ + diag(ψ_c)`.
    pub loads: Vec<Vec<Vec<R>>>,
    /// Per-dimension noise variance `ψ_c` per component — the one thing this head has that
    /// [`Mppca`](super::mppca::Mppca) does not.
    pub noise: Vec<Vec<R>>,
    /// Weighted data log-likelihood at convergence.
    pub loglik: R,
    /// The fitted density, for scoring raw points.
    pub mixture: Mixture,
}

fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y)
        .fold(R::zero(), |p, q| p + q)
}

/// Cholesky of a `q×q` matrix this head builds to be positive definite (`M ⪰ I` and `M + G ⪰ I`),
/// ridged only as a numerical fallback — the same reasoning as `mppca`'s: a ridge that is always
/// applied biases the M-step's fixed point.
fn chol_spd<R: Real>(a: &[Vec<R>], scale: R) -> (Vec<Vec<R>>, R) {
    match crate::linalg::cholesky_lower(a) {
        Some(l) => {
            let ld = crate::linalg::logdet_from_chol(&l);
            (l, ld)
        }
        None => chol_regularized(a, scale, R::from_f64(1e-9).unwrap()),
    }
}

/// Gram-Schmidt with one re-orthogonalisation pass, in place; a collapsed row is left at zero.
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

/// `out[r] += w · (Σ + δδᵀ) v_r` for every row of `v` — one leaf's contribution to `S V`.
fn accumulate_scatter_rows<R: Real>(
    v: &[Vec<R>],
    sig: &SecondMoment<R>,
    delta: &[R],
    w: R,
    out: &mut [Vec<R>],
) {
    sig.apply_rows(v, out, w);
    for (vr, o) in v.iter().zip(out.iter_mut()) {
        let c = w * dot(vr, delta);
        if c != R::zero() {
            for (ov, &dv) in o.iter_mut().zip(delta) {
                *ov = *ov + c * dv;
            }
        }
    }
}

/// Fit a `k`-component MFA with `q` factors, warm-started from k-means and a per-cluster subspace
/// iteration.
#[allow(clippy::needless_range_loop)] // component/factor/dimension indices read clearest explicitly
fn mfa_once<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> Mfa<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let dim = features[0].dim();
    let m = features.len();
    // `q = d` leaves no residual for `Ψ` to explain; `q = 0` is the diagonal-Gaussian rung and stays
    // reachable, which is the honest lower end of this ladder.
    let q = rank.min(dim.saturating_sub(1));
    let mu: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let n: Vec<R> = features.iter().map(|f| f.weight()).collect();
    let sig: Vec<SecondMoment<R>> = features.iter().map(|f| f.second_moment()).collect();
    // The per-leaf diagonal, read once. Every component's E-step needs `Σ_j Σ_i[j][j]/ψ_cj`, and on
    // an `FdSketch` leaf recovering it costs `O(ℓ·d)` — paid `m` times here rather than `m·k` times
    // inside the loop.
    let diag_sig: Vec<Vec<R>> = sig
        .iter()
        .map(|s| {
            let mut v = vec![R::zero(); dim];
            s.add_diagonal_scaled(&mut v, R::one());
            v
        })
        .collect();
    let tr_sig: Vec<R> = diag_sig
        .iter()
        .map(|d| d.iter().copied().fold(R::zero(), |a, b| a + b))
        .collect();

    let half = R::from_f64(0.5).unwrap();
    let log_two_pi = R::from_f64(std::f64::consts::TAU).unwrap().ln();
    let dimr = R::from_usize(dim).unwrap();

    // Mean global variance, matrix-free: the scale every floor here is expressed against.
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
    // Without a floor on the loading scale a component whose leading eigenvalue does not clear its
    // noise gets `W_c = 0`, and `W = 0` is a fixed point of the M-step, so it would stay diagonal
    // for the rest of the run.
    let load_floor = R::from_f64(1e-6).unwrap() * scale;

    let km = kmeans(features, k, 50, 1, seed);
    let mut means = km.centers;
    let mut loads = vec![vec![vec![R::zero(); dim]; q]; k];
    let mut noise = vec![vec![scale; dim]; k];
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
            // Per-dimension cluster variance, which is where the diagonal init differs from
            // `mppca`'s scalar one: `Ψ` starts at the residual the subspace has not taken, per axis.
            let mut nk = R::zero();
            let mut var = vec![R::zero(); dim];
            for i in 0..m {
                if km.labels[i] != c {
                    continue;
                }
                nk = nk + n[i];
                for j in 0..dim {
                    let d = mu[i][j] - means[c][j];
                    var[j] = var[j] + n[i] * (diag_sig[i][j] + d * d);
                }
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
            var.iter_mut().for_each(|x| *x = *x / nk);
            // **The subspace iteration runs on the standardised scatter, not the raw one.** That is
            // the one place a factor analyser must not copy MPPCA. Under a shared `σ²` the leading
            // principal direction is also the leading factor, so MPPCA can seed `W` from the raw
            // scatter; under `diag(ψ)` any *axis-aligned* variance is free — `ψ_j` absorbs it at no
            // likelihood cost — so the raw top eigenvector is the direction `W` is least needed for.
            // Seeded that way the head measurably lands on a stationary point where `W` sits on the
            // loudest coordinate and explains nothing: on a 6-D blob whose only real structure is a
            // correlation between two quiet axes, it reached log-likelihood −5974.9 against the
            // diagonal Gaussian head's −5976.7, i.e. the factor bought 2 nats out of the 760 the
            // full head finds. Standardising by `√var` first makes the eigenvalue scale
            // *correlation*, where an uncorrelated axis sits at 1 however loud it is.
            let inv_sd: Vec<R> = var
                .iter()
                .map(|&x| R::one() / x.max(noise_floor).sqrt())
                .collect();
            let mut y = vec![vec![R::zero(); dim]; q];
            let mut scaled = vec![vec![R::zero(); dim]; q];
            for pass in 0..=INIT_SUBSPACE_ITERS {
                y.iter_mut().for_each(|row| row.fill(R::zero()));
                for (sr, vr) in scaled.iter_mut().zip(&v) {
                    for ((s, &x), &iv) in sr.iter_mut().zip(vr).zip(&inv_sd) {
                        *s = x * iv;
                    }
                }
                for i in 0..m {
                    if km.labels[i] != c {
                        continue;
                    }
                    let delta: Vec<R> = mu[i].iter().zip(&means[c]).map(|(&a, &b)| a - b).collect();
                    accumulate_scatter_rows(&scaled, &sig[i], &delta, n[i] / nk, &mut y);
                }
                for row in y.iter_mut() {
                    for (t, &iv) in row.iter_mut().zip(&inv_sd) {
                        *t = *t * iv;
                    }
                }
                if pass < INIT_SUBSPACE_ITERS {
                    orthonormalize(&mut y);
                    v = std::mem::replace(&mut y, vec![vec![R::zero(); dim]; q]);
                }
            }
            // Rayleigh quotients of the standardised scatter, where the noise floor is 1 per axis,
            // so `λ_r − 1` is the variance the factor genuinely adds.
            let lam: Vec<R> = (0..q).map(|r| dot(&v[r], &y[r]).max(R::zero())).collect();
            for r in 0..q {
                let s = (lam[r] - R::one()).max(load_floor).sqrt();
                for ((lv, &vv), &x) in loads[c][r].iter_mut().zip(&v[r]).zip(&var) {
                    *lv = s * vv * x.max(noise_floor).sqrt();
                }
            }
            for j in 0..dim {
                let taken = (0..q)
                    .map(|r| loads[c][r][j] * loads[c][r][j])
                    .fold(R::zero(), |a, b| a + b);
                noise[c][j] = (var[j] - taken).max(noise_floor);
            }
        }
    }
    let mut weights = vec![R::one() / R::from_usize(k).unwrap(); k];

    let mut resp = vec![vec![R::zero(); k]; m];
    let mut loglik = R::neg_infinity();
    let tol = R::from_f64(1e-7).unwrap();

    for it in 0..max_iter {
        // Per-component constants: `U_c = Ψ_c⁻¹ W_c`, the factor of `M_c`, its inverse, `log|Σ_c|`.
        let mut m_chol = Vec::with_capacity(k);
        let mut m_inv = Vec::with_capacity(k);
        let mut u = Vec::with_capacity(k);
        let mut logdet = vec![R::zero(); k];
        for c in 0..k {
            let uc: Vec<Vec<R>> = loads[c]
                .iter()
                .map(|w| w.iter().zip(&noise[c]).map(|(&x, &p)| x / p).collect())
                .collect();
            let mut mm = vec![vec![R::zero(); q]; q];
            for a in 0..q {
                for b in 0..=a {
                    let v = dot(&loads[c][a], &uc[b]);
                    mm[a][b] = v;
                    mm[b][a] = v;
                }
                mm[a][a] = mm[a][a] + R::one();
            }
            let (l, ld) = chol_spd(&mm, R::one());
            logdet[c] = noise[c]
                .iter()
                .map(|&p| p.ln())
                .fold(R::zero(), |a, b| a + b)
                + ld;
            m_inv.push(crate::linalg::inv_from_chol(&l));
            m_chol.push(l);
            u.push(uc);
        }

        // E-step, fused with the part of the M-step that does not need the new means: `Σ_i w_ic Σ_i U_c`
        // is mean-free, and it is the only term costing more than `O(q·d)`.
        let mut nk = vec![R::zero(); k];
        let mut mean_sum = vec![vec![R::zero(); dim]; k];
        let mut su = vec![vec![vec![R::zero(); dim]; q]; k];
        let mut diag_s = vec![vec![R::zero(); dim]; k];
        let mut new_ll = R::zero();
        let mut logr = vec![R::zero(); k];
        let mut sig_u = vec![vec![R::zero(); dim]; q];
        let mut per_leaf = vec![vec![vec![R::zero(); dim]; q]; k];
        for i in 0..m {
            for c in 0..k {
                let delta: Vec<R> = mu[i].iter().zip(&means[c]).map(|(&a, &b)| a - b).collect();
                // `δᵀΨ⁻¹δ − (Uᵀδ)ᵀ M⁻¹ (Uᵀδ)`.
                let iso = delta
                    .iter()
                    .zip(&noise[c])
                    .map(|(&d, &p)| d * d / p)
                    .fold(R::zero(), |a, b| a + b);
                let p: Vec<R> = u[c].iter().map(|ur| dot(ur, &delta)).collect();
                let quad =
                    (iso - crate::linalg::mahalanobis_sq_from_chol(&m_chol[c], &p)).max(R::zero());
                sig_u.iter_mut().for_each(|row| row.fill(R::zero()));
                sig[i].apply_rows(&u[c], &mut sig_u, R::one());
                // `tr(Ψ⁻¹Σ_i) − tr(M⁻¹ UᵀΣ_i U)`, with `(UᵀΣ_iU)[a][b] = u_b · (Σ_i u_a)`.
                let raw = diag_sig[i]
                    .iter()
                    .zip(&noise[c])
                    .map(|(&s, &p)| s / p)
                    .fold(R::zero(), |a, b| a + b);
                let mut folded = R::zero();
                for a in 0..q {
                    for b in 0..q {
                        folded = folded + m_inv[c][a][b] * dot(&u[c][b], &sig_u[a]);
                    }
                }
                let trace = (raw - folded).max(R::zero());
                logr[c] =
                    weights[c].ln() - half * (dimr * log_two_pi + logdet[c] + quad) - half * trace;
                per_leaf[c].clone_from(&sig_u);
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
                for (ds, &v) in diag_s[c].iter_mut().zip(&diag_sig[i]) {
                    *ds = *ds + w * v;
                }
                for (dst, src) in su[c].iter_mut().zip(&per_leaf[c]) {
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
                // An emptied component keeps its parameters; `ln π_c` floors it out of the posterior.
                weights[c] = R::zero();
            }
        }
        // The between-leaf term of `S_c`, which needs the new means. `δ (δᵀU)` is `O(q·d)`, so this
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
                for (ds, &d) in diag_s[c].iter_mut().zip(&delta) {
                    *ds = *ds + w * d * d;
                }
                for (row, ur) in su[c].iter_mut().zip(&u[c]) {
                    let coef = w * dot(ur, &delta);
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
            for row in su[c].iter_mut() {
                for v in row.iter_mut() {
                    *v = *v / nk[c];
                }
            }
            for v in diag_s[c].iter_mut() {
                *v = *v / nk[c];
            }
            if q > 0 {
                // `W_new = (S U)(M + G)⁻¹ M` with `G = Uᵀ S U`. `M + G ⪰ I`, so it factors; the
                // unsimplified `I − βW + βSβᵀ` would need a general inverse.
                let mut kmat = vec![vec![R::zero(); q]; q];
                for a in 0..q {
                    for b in 0..=a {
                        let g = dot(&u[c][b], &su[c][a]);
                        let mab = dot(&loads[c][a], &u[c][b]);
                        let v = if a == b { R::one() + mab + g } else { mab + g };
                        kmat[a][b] = v;
                        kmat[b][a] = v;
                    }
                }
                let (kl, _) = chol_spd(&kmat, R::one());
                // `z = (M + G)⁻¹ M`, solved column by column through the factor.
                let mut z = vec![vec![R::zero(); q]; q];
                for j in 0..q {
                    let col: Vec<R> = (0..q)
                        .map(|a| {
                            let base = dot(&loads[c][a], &u[c][j]);
                            if a == j { base + R::one() } else { base }
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
                            acc = acc + su[c][a][j] * z[a][r];
                        }
                        new_loads[c][r][j] = acc;
                    }
                }
                // `ψ_new[j] = S[j][j] − Σ_a W_new[a][j] (M⁻¹ (S U)ᵀ)[a][j]`, with the *old* `M`.
                for j in 0..dim {
                    let mut folded = R::zero();
                    for a in 0..q {
                        let mut msu = R::zero();
                        for b in 0..q {
                            msu = msu + m_inv[c][a][b] * su[c][b][j];
                        }
                        folded = folded + new_loads[c][a][j] * msu;
                    }
                    new_noise[c][j] = (diag_s[c][j] - folded).max(noise_floor);
                }
            } else {
                for j in 0..dim {
                    new_noise[c][j] = diag_s[c][j].max(noise_floor);
                }
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
    let mixture = Mixture::factor_analysis(&weights, &means, &loads, &noise);
    Mfa {
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

/// Fit a `k`-component MFA with `rank` factors, keeping the best of [`MFA_N_INIT`] EM restarts by
/// log-likelihood.
pub fn mfa<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> Mfa<R> {
    best_of_restarts(
        MFA_N_INIT,
        seed,
        |g: &Mfa<R>| g.loglik,
        |s| mfa_once(features, k, rank, max_iter, s),
    )
}

/// MFA with automatic component count (BIC over `k ∈ [k_min, k_max]`). A component costs `d` mean
/// parameters, `d` noise variances and `d·q − q(q−1)/2` free loadings — the Stiefel rotation of
/// `W_c` is unidentifiable, so that part of it is not counted. The `d − 1` difference from `mppca`
/// is exactly what BIC charges the head for its extra freedom.
pub fn mfa_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> Mfa<R> {
    let d = features[0].dim();
    let q = rank.min(d.saturating_sub(1));
    let ntot = total_weight(features);
    let k_hi = k_max.min(features.len()).max(1);
    let k_lo = k_min.max(1).min(k_hi);
    let mut best_score = R::infinity();
    let mut best: Option<Mfa<R>> = None;
    for k in k_lo..=k_hi {
        let g = mfa_once(features, k, rank, max_iter, seed);
        let p = k * (d + d + d * q - q * q.saturating_sub(1) / 2) + (k - 1);
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
    use crate::clustering::mppca::mppca;
    use crate::clustering::testutil::ari;
    use crate::feature::{FdSketch, Full};

    /// Dense `Σ_c = W Wᵀ + diag(ψ)`, the matrix the head never forms — the reference every
    /// matrix-free claim below is checked against.
    fn dense_cov(loads: &[Vec<f64>], noise: &[f64]) -> Vec<Vec<f64>> {
        let d = noise.len();
        let mut cov = vec![vec![0.0; d]; d];
        for w in loads {
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] += w[i] * w[j];
                }
            }
        }
        for j in 0..d {
            cov[j][j] += noise[j];
        }
        cov
    }

    fn leaves(pts: &[Vec<f64>], per: usize) -> Vec<Full<f64>> {
        pts.chunks(per)
            .map(|c| {
                let mut f = Full::new(c[0].len());
                for p in c {
                    f.push(p, 1.0);
                }
                f
            })
            .collect()
    }

    /// The generator's label per **leaf**. Every fixture here packs whole clusters into whole
    /// leaves, so a leaf's label is its first point's — and scoring a 180-leaf labelling against a
    /// 900-point truth is the silent way to read ARI 0 off a head that works.
    fn leaf_truth(truth: &[usize], per: usize) -> Vec<usize> {
        truth.chunks(per).map(|c| c[0]).collect()
    }

    /// Three clusters separated on a **quiet** axis while two loud axes carry no signal — the case
    /// an isotropic residual cannot express.
    ///
    /// `mppca` fits one `σ²` to the whole residual, which the loud axes dominate (`σ² ≈ 0.31`
    /// here). Under that scale the separation on axis 0 contributes `1.2²/0.31 ≈ 4.6` to the
    /// Mahalanobis distance while the noise on axes 4-5 contributes several times more, so the
    /// signal is drowned. Under `diag(ψ)` the same separation contributes `1.2²/0.014 ≈ 100`.
    fn heteroscedastic(seed: u64, n: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let d = 6;
        let sd = [0.12, 0.12, 0.35, 0.35, 1.6, 1.6];
        let (mut pts, mut truth) = (Vec::new(), Vec::new());
        for (c, &offset) in [0.0, 1.2, -1.2].iter().enumerate() {
            for _ in 0..n {
                let z = rng.gauss();
                let mut p = vec![0.0; d];
                for j in 0..d {
                    // One shared factor tilting axes 2-3, then independent per-axis noise.
                    let load = if (2..4).contains(&j) { 0.8 } else { 0.0 };
                    let centre = if j == 0 { offset } else { 0.0 };
                    p[j] = centre + load * z + sd[j] * rng.gauss();
                }
                pts.push(p);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    /// Three thin lines through a common origin at 0°, 60° and 120°, with two nuisance axes. Every
    /// cluster has the *same* per-axis scale profile and differs only in orientation — the case
    /// `mppca` owns and this head does not. See the loss test below.
    fn crossing_lines(seed: u64, n: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let d = 6;
        let sd = [0.15, 0.15, 0.3, 0.3, 0.9, 0.9];
        let (mut pts, mut truth) = (Vec::new(), Vec::new());
        for (c, &ang) in [0.0_f64, 1.047, 2.094].iter().enumerate() {
            for _ in 0..n {
                let z = rng.gauss();
                let mut p: Vec<f64> = (0..d).map(|j| sd[j] * rng.gauss()).collect();
                p[0] += 2.0 * z * ang.cos();
                p[1] += 2.0 * z * ang.sin();
                pts.push(p);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    /// The head's whole claim: `Σ_c⁻¹` and `log|Σ_c|` computed through `M_c` must equal the dense
    /// ones. Checked against a matrix built and inverted the expensive way.
    #[test]
    fn the_woodbury_form_matches_the_covariance_it_never_builds() {
        let loads = vec![vec![1.0, -2.0, 0.5, 3.0], vec![0.0, 1.0, 2.0, -1.0]];
        let noise = vec![0.7, 2.5, 0.1, 1.3];
        let d = noise.len();
        let q = loads.len();
        let cov = dense_cov(&loads, &noise);

        let u: Vec<Vec<f64>> = loads
            .iter()
            .map(|w| w.iter().zip(&noise).map(|(&x, &p)| x / p).collect())
            .collect();
        let mut mm = vec![vec![0.0; q]; q];
        for a in 0..q {
            for b in 0..q {
                mm[a][b] = dot(&loads[a], &u[b]) + f64::from(u8::from(a == b));
            }
        }
        let l = crate::linalg::cholesky_lower(&mm).unwrap();
        let minv = crate::linalg::inv_from_chol(&l);

        // Σ · [Ψ⁻¹ − U M⁻¹ Uᵀ] = I, the same residual the Maxima file reports as exactly zero.
        let mut inv = vec![vec![0.0; d]; d];
        for i in 0..d {
            inv[i][i] = 1.0 / noise[i];
            for j in 0..d {
                for a in 0..q {
                    for b in 0..q {
                        inv[i][j] -= u[a][i] * minv[a][b] * u[b][j];
                    }
                }
            }
        }
        for (i, crow) in cov.iter().enumerate() {
            for j in 0..d {
                let prod: f64 = crow.iter().zip(&inv).map(|(&c, ir)| c * ir[j]).sum();
                let want = f64::from(u8::from(i == j));
                assert!((prod - want).abs() < 1e-9, "({i},{j}): {prod} vs {want}");
            }
        }

        // log|Σ| = Σ ln ψ_j + log|M|.
        let claim: f64 =
            noise.iter().map(|p| p.ln()).sum::<f64>() + crate::linalg::logdet_from_chol(&l);
        let dense = crate::linalg::logdet_from_chol(&crate::linalg::cholesky_lower(&cov).unwrap());
        assert!((claim - dense).abs() < 1e-9, "{claim} vs {dense}");
    }

    /// The reason the head exists. Where the axes have different scales and the separation sits on
    /// a quiet one, a per-dimension `Ψ` recovers the partition outright and an isotropic `σ² I`
    /// does not — it has to average the loud axes against the tight one carrying the signal.
    #[test]
    fn diagonal_noise_beats_isotropic_where_the_axes_have_different_scales() {
        for seed in 0..5 {
            let (pts, truth) = heteroscedastic(seed, 300);
            let feats = leaves(&pts, 5);
            let truth = leaf_truth(&truth, 5);
            let a = ari(&mfa(&feats, 3, 1, 200, seed).labels, &truth);
            let b = ari(&mppca(&feats, 3, 1, 200, seed).labels, &truth);
            assert!(a > 0.9, "seed {seed}: mfa only {a}");
            assert!(
                b < 0.6,
                "seed {seed}: mppca reached {b}, fixture no longer discriminates"
            );
        }
    }

    /// …and the other side of the same coin, which the docs promise and this pins.
    ///
    /// Where every cluster has the same per-axis scale and differs only in **orientation**, the
    /// extra freedom is a liability: `ψ` can absorb an elongation that should have gone into `W`,
    /// so the factor is under-determined and EM settles on a diagonal solution with no orientation
    /// at all. `mppca`'s single `σ²` cannot absorb it, which forces the anisotropy into `W` and
    /// wins the fixture outright. This is not a bug to fix here — it is why both heads ship.
    #[test]
    fn isotropic_noise_beats_diagonal_where_only_the_orientation_differs() {
        let (mut mine, mut theirs) = (0.0, 0.0);
        for seed in 0..3 {
            let (pts, truth) = crossing_lines(seed, 300);
            let feats = leaves(&pts, 5);
            let truth = leaf_truth(&truth, 5);
            mine += ari(&mfa(&feats, 3, 1, 200, seed).labels, &truth);
            theirs += ari(&mppca(&feats, 3, 1, 200, seed).labels, &truth);
        }
        assert!(
            theirs > mine + 1.0,
            "mppca {theirs:.2} vs mfa {mine:.2} over 3 seeds — the documented loss has moved"
        );
    }

    /// `Ψ` must track the generator's per-axis noise, not merely differ from a scalar. Axes 4-5 are
    /// 13x louder than axis 0 by construction.
    #[test]
    fn the_fitted_noise_recovers_the_per_axis_scale() {
        let (pts, _truth) = heteroscedastic(7, 400);
        let feats = leaves(&pts, 5);
        let fit = mfa(&feats, 3, 1, 300, 7);
        let biggest = fit
            .weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let psi = &fit.noise[biggest];
        assert!(
            psi[4] > 4.0 * psi[0] && psi[5] > 4.0 * psi[0],
            "psi did not separate the loud axes from the tight one: {psi:?}"
        );
    }

    /// `rank=0` leaves no subspace, so the head is exactly a diagonal Gaussian mixture — the bottom
    /// rung of the ladder, and a check that the `q = 0` branch is not dead code.
    #[test]
    fn rank_zero_is_a_diagonal_gaussian_mixture() {
        use crate::clustering::gmm::gmm_diagonal;
        let (pts, truth) = heteroscedastic(3, 200);
        let feats = leaves(&pts, 5);
        let truth = leaf_truth(&truth, 5);
        let a = mfa(&feats, 3, 0, 200, 3);
        let b = gmm_diagonal(&feats, 3, 200, 3);
        assert!(
            (ari(&a.labels, &truth) - ari(&b.labels, &truth)).abs() < 0.05,
            "rank-0 MFA {} vs diagonal GMM {}",
            ari(&a.labels, &truth),
            ari(&b.labels, &truth)
        );
        assert!(a.loads.iter().all(|w| w.is_empty()));
    }

    /// The E-step reads the leaf's *scatter*, not only its mean: an `FdSketch` leaf must reach the
    /// same partition a `Full` one does on data whose rank the sketch retains.
    #[test]
    fn an_fd_sketch_leaf_reaches_the_same_partition_as_a_full_one() {
        let (pts, truth) = heteroscedastic(11, 300);
        let truth = leaf_truth(&truth, 5);
        let full = leaves(&pts, 5);
        let sketch: Vec<FdSketch<f64>> = pts
            .chunks(5)
            .map(|c| {
                let mut f = FdSketch::with_ell(c[0].len(), 6);
                for p in c {
                    f.push(p, 1.0);
                }
                f
            })
            .collect();
        let a = ari(&mfa(&full, 3, 1, 200, 11).labels, &truth);
        let b = ari(&mfa(&sketch, 3, 1, 200, 11).labels, &truth);
        assert!(a > 0.7 && b > 0.7, "full {a}, sketch {b}");
        assert!((a - b).abs() < 0.15, "full {a} vs sketch {b}");
    }

    /// The mixture the head publishes must agree with the labels it publishes, or `predict` and
    /// `labels_` would disagree on the very points that were fitted.
    #[test]
    fn the_published_mixture_reproduces_the_head_labels() {
        let (pts, _truth) = heteroscedastic(5, 200);
        let feats = leaves(&pts, 5);
        let fit = mfa(&feats, 3, 1, 200, 5);
        for (f, &want) in feats.iter().zip(&fit.labels) {
            assert_eq!(fit.mixture.assign(f.mean()), want);
        }
    }

    /// BIC has to charge the head for its `d − 1` extra parameters per component. The sharp form of
    /// that: on a **single** Gaussian blob no extra component may pay for itself.
    ///
    /// This is the test the PCA-style initialisation failed. Seeded from the raw scatter, `W` landed
    /// on the loudest coordinate — where `ψ` already explains everything — the `k = 1` likelihood
    /// came out 755 nats short of what the model can reach, and the sweep bought that gap back by
    /// adding components, answering `k = 3` on one blob at every leaf budget tried. With the
    /// standardised init the likelihood is flat in `k` here, which is what makes the penalty decide.
    #[test]
    fn auto_k_does_not_split_a_single_blob() {
        let mut rng = SplitMix64::new(2);
        let sd = [0.12, 0.12, 0.35, 0.35, 1.6, 1.6];
        let mut pts = Vec::new();
        for _ in 0..1200 {
            let z = rng.gauss();
            let mut p = vec![0.0; 6];
            for j in 0..6 {
                let load = if (2..4).contains(&j) { 0.8 } else { 0.0 };
                p[j] = load * z + sd[j] * rng.gauss();
            }
            pts.push(p);
        }
        for per in [5, 20] {
            let feats = leaves(&pts, per);
            let fit = mfa_auto(&feats, 1, 6, 1, 200, 2);
            assert_eq!(
                fit.weights.len(),
                1,
                "{per} points per leaf: auto-k split one blob"
            );
        }
    }

    /// …and the factor it fits on that blob is the generator's, not the loudest axis. `W` must load
    /// on the two *correlated* axes (2 and 3, shared factor 0.8) and not on axes 4-5, whose variance
    /// is 13x larger but entirely independent — `ψ` explains those for free.
    #[test]
    fn the_factor_loads_on_correlation_not_on_variance() {
        let mut rng = SplitMix64::new(2);
        let sd = [0.12, 0.12, 0.35, 0.35, 1.6, 1.6];
        let mut pts = Vec::new();
        for _ in 0..1200 {
            let z = rng.gauss();
            let mut p = vec![0.0; 6];
            for j in 0..6 {
                let load = if (2..4).contains(&j) { 0.8 } else { 0.0 };
                p[j] = load * z + sd[j] * rng.gauss();
            }
            pts.push(p);
        }
        let feats = leaves(&pts, 5);
        let w = &mfa(&feats, 1, 1, 300, 2).loads[0][0];
        let correlated = w[2].abs().min(w[3].abs());
        let loudest = w[4].abs().max(w[5].abs());
        assert!(
            correlated > 5.0 * loudest,
            "W chased variance instead of correlation: {w:?}"
        );
        assert!(
            (w[2].abs() - 0.8).abs() < 0.2 && (w[3].abs() - 0.8).abs() < 0.2,
            "loadings are not the generator's 0.8: {w:?}"
        );
    }
}
