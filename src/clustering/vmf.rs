//! Directional clustering on the unit hypersphere — spherical k-means and a mixture of
//! von Mises–Fisher (movMF) distributions over leaf clustering features.
//!
//! On L2-normalized data every point lies on `S^{d-1}`, so a leaf's mean `μ_i` points in its
//! resultant direction with length `R̄_i = ‖μ_i‖ ∈ [0, 1]` — a length that already encodes the
//! within-leaf angular spread. Each leaf is used as its weighted mean `(n_i, μ_i)` directly (kept
//! un-normalized), exactly as the k-means / GMM heads use a leaf's mean. The cluster resultant
//! `R_c = Σ_{i∈c} n_i μ_i` is additive, so the BETULA exact-merge property carries straight through
//! to the directional model.
//!
//! - [`spherical_kmeans`]: hard assignment by maximal cosine (`argmax_c μ_i·μ_c`), centers
//!   re-normalized to the sphere. The objective is cohesion `Σ_c ‖R_c‖` (maximized).
//! - [`movmf`] / [`movmf_auto`]: soft EM for a vMF mixture. Concentration `κ` is estimated with
//!   the Banerjee et al. (2005) approximation `κ̂ ≈ R̄(d − R̄²)/(1 − R̄²)`; the normalizer
//!   `C_d(κ)` uses a stable log-space series for `log I_ν(κ)` (no Bessel library).

use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::mixture::Mixture;
use crate::types::Real;

/// Concentration is capped for numerical stability: `κ·(d_i·μ_c)` stays representable, and the
/// `log I_ν(κ)` series stays short. A cluster tighter than this is already effectively a point.
const KAPPA_MAX: f64 = 1e4;
/// EM restarts kept by best data log-likelihood (mirrors the GMM head's restart budget).
const MOVMF_N_INIT: u64 = 4;

// ───────────────────────── special functions (f64) ─────────────────────────

/// `ln Γ(x)` via the Lanczos approximation (g = 7). Only called with `x ≥ 0.5` here
/// (`x = ν+m+1`, `ν = d/2 − 1 ≥ −1/2`), so no reflection formula is needed.
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let g = 7.0_f64;
    let x = x - 1.0;
    let t = x + g + 0.5;
    let mut a = C[0];
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// `log I_ν(κ)` for `κ > 0`, `ν ≥ 0`, via the all-positive power series
/// `I_ν(κ) = Σ_m (κ/2)^{2m+ν} / (m! Γ(ν+m+1))`. The `(κ/2)^ν` factor is pulled out and the
/// term ratio `a_m/a_{m-1} = (κ/2)² / (m(ν+m))` is accumulated in log-space with an online
/// log-sum-exp, so nothing overflows even for large `κ` (the peak term sits near `m ≈ κ/2`).
pub(crate) fn log_iv(nu: f64, kappa: f64) -> f64 {
    let half_ln = (kappa * 0.5).ln(); // ln(κ/2)
    let mut log_a = -ln_gamma(nu + 1.0); // log a_0  (a_0 = 1/Γ(ν+1))
    let mut max_log = log_a;
    let mut sum_exp = 1.0_f64; // Σ exp(log a_m − max_log), starts at exp(0)
    let mut m = 1.0_f64;
    loop {
        log_a += 2.0 * half_ln - m.ln() - (nu + m).ln();
        if log_a > max_log {
            sum_exp = sum_exp * (max_log - log_a).exp() + 1.0;
            max_log = log_a;
        } else {
            sum_exp += (log_a - max_log).exp();
        }
        // Stop once past the peak (m > κ/2) and the tail is negligible; hard cap for safety.
        if (m > kappa * 0.5 && log_a < max_log - 40.0) || m > 200_000.0 {
            break;
        }
        m += 1.0;
    }
    nu * half_ln + max_log + sum_exp.ln()
}

/// `log C_d(κ)` — the log normalizing constant of the vMF density on `S^{d-1}`:
/// `C_d(κ) = κ^{d/2−1} / ((2π)^{d/2} I_{d/2−1}(κ))`.
fn log_vmf_norm(dim: usize, kappa: f64) -> f64 {
    let d = dim as f64;
    let nu = d / 2.0 - 1.0;
    nu * kappa.ln() - (d / 2.0) * std::f64::consts::TAU.ln() - log_iv(nu, kappa)
}

/// Banerjee et al. (2005) concentration estimate from the mean resultant length `R̄ = ‖R‖/n`.
fn estimate_kappa(rbar: f64, dim: usize) -> f64 {
    let d = dim as f64;
    let r = rbar.clamp(1e-8, 1.0 - 1e-9);
    ((r * (d - r * r)) / (1.0 - r * r)).clamp(1e-8, KAPPA_MAX)
}

// ───────────────────────── shared helpers ─────────────────────────

fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn norm<R: Real>(v: &[R]) -> R {
    v.iter().map(|&x| x * x).sum::<R>().sqrt()
}

/// Unit-normalize a vector; a (near-)zero vector is returned unchanged.
fn unit<R: Real>(v: &[R]) -> Vec<R> {
    let nrm = norm(v);
    if nrm > R::zero() {
        v.iter().map(|&x| x / nrm).collect()
    } else {
        v.to_vec()
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

/// Reduce each leaf to its (weighted) mean vector `μ_i` and weight `n_i`. On L2-normalized data
/// `μ_i` is the leaf's resultant direction with length `‖μ_i‖ = R̄_i ≤ 1` carrying the *within-leaf*
/// angular spread. The mean is kept as-is — **not** re-normalized to a unit direction — so the
/// cluster resultant `R_c = Σ_{i∈c} n_i μ_i` reflects both between- and within-leaf dispersion, which
/// is what a correct concentration `κ_c` requires (re-normalizing per leaf would discard the
/// within-leaf spread and wildly over-estimate `κ`). A degenerate leaf (`μ_i ≈ 0`) contributes
/// nothing to any resultant and is a no-op in the max-dot assignment.
fn leaf_means<R: Real, C: ClusterFeature<R>>(features: &[C]) -> (Vec<Vec<R>>, Vec<R>) {
    features
        .iter()
        .map(|f| (f.mean().to_vec(), f.weight()))
        .unzip()
}

/// Sample an index with probability proportional to `w` (deterministic given `rng`).
fn weighted_pick(w: &[f64], rng: &mut SplitMix64) -> usize {
    let tot: f64 = w.iter().sum();
    if tot <= 0.0 {
        return (rng.next_u64() as usize) % w.len().max(1);
    }
    let mut t = rng.next_f64() * tot;
    for (i, &wi) in w.iter().enumerate() {
        t -= wi;
        if t <= 0.0 {
            return i;
        }
    }
    w.len() - 1
}

// ───────────────────────── spherical k-means ─────────────────────────

/// Result of a spherical k-means run over leaf features.
pub struct SphericalKMeans<R: Real> {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
    /// Unit cluster mean directions `μ_c`.
    pub centers: Vec<Vec<R>>,
    /// Cohesion `Σ_c ‖Σ_{i∈c} n_i μ_i‖` — the maximized objective (higher is better).
    pub cohesion: R,
}

/// Cluster `features` into `k` directional groups on the unit sphere. `n_init` k-means++-style
/// restarts (angular `D²` seeding) are run and the highest-cohesion labelling is kept.
pub fn spherical_kmeans<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> SphericalKMeans<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let (means, n) = leaf_means(features);
    let dim = features[0].dim();
    let mut rng = SplitMix64::new(seed);
    let mut best: Option<SphericalKMeans<R>> = None;
    for _ in 0..n_init.max(1) {
        let init = spherical_pp(&means, &n, k, &mut rng);
        let res = spherical_lloyd(&means, &n, init, max_iter, dim);
        match &best {
            Some(b) if res.cohesion <= b.cohesion => {}
            _ => best = Some(res),
        }
    }
    best.expect("at least one init")
}

/// Weighted k-means++ seeding with angular distance `1 − ĉ·μ̂` in place of squared Euclidean.
/// Seed centers are unit vectors (the leaf means are normalized when chosen).
fn spherical_pp<R: Real>(means: &[Vec<R>], n: &[R], k: usize, rng: &mut SplitMix64) -> Vec<Vec<R>> {
    let ang = |a: &[R], c: &[R]| (1.0 - dot(a, c).to_f64().unwrap_or(0.0)).max(0.0);
    let w0: Vec<f64> = n.iter().map(|&w| w.to_f64().unwrap_or(0.0)).collect();
    let mut centers = Vec::with_capacity(k);
    centers.push(unit(&means[weighted_pick(&w0, rng)]));
    let mut d2: Vec<f64> = means.iter().map(|x| ang(x, &centers[0])).collect();
    while centers.len() < k {
        let probs: Vec<f64> = w0.iter().zip(&d2).map(|(&w, &d)| w * d).collect();
        let c = unit(&means[weighted_pick(&probs, rng)]);
        for (di, x) in d2.iter_mut().zip(means) {
            let nd = ang(x, &c);
            if nd < *di {
                *di = nd;
            }
        }
        centers.push(c);
    }
    centers
}

#[allow(clippy::needless_range_loop)] // spherical accumulation reads clearest with explicit d/c indices
fn spherical_lloyd<R: Real>(
    means: &[Vec<R>],
    n: &[R],
    mut centers: Vec<Vec<R>>,
    max_iter: usize,
    dim: usize,
) -> SphericalKMeans<R> {
    let m = means.len();
    let k = centers.len();
    let mut labels = vec![0usize; m];
    for it in 0..max_iter.max(1) {
        // Assign each leaf to the center with the largest cosine (‖μ_i‖ is constant across centers,
        // so max_c μ_i·μ_c is the max-cosine assignment); track how well each leaf is served.
        let mut changed = false;
        let mut served = vec![R::neg_infinity(); m];
        for i in 0..m {
            let mut best = 0;
            let mut best_dot = dot(&means[i], &centers[0]);
            for c in 1..k {
                let d = dot(&means[i], &centers[c]);
                if d > best_dot {
                    best_dot = d;
                    best = c;
                }
            }
            served[i] = best_dot;
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed && it > 0 {
            break;
        }
        // Update: μ_c ← normalize(Σ_{i∈c} n_i μ_i); reseed an empty cluster from the worst-served leaf.
        let mut acc = vec![vec![R::zero(); dim]; k];
        let mut count = vec![0usize; k];
        for i in 0..m {
            let c = labels[i];
            count[c] += 1;
            for d in 0..dim {
                acc[c][d] = acc[c][d] + n[i] * means[i][d];
            }
        }
        for c in 0..k {
            if count[c] == 0 {
                let worst = argmin(&served);
                centers[c] = unit(&means[worst]);
                served[worst] = R::infinity(); // don't reseed two clusters onto the same leaf
                continue;
            }
            let nrm = norm(&acc[c]);
            if nrm > R::zero() {
                for d in 0..dim {
                    centers[c][d] = acc[c][d] / nrm;
                }
            }
        }
    }
    let cohesion = cohesion_of(means, n, &labels, k, dim);
    SphericalKMeans {
        labels,
        centers,
        cohesion,
    }
}

fn argmin<R: Real>(v: &[R]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x < v[best] {
            best = i;
        }
    }
    best
}

#[allow(clippy::needless_range_loop)]
fn cohesion_of<R: Real>(means: &[Vec<R>], n: &[R], labels: &[usize], k: usize, dim: usize) -> R {
    let mut acc = vec![vec![R::zero(); dim]; k];
    for (i, &c) in labels.iter().enumerate() {
        for d in 0..dim {
            acc[c][d] = acc[c][d] + n[i] * means[i][d];
        }
    }
    acc.iter().map(|a| norm(a)).fold(R::zero(), |s, x| s + x)
}

// ───────────────────────── mixture of von Mises–Fisher ─────────────────────────

/// Result of a movMF EM run over leaf features.
pub struct Movmf<R: Real> {
    /// Hard label (argmax responsibility) per input feature.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[feature][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_c`.
    pub weights: Vec<R>,
    /// Unit component mean directions `μ_c`.
    pub means: Vec<Vec<R>>,
    /// Component concentrations `κ_c`.
    pub kappas: Vec<R>,
    /// Weighted data log-likelihood at convergence.
    pub loglik: R,
    /// The fitted density, for scoring raw points.
    pub mixture: Mixture,
}

/// Fit a `k`-component vMF mixture, keeping the best of [`MOVMF_N_INIT`] EM restarts by
/// log-likelihood.
pub fn movmf<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Movmf<R> {
    let mut best: Option<Movmf<R>> = None;
    for s in 0..MOVMF_N_INIT {
        let r = movmf_once(
            features,
            k,
            max_iter,
            seed.wrapping_add(s.wrapping_mul(0x9E37_79B9)),
        );
        match &best {
            Some(b) if r.loglik <= b.loglik => {}
            _ => best = Some(r),
        }
    }
    best.expect("MOVMF_N_INIT >= 1")
}

#[allow(clippy::needless_range_loop)] // EM over (leaf i, component c, dim d) is clearest indexed
fn movmf_once<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Movmf<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let (mu, n) = leaf_means(features);
    let dim = features[0].dim();
    let m = mu.len();

    // Warm-start directions from spherical k-means; seed κ from the hard-assignment resultants.
    let km = spherical_kmeans(features, k, 50, 1, seed);
    let mut means = km.centers;
    let mut weights = vec![R::one() / R::from_usize(k).unwrap(); k];
    let mut kappas = init_kappas(&mu, &n, &km.labels, k, dim);

    let mut resp = vec![vec![R::zero(); k]; m];
    let mut loglik = R::neg_infinity();
    let tol = R::from_f64(1e-7).unwrap();

    for it in 0..max_iter {
        let logc: Vec<f64> = kappas
            .iter()
            .map(|&kap| log_vmf_norm(dim, kap.to_f64().unwrap().max(1e-8)))
            .collect();
        let lw: Vec<f64> = weights
            .iter()
            .map(|&w| w.to_f64().unwrap_or(0.0).max(1e-300).ln())
            .collect();
        let kf: Vec<f64> = kappas.iter().map(|&kap| kap.to_f64().unwrap()).collect();

        // E-step (log-space softmax): log r_ic = ln π_c + log C_d(κ_c) + κ_c (d_i·μ_c).
        let mut new_ll = R::zero();
        for i in 0..m {
            let mut logr = vec![0.0_f64; k];
            for c in 0..k {
                let dv = dot(&mu[i], &means[c]).to_f64().unwrap_or(0.0);
                logr[c] = lw[c] + logc[c] + kf[c] * dv;
            }
            let mx = logr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut s = 0.0;
            for &l in &logr {
                s += (l - mx).exp();
            }
            let lse = mx + s.ln();
            new_ll = new_ll + n[i] * R::from_f64(lse).unwrap();
            for c in 0..k {
                resp[i][c] = R::from_f64((logr[c] - lse).exp()).unwrap();
            }
        }

        // M-step: π_c, μ_c = normalize(Σ n_i r_ic d_i), κ_c via Banerjee.
        let mut rvec = vec![vec![R::zero(); dim]; k];
        let mut nk = vec![R::zero(); k];
        for i in 0..m {
            for c in 0..k {
                let w = n[i] * resp[i][c];
                nk[c] = nk[c] + w;
                for d in 0..dim {
                    rvec[c][d] = rvec[c][d] + w * mu[i][d];
                }
            }
        }
        let ntot: R = nk.iter().copied().sum();
        for c in 0..k {
            weights[c] = if ntot > R::zero() {
                nk[c] / ntot
            } else {
                R::one() / R::from_usize(k).unwrap()
            };
            let rnorm = norm(&rvec[c]);
            if rnorm > R::zero() {
                for d in 0..dim {
                    means[c][d] = rvec[c][d] / rnorm;
                }
            }
            let rbar = if nk[c] > R::zero() {
                (rnorm / nk[c]).to_f64().unwrap_or(0.0)
            } else {
                0.0
            };
            kappas[c] = R::from_f64(estimate_kappa(rbar, dim)).unwrap();
        }

        if it > 0 && (new_ll - loglik).abs() <= tol * loglik.abs().max(R::one()) {
            loglik = new_ll;
            break;
        }
        loglik = new_ll;
    }

    let labels = resp.iter().map(|r| argmax(r)).collect();
    let logc: Vec<f64> = kappas
        .iter()
        .map(|&kap| log_vmf_norm(dim, kap.to_f64().unwrap_or(0.0).max(1e-8)))
        .collect();
    let mixture = Mixture::vmf(&weights, &means, &kappas, &logc);
    Movmf {
        labels,
        resp,
        weights,
        means,
        kappas,
        loglik,
        mixture,
    }
}

/// Per-component κ from a hard labelling's resultants (used to seed EM).
fn init_kappas<R: Real>(mu: &[Vec<R>], n: &[R], labels: &[usize], k: usize, dim: usize) -> Vec<R> {
    let mut acc = vec![vec![R::zero(); dim]; k];
    let mut nk = vec![R::zero(); k];
    for (i, &c) in labels.iter().enumerate() {
        nk[c] = nk[c] + n[i];
        for (a, &v) in acc[c].iter_mut().zip(&mu[i]) {
            *a = *a + n[i] * v;
        }
    }
    (0..k)
        .map(|c| {
            let rbar = if nk[c] > R::zero() {
                (norm(&acc[c]) / nk[c]).to_f64().unwrap_or(0.0)
            } else {
                0.0
            };
            R::from_f64(estimate_kappa(rbar, dim)).unwrap()
        })
        .collect()
}

/// Fit a vMF mixture with automatic component count: fit every `k ∈ [k_min, k_max]` and keep the
/// lowest-BIC model. Free parameters per component: `d` (mean direction) + `1` (κ), plus `k−1`
/// mixing weights.
pub fn movmf_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> Movmf<R> {
    let d = features[0].dim();
    let ntot: R = features.iter().map(|f| f.weight()).sum();
    let two = R::from_f64(2.0).unwrap();
    let k_hi = k_max.min(features.len()).max(1);
    let k_lo = k_min.max(1).min(k_hi);
    let mut best_score = R::infinity();
    let mut best: Option<Movmf<R>> = None;
    for k in k_lo..=k_hi {
        let g = movmf(features, k, max_iter, seed);
        let p = k * (d + 1) + (k - 1);
        let score = -two * g.loglik + R::from_usize(p).unwrap() * ntot.ln();
        if score < best_score {
            best_score = score;
            best = Some(g);
        }
    }
    best.expect("k_lo <= k_hi")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::ari;
    use crate::feature::{ClusterFeature, Spherical};

    /// `I_{1/2}(κ) = √(2/(πκ))·sinh κ` exactly — a closed form to validate `log_iv`.
    fn log_i_half(kappa: f64) -> f64 {
        0.5 * (2.0 / (std::f64::consts::PI * kappa)).ln() + kappa.sinh().ln()
    }

    #[test]
    fn log_iv_matches_half_integer_order() {
        for &k in &[0.25_f64, 1.0, 5.0, 20.0, 100.0] {
            let got = log_iv(0.5, k);
            let want = log_i_half(k);
            assert!((got - want).abs() < 1e-8, "κ={k}: got {got}, want {want}");
        }
    }

    #[test]
    fn log_iv_matches_known_i0_i1() {
        // I_0(1) = 1.2660658777520082, I_1(1) = 0.5651591039924850.
        assert!((log_iv(0.0, 1.0).exp() - 1.266_065_877_752_008).abs() < 1e-9);
        assert!((log_iv(1.0, 1.0).exp() - 0.565_159_103_992_485).abs() < 1e-9);
    }

    #[test]
    fn a_fully_concentrated_resultant_stays_inside_the_kappa_range() {
        // R̄ = 1 is the pole of the Banerjee estimate: the denominator 1 − R̄² vanishes, and in one
        // dimension the numerator vanishes with it. Holding R̄ off the pole is what keeps that ratio
        // from being 0/0 -- a NaN seed κ that every later E-step then carries.
        for dim in 1..=5usize {
            let got = estimate_kappa(1.0, dim);
            assert!(
                got.is_finite() && (1e-8..=KAPPA_MAX).contains(&got),
                "dim {dim}: κ = {got}"
            );
        }
        assert!(
            estimate_kappa(1.0, 1) < KAPPA_MAX,
            "one dimension saturates the concentration cap, so the fixture never reaches the 0/0 \
             corner that only the degenerate sphere has"
        );
    }

    /// `per` unit points around each random unit direction in `dim` dims, one Spherical leaf each.
    fn unit_blobs(
        rng: &mut SplitMix64,
        dim: usize,
        centers: usize,
        per: usize,
        noise: f64,
    ) -> (Vec<Spherical<f64>>, Vec<usize>) {
        let dirs: Vec<Vec<f64>> = (0..centers)
            .map(|_| {
                let mut v: Vec<f64> = (0..dim).map(|_| rng.gauss()).collect();
                let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                for x in &mut v {
                    *x /= nrm;
                }
                v
            })
            .collect();
        let mut feats = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in dirs.iter().enumerate() {
            for _ in 0..per {
                let mut p: Vec<f64> = ctr.iter().map(|&m| m + noise * rng.gauss()).collect();
                let nrm = p.iter().map(|x| x * x).sum::<f64>().sqrt();
                for x in &mut p {
                    *x /= nrm;
                }
                let mut f = Spherical::new(dim);
                f.push(&p, 1.0);
                feats.push(f);
                truth.push(c);
            }
        }
        (feats, truth)
    }

    #[test]
    fn spherical_kmeans_separates_directions() {
        let mut rng = SplitMix64::new(7);
        let (feats, truth) = unit_blobs(&mut rng, 8, 4, 60, 0.15);
        let res = spherical_kmeans(&feats, 4, 100, 4, 1);
        assert!(
            ari(&res.labels, &truth) > 0.95,
            "ARI = {}",
            ari(&res.labels, &truth)
        );
        for c in &res.centers {
            assert!((norm(c) - 1.0).abs() < 1e-9, "center not unit");
        }
    }

    #[test]
    fn movmf_recovers_directions() {
        let mut rng = SplitMix64::new(11);
        let (feats, truth) = unit_blobs(&mut rng, 10, 3, 80, 0.2);
        let res = movmf(&feats, 3, 100, 3);
        assert!(
            ari(&res.labels, &truth) > 0.9,
            "ARI = {}",
            ari(&res.labels, &truth)
        );
        for &kap in &res.kappas {
            assert!(kap > 0.0 && kap.is_finite(), "bad κ = {kap}");
        }
    }

    #[test]
    fn movmf_auto_picks_component_count() {
        let mut rng = SplitMix64::new(3);
        let (feats, truth) = unit_blobs(&mut rng, 12, 3, 70, 0.15);
        let res = movmf_auto(&feats, 1, 6, 100, 1);
        assert_eq!(res.means.len(), 3, "BIC should recover 3 components");
        assert!(ari(&res.labels, &truth) > 0.9);
    }

    /// Leaves that AGGREGATE many points (so `‖μ_i‖ < 1`). Regression guard: re-normalizing each
    /// leaf to a unit direction would discard the within-leaf spread, over-estimate `κ`, and make
    /// BIC fragment the mixture into far more than the true 3 components.
    #[test]
    fn movmf_handles_aggregated_microclusters() {
        let mut rng = SplitMix64::new(21);
        let dim = 12;
        let dirs: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                let mut v: Vec<f64> = (0..dim).map(|_| rng.gauss()).collect();
                let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                for x in &mut v {
                    *x /= nrm;
                }
                v
            })
            .collect();
        let mut feats = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in dirs.iter().enumerate() {
            for _ in 0..20 {
                let mut f = Spherical::new(dim);
                for _ in 0..25 {
                    let mut p: Vec<f64> = ctr.iter().map(|&m| m + 0.35 * rng.gauss()).collect();
                    let nrm = p.iter().map(|x| x * x).sum::<f64>().sqrt();
                    for x in &mut p {
                        *x /= nrm;
                    }
                    f.push(&p, 1.0);
                }
                feats.push(f);
                truth.push(c);
            }
        }
        let res = movmf_auto(&feats, 1, 8, 100, 1);
        assert_eq!(
            res.means.len(),
            3,
            "BIC must not over-fragment aggregated leaves"
        );
        assert!(
            ari(&res.labels, &truth) > 0.9,
            "ARI = {}",
            ari(&res.labels, &truth)
        );
        for &kap in &res.kappas {
            assert!(kap.is_finite() && kap < 9_000.0, "κ over-estimated: {kap}");
        }
    }

    /// `log C_3(κ) = ln κ − ln(4π sinh κ)` — the three-dimensional vMF normaliser in closed form,
    /// derived from the density rather than from [`log_vmf_norm`]'s own expression.
    #[test]
    fn log_vmf_norm_matches_the_closed_form_on_the_sphere() {
        for &kap in &[0.3_f64, 1.0, 4.0, 12.0] {
            let want = kap.ln() - (4.0 * std::f64::consts::PI).ln() - kap.sinh().ln();
            let got = log_vmf_norm(3, kap);
            assert!((got - want).abs() < 1e-9, "κ={kap}: got {got}, want {want}");
        }
    }

    #[test]
    fn argmin_keeps_the_first_of_equal_scores() {
        assert_eq!(argmin(&[3.0, 1.0, 2.0]), 1);
        assert_eq!(argmin(&[1.0, 1.0, 5.0]), 0);
        assert_eq!(argmin(&[5.0, 2.0, 2.0]), 1);
        assert_eq!(argmin(&[7.0]), 0);
    }

    #[test]
    fn cohesion_is_the_summed_length_of_the_weighted_resultants() {
        // Cluster 0 gets 2·[1,0] and 3·[0,1] → resultant [2,3], length √13.
        // Cluster 1 gets 4·[0.6,0.8] → resultant [2.4,3.2], length 4.
        let means = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.8]];
        let n = vec![2.0, 3.0, 4.0];
        let got = cohesion_of(&means, &n, &[0, 0, 1], 2, 2);
        let want = 13.0_f64.sqrt() + 4.0;
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    /// Independent re-derivation of [`movmf_once`], written from the movMF EM equations. The
    /// end-to-end tests assert an ARI over well-separated directions, which stays above 0.9 even
    /// when the E-step or the M-step is arithmetically wrong.
    ///
    /// Returns `(resp, weights, means, kappas, loglik, iterations)`.
    #[allow(clippy::type_complexity)]
    fn reference_movmf_em(
        features: &[Spherical<f64>],
        k: usize,
        max_iter: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>, f64, usize) {
        let (mu, n) = leaf_means(features);
        let dim = features[0].dim();
        let m = mu.len();
        let km = spherical_kmeans(features, k, 50, 1, seed);
        let mut means = km.centers.clone();
        let mut weights = vec![1.0 / k as f64; k];
        let mut kappas: Vec<f64> = init_kappas(&mu, &n, &km.labels, k, dim);

        let mut resp = vec![vec![0.0; k]; m];
        let mut loglik = f64::NEG_INFINITY;
        let mut iters = 0;

        for it in 0..max_iter {
            iters = it + 1;
            let mut new_ll = 0.0;
            for i in 0..m {
                let logr: Vec<f64> = (0..k)
                    .map(|c| {
                        let dv: f64 = (0..dim).map(|d| mu[i][d] * means[c][d]).sum();
                        weights[c].max(1e-300).ln()
                            + log_vmf_norm(dim, kappas[c].max(1e-8))
                            + kappas[c] * dv
                    })
                    .collect();
                let mx = logr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = mx + logr.iter().map(|&l| (l - mx).exp()).sum::<f64>().ln();
                new_ll += n[i] * lse;
                for c in 0..k {
                    resp[i][c] = (logr[c] - lse).exp();
                }
            }

            let mut rvec = vec![vec![0.0; dim]; k];
            let mut nk = vec![0.0; k];
            for i in 0..m {
                for c in 0..k {
                    let w = n[i] * resp[i][c];
                    nk[c] += w;
                    for d in 0..dim {
                        rvec[c][d] += w * mu[i][d];
                    }
                }
            }
            let ntot: f64 = nk.iter().sum();
            for c in 0..k {
                weights[c] = if ntot > 0.0 {
                    nk[c] / ntot
                } else {
                    1.0 / k as f64
                };
                let rnorm = rvec[c].iter().map(|x| x * x).sum::<f64>().sqrt();
                if rnorm > 0.0 {
                    for d in 0..dim {
                        means[c][d] = rvec[c][d] / rnorm;
                    }
                }
                let rbar = if nk[c] > 0.0 { rnorm / nk[c] } else { 0.0 };
                kappas[c] = estimate_kappa(rbar, dim);
            }

            if it > 0 && (new_ll - loglik).abs() <= 1e-7 * loglik.abs().max(1.0) {
                loglik = new_ll;
                break;
            }
            loglik = new_ll;
        }
        (resp, weights, means, kappas, loglik, iters)
    }

    #[track_caller]
    fn assert_close_rel(got: f64, want: f64, tol: f64, what: &str) {
        let scale = got.abs().max(want.abs()).max(1.0);
        assert!(
            (got - want).abs() <= tol * scale,
            "{what}: got {got}, want {want}"
        );
    }

    /// Directions close enough together that every responsibility stays strictly inside `(0, 1)`.
    /// Well-separated directions saturate the posterior and hide the arithmetic under it.
    fn overlapping_directions() -> Vec<Spherical<f64>> {
        let mut rng = SplitMix64::new(1234);
        unit_blobs(&mut rng, 6, 3, 40, 0.55).0
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index form mirrors the movMF equations
    fn movmf_em_matches_an_independent_reference_iteration_for_iteration() {
        let feats = overlapping_directions();
        let (k, iters, seed) = (3, 4, 5);
        let got: Movmf<f64> = movmf_once(&feats, k, iters, seed);
        let (rresp, rweights, rmeans, rkappas, rloglik, ran) =
            reference_movmf_em(&feats, k, iters, seed);
        assert_eq!(
            ran, iters,
            "fixture converged early, so the path is untested"
        );

        assert_close_rel(got.loglik, rloglik, 1e-9, "loglik");
        let dim = feats[0].dim();
        let mut soft = 0;
        for c in 0..k {
            assert_close_rel(got.weights[c], rweights[c], 1e-9, "weight");
            assert_close_rel(got.kappas[c], rkappas[c], 1e-9, "kappa");
            for d in 0..dim {
                assert_close_rel(got.means[c][d], rmeans[c][d], 1e-9, "mean");
            }
        }
        for i in 0..rresp.len() {
            for c in 0..k {
                assert_close_rel(got.resp[i][c], rresp[i][c], 1e-9, "resp");
                if got.resp[i][c] > 1e-3 && got.resp[i][c] < 1.0 - 1e-3 {
                    soft += 1;
                }
            }
            assert_eq!(got.labels[i], argmax(&rresp[i]), "label disagrees");
        }
        assert!(soft > 20, "fixture is not soft enough: {soft} soft entries");
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index form mirrors the movMF equations
    fn movmf_em_stops_on_the_relative_loglik_test() {
        let feats = overlapping_directions();
        let (k, seed) = (3, 5);
        let got: Movmf<f64> = movmf_once(&feats, k, 500, seed);
        let (_, _, _, rkappas, rloglik, ran) = reference_movmf_em(&feats, k, 500, seed);
        assert!(ran > 3 && ran < 500, "expected convergence, ran {ran}");
        assert_close_rel(got.loglik, rloglik, 1e-6, "converged loglik");
        for c in 0..k {
            assert_close_rel(got.kappas[c], rkappas[c], 1e-4, "converged kappa");
        }
        // movMF EM ascends the same likelihood as any other EM: no iteration may lower it.
        let mut prev = f64::NEG_INFINITY;
        for t in 1..=10 {
            let ll = movmf_once::<f64, _>(&feats, k, t, seed).loglik;
            assert!(
                ll >= prev - 1e-9,
                "loglik fell at iteration {t}: {prev} -> {ll}"
            );
            prev = ll;
        }
    }

    #[test]
    fn unit_normalizes_and_leaves_a_zero_vector_alone() {
        let u = unit(&[3.0f64, 4.0]);
        assert!(
            (u[0] - 0.6).abs() < 1e-15 && (u[1] - 0.8).abs() < 1e-15,
            "{u:?}"
        );
        // There is no direction to return, and dividing by the norm would hand back two NaNs.
        assert_eq!(unit(&[0.0f64, 0.0, 0.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn argmax_keeps_the_first_of_equal_scores() {
        assert_eq!(argmax(&[1.0f64, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0f64, 5.0]), 0);
        assert_eq!(argmax(&[-2.0f64, -1.0]), 1);
    }

    #[test]
    fn weighted_pick_is_proportional_and_never_leaves_the_slice() {
        let mut rng = SplitMix64::new(13);
        let mut hits = [0usize; 3];
        for _ in 0..6000 {
            hits[weighted_pick(&[2.0, 0.0, 4.0], &mut rng)] += 1;
        }
        assert_eq!(hits[1], 0, "a zero-probability entry was drawn");
        let ratio = hits[2] as f64 / hits[0] as f64;
        assert!((1.7..2.3).contains(&ratio), "0:2 ratio {ratio}, want 2");

        // A degenerate total falls back to a uniform draw, which must still be an index.
        let mut seen = [false; 4];
        for _ in 0..400 {
            let i = weighted_pick(&[0.0; 4], &mut rng);
            assert!(i < 4, "index {i} out of range");
            seen[i] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "uniform fallback never reached some index"
        );

        // The other fallback: `t` is `u · Σw` and the loop subtracts the terms one at a time, so
        // when the sum is not representable the running remainder need never cross zero and the
        // loop ends without returning. What it falls back to has to be an index -- the caller uses
        // it to subscript, and one past the end is a panic rather than a slightly wrong draw.
        let overflow = [f64::MAX, f64::MAX, f64::MAX];
        assert!(
            overflow.iter().sum::<f64>().is_infinite(),
            "the fixture's total is finite, so the loop still returns from inside"
        );
        for _ in 0..8 {
            let i = weighted_pick(&overflow, &mut rng);
            assert!(
                i < overflow.len(),
                "index {i} is past the end of a 3-slot slice"
            );
        }
    }

    /// Angular k-means++ re-derived: the first centre is drawn ∝ weight and normalized, then each
    /// further one ∝ `weight · (1 − ĉ·μ̂)` against the closest centre so far. It shares
    /// [`weighted_pick`], so it consumes the same rng stream.
    fn reference_spherical_pp(
        means: &[Vec<f64>],
        n: &[f64],
        k: usize,
        rng: &mut SplitMix64,
    ) -> Vec<Vec<f64>> {
        let ang = |a: &[f64], c: &[f64]| {
            (1.0 - a.iter().zip(c).map(|(x, y)| x * y).sum::<f64>()).max(0.0)
        };
        let mut centers = vec![unit(&means[weighted_pick(n, rng)])];
        let mut d2: Vec<f64> = means.iter().map(|x| ang(x, &centers[0])).collect();
        while centers.len() < k {
            let probs: Vec<f64> = n.iter().zip(&d2).map(|(&w, &d)| w * d).collect();
            let c = unit(&means[weighted_pick(&probs, rng)]);
            for (di, x) in d2.iter_mut().zip(means) {
                *di = di.min(ang(x, &c));
            }
            centers.push(c);
        }
        centers
    }

    #[test]
    fn spherical_seeding_matches_the_angular_reference_draw_for_draw() {
        // Only the seeds are observable downstream, and on separated directions every sampler
        // returns the same ones -- so the fixture is deliberately crowded and the assertion is on
        // the exact sequence, with the rng streams asserted to end in the same place.
        let mut rng = SplitMix64::new(55);
        let (feats, _truth) = unit_blobs(&mut rng, 5, 4, 25, 0.75);
        let means: Vec<Vec<f64>> = feats.iter().map(|f| unit(f.mean())).collect();
        let n: Vec<f64> = feats.iter().map(|f| f.weight()).collect();
        for k in [2usize, 4, 6] {
            let mut a = SplitMix64::new(808);
            let mut b = SplitMix64::new(808);
            assert_eq!(
                spherical_pp(&means, &n, k, &mut a),
                reference_spherical_pp(&means, &n, k, &mut b),
                "k = {k}"
            );
            assert_eq!(a.next_u64(), b.next_u64(), "k = {k}: rng streams diverged");
        }
    }

    #[test]
    fn more_restarts_never_return_less_cohesion() {
        // One rng stream feeds the restarts in order, so `n_init = m + 1` sees exactly the seeds
        // `n_init = m` saw plus one more: keeping the best can only raise the cohesion.
        let mut rng = SplitMix64::new(6);
        let (feats, _truth) = unit_blobs(&mut rng, 6, 5, 30, 0.7);
        let mut prev = f64::NEG_INFINITY;
        let mut improvements = 0;
        for n_init in 1..=10 {
            let got = spherical_kmeans(&feats, 5, 100, n_init, 4).cohesion;
            assert!(got >= prev - 1e-9, "n_init = {n_init}: {got} < {prev}");
            if got > prev + 1e-9 {
                improvements += 1;
            }
            prev = got;
        }
        assert!(
            improvements > 1,
            "every restart found the same optimum; the fixture cannot see the choice"
        );
    }

    /// `κ_c` re-derived from Banerjee et al. (2005): the resultant is built by *filtering* the
    /// leaves that carry the label, so nothing is accumulated into an indexed slot, and the estimate
    /// `R̄(d − R̄²)/(1 − R̄²)` is written out rather than called. Returns one κ per component.
    fn reference_init_kappas(
        mu: &[Vec<f64>],
        n: &[f64],
        labels: &[usize],
        k: usize,
        dim: usize,
    ) -> Vec<f64> {
        (0..k)
            .map(|c| {
                let members: Vec<usize> = (0..mu.len()).filter(|&i| labels[i] == c).collect();
                let mass: f64 = members.iter().map(|&i| n[i]).sum();
                let resultant: Vec<f64> = (0..dim)
                    .map(|d| members.iter().map(|&i| n[i] * mu[i][d]).sum())
                    .collect();
                let rbar = if mass > 0.0 {
                    resultant.iter().map(|v| v * v).sum::<f64>().sqrt() / mass
                } else {
                    0.0
                };
                let r = rbar.clamp(1e-8, 1.0 - 1e-9);
                ((r * (dim as f64 - r * r)) / (1.0 - r * r)).clamp(1e-8, KAPPA_MAX)
            })
            .collect()
    }

    #[test]
    fn init_kappas_matches_an_independent_reference() {
        // κ seeds every component's EM, and the resultant it comes from is a weighted sum -- so the
        // fixture gives every leaf a distinct weight and spreads each component's directions by a
        // different amount, or a concentration read off the wrong resultant lands on the right
        // answer anyway. Component 2 is deliberately left empty: its mass is exactly zero, and
        // dividing the resultant by it would return NaN through a value the caller cannot check.
        let dim = 3usize;
        let mu: Vec<Vec<f64>> = [
            [1.0, 0.0, 0.0],
            [0.995, 0.100, 0.0],
            [0.980, 0.199, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.707, 0.707],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|v| v.to_vec())
        .collect();
        let n = vec![3.0, 11.0, 2.0, 7.0, 5.0, 13.0];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let k = 3;

        let want = reference_init_kappas(&mu, &n, &labels, k, dim);
        assert!(
            !labels.contains(&2),
            "the fixture gave component 2 a member, so it cannot see the empty-mass guard"
        );
        assert!(
            want[0] > 2.0 * want[1],
            "the two populated components have the same concentration ({want:?}); \
             the fixture cannot tell one resultant from another"
        );
        assert!(
            want.iter().all(|k| k.is_finite() && *k < KAPPA_MAX),
            "a reference κ sits on the clamp, where any resultant gives the same answer: {want:?}"
        );

        let got = init_kappas(&mu, &n, &labels, k, dim);
        for (c, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                "component {c}: got {g}, want {w}"
            );
        }
    }

    #[test]
    fn a_component_whose_resultant_cancels_keeps_a_unit_direction() {
        // Antipodal leaves of equal mass sum to the zero vector exactly, so the mean-direction
        // update has nothing to normalize by. Dividing anyway gives 0/0, and the NaN direction then
        // poisons every later E-step and the reported log-likelihood with it.
        let dim = 2;
        let feats: Vec<Spherical<f64>> = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
            .iter()
            .map(|p| {
                let mut f = Spherical::new(dim);
                f.push(p, 1.0);
                f
            })
            .collect();
        let (mu, n) = leaf_means(&feats);
        let resultant: Vec<f64> = (0..dim)
            .map(|d| mu.iter().zip(&n).map(|(v, w)| w * v[d]).sum())
            .collect();
        assert!(
            resultant.iter().all(|&v| v == 0.0),
            "the fixture's resultant is {resultant:?}, not exactly zero, so it cannot see the guard"
        );

        let got: Movmf<f64> = movmf_once(&feats, 1, 10, 5);
        assert!(
            got.loglik.is_finite(),
            "the log-likelihood is {}",
            got.loglik
        );
        let nrm = norm(&got.means[0]);
        assert!(
            (nrm - 1.0).abs() < 1e-12,
            "the mean direction is no longer a unit vector: {:?}",
            got.means[0]
        );
    }

    #[test]
    fn movmf_keeps_the_best_restart_and_not_the_last() {
        // EM is non-convex, which is the whole reason for `MOVMF_N_INIT` restarts; keeping whichever
        // one happened to run last would spend four fits and report a random draw from them. The
        // restart seeds are derived from the caller's, so the individual runs are reproducible here
        // and the winner can be named rather than inferred.
        let mut rng = SplitMix64::new(31);
        let (feats, _truth) = unit_blobs(&mut rng, 4, 4, 20, 0.55);
        let lls: Vec<f64> = (0..MOVMF_N_INIT)
            .map(|s| {
                movmf_once::<f64, _>(
                    &feats,
                    4,
                    100,
                    17u64.wrapping_add(s.wrapping_mul(0x9E37_79B9)),
                )
                .loglik
            })
            .collect();
        let best = lls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            *lls.last().unwrap() < best,
            "the last restart is already the best one ({lls:?}); the fixture cannot see the choice"
        );

        let got: Movmf<f64> = movmf(&feats, 4, 100, 17);
        assert_eq!(got.loglik, best, "restarts were {lls:?}");
    }

    #[test]
    fn movmf_auto_minimises_an_independently_counted_bic() {
        // `p = k(d + 1) + (k - 1)`: a mean direction and a concentration per component, plus the
        // free mixing weights. Every rival count below differs from it by one or two parameters per
        // component -- one or two `ln n` of penalty -- so it can only decide where the likelihood an
        // extra component buys is the same order as what it costs. Sweeping the *separation* walks
        // the fixture through that regime; sweeping the dimension does not, because the difference
        // between these counts is constant in `d`.
        let dim = 6usize;
        /// A rival free-parameter count, in `(k, d)`.
        type ParamCount = fn(usize, usize) -> usize;
        let rivals: [(&str, ParamCount); 3] = [
            ("k·d + (k-1)", |k, d| k * d + (k - 1)),
            ("k(d-1) + (k-1)", |k, d| k * (d - 1) + (k - 1)),
            ("k(d+1) - (k-1)", |k, d| k * (d + 1) - (k - 1)),
        ];
        let mut discriminating = [false; 3];
        let mut chosen = Vec::new();
        for spread in [0.30f64, 0.45, 0.60, 0.75, 0.90] {
            let mut rng = SplitMix64::new(70);
            let (feats, _truth) = unit_blobs(&mut rng, dim, 3, 22, spread);
            let ntot: f64 = feats.iter().map(|f| f.weight()).sum();
            let ll: Vec<f64> = (1..=5)
                .map(|k| movmf::<f64, _>(&feats, k, 100, 9).loglik)
                .collect();
            let pick = |params: &dyn Fn(usize) -> usize| {
                (1..=5)
                    .min_by(|&a, &b| {
                        let s = |k: usize| -2.0 * ll[k - 1] + params(k) as f64 * ntot.ln();
                        s(a).partial_cmp(&s(b)).unwrap()
                    })
                    .unwrap()
            };
            let want = pick(&|k| k * (dim + 1) + (k - 1));
            for (i, (_, f)) in rivals.iter().enumerate() {
                if pick(&|k| f(k, dim)) != want {
                    discriminating[i] = true;
                }
            }

            let got: Movmf<f64> = movmf_auto(&feats, 1, 5, 100, 9);
            assert!(
                (got.loglik - ll[want - 1]).abs() <= 1e-9 * ll[want - 1].abs().max(1.0),
                "spread {spread}: selected k is not the argmin (k = {want})"
            );
            chosen.push(want);
        }
        for (i, (name, _)) in rivals.iter().enumerate() {
            assert!(
                discriminating[i],
                "no separation in the sweep lets `{name}` choose differently, so the penalty \
                 accompanies the choice here rather than deciding it"
            );
        }
        assert!(
            chosen.windows(2).any(|w| w[0] != w[1]),
            "the sweep never changes its mind, so it crosses no boundary: {chosen:?}"
        );
    }

    /// Independent re-derivation of [`spherical_lloyd`], written from the max-cosine assignment and
    /// the resultant-normalization update rather than from the production loop: the assignment
    /// materializes the whole dot-product row before choosing, the empty-cluster test asks whether
    /// any label mentions the cluster instead of counting members, and the accumulation is a
    /// separate pass. Returns `(labels, centers, cohesion)`.
    fn reference_spherical_lloyd(
        means: &[Vec<f64>],
        n: &[f64],
        init: &[Vec<f64>],
        max_iter: usize,
    ) -> (Vec<usize>, Vec<Vec<f64>>, f64) {
        let (m, k, dim) = (means.len(), init.len(), means[0].len());
        let resultants = |labels: &[usize]| {
            let mut acc = vec![vec![0.0; dim]; k];
            for (i, &c) in labels.iter().enumerate() {
                for d in 0..dim {
                    acc[c][d] += n[i] * means[i][d];
                }
            }
            acc
        };
        let length = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let direction = |v: &[f64]| {
            let l = length(v);
            if l > 0.0 {
                v.iter().map(|x| x / l).collect()
            } else {
                v.to_vec()
            }
        };

        let mut centers: Vec<Vec<f64>> = init.to_vec();
        let mut labels = vec![0usize; m];
        for it in 0..max_iter.max(1) {
            let mut changed = false;
            let mut served = vec![f64::NEG_INFINITY; m];
            for (i, mu) in means.iter().enumerate() {
                let row: Vec<f64> = centers
                    .iter()
                    .map(|c| c.iter().zip(mu).map(|(a, b)| a * b).sum())
                    .collect();
                let mut best = 0usize;
                for c in 1..k {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                served[i] = row[best];
                if labels[i] != best {
                    labels[i] = best;
                    changed = true;
                }
            }
            if !changed && it > 0 {
                break;
            }
            let acc = resultants(&labels);
            for c in 0..k {
                if !labels.contains(&c) {
                    let mut worst = 0usize;
                    for i in 1..m {
                        if served[i] < served[worst] {
                            worst = i;
                        }
                    }
                    centers[c] = direction(&means[worst]);
                    served[worst] = f64::INFINITY;
                    continue;
                }
                if length(&acc[c]) > 0.0 {
                    centers[c] = direction(&acc[c]);
                }
            }
        }
        let cohesion = resultants(&labels).iter().map(|a| length(a)).sum();
        (labels, centers, cohesion)
    }

    /// A fixture whose weights are all distinct and whose directions straddle two seeds, so the
    /// resultant depends on *which* leaf carries *which* weight — an accumulation that adds the
    /// weight instead of multiplying by it lands somewhere else.
    fn lloyd_fixture() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
        let means: Vec<Vec<f64>> = [
            [0.980, 0.199],
            [0.940, 0.342],
            [0.766, 0.643],
            [0.174, 0.985],
            [-0.259, 0.966],
            [-0.707, 0.707],
        ]
        .iter()
        .map(|v| v.to_vec())
        .collect();
        let n = vec![3.0, 11.0, 2.0, 7.0, 5.0, 13.0];
        let init = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        (means, n, init)
    }

    #[test]
    fn spherical_lloyd_matches_an_independent_reference() {
        let (means, n, init) = lloyd_fixture();
        let got = spherical_lloyd(&means, &n, init.clone(), 50, 2);
        let (want_labels, want_centers, want_cohesion) =
            reference_spherical_lloyd(&means, &n, &init, 50);

        assert!(
            want_labels.iter().any(|&c| c != want_labels[0]),
            "the fixture put every leaf in one cluster; it cannot see the update"
        );
        assert!(
            want_centers
                .iter()
                .zip(&init)
                .any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6)),
            "the fixture never moved a center; it cannot see the accumulation"
        );

        assert_eq!(got.labels, want_labels);
        for (c, (a, b)) in got.centers.iter().zip(&want_centers).enumerate() {
            for (d, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() < 1e-12, "center {c}[{d}]: got {x}, want {y}");
            }
        }
        assert!(
            (got.cohesion - want_cohesion).abs() < 1e-12,
            "cohesion: got {}, want {want_cohesion}",
            got.cohesion
        );
    }

    /// Leaves spread evenly along half of the unit circle, with all three centers seeded within six
    /// degrees of one end. A continuum has no gap for the first update to snap to, so each round
    /// moves the boundaries only partway and the labelling is still shifting after the second
    /// assignment -- the only regime in which "stop once nothing changed" differs from "stop as soon
    /// as something does".
    fn slow_lloyd_fixture() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
        let at = |deg: f64| {
            let r: f64 = deg.to_radians();
            vec![r.cos(), r.sin()]
        };
        let means: Vec<Vec<f64>> = (0..24).map(|i| at(7.5 * i as f64)).collect();
        let n: Vec<f64> = (0..24).map(|i| 1.0 + 0.05 * i as f64).collect();
        (means, n, vec![at(2.0), at(4.0), at(6.0)])
    }

    #[test]
    fn spherical_lloyd_runs_until_the_labelling_stops_moving() {
        let (means, n, init) = slow_lloyd_fixture();
        let k = init.len();
        let early = spherical_lloyd(&means, &n, init.clone(), 2, 2);
        let got = spherical_lloyd(&means, &n, init, 50, 2);
        assert_ne!(
            early.labels, got.labels,
            "the fixture settles within two assignments, so it cannot tell a loop that stops when \
             nothing changed from one that stops as soon as something does"
        );

        // Both halves of the iteration are stationary at a Lloyd fixed point: every leaf sits on its
        // largest-cosine center, and every center is the normalized weighted resultant of its own
        // members. A run that stopped early still holds centers built from a labelling it has left.
        for (i, mu) in means.iter().enumerate() {
            let mut best = 0;
            for c in 1..k {
                if dot(mu, &got.centers[c]) > dot(mu, &got.centers[best]) {
                    best = c;
                }
            }
            assert_eq!(got.labels[i], best, "leaf {i} is not on its nearest center");
        }
        for c in 0..k {
            let members: Vec<usize> = (0..means.len()).filter(|&i| got.labels[i] == c).collect();
            assert!(
                !members.is_empty(),
                "component {c} is empty, so its center was reseeded rather than averaged"
            );
            let mut acc = vec![0.0f64; 2];
            for &i in &members {
                for (d, a) in acc.iter_mut().enumerate() {
                    *a += n[i] * means[i][d];
                }
            }
            let want = unit(&acc);
            for (d, w) in want.iter().enumerate() {
                assert!(
                    (got.centers[c][d] - w).abs() < 1e-12,
                    "center {c} is not the resultant of its members: {:?} vs {want:?}",
                    got.centers[c]
                );
            }
        }
    }

    #[test]
    fn a_first_pass_that_changes_nothing_still_reseeds_the_empty_cluster() {
        // Labels start at zero, so leaves that all prefer the first center leave the opening
        // assignment with nothing changed. That is the one round whose update has to run regardless,
        // because it is what fills the second center -- returning here hands back the seed.
        let means: Vec<Vec<f64>> = [[1.0, 0.0], [0.940, 0.342], [0.766, 0.643], [0.174, 0.985]]
            .iter()
            .map(|v| v.to_vec())
            .collect();
        let n = vec![3.0, 11.0, 2.0, 7.0];
        let init = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];

        let first = spherical_lloyd(&means, &n, init.clone(), 1, 2);
        assert!(
            first.labels.iter().all(|&c| c == 0),
            "a leaf left cluster 0 on the opening pass, so the fixture cannot see a round that \
             changed nothing: {:?}",
            first.labels
        );

        let got = spherical_lloyd(&means, &n, init.clone(), 50, 2);
        assert!(
            got.labels.contains(&1),
            "the second cluster stayed empty, so the run returned its seed rather than a fit"
        );
        assert!(
            (got.centers[1][0] - init[1][0]).abs() > 1e-9,
            "the second center is still its seed: {:?}",
            got.centers[1]
        );
    }

    #[test]
    fn spherical_lloyd_breaks_an_exact_dot_tie_towards_the_first_center() {
        // `[s, s]` with `s = 1/√2` has dot `s` with both axis centers, bit for bit: the products
        // are `s·1 + s·0` and `s·0 + s·1`. Only the comparison decides, so `>` (first wins) and
        // `>=` (last wins) put the leaf in different clusters.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let means = vec![vec![s, s], vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = vec![1.0, 1.0, 1.0];
        let init = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let a: f64 = means[0].iter().zip(&init[0]).map(|(x, y)| x * y).sum();
        let b: f64 = means[0].iter().zip(&init[1]).map(|(x, y)| x * y).sum();
        assert_eq!(a.to_bits(), b.to_bits(), "the fixture is not an exact tie");

        let got = spherical_lloyd(&means, &n, init, 1, 2);
        assert_eq!(got.labels, vec![0, 0, 1]);
    }

    #[test]
    fn spherical_lloyd_always_runs_one_update_pass() {
        // Both leaves already sit on center 0 and the labels start there, so the first assignment
        // changes nothing. Skipping the update on that basis would return the seed untouched and
        // leave cluster 1 anti-aligned; the loop must still reseed it from the worst-served leaf.
        let means = vec![vec![1.0, 0.0], vec![0.6, 0.8]];
        let n = vec![1.0, 1.0];
        let init = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];

        let got = spherical_lloyd(&means, &n, init.clone(), 20, 2);
        let (want_labels, want_centers, _) = reference_spherical_lloyd(&means, &n, &init, 20);
        assert_eq!(got.labels, want_labels);
        assert!(
            got.centers[1][0] > 0.0,
            "cluster 1 kept its anti-aligned seed {:?}; the update pass was skipped",
            got.centers[1]
        );
        for (a, b) in got
            .centers
            .iter()
            .flatten()
            .zip(want_centers.iter().flatten())
        {
            assert!((a - b).abs() < 1e-12, "got {a}, want {b}");
        }
    }

    #[test]
    fn spherical_lloyd_keeps_a_center_whose_resultant_cancels() {
        // Two antipodal leaves of equal weight in one cluster sum to exactly `[0, 0]`, so the
        // normalization has nothing to divide by. The guard must leave the seed in place rather
        // than emit `0/0`.
        let means = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        let n = vec![4.0, 4.0];
        let init = vec![vec![1.0, 0.0]];

        let got = spherical_lloyd(&means, &n, init, 5, 2);
        assert!(
            got.centers[0].iter().all(|x: &f64| x.is_finite()),
            "the cancelled resultant escaped the zero guard: {:?}",
            got.centers[0]
        );
        assert_eq!(got.centers[0], vec![1.0, 0.0]);
        assert_eq!(got.cohesion, 0.0);
    }
}
