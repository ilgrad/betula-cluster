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
}
