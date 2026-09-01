//! Watson axial mixture — directions **without** a sign, on the scatter the leaves already carry.
//!
//! The von Mises-Fisher head in [`vmf`](super::vmf) models a direction: `p(x) ∝ exp(κ μᵀx)` puts
//! `−μ` at the far end of its density from `μ`. A great deal of directional data is not like that.
//! An eigenvector, a line's orientation, a fibre direction, a PCA axis, a symmetric-embedding
//! feature — all are defined only up to sign, and `x` and `−x` mean the same thing. Handed such
//! data, `movmf` spends half its components chasing the antipodes of the other half, and if the two
//! halves are equally populated its resultant `Σ n_i μ_i` cancels to nothing.
//!
//! The Watson distribution (Watson 1965; Mardia & Jupp 2000 §9.4.1) is the axial answer:
//!
//! ```text
//! p(x | μ, κ) = M(1/2, d/2, κ)^-1 * Γ(d/2) / (2 π^(d/2)) * exp(κ (μᵀx)²),   x, μ ∈ S^(d-1)
//! ```
//!
//! `(μᵀx)²` is invariant under `x ↦ −x`, which is the whole point. `M` is Kummer's confluent
//! hypergeometric `₁F₁`. `κ > 0` is **bipolar** — mass at both poles `±μ`; `κ < 0` is **girdle** —
//! mass on the equator orthogonal to `μ`, which is how a great-circle / co-planar structure is
//! described. Both signs ship.
//!
//! ## Why the leaf summary is enough, exactly
//!
//! The sufficient statistic is the second moment about the **origin**, `T = Σ_i w_i x_i x_iᵀ`, and a
//! cluster feature carries it: `E[x xᵀ] = Σ_i + μ_i μ_iᵀ` from the leaf's covariance and mean. So
//! the E-step term is available in closed form and *exactly*,
//!
//! ```text
//! E_{x ∈ leaf i}[(μ_cᵀ x)²] = μ_cᵀ (Σ_i + μ_i μ_iᵀ) μ_c
//! ```
//!
//! — the same expected-log E-step the Gaussian heads use, with the same status: tying the
//! responsibility within a leaf is the only approximation, and the within-leaf spread is not dropped
//! but integrated. The M-step needs `T_c = Σ_i r_ic n_i (Σ_i + μ_i μ_iᵀ)`, also exact, whose
//! dominant eigenvector is `μ̂_c` (smallest, for a girdle component).
//!
//! **`feature="full"`, and the margin over a diagonal leaf is narrower than that derivation makes it
//! sound.** What `Spherical` and `Diagonal` drop is the *off-diagonal* block of `Σ_i`, and that block
//! carries axial information only where a leaf straddles **both** poles of its axis — there the leaf
//! mean cancels toward zero and the off-diagonal scatter is the only thing left naming the axis.
//! Measured on a 32-D four-axis fixture, median of seeds 0/1/2: at `max_leaves ≥ 40` every leaf mean
//! already has unit norm and `spherical` / `diagonal` / `full` tie at ARI ≈ 0.95; at `max_leaves ≤ 12`,
//! where the median leaf-mean norm falls to ≈ 0.5, `full` leads by 0.10–0.15 — on a budget whose
//! absolute quality is poor either way. `fd` is the one to avoid when the budget is coarse: the
//! sketch's rank truncation costs it 0.2–0.55 there.
//!
//! ## The special function, and where its numbers come from
//!
//! The head needs `log M(1/2, d/2, κ)` for the normalizer and
//! `g(κ) = M'(1/2,d/2,κ)/M(1/2,d/2,κ)` for the concentration MLE, whose equation is `g(κ) = r̄` with
//! `r̄ = μ̂ᵀ T μ̂`. Two identities carry the implementation, both verified symbolically in Maxima
//! (`local/scratch/watson_kummer.mac`, residual exactly `0` as a series identity in `z` for symbolic
//! `a` and `b`) before any of this was written:
//!
//! - `M'(a,b,z) = (a/b) M(a+1,b+1,z)`, so `g(κ) = (1/d) M(3/2, d/2+1, κ) / M(1/2, d/2, κ)`;
//! - Kummer's transformation `M(a,b,z) = e^z M(b−a, b, −z)`, which is what makes `κ < 0` computable:
//!   the ascending series alternates for negative argument and loses every digit to cancellation,
//!   while for positive argument every term is positive and the sum is exact to roundoff.
//!
//! The leading asymptotic `M(a,b,κ) ≈ Γ(b)/Γ(a) e^κ κ^(a−b)` is **not** used: measured against
//! Maxima it is still 42 % low at `d = 50, κ = 50` and 1.2 % low at `κ = 1000`, so the series runs
//! everywhere and `κ` is capped at [`KAPPA_MAX`] instead, where its length is bounded. The unit
//! tests check `log M` and `g` against the Maxima table directly.
//!
//! `g` is strictly increasing with `g(0) = 1/d`, `g(−∞) = 0`, `g(+∞) = 1`, so `r̄ ≷ 1/d` fixes the
//! sign of `κ` before any solving and the solver cannot pick the wrong branch.

use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::mixture::Mixture;
use crate::types::Real;

/// Concentration cap. The ascending series has its peak term near `n ≈ κ`, so this is what bounds
/// its length; a component tighter than this is already a point on the sphere in any usable sense.
const KAPPA_MAX: f64 = 1e4;
/// EM restarts kept by best data log-likelihood — the same budget the vMF head uses.
const WATSON_N_INIT: u64 = 4;
/// Relative tolerance of the `κ` solve, as a fraction of the starting bracket.
const KAPPA_TOL: f64 = 1e-10;

// ───────────────────────── the confluent hypergeometric ─────────────────────────

/// `ln Γ(x)` for `x > 0` via Lanczos (g = 7) — the same coefficients [`super::vmf`] uses.
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
    0.5 * (std::f64::consts::TAU).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// `log M(a, b, z)` — the log of Kummer's `₁F₁(a; b; z)`, for `a, b > 0`.
///
/// For `z ≥ 0` this is the ascending series `Σ_n (a)_n/(b)_n z^n/n!`, accumulated through the term
/// ratio `t_{n+1}/t_n = (a+n)z / ((b+n)(n+1))`. Every term is positive, so there is no cancellation
/// and the only hazard is overflow — which one rescaling of the running sum removes, rather than a
/// logarithm per term: the peak sits near `n ≈ z` and the loop is `O(z)` long, so three `ln` calls
/// per term is what made the concentration solve the head's whole cost. For `z < 0` Kummer's
/// transformation moves the evaluation to the positive side rather than summing an alternating
/// series — the difference between "exact to roundoff" and "no correct digits at `κ = −20`".
pub(crate) fn log_kummer_m(a: f64, b: f64, z: f64) -> f64 {
    if z < 0.0 {
        return z + log_kummer_m(b - a, b, -z);
    }
    if z == 0.0 {
        return 0.0;
    }
    /// Rescale the accumulator here; `2^512` is exact and leaves ~500 binades of headroom either way.
    const BIG: f64 = 1.340_780_792_994_259_7e154; // 2^512
    let (mut t, mut sum, mut n, mut log_scale) = (1.0_f64, 1.0_f64, 0.0_f64, 0.0_f64);
    loop {
        t *= (a + n) * z / ((b + n) * (n + 1.0));
        sum += t;
        n += 1.0;
        if sum > BIG {
            sum /= BIG;
            t /= BIG;
            log_scale += BIG.ln();
        }
        // Past the peak (which sits near n ≈ z) and contributing nothing further.
        if n > z && t < 1e-18 * sum {
            break;
        }
    }
    log_scale + sum.ln()
}

/// `g(κ) = M'(1/2,d/2,κ)/M(1/2,d/2,κ)` — the mean resultant `r̄ = μᵀTμ` the concentration `κ`
/// implies. Strictly increasing, `g(0) = 1/d`, range `(0, 1)`.
fn kummer_ratio(dim: usize, kappa: f64) -> f64 {
    let d = dim as f64;
    (log_kummer_m(1.5, d / 2.0 + 1.0, kappa) - log_kummer_m(0.5, d / 2.0, kappa)).exp() / d
}

/// Invert `g` for the Watson MLE `κ̂`, on the branch `r̄` selects.
///
/// The sign is decided before the solve, not by it: `g(0) = 1/d` exactly, so `r̄ > 1/d` is bipolar
/// and `r̄ < 1/d` is girdle. A *bracketed* solver rather than Newton because `g` is monotone but very
/// flat far from the origin in high `d`, where a Newton step happily leaves the bracket.
pub(crate) fn solve_kappa(dim: usize, rbar: f64) -> f64 {
    let iso = 1.0 / dim as f64;
    let r = rbar.clamp(1e-12, 1.0 - 1e-12);
    if (r - iso).abs() < 1e-12 {
        return 0.0;
    }
    let (mut lo, mut hi) = if r > iso {
        (0.0, KAPPA_MAX)
    } else {
        (-KAPPA_MAX, 0.0)
    };
    if r > iso && kummer_ratio(dim, hi) <= r {
        return hi;
    }
    if r < iso && kummer_ratio(dim, lo) >= r {
        return lo;
    }
    // Illinois: regula falsi with the retained endpoint's residual halved, which is what stops the
    // stalling that plain false position suffers where `g` is flat. Every step stays inside the
    // bracket, so it keeps bisection's guarantee and not Newton's failure mode; every eighth step is
    // a bisection anyway, which bounds the bracket width and therefore the iteration count.
    let (mut flo, mut fhi) = (kummer_ratio(dim, lo) - r, kummer_ratio(dim, hi) - r);
    let mut step = 0usize;
    while hi - lo > KAPPA_TOL * KAPPA_MAX {
        let secant = hi - fhi * (hi - lo) / (fhi - flo);
        let mid = if step % 8 == 7 || !secant.is_finite() || secant <= lo || secant >= hi {
            0.5 * (lo + hi)
        } else {
            secant
        };
        let fmid = kummer_ratio(dim, mid) - r;
        if fmid < 0.0 {
            (lo, flo) = (mid, fmid);
            fhi *= 0.5;
        } else {
            (hi, fhi) = (mid, fmid);
            flo *= 0.5;
        }
        step += 1;
    }
    0.5 * (lo + hi)
}

/// `log` of the Watson normalizing constant `Γ(d/2) / (2 π^(d/2) M(1/2, d/2, κ))`.
fn log_watson_norm(dim: usize, kappa: f64) -> f64 {
    let d = dim as f64;
    ln_gamma(d / 2.0)
        - std::f64::consts::LN_2
        - (d / 2.0) * std::f64::consts::PI.ln()
        - log_kummer_m(0.5, d / 2.0, kappa)
}

// ───────────────────────── the head ─────────────────────────

/// A fitted Watson axial mixture over the leaf summary.
pub struct Watson<R: Real> {
    /// Hard label (argmax responsibility) per leaf.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[leaf][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_c`.
    pub weights: Vec<R>,
    /// Unit component **axes** `μ_c`. Sign is arbitrary — `μ_c` and `−μ_c` are the same component.
    pub axes: Vec<Vec<R>>,
    /// Component concentrations. Positive is bipolar, negative is girdle.
    pub kappas: Vec<R>,
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

fn unit<R: Real>(v: &[R]) -> Vec<R> {
    let n = dot(v, v).sqrt();
    if n > R::zero() {
        v.iter().map(|&x| x / n).collect()
    } else {
        let mut e = vec![R::zero(); v.len()];
        if !e.is_empty() {
            e[0] = R::one();
        }
        e
    }
}

fn argmax<R: Real>(v: &[R]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// `k`-means++ in the **axial** distance `1 − (u_i·μ_c)²`, which is what makes the seeding blind to
/// sign: a leaf sitting at the antipode of a chosen axis is at distance 0 from it, not 2.
///
/// The candidates are leaf mean *directions*, exactly as `k`-means++ draws its centres from the data.
/// That is also the limit: an axis no leaf's mean points along has to be reached by EM from a
/// neighbouring seed, so a summary whose leaves all share one mean direction — every leaf straddling
/// both poles of its own axis — offers this pass nothing to tell them apart. A leaf that coarse is
/// what a very small `max_leaves` produces, and the fix is leaf budget, not a different seeder.
fn axial_pp<R: Real>(u: &[Vec<R>], n: &[R], k: usize, rng: &mut SplitMix64) -> Vec<Vec<R>> {
    let m = u.len();
    let mut axes: Vec<Vec<R>> = Vec::with_capacity(k);
    let mut best = vec![f64::INFINITY; m];
    let first = (rng.next_u64() % m as u64) as usize;
    axes.push(u[first].clone());
    for _ in 1..k {
        let last = axes.last().unwrap();
        let mut total = 0.0;
        for i in 0..m {
            let c = dot(&u[i], last).to_f64().unwrap_or(0.0);
            let d = (1.0 - c * c).max(0.0) * n[i].to_f64().unwrap_or(0.0);
            best[i] = best[i].min(d);
            total += best[i];
        }
        let mut target = rng.next_f64() * total;
        let mut pick = m - 1;
        for (i, &w) in best.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                pick = i;
                break;
            }
        }
        axes.push(u[pick].clone());
    }
    axes
}

/// Fit a `k`-component Watson mixture, keeping the best of [`WATSON_N_INIT`] EM restarts.
pub fn watson<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Watson<R> {
    let mut best: Option<Watson<R>> = None;
    for s in 0..WATSON_N_INIT {
        let r = watson_once(
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
    best.expect("WATSON_N_INIT >= 1")
}

#[allow(clippy::needless_range_loop)] // EM over (leaf i, component c, dim d) is clearest indexed
fn watson_once<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Watson<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let m = features.len();
    let dim = features[0].dim();
    let n: Vec<R> = features.iter().map(|f| f.weight()).collect();
    let mu: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let u: Vec<Vec<R>> = mu.iter().map(|v| unit(v)).collect();

    let mut rng = SplitMix64::new(seed);
    let mut axes = axial_pp(&u, &n, k, &mut rng);
    let mut weights = vec![R::one() / R::from_usize(k).unwrap(); k];
    let mut kappas = vec![R::from_f64(solve_kappa(dim, 2.0 / dim as f64)).unwrap(); k];

    let mut resp = vec![vec![R::zero(); k]; m];
    let mut loglik = R::neg_infinity();
    let tol = R::from_f64(1e-7).unwrap();
    // Reused across leaves so the E-step allocates once, not once per leaf.
    let mut proj = vec![vec![R::zero(); dim]; k];
    let mut q = vec![0.0_f64; k];

    for it in 0..max_iter {
        let kf: Vec<f64> = kappas.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();
        let logc: Vec<f64> = kf.iter().map(|&x| log_watson_norm(dim, x)).collect();
        let lw: Vec<f64> = weights
            .iter()
            .map(|&w| w.to_f64().unwrap_or(0.0).max(1e-300).ln())
            .collect();

        // E-step. `q[c] = μ_cᵀ (Σ_i + μ_i μ_iᵀ) μ_c` is the exact within-leaf expectation of
        // `(μ_c·x)²`, so the leaf's scatter enters the responsibility rather than being dropped.
        let mut new_ll = R::zero();
        for i in 0..m {
            for row in proj.iter_mut() {
                row.iter_mut().for_each(|v| *v = R::zero());
            }
            features[i]
                .second_moment()
                .apply_rows(&axes, &mut proj, R::one());
            let mut logr = vec![0.0_f64; k];
            for c in 0..k {
                let cross = dot(&axes[c], &mu[i]).to_f64().unwrap_or(0.0);
                q[c] = dot(&axes[c], &proj[c]).to_f64().unwrap_or(0.0) + cross * cross;
                logr[c] = lw[c] + logc[c] + kf[c] * q[c];
            }
            let mx = logr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = mx + logr.iter().map(|&l| (l - mx).exp()).sum::<f64>().ln();
            new_ll = new_ll + n[i] * R::from_f64(lse).unwrap();
            for c in 0..k {
                resp[i][c] = R::from_f64((logr[c] - lse).exp()).unwrap();
            }
        }

        // M-step: `T_c = Σ_i r_ic n_i (Σ_i + μ_i μ_iᵀ) / Σ_i r_ic n_i`, then the axis is one end of
        // its spectrum and `κ` the MLE for that end's eigenvalue.
        let mut t = vec![vec![vec![R::zero(); dim]; dim]; k];
        let mut nk = vec![R::zero(); k];
        for i in 0..m {
            let sm = features[i].second_moment();
            for c in 0..k {
                let w = n[i] * resp[i][c];
                if w == R::zero() {
                    continue;
                }
                nk[c] = nk[c] + w;
                sm.add_scaled(&mut t[c], w);
                for a in 0..dim {
                    let wa = w * mu[i][a];
                    for b in 0..dim {
                        t[c][a][b] = t[c][a][b] + wa * mu[i][b];
                    }
                }
            }
        }
        let ntot: R = nk.iter().copied().fold(R::zero(), |a, b| a + b);
        for c in 0..k {
            weights[c] = if ntot > R::zero() {
                nk[c] / ntot
            } else {
                R::one() / R::from_usize(k).unwrap()
            };
            if nk[c] <= R::zero() {
                continue;
            }
            for row in t[c].iter_mut() {
                for v in row.iter_mut() {
                    *v = *v / nk[c];
                }
            }
            let (top, bottom) = extreme_eigenpairs(&t[c], &mut rng);
            let (axis, kappa) = better_pole(dim, top, bottom);
            axes[c] = axis;
            kappas[c] = R::from_f64(kappa).unwrap();
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
        .map(|&x| log_watson_norm(dim, x.to_f64().unwrap_or(0.0)))
        .collect();
    let mixture = Mixture::watson(&weights, &axes, &kappas, &logc);
    Watson {
        labels,
        resp,
        weights,
        axes,
        kappas,
        loglik,
        mixture,
    }
}

/// Both ends of a symmetric positive-semidefinite matrix's spectrum, without decomposing it.
///
/// The M-step needs the largest and the smallest eigenpair and nothing in between, and a full
/// Jacobi sweep is the most expensive route to that in the crate: it rotates the whole matrix to
/// diagonal in ~10 passes of `O(d³)`, where a power step costs `O(d²)`. Measured on the axial
/// fixture at `d = 256`, that difference *was* the head.
///
/// `σ = tr T` bounds every eigenvalue of a PSD `T` from above, so `σI − T` is PSD too and its
/// **largest** eigenvalue sits at `T`'s smallest — one routine answers both ends.
fn extreme_eigenpairs<R: Real>(
    t: &[Vec<R>],
    rng: &mut SplitMix64,
) -> ((f64, Vec<R>), (f64, Vec<R>)) {
    let sigma = (0..t.len())
        .map(|i| t[i][i].to_f64().unwrap_or(0.0))
        .sum::<f64>();
    let top = power_iterate(t, 0.0, rng);
    let (bottom_val, bottom_vec) = power_iterate(t, sigma, rng);
    (top, (sigma - bottom_val, bottom_vec))
}

/// Largest eigenpair of the PSD operator `shift·I − T` (`shift = 0` meaning `T` itself), by power
/// iteration from a **random** start.
///
/// Random and not a coordinate vector: a leaf summary is very often coordinate-aligned, and there
/// `e_j` is an exact eigenvector, so a coordinate start is a fixed point that never reaches the
/// dominant axis. The limit does not depend on the start, which is what keeps the head rotation
/// equivariant; the start only has to avoid being orthogonal to it, which a random draw is with
/// probability 1.
fn power_iterate<R: Real>(t: &[Vec<R>], shift: f64, rng: &mut SplitMix64) -> (f64, Vec<R>) {
    /// Steps allowed per end. The Rayleigh quotient settles far sooner on any spectrum with a gap,
    /// and a spectrum without one has no axis to find.
    const MAX_STEPS: usize = 500;
    /// Relative change in the eigenvalue estimate that counts as converged.
    const TOL: f64 = 1e-13;

    let d = t.len();
    let mut v: Vec<R> = (0..d)
        .map(|_| R::from_f64(rng.next_f64() - 0.5).unwrap())
        .collect();
    let mut w = vec![R::zero(); d];
    let mut lambda = f64::INFINITY;
    let shift = R::from_f64(shift).unwrap();
    for _ in 0..MAX_STEPS {
        let v_unit = unit(&v);
        for i in 0..d {
            let dot = dot(&t[i], &v_unit);
            w[i] = if shift == R::zero() {
                dot
            } else {
                shift * v_unit[i] - dot
            };
        }
        let norm = dot(&w, &w).sqrt().to_f64().unwrap_or(0.0);
        v = w.clone();
        if norm <= 0.0 {
            // A zero operator: every direction is an eigenvector, and `v_unit` is as good as any.
            return (0.0, v_unit);
        }
        if (norm - lambda).abs() <= TOL * norm {
            lambda = norm;
            break;
        }
        lambda = norm;
    }
    (lambda, unit(&v))
}

/// Pick the bipolar or the girdle solution, whichever fits better.
///
/// The Watson MLE has two stationary points — the axis of largest eigenvalue with `κ > 0`, and the
/// axis of smallest with `κ < 0` — and the model is not free to prefer one a priori. Both are
/// scored on the term of the expected complete log-likelihood that distinguishes them,
/// `κ r̄ − log M(1/2, d/2, κ)`, and the larger wins. Skipping this and always taking the top
/// eigenvector would make the head unable to fit the equatorial structure `κ < 0` exists for.
fn better_pole<R: Real>(dim: usize, top: (f64, Vec<R>), bottom: (f64, Vec<R>)) -> (Vec<R>, f64) {
    let score = |val: f64| -> (f64, f64) {
        let r = val.clamp(0.0, 1.0);
        let kappa = solve_kappa(dim, r);
        (
            kappa * r - log_kummer_m(0.5, dim as f64 / 2.0, kappa),
            kappa,
        )
    };
    let (s_hi, k_hi) = score(top.0);
    let (s_lo, k_lo) = score(bottom.0);
    if s_lo > s_hi {
        (bottom.1, k_lo)
    } else {
        (top.1, k_hi)
    }
}

/// Fit a Watson mixture with automatic component count by BIC over `k ∈ [k_min, k_max]`.
///
/// Free parameters per component: `d − 1` for the axis (a unit vector up to sign) plus 1 for `κ`,
/// and `k − 1` mixing weights.
pub fn watson_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> Watson<R> {
    let dim = features.first().map_or(0, |f| f.dim()) as f64;
    let ntot: f64 = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap_or(0.0))
        .sum();
    let hi = k_max.min(features.len()).max(k_min);
    let mut best: Option<(f64, Watson<R>)> = None;
    for k in k_min..=hi {
        let fit = watson(features, k, max_iter, seed);
        let p = k as f64 * dim + (k - 1) as f64;
        let bic = p * ntot.max(1.0).ln() - 2.0 * fit.loglik.to_f64().unwrap_or(0.0);
        match &best {
            Some((b, _)) if *b <= bic => {}
            _ => best = Some((bic, fit)),
        }
    }
    best.expect("k_min <= k_max").1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::testutil::ari;
    use crate::clustering::vmf::movmf;
    use crate::feature::{ClusterFeature, Full};

    /// `(d, κ, log M(1/2, d/2, κ), g(κ))` from Maxima at `fpprec: 40`, rounded to `f64` —
    /// `local/scratch/watson_kummer.mac`, section 4. Every `κ` there is an exact integer or
    /// rational: a float literal makes `hypergeometric` evaluate in double precision, and `bfloat`
    /// then pads the result out to 40 digits of which only ~15 are real. That is how the two
    /// `κ = ±1/4` rows were once up to 5 ulp wrong here.
    const GOLDEN: [(usize, f64, f64, f64); 24] = [
        (
            2,
            -500.0,
            -3.679_167_987_942_912_4,
            1.001_004_025_210_173e-3,
        ),
        (2, -20.0, -2.057_027_916_881_304_4, 2.570_008_702_257_702e-2),
        (
            2,
            -1.0,
            -4.384_502_808_145_187e-1,
            3.787_501_937_095_990_6e-1,
        ),
        (2, 1.0, 5.615_497_191_854_814e-1, 6.212_498_062_904_01e-1),
        (2, 100.0, 9.712_757_550_187_18e1, 9.949_744_836_892_489e-1),
        (3, -100.0, -2.423_367_330_629_291, 5e-3),
        (
            3,
            -0.25,
            -8.060_068_275_163_107e-2,
            3.116_565_125_283_536_4e-1,
        ),
        (
            3,
            0.25,
            8.615_403_391_270_206e-2,
            3.560_656_882_169_370_7e-1,
        ),
        (3, 5.0, 2.843_289_338_174_84, 7.642_662_212_704_322e-1),
        (3, 500.0, 4.930_932_472_334_403e2, 9.979_979_899_252_858e-1),
        (5, -5.0, -6.250_730_856_407_12e-1, 7.819_741_318_739_432e-2),
        (5, 1.0, 2.249_459_715_360_156e-1, 2.520_213_687_217_308_7e-1),
        (5, 20.0, 1.377_614_931_459_231e1, 8.969_074_069_413_289e-1),
        (
            10,
            -20.0,
            -8.567_110_352_587_666e-1,
            2.105_599_649_999_264_5e-2,
        ),
        (
            10,
            5.0,
            7.839_827_883_901_322e-1,
            2.377_414_514_343_994_5e-1,
        ),
        (
            10,
            500.0,
            4.746_444_797_135_067_7e2,
            9.909_908_903_790_221e-1,
        ),
        (
            20,
            -1.0,
            -4.794_383_864_852_173e-2,
            4.598_579_569_471_381e-2,
        ),
        (20, 20.0, 4.141_541_515_247_125, 4.927_736_323_814_999_6e-1),
        (20, 100.0, 6.853_070_024_836_767e1, 9.044_652_000_281_247e-1),
        (
            64,
            -500.0,
            -1.415_925_053_521_573_3,
            9.423_531_930_379_572e-4,
        ),
        (64, 5.0, 8.456_741_476_994_93e-2, 1.833_928_707_470_880_4e-2),
        (
            64,
            100.0,
            3.264_878_555_589_809_4e1,
            6.826_220_598_634_248e-1,
        ),
        (
            200,
            -100.0,
            -3.475_126_394_350_868_7e-1,
            2.509_422_034_699_247_2e-3,
        ),
        (
            200,
            500.0,
            2.403_195_156_891_966_3e2,
            8.007_503_886_287_254e-1,
        ),
    ];

    #[test]
    fn the_confluent_hypergeometric_matches_maxima_on_both_signs_of_kappa() {
        // Measured worst over the whole table on the unmutated code is 3.2e-14 for `log M` and
        // 1.5e-14 for `g`, both at d = 64, κ = -500 -- the deepest cancellation in the grid. The
        // bound is three times that, in the style of `stats.rs`. It was 1e-11, which is 300× the
        // residual and so blind to any regression in the series short of a gross one.
        //
        // What it still cannot see: a golden corrupted by a few ulp. The implementation's own
        // 3.2e-14 error is an order above the ~1e-15 such a corruption moves the target by, so no
        // tolerance separates the two. That failure has happened here -- two rows sat ~5 ulp wrong
        // -- and the guard against it is not this test but recomputing the table from an engine
        // other than the one that produced it (`local/scratch/recheck_goldens.py`, mpmath).
        for (d, kappa, log_m, g) in GOLDEN {
            let got = log_kummer_m(0.5, d as f64 / 2.0, kappa);
            assert!(
                (got - log_m).abs() <= 1e-13 * log_m.abs().max(1.0),
                "log M(1/2,{}/2,{kappa}) = {got}, Maxima says {log_m}",
                d
            );
            let gr = kummer_ratio(d, kappa);
            assert!(
                (gr - g).abs() <= 1e-13 * g,
                "g({d},{kappa}) = {gr}, Maxima says {g}"
            );
        }
    }

    #[test]
    fn kummers_transformation_is_what_makes_a_negative_kappa_computable() {
        // The alternating ascending series, written out here, is what the head would use without
        // the transformation. It agrees at small |z| and loses everything by z = -40.
        let naive = |a: f64, b: f64, z: f64| {
            let (mut term, mut sum) = (1.0_f64, 1.0_f64);
            for n in 0..2000 {
                let n = n as f64;
                term *= (a + n) / (b + n) * z / (n + 1.0);
                sum += term;
            }
            sum.ln()
        };
        assert!((naive(0.5, 1.5, -1.0) - log_kummer_m(0.5, 1.5, -1.0)).abs() < 1e-12);
        let exact = log_kummer_m(0.5, 1.5, -40.0);
        assert!((naive(0.5, 1.5, -40.0) - exact).abs() > 1e-3);
        // ...and the transformed value is the right one: M(1/2,3/2,-z) = sqrt(pi/(4z)) erf(sqrt z),
        // which at z = 40 is sqrt(pi/160) to every digit erf can spare.
        assert!((exact - (std::f64::consts::PI / 160.0).sqrt().ln()).abs() < 1e-12);
    }

    #[test]
    fn the_concentration_solve_inverts_its_own_ratio() {
        for d in [2usize, 3, 5, 20, 64, 200] {
            for kappa in [-800.0, -50.0, -2.0, -0.3, 0.3, 2.0, 50.0, 800.0] {
                let r = kummer_ratio(d, kappa);
                let back = solve_kappa(d, r);
                assert!(
                    (back - kappa).abs() <= 1e-5 * kappa.abs().max(1.0),
                    "d={d} kappa={kappa} -> r={r} -> {back}"
                );
            }
        }
        // The isotropic point is exact, and it is what fixes the sign of every other solve.
        for d in [3usize, 10, 64] {
            assert_eq!(solve_kappa(d, 1.0 / d as f64), 0.0);
            assert!(solve_kappa(d, 1.0 / d as f64 + 0.05) > 0.0);
            assert!(solve_kappa(d, 1.0 / d as f64 - 0.004) < 0.0);
        }
    }

    fn axial_leaves(rng_seed: u64, per: usize, axes: &[[f64; 3]], spread: f64) -> Vec<Full<f64>> {
        let mut rng = SplitMix64::new(rng_seed);
        let mut out = Vec::new();
        for a in axes {
            for j in 0..per {
                let mut cf = Full::new(3);
                // Every point is placed on the axis, then half of them are reflected through the
                // origin: the sign carries no information, which is the fixture's whole content.
                let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                let mut p = [0.0; 3];
                for d in 0..3 {
                    p[d] = sign * a[d] + spread * (rng.next_f64() - 0.5);
                }
                let nrm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                for v in p.iter_mut() {
                    *v /= nrm;
                }
                cf.push(&p, 1.0);
                out.push(cf);
            }
        }
        out
    }

    #[test]
    fn the_seeding_spends_its_second_axis_on_a_second_axis_not_on_an_antipode() {
        // 80 leaves on one line, at both of its ends, and 5 on another. A directional `k`-means++
        // would score `−e0` as the point farthest from `e0` and burn both seeds on the same axis;
        // in `1 − (u·μ)²` an antipode is at distance 0, so the second seed has to leave the line.
        let mut u: Vec<Vec<f64>> = Vec::new();
        for i in 0..80 {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            u.push(vec![s, 0.0, 0.0]);
        }
        u.extend((0..5).map(|_| vec![0.0, 1.0, 0.0]));
        let n = vec![1.0; u.len()];
        for seed in 0..20 {
            let mut rng = SplitMix64::new(seed);
            let axes = axial_pp(&u, &n, 2, &mut rng);
            let cos = dot(&axes[0], &axes[1]).abs();
            assert!(cos < 0.5, "seed {seed} seeded one line twice: {axes:?}");
        }
    }

    #[test]
    fn the_axial_head_reads_a_sign_free_direction_that_the_vmf_head_cannot() {
        // Two axes, each populated at both poles. To a vMF mixture this is four clusters whose
        // resultants cancel in pairs; to a Watson mixture it is two.
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let feats = axial_leaves(7, 60, &axes, 0.25);
        let truth: Vec<usize> = (0..120).map(|i| i / 60).collect();

        let w = watson(&feats, 2, 100, 0);
        let v = movmf(&feats, 2, 100, 0);
        assert!(
            ari(&w.labels, &truth) > 0.95,
            "watson scored {}",
            ari(&w.labels, &truth)
        );
        assert!(
            ari(&v.labels, &truth) < 0.5,
            "vmf scored {} — the fixture no longer separates the two models",
            ari(&v.labels, &truth)
        );
    }

    #[test]
    fn a_component_is_the_same_component_under_reflection() {
        // The defining symmetry: negating every point must not move a single label.
        let axes = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let feats = axial_leaves(11, 40, &axes, 0.3);
        let flipped: Vec<Full<f64>> = feats
            .iter()
            .map(|f| {
                let mut cf = Full::new(3);
                let neg: Vec<f64> = f.mean().iter().map(|&v| -v).collect();
                cf.push(&neg, f.weight());
                cf
            })
            .collect();
        let a = watson(&feats, 2, 100, 3);
        let b = watson(&flipped, 2, 100, 3);
        assert!((ari(&a.labels, &b.labels) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_girdle_component_is_fitted_with_a_negative_concentration() {
        // Points spread over the x-y great circle: the structure is the *pole* they avoid, which is
        // only expressible with kappa < 0.
        let mut rng = SplitMix64::new(5);
        let feats: Vec<Full<f64>> = (0..200)
            .map(|_| {
                let t = rng.next_f64() * std::f64::consts::TAU;
                let mut cf = Full::new(3);
                cf.push(&[t.cos(), t.sin(), 0.02 * (rng.next_f64() - 0.5)], 1.0);
                cf
            })
            .collect();
        let fit = watson(&feats, 1, 100, 0);
        assert!(fit.kappas[0] < -10.0, "kappa = {}", fit.kappas[0]);
        // ...and the axis it names is the pole of that circle, up to sign.
        assert!(fit.axes[0][2].abs() > 0.99, "axis = {:?}", fit.axes[0]);
    }

    #[test]
    fn the_within_leaf_scatter_reaches_the_e_step() {
        // Family C's points straddle `e0` at ±70° toward `e2`, so its *mean* points at `e0` — where
        // family A is — while what it actually holds lies mostly along `e2`, where family B is.
        // The exact expectation `μᵀ(Σ_i + m_i m_iᵀ)μ` puts C with B; the mean alone puts it with A.
        let tilt = 70.0_f64.to_radians();
        let mut rng = SplitMix64::new(19);
        let jitter = |rng: &mut SplitMix64| 0.3 * (0..4).map(|_| rng.next_f64() - 0.5).sum::<f64>();
        let mut feats = Vec::new();
        for (leaves, pole, half) in [(30usize, 0usize, 0.0), (20, 2, 0.0), (30, 0, tilt)] {
            for _ in 0..leaves {
                let mut cf = Full::new(3);
                for t in 0..8 {
                    let a = if t % 2 == 0 { half } else { -half };
                    let mut p = [0.0; 3];
                    p[pole] = a.cos();
                    p[if pole == 0 { 2 } else { 0 }] = a.sin();
                    p.iter_mut().for_each(|v| *v += jitter(&mut rng));
                    cf.push(&unit(&p), 1.0);
                }
                feats.push(cf);
            }
        }
        let q = |cf: &Full<f64>, axis: [f64; 3]| {
            let axis = axis.to_vec();
            let mut proj = vec![vec![0.0; 3]];
            cf.second_moment()
                .apply_rows(std::slice::from_ref(&axis), &mut proj, 1.0);
            let cross: f64 = dot(&axis, cf.mean());
            dot(&axis, &proj[0]) + cross * cross
        };
        // C leans on `e0` by its mean and on `e2` by its scatter, and the scatter is the larger of
        // the two. `(μ_c·m_i)²` alone sees only the first column of this.
        let c = &feats[65];
        assert!(dot(&[1.0, 0.0, 0.0], &unit(c.mean())) > 0.99);
        assert!(q(c, [0.0, 0.0, 1.0]) > 2.0 * q(c, [1.0, 0.0, 0.0]));

        let fit = watson(&feats, 2, 100, 0);
        let truth: Vec<usize> = (0..80).map(|i| usize::from(i >= 30)).collect();
        assert!(
            (ari(&fit.labels, &truth) - 1.0).abs() < 1e-12,
            "labels {:?}",
            fit.labels
        );
    }

    #[test]
    fn the_degenerate_inputs_answer_rather_than_panic() {
        let one = axial_leaves(2, 1, &[[1.0, 0.0, 0.0]], 0.0);
        let fit = watson(&one, 1, 20, 0);
        assert_eq!(fit.labels, vec![0]);
        assert_eq!(fit.axes.len(), 1);
        // A cap is a cap: an exactly concentrated component asks for infinite kappa and gets the
        // documented ceiling instead of a NaN.
        assert!(solve_kappa(3, 1.0 - 1e-15) <= KAPPA_MAX);
        assert!(solve_kappa(3, 1e-15) >= -KAPPA_MAX);
        assert!(solve_kappa(3, 0.5).is_finite());
    }

    #[test]
    fn auto_k_picks_the_axis_count_by_bic() {
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let feats = axial_leaves(13, 40, &axes, 0.2);
        let fit = watson_auto(&feats, 1, 6, 100, 0);
        assert_eq!(fit.weights.len(), 3);
    }
}
