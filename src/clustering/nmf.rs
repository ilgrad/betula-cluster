//! CF-weighted nonnegative matrix factorization (weighted HALS) over microcluster centroids.
//!
//! BETULA compresses `N` points into `M ≪ N` leaf microclusters. Assigning every point in leaf `C_j`
//! the same nonnegative code `z_j` (the hard-leaf approximation every Phase-3 head already makes), the
//! full-data NMF objective factors by König-Huygens / parallel-axis:
//! `Σ_{x∈C_j} ‖x − z_j H‖² = Σ_{x∈C_j} ‖x − μ_j‖² + n_j ‖μ_j − z_j H‖²`. The first term is the
//! within-leaf scatter (independent of the factorization, already in the CF stats), so minimizing the
//! full objective is equivalent — up to that constant — to the **weighted centroid** problem
//! `min_{Z,H ≥ 0} Σ_j n_j ‖μ_j − z_j H‖² = ‖X̃ − W H‖²_F` with `X̃_j = √n_j·μ_j`, `W_j = √n_j·z_j`.
//!
//! So the expensive factorization runs over the `M×d` centroid matrix, not the `N×d` raw one — the
//! compression *is* the acceleration (`O(M·d·r)` per sweep, `M ≪ N`), and the matrices are small enough
//! that no BLAS is needed. The solver is **weighted HALS** (coordinate descent, Cichocki-Phan): it
//! reuses the Gram / cross-product matrices `HHᵀ`, `X̃Hᵀ`, `WᵀW`, `WᵀX̃` across the column/row sweeps.
//! Nonnegative data only (TF-IDF / counts / spectrograms / histograms) — validated at the boundary.

// The projection is reached only through the Python bindings (`feature = "python"`); without them the
// factorizer is exercised solely by the unit tests, so silence dead-code analysis of the lib-only view.
#![cfg_attr(not(feature = "python"), allow(dead_code))]

use crate::clustering::rng::SplitMix64;
use crate::feature::{ClusterFeature, Spherical};
use crate::linalg::jacobi_eigen;
use crate::types::Real;

/// Build `n` output rows, in parallel over rows when the `parallel` feature is on.
#[cfg(feature = "parallel")]
fn build_rows<R: Real>(n: usize, f: impl Fn(usize) -> Vec<R> + Sync + Send) -> Vec<Vec<R>> {
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}
#[cfg(not(feature = "parallel"))]
fn build_rows<R: Real>(n: usize, f: impl Fn(usize) -> Vec<R>) -> Vec<Vec<R>> {
    (0..n).map(f).collect()
}

/// `r×r` Gram matrix of the rows of `a` (`a·aᵀ`).
fn gram_rows<R: Real>(a: &[Vec<R>]) -> Vec<Vec<R>> {
    let r = a.len();
    let mut g = vec![vec![R::zero(); r]; r];
    for i in 0..r {
        for j in i..r {
            let s: R = a[i].iter().zip(&a[j]).map(|(&x, &y)| x * y).sum();
            g[i][j] = s;
            g[j][i] = s;
        }
    }
    g
}

/// `r×r` Gram matrix of the columns of `w` (`wᵀ·w`, `w` is `m×r`).
#[allow(clippy::needless_range_loop)]
fn gram_cols<R: Real>(w: &[Vec<R>], r: usize) -> Vec<Vec<R>> {
    let mut g = vec![vec![R::zero(); r]; r];
    for row in w {
        for a in 0..r {
            for b in a..r {
                g[a][b] = g[a][b] + row[a] * row[b];
            }
        }
    }
    for a in 0..r {
        for b in 0..a {
            g[a][b] = g[b][a];
        }
    }
    g
}

/// Rank-`r` truncated SVD of `x` (`m×d`) by a randomized range finder (Halko-Martinsson-Tropp):
/// sketch `Y = XΩ` with a Gaussian `Ω`, two power iterations with re-orthonormalization, then a small
/// eigendecomposition of `BBᵀ` where `B = QᵀX`. Cost is `O(m·d·l)` with `l = r + oversampling`, versus
/// `O(min(m,d)³)` for a dense factorization — and the crate has no LAPACK to call anyway.
///
/// Returns `(σ, U, V)` with `σ` descending, `U[k]` the `k`-th left vector (length `m`) and `V[k]` the
/// `k`-th right vector (length `d`). Vectors are unit-norm; signs are arbitrary (as always for an SVD).
fn randomized_svd<R: Real>(
    x: &[Vec<R>],
    r: usize,
    seed: u64,
) -> (Vec<R>, Vec<Vec<R>>, Vec<Vec<R>>) {
    let m = x.len();
    let d = x[0].len();
    let l = (r + 10).min(m).min(d);
    let mut rng = SplitMix64::new(seed ^ 0x5eed_05bd_u64);

    // Y = X Ω  (m×l), Ω ~ N(0,1) (d×l), then two power iterations Y ← X(XᵀY) for spectral decay.
    let omega: Vec<Vec<R>> = (0..d)
        .map(|_| (0..l).map(|_| R::from_f64(rng.gauss()).unwrap()).collect())
        .collect();
    let mut q: Vec<Vec<R>> = build_rows(m, |j| {
        (0..l)
            .map(|c| {
                (0..d)
                    .map(|t| x[j][t] * omega[t][c])
                    .fold(R::zero(), |a, b| a + b)
            })
            .collect()
    });
    for _ in 0..2 {
        orthonormalize(&mut q, l);
        let xtq: Vec<Vec<R>> = build_rows(d, |t| {
            (0..l)
                .map(|c| {
                    (0..m)
                        .map(|j| x[j][t] * q[j][c])
                        .fold(R::zero(), |a, b| a + b)
                })
                .collect()
        });
        q = build_rows(m, |j| {
            (0..l)
                .map(|c| {
                    (0..d)
                        .map(|t| x[j][t] * xtq[t][c])
                        .fold(R::zero(), |a, b| a + b)
                })
                .collect()
        });
    }
    orthonormalize(&mut q, l);

    // B = Qᵀ X (l×d); the SVD of B lifts back through Q.
    let b: Vec<Vec<R>> = build_rows(l, |c| {
        (0..d)
            .map(|t| {
                (0..m)
                    .map(|j| q[j][c] * x[j][t])
                    .fold(R::zero(), |a, b| a + b)
            })
            .collect()
    });
    let (eigvals, eigvecs) = jacobi_eigen(&gram_rows(&b));
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());

    // Numerical-rank cutoff (LAPACK convention). `v = Bᵀu/σ` is only meaningful while `σ` is above the
    // noise floor: past it the division amplifies round-off into a vector of arbitrary magnitude, which
    // then seeds a component so far out of scale that the first HALS sweep annihilates it — measured 28
    // of 32 components dead on a rank-12 matrix. Below the cutoff the triplet carries no information, so
    // report an honest zero and let NNDSVDar's fill seed that component instead.
    let sigma_max = order
        .first()
        .map_or(R::zero(), |&i| eigvals[i].max(R::zero()).sqrt());
    let cutoff = sigma_max * R::from_usize(m.max(d)).unwrap() * R::epsilon();
    let mut sigma = Vec::with_capacity(r);
    let mut u = Vec::with_capacity(r);
    let mut v = Vec::with_capacity(r);
    for &idx in order.iter().take(r.min(l)) {
        let s = eigvals[idx].max(R::zero()).sqrt();
        if s <= cutoff {
            sigma.push(R::zero());
            u.push(vec![R::zero(); m]);
            v.push(vec![R::zero(); d]);
            continue;
        }
        // left vector in the sketch basis → lift to R^m through Q
        let ub: Vec<R> = (0..l).map(|i| eigvecs[i][idx]).collect();
        let uk: Vec<R> = (0..m)
            .map(|j| {
                (0..l)
                    .map(|c| q[j][c] * ub[c])
                    .fold(R::zero(), |a, b| a + b)
            })
            .collect();
        // right vector: v = Bᵀ u_B / σ
        let vk: Vec<R> = (0..d)
            .map(|t| {
                (0..l)
                    .map(|c| b[c][t] * ub[c])
                    .fold(R::zero(), |a, b| a + b)
                    / s
            })
            .collect();
        sigma.push(s);
        u.push(uk);
        v.push(vk);
    }
    (sigma, u, v)
}

/// `WᵀX` (`r×d`) accumulated row-by-row over `X`.
///
/// The transpose-product is the sweep's hot loop, and the obvious expression of it —
/// `wtx[k][c] = Σ_j w[j][k]·x[j][c]` evaluated per output cell — walks a whole column of `X` for each
/// of the `r·d` cells, striding `d` floats per step through a matrix far larger than L2. Accumulating
/// into the (small, cache-resident) `r×d` output while reading each row of `X` once, sequentially,
/// computes the same product with the access pattern the hardware wants.
fn wt_x<R: Real>(w: &[Vec<R>], x: &[Vec<R>], r: usize, d: usize) -> Vec<Vec<R>> {
    let fold = |mut acc: Vec<Vec<R>>, (wj, xj): (&Vec<R>, &Vec<R>)| {
        for k in 0..r {
            let wjk = wj[k];
            if wjk != R::zero() {
                for (a, &v) in acc[k].iter_mut().zip(xj.iter()) {
                    *a = *a + wjk * v;
                }
            }
        }
        acc
    };
    let zero = || vec![vec![R::zero(); d]; r];
    let merge = |mut a: Vec<Vec<R>>, b: Vec<Vec<R>>| {
        for (ra, rb) in a.iter_mut().zip(b) {
            for (va, vb) in ra.iter_mut().zip(rb) {
                *va = *va + vb;
            }
        }
        a
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        w.par_iter()
            .zip(x.par_iter())
            .fold(zero, fold)
            .reduce(zero, merge)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = merge;
        w.iter().zip(x.iter()).fold(zero(), fold)
    }
}

/// Modified Gram-Schmidt orthonormalization of the `cols` columns of `a` (`m×cols`), in place.
fn orthonormalize<R: Real>(a: &mut [Vec<R>], cols: usize) {
    let eps = R::from_f64(1e-12).unwrap();
    for c in 0..cols {
        for prev in 0..c {
            let dot = a
                .iter()
                .map(|row| row[c] * row[prev])
                .fold(R::zero(), |x, y| x + y);
            for row in a.iter_mut() {
                row[c] = row[c] - dot * row[prev];
            }
        }
        let norm = a
            .iter()
            .map(|row| row[c] * row[c])
            .fold(R::zero(), |x, y| x + y)
            .sqrt();
        if norm > eps {
            for row in a.iter_mut() {
                row[c] = row[c] / norm;
            }
        } else {
            // Rank-deficient sketch column: zero it rather than amplify noise.
            for row in a.iter_mut() {
                row[c] = R::zero();
            }
        }
    }
}

/// NNDSVDar initialization (Boutsidis & Gallopoulos 2008): take the rank-`r` SVD and build a
/// nonnegative pair from each singular triplet by keeping whichever of the positive / negative parts
/// carries more energy. Deterministic given the seed, and far better conditioned than a random start —
/// HALS is a non-convex coordinate descent, so the basin it lands in is decided here.
///
/// Plain NNDSVD leaves the resulting zeros at zero, where both HALS and the multiplicative updates lock
/// them forever (zero is a fixed point of both), so the zeros must be filled. The **`ar`** variant fills
/// them with `mean(X)·U(0,1)/100` rather than the `a` variant's `mean(X)`, and both halves of that matter
/// here:
///
/// * **Scale.** A filled component is a rank-1 block of constant magnitude `f²` per entry, and `r` of them
///   stack additively against data entries of size `mean(X)`. At `f = mean(X)` the fill dominates the data
///   — measured on a rank-12 matrix at `r = 32`: initial relative residual 13.5, and the first HALS sweep
///   annihilated 28 of the 32 components (zero being absorbing, they never returned). Dividing by 100 puts
///   the fill back in the perturbation regime: 0 components dead, converged residual 212× lower, and on
///   `digits` at `r = 24` the downstream ARI rose 0.54 → 0.63 with the reconstruction error 0.33 → 0.20.
/// * **Randomness.** A constant fill gives every zero-seeded component the *same* column, so they are
///   linearly dependent and no coordinate descent can pull them apart. `U(0,1)` breaks that degeneracy.
///
/// Rank-deficient triplets (`σ` below the numerical-rank cutoff) arrive as exact zeros from the SVD and
/// are seeded entirely by this fill — which is the intended path, not a fallback.
fn nndsvdar<R: Real>(x: &[Vec<R>], r: usize, seed: u64) -> (Vec<Vec<R>>, Vec<Vec<R>>) {
    let m = x.len();
    let d = x[0].len();
    let (sigma, u, v) = randomized_svd(x, r, seed);
    let mut w = vec![vec![R::zero(); r]; m];
    let mut h = vec![vec![R::zero(); d]; r];
    let eps = R::from_f64(1e-12).unwrap();

    for k in 0..sigma.len() {
        let (uk, vk, s) = (&u[k], &v[k], sigma[k]);
        let (wk, hk) = if k == 0 {
            // The leading pair is sign-definite for a nonnegative X (Perron-Frobenius), so |·| is exact.
            let root = s.sqrt();
            (
                uk.iter().map(|&t| root * t.abs()).collect::<Vec<R>>(),
                vk.iter().map(|&t| root * t.abs()).collect::<Vec<R>>(),
            )
        } else {
            let split = |z: &[R]| -> (Vec<R>, Vec<R>, R, R) {
                let p: Vec<R> = z.iter().map(|&t| t.max(R::zero())).collect();
                let n: Vec<R> = z.iter().map(|&t| (-t).max(R::zero())).collect();
                let pn = p
                    .iter()
                    .map(|&t| t * t)
                    .fold(R::zero(), |a, b| a + b)
                    .sqrt();
                let nn = n
                    .iter()
                    .map(|&t| t * t)
                    .fold(R::zero(), |a, b| a + b)
                    .sqrt();
                (p, n, pn, nn)
            };
            let (up, un, upn, unn) = split(uk);
            let (vp, vn, vpn, vnn) = split(vk);
            // Keep whichever signed half carries more energy, and normalize by *its own* norms.
            let (uu, vv, un_norm, vn_norm, mu) = if upn * vpn >= unn * vnn {
                (up, vp, upn, vpn, upn * vpn)
            } else {
                (un, vn, unn, vnn, unn * vnn)
            };
            let lbd = (s * mu).sqrt();
            (
                uu.iter().map(|&t| lbd * t / un_norm.max(eps)).collect(),
                vv.iter().map(|&t| lbd * t / vn_norm.max(eps)).collect(),
            )
        };
        for j in 0..m {
            w[j][k] = wk[j];
        }
        h[k][..d].copy_from_slice(&hk[..d]);
    }

    let total = x
        .iter()
        .map(|row| row.iter().fold(R::zero(), |a, &b| a + b))
        .fold(R::zero(), |a, b| a + b);
    let avg = (total / R::from_usize((m * d).max(1)).unwrap()).max(R::from_f64(1e-8).unwrap())
        * R::from_f64(0.01).unwrap();
    let mut fill = SplitMix64::new(seed ^ 0x00f1_115c_a1e0_u64);
    for row in w.iter_mut().chain(h.iter_mut()) {
        for t in row.iter_mut() {
            if *t <= R::zero() {
                *t = avg * R::from_f64(fill.next_f64()).unwrap();
            }
        }
    }
    (w, h)
}

/// Generalized KL divergence `Σ_ij [x log(x/wh) − x + wh]` (I-divergence), the objective the
/// multiplicative updates minimize. `0·log 0` is taken as `0`, as the limit requires.
fn kl_divergence<R: Real>(x: &[Vec<R>], w: &[Vec<R>], h: &[Vec<R>], d: usize) -> R {
    let r = h.len();
    let eps = R::from_f64(1e-10).unwrap();
    let per: Vec<Vec<R>> = build_rows(x.len(), |i| {
        let mut s = R::zero();
        for c in 0..d {
            let mut wh = R::zero();
            for k in 0..r {
                wh = wh + w[i][k] * h[k][c];
            }
            let xv = x[i][c];
            if xv > eps {
                s = s + xv * (xv / wh.max(eps)).ln() - xv + wh;
            } else {
                s = s + wh;
            }
        }
        vec![s]
    });
    per.iter().map(|v| v[0]).fold(R::zero(), |a, b| a + b)
}

/// `‖X − W H‖²_F` (residual sum of squares), summed over rows.
fn residual<R: Real>(x: &[Vec<R>], w: &[Vec<R>], h: &[Vec<R>], d: usize) -> R {
    let r = h.len();
    let per: Vec<Vec<R>> = build_rows(x.len(), |j| {
        let mut s = R::zero();
        for c in 0..d {
            let mut wh = R::zero();
            for k in 0..r {
                wh = wh + w[j][k] * h[k][c];
            }
            let e = x[j][c] - wh;
            s = s + e * e;
        }
        vec![s]
    });
    per.iter().map(|v| v[0]).fold(R::zero(), |a, b| a + b)
}

/// Resolve the `(W D, D⁻¹H)` scale indeterminacy every NMF objective has: normalize each component row
/// of `H` to unit L2 norm, pushing the scale into the matching column of `W`, then order components by
/// descending energy `‖W_k‖`.
///
/// This is not cosmetic. The reconstruction `W H` is invariant to `D`, so the optimizer leaves whatever
/// split it happened to land on — measured spreads of 70× between component scales on a converged fit.
/// But `W` leaves this module as a **Euclidean feature vector** for a Phase-3 head, where a per-component
/// scale is a per-dimension weight: without this the head silently clusters along whichever component
/// drew the largest number. Canonicalizing also makes `H` comparable across runs and ranks.
///
/// L2 (rather than L1, which would make the KL components read as distributions) because both solvers'
/// output is consumed as a Euclidean feature — one invariant for both.
fn canonicalize<R: Real>(w: &mut [Vec<R>], h: &mut [Vec<R>]) {
    let r = h.len();
    let eps = R::from_f64(1e-12).unwrap();
    for k in 0..r {
        let hn = h[k]
            .iter()
            .map(|&t| t * t)
            .fold(R::zero(), |a, b| a + b)
            .sqrt();
        if hn > eps {
            for t in h[k].iter_mut() {
                *t = *t / hn;
            }
            for row in w.iter_mut() {
                row[k] = row[k] * hn;
            }
        }
    }
    let energy: Vec<R> = (0..r)
        .map(|k| {
            w.iter()
                .map(|row| row[k] * row[k])
                .fold(R::zero(), |a, b| a + b)
        })
        .collect();
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| energy[b].partial_cmp(&energy[a]).unwrap());
    if order.iter().enumerate().any(|(i, &k)| i != k) {
        for row in w.iter_mut() {
            let permuted: Vec<R> = order.iter().map(|&k| row[k]).collect();
            row.copy_from_slice(&permuted);
        }
        let permuted: Vec<Vec<R>> = order.iter().map(|&k| h[k].clone()).collect();
        h.clone_from_slice(&permuted);
    }
}

/// Weighted NMF `X̃ ≈ W H` with `X̃_j = √w_j·μ_j` (`m×d`), `W ≥ 0` (`m×r`), `H ≥ 0` (`r×d`), by weighted
/// HALS. Returns per-microcluster codes `z_j = W_j / √w_j` (`m×r`) and components `H` (`r×d`).
fn weighted_nmf<R: Real>(
    centroids: &[Vec<R>],
    weights: &[R],
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<R>>, Vec<Vec<R>>) {
    let m = centroids.len();
    let d = centroids[0].len();
    // Rank is bounded by both matrix dimensions: r > m makes the factorization rank-deficient by
    // construction and leaves whole components with nothing to fit.
    let r = rank.min(d).min(m).max(1);
    let eps = R::from_f64(1e-10).unwrap();

    // X̃ = √w · μ (row-scaled); clamp tiny negatives from float error (no data shifting).
    let sw: Vec<R> = weights.iter().map(|&w| w.max(R::zero()).sqrt()).collect();
    let x: Vec<Vec<R>> = (0..m)
        .map(|j| {
            (0..d)
                .map(|c| sw[j] * centroids[j][c].max(R::zero()))
                .collect()
        })
        .collect();

    let (mut w, mut h) = nndsvdar(&x, r, seed);

    let tol = R::from_f64(1e-4).unwrap();
    let mut first_movement = R::zero();

    for it in 0..max_iter {
        // Stopping rule follows the size of the update, not the size of the objective. A relative test
        // on the residual never fires: HALS converges sublinearly, so it keeps buying more than `tol`
        // of relative improvement for hundreds of sweeps and `max_iter` ends up the only brake
        // (measured: a 4x budget cost 6x the time, so the check had never once triggered). The total
        // coordinate movement, compared against the first sweep's, is scale-free and does converge —
        // and it falls out of the sweep for free, with no extra pass over the data.
        let mut movement = R::zero();

        // ── update W (columns): A = X Hᵀ (m×r), B = H Hᵀ (r×r) ──
        let hht = gram_rows(&h);
        let xht = build_rows(m, |j| {
            (0..r)
                .map(|k| x[j].iter().zip(&h[k]).map(|(&a, &b)| a * b).sum::<R>())
                .collect()
        });
        for j in 0..m {
            for k in 0..r {
                let mut s = xht[j][k];
                for l in 0..r {
                    s = s - w[j][l] * hht[l][k];
                }
                s = s + w[j][k] * hht[k][k];
                let next = (s / hht[k][k].max(eps)).max(R::zero());
                movement = movement + (next - w[j][k]).abs();
                w[j][k] = next;
            }
        }
        // ── update H (rows): C = Wᵀ X (r×d), G = Wᵀ W (r×r) ──
        let wtw = gram_cols(&w, r);
        let wtx = wt_x(&w, &x, r, d);
        for k in 0..r {
            for c in 0..d {
                let mut s = wtx[k][c];
                for l in 0..r {
                    s = s - wtw[k][l] * h[l][c];
                }
                s = s + wtw[k][k] * h[k][c];
                let next = (s / wtw[k][k].max(eps)).max(R::zero());
                movement = movement + (next - h[k][c]).abs();
                h[k][c] = next;
            }
        }

        if it == 0 {
            first_movement = movement;
        } else if movement <= tol * first_movement {
            break;
        }
    }

    canonicalize(&mut w, &mut h);
    let codes: Vec<Vec<R>> = (0..m)
        .map(|j| {
            let inv = if sw[j] > eps {
                R::one() / sw[j]
            } else {
                R::zero()
            };
            w[j].iter().map(|&v| v * inv).collect()
        })
        .collect();
    (codes, h)
}

/// Weighted **KL-divergence** NMF `X ≈ W H` with `X` = the raw nonnegative centroids `μ` (`M×d`), by
/// Lee-Seung multiplicative updates. The generalized-KL (I-divergence) objective is the right noise
/// model for **count** data (Poisson), where the Frobenius (Gaussian) HALS is mis-specified. Row weights
/// `n_j` scale the shared-component (`H`) update — heavier leaves shape the parts more — while the
/// per-row `W` update is weight-invariant (each row's divergence is minimized independently). Returns
/// codes `W` (`M×r`) and components `H` (`r×d`).
fn weighted_nmf_kl<R: Real>(
    centroids: &[Vec<R>],
    weights: &[R],
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<R>>, Vec<Vec<R>>) {
    let m = centroids.len();
    let d = centroids[0].len();
    let r = rank.min(d).min(m).max(1);
    let eps = R::from_f64(1e-10).unwrap();
    let x: Vec<Vec<R>> = centroids
        .iter()
        .map(|row| row.iter().map(|&v| v.max(R::zero())).collect())
        .collect();

    // Same NNDSVDar start as the Frobenius solver. Zero is a fixed point of the multiplicative updates
    // too — more starkly, since they are purely multiplicative — so the fill is not optional here.
    let (mut w, mut h) = nndsvdar(&x, r, seed);

    let wh_of = |w: &[Vec<R>], h: &[Vec<R>]| -> Vec<Vec<R>> {
        build_rows(m, |i| {
            (0..d)
                .map(|j| {
                    (0..r)
                        .map(|k| w[i][k] * h[k][j])
                        .fold(R::zero(), |a, b| a + b)
                })
                .collect()
        })
    };
    let tol = R::from_f64(1e-4).unwrap();
    let mut prev = kl_divergence(&x, &w, &h, d);
    for it in 0..max_iter {
        // W_ik *= [Σ_j (X_ij/WH_ij) H_kj] / [Σ_j H_kj]
        let wh = wh_of(&w, &h);
        let colsum_h: Vec<R> = (0..r)
            .map(|k| (0..d).map(|j| h[k][j]).fold(R::zero(), |a, b| a + b))
            .collect();
        w = build_rows(m, |i| {
            (0..r)
                .map(|k| {
                    let num: R = (0..d)
                        .map(|j| (x[i][j] / wh[i][j].max(eps)) * h[k][j])
                        .fold(R::zero(), |a, b| a + b);
                    w[i][k] * num / colsum_h[k].max(eps)
                })
                .collect()
        });
        // H_kj *= [Σ_i n_i (X_ij/WH_ij) W_ik] / [Σ_i n_i W_ik]
        let wh = wh_of(&w, &h);
        let wsum: Vec<R> = (0..r)
            .map(|k| {
                (0..m)
                    .map(|i| weights[i].max(R::zero()) * w[i][k])
                    .fold(R::zero(), |a, b| a + b)
            })
            .collect();
        h = build_rows(r, |k| {
            (0..d)
                .map(|j| {
                    let num: R = (0..m)
                        .map(|i| {
                            weights[i].max(R::zero()) * (x[i][j] / wh[i][j].max(eps)) * w[i][k]
                        })
                        .fold(R::zero(), |a, b| a + b);
                    h[k][j] * num / wsum[k].max(eps)
                })
                .collect()
        });
        // ── convergence check every few sweeps ──
        if it % 5 == 4 {
            let err = kl_divergence(&x, &w, &h, d);
            if (prev - err).abs() <= tol * prev.max(R::one()) {
                break;
            }
            prev = err;
        }
    }
    canonicalize(&mut w, &mut h);
    (w, h)
}

/// How to run the optional Phase-3 NMF projection. A value object rather than a tuple: the settings are
/// threaded through every dispatch signature, and a bare `(usize, bool, usize)` at those call sites says
/// nothing about which number is which.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NmfSpec {
    /// Target rank. Capped by the centroid matrix's own dimensions inside the solvers.
    pub rank: usize,
    /// Minimize the generalized KL divergence (the Poisson model for counts) instead of Frobenius.
    pub kl: bool,
    /// Solver sweeps. Separate from the clustering head's budget — the two converge at different rates,
    /// and sharing one number made a larger `max_iter` for the head silently pay for NMF sweeps too.
    pub max_iter: usize,
}

/// The factorization of a set of leaf microclusters: per-leaf codes as single-point `Spherical` features
/// (mean = code, weight = original mass) for a downstream Phase-3 head, plus the shared parts.
pub(crate) struct Projection<R: Real> {
    pub coded: Vec<Spherical<R>>,
    pub components: Vec<Vec<R>>,
    /// Relative reconstruction error `‖X̃ − W H‖_F / ‖X̃‖_F` of the weighted centroid matrix.
    pub reconstruction_err: R,
}

/// Project leaf microclusters to `spec.rank`-dimensional CF-weighted NMF codes (Frobenius HALS, or the
/// KL-divergence multiplicative variant for count data).
pub(crate) fn project<R, C>(feats: &[C], spec: NmfSpec, seed: u64) -> Projection<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    if feats.is_empty() {
        return Projection {
            coded: Vec::new(),
            components: Vec::new(),
            reconstruction_err: R::zero(),
        };
    }
    let centroids: Vec<Vec<R>> = feats.iter().map(|f| f.mean().to_vec()).collect();
    let weights: Vec<R> = feats.iter().map(|f| f.weight()).collect();
    let iters = spec.max_iter.max(1);
    let (codes, components) = if spec.kl {
        weighted_nmf_kl(&centroids, &weights, spec.rank, iters, seed)
    } else {
        weighted_nmf(&centroids, &weights, spec.rank, iters, seed)
    };

    // Scored on the matrix each solver actually fits: √n·μ for Frobenius, raw μ for KL.
    let d = centroids[0].len();
    let x: Vec<Vec<R>> = (0..centroids.len())
        .map(|j| {
            let s = if spec.kl {
                R::one()
            } else {
                weights[j].max(R::zero()).sqrt()
            };
            centroids[j].iter().map(|&v| s * v.max(R::zero())).collect()
        })
        .collect();
    let w: Vec<Vec<R>> = (0..codes.len())
        .map(|j| {
            let s = if spec.kl {
                R::one()
            } else {
                weights[j].max(R::zero()).sqrt()
            };
            codes[j].iter().map(|&v| s * v).collect()
        })
        .collect();
    let energy = x
        .iter()
        .flatten()
        .map(|&v| v * v)
        .fold(R::zero(), |a, b| a + b);
    let reconstruction_err = if energy > R::zero() {
        (residual(&x, &w, &components, d) / energy)
            .max(R::zero())
            .sqrt()
    } else {
        R::zero()
    };

    Projection {
        coded: codes
            .into_iter()
            .zip(&weights)
            .map(|(code, &w)| Spherical::from_moments(w, code, R::zero()))
            .collect(),
        components,
        reconstruction_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build spherical leaves directly from centroids (unit mass) for testing the projection.
    fn leaves(centroids: &[Vec<f64>]) -> Vec<Spherical<f64>> {
        centroids
            .iter()
            .map(|c| Spherical::from_moments(1.0, c.clone(), 0.0))
            .collect()
    }

    #[test]
    fn recovers_planted_parts() {
        // Two nonnegative "parts"; every centroid is a nonnegative mix of them. Rank-2 NMF must
        // reconstruct the centroids well (small residual) and keep codes nonnegative.
        let base = [[3.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 4.0]];
        let mut cents = Vec::new();
        let mut rng = SplitMix64::new(1);
        for _ in 0..40 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            cents.push(
                (0..4)
                    .map(|c| a * base[0][c] + b * base[1][c])
                    .collect::<Vec<_>>(),
            );
        }
        let (codes, h) = weighted_nmf(&cents, &vec![1.0; cents.len()], 2, 200, 7);
        assert!(codes.iter().flatten().all(|&v| v >= 0.0 && v.is_finite()));
        assert!(h.iter().flatten().all(|&v| v >= 0.0));
        // reconstruction error should be a tiny fraction of the signal energy
        let x: Vec<Vec<f64>> = cents.clone();
        let sw = vec![1.0; cents.len()];
        let w: Vec<Vec<f64>> = codes
            .iter()
            .zip(&sw)
            .map(|(z, &s)| z.iter().map(|&v| v * s).collect())
            .collect();
        let err = residual(&x, &w, &h, 4);
        let energy: f64 = x.iter().flatten().map(|v| v * v).sum();
        assert!(err / energy < 0.02, "relative residual {}", err / energy);
    }

    #[test]
    fn project_features_reduces_dim_and_weights() {
        let cents = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 0.0],
            vec![2.0, 1.0, 1.0],
        ];
        let feats = leaves(&cents);
        let out = project(
            &feats,
            NmfSpec {
                rank: 2,
                kl: false,
                max_iter: 100,
            },
            3,
        );
        let coded = out.coded;
        assert_eq!(coded.len(), 3);
        assert_eq!(out.components.len(), 2);
        assert!((0.0..=1.0).contains(&out.reconstruction_err));
        assert!(coded.iter().all(|f| f.dim() == 2));
        assert!(coded.iter().all(|f| f.weight() == 1.0));
        assert!(coded
            .iter()
            .all(|f| f.mean().iter().all(|&v| v >= 0.0 && v.is_finite())));
    }

    #[test]
    fn more_sweeps_never_worsen_the_fit() {
        // The convergence test compares `|prev − err|` against `tol · prev`. Seeding `prev` with
        // `+inf` makes both sides infinite and IEEE-754 says `inf <= inf`, so the solver breaks at
        // its first check and `max_iter` becomes dead code — a bug that shipped once and was
        // reintroduced once. A residual that keeps falling as the budget grows is what catches it.
        for (m, d, r, seed) in [(400usize, 60usize, 8usize, 3u64), (800, 40, 12, 9)] {
            let mut rng = SplitMix64::new(seed);
            let base: Vec<Vec<f64>> = (0..r)
                .map(|_| {
                    (0..d)
                        .map(|_| {
                            if rng.next_f64() < 0.3 {
                                rng.next_f64()
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            let x: Vec<Vec<f64>> = (0..m)
                .map(|_| {
                    let c: Vec<f64> = (0..r).map(|_| rng.next_f64()).collect();
                    (0..d)
                        .map(|j| {
                            (0..r).map(|k| c[k] * base[k][j]).sum::<f64>() + 0.02 * rng.next_f64()
                        })
                        .collect()
                })
                .collect();
            let wts = vec![1.0f64; m];
            let energy: f64 = x.iter().flatten().map(|v| v * v).sum();

            let curve: Vec<f64> = [5usize, 20, 80]
                .iter()
                .map(|&it| {
                    let (codes, h) = weighted_nmf(&x, &wts, r, it, 7);
                    residual(&x, &codes, &h, d) / energy
                })
                .collect();
            assert!(
                curve.windows(2).all(|p| p[1] <= p[0]),
                "residual must not grow with the sweep budget: {curve:?}"
            );
            assert!(
                curve[2] < curve[0] * 0.5,
                "80 sweeps should clearly beat 5, got {curve:?}"
            );
        }
    }

    #[test]
    fn overcomplete_rank_keeps_components_alive() {
        // Two ways a component gets seeded so far out of scale that the first HALS sweep annihilates
        // it — and zero is absorbing, so it never returns. (1) A rank-deficient singular triplet:
        // `v = Bᵀu/σ` divided by a clamped near-zero `σ` amplifies round-off into a huge vector.
        // (2) The NNDSVD zero-fill: `r` filled components are rank-1 blocks of constant magnitude that
        // stack additively, so a fill at `mean(X)` swamps the data. Together these left 28 of 32
        // components dead on this matrix and the residual stuck 270× above where it converges now.
        let (m, d, r, true_rank) = (600usize, 64usize, 32usize, 12usize);
        let mut rng = SplitMix64::new(5);
        let parts: Vec<Vec<f64>> = (0..true_rank)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        if rng.next_f64() < 0.4 {
                            rng.next_f64()
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let x: Vec<Vec<f64>> = (0..m)
            .map(|_| {
                let c: Vec<f64> = (0..true_rank).map(|_| rng.next_f64().powi(3)).collect();
                (0..d)
                    .map(|j| (0..true_rank).map(|k| c[k] * parts[k][j]).sum::<f64>())
                    .collect()
            })
            .collect();
        let energy: f64 = x.iter().flatten().map(|v| v * v).sum();

        let (w0, h0) = nndsvdar(&x, r, 7);
        let init = (residual(&x, &w0, &h0, d) / energy).sqrt();
        assert!(
            init < 2.0,
            "initialization must not swamp the data: rel resid {init}"
        );

        let (codes, h) = weighted_nmf(&x, &vec![1.0; m], r, 200, 7);
        let dead = h
            .iter()
            .filter(|row| row.iter().all(|&v| v.abs() < 1e-12))
            .count();
        assert!(
            dead <= r - true_rank,
            "{dead}/{r} components collapsed; at most {} may, since the data is rank {true_rank}",
            r - true_rank
        );
        let err = (residual(&x, &codes, &h, d) / energy).sqrt();
        assert!(
            err < 0.01,
            "overcomplete fit should be near-exact, got {err}"
        );
    }

    #[test]
    fn transpose_product_matches_the_direct_form() {
        // `wt_x` accumulates into the small `r×d` output while reading `X` row-sequentially, rather
        // than walking a column of `X` per output cell. Measured 2.4-3.4x faster; this pins that it
        // still computes `WᵀX`.
        let (m, d, r) = (500usize, 48usize, 9usize);
        let mut rng = SplitMix64::new(3);
        let x: Vec<Vec<f64>> = (0..m)
            .map(|_| (0..d).map(|_| rng.next_f64()).collect())
            .collect();
        let w: Vec<Vec<f64>> = (0..m)
            .map(|_| {
                (0..r)
                    .map(|_| {
                        if rng.next_f64() < 0.3 {
                            0.0
                        } else {
                            rng.next_f64()
                        }
                    })
                    .collect()
            })
            .collect();
        let got = wt_x(&w, &x, r, d);
        for k in 0..r {
            for c in 0..d {
                let want: f64 = (0..m).map(|j| w[j][k] * x[j][c]).sum();
                assert!(
                    (got[k][c] - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "WᵀX[{k}][{c}]: {} vs {want}",
                    got[k][c]
                );
            }
        }
    }

    #[test]
    fn components_are_canonical() {
        // NMF is invariant to `(W D, D⁻¹H)`, so the split is arbitrary unless it is pinned down. The
        // codes leave as a Euclidean feature vector, where a per-component scale is a per-dimension
        // weight — an unnormalized `H` silently reweights the downstream clustering.
        let mut rng = SplitMix64::new(11);
        let base: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..12).map(|_| rng.next_f64()).collect())
            .collect();
        let x: Vec<Vec<f64>> = (0..90)
            .map(|_| {
                let c: Vec<f64> = (0..3).map(|_| rng.next_f64()).collect();
                (0..12)
                    .map(|j| (0..3).map(|k| c[k] * base[k][j]).sum::<f64>())
                    .collect()
            })
            .collect();
        let (codes, h) = weighted_nmf(&x, &vec![1.0; 90], 3, 100, 4);
        for row in &h {
            let n: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-9, "component row norm {n}");
        }
        let energy: Vec<f64> = (0..3)
            .map(|k| codes.iter().map(|row| row[k] * row[k]).sum())
            .collect();
        assert!(
            energy.windows(2).all(|p| p[1] <= p[0] + 1e-12),
            "components must be ordered by descending energy: {energy:?}"
        );
    }

    #[test]
    fn weighting_follows_mass() {
        // A heavy centroid and many light ones far away: the weighted fit must favour the heavy one,
        // so its reconstruction error is smaller than an equal-weight fit would give.
        let mut cents: Vec<Vec<f64>> = vec![vec![10.0, 0.0]];
        for _ in 0..20 {
            cents.push(vec![0.0, 1.0]);
        }
        let mut wts: Vec<f64> = vec![100.0];
        wts.extend(std::iter::repeat_n(1.0, 20));
        let (codes, h) = weighted_nmf(&cents, &wts, 2, 200, 5);
        let recon0: f64 = (0..2)
            .map(|c| {
                let wh: f64 = (0..2)
                    .map(|k| codes[0][k] * wts[0].sqrt() * h[k][c])
                    .sum::<f64>()
                    / wts[0].sqrt();
                (cents[0][c] - wh).powi(2)
            })
            .sum();
        assert!(recon0 < 1.0, "heavy centroid poorly fit: {recon0}");
    }

    #[test]
    fn kl_recovers_nonnegative_parts() {
        // The KL multiplicative variant reconstructs nonnegative mixtures and keeps codes nonnegative.
        let base = [[3.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 4.0]];
        let mut cents = Vec::new();
        let mut rng = SplitMix64::new(2);
        for _ in 0..40 {
            let (a, b) = (rng.next_f64() + 0.1, rng.next_f64() + 0.1);
            cents.push(
                (0..4)
                    .map(|c| a * base[0][c] + b * base[1][c])
                    .collect::<Vec<_>>(),
            );
        }
        let (codes, h) = weighted_nmf_kl(&cents, &vec![1.0; cents.len()], 2, 200, 7);
        assert!(codes.iter().flatten().all(|&v| v >= 0.0 && v.is_finite()));
        assert!(h.iter().flatten().all(|&v| v >= 0.0 && v.is_finite()));
        let (mut err, mut energy) = (0.0, 0.0);
        for (i, row) in cents.iter().enumerate() {
            for (j, &xij) in row.iter().enumerate() {
                let wh: f64 = (0..2).map(|kk| codes[i][kk] * h[kk][j]).sum();
                err += (xij - wh).powi(2);
                energy += xij * xij;
            }
        }
        assert!(err / energy < 0.05, "relative residual {}", err / energy);
    }

    /// Independent re-derivation of the weighted-HALS sweeps in [`weighted_nmf`], written from the
    /// block-coordinate update `w_jk ← max(0, (X Hᵀ − W H Hᵀ)_jk / (H Hᵀ)_kk + w_jk)` rather than
    /// from the source. The end-to-end tests assert a reconstruction *bound*, which a factorisation
    /// can meet while every individual coordinate update is wrong.
    fn reference_hals(
        centroids: &[Vec<f64>],
        weights: &[f64],
        rank: usize,
        max_iter: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let m = centroids.len();
        let d = centroids[0].len();
        let r = rank.min(d).min(m).max(1);
        let eps = 1e-10;
        let sw: Vec<f64> = weights.iter().map(|&w| w.max(0.0).sqrt()).collect();
        let x: Vec<Vec<f64>> = (0..m)
            .map(|j| (0..d).map(|c| sw[j] * centroids[j][c].max(0.0)).collect())
            .collect();
        let (mut w, mut h) = nndsvdar(&x, r, seed);

        let mut first_movement = 0.0;
        for it in 0..max_iter {
            let mut movement = 0.0;
            for j in 0..m {
                for k in 0..r {
                    let hkk: f64 = h[k].iter().map(|&t| t * t).sum();
                    let xh: f64 = (0..d).map(|c| x[j][c] * h[k][c]).sum();
                    let whh: f64 = (0..r)
                        .filter(|&l| l != k)
                        .map(|l| w[j][l] * (0..d).map(|c| h[l][c] * h[k][c]).sum::<f64>())
                        .sum();
                    let next = ((xh - whh) / hkk.max(eps)).max(0.0);
                    movement += (next - w[j][k]).abs();
                    w[j][k] = next;
                }
            }
            for k in 0..r {
                for c in 0..d {
                    let wkk: f64 = w.iter().map(|row| row[k] * row[k]).sum();
                    let wx: f64 = (0..m).map(|j| w[j][k] * x[j][c]).sum();
                    let wwh: f64 = (0..r)
                        .filter(|&l| l != k)
                        .map(|l| h[l][c] * (0..m).map(|j| w[j][k] * w[j][l]).sum::<f64>())
                        .sum();
                    let next = ((wx - wwh) / wkk.max(eps)).max(0.0);
                    movement += (next - h[k][c]).abs();
                    h[k][c] = next;
                }
            }
            if it == 0 {
                first_movement = movement;
            } else if movement <= 1e-4 * first_movement {
                break;
            }
        }
        canonicalize(&mut w, &mut h);
        let codes: Vec<Vec<f64>> = (0..m)
            .map(|j| {
                let inv = if sw[j] > eps { 1.0 / sw[j] } else { 0.0 };
                w[j].iter().map(|&v| v * inv).collect()
            })
            .collect();
        (codes, h)
    }

    fn hals_fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        let base = [[3.0, 0.2, 1.0, 0.4], [0.5, 2.0, 0.3, 4.0]];
        let mut rng = SplitMix64::new(9);
        let mut cents = Vec::new();
        let mut ws = Vec::new();
        for _ in 0..25 {
            let (a, b) = (rng.next_f64(), rng.next_f64());
            cents.push(
                (0..4)
                    .map(|c| a * base[0][c] + b * base[1][c] + 0.05 * rng.next_f64())
                    .collect::<Vec<f64>>(),
            );
            ws.push(1.0 + 4.0 * rng.next_f64());
        }
        (cents, ws)
    }

    #[test]
    fn hals_sweeps_match_an_independent_reference() {
        let (cents, ws) = hals_fixture();
        for iters in [1usize, 3, 40] {
            let (codes, h) = weighted_nmf(&cents, &ws, 2, iters, 4);
            let (rcodes, rh) = reference_hals(&cents, &ws, 2, iters, 4);
            for (j, (a, b)) in codes.iter().zip(&rcodes).enumerate() {
                for (k, (&x, &y)) in a.iter().zip(b).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
                        "iters {iters} code[{j}][{k}]: {x} vs {y}"
                    );
                }
            }
            for (k, (a, b)) in h.iter().zip(&rh).enumerate() {
                for (c, (&x, &y)) in a.iter().zip(b).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
                        "iters {iters} h[{k}][{c}]: {x} vs {y}"
                    );
                }
            }
        }
    }

    /// Every HALS coordinate update is the exact nonnegative minimiser along that coordinate, so the
    /// Frobenius residual can never rise from one sweep to the next. Likewise the Lee–Seung
    /// multiplicative updates never raise the generalized KL divergence.
    #[test]
    fn both_solvers_descend_their_own_objective() {
        let (cents, ws) = hals_fixture();
        let d = cents[0].len();
        let sw: Vec<f64> = ws.iter().map(|&w| w.sqrt()).collect();
        let xf: Vec<Vec<f64>> = (0..cents.len())
            .map(|j| cents[j].iter().map(|&v| sw[j] * v).collect())
            .collect();

        let mut prev = f64::INFINITY;
        for t in 1..=8 {
            let (codes, h) = weighted_nmf(&cents, &ws, 2, t, 4);
            let w: Vec<Vec<f64>> = (0..codes.len())
                .map(|j| codes[j].iter().map(|&v| sw[j] * v).collect())
                .collect();
            let res = residual(&xf, &w, &h, d);
            assert!(res <= prev + 1e-9, "Frobenius residual rose at sweep {t}");
            prev = res;
        }

        let mut prev = f64::INFINITY;
        for t in 1..=8 {
            let (w, h) = weighted_nmf_kl(&cents, &ws, 2, t, 4);
            let kl = kl_divergence(&cents, &w, &h, d);
            assert!(
                kl <= prev + 1e-9,
                "KL divergence rose at sweep {t}: {prev} -> {kl}"
            );
            prev = kl;
        }
    }

    #[test]
    fn canonicalize_normalizes_h_preserves_wh_and_orders_by_energy() {
        let mut w = vec![vec![1.0, 4.0], vec![2.0, 1.0], vec![0.5, 3.0]];
        let mut h = vec![vec![3.0, 4.0, 0.0], vec![0.0, 0.0, 2.0]];
        let before: Vec<Vec<f64>> = (0..3)
            .map(|j| {
                (0..3)
                    .map(|c| (0..2).map(|k| w[j][k] * h[k][c]).sum())
                    .collect()
            })
            .collect();

        canonicalize(&mut w, &mut h);

        for (k, row) in h.iter().enumerate() {
            let n: f64 = row.iter().map(|&t| t * t).sum::<f64>().sqrt();
            assert!(
                (n - 1.0).abs() < 1e-12,
                "component {k} is not unit-norm: {n}"
            );
        }
        for j in 0..3 {
            for c in 0..3 {
                let got: f64 = (0..2).map(|k| w[j][k] * h[k][c]).sum();
                assert!(
                    (got - before[j][c]).abs() < 1e-12,
                    "W·H changed at ({j},{c})"
                );
            }
        }
        // Column 1 carries √(4²+1²+3²)·2 = √26·2 ≈ 10.2 of energy against column 0's √(1+4+0.25)·5
        // = 11.5, so the order is unchanged here; swapping the two inputs must swap the output.
        let energy: Vec<f64> = (0..2)
            .map(|k| w.iter().map(|row| row[k] * row[k]).sum())
            .collect();
        assert!(
            energy[0] >= energy[1],
            "components are not ordered by energy"
        );

        let mut w2 = vec![vec![4.0, 1.0], vec![1.0, 2.0], vec![3.0, 0.5]];
        let mut h2 = vec![vec![0.0, 0.0, 2.0], vec![3.0, 4.0, 0.0]];
        canonicalize(&mut w2, &mut h2);
        let e2: Vec<f64> = (0..2)
            .map(|k| w2.iter().map(|row| row[k] * row[k]).sum())
            .collect();
        assert!(e2[0] >= e2[1], "reordering did not fire on a swapped input");
        assert!(
            (h2[0][0] - 0.6).abs() < 1e-12,
            "the heavier component is not first: {:?}",
            h2[0]
        );
    }

    #[test]
    fn projection_error_is_the_relative_frobenius_norm() {
        let (cents, ws) = hals_fixture();
        let feats: Vec<Spherical<f64>> = cents
            .iter()
            .zip(&ws)
            .map(|(c, &w)| Spherical::from_moments(w, c.clone(), 0.0))
            .collect();
        let spec = NmfSpec {
            rank: 2,
            max_iter: 40,
            kl: false,
        };
        let out: Projection<f64> = project(&feats, spec, 4);

        let d = cents[0].len();
        let sw: Vec<f64> = ws.iter().map(|&w| w.sqrt()).collect();
        let x: Vec<Vec<f64>> = (0..cents.len())
            .map(|j| cents[j].iter().map(|&v| sw[j] * v).collect())
            .collect();
        let w: Vec<Vec<f64>> = (0..out.coded.len())
            .map(|j| out.coded[j].mean().iter().map(|&v| sw[j] * v).collect())
            .collect();
        let energy: f64 = x.iter().flatten().map(|&v| v * v).sum();
        let want = (residual(&x, &w, &out.components, d) / energy).sqrt();
        assert!(
            (out.reconstruction_err - want).abs() < 1e-9,
            "got {}, want {want}",
            out.reconstruction_err
        );
    }
}
