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
    let r = rank.min(d).max(1);
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

    // Scale-aware random nonnegative init (deterministic per seed).
    let mut total = R::zero();
    let mut cnt = 0usize;
    for row in &x {
        for &v in row {
            total = total + v;
            cnt += 1;
        }
    }
    let mean = total / R::from_usize(cnt.max(1)).unwrap();
    let scale = (mean / R::from_usize(r).unwrap())
        .max(R::from_f64(1e-6).unwrap())
        .sqrt();
    let mut rng = SplitMix64::new(seed);
    let mut w = vec![vec![R::zero(); r]; m];
    for row in w.iter_mut() {
        for v in row.iter_mut() {
            *v = R::from_f64(rng.next_f64()).unwrap() * scale;
        }
    }
    let mut h = vec![vec![R::zero(); d]; r];
    for row in h.iter_mut() {
        for v in row.iter_mut() {
            *v = R::from_f64(rng.next_f64()).unwrap() * scale;
        }
    }

    let tol = R::from_f64(1e-4).unwrap();
    let mut prev = R::infinity();
    for it in 0..max_iter {
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
                w[j][k] = (s / hht[k][k].max(eps)).max(R::zero());
            }
        }
        // ── update H (rows): C = Wᵀ X (r×d), G = Wᵀ W (r×r) ──
        let wtw = gram_cols(&w, r);
        let wtx = build_rows(r, |k| {
            (0..d)
                .map(|c| {
                    (0..m)
                        .map(|j| w[j][k] * x[j][c])
                        .fold(R::zero(), |a, b| a + b)
                })
                .collect()
        });
        for k in 0..r {
            for c in 0..d {
                let mut s = wtx[k][c];
                for l in 0..r {
                    s = s - wtw[k][l] * h[l][c];
                }
                s = s + wtw[k][k] * h[k][c];
                h[k][c] = (s / wtw[k][k].max(eps)).max(R::zero());
            }
        }
        // ── convergence check every few sweeps ──
        if it % 5 == 4 {
            let err = residual(&x, &w, &h, d);
            if (prev - err).abs() <= tol * prev.max(R::one()) {
                break;
            }
            prev = err;
        }
    }

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

/// Project leaf microclusters to `rank`-dimensional CF-weighted NMF codes, returned as single-point
/// `Spherical` features (mean = code, weight = original mass) for a downstream Phase-3 head to cluster.
pub(crate) fn project_features<R, C>(
    feats: &[C],
    rank: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<Spherical<R>>
where
    R: Real,
    C: ClusterFeature<R>,
{
    if feats.is_empty() {
        return Vec::new();
    }
    let centroids: Vec<Vec<R>> = feats.iter().map(|f| f.mean().to_vec()).collect();
    let weights: Vec<R> = feats.iter().map(|f| f.weight()).collect();
    let (codes, _components) = weighted_nmf(&centroids, &weights, rank, max_iter.max(1), seed);
    codes
        .into_iter()
        .zip(&weights)
        .map(|(code, &w)| Spherical::from_moments(w, code, R::zero()))
        .collect()
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
        let coded = project_features(&feats, 2, 100, 3);
        assert_eq!(coded.len(), 3);
        assert!(coded.iter().all(|f| f.dim() == 2));
        assert!(coded.iter().all(|f| f.weight() == 1.0));
        assert!(coded
            .iter()
            .all(|f| f.mean().iter().all(|&v| v >= 0.0 && v.is_finite())));
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
}
