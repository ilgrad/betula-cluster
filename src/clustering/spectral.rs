//! Spectral clustering on the CF-tree leaf microclusters.
//!
//! Builds a self-tuning RBF affinity over the leaf means (local scaling of Zelnik-Manor & Perona,
//! NIPS 2004), forms the symmetric normalized affinity `P = D^{-1/2} A D^{-1/2}` whose top
//! eigenvectors are the bottom eigenvectors of the normalized Laplacian `L_sym = I − P`
//! (Ng-Jordan-Weiss, NIPS 2001), embeds each microcluster in the space of the `k` such
//! eigenvectors, row-normalizes, and k-means-clusters the embedding. This separates non-convex /
//! manifold clusters (rings, moons, spirals) that the centroid heads cannot.
//!
//! **Two solvers, one boundary.** Up to [`SPECTRAL_EXACT_NODES`] the eigenproblem is solved exactly
//! by the in-house cyclic-Jacobi routine (`O(M³)`; no LAPACK/ARPACK — the crate stays LEAN) on the
//! complete `M×M` affinity. Above it, neither the `O(M³)` solve nor the `O(M²)` distance matrix that
//! feeds it is affordable, and the head used to answer by *throwing resolution away*: reduce the
//! leaves to 256 weighted k-means landmarks, cluster those, and let every leaf inherit its
//! landmark's label — a second lossy summarisation stacked on the one the tree already did.
//!
//! It no longer does. Above the boundary the top-`k` eigenvectors come from **Chebyshev-filtered
//! subspace iteration**: the only thing it needs of `P` is the product `P x`, which is `O(nnz)` on
//! the sparse graph. A degree-`m` Chebyshev polynomial in `P`, evaluated by the three-term
//! recurrence, amplifies the wanted end of the spectrum by `cosh(m · arccosh σ)` while staying
//! bounded by 1 on the damped interval — that gap, not the matrix size, is what sets the iteration
//! count. Nothing `M×M` is ever formed. Past [`SPECTRAL_DENSE_GRAPH_MAX`] the graph itself comes
//! from the bounded-degree beam-search index too ([`knn_affinity_approx`], the same graph the
//! HDBSCAN head runs on), since the `O(M²)` distance matrix is the next thing to go.
//!
//! **Measured, A/B against the landmark path on the same trees, median of seeds 0/1/2.** Quality is
//! a tie or a win wherever the two differ at all, and the cost falls by 2–12×: `two-moons` and
//! `two-circles` at 20 000 points hold ARI 1.000 at every budget from 500 to 5000 leaves while the
//! fit drops from 0.34 s to 0.02 s at 500 and from 0.43 s to 0.36 s at 5000; `digits`-PCA20 goes
//! **0.660 → 0.779** at 500 leaves and **0.786 → 0.801** at 1000, for 0.36 s → 0.03 s and
//! 0.40 s → 0.06 s, and loses one budget, 0.766 → 0.735 at one leaf per point. `covtype` scores
//! within ±0.01 of zero on both paths, which is the head's documented failure case and not a
//! comparison.
//!
//! **Where it stops, and why it is not the approximation.** At 10 000 leaves the head still returns
//! ARI 1.000 on both non-convex fixtures in 0.7 s. At 20 000 — one leaf per point — it collapses to
//! ≈ 0.6. Forcing the *exact* `O(M²)` affinity there returns **0.603 / 0.594**, the same answer for
//! 17.2 s instead of 1.7: the approximate graph is not what breaks. The fixed `KNN = 10` is. It is
//! `1.0 · log n` at 20 000 nodes, at the connectivity threshold below which a k-NN Laplacian's
//! spectrum stops describing the manifold (see [`knn_degree`](super::graph::knn_degree)), and no
//! eigensolver recovers a graph that has already fragmented.
//!
//! Spectral has no built-in cluster-count selection (the eigengap is unreliable on k-NN graphs), so
//! `k == 0` (auto) falls back to [`SPECTRAL_DEFAULT_K`].

use crate::clustering::graph::{knn_affinity, knn_affinity_approx, knn_affinity_with_degree};
use crate::clustering::kmeans::kmeans;
use crate::clustering::rng::SplitMix64;
use crate::feature::{ClusterFeature, Spherical};
use crate::linalg::{jacobi_eigen, orthonormalize_rows};
use crate::types::Real;

/// Microclusters solved by the exact dense eigensolver on the exact `O(M²)` affinity. Above this
/// the sparse graph and the Chebyshev solver take over; there is no landmark reduction either side.
///
/// The boundary is where the dense route stops being the cheap one, not where it stops being
/// correct: `O(M³)` Jacobi at 256 is well under a second, and the `M×M` distance matrix the exact
/// affinity builds is 0.5 MB at 256, 134 MB at 4096 and 3.2 GB at 20 000.
pub const SPECTRAL_EXACT_NODES: usize = 256;
/// Nodes above which the affinity graph is built by approximate search rather than from the
/// complete distance matrix.
///
/// A second, independent boundary, because the exact graph and the exact solver stop being
/// affordable for different reasons and at different sizes. The `O(M³)` Jacobi is the first to go;
/// the `O(M²)` distance matrix survives well past it — 33 MB at 2048 — and is worth keeping, because
/// the approximate graph is where the remaining quality difference lives, not the solver.
pub const SPECTRAL_DENSE_GRAPH_MAX: usize = 2048;
/// Cluster count used when `k == 0` (auto) is requested — spectral has no reliable built-in
/// selection, and two clusters is the canonical non-convex case (moons / two rings).
pub const SPECTRAL_DEFAULT_K: usize = 2;
/// k-means restarts for the embedding (mirrors the k-means head).
const N_INIT: usize = 4;

/// Extra vectors carried above `k` in the subspace iteration.
///
/// Subspace iteration converges at the rate `(λ_{s+1}/λ_k)^m` in the block size `s`, so the padding
/// buys convergence speed directly; it also supplies the Ritz value `θ_{k+1}` the adaptive filter
/// bound needs, which a block of exactly `k` cannot.
const CHEB_OVERSAMPLE: usize = 4;
/// Degree of the Chebyshev polynomial applied per outer iteration — `CHEB_DEGREE` sparse products
/// per vector, buying `cosh(CHEB_DEGREE · arccosh σ)` amplification instead of the `σ^1` a plain
/// power iteration gets for the same one product.
const CHEB_DEGREE: usize = 12;
/// Outer iterations before the solver returns what it has. Reached only when the wanted and
/// unwanted parts of the spectrum are not separated, where no iteration count would help.
const CHEB_MAX_OUTER: usize = 40;
/// Accepted residual `‖P x − θ x‖` for every wanted Ritz pair.
const CHEB_TOL: f64 = 1e-8;
/// Keeps the subspace-init random stream apart from the k-means one drawn at the same seed.
const CHEB_STREAM_OFFSET: u64 = 0x9E37_79B9_7F4A_7C15;

/// Result of a spectral run: one cluster label per input microcluster.
pub struct Spectral {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
}

/// Spectral-cluster `features` into `k` groups (`k == 0` ⇒ [`SPECTRAL_DEFAULT_K`]). Every feature
/// is a node of the graph at every size; above [`SPECTRAL_EXACT_NODES`] the graph and the
/// eigenvectors are approximated, but the node set is not reduced.
pub fn spectral<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Spectral {
    assert!(!features.is_empty(), "spectral needs at least one feature");
    let k = if k == 0 { SPECTRAL_DEFAULT_K } else { k };
    let centers: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    Spectral {
        labels: spectral_core::<R>(&centers, k, max_iter, seed),
    }
}

/// Diffusion-maps density normalisation exponent `α` (Coifman & Lafon, *Diffusion maps*, ACHA 21(1),
/// 2006, §3). Replacing `A_ij` by `A_ij / (q_i^α q_j^α)` with `q_i = Σ_j A_ij` before the usual
/// symmetric normalisation interpolates between two different operators: `α = 0` is the
/// density-biased normalized Laplacian, `α = 1` divides the sampling density out entirely and the
/// limit operator is the Laplace–Beltrami of the underlying manifold, `α = 1/2` is the
/// Fokker–Planck point between them.
///
/// It matters more here than on raw points, because a leaf is a *mass-weighted* microcluster: where
/// the data is dense the tree puts more leaves **and** each leaf is tighter, so sampling density
/// enters the affinity twice.
///
/// `0.0` is the shipped value, and it is a measurement rather than an inheritance — see the
/// `measure_alpha_normalization` harness and `bench/RESULTS.md`.
const DIFFUSION_ALPHA: f64 = 0.0;

fn spectral_core<R: Real>(centers: &[Vec<R>], k: usize, max_iter: usize, seed: u64) -> Vec<usize> {
    spectral_core_alpha(centers, k, max_iter, seed, DIFFUSION_ALPHA, None)
}

fn spectral_core_alpha<R: Real>(
    centers: &[Vec<R>],
    k: usize,
    max_iter: usize,
    seed: u64,
    alpha: f64,
    degree: Option<usize>,
) -> Vec<usize> {
    let n = centers.len();
    if n == 1 {
        return vec![0];
    }
    if k <= 1 {
        return vec![0; n]; // single cluster
    }
    if k >= n {
        return (0..n).collect(); // more clusters requested than nodes ⇒ each its own
    }
    let tiny = R::from_f64(1e-12).unwrap();

    // Symmetric self-tuning k-NN affinity graph, kept sparse. `A` is exactly symmetric by
    // construction, so the normalized affinity `P = D^{-1/2} A D^{-1/2}` is too.
    //
    // The exact builder survives past the exact *solver*: what retires it is the `O(M²)` distance
    // matrix, at [`SPECTRAL_DENSE_GRAPH_MAX`], not the `O(M³)` eigendecomposition at
    // [`SPECTRAL_EXACT_NODES`]. A forced degree means the caller is measuring the graph itself, so
    // it gets the exact one whatever the size.
    let mut adj = match degree {
        Some(d) => knn_affinity_with_degree(centers, d),
        None if n <= SPECTRAL_DENSE_GRAPH_MAX => knn_affinity(centers),
        None => knn_affinity_approx(centers, seed),
    };
    if alpha != 0.0 {
        // `q` is the kernel density estimate the affinity itself induces; dividing it out is what
        // separates the geometry from the sampling. It must be taken *before* any weight is
        // rewritten, or later rows would be normalised against already-normalised ones.
        let exponent = R::from_f64(alpha).unwrap_or_else(R::zero);
        let q: Vec<R> = adj
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(_, w)| w)
                    .sum::<R>()
                    .max(tiny)
                    .powf(exponent)
            })
            .collect();
        for (i, row) in adj.iter_mut().enumerate() {
            for (j, w) in row.iter_mut() {
                *w = *w / (q[i] * q[*j]);
            }
        }
    }
    let p = normalized_affinity(&adj);

    // Top-`k` eigenvectors of `P` are the bottom-`k` of `L_sym = I − P` (here `2 ≤ k < n`).
    let vecs = if n <= SPECTRAL_EXACT_NODES {
        dense_top_eigenvectors(&p, k)
    } else {
        chebyshev_top_eigenvectors(&p, k, seed)
    };

    // Row-normalized eigenvector embedding (Ng-Jordan-Weiss), then k-means on the rows.
    let embed: Vec<Spherical<R>> = vecs
        .into_iter()
        .map(|mut row| {
            let norm = row.iter().map(|&x| x * x).sum::<R>().sqrt().max(tiny);
            for x in row.iter_mut() {
                *x = *x / norm;
            }
            let mut f = Spherical::new(k);
            f.push(&row, R::one());
            f
        })
        .collect();
    kmeans(&embed, k, max_iter, N_INIT, seed).labels
}

/// `P = D^{-1/2} A D^{-1/2}` from a sparse symmetric affinity, in the same sparse layout.
fn normalized_affinity<R: Real>(adj: &[Vec<(usize, R)>]) -> Vec<Vec<(usize, R)>> {
    let tiny = R::from_f64(1e-12).unwrap();
    let dinv: Vec<R> = adj
        .iter()
        .map(|row| R::one() / row.iter().map(|&(_, w)| w).sum::<R>().max(tiny).sqrt())
        .collect();
    adj.iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .map(|&(j, w)| (j, w * dinv[i] * dinv[j]))
                .collect()
        })
        .collect()
}

/// `y ← P x` for the sparse symmetric `P`.
fn spmv<R: Real>(p: &[Vec<(usize, R)>], x: &[R], y: &mut [R]) {
    for (yi, row) in y.iter_mut().zip(p) {
        *yi = row
            .iter()
            .map(|&(j, w)| w * x[j])
            .fold(R::zero(), |a, b| a + b);
    }
}

fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y)
        .fold(R::zero(), |p, q| p + q)
}

/// The `k` leading eigenvectors of `P`, exactly, as `n` rows of `k` components.
///
/// Expands the sparse graph into the dense matrix cyclic Jacobi needs. Only reachable at
/// `n ≤ SPECTRAL_EXACT_NODES`, where that matrix is at most 0.5 MB.
fn dense_top_eigenvectors<R: Real>(p: &[Vec<(usize, R)>], k: usize) -> Vec<Vec<R>> {
    let n = p.len();
    let mut dense = vec![vec![R::zero(); n]; n];
    for (i, row) in p.iter().enumerate() {
        for &(j, w) in row {
            dense[i][j] = w;
        }
    }
    let (eigvals, vecs) = jacobi_eigen(&dense);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());
    let sel = &order[0..k];
    (0..n)
        .map(|i| sel.iter().map(|&c| vecs[i][c]).collect())
        .collect()
}

/// One degree-[`CHEB_DEGREE`] Chebyshev polynomial of `P`, applied in place to every row of `x`.
///
/// The affine map `σ(λ) = (λ − c)/e` with `c = (lo+hi)/2`, `e = (hi−lo)/2` sends `[lo, hi]` onto
/// `[−1, 1]`, where `|T_m| ≤ 1`; every eigenvalue above `hi` lands outside, where
/// `T_m(σ) = cosh(m · arccosh σ)` grows exponentially in `m`. So the polynomial multiplies the
/// wanted end of the spectrum by a large factor and leaves the damped interval alone — that ratio,
/// not the matrix size, is what the subspace iteration converges on.
///
/// The three-term recurrence `T_{m+1} = 2σT_m − T_{m−1}` is evaluated on vectors, so the cost is
/// `CHEB_DEGREE` sparse products per row and nothing is ever squared. Its own growth is the reason
/// for the rescale: `T_12(3) ≈ 1.4·10⁸` already, and a narrow damping interval sends `σ` much higher
/// than 3. Dividing the current *and* previous terms by the same factor leaves the recurrence exact
/// — it is linear — and the span, which is all the caller wants, is unchanged by any scaling.
fn chebyshev_filter<R: Real>(p: &[Vec<(usize, R)>], x: &mut [Vec<R>], lo: R, hi: R) {
    let n = p.len();
    let two = R::one() + R::one();
    let c = (lo + hi) / two;
    let e = ((hi - lo) / two).max(R::from_f64(1e-6).unwrap());
    let huge = R::from_f64(1e150).unwrap();
    let mut work = vec![R::zero(); n];
    for v in x.iter_mut() {
        let mut prev = std::mem::take(v);
        spmv(p, &prev, &mut work);
        let mut cur: Vec<R> = work
            .iter()
            .zip(&prev)
            .map(|(&w, &t)| (w - c * t) / e)
            .collect();
        for _ in 1..CHEB_DEGREE {
            spmv(p, &cur, &mut work);
            let mut next: Vec<R> = work
                .iter()
                .zip(&cur)
                .zip(&prev)
                .map(|((&w, &y), &z)| two * (w - c * y) / e - z)
                .collect();
            let mx = next.iter().fold(R::zero(), |m, &t| m.max(t.abs()));
            if mx > huge {
                for t in next.iter_mut() {
                    *t = *t / mx;
                }
                for t in cur.iter_mut() {
                    *t = *t / mx;
                }
            }
            prev = cur;
            cur = next;
        }
        *v = cur;
    }
}

/// The `k` leading eigenvectors of `P` by Chebyshev-filtered subspace iteration, as `n` rows of `k`
/// components, ordered by descending Ritz value.
///
/// Each outer step filters the block, re-orthonormalises it, and takes the Rayleigh–Ritz rotation
/// from the `s×s` projected matrix `H = XᵀPX` — small enough that the same cyclic-Jacobi routine the
/// dense path uses for the whole problem handles it in microseconds. The filter's damping bound is
/// then reset to `θ_{k+1}`, the largest Ritz value that is *not* wanted, which is the closest thing
/// to `λ_{k+1}` available without knowing the answer. It is clamped away from both ends of the
/// spectrum: at `hi → −1` the interval collapses and `σ` overflows, at `hi → 1` there is nothing
/// left outside the interval to amplify.
fn chebyshev_top_eigenvectors<R: Real>(p: &[Vec<(usize, R)>], k: usize, seed: u64) -> Vec<Vec<R>> {
    let n = p.len();
    let s = (k + CHEB_OVERSAMPLE).min(n);
    let mut rng = SplitMix64::new(seed ^ CHEB_STREAM_OFFSET);
    let mut x: Vec<Vec<R>> = (0..s)
        .map(|_| {
            (0..n)
                .map(|_| R::from_f64(rng.gauss()).unwrap_or_else(R::zero))
                .collect()
        })
        .collect();
    orthonormalize_rows(&mut x);

    let lo = -R::one();
    let mut hi = R::zero();
    let clamp_lo = R::from_f64(-0.9).unwrap();
    let clamp_hi = R::from_f64(0.95).unwrap();
    let tol = R::from_f64(CHEB_TOL).unwrap();
    let mut px = vec![vec![R::zero(); n]; s];
    let mut theta = vec![R::zero(); s];

    for _ in 0..CHEB_MAX_OUTER {
        chebyshev_filter(p, &mut x, lo, hi);
        orthonormalize_rows(&mut x);
        for (row, v) in px.iter_mut().zip(&x) {
            spmv(p, v, row);
        }
        let mut h = vec![vec![R::zero(); s]; s];
        for a in 0..s {
            for b in 0..=a {
                let v = dot(&x[a], &px[b]);
                h[a][b] = v;
                h[b][a] = v;
            }
        }
        let (vals, q) = jacobi_eigen(&h);
        let mut order: Vec<usize> = (0..s).collect();
        order.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());

        // Rotate the block and its image together: `P(XQ) = (PX)Q`, so the residual below costs no
        // extra sparse products.
        let mut nx = vec![vec![R::zero(); n]; s];
        let mut npx = vec![vec![R::zero(); n]; s];
        for (r, &col) in order.iter().enumerate() {
            for a in 0..s {
                let w = q[a][col];
                if w == R::zero() {
                    continue;
                }
                for t in 0..n {
                    nx[r][t] = nx[r][t] + w * x[a][t];
                    npx[r][t] = npx[r][t] + w * px[a][t];
                }
            }
            theta[r] = vals[col];
        }
        x = nx;
        px = npx;

        let worst = (0..k.min(s))
            .map(|r| {
                px[r]
                    .iter()
                    .zip(&x[r])
                    .map(|(&pv, &v)| (pv - theta[r] * v) * (pv - theta[r] * v))
                    .sum::<R>()
                    .sqrt()
            })
            .fold(R::zero(), R::max);
        if worst <= tol {
            break;
        }
        hi = if s > k { theta[k] } else { R::zero() }
            .max(clamp_lo)
            .min(clamp_hi);
    }
    (0..n)
        .map(|t| (0..k).map(|r| x[r.min(s - 1)][t]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros, two_moons};
    use std::collections::HashSet;

    fn n_distinct(labels: &[usize]) -> usize {
        labels.iter().copied().collect::<HashSet<_>>().len()
    }

    /// Two concentric rings — the canonical case where the graph, not the centroid, is the model.
    fn circles(rng: &mut SplitMix64, per: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, r) in [(0usize, 1.0f64), (1, 2.6)] {
            for i in 0..per {
                let t = std::f64::consts::TAU * (i as f64) / (per as f64);
                pts.push(vec![
                    r * t.cos() + noise * rng.gauss(),
                    r * t.sin() + noise * rng.gauss(),
                ]);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    /// Blobs sheared into long thin ribbons: the Euclidean heads' worst shape, and a graph whose
    /// local scales differ wildly along and across the ribbon.
    fn aniso(rng: &mut SplitMix64, per: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in [(0usize, [0.0f64, 0.0]), (1, [3.0, 3.0]), (2, [-3.0, 4.0])] {
            for _ in 0..per {
                let (u, v) = (2.5 * rng.gauss(), 0.25 * rng.gauss());
                pts.push(vec![ctr[0] + u + 0.6 * v, ctr[1] + 0.35 * u + v]);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    /// Two moons sampled at different rates. This is the case `α` exists for: the affinity's own
    /// density estimate differs by a factor of four between the two clusters, and `α = 1` is
    /// supposed to divide exactly that out.
    fn lopsided_moons(rng: &mut SplitMix64, per: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
        thinned_moons(rng, per, noise, 4)
    }

    fn thinned_moons(
        rng: &mut SplitMix64,
        per: usize,
        noise: f64,
        thin: usize,
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let (pts, truth) = two_moons(rng, per, noise);
        let mut kept_pts = Vec::new();
        let mut kept_truth = Vec::new();
        // Count within the class, not over the interleaved list: `two_moons` alternates the two
        // moons, so thinning on the global index would drop one class entirely.
        let mut seen = 0usize;
        for (p, &t) in pts.iter().zip(&truth) {
            let keep = if t == 0 {
                true
            } else {
                seen += 1;
                seen % thin == 0
            };
            if keep {
                kept_pts.push(p.clone());
                kept_truth.push(t);
            }
        }
        (kept_pts, kept_truth)
    }

    #[test]
    fn the_shipped_head_is_the_alpha_the_constant_names() {
        let mut rng = SplitMix64::new(3);
        let (pts, _) = circles(&mut rng, 220, 0.08);
        let (micros, _) = grid_micros(&pts, 0.16);
        let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        assert_eq!(
            spectral(&micros, 2, 100, 1).labels,
            spectral_core_alpha(&centers, 2, 100, 1, DIFFUSION_ALPHA, None),
            "the head and the constant have drifted apart"
        );
    }

    #[test]
    fn alpha_normalisation_changes_the_operator_it_is_supposed_to_change() {
        // Without this the `alpha != 0.0` branch could be dead and every measurement below would be
        // measuring the same operator three times.
        let mut rng = SplitMix64::new(5);
        let (pts, _) = lopsided_moons(&mut rng, 400, 0.06);
        let (micros, _) = grid_micros(&pts, 0.1);
        let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        assert_ne!(
            spectral_core_alpha(&centers, 2, 100, 1, 0.0, None),
            spectral_core_alpha(&centers, 2, 100, 1, 1.0, None),
            "alpha = 1 produced the same labelling as alpha = 0"
        );
    }

    /// Is the shipped neighbour count load-bearing, and does the Γ-convergence floor bite? Median
    /// ARI of seeds 0/1/2 per (fixture, degree). `knn_degree` returns 10 at every node count the
    /// spectral head can reach, so the comparison is against fixed alternatives around it.
    ///
    /// `cargo test --lib clustering::spectral::tests::measure_graph_degree -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn measure_graph_degree() {
        type Fixture = fn(&mut SplitMix64, usize, f64) -> (Vec<Vec<f64>>, Vec<usize>);
        let cases: [(&str, Fixture, usize, f64, f64, usize); 4] = [
            ("two-moons", two_moons, 400, 0.06, 0.10, 2),
            ("lopsided-moons", lopsided_moons, 400, 0.06, 0.10, 2),
            ("circles", circles, 400, 0.08, 0.16, 2),
            ("aniso", |r, per, _| aniso(r, per), 400, 0.0, 0.30, 3),
        ];
        let degrees = [3usize, 5, 7, 10, 16, 24];
        print!("\n{:>16} {:>7}", "fixture", "nodes");
        for d in degrees {
            print!("{:>9}", format!("k={d}"));
        }
        println!("   (shipped k = 10)");
        for (name, build, per, noise, cell, k) in cases {
            let mut row = Vec::new();
            let mut nodes = 0usize;
            for degree in degrees {
                let mut scores = Vec::new();
                for seed in 0u64..3 {
                    let mut rng = SplitMix64::new(seed);
                    let (pts, truth) = build(&mut rng, per, noise);
                    let (micros, assign) = grid_micros(&pts, cell);
                    let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
                    nodes = centers.len();
                    let lab = spectral_core_alpha(&centers, k, 100, seed, 0.0, Some(degree));
                    let per_point: Vec<usize> = assign.iter().map(|&m| lab[m]).collect();
                    scores.push(ari(&per_point, &truth));
                }
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                row.push(scores[1]);
            }
            print!("{name:>16} {nodes:>7}");
            for v in &row {
                print!("{v:>9.4}");
            }
            println!();
        }
    }

    /// Does dividing the sampling density out of the affinity help this crate's mass-weighted
    /// leaves? Median ARI of seeds 0/1/2 per (fixture, alpha).
    ///
    /// `cargo test --lib clustering::spectral::tests::measure_alpha -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn measure_alpha_normalization() {
        type Fixture = fn(&mut SplitMix64, usize, f64) -> (Vec<Vec<f64>>, Vec<usize>);
        let cases: [(&str, Fixture, usize, f64, f64, usize); 4] = [
            ("two-moons", two_moons, 400, 0.06, 0.10, 2),
            ("lopsided-moons", lopsided_moons, 400, 0.06, 0.10, 2),
            ("circles", circles, 400, 0.08, 0.16, 2),
            ("aniso", |r, per, _| aniso(r, per), 400, 0.0, 0.30, 3),
        ];
        println!(
            "\n{:>16} {:>8} {:>10} {:>10} {:>10}",
            "fixture", "leaves", "a=0", "a=0.5", "a=1"
        );
        for (name, build, per, noise, cell, k) in cases {
            let mut row = Vec::new();
            let mut leaves = 0usize;
            for alpha in [0.0f64, 0.5, 1.0] {
                let mut scores = Vec::new();
                for seed in 0u64..3 {
                    let mut rng = SplitMix64::new(seed);
                    let (pts, truth) = build(&mut rng, per, noise);
                    let (micros, assign) = grid_micros(&pts, cell);
                    leaves = micros.len();
                    let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
                    let lab = spectral_core_alpha(&centers, k, 100, seed, alpha, None);
                    let per_point: Vec<usize> = assign.iter().map(|&m| lab[m]).collect();
                    scores.push(ari(&per_point, &truth));
                }
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                row.push(scores[1]);
            }
            println!(
                "{name:>16} {leaves:>8} {:>10.4} {:>10.4} {:>10.4}",
                row[0], row[1], row[2]
            );
        }

        // The regime `α` exists for: one moon sampled `thin` times more sparsely than the other.
        println!(
            "\n{:>16} {:>8} {:>10} {:>10} {:>10}",
            "moons 1:thin", "noise", "a=0", "a=0.5", "a=1"
        );
        for thin in [4usize, 10, 25] {
            for noise in [0.06f64, 0.10, 0.14] {
                let mut row = Vec::new();
                for alpha in [0.0f64, 0.5, 1.0] {
                    let mut scores = Vec::new();
                    for seed in 0u64..3 {
                        let mut rng = SplitMix64::new(seed);
                        let (pts, truth) = thinned_moons(&mut rng, 400, noise, thin);
                        let (micros, assign) = grid_micros(&pts, 0.10);
                        let centers: Vec<Vec<f64>> =
                            micros.iter().map(|f| f.mean().to_vec()).collect();
                        let lab = spectral_core_alpha(&centers, 2, 100, seed, alpha, None);
                        let per_point: Vec<usize> = assign.iter().map(|&m| lab[m]).collect();
                        scores.push(ari(&per_point, &truth));
                    }
                    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    row.push(scores[1]);
                }
                println!(
                    "{:>16} {noise:>8.2} {:>10.4} {:>10.4} {:>10.4}",
                    format!("1:{thin}"),
                    row[0],
                    row[1],
                    row[2]
                );
            }
        }
    }

    #[test]
    fn spectral_separates_two_moons_where_kmeans_cannot() {
        let mut rng = SplitMix64::new(7);
        let (pts, truth) = two_moons(&mut rng, 250, 0.05);
        let (micros, point_to_micro) = grid_micros(&pts, 0.12);
        let micro_labels = spectral(&micros, 2, 100, 1).labels;
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| micro_labels[m]).collect();
        assert_eq!(n_distinct(&micro_labels), 2);
        // The non-convex moons are recovered — a centroid head (k-means) scores ~0 here.
        assert!(ari(&pred, &truth) > 0.85, "moons ARI too low");
    }

    #[test]
    fn spectral_auto_k_defaults_to_two() {
        // k == 0 (auto) ⇒ SPECTRAL_DEFAULT_K = 2, which is exactly right for the two moons.
        let mut rng = SplitMix64::new(5);
        let (pts, truth) = two_moons(&mut rng, 250, 0.05);
        let (micros, point_to_micro) = grid_micros(&pts, 0.12);
        let micro_labels = spectral(&micros, 0, 100, 1).labels;
        assert_eq!(n_distinct(&micro_labels), SPECTRAL_DEFAULT_K);
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| micro_labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.85, "auto-k moons ARI too low");
    }

    #[test]
    fn spectral_more_clusters_than_nodes_is_identity() {
        let (micros, _) = grid_micros(&[vec![0.0, 0.0], vec![9.0, 0.0], vec![0.0, 9.0]], 0.5);
        let labels = spectral(&micros, 5, 100, 1).labels;
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn spectral_single_feature_is_one_cluster() {
        let (micros, _) = grid_micros(&[vec![1.0, 2.0]], 1.0);
        assert_eq!(spectral(&micros, 1, 100, 1).labels, vec![0]);
    }

    #[test]
    fn spectral_k_one_collapses_to_a_single_cluster() {
        let (micros, _) = grid_micros(&[vec![0.0, 0.0], vec![9.0, 0.0], vec![0.0, 9.0]], 0.5);
        let labels = spectral(&micros, 1, 100, 1).labels;
        assert_eq!(labels, vec![0, 0, 0]);
    }

    /// Largest principal angle between two `n×k` column-orthonormal bases, as `sin θ_max`.
    ///
    /// Eigenvectors are only defined up to sign, and up to an arbitrary rotation inside any
    /// degenerate eigenspace — a k-NN Laplacian has plenty of near-degenerate pairs. Comparing the
    /// vectors would therefore fail on a correct solver. The *subspace* they span is what the
    /// embedding actually uses, and it is unique; `sin θ_max` is zero exactly when the two spans
    /// coincide.
    fn sin_theta_max(u: &[Vec<f64>], v: &[Vec<f64>]) -> f64 {
        let k = u[0].len();
        let mut m = vec![vec![0.0; k]; k];
        for (ur, vr) in u.iter().zip(v) {
            for a in 0..k {
                for b in 0..k {
                    m[a][b] += ur[a] * vr[b];
                }
            }
        }
        // σ_min(M) via the smallest eigenvalue of MᵀM — k is 2..8 here, so the k×k Jacobi is free.
        let mut g = vec![vec![0.0; k]; k];
        for a in 0..k {
            for b in 0..k {
                g[a][b] = (0..k).map(|t| m[t][a] * m[t][b]).sum();
            }
        }
        let (vals, _) = jacobi_eigen(&g);
        let smallest = vals.iter().copied().fold(f64::INFINITY, f64::min);
        (1.0 - smallest.clamp(0.0, 1.0)).max(0.0).sqrt()
    }

    /// The acceptance check for the Chebyshev solver: on a graph the exact solver can still handle,
    /// the two must return the same invariant subspace — as accurately as the eigengap permits.
    ///
    /// The bound is Davis–Kahan's, not a tuned constant: `sin θ_max ≤ ‖R‖_F / gap`, where `R` is the
    /// residual block `P V − V Θ` and `gap = λ_k − λ_{k+1}`. That distinction matters here because
    /// the gap varies by an order of magnitude across these three fixtures — `aniso` closes to
    /// `1.6·10⁻³` where `two-moons` sits at `1.2·10⁻²` — and the measured subspace angle tracks it,
    /// `1.5·10⁻⁵` against `1.5·10⁻⁸`. A fixed tolerance would either pass a broken solver on the
    /// well-separated fixture or fail a correct one on the tight fixture; the inequality does
    /// neither, and it is the same theorem that says a near-degenerate eigenspace has no
    /// individually meaningful eigenvectors to compare in the first place.
    ///
    /// This is the test that would fail if the filter interval, the Rayleigh–Ritz rotation or the
    /// residual criterion were wrong — every one of which produces a plausible embedding and a
    /// plausible ARI while pointing somewhere else in eigenspace.
    #[test]
    fn the_chebyshev_solver_finds_the_subspace_the_exact_one_does() {
        // Cell sizes chosen so every fixture lands near 100 nodes: the dense reference is `O(n³)`
        // and this test runs in the default suite.
        for (name, k, cell) in [
            ("moons", 2usize, 0.18),
            ("aniso", 3, 0.40),
            ("circles", 2, 0.24),
        ] {
            let mut rng = SplitMix64::new(7);
            let (pts, _) = match name {
                "moons" => two_moons(&mut rng, 300, 0.06),
                "circles" => circles(&mut rng, 300, 0.08),
                _ => aniso(&mut rng, 300),
            };
            let (micros, _) = grid_micros(&pts, cell);
            let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
            let p = normalized_affinity(&knn_affinity(&centers));
            let exact = dense_top_eigenvectors(&p, k);
            let cheb = chebyshev_top_eigenvectors(&p, k, 3);
            let angle = sin_theta_max(&exact, &cheb);

            // `‖R‖_F` for the returned block, at its own Rayleigh quotients.
            let n = p.len();
            let mut resid_sq = 0.0;
            let columns: Vec<Vec<f64>> = (0..k)
                .map(|c| cheb.iter().map(|row| row[c]).collect())
                .collect();
            for v in &columns {
                let mut pv = vec![0.0; n];
                spmv(&p, v, &mut pv);
                let theta: f64 = v.iter().zip(&pv).map(|(a, b)| a * b).sum();
                resid_sq += pv
                    .iter()
                    .zip(v)
                    .map(|(&y, &x)| (y - theta * x) * (y - theta * x))
                    .sum::<f64>();
            }
            let resid = resid_sq.sqrt();

            let mut dense = vec![vec![0.0; n]; n];
            for (i, row) in p.iter().enumerate() {
                for &(j, w) in row {
                    dense[i][j] = w;
                }
            }
            let (mut vals, _) = jacobi_eigen(&dense);
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let gap = vals[k - 1] - vals[k];

            assert!(
                resid < 1e-6,
                "{name}: the block did not converge, ‖R‖_F = {resid:e}"
            );
            assert!(
                angle <= resid / gap,
                "{name}: sin θ_max = {angle:e} exceeds Davis–Kahan ‖R‖_F/gap = {:e}",
                resid / gap
            );
        }
    }

    /// The filter must not be what decides the answer. Different random starts explore different
    /// Krylov spaces; if the iteration has converged they still land on the same invariant subspace.
    #[test]
    fn the_chebyshev_subspace_does_not_depend_on_the_random_start() {
        let mut rng = SplitMix64::new(21);
        let (pts, _) = two_moons(&mut rng, 300, 0.06);
        let (micros, _) = grid_micros(&pts, 0.18);
        let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let p = normalized_affinity(&knn_affinity(&centers));
        let a = chebyshev_top_eigenvectors(&p, 2, 1);
        let b = chebyshev_top_eigenvectors(&p, 2, 99);
        assert!(sin_theta_max(&a, &b) < 1e-6);
    }

    #[test]
    fn the_sparse_solver_keeps_the_blobs_above_the_exact_cap() {
        // > SPECTRAL_EXACT_NODES microclusters route through the approximate graph and the
        // Chebyshev solver. Every microcluster stays a node — there is no landmark reduction —
        // so the label vector is as long as the input, which the old path also guaranteed but
        // for a different reason.
        let mut rng = SplitMix64::new(11);
        let centers = [[0.0, 0.0], [15.0, 0.0], [0.0, 15.0]];
        let (pts, truth) = blobs(&mut rng, 700, &centers, 0.7);
        let (micros, point_to_micro) = grid_micros(&pts, 0.12);
        assert!(
            micros.len() > SPECTRAL_EXACT_NODES,
            "need > cap microclusters"
        );
        let micro_labels = spectral(&micros, 3, 100, 1).labels;
        assert_eq!(micro_labels.len(), micros.len());
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| micro_labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.9, "sparse spectral lost the blobs");
    }

    /// Three groups of very different size. Eigenvector entries scale like `1/sqrt(size)`, so the
    /// raw embedding rows of the small groups are several times longer than the large group's --
    /// which is exactly the magnitude the Ng-Jordan-Weiss row normalization exists to remove.
    fn lopsided_groups() -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(404);
        let spec = [
            ([0.0f64, 0.0], 80usize),
            ([14.0, 0.0], 10),
            ([7.0, 13.0], 5),
        ];
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, (ctr, count)) in spec.iter().enumerate() {
            for _ in 0..*count {
                pts.push(vec![
                    ctr[0] + 0.35 * rng.gauss(),
                    ctr[1] + 0.35 * rng.gauss(),
                ]);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    #[test]
    fn the_embedding_is_row_normalized_before_the_final_kmeans() {
        // Without the normalization the k-means at the end separates rows by *length* -- the large
        // group's short rows cluster together with whichever small group is nearest the origin --
        // so an exact partition, not an ARI threshold, is what makes the step visible.
        let (pts, truth) = lopsided_groups();
        let labels = spectral_core(&pts, 3, 100, 5);
        assert_eq!(n_distinct(&labels), 3, "{labels:?}");
        assert!(
            (ari(&labels, &truth) - 1.0).abs() < 1e-12,
            "ARI = {}",
            ari(&labels, &truth)
        );
    }
}
