//! Shared self-tuning k-NN affinity graph over microcluster means.
//!
//! Both the spectral and community-detection heads cluster the same object — a similarity graph over
//! the leaf means — so they build it here, once, the same way: a symmetric k-NN graph (edges follow
//! the data manifold instead of a dense RBF bridging clusters through the ambient gap) weighted by
//! the self-tuning RBF of Zelnik-Manor & Perona (NIPS 2004). The degree scales down with the node
//! count so the graph stays local on coarse trees.

use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::linalg::{frobenius_sq_diff, jacobi_eigen, matrix_log};
use crate::types::Real;

/// Max neighbours kept per node.
const KNN: usize = 10;
/// Floor on the k-NN degree so the graph stays connected on small node counts.
const MIN_KNN: usize = 4;
/// Neighbour rank for the self-tuning local scale `σ_i`.
const LOCAL_SCALE_NN: usize = 7;

/// Symmetric self-tuning k-NN affinity graph over `centers` (requires `n ≥ 2`), as a per-node
/// adjacency list of `(neighbour, weight)` with `weight = exp(-d²/(σ_i σ_j))`. Edges are the union
/// of each node's nearest neighbours, so `adj[i]` may list `j` because `i` kept `j` or `j` kept `i`;
/// weights are symmetric (`w_ij = w_ji`).
pub fn knn_affinity<R: Real>(centers: &[Vec<R>]) -> Vec<Vec<(usize, R)>> {
    knn_affinity_impl(centers, None, None)
}

/// Geometry-aware [`knn_affinity`] (GeoBETULA): the pairwise distance gains an optional log-Euclidean
/// **shape** term `β · ‖logΣ_i − logΣ_j‖²_F` *and* an optional **tangent** term
/// `γ · d²_Gr(U_i, U_j)` (projection-Grassmann distance between local `r`-dim principal subspaces).
/// So two microclusters must agree in centroid, covariance shape, **and** manifold orientation to be
/// neighbours — this separates crossing / adjacent manifolds that share a centroid neighbourhood.
/// Passing `None` for a term drops it; both `None` reproduces [`knn_affinity`] exactly.
pub fn knn_affinity_geo<R: Real>(
    centers: &[Vec<R>],
    cov: Option<(&[Vec<Vec<R>>], R)>,
    tangent: Option<(&[Vec<Vec<R>>], R)>,
) -> Vec<Vec<(usize, R)>> {
    knn_affinity_impl(centers, cov, tangent)
}

/// Squared projection (chordal) Grassmann distance between two `d×r` column-orthonormal bases:
/// `r − ‖AᵀB‖²_F ∈ [0, r]` — `0` when the subspaces coincide, `r` when orthogonal.
#[allow(clippy::needless_range_loop)] // Aᵀ B reads clearest with explicit (p, q, k) matrix indices
fn grassmann_sq<R: Real>(a: &[Vec<R>], b: &[Vec<R>]) -> R {
    let d = a.len();
    let r = a.first().map_or(0, |row| row.len());
    let mut fro = R::zero();
    for p in 0..r {
        for q in 0..r {
            let mut m = R::zero();
            for k in 0..d {
                m = m + a[k][p] * b[k][q];
            }
            fro = fro + m * m;
        }
    }
    (R::from_usize(r).unwrap() - fro).max(R::zero())
}

/// Per-microcluster local tangent basis: the top-`rank` eigenvectors of `Σ_i` (its principal
/// directions), as a `d×rank` column-orthonormal matrix, for [`knn_affinity_geo`]. `rank` is clamped
/// to `[1, d]`. Most informative for the `full` feature.
pub fn tangent_bases<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    rank: usize,
) -> Vec<Vec<Vec<R>>> {
    features
        .iter()
        .map(|f| {
            let cov = f.cov_dense();
            let d = cov.len();
            let r = rank.clamp(1, d.max(1));
            let (eig, v) = jacobi_eigen(&cov);
            // Indices of the `r` largest eigenvalues (jacobi_eigen returns them unsorted).
            let mut idx: Vec<usize> = (0..d).collect();
            idx.sort_by(|&a, &b| {
                eig[b]
                    .partial_cmp(&eig[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(r);
            (0..d)
                .map(|k| idx.iter().map(|&j| v[k][j]).collect())
                .collect()
        })
        .collect()
}

/// Per-microcluster matrix log of the covariance `Σ_i`, for [`knn_affinity_cov`]. Each covariance's
/// eigenvalues are floored relative to its own scale (`trace/d`) before the log, so a rank-deficient
/// or empty leaf still yields a finite `logΣ_i`. Most informative for the `full` feature (a diagonal
/// feature yields a diagonal `logΣ` = log-variances).
pub fn log_covariances<R: Real, C: ClusterFeature<R>>(features: &[C]) -> Vec<Vec<Vec<R>>> {
    let eps = R::from_f64(1e-8).unwrap();
    let tiny = R::from_f64(1e-12).unwrap();
    features
        .iter()
        .map(|f| {
            let cov = f.cov_dense();
            let d = cov.len();
            let trace = (0..d).map(|i| cov[i][i]).fold(R::zero(), |a, b| a + b);
            let scale = if d > 0 {
                trace / R::from_usize(d).unwrap()
            } else {
                R::zero()
            };
            matrix_log(&cov, (eps * scale).max(tiny))
        })
        .collect()
}

fn knn_affinity_impl<R: Real>(
    centers: &[Vec<R>],
    cov: Option<(&[Vec<Vec<R>>], R)>,
    tangent: Option<(&[Vec<Vec<R>>], R)>,
) -> Vec<Vec<(usize, R)>> {
    let n = centers.len();
    let tiny = R::from_f64(1e-12).unwrap();

    // Pairwise squared distances: centroid term + optional log-Euclidean shape + optional tangent.
    let mut d2 = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut v = sq_euclidean(&centers[i], &centers[j]);
            if let Some((lc, beta)) = cov {
                v = v + beta * frobenius_sq_diff(&lc[i], &lc[j]);
            }
            if let Some((ub, gamma)) = tangent {
                v = v + gamma * grassmann_sq(&ub[i], &ub[j]);
            }
            d2[i][j] = v;
            d2[j][i] = v;
        }
    }

    // Per node, sort neighbours once, then read off the self-tuning local scale `σ_i` (distance to
    // the `LOCAL_SCALE_NN`-th neighbour) and the `knn` nearest for the graph. `scale_rank` ≥ 1 since
    // n ≥ 2; `knn` scales with the node count so the graph stays local on small trees.
    let scale_rank = LOCAL_SCALE_NN.min(n - 1);
    let knn = (n / 10).clamp(MIN_KNN, KNN).min(n - 1);
    let mut sigma = vec![R::zero(); n];
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        idx.sort_by(|&x, &y| d2[i][x].partial_cmp(&d2[i][y]).unwrap());
        sigma[i] = d2[i][idx[scale_rank - 1]].sqrt().max(tiny);
        for &j in &idx[0..knn] {
            adj[i][j] = true; // symmetric k-NN: an edge if either endpoint keeps the other
            adj[j][i] = true;
        }
    }

    let mut out: Vec<Vec<(usize, R)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i != j && adj[i][j] {
                out[i].push((j, (-d2[i][j] / (sigma[i] * sigma[j])).exp()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::Full;

    #[test]
    fn cov_affinity_reduces_to_plain_when_beta_zero() {
        let centers = vec![
            vec![0.0_f64, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let log_covs = vec![vec![vec![0.7, 0.1], vec![0.1, 0.7]]; 4];
        assert_eq!(
            knn_affinity(&centers),
            knn_affinity_geo(&centers, Some((&log_covs, 0.0)), None)
        );
    }

    #[test]
    fn cov_affinity_shape_term_changes_the_graph() {
        let centers = vec![
            vec![0.0_f64, 0.0],
            vec![0.1, 0.0],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
        ];
        let same = vec![vec![vec![0.0; 2]; 2]; 4];
        let mut diff = same.clone();
        diff[1] = vec![vec![4.0, 0.0], vec![0.0, 4.0]]; // node 1 gets a very different logΣ
        let flat = |g: &[Vec<(usize, f64)>]| -> Vec<f64> {
            g.iter().flat_map(|r| r.iter().map(|&(_, w)| w)).collect()
        };
        assert_ne!(
            flat(&knn_affinity_geo(&centers, Some((&same, 1.0)), None)),
            flat(&knn_affinity_geo(&centers, Some((&diff, 1.0)), None)),
        );
    }

    #[test]
    fn log_covariances_are_finite_and_shaped() {
        let mut f = Full::<f64>::new(2);
        for p in [[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [-0.5, 0.4]] {
            f.push(&p, 1.0);
        }
        let lc = log_covariances(&[f]);
        assert_eq!(lc.len(), 1);
        assert_eq!(lc[0].len(), 2);
        assert!(lc[0].iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn grassmann_sq_zero_for_same_subspace_full_for_orthogonal() {
        let e1 = vec![vec![1.0_f64], vec![0.0]]; // x-axis (d=2, r=1)
        let e2 = vec![vec![0.0_f64], vec![1.0]]; // y-axis
        assert!(grassmann_sq(&e1, &e1).abs() < 1e-12);
        assert!((grassmann_sq(&e1, &e2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn grassmann_sq_sums_squares_over_a_genuinely_mixed_basis() {
        // Rank 1 with axis-aligned bases makes every inner product a single term and every square
        // its own value, so the Frobenius sum can be corrupted without moving the answer. Two
        // mixed columns at a 30-degree tilt give ‖AᵀB‖²_F four nonzero terms and a known result.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let a = vec![vec![s, -s], vec![s, s], vec![0.0, 0.0]];
        let (c, sn) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let b = vec![vec![c, 0.0], vec![0.0, 1.0], vec![sn, 0.0]];
        // A spans the xy-plane; B tilts one of its two axes out of it by 30 degrees, so the
        // squared projection distance is r − (cos²θ + 1) = sin²θ.
        let got = grassmann_sq(&a, &b);
        assert!((got - sn * sn).abs() < 1e-12, "{got} vs {}", sn * sn);
        assert!(grassmann_sq(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn tangent_bases_recover_principal_axis() {
        // Points along y ≈ 0.05·x: the rank-1 tangent basis is the long (≈ x) axis.
        let mut f = Full::<f64>::new(2);
        for x in [-3.0, -1.5, 0.0, 1.5, 3.0] {
            f.push(&[x, 0.05 * x], 1.0);
        }
        let bases = tangent_bases(&[f], 1);
        assert_eq!(bases[0].len(), 2); // d rows
        assert_eq!(bases[0][0].len(), 1); // r = 1 column
        assert!(
            bases[0][0][0].abs() > bases[0][1][0].abs(),
            "leading tangent should align with x"
        );
    }

    #[test]
    fn geo_affinity_tangent_term_changes_graph() {
        let centers = vec![
            vec![0.0_f64, 0.0],
            vec![0.1, 0.0],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
        ];
        let same = vec![vec![vec![1.0], vec![0.0]]; 4]; // all x-axis subspaces
        let mut diff = same.clone();
        diff[1] = vec![vec![0.0], vec![1.0]]; // node 1 perpendicular
        let flat = |g: &[Vec<(usize, f64)>]| -> Vec<f64> {
            g.iter().flat_map(|r| r.iter().map(|&(_, w)| w)).collect()
        };
        assert_ne!(
            flat(&knn_affinity_geo(&centers, None, Some((&same, 1.0)))),
            flat(&knn_affinity_geo(&centers, None, Some((&diff, 1.0)))),
        );
    }
}
