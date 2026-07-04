//! Shared self-tuning k-NN affinity graph over microcluster means.
//!
//! Both the spectral and community-detection heads cluster the same object — a similarity graph over
//! the leaf means — so they build it here, once, the same way: a symmetric k-NN graph (edges follow
//! the data manifold instead of a dense RBF bridging clusters through the ambient gap) weighted by
//! the self-tuning RBF of Zelnik-Manor & Perona (NIPS 2004). The degree scales down with the node
//! count so the graph stays local on coarse trees.

use crate::kernels::sq_euclidean;
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
    let n = centers.len();
    let tiny = R::from_f64(1e-12).unwrap();

    // Pairwise squared distances.
    let mut d2 = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = sq_euclidean(&centers[i], &centers[j]);
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
