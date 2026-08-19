//! Spectral clustering on the CF-tree leaf microclusters.
//!
//! Builds a self-tuning RBF affinity over the leaf means (local scaling of Zelnik-Manor & Perona,
//! NIPS 2004), forms the symmetric normalized affinity `P = D^{-1/2} A D^{-1/2}` whose top
//! eigenvectors are the bottom eigenvectors of the normalized Laplacian `L_sym = I − P`
//! (Ng-Jordan-Weiss, NIPS 2001), embeds each microcluster in the space of the `k` such
//! eigenvectors, row-normalizes, and k-means-clusters the embedding. This separates non-convex /
//! manifold clusters (rings, moons, spirals) that the centroid heads cannot.
//!
//! The eigensolver is the in-house cyclic-Jacobi routine (`O(M³)`; no LAPACK/ARPACK — the crate
//! stays LEAN), so the microcluster count handed to it is capped: above [`SPECTRAL_MAX_NODES`] the
//! leaves are first reduced to that many weighted k-means landmarks, spectral-clustered, and each
//! leaf inherits its landmark's label. Spectral has no built-in cluster-count selection (the
//! eigengap is unreliable on k-NN graphs), so `k == 0` (auto) falls back to [`SPECTRAL_DEFAULT_K`].

use crate::clustering::graph::knn_affinity;
use crate::clustering::kmeans::kmeans;
use crate::feature::{ClusterFeature, Spherical};
use crate::linalg::jacobi_eigen;
use crate::types::Real;

/// Microclusters solved directly by the eigensolver; above this, reduce to this many k-means
/// landmarks first. Keeps the `O(M³)` Jacobi eigendecomposition well under a second.
pub const SPECTRAL_MAX_NODES: usize = 256;
/// Cluster count used when `k == 0` (auto) is requested — spectral has no reliable built-in
/// selection, and two clusters is the canonical non-convex case (moons / two rings).
pub const SPECTRAL_DEFAULT_K: usize = 2;
/// k-means restarts for the embedding and landmark reductions (mirrors the k-means head).
const N_INIT: usize = 4;

/// Result of a spectral run: one cluster label per input microcluster.
pub struct Spectral {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
}

/// Spectral-cluster `features` into `k` groups (`k == 0` ⇒ [`SPECTRAL_DEFAULT_K`]). Above
/// [`SPECTRAL_MAX_NODES`] features the graph is reduced to that many k-means landmarks first.
pub fn spectral<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Spectral {
    assert!(!features.is_empty(), "spectral needs at least one feature");
    let k = if k == 0 { SPECTRAL_DEFAULT_K } else { k };
    let m = features.len();
    if m > SPECTRAL_MAX_NODES {
        // Landmark reduction: weighted k-means to SPECTRAL_MAX_NODES centers, spectral on those,
        // then every leaf inherits the label of the landmark it was assigned to.
        let land = kmeans(features, SPECTRAL_MAX_NODES, max_iter, N_INIT, seed);
        let sub = spectral_core::<R>(&land.centers, k, max_iter, seed);
        let labels = land.labels.iter().map(|&a| sub[a]).collect();
        return Spectral { labels };
    }
    let centers: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    Spectral {
        labels: spectral_core::<R>(&centers, k, max_iter, seed),
    }
}

fn spectral_core<R: Real>(centers: &[Vec<R>], k: usize, max_iter: usize, seed: u64) -> Vec<usize> {
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

    // Symmetric self-tuning k-NN affinity graph, then its degrees and the normalized affinity
    // `P = D^{-1/2} A D^{-1/2}` (`A` is exactly symmetric by construction, so `P` is too).
    let adj = knn_affinity(centers);
    let mut deg = vec![R::zero(); n];
    let mut a = vec![vec![R::zero(); n]; n];
    for (i, row) in adj.iter().enumerate() {
        for &(j, w) in row {
            a[i][j] = w;
            deg[i] = deg[i] + w;
        }
    }
    let dinv: Vec<R> = deg.iter().map(|&d| R::one() / d.max(tiny).sqrt()).collect();
    let mut p = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            p[i][j] = a[i][j] * dinv[i] * dinv[j];
        }
    }

    // Top-`k` eigenvectors of `P` are the bottom-`k` of `L_sym = I − P` (here `2 ≤ k < n`).
    let (eigvals, vecs) = jacobi_eigen(&p);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap()); // P eigenvalues descending

    // Row-normalized eigenvector embedding (Ng-Jordan-Weiss), then k-means on the rows.
    let sel = &order[0..k];
    let embed: Vec<Spherical<R>> = (0..n)
        .map(|i| {
            let mut row: Vec<R> = sel.iter().map(|&c| vecs[i][c]).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros, two_moons};
    use std::collections::HashSet;

    fn n_distinct(labels: &[usize]) -> usize {
        labels.iter().copied().collect::<HashSet<_>>().len()
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

    #[test]
    fn spectral_landmark_reduction_above_the_cap() {
        // > SPECTRAL_MAX_NODES microclusters force the k-means-landmark reduction path.
        let mut rng = SplitMix64::new(11);
        let centers = [[0.0, 0.0], [15.0, 0.0], [0.0, 15.0]];
        let (pts, truth) = blobs(&mut rng, 700, &centers, 0.7);
        let (micros, point_to_micro) = grid_micros(&pts, 0.12);
        assert!(
            micros.len() > SPECTRAL_MAX_NODES,
            "need > cap microclusters"
        );
        let micro_labels = spectral(&micros, 3, 100, 1).labels;
        assert_eq!(micro_labels.len(), micros.len());
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| micro_labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.9, "landmark spectral lost the blobs");
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
