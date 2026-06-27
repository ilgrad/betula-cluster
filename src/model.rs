//! End-to-end model: build a CF-tree, cluster its leaves (Phase 3), and label points.

use crate::clustering::{
    gmm_diagonal, gmm_diagonal_auto, gmm_full, gmm_full_auto, kmeans, ward_hac, ward_hac_auto,
    xmeans,
};
use crate::distance::CFDistance;
use crate::feature::ClusterFeature;
use crate::tree::CFTree;
use crate::types::Real;

/// Upper bound on `k` swept by the BIC auto-selection (`n_clusters = 0`).
const AUTO_K_MAX: usize = 20;

/// Global-clustering method applied to the CF-tree leaves.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum Method {
    /// Weighted k-means (k-means++ init, exact Lloyd).
    KMeans,
    /// Diagonal GMM-EM with the expected-log E-step.
    Gmm,
    /// Full-covariance GMM-EM (captures rotated / correlated clusters).
    GmmFull,
    /// Ward agglomerative hierarchical clustering (variance-increase linkage).
    Ward,
}

/// A fitted model: a CF-tree plus a cluster label per leaf entry. A point is labelled by routing
/// it to its nearest leaf entry and reading that entry's cluster.
pub struct Model<R: Real, C: ClusterFeature<R>, D: CFDistance<R, C>, A: CFDistance<R, C>> {
    tree: CFTree<R, C, D, A>,
    entry_labels: Vec<usize>,
    n_clusters: usize,
}

impl<R: Real, C: ClusterFeature<R>, D: CFDistance<R, C>, A: CFDistance<R, C>> Model<R, C, D, A> {
    /// Cluster the leaves of a tree that already contains the data. `k` is clamped to the number of
    /// available leaf micro-clusters; `k == 0` requests automatic BIC selection of the component
    /// count (GMM heads only — k-means falls back to a single cluster). The realised cluster count
    /// is available via [`Model::n_clusters`].
    pub fn fit(
        tree: CFTree<R, C, D, A>,
        k: usize,
        method: Method,
        max_iter: usize,
        seed: u64,
    ) -> Self {
        let entry_labels = cluster_leaves(tree.leaf_features(), k, method, max_iter, seed);
        let n_clusters = distinct_count(&entry_labels);
        Self {
            tree,
            entry_labels,
            n_clusters,
        }
    }

    /// Cluster label of point `x` (via its nearest leaf entry).
    pub fn predict(&self, x: &[R]) -> usize {
        self.entry_labels[self.tree.nearest_entry(x)]
    }

    /// Number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// The underlying CF-tree.
    pub fn tree(&self) -> &CFTree<R, C, D, A> {
        &self.tree
    }
}

/// Label leaf features with a parametric head. `k == 0` requests BIC auto-selection of the
/// component count for the GMM heads; k-means clamps to `[1, n_features]`. Shared by [`Model::fit`]
/// and the streaming Python estimator so both honour the same `k`/auto semantics.
pub(crate) fn cluster_leaves<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    method: Method,
    max_iter: usize,
    seed: u64,
) -> Vec<usize> {
    let nlv = features.len();
    let auto_hi = nlv.min(AUTO_K_MAX);
    match method {
        Method::KMeans if k == 0 => xmeans(features, 1, auto_hi, max_iter, seed).labels,
        Method::KMeans => kmeans(features, k.min(nlv).max(1), max_iter, 4, seed).labels,
        Method::Gmm if k == 0 => gmm_diagonal_auto(features, 1, auto_hi, max_iter, seed).labels,
        Method::Gmm => gmm_diagonal(features, k.min(nlv).max(1), max_iter, seed).labels,
        Method::GmmFull if k == 0 => gmm_full_auto(features, 1, auto_hi, max_iter, seed).labels,
        Method::GmmFull => gmm_full(features, k.min(nlv).max(1), max_iter, seed).labels,
        Method::Ward if k == 0 => ward_hac_auto(features, 2, auto_hi).labels,
        Method::Ward => ward_hac(features, k.min(nlv).max(1)).labels,
    }
}

/// Like [`cluster_leaves`], but for the GMM heads also returns the per-leaf soft responsibility
/// matrix flattened row-major as `(resp, k)` (`n_leaves × k`); `None` for k-means / Ward / (caller's)
/// HDBSCAN, which have no posterior. Used to expose `predict_proba` without recomputing the E-step.
#[cfg(feature = "python")]
#[allow(clippy::type_complexity)]
pub(crate) fn cluster_leaves_proba<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    method: Method,
    max_iter: usize,
    seed: u64,
) -> (Vec<usize>, Option<(Vec<f64>, usize)>) {
    let nlv = features.len();
    let auto_hi = nlv.min(AUTO_K_MAX);
    let flatten = |resp: &[Vec<R>]| -> (Vec<f64>, usize) {
        let kk = resp.first().map_or(0, |r| r.len());
        let flat = resp
            .iter()
            .flat_map(|r| r.iter().map(|v| v.to_f64().unwrap()))
            .collect();
        (flat, kk)
    };
    match method {
        Method::Gmm if k == 0 => {
            let g = gmm_diagonal_auto(features, 1, auto_hi, max_iter, seed);
            let p = flatten(&g.resp);
            (g.labels, Some(p))
        }
        Method::Gmm => {
            let g = gmm_diagonal(features, k.min(nlv).max(1), max_iter, seed);
            let p = flatten(&g.resp);
            (g.labels, Some(p))
        }
        Method::GmmFull if k == 0 => {
            let g = gmm_full_auto(features, 1, auto_hi, max_iter, seed);
            let p = flatten(&g.resp);
            (g.labels, Some(p))
        }
        Method::GmmFull => {
            let g = gmm_full(features, k.min(nlv).max(1), max_iter, seed);
            let p = flatten(&g.resp);
            (g.labels, Some(p))
        }
        _ => (cluster_leaves(features, k, method, max_iter, seed), None),
    }
}

/// Number of distinct labels actually used (empty components are not counted).
fn distinct_count(labels: &[usize]) -> usize {
    let mut v = labels.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs};
    use crate::distance::CentroidEuclidean;
    use crate::feature::{Diagonal, Spherical};

    #[test]
    fn end_to_end_kmeans_from_points() {
        let mut rng = SplitMix64::new(99);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.6);
        let mut tree: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(2, 16, 16, 0.05, 200, CentroidEuclidean, CentroidEuclidean);
        for p in &pts {
            tree.insert(p);
        }
        let model = Model::fit(tree, 4, Method::KMeans, 100, 7);
        let labels: Vec<usize> = pts.iter().map(|p| model.predict(p)).collect();
        let score = ari(&labels, &truth);
        assert!(score > 0.95, "ARI = {score}");
    }

    #[test]
    fn end_to_end_gmm_from_points() {
        let mut rng = SplitMix64::new(5);
        let centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.7);
        let mut tree: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(2, 16, 16, 0.05, 200, CentroidEuclidean, CentroidEuclidean);
        for p in &pts {
            tree.insert(p);
        }
        let model = Model::fit(tree, 3, Method::Gmm, 200, 3);
        let labels: Vec<usize> = pts.iter().map(|p| model.predict(p)).collect();
        let score = ari(&labels, &truth);
        assert!(score > 0.95, "ARI = {score}");
    }

    #[test]
    fn model_exposes_n_clusters_and_tree() {
        let mut rng = SplitMix64::new(1);
        let centers = [[0.0, 0.0], [9.0, 0.0]];
        let (pts, _truth) = blobs(&mut rng, 200, &centers, 0.5);
        let mut tree: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(2, 16, 16, 0.05, 200, CentroidEuclidean, CentroidEuclidean);
        for p in &pts {
            tree.insert(p);
        }
        let model = Model::fit(tree, 2, Method::KMeans, 100, 1);
        assert_eq!(model.n_clusters(), 2);
        assert!(model.tree().num_leaves() > 0);
    }

    #[test]
    fn cluster_leaves_dispatches_every_method_and_auto_k() {
        let mut rng = SplitMix64::new(2);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]];
        let (pts, _t) = blobs(&mut rng, 300, &centers, 0.5);
        let mut tree: CFTree<f64, Diagonal<f64>, _, _> =
            CFTree::new(2, 16, 16, 0.05, 200, CentroidEuclidean, CentroidEuclidean);
        for p in &pts {
            tree.insert(p);
        }
        let feats = tree.leaf_features().to_vec();
        // every head, both fixed-k and auto-k (k == 0), hits its `cluster_leaves` arm.
        for method in [Method::KMeans, Method::Gmm, Method::GmmFull, Method::Ward] {
            for k in [3usize, 0usize] {
                let labels = cluster_leaves(&feats, k, method, 100, 1);
                assert_eq!(labels.len(), feats.len());
            }
        }
    }
}
