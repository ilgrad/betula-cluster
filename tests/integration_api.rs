//! Integration tests over the public Rust API (`betula_cluster::*`).
//!
//! Unit tests live in-module (they exercise private internals — idiomatic Rust); these drive only
//! the public surface — features, distances, the CF-tree, the clustering heads, and the end-to-end
//! `Model` — exactly as an external crate would, with self-contained data (no test-only helpers).

use betula_cluster::clustering::{gmm_diagonal, kmeans};
use betula_cluster::distance::{CFDistance, CentroidEuclidean};
use betula_cluster::feature::{ClusterFeature, Diagonal, Full, Spherical};
use betula_cluster::model::{Method, Model};
use betula_cluster::tree::CFTree;

/// Four tight, well-separated 2-D blobs (deterministic; 6 points each).
fn blobs() -> (Vec<Vec<f64>>, Vec<usize>) {
    let centers = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
    let offsets = [
        [-0.3, 0.2],
        [0.25, -0.15],
        [0.1, 0.3],
        [-0.2, -0.25],
        [0.0, 0.0],
        [0.15, 0.1],
    ];
    let mut pts = Vec::new();
    let mut truth = Vec::new();
    for (c, ctr) in centers.iter().enumerate() {
        for off in &offsets {
            pts.push(vec![ctr[0] + off[0], ctr[1] + off[1]]);
            truth.push(c);
        }
    }
    (pts, truth)
}

/// Every blob must be pure (its points share one label) and the blobs must use distinct labels.
fn assert_recovered(labels: &[usize], truth: &[usize], k: usize) {
    let mut blob_label: Vec<Option<usize>> = vec![None; k];
    for (&l, &t) in labels.iter().zip(truth) {
        match blob_label[t] {
            None => blob_label[t] = Some(l),
            Some(bl) => assert_eq!(bl, l, "blob {t} is not pure"),
        }
    }
    let distinct: std::collections::HashSet<usize> = blob_label.iter().flatten().copied().collect();
    assert_eq!(distinct.len(), k, "expected {k} distinct cluster labels");
}

#[test]
fn cf_tree_summarises_all_points() {
    let (pts, _) = blobs();
    let mut tree: CFTree<f64, Spherical<f64>, _, _> =
        CFTree::new(2, 16, 16, 0.05, 100, CentroidEuclidean, CentroidEuclidean);
    for p in &pts {
        tree.insert(p);
    }
    assert!(tree.num_leaves() >= 4 && tree.num_leaves() <= pts.len());
    assert!((tree.summary().weight() - pts.len() as f64).abs() < 1e-9);
    assert_eq!(tree.leaf_features().len(), tree.num_leaves());
}

#[test]
fn model_fit_predict_kmeans_and_gmm() {
    let (pts, truth) = blobs();
    for method in [Method::KMeans, Method::Gmm] {
        let mut tree: CFTree<f64, Diagonal<f64>, _, _> =
            CFTree::new(2, 16, 16, 0.05, 100, CentroidEuclidean, CentroidEuclidean);
        for p in &pts {
            tree.insert(p);
        }
        let model = Model::fit(tree, 4, method, 100, 1);
        assert_eq!(model.n_clusters(), 4);
        let labels: Vec<usize> = pts.iter().map(|p| model.predict(p)).collect();
        assert_recovered(&labels, &truth, 4);
    }
}

#[test]
fn clustering_heads_on_leaf_features() {
    // Build leaf features, then cluster them with the public head functions directly.
    let (pts, _) = blobs();
    let mut tree: CFTree<f64, Diagonal<f64>, _, _> =
        CFTree::new(2, 8, 8, 0.0, 100, CentroidEuclidean, CentroidEuclidean);
    for p in &pts {
        tree.insert(p);
    }
    let feats = tree.leaf_features();
    let km = kmeans(feats, 4, 100, 4, 1);
    assert_eq!(km.labels.len(), feats.len());
    let gmm = gmm_diagonal(feats, 4, 100, 1);
    assert_eq!(gmm.labels.len(), feats.len());
}

#[test]
fn feature_models_and_distance_api() {
    // Features: weighted mean / variance / merge are exact across the public models.
    let mut a: Full<f64> = Full::new(2);
    for p in [[0.0, 0.0], [2.0, 4.0]] {
        a.push(&p, 1.0);
    }
    assert!((a.mean()[0] - 1.0).abs() < 1e-12 && (a.mean()[1] - 2.0).abs() < 1e-12);
    let cov = a.covariance();
    assert!(cov[0][1].abs() > 0.0 && (cov[0][1] - cov[1][0]).abs() < 1e-12);

    let mut s: Spherical<f64> = Spherical::new(2);
    s.push(&[0.0, 0.0], 1.0);
    let mut s2: Spherical<f64> = Spherical::new(2);
    s2.push(&[2.0, 0.0], 1.0);
    s.merge(&s2);
    assert!((s.weight() - 2.0).abs() < 1e-12 && (s.mean()[0] - 1.0).abs() < 1e-12);

    // Distance: feature-to-point and feature-to-feature are the squared Euclidean centroid forms.
    assert!((CentroidEuclidean.point(&s, &[4.0, 0.0]) - 9.0).abs() < 1e-12);
    assert!((CentroidEuclidean.between(&s, &s2) - 1.0).abs() < 1e-12);
}
