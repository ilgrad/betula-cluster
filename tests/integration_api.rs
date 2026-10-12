//! Integration tests over the public Rust API (`betula_cluster::*`).
//!
//! Unit tests live in-module (they exercise private internals — idiomatic Rust); these drive only
//! the public surface — features, distances, the CF-tree, the clustering heads, and the end-to-end
//! `Model` — exactly as an external crate would, with self-contained data (no test-only helpers).

use betula_cluster::clustering::{KMEANS_N_INIT, gmm_diagonal, kmeans};
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
        tree.try_insert(p).unwrap();
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
            tree.try_insert(p).unwrap();
        }
        let model = Model::fit(tree, 4, method, 100, 0, 1, 0);
        assert_eq!(model.n_clusters(), 4);
        let labels: Vec<usize> = pts.iter().map(|p| model.try_predict(p).unwrap()).collect();
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
        tree.try_insert(p).unwrap();
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

#[test]
fn a_row_of_the_wrong_width_is_rejected_and_changes_nothing() {
    // The kernels compare `a.len().min(b.len())` coordinates, so a row one column short used to be
    // clustered on its prefix and a row one column long used to drop its tail — both silently. The
    // checked entry reads the length before anything is absorbed, so the summary is untouched.
    let (pts, _) = blobs();
    let mut tree: CFTree<f64, Spherical<f64>, _, _> =
        CFTree::new(2, 16, 16, 0.05, 100, CentroidEuclidean, CentroidEuclidean);
    for p in &pts {
        tree.try_insert(p).unwrap();
    }
    let (weight, leaves) = (tree.summary().weight(), tree.num_leaves());
    for bad in [vec![1.0], vec![1.0, 2.0, 3.0]] {
        let err = tree.try_insert(&bad).unwrap_err();
        assert_eq!(err.expected, 2);
        assert_eq!(err.got, bad.len());
    }
    assert_eq!(tree.summary().weight(), weight);
    assert_eq!(tree.num_leaves(), leaves);

    let model = Model::fit(tree, 4, Method::KMeans, 100, 0, 1, 0);
    assert!(model.try_predict(&[1.0]).is_err());
    assert!(model.try_predict(&pts[0]).is_ok());
}

/// Standard normals from SplitMix64 (Vigna's constants) through Box–Muller, so the sample is pinned
/// without a dependency.
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn sample(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform(); // (0, 1], so the logarithm is finite
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The population k-means cost per point of `N(0, I_2)` has a closed form, so the tree plus the head
/// can be checked end to end with no labels and no reference implementation. Three 120° sectors
/// cost `2 − 27/(8π)` (centres at radius `3^{3/2}/(2^{3/2}√π)`), two half-planes cost `2 − 2/π`.
/// Both partitions are Lloyd fixed points by symmetry; that the three sectors are the *global*
/// optimum is supported numerically, not proven.
///
/// The score is the k-means cost of the centres the head returns, over the raw rows: the rule a
/// fitted `Model` labels a point by. The head's own `inertia` is the cost of its leaf-level
/// partition and carries the summary's granularity on top: up to +0.012 at 100 leaves here.
///
/// The per-point cost has variance `(64π² + 2·3^{9/2} − 729)/(16π²) = 1.160` at k = 3 and
/// `(4π² − 16)/π² = 2.379` at k = 2 (Maxima), so at 2·10⁵ rows one standard error is 0.0024 and
/// 0.0034, against 0.0034 and 0.0049 at 10⁵ -- half the window. The 0.01 window rejects a
/// collinear 3-means fixed point (1.19017: 1 + the 3-level Lloyd–Max MSE of `N(0, 1)`) and an
/// answer collapsed to two clusters (1.36338).
#[test]
fn kmeans_through_the_tree_reaches_the_gaussian_population_optimum() {
    const ROWS: usize = 200_000;
    let pi = std::f64::consts::PI;
    let mut normal = Normal(20_261_012);
    let pts: Vec<[f64; 2]> = (0..ROWS)
        .map(|_| [normal.sample(), normal.sample()])
        .collect();
    for max_leaves in [100, 500] {
        let mut tree: CFTree<f64, Diagonal<f64>, _, _> = CFTree::new(
            2,
            32,
            32,
            0.0,
            max_leaves,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for p in &pts {
            tree.try_insert(p).unwrap();
        }
        for (k, target) in [(3, 2.0 - 27.0 / (8.0 * pi)), (2, 2.0 - 2.0 / pi)] {
            for seed in 0..3 {
                let centers = kmeans(tree.leaf_features(), k, 100, KMEANS_N_INIT, seed).centers;
                let cost = pts
                    .iter()
                    .map(|p| {
                        centers
                            .iter()
                            .map(|c| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2))
                            .fold(f64::INFINITY, f64::min)
                    })
                    .sum::<f64>()
                    / ROWS as f64;
                assert!(
                    (cost - target).abs() <= 0.01,
                    "max_leaves={max_leaves} k={k} seed={seed}: SSE/N {cost:.5}, \
                     population optimum {target:.5}"
                );
            }
        }
    }
}
