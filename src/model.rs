//! End-to-end model: build a CF-tree, cluster its leaves (Phase 3), and label points.

use crate::clustering::{
    gmm_diagonal, gmm_diagonal_auto, gmm_full, gmm_full_auto, gmm_toeplitz, gmm_toeplitz_auto,
    gmm_toeplitz_full, gmm_toeplitz_full_auto, gmm_toeplitz_gs, gmm_toeplitz_gs_auto, kmeans,
    leiden, movmf, movmf_auto, spectral, spherical_kmeans, ward_hac, ward_hac_auto, xmeans, Gmm,
    GmmFull, GmmToeplitz, Movmf, Objective,
};
use crate::distance::CFDistance;
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::mixture::Mixture;
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
    /// Spectral clustering (self-tuning affinity + normalized Laplacian) for non-convex clusters.
    Spectral,
    /// Leiden community detection on the microcluster affinity graph (auto community count).
    /// `resolution` is γ; `cpm` selects the Constant Potts Model over modularity. `cov_weight > 0`
    /// adds a log-Euclidean covariance/shape term and `tangent_weight > 0` a Grassmann
    /// tangent-subspace term (rank `tangent_rank`) to the affinity — GeoBETULA, best with
    /// `feature="full"`.
    Leiden {
        resolution: f64,
        cpm: bool,
        cov_weight: f64,
        tangent_weight: f64,
        tangent_rank: usize,
    },
    /// Spherical k-means on the unit sphere (hard cosine assignment) for L2-normalized embeddings.
    SphericalKMeans,
    /// Mixture of von Mises–Fisher distributions (soft directional EM; BIC auto-`k` when `k == 0`).
    Movmf,
    /// AR / Toeplitz-structured GMM for ordered, wide-sense-stationary signals (time-series windows,
    /// trajectories). BIC auto-`k` when `k == 0`; use `feature="spherical"` or `"diagonal"`.
    GmmToeplitz,
    /// General (non-AR) positive-definite Toeplitz-covariance GMM for ordered stationary signals whose
    /// autocovariance a low-order AR cannot capture. BIC auto-`k` when `k == 0`.
    GmmToeplitzFull,
    /// Full-order Gohberg-Semencul MLE Toeplitz-precision GMM (Yule-Walker warm start + likelihood
    /// coordinate ascent) for ordered stationary signals. BIC auto-`k` when `k == 0`.
    GmmToeplitzGs,
}

/// How a head labels a raw point — by its own objective, not by a routing shortcut.
///
/// Only k-means and its spherical twin are *centroid* models: "assign to the nearest centre" is
/// literally what they optimise. The mixture heads assign by maximum posterior, which weighs each
/// component by its own covariance / concentration and its mixing weight — a nearest-centre rule is
/// a different partition, not a faster route to the same one. Ward, Spectral and Leiden have neither:
/// their clusters need not be convex, so any centre rule would impose exactly the Voronoi partition
/// they exist to avoid, and the microcluster route is the only thing defined for them.
pub(crate) enum Rule {
    /// Argmin over the cluster centres. `unit` compares them as unit vectors, where the Euclidean
    /// argmin and the cosine argmax agree.
    Centroid { unit: bool },
    /// Argmax of `ln π_c + ln p(x | θ_c)` under the fitted mixture.
    Posterior,
    /// Route down the tree to a leaf and read its label.
    Microcluster,
}

pub(crate) fn assignment_rule(method: Method) -> Rule {
    match method {
        Method::KMeans => Rule::Centroid { unit: false },
        Method::SphericalKMeans => Rule::Centroid { unit: true },
        Method::Gmm
        | Method::GmmFull
        | Method::GmmToeplitz
        | Method::GmmToeplitzFull
        | Method::GmmToeplitzGs
        | Method::Movmf => Rule::Posterior,
        Method::Ward | Method::Spectral | Method::Leiden { .. } => Rule::Microcluster,
    }
}

/// Mass-weighted centroid of each non-empty cluster, paired with its label — the same quantity the
/// Python `cluster_centers_` accessor reports. Empty clusters are dropped rather than emitted as
/// zero rows, which would sit at the origin and attract every point near it.
fn cluster_centroids<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    labels: &[usize],
    unit: bool,
) -> Vec<(usize, Vec<R>)> {
    let dim = features.first().map_or(0, |f| f.dim());
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    let mut sums = vec![vec![R::zero(); dim]; k];
    let mut wsum = vec![R::zero(); k];
    for (f, &l) in features.iter().zip(labels) {
        let w = f.weight();
        wsum[l] = wsum[l] + w;
        for (s, &m) in sums[l].iter_mut().zip(f.mean()) {
            *s = *s + w * m;
        }
    }
    let mut out = Vec::new();
    for (l, &ws) in wsum.iter().enumerate() {
        if ws <= R::zero() {
            continue;
        }
        let mut c: Vec<R> = sums[l].iter().map(|&s| s / ws).collect();
        if unit {
            let norm = c.iter().fold(R::zero(), |a, &v| a + v * v).sqrt();
            if norm > R::zero() {
                for v in &mut c {
                    *v = *v / norm;
                }
            }
        }
        out.push((l, c));
    }
    out
}

/// What a fitted head does with a raw point. The three cases are mutually exclusive by
/// construction, so no combination of them can be represented.
enum Assignment<R: Real> {
    /// Nearest of the `(label, centre)` pairs — the partition a centroid head *is*.
    Centers(Vec<(usize, Vec<R>)>),
    /// Maximum posterior under the fitted mixture.
    Posterior(Mixture),
    /// Nearest leaf entry, then that entry's label.
    Microcluster,
}

/// A fitted model: a CF-tree plus a cluster label per leaf entry.
///
/// A point is labelled by the head's own model of a point (see [`Rule`]). The alternative, routing
/// the point down the tree to a leaf and reading that leaf's label, answers a different question —
/// *which cluster owns the nearest microcluster* — and answers it approximately, since the descent is
/// greedy. Measured on 20-newsgroups TF-IDF it relabels 18% of points for `kmeans` and 43% for `gmm`,
/// and it could reach only 14 of the 20 clusters the `kmeans` head found: the rest were unreachable
/// by descent at any point. Heads with no point model keep the microcluster route, which is all they
/// define.
pub struct Model<R: Real, C: ClusterFeature<R>, D: CFDistance<R, C>, A: CFDistance<R, C>> {
    tree: CFTree<R, C, D, A>,
    entry_labels: Vec<usize>,
    assign: Assignment<R>,
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
        let fit = fit_head(tree.leaf_features(), k, method, max_iter, seed);
        let n_clusters = distinct_count(&fit.labels);
        let assign = match (assignment_rule(method), fit.mixture) {
            (Rule::Centroid { unit }, _) => {
                Assignment::Centers(cluster_centroids(tree.leaf_features(), &fit.labels, unit))
            }
            (Rule::Posterior, Some(m)) => Assignment::Posterior(m),
            _ => Assignment::Microcluster,
        };
        Self {
            tree,
            entry_labels: fit.labels,
            assign,
            n_clusters,
        }
    }

    /// Cluster label of point `x` under the head's own assignment rule.
    pub fn predict(&self, x: &[R]) -> usize {
        match &self.assign {
            Assignment::Centers(centers) => nearest_center(centers, x),
            Assignment::Posterior(mixture) => mixture.assign(x),
            Assignment::Microcluster => self.entry_labels[self.tree.nearest_entry(x)],
        }
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

/// Label of the nearest `(label, centre)` pair. `centers` is never empty on this path: a fitted
/// partition has at least one non-empty cluster.
fn nearest_center<R: Real>(centers: &[(usize, Vec<R>)], x: &[R]) -> usize {
    let mut best = 0;
    let mut bd = R::infinity();
    for (i, (_, c)) in centers.iter().enumerate() {
        let d = sq_euclidean(x, c);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    centers[best].0
}

/// The three things a mixture head returns that outlive the fit.
trait MixtureFit<R: Real> {
    fn parts(self) -> (Vec<usize>, Vec<Vec<R>>, Mixture);
}

macro_rules! mixture_fit {
    ($t:ident) => {
        impl<R: Real> MixtureFit<R> for $t<R> {
            fn parts(self) -> (Vec<usize>, Vec<Vec<R>>, Mixture) {
                (self.labels, self.resp, self.mixture)
            }
        }
    };
}

mixture_fit!(Gmm);
mixture_fit!(GmmFull);
mixture_fit!(GmmToeplitz);
mixture_fit!(Movmf);

/// What one Phase-3 head fit produced.
pub(crate) struct HeadFit<R: Real> {
    /// One cluster label per leaf feature.
    pub labels: Vec<usize>,
    /// Per-leaf soft responsibilities `[leaf][component]`, for the heads that have a posterior.
    /// Read by the Python estimator's `microcluster_proba_`; the core [`Model`] has no use for it.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub resp: Option<Vec<Vec<R>>>,
    /// The point-level density, for the heads that are generative.
    pub mixture: Option<Mixture>,
}

impl<R: Real> HeadFit<R> {
    /// A head with no point-level model.
    fn hard(labels: Vec<usize>) -> Self {
        Self {
            labels,
            resp: None,
            mixture: None,
        }
    }

    /// A mixture head. Components that no leaf claims are silenced here — the single place that can
    /// forget to, so it does not.
    fn soft(fit: impl MixtureFit<R>) -> Self {
        let (labels, resp, mut mixture) = fit.parts();
        mixture.restrict_to(&labels);
        Self {
            labels,
            resp: Some(resp),
            mixture: Some(mixture),
        }
    }
}

/// Label leaf features with a parametric head. `k == 0` requests BIC auto-selection of the
/// component count for the GMM heads; k-means clamps to `[1, n_features]`. Shared by [`Model::fit`]
/// and the streaming Python estimator so both honour the same `k`/auto semantics — and so both get
/// the same point-level model out of it.
pub(crate) fn fit_head<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    method: Method,
    max_iter: usize,
    seed: u64,
) -> HeadFit<R> {
    let nlv = features.len();
    let auto_hi = nlv.min(AUTO_K_MAX);
    let kk = k.min(nlv).max(1);
    match method {
        Method::KMeans if k == 0 => {
            HeadFit::hard(xmeans(features, 1, auto_hi, max_iter, seed).labels)
        }
        Method::KMeans => HeadFit::hard(kmeans(features, kk, max_iter, 4, seed).labels),
        Method::Gmm if k == 0 => {
            HeadFit::soft(gmm_diagonal_auto(features, 1, auto_hi, max_iter, seed))
        }
        Method::Gmm => HeadFit::soft(gmm_diagonal(features, kk, max_iter, seed)),
        Method::GmmFull if k == 0 => {
            HeadFit::soft(gmm_full_auto(features, 1, auto_hi, max_iter, seed))
        }
        Method::GmmFull => HeadFit::soft(gmm_full(features, kk, max_iter, seed)),
        Method::Ward if k == 0 => HeadFit::hard(ward_hac_auto(features, 2, auto_hi).labels),
        Method::Ward => HeadFit::hard(ward_hac(features, kk).labels),
        // Spectral resolves `k == 0` (eigengap) and clamps internally, so one arm covers both.
        Method::Spectral => HeadFit::hard(spectral(features, k, max_iter, seed).labels),
        // Leiden discovers the community count from the graph; `k` is ignored (like HDBSCAN).
        Method::Leiden {
            resolution,
            cpm,
            cov_weight,
            tangent_weight,
            tangent_rank,
        } => {
            let obj = if cpm {
                Objective::Cpm
            } else {
                Objective::Modularity
            };
            HeadFit::hard(
                leiden(
                    features,
                    resolution,
                    obj,
                    seed,
                    cov_weight,
                    tangent_weight,
                    tangent_rank,
                )
                .labels,
            )
        }
        // Spherical k-means needs a `k`; `k == 0` selects it by BIC via the vMF mixture.
        Method::SphericalKMeans if k == 0 => {
            let auto = movmf_auto(features, 1, auto_hi, max_iter, seed).means.len();
            HeadFit::hard(
                spherical_kmeans(features, auto.min(nlv).max(1), max_iter, 4, seed).labels,
            )
        }
        Method::SphericalKMeans => {
            HeadFit::hard(spherical_kmeans(features, kk, max_iter, 4, seed).labels)
        }
        Method::Movmf if k == 0 => HeadFit::soft(movmf_auto(features, 1, auto_hi, max_iter, seed)),
        Method::Movmf => HeadFit::soft(movmf(features, kk, max_iter, seed)),
        Method::GmmToeplitz if k == 0 => {
            HeadFit::soft(gmm_toeplitz_auto(features, 1, auto_hi, max_iter, seed))
        }
        Method::GmmToeplitz => HeadFit::soft(gmm_toeplitz(features, kk, max_iter, seed)),
        Method::GmmToeplitzFull if k == 0 => {
            HeadFit::soft(gmm_toeplitz_full_auto(features, 1, auto_hi, max_iter, seed))
        }
        Method::GmmToeplitzFull => HeadFit::soft(gmm_toeplitz_full(features, kk, max_iter, seed)),
        Method::GmmToeplitzGs if k == 0 => {
            HeadFit::soft(gmm_toeplitz_gs_auto(features, 1, auto_hi, max_iter, seed))
        }
        Method::GmmToeplitzGs => HeadFit::soft(gmm_toeplitz_gs(features, kk, max_iter, seed)),
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
    use std::collections::HashMap;

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
        for method in [
            Method::KMeans,
            Method::Gmm,
            Method::GmmFull,
            Method::GmmToeplitz,
            Method::GmmToeplitzFull,
            Method::GmmToeplitzGs,
            Method::Ward,
            Method::Spectral,
            Method::Leiden {
                resolution: 1.0,
                cpm: false,
                cov_weight: 0.0,
                tangent_weight: 0.0,
                tangent_rank: 2,
            },
            Method::Leiden {
                resolution: 0.05,
                cpm: true,
                cov_weight: 0.0,
                tangent_weight: 0.0,
                tangent_rank: 2,
            },
            Method::Leiden {
                resolution: 1.0,
                cpm: false,
                cov_weight: 0.5,
                tangent_weight: 0.5,
                tangent_rank: 2,
            },
            Method::SphericalKMeans,
            Method::Movmf,
        ] {
            for k in [3usize, 0usize] {
                let labels = fit_head(&feats, k, method, 100, 1).labels;
                assert_eq!(labels.len(), feats.len());
            }
        }
    }

    /// Three well-separated blobs collapsed onto a coarse grid: few enough leaves that the
    /// automatic sweep is cheap, separated enough — in direction as well as in position — that
    /// every head's automatic arm finds more than one component.
    fn dispatch_leaves() -> Vec<Diagonal<f64>> {
        let mut rng = SplitMix64::new(11);
        let centers = [[10.0, 0.0], [0.0, 10.0], [-7.0, -7.0]];
        let (pts, _truth) = blobs(&mut rng, 40, &centers, 0.6);
        let mut map: HashMap<(i64, i64), usize> = HashMap::new();
        let mut cfs: Vec<Diagonal<f64>> = Vec::new();
        for p in &pts {
            let key = ((p[0] / 1.5).round() as i64, (p[1] / 1.5).round() as i64);
            let idx = *map.entry(key).or_insert_with(|| {
                cfs.push(<Diagonal<f64> as ClusterFeature<f64>>::new(2));
                cfs.len() - 1
            });
            cfs[idx].push(p, 1.0);
        }
        cfs
    }

    #[test]
    fn every_head_takes_its_automatic_arm_only_when_k_is_zero() {
        let feats = dispatch_leaves();
        for (name, method) in [
            ("kmeans", Method::KMeans),
            ("gmm", Method::Gmm),
            ("gmm-full", Method::GmmFull),
            ("ward", Method::Ward),
            ("spherical-kmeans", Method::SphericalKMeans),
            ("movmf", Method::Movmf),
            ("gmm-toeplitz", Method::GmmToeplitz),
            ("gmm-toeplitz-full", Method::GmmToeplitzFull),
            ("gmm-toeplitz-gs", Method::GmmToeplitzGs),
        ] {
            let auto = distinct_count(&fit_head(&feats, 0, method, 100, 1).labels);
            let one = distinct_count(&fit_head(&feats, 1, method, 100, 1).labels);
            let two = distinct_count(&fit_head(&feats, 2, method, 100, 1).labels);
            assert!(auto > 1, "{name}: the automatic arm collapsed to {auto}");
            assert_eq!(one, 1, "{name}: k = 1 did not take the fixed arm");
            assert_eq!(two, 2, "{name}: k = 2 did not take the fixed arm");
        }
    }

    #[test]
    fn cluster_centroids_are_mass_weighted_and_normalized_only_when_asked() {
        let leaf = |p: [f64; 2], w: f64| {
            let mut f = <Diagonal<f64> as ClusterFeature<f64>>::new(2);
            f.push(&p, w);
            f
        };
        // Label 1 is empty, and label 2's two leaves cancel to the origin — the one centroid whose
        // norm is zero, so `unit` must leave it alone rather than divide by it.
        let feats = vec![
            leaf([0.0, 4.0], 1.0),
            leaf([4.0, 0.0], 3.0),
            leaf([1.0, 0.0], 2.0),
            leaf([-1.0, 0.0], 2.0),
        ];
        let labels = [0usize, 0, 2, 2];

        let raw = cluster_centroids(&feats, &labels, false);
        assert_eq!(raw.len(), 2, "the empty label was emitted");
        assert_eq!(raw[0].0, 0);
        assert_eq!(raw[1].0, 2);
        for (got, want) in raw[0].1.iter().zip(&[3.0, 1.0]) {
            assert!((got - want).abs() < 1e-12, "{raw:?}");
        }
        assert!(raw[1].1.iter().all(|v| v.abs() < 1e-12), "{raw:?}");

        let unit = cluster_centroids(&feats, &labels, true);
        let n = 10f64.sqrt();
        for (got, want) in unit[0].1.iter().zip(&[3.0 / n, 1.0 / n]) {
            assert!((got - want).abs() < 1e-12, "{unit:?}");
        }
        assert!(unit[1].1.iter().all(|v| *v == 0.0), "{unit:?}");
    }

    #[test]
    fn nearest_center_keeps_the_first_of_two_equidistant_centers() {
        let centers = vec![(3usize, vec![-1.0, 0.0]), (7usize, vec![1.0, 0.0])];
        assert_eq!(nearest_center(&centers, &[0.0, 0.0]), 3);
        assert_eq!(nearest_center(&centers, &[0.9, 0.0]), 7);
        assert_eq!(nearest_center(&centers, &[-0.9, 0.0]), 3);
    }

    #[test]
    fn each_head_installs_the_assignment_rule_it_declares() {
        let mut rng = SplitMix64::new(4);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]];
        let (pts, _truth) = blobs(&mut rng, 60, &centers, 0.5);
        for (name, method) in [
            ("kmeans", Method::KMeans),
            ("spherical-kmeans", Method::SphericalKMeans),
            ("gmm", Method::Gmm),
            ("movmf", Method::Movmf),
            ("ward", Method::Ward),
            ("spectral", Method::Spectral),
            (
                "leiden",
                Method::Leiden {
                    resolution: 1.0,
                    cpm: false,
                    cov_weight: 0.0,
                    tangent_weight: 0.0,
                    tangent_rank: 2,
                },
            ),
        ] {
            let mut tree: CFTree<f64, Diagonal<f64>, _, _> =
                CFTree::new(2, 16, 16, 0.05, 200, CentroidEuclidean, CentroidEuclidean);
            for p in &pts {
                tree.insert(p);
            }
            let model = Model::fit(tree, 3, method, 100, 1);
            let ok = matches!(
                (assignment_rule(method), &model.assign),
                (Rule::Centroid { .. }, Assignment::Centers(_))
                    | (Rule::Posterior, Assignment::Posterior(_))
                    | (Rule::Microcluster, Assignment::Microcluster)
            );
            assert!(ok, "{name}: fit installed a rule the head does not declare");
        }
    }

    #[test]
    fn the_centroid_rule_outvotes_the_nearest_microcluster() {
        // One cluster is a heavy core at the origin plus a light satellite at x = 5; the other is
        // compact at x = 12. The probe at x = 7 has the satellite as its nearest microcluster, but
        // the satellite barely moves its cluster's centroid, so the far compact centre is nearer.
        let mut rng = SplitMix64::new(17);
        let (core, _t) = blobs(&mut rng, 100, &[[0.0, 0.0]], 0.4);
        let (satellite, _t) = blobs(&mut rng, 20, &[[5.0, 0.0]], 0.2);
        let (far, _t) = blobs(&mut rng, 40, &[[12.0, 0.0]], 0.2);
        let mut tree: CFTree<f64, Diagonal<f64>, _, _> =
            CFTree::new(2, 16, 16, 0.05, 200, CentroidEuclidean, CentroidEuclidean);
        for p in core.iter().chain(&satellite).chain(&far) {
            tree.insert(p);
        }
        let model = Model::fit(tree, 2, Method::KMeans, 100, 1);
        let probe = [7.0, 0.0];
        let micro = model.entry_labels[model.tree.nearest_entry(&probe)];
        assert_eq!(
            model.predict(&probe),
            model.predict(&[12.0, 0.0]),
            "the probe did not fall to the far centre"
        );
        assert_ne!(
            model.predict(&probe),
            micro,
            "the centroid rule agreed with the descent, so the fixture proves nothing"
        );
    }
}
