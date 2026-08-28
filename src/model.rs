//! End-to-end model: build a CF-tree, cluster its leaves (Phase 3), and label points.

use crate::clustering::{
    Gmm, GmmFull, GmmToeplitz, Linkage, Movmf, Mppca, Objective, agglomerative, agglomerative_auto,
    gmm_diagonal, gmm_diagonal_auto, gmm_full, gmm_full_auto, gmm_toeplitz, gmm_toeplitz_auto,
    gmm_toeplitz_full, gmm_toeplitz_full_auto, gmm_toeplitz_gs, gmm_toeplitz_gs_auto, kmeans,
    kmeans_auto, leiden, movmf, movmf_auto, mppca, mppca_auto, spectral, spherical_kmeans,
    ward_hac, ward_hac_auto, xmeans,
};
use crate::distance::CFDistance;
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::mixture::Mixture;
use crate::tree::CFTree;
use crate::types::Real;

/// Default ceiling on `k` for the automatic selectors that **sweep** — those that refit the whole
/// head at every candidate `k` and keep the best score. Their work is `Σ_{k≤K} k = O(K²)`, so the
/// ceiling is the only thing bounding them: measured on 480 leaves in 64 dimensions, raising it from
/// 20 to 120 takes `kmeans` from 46 ms to 1.4 s and `gmm` from 0.5 s to 4.1 s. It binds hard on data
/// with more groups than that — the same fixture holds 120 — so it is an override, not a law; see
/// `fit_head`'s `auto_k_max`. The selectors that only *cut* an already-built dendrogram do not pay
/// this and are not bounded by it.
const AUTO_K_MAX: usize = 20;

/// Global-clustering method applied to the CF-tree leaves.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum Method {
    /// Weighted k-means (k-means++ init, exact Lloyd). `k == 0` sweeps `k` and keeps the best
    /// Pelleg-Moore BIC, which is bounded by `AUTO_K_MAX` because the sweep costs `O(k_max²)`.
    KMeans,
    /// X-means (Pelleg & Moore 2000): recursive splitting, where each centre is tested separately
    /// for a 2-way split by BIC and the algorithm stops when no centre wants one. `k` is an **upper
    /// bound**, not a target; `k == 0` bounds it only by the leaf count, since the split test — not a
    /// cost guard — is what stops the recursion.
    XMeans,
    /// Diagonal GMM-EM with the expected-log E-step.
    Gmm,
    /// Full-covariance GMM-EM (captures rotated / correlated clusters).
    GmmFull,
    /// Ward agglomerative hierarchical clustering (variance-increase linkage).
    Ward,
    /// Agglomerative hierarchical clustering under one of the four non-Ward linkages, driven by
    /// Anderberg's algorithm. Auto-`k` (Calinski-Harabasz over the cuts) when `k == 0`.
    Agglomerative { linkage: Linkage },
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
    /// Mixture of probabilistic PCA: `Σ_c = W_c W_cᵀ + σ_c² I` with `W_c` of rank `rank`. Captures
    /// orientation like `GmmFull` at `O(d·rank)` per component instead of `O(d²)`; pair it with
    /// `feature="fd"`, whose low-rank leaf scatter the E-step consumes without forming `d×d`. BIC
    /// auto-`k` when `k == 0`.
    Mppca { rank: usize },
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
        Method::KMeans | Method::XMeans => Rule::Centroid { unit: false },
        Method::SphericalKMeans => Rule::Centroid { unit: true },
        Method::Gmm
        | Method::GmmFull
        | Method::GmmToeplitz
        | Method::GmmToeplitzFull
        | Method::GmmToeplitzGs
        | Method::Movmf
        | Method::Mppca { .. } => Rule::Posterior,
        Method::Ward | Method::Agglomerative { .. } | Method::Spectral | Method::Leiden { .. } => {
            Rule::Microcluster
        }
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
    /// Nearest of the `(label, centre)` pairs — the partition a centroid head *is*. `unit` is the
    /// head's own flag, kept so a later refinement re-normalizes the centres the same way the fit did.
    Centers {
        centers: Vec<(usize, Vec<R>)>,
        unit: bool,
    },
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
    /// count (GMM heads only — k-means falls back to a single cluster). `auto_k_max` overrides the
    /// ceiling that selection searches under, and `0` takes the default. The realised cluster count
    /// is available via [`Model::n_clusters`].
    pub fn fit(
        tree: CFTree<R, C, D, A>,
        k: usize,
        method: Method,
        max_iter: usize,
        seed: u64,
        auto_k_max: usize,
    ) -> Self {
        let fit = fit_head(tree.leaf_features(), k, method, max_iter, seed, auto_k_max);
        let n_clusters = distinct_count(&fit.labels);
        let assign = match (assignment_rule(method), fit.mixture) {
            (Rule::Centroid { unit }, _) => Assignment::Centers {
                centers: cluster_centroids(tree.leaf_features(), &fit.labels, unit),
                unit,
            },
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
            Assignment::Centers { centers, .. } => nearest_center(centers, x),
            Assignment::Posterior(mixture) => mixture.assign(x),
            Assignment::Microcluster => self.entry_labels[self.tree.nearest_entry(x)],
        }
    }

    /// BIRCH Phase 4 over the raw rows `flat` (`n × dim`), warm-started from the Phase-3 centres.
    /// Returns the number of Lloyd sweeps run, `0` for a head that has no centre model — for
    /// [`Rule::Posterior`] and [`Rule::Microcluster`] "nearest centre" is not the partition the head
    /// defines, so a centre sweep would silently replace it with a different one.
    pub fn refine(&mut self, flat: &[R], n: usize, dim: usize, iters: usize) -> usize {
        let Assignment::Centers { centers, unit } = &mut self.assign else {
            return 0;
        };
        if centers.first().is_none_or(|(_, c)| c.len() != dim) {
            return 0;
        }
        let mut buf: Vec<R> = centers
            .iter()
            .flat_map(|(_, c)| c.iter().copied())
            .collect();
        let sweeps = refine_centers(&mut buf, flat, n, dim, *unit, iters);
        for (i, (_, c)) in centers.iter_mut().enumerate() {
            c.copy_from_slice(&buf[i * dim..(i + 1) * dim]);
        }
        sweeps
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

/// BIRCH Phase 4, never previously implemented here: Lloyd iterations over the **raw** points,
/// warm-started from the Phase-3 centres. Returns the number of sweeps actually run, which is fewer
/// than `iters` when the assignment stops moving.
///
/// The warm start is the point of it. scikit-learn pays `n_init` cold restarts to escape a bad
/// seeding; a CF-tree summary is already a good seeding, so 1–3 sweeps reach a comparable objective
/// at a fraction of the cost. It is `O(iters·n·k·d)` in time and `O(k·d)` in extra memory, so it
/// touches the raw data again — which is why it is opt-in and why streaming cannot have it.
///
/// **A better objective is not a better partition.** On `covtype` scikit-learn's k-means reaches a
/// lower within-cluster sum of squares than ours and a *worse* ARI (0.054 against 0.088), so
/// refining toward that objective can move ARI either way. The caller decides.
///
/// A cluster that attracts no point keeps its previous centre rather than being re-seeded: dropping
/// it would change `k` mid-refinement and re-seeding it is a second algorithm with its own seeding
/// policy. `unit` re-normalizes each centre after the update, for the spherical head where the
/// Euclidean argmin and the cosine argmax agree only on the unit sphere.
pub(crate) fn refine_centers<R: Real>(
    centers: &mut [R],
    flat: &[R],
    n: usize,
    dim: usize,
    unit: bool,
    iters: usize,
) -> usize {
    let k = centers.len().checked_div(dim).unwrap_or(0);
    if k == 0 || n == 0 || iters == 0 {
        return 0;
    }
    let mut sums = vec![R::zero(); k * dim];
    let mut counts = vec![0usize; k];
    let mut owner = vec![usize::MAX; n];
    for sweep in 0..iters {
        sums.iter_mut().for_each(|v| *v = R::zero());
        counts.iter_mut().for_each(|c| *c = 0);
        let mut moved = false;
        for i in 0..n {
            let x = &flat[i * dim..(i + 1) * dim];
            let mut best = 0;
            let mut bd = R::infinity();
            for c in 0..k {
                let d = sq_euclidean(x, &centers[c * dim..(c + 1) * dim]);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            moved |= owner[i] != best;
            owner[i] = best;
            counts[best] += 1;
            for (j, &v) in x.iter().enumerate() {
                sums[best * dim + j] = sums[best * dim + j] + v;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let m = R::from_usize(counts[c]).unwrap();
            let centre = &mut centers[c * dim..(c + 1) * dim];
            for (j, v) in centre.iter_mut().enumerate() {
                *v = sums[c * dim + j] / m;
            }
            if unit {
                let norm = centre.iter().fold(R::zero(), |a, &v| a + v * v).sqrt();
                if norm > R::zero() {
                    centre.iter_mut().for_each(|v| *v = *v / norm);
                }
            }
        }
        if !moved {
            return sweep + 1;
        }
    }
    iters
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
mixture_fit!(Mppca);

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

/// The ceiling an automatic arm (`k == 0`) searches under, and the one thing that decides it: how
/// the selector pays for a wider search.
///
/// A **sweep** refits the whole head at every candidate `k` and keeps the best score, so its work is
/// `Σ_{k≤K} k = O(K²)` and the ceiling is the only thing bounding it — measured on 480 leaves in 64
/// dimensions, raising it from 20 to 120 takes `kmeans` from 46 ms to 1.4 s and `gmm` from 0.5 s to
/// 4.1 s. A **cut** selector builds one dendrogram and scores its cuts, or stops on its own test, so
/// a wider ceiling costs it a linear pass and nothing more: `ward` over the same leaves spends
/// 5.7 ms at 20 and 23.8 ms at the leaf count. The first keeps [`AUTO_K_MAX`]; the second has no
/// reason to and takes the leaf count, which on that fixture is the difference between ARI 0.009 and
/// 1.000 because the true count is 120.
///
/// `auto_k_max` overrides both; `0` takes the default. The result is always at least 1 and never
/// exceeds the leaf count, so a caller cannot ask a head for more clusters than it has leaves.
pub(crate) fn auto_k_ceiling(method: Method, n_leaves: usize, auto_k_max: usize) -> usize {
    let sweeps = !matches!(
        method,
        Method::Ward | Method::Agglomerative { .. } | Method::XMeans
    );
    let default = if sweeps {
        n_leaves.min(AUTO_K_MAX)
    } else {
        n_leaves
    };
    if auto_k_max == 0 {
        default
    } else {
        n_leaves.min(auto_k_max)
    }
    .max(1)
}

/// Label leaf features with a parametric head. `k == 0` requests BIC auto-selection of the
/// component count for the GMM heads; k-means clamps to `[1, n_features]`. Shared by [`Model::fit`]
/// and the streaming Python estimator so both honour the same `k`/auto semantics — and so both get
/// the same point-level model out of it.
///
/// `auto_k_max` overrides the ceiling the automatic arms search under; `0` takes the default.
pub(crate) fn fit_head<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    method: Method,
    max_iter: usize,
    seed: u64,
    auto_k_max: usize,
) -> HeadFit<R> {
    let nlv = features.len();
    let hi = auto_k_ceiling(method, nlv, auto_k_max);
    let kk = k.min(nlv).max(1);
    match method {
        Method::KMeans if k == 0 => {
            HeadFit::hard(kmeans_auto(features, 1, hi, max_iter, seed).labels)
        }
        Method::KMeans => HeadFit::hard(kmeans(features, kk, max_iter, 4, seed).labels),
        // `AUTO_K_MAX` bounds the *sweep*, whose cost is quadratic in the cap. X-means stops when no
        // centre wants to split, so at `k == 0` the only bound it needs is the leaf count -- taking
        // the sweep's cost guard here would silently truncate the answer this head exists to give.
        Method::XMeans if k == 0 => HeadFit::hard(xmeans(features, 2, hi, max_iter, seed).labels),
        Method::XMeans => HeadFit::hard(xmeans(features, 2.min(kk), kk, max_iter, seed).labels),
        Method::Gmm if k == 0 => HeadFit::soft(gmm_diagonal_auto(features, 1, hi, max_iter, seed)),
        Method::Gmm => HeadFit::soft(gmm_diagonal(features, kk, max_iter, seed)),
        Method::GmmFull if k == 0 => HeadFit::soft(gmm_full_auto(features, 1, hi, max_iter, seed)),
        Method::GmmFull => HeadFit::soft(gmm_full(features, kk, max_iter, seed)),
        Method::Mppca { rank } if k == 0 => {
            HeadFit::soft(mppca_auto(features, 1, hi, rank, max_iter, seed))
        }
        Method::Mppca { rank } => HeadFit::soft(mppca(features, kk, rank, max_iter, seed)),
        Method::Ward if k == 0 => HeadFit::hard(ward_hac_auto(features, 2, hi).labels),
        Method::Ward => HeadFit::hard(ward_hac(features, kk).labels),
        Method::Agglomerative { linkage } if k == 0 => {
            HeadFit::hard(agglomerative_auto(features, linkage, 2, hi).labels)
        }
        Method::Agglomerative { linkage } => {
            HeadFit::hard(agglomerative(features, linkage, kk).labels)
        }
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
            let auto = movmf_auto(features, 1, hi, max_iter, seed).means.len();
            HeadFit::hard(
                spherical_kmeans(features, auto.min(nlv).max(1), max_iter, 4, seed).labels,
            )
        }
        Method::SphericalKMeans => {
            HeadFit::hard(spherical_kmeans(features, kk, max_iter, 4, seed).labels)
        }
        Method::Movmf if k == 0 => HeadFit::soft(movmf_auto(features, 1, hi, max_iter, seed)),
        Method::Movmf => HeadFit::soft(movmf(features, kk, max_iter, seed)),
        Method::GmmToeplitz if k == 0 => {
            HeadFit::soft(gmm_toeplitz_auto(features, 1, hi, max_iter, seed))
        }
        Method::GmmToeplitz => HeadFit::soft(gmm_toeplitz(features, kk, max_iter, seed)),
        Method::GmmToeplitzFull if k == 0 => {
            HeadFit::soft(gmm_toeplitz_full_auto(features, 1, hi, max_iter, seed))
        }
        Method::GmmToeplitzFull => HeadFit::soft(gmm_toeplitz_full(features, kk, max_iter, seed)),
        Method::GmmToeplitzGs if k == 0 => {
            HeadFit::soft(gmm_toeplitz_gs_auto(features, 1, hi, max_iter, seed))
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
    use crate::clustering::testutil::{ari, blob_leaves, blobs};
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
        let model = Model::fit(tree, 4, Method::KMeans, 100, 7, 0);
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
        let model = Model::fit(tree, 3, Method::Gmm, 200, 3, 0);
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
        let model = Model::fit(tree, 2, Method::KMeans, 100, 1, 0);
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
            Method::Mppca { rank: 1 },
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
                let labels = fit_head(&feats, k, method, 100, 1, 0).labels;
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
    fn no_automatic_selector_is_maximised_by_one_cluster_per_leaf() {
        // `kmeans_auto`'s Pelleg-Moore BIC scored the between-leaf sum of squares alone. `Σ_l S_l` is
        // constant in `k`, but it sits inside `ln σ̂²`, so at `k = n_leaves` the estimate reached its
        // floor and the score diverged: one cluster per leaf beat every real answer. `AUTO_K_MAX`
        // hid it — no shipped path passes a `k_max` anywhere near the leaf count.
        //
        // Every other selector here reads the leaf scatter through a different route (`gmm`'s M-step
        // adds `var[i][d]`, `mppca` reads `second_moment`, the Toeplitz ladder reads `variance(t)`,
        // the two HAC drivers read `ssd`), so none should have the defect. That is an argument from
        // reading the code, and this is the measurement: hand each selector a `k_max` equal to the
        // leaf count — the regime the cap has always kept them out of — and require an answer nearer
        // the real groups than the leaf count.
        //
        // Ten dimensions, not the 2-D fixture the rest of this module uses. A split has to buy
        // `½·n·d·ln(S₁/S₂)` against a cost of `n·ln 2`, so at `d = 2` a greedy splitter refuses
        // almost everything and passes this whatever its score does; the runaway only becomes
        // reachable as `d` grows.
        let (feats, _truth) = blob_leaves(6, 10, 40, 0);
        let n = feats.len();
        assert_eq!(n, 24, "the fixture is four leaves per blob");
        let selected: [(&str, usize); 7] = [
            ("kmeans", kmeans_auto(&feats, 1, n, 100, 1).centers.len()),
            ("xmeans", xmeans(&feats, 2, n, 100, 1).centers.len()),
            ("gmm", gmm_diagonal_auto(&feats, 1, n, 100, 1).means.len()),
            ("gmm-full", gmm_full_auto(&feats, 1, n, 100, 1).means.len()),
            ("mppca", mppca_auto(&feats, 1, n, 1, 100, 1).means.len()),
            ("ward", distinct_count(&ward_hac_auto(&feats, 2, n).labels)),
            (
                "average",
                distinct_count(&agglomerative_auto(&feats, Linkage::Average, 2, n).labels),
            ),
        ];
        for (name, k) in selected {
            assert!(
                k < n / 2,
                "{name} chose {k} of {n} leaves — a selector that runs to the leaf count is \
                 reporting the summary back, not clustering it"
            );
        }
    }

    #[test]
    fn every_head_takes_its_automatic_arm_only_when_k_is_zero() {
        let feats = dispatch_leaves();
        for (name, method) in [
            ("kmeans", Method::KMeans),
            ("gmm", Method::Gmm),
            ("gmm-full", Method::GmmFull),
            ("ward", Method::Ward),
            (
                "average",
                Method::Agglomerative {
                    linkage: Linkage::Average,
                },
            ),
            (
                "median",
                Method::Agglomerative {
                    linkage: Linkage::Median,
                },
            ),
            ("spherical-kmeans", Method::SphericalKMeans),
            ("movmf", Method::Movmf),
            ("gmm-toeplitz", Method::GmmToeplitz),
            ("gmm-toeplitz-full", Method::GmmToeplitzFull),
            ("gmm-toeplitz-gs", Method::GmmToeplitzGs),
            ("mppca", Method::Mppca { rank: 2 }),
        ] {
            let auto = distinct_count(&fit_head(&feats, 0, method, 100, 1, 0).labels);
            let one = distinct_count(&fit_head(&feats, 1, method, 100, 1, 0).labels);
            let two = distinct_count(&fit_head(&feats, 2, method, 100, 1, 0).labels);
            assert!(auto > 1, "{name}: the automatic arm collapsed to {auto}");
            assert_eq!(one, 1, "{name}: k = 1 did not take the fixed arm");
            assert_eq!(two, 2, "{name}: k = 2 did not take the fixed arm");
        }
    }

    #[test]
    fn xmeans_reads_k_as_a_cap_rather_than_as_a_target() {
        // The head deliberately sits outside `every_head_takes_its_automatic_arm_only_when_k_is_zero`:
        // `n_clusters` bounds x-means rather than fixing it, so `k = 2` may legitimately answer 1 and
        // there is no "fixed arm" to take. What must hold is that the bound binds, that `k = 0` does
        // not silently inherit the sweep's `AUTO_K_MAX`, and that the head still finds structure.
        let feats = dispatch_leaves();
        for k in 1..=4 {
            let got = distinct_count(&fit_head(&feats, k, Method::XMeans, 100, 1, 0).labels);
            assert!(got <= k, "k = {k} is a cap, but the head returned {got}");
        }
        assert_eq!(
            distinct_count(&fit_head(&feats, 1, Method::XMeans, 100, 1, 0).labels),
            1,
            "a cap of 1 leaves nothing to split"
        );
        let auto = distinct_count(&fit_head(&feats, 0, Method::XMeans, 100, 1, 0).labels);
        assert!(auto > 1, "the automatic arm collapsed to {auto}");
        assert!(
            auto <= feats.len(),
            "the automatic arm is bounded by the leaf count, not by AUTO_K_MAX"
        );
    }

    /// What the automatic-`k` ceiling costs and what it hides — the table in `docs/USAGE.md`.
    ///
    /// A sweep refits at every candidate `k`, so its work is `Σ_{k≤K} k = O(K²)`; the two HAC
    /// drivers build one dendrogram and re-cut it, and x-means stops on its own split test, so for
    /// those the ceiling buys nothing and costs the answer. Reported, not asserted — run with
    /// `cargo test --release --all-features --lib -- --ignored --nocapture the_cost_of_the_auto_k`.
    #[test]
    #[ignore = "measurement, not an assertion"]
    fn the_cost_of_the_auto_k_ceiling() {
        use std::time::Instant;
        let (feats, truth) = blob_leaves(120, 64, 64, 4);
        let n = feats.len();
        println!("{n} leaves, d = 64, 120 true groups; AUTO_K_MAX = {AUTO_K_MAX}");
        let row = |name: &str, cap: usize, t: Instant, labels: &[usize], k: usize| {
            println!(
                "{name:<8} cap={cap:>5} {:>9.1} ms  k={k:>4}  ARI={:.3}",
                t.elapsed().as_secs_f64() * 1e3,
                ari(labels, &truth)
            );
        };
        for &cap in &[AUTO_K_MAX, 120, n] {
            let t = Instant::now();
            let w = ward_hac_auto(&feats, 2, cap);
            row("ward", cap, t, &w.labels, distinct_count(&w.labels));
        }
        // The two sweeps are quadratic in the cap, so the leaf-count column is left unmeasured
        // rather than run for minutes to restate what 120 already shows.
        for &cap in &[AUTO_K_MAX, 120] {
            let t = Instant::now();
            let km = kmeans_auto(&feats, 1, cap, 100, 0);
            row("kmeans", cap, t, &km.labels, km.centers.len());
            let t = Instant::now();
            let g = gmm_diagonal_auto(&feats, 1, cap, 100, 0);
            row("gmm", cap, t, &g.labels, g.means.len());
        }
        let t = Instant::now();
        let x = xmeans(&feats, 2, n, 100, 0);
        row("xmeans", n, t, &x.labels, x.centers.len());
    }

    #[test]
    fn only_the_selectors_that_pay_for_a_wider_search_are_bounded_by_the_default() {
        // Forty groups, more than double the shipped ceiling. A sweep refits at every candidate `k`
        // so its ceiling is a cost guard and it stops at 20; a driver that cuts one dendrogram, and
        // a splitter that stops on its own test, pay a linear pass for the same reach and have no
        // reason to be bounded below the leaf count.
        let (feats, truth) = blob_leaves(40, 10, 40, 7);
        for (name, method) in [
            ("kmeans", Method::KMeans),
            ("gmm", Method::Gmm),
            ("gmm-full", Method::GmmFull),
            ("mppca", Method::Mppca { rank: 2 }),
        ] {
            let got = distinct_count(&fit_head(&feats, 0, method, 100, 1, 0).labels);
            assert_eq!(
                got, AUTO_K_MAX,
                "{name} is a sweep and must stop at the ceiling"
            );
        }
        for (name, method) in [
            ("ward", Method::Ward),
            (
                "average",
                Method::Agglomerative {
                    linkage: Linkage::Average,
                },
            ),
            ("xmeans", Method::XMeans),
        ] {
            let fit = fit_head(&feats, 0, method, 100, 1, 0);
            assert_eq!(
                distinct_count(&fit.labels),
                40,
                "{name} did not reach the true count"
            );
            assert!(ari(&fit.labels, &truth) > 0.99, "{name}");
        }
    }

    #[test]
    fn auto_k_max_is_the_override_that_lets_a_sweep_reach_the_same_count() {
        // The ceiling is a cost guard, not a statement about the data, so it has to be liftable —
        // and lifting it is what a sweep needs to answer 40 rather than 20. Raised to 60 rather
        // than to the leaf count because the sweeps are `O(K²)` and this is a unit test;
        // `the_cost_of_the_auto_k_cap` is where the whole curve is measured. `gmm-full` is left out
        // deliberately: with the ceiling lifted it answers 38 on this fixture at ARI 0.936, which is
        // its own score's choice and not the ceiling's doing.
        let (feats, truth) = blob_leaves(40, 10, 40, 7);
        for (name, method) in [("kmeans", Method::KMeans), ("gmm", Method::Gmm)] {
            let fit = fit_head(&feats, 0, method, 100, 1, 60);
            assert_eq!(
                distinct_count(&fit.labels),
                40,
                "{name} with the ceiling lifted"
            );
            assert!(ari(&fit.labels, &truth) > 0.99, "{name}");
        }
        // And it binds downward too, on a head the default does not bound at all: `ward` answers
        // 40 here when left alone. A ceiling is not a target, so what must hold is `<=`.
        let held = distinct_count(&fit_head(&feats, 0, Method::Ward, 100, 1, 6).labels);
        assert!(
            (2..=6).contains(&held),
            "auto_k_max must cap as well as raise, but ward answered {held}"
        );
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
            ("xmeans", Method::XMeans),
            ("spherical-kmeans", Method::SphericalKMeans),
            ("gmm", Method::Gmm),
            ("movmf", Method::Movmf),
            ("ward", Method::Ward),
            (
                "centroid",
                Method::Agglomerative {
                    linkage: Linkage::Centroid,
                },
            ),
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
            let model = Model::fit(tree, 3, method, 100, 1, 0);
            let ok = matches!(
                (assignment_rule(method), &model.assign),
                (Rule::Centroid { .. }, Assignment::Centers { .. })
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
        let model = Model::fit(tree, 2, Method::KMeans, 100, 1, 0);
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

    /// The four-corner square around each of two well-separated cells, so the exact per-cluster mean
    /// is known in closed form and the fixture does not have to trust the routine that computes it.
    fn two_squares() -> Vec<f64> {
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, // cell A, mean (0.5, 0.5)
            10.0, 0.0, 10.0, 1.0, 11.0, 0.0, 11.0, 1.0, // cell B, mean (10.5, 0.5)
        ]
    }

    #[test]
    fn a_lloyd_sweep_lands_on_the_exact_group_means_and_then_stops() {
        // Both centres start displaced but on the right side of the gap, so the very first
        // assignment is already final: the second sweep can only confirm it. A routine that keeps
        // sweeping regardless would return 10 here, and one that never updates would leave the 2.0.
        let mut centers = vec![2.0, 0.5, 9.0, 0.5];
        let sweeps = refine_centers(&mut centers, &two_squares(), 8, 2, false, 10);
        assert_eq!(sweeps, 2, "the fixed point was not detected");
        for (got, want) in centers.iter().zip([0.5, 0.5, 10.5, 0.5]) {
            assert!((got - want).abs() < 1e-12, "{centers:?}");
        }
    }

    #[test]
    fn a_centre_that_attracts_no_point_keeps_its_position() {
        // Three centres over two cells: the third is far enough that every point prefers one of the
        // first two. The documented policy is to leave it where it is rather than re-seed it, which
        // would be a second algorithm with its own seeding rule.
        let mut centers = vec![0.0, 0.0, 10.0, 0.0, 500.0, 500.0];
        refine_centers(&mut centers, &two_squares(), 8, 2, false, 10);
        assert_eq!(
            &centers[..4],
            &[0.5, 0.5, 10.5, 0.5],
            "the sweep did not run"
        );
        assert_eq!(&centers[4..], &[500.0, 500.0]);
    }

    #[test]
    fn the_unit_flag_returns_each_refined_centre_to_the_sphere() {
        // Raw means of points on the sphere sit strictly inside it, so an un-normalized update is
        // visible as a norm below one — this asserts the re-projection, not merely that it ran.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let pts = vec![1.0, 0.0, s, s, 0.0, 1.0, -1.0, 0.0, -s, -s, 0.0, -1.0];
        let mut centers = vec![1.0, 0.1, -1.0, -0.1];
        refine_centers(&mut centers, &pts, 6, 2, true, 10);
        for c in centers.chunks(2) {
            let norm = (c[0] * c[0] + c[1] * c[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "norm = {norm}");
        }
    }

    #[test]
    fn refinement_lowers_the_k_means_objective_it_optimizes() {
        // Phase 3 minimizes the objective over the *leaf summary*; Phase 4 minimizes it over the raw
        // points. Lloyd is monotone, so the second can only improve on the first — and on a summary
        // this coarse (16 leaves for 4 blobs) it has room to.
        let mut rng = SplitMix64::new(11);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, _) = blobs(&mut rng, 300, &centers, 1.4);
        let flat: Vec<f64> = pts.iter().flatten().copied().collect();
        let mut tree: CFTree<f64, Diagonal<f64>, _, _> =
            CFTree::new(2, 4, 4, 4.0, 16, CentroidEuclidean, CentroidEuclidean);
        for p in &pts {
            tree.insert(p);
        }
        let mut model = Model::fit(tree, 4, Method::KMeans, 100, 3, 0);
        let wcss = |m: &Model<f64, Diagonal<f64>, _, _>| -> f64 {
            let Assignment::Centers { centers, .. } = &m.assign else {
                unreachable!("k-means is a centroid head")
            };
            pts.iter()
                .map(|p| {
                    centers
                        .iter()
                        .map(|(_, c)| sq_euclidean(p, c))
                        .fold(f64::INFINITY, f64::min)
                })
                .sum()
        };
        let before = wcss(&model);
        let sweeps = model.refine(&flat, pts.len(), 2, 20);
        let after = wcss(&model);
        assert!(sweeps > 0, "no sweep ran");
        assert!(after < before, "{after} !< {before}");
    }

    #[test]
    fn each_degenerate_argument_declines_the_sweep_on_its_own() {
        // The guard is a disjunction, so every clause has to be lethal by itself. With no centre the
        // per-cluster counter is empty and the first assignment indexes past its end; with no point
        // nothing can move, which a sweep that ran anyway would mistake for a converged pass.
        let mut empty: Vec<f64> = vec![];
        assert_eq!(
            refine_centers(&mut empty, &two_squares(), 8, 2, false, 10),
            0
        );
        let mut centers = vec![0.0, 0.0, 10.0, 0.0];
        assert_eq!(refine_centers(&mut centers, &[], 0, 2, false, 10), 0);
        assert_eq!(
            refine_centers(&mut centers, &two_squares(), 8, 2, false, 0),
            0
        );
        assert_eq!(
            centers,
            [0.0, 0.0, 10.0, 0.0],
            "a declined call still moved a centre"
        );
    }

    #[test]
    fn a_point_equidistant_from_two_centres_goes_to_the_earlier_one() {
        // Ties are not measure-zero on a summary: leaf centroids land exactly between two centres
        // often enough to matter, and which side takes them decides both means. A strict `<` keeps
        // the earlier centre; `<=` hands the tie to the later one, leaving the earlier stranded.
        let mut centers = vec![0.0, 0.0, 2.0, 0.0];
        let sweeps = refine_centers(&mut centers, &[1.0, 0.0], 1, 2, false, 10);
        assert_eq!(sweeps, 2, "the fixed point was not detected");
        assert_eq!(
            centers,
            [1.0, 0.0, 2.0, 0.0],
            "the tie went to the wrong centre"
        );
    }

    #[test]
    fn a_centre_whose_points_cancel_survives_the_unit_projection() {
        // Antipodal points average to the origin, which has no direction to project onto. Dividing
        // by that norm writes NaN into the model, and NaN loses every later `<` comparison silently
        // rather than failing — so the guard is what keeps a degenerate centre merely useless.
        let mut centers = vec![0.0, 1.0];
        refine_centers(&mut centers, &[1.0, 0.0, -1.0, 0.0], 2, 2, true, 10);
        assert_eq!(centers, [0.0, 0.0], "{centers:?}");
    }

    #[test]
    fn refinement_is_declined_by_every_head_without_a_centre_model() {
        // A mixture assigns by maximum posterior and Ward/Spectral/Leiden by microcluster; sweeping
        // centres over any of them would silently substitute the Voronoi partition they do not use.
        let mut rng = SplitMix64::new(4);
        let centers = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let (pts, _) = blobs(&mut rng, 200, &centers, 0.8);
        let flat: Vec<f64> = pts.iter().flatten().copied().collect();
        for method in [Method::Gmm, Method::Ward, Method::Spectral] {
            let mut tree: CFTree<f64, Diagonal<f64>, _, _> =
                CFTree::new(2, 16, 16, 0.05, 200, CentroidEuclidean, CentroidEuclidean);
            for p in &pts {
                tree.insert(p);
            }
            let mut model = Model::fit(tree, 3, method, 100, 2, 0);
            let before: Vec<usize> = pts.iter().map(|p| model.predict(p)).collect();
            assert_eq!(model.refine(&flat, pts.len(), 2, 10), 0, "{method:?}");
            let after: Vec<usize> = pts.iter().map(|p| model.predict(p)).collect();
            assert_eq!(before, after, "{method:?} relabelled points");
        }
    }
}
