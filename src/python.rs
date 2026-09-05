//! Python bindings (feature = "python").
//!
//! Two entry points over the Rust core:
//! * [`fit_predict`] — one-shot function: build a CF-tree from a 2-D array and label every row
//!   (the heavy work runs detached from the interpreter via `Python::detach`).
//! * [`Betula`] — a stateful, scikit-learn-style estimator with `partial_fit` for streaming /
//!   out-of-core data (memory-bounded CF-tree), then `fit` / `predict` / `fit_predict`.
//!
//! Parametric heads (`kmeans`, `gmm`, `gmm-full`; `n_clusters=0` ⇒ BIC auto-k) and the density
//! head (`hdbscan`, where `-1` marks noise).

use numpy::ndarray::{Array1, Array2};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use std::ffi::CString;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::bregman::{
    BregmanCentroid, BregmanCf, BregmanDivergence, BregmanIncrease, ItakuraSaito, KullbackLeibler,
    Logistic, SquaredEuclidean,
};
use crate::clustering::hdbscan::hdbscan_with;
use crate::clustering::hyperbolic::lorentz_dot;
use crate::clustering::nmf::{Projection, ProjectionKind, ProjectionSpec};
use crate::clustering::scalespace::scale_space;
use crate::clustering::{
    BlockWeights, ConstraintError, Linkage, MixedCf, MixedRows, MixedSchema, bregman_agglomerative,
    bregman_em, bregman_kmeans, cop_kmeans, kprototypes, nearest_micro, project_to_sheet,
    summarize_mixed,
};
use crate::clustering::{DcObjective, dc_clustering};
use crate::clustering::{Reachability, optics};
use crate::distance::{
    AverageIntercluster, AverageIntracluster, CFDistance, CentroidEuclidean, CentroidManhattan,
    MahalanobisChi2, Radius, SubspaceChi2, VarianceIncrease,
};
use crate::feature::{ClusterFeature, Diagonal, FdSketch, Full, Spherical};
use crate::linalg::{cholesky_lower, mahalanobis_sq_from_chol};
use crate::mixture::{Mixture, SparseAssigner};
use crate::model::{
    Method, Model, Rule, assignment_rule, auto_k_ceiling, fit_head, refinable, refine_centers,
};
use crate::order::{canonical_permutation, canonical_permutation_csr, canonical_shards};
use crate::sparse::{SparseCentroids, normalize_csr_rows, summarize_sparse};
use crate::stats::chi2_quantile;
use crate::stream::{DbStream, DenStream, DriftReport};
use crate::topology::{Lens, Link, MapperGraph, MapperParams, mapper};
use crate::tree::CFTree;
use crate::types::Real;
use crate::wasserstein::{GaussianMixture, Spread, mixture_w2};
use crate::window::WindowStream;

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum Kind {
    Parametric(Method),
    Hdbscan {
        min_samples: usize,
        min_cluster_size: usize,
        /// Out-degree of the proximity graph the mutual-reachability MST is taken over; `0` keeps
        /// the exact complete graph.
        graph_degree: usize,
    },
    /// Exact `k`-center / `k`-median in the density-connectivity ultrametric — HDBSCAN\*'s own
    /// spanning tree, cut for a `k` the caller names instead of a `min_cluster_size`.
    DcDist {
        objective: DcObjective,
        min_samples: usize,
        graph_degree: usize,
    },
    /// Scale-space KDE-mode clustering with persistence-selected scale (no `k`, no bandwidth).
    ScaleSpace,
}

impl Kind {
    /// Whether the head partitions the leaves into a caller-supplied `k`. HDBSCAN, scale-space and
    /// Leiden discover their own count, so a leaf budget stated relative to `k` says nothing there.
    fn consumes_k(self) -> bool {
        matches!(self, Kind::DcDist { .. })
            || matches!(self, Kind::Parametric(m) if !matches!(m, Method::Leiden { .. }))
    }
}

/// Leaves per requested cluster below which the summary is too coarse to carry `k` clusters.
///
/// Measured on the `ward` head over three seeds (`local/scratch/leaves_per_k_sweep.py`): on
/// well-separated synthetic data the score is already at its plateau at ≈2 leaves per cluster
/// (1.03× and 1.00× of the ratio-64 score at k=50 and k=200) and collapses to 0.71× / 0.45× at ≈1,
/// while `digits` and `covtype` score 0.000 and 0.003 at ≈1. Lang's thesis reports the same floor
/// from the other end (Sec. 5.5.4: k=5000 under a 10 000-leaf cap, "fewer than two cluster features
/// per center"). It is a *floor*, not a recommendation — above it there is no universal direction,
/// `covtype` peaking at ≈8 leaves per cluster and declining after.
const MIN_LEAVES_PER_CLUSTER: usize = 2;

/// Below this many points the cost of an uncompressed tree is not worth a warning.
const NO_COMPRESSION_FLOOR: usize = 5_000;

/// Warn once per fit when the realised leaf count cannot support `n_clusters`.
///
/// The check uses the **realised** leaf count rather than `max_leaves`: the tree can settle well
/// below its cap, and with `n < max_leaves` the cap is never the binding constraint at all.
///
/// Auto-`k` (`k == 0`) needs no separate arm — it selects a count from the leaves it has, and the
/// ratio test is vacuously satisfied at `k == 0`. `k == 1` does need one: a single cluster
/// separates nothing, so a one-leaf summary answers it exactly and the floor does not apply.
/// `leaves == 0` likewise, since nothing was summarized at all.
///
/// Only call this for a head that partitions the summary into a caller-supplied `k` — the ratio is
/// meaningless for one that discovers its own count. [`Kind::consumes_k`] is that predicate.
fn warn_leaf_budget(py: Python<'_>, leaves: usize, k: usize, max_leaves: usize) -> PyResult<()> {
    if k < 2 || leaves == 0 || leaves >= MIN_LEAVES_PER_CLUSTER * k {
        return Ok(());
    }
    let msg = CString::new(format!(
        "the CF-tree summarized the data into {leaves} leaves but n_clusters={k} was requested \
         ({:.2} leaves per cluster). Below {MIN_LEAVES_PER_CLUSTER} the summary cannot separate \
         that many clusters and the partition degrades. Raise max_leaves (currently {max_leaves}), \
         lower threshold, or lower n_clusters.",
        leaves as f64 / k as f64,
    ))
    .expect("the formatted warning contains no interior NUL");
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 1)
}

/// Warn when an automatic `k` lands exactly on its own ceiling.
///
/// The sweep-based selectors refit the whole head at every candidate `k`, so their cost is quadratic
/// in the ceiling and it cannot simply be removed. What it can stop being is silent. An argmax that
/// sits on the last candidate is not evidence that the data has that many groups — it is evidence
/// that the search stopped before the score turned over. Measured on 480 leaves in 64 dimensions
/// holding 120 true groups, `method="gmm"` with `n_clusters=0` returns 20 at the default ceiling and
/// scores ARI 0.109; at `auto_k_max=120` it returns 120 and scores 1.000.
///
/// Only for the automatic arm of a head that reads the ceiling: `spectral` resolves `k` from the
/// eigengap and `leiden` from the graph, so neither is bounded by it.
fn warn_auto_k_saturated(
    py: Python<'_>,
    kind: Kind,
    n_clusters: usize,
    chosen: usize,
    leaves: usize,
    auto_k_max: usize,
) -> PyResult<()> {
    let Kind::Parametric(method) = kind else {
        return Ok(());
    };
    if n_clusters != 0
        || matches!(method, Method::Spectral | Method::Leiden { .. })
        || chosen < auto_k_ceiling(method, leaves, auto_k_max)
    {
        return Ok(());
    }
    let msg = CString::new(format!(
        "n_clusters=0 selected k={chosen}, which is the ceiling the search was bounded by, so the \
         true count may be higher — the score never turned over. Raise auto_k_max (currently \
         {auto_k_max}, 0 = the default), set n_clusters explicitly, or use method='xmeans', whose \
         stopping rule is a split test rather than a cost guard."
    ))
    .expect("the formatted warning contains no interior NUL");
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 1)
}

/// Warn when the CF-tree compressed nothing: one leaf per point.
///
/// With `threshold = 0` a point is absorbed only by an entry it equals exactly, so a budget that
/// never binds leaves the summary at `n` micro-clusters — the tree is built, descended and split for
/// every point, and then Phase 3 runs on the raw rows anyway. It is not a wrong answer, it is a
/// silent price: measured single-threaded on 64-dimensional blobs, the same fit costs **3.8× more at
/// n = 8 000 and 14× more at n = 40 000** than the same call with a binding `max_leaves = 2000`, and
/// 36–46× more than clustering the raw rows directly.
///
/// The check reads the **realised** leaf count rather than comparing `n` against `max_leaves`,
/// because that is the thing that is actually true — a tree can settle below its cap for other
/// reasons, and only `leaves == n` says no two points ever shared a micro-cluster.
///
/// `NO_COMPRESSION_FLOOR` keeps the warning off inputs small enough that the absolute cost is
/// irrelevant, where "your summary is your data" is the obviously intended reading.
fn warn_no_compression(py: Python<'_>, leaves: usize, n: usize, max_leaves: usize) -> PyResult<()> {
    if leaves < NO_COMPRESSION_FLOOR || leaves != n {
        return Ok(());
    }
    let msg = CString::new(format!(
        "the CF-tree summarized {n} points into {leaves} leaves — one per point, so nothing was \
         compressed and the tree is pure overhead (measured 3.8x the fit time at n=8000 and 14x at \
         n=40000 against a binding budget). max_leaves={max_leaves} never binds at this n and \
         threshold=0.0 absorbs only exact duplicates. Lower max_leaves, or raise threshold."
    ))
    .expect("the formatted warning contains no interior NUL");
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 1)
}

/// Warn when a Gaussian head that wants per-dimension variance is fed a feature that has none.
///
/// `Spherical::variance(_d)` ignores its argument — it returns `ssd / (w · dim)`, one **isotropic**
/// number for every dimension, because a spherical cluster feature carries a scalar scatter and
/// cannot carry more. The diagonal M-step adds that number to all `dim` per-component variances (and
/// `gmm-full` inherits the same thing through `cov_dense`'s diagonal default). At zero compression
/// every leaf is a singleton, `ssd = 0`, and nothing is added. Under compression each component is
/// inflated **equally in every dimension** by however much leaf scatter it happens to cover, so a
/// dimension with genuinely near-zero variance is lifted to the isotropic average. The fit survives
/// this; the labelling does not, because the maximum-posterior argmax is dominated by
/// `ln|Σ_c| = Σ_d ln σ²_cd` while a nearest-centroid rule ignores `Σ` entirely.
///
/// Measured on `digits` (1797 × 64, `k = 10`, medians of seeds 0/1/2), ARI by cluster feature:
///
/// | leaf budget | `spherical` | `fd` | `full` |
/// |---|---|---|---|
/// | 1797 (×1.0) | 0.4613 | 0.4613 | 0.4613 |
/// | 900 (×2.0) | **0.0088** | 0.3840 | 0.4403 |
/// | 500 (×3.7) | **0.0104** | 0.4562 | 0.5083 |
///
/// The ×1.0 row is the control: with no within-leaf scatter to add, all three agree to the digit.
/// `gmm-full` shows the same collapse on the same feature (0.0096 at 1200 leaves, 0.0115 at 500) and
/// none of it on `feature="full"`. The other heads read the same isotropic `variance(d)` but were not
/// measured, so they are not covered here.
fn warn_isotropic_gaussian(
    py: Python<'_>,
    method: &str,
    feature: &str,
    leaves: usize,
    n: usize,
) -> PyResult<()> {
    if feature != "spherical" || leaves >= n || !matches!(method, "gmm" | "gmm-full") {
        return Ok(());
    }
    let msg = CString::new(format!(
        "method=\"{method}\" fits a per-dimension covariance, but feature=\"spherical\" carries only \
         a scalar within-leaf scatter, so the same isotropic number is added to every dimension. \
         With {leaves} leaves for {n} points that is a real compression, and it distorts each \
         component's log-determinant — measured on digits at x2.0 compression the labels fall to \
         ARI 0.0088 against 0.4403 for feature=\"full\", while the fitted centres stay healthy. Pass \
         feature=\"full\" (or \"fd\" for high dimension), or keep max_leaves at n so the tree does \
         not compress."
    ))
    .expect("the formatted warning contains no interior NUL");
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 1)
}

/// Map the `method` keyword (+ HDBSCAN params) to an internal [`Kind`].
#[allow(clippy::too_many_arguments)] // one parameter per head-specific keyword, as the callers have
fn parse_method(
    method: &str,
    min_samples: usize,
    min_cluster_size: usize,
    resolution: f64,
    covariance_weight: f64,
    tangent_weight: f64,
    tangent_rank: usize,
    rank: usize,
    graph_degree: usize,
    fuzzifier: f64,
) -> PyResult<Kind> {
    match method {
        "kmeans" => Ok(Kind::Parametric(Method::KMeans)),
        "xmeans" => Ok(Kind::Parametric(Method::XMeans)),
        "gmm" => Ok(Kind::Parametric(Method::Gmm)),
        "gmm-full" => Ok(Kind::Parametric(Method::GmmFull)),
        "ward" => Ok(Kind::Parametric(Method::Ward)),
        "average" => Ok(Kind::Parametric(Method::Agglomerative {
            linkage: Linkage::Average,
        })),
        "weighted" => Ok(Kind::Parametric(Method::Agglomerative {
            linkage: Linkage::Weighted,
        })),
        "centroid" => Ok(Kind::Parametric(Method::Agglomerative {
            linkage: Linkage::Centroid,
        })),
        "median" => Ok(Kind::Parametric(Method::Agglomerative {
            linkage: Linkage::Median,
        })),
        "spectral" => Ok(Kind::Parametric(Method::Spectral)),
        "leiden" => Ok(Kind::Parametric(Method::Leiden {
            resolution,
            cpm: false,
            cov_weight: covariance_weight,
            tangent_weight,
            tangent_rank,
        })),
        "leiden-cpm" => Ok(Kind::Parametric(Method::Leiden {
            resolution,
            cpm: true,
            cov_weight: covariance_weight,
            tangent_weight,
            tangent_rank,
        })),
        "kmedoids" => Ok(Kind::Parametric(Method::KMedoids)),
        "fuzzy-cmeans" => {
            // The exponent is a modelling choice with no likelihood behind it, so it is validated
            // here rather than clamped silently: at `m <= 1` the membership exponent `1/(m−1)` is
            // undefined or negative, which inverts the rule into "farthest centre wins".
            if !fuzzifier.is_finite() || fuzzifier <= 1.0 {
                return Err(PyValueError::new_err(
                    "fuzzifier must be finite and > 1 (2.0 is the default; m -> 1 is k-means)",
                ));
            }
            Ok(Kind::Parametric(Method::FuzzyCMeans { fuzzifier }))
        }
        "spherical-kmeans" => Ok(Kind::Parametric(Method::SphericalKMeans)),
        "vmf" => Ok(Kind::Parametric(Method::Movmf)),
        "watson" => Ok(Kind::Parametric(Method::Watson)),
        "gmm-toeplitz" => Ok(Kind::Parametric(Method::GmmToeplitz)),
        "gmm-toeplitz-full" => Ok(Kind::Parametric(Method::GmmToeplitzFull)),
        "gmm-toeplitz-gs" => Ok(Kind::Parametric(Method::GmmToeplitzGs)),
        "mppca" => Ok(Kind::Parametric(Method::Mppca { rank })),
        "mfa" => Ok(Kind::Parametric(Method::Mfa { rank })),
        "hyperbolic" => Ok(Kind::Parametric(Method::Hyperbolic)),
        "dc-center" => Ok(Kind::DcDist {
            objective: DcObjective::Center,
            min_samples,
            graph_degree,
        }),
        "dc-median" => Ok(Kind::DcDist {
            objective: DcObjective::Median,
            min_samples,
            graph_degree,
        }),
        "hdbscan" => Ok(Kind::Hdbscan {
            min_samples,
            min_cluster_size,
            graph_degree,
        }),
        "scale-space" => Ok(Kind::ScaleSpace),
        _ => Err(PyValueError::new_err(
            "method must be 'kmeans', 'xmeans', 'kmedoids', 'fuzzy-cmeans', 'gmm', 'gmm-full', \
             'mppca', 'mfa', 'ward', 'average', 'weighted', 'centroid', 'median', 'spectral', 'leiden', \
             'leiden-cpm', 'spherical-kmeans', 'vmf', 'watson', 'hyperbolic', 'gmm-toeplitz', \
             'gmm-toeplitz-full', 'gmm-toeplitz-gs', 'hdbscan', 'dc-center', 'dc-median' or \
             'scale-space'",
        )),
    }
}

/// Label leaf features with the configured head and, for the GMM heads, also return the per-leaf
/// soft responsibility matrix flattened `(resp, k)` (`None` otherwise) so `predict_proba` can read a
/// true posterior without recomputing the E-step. HDBSCAN keeps `-1` for noise; parametric labels are
/// cast to `i64`. Generic over the element type so it serves both the `f64` and `f32` trees.
/// The `projection` values this binding accepts, as one string so the parser and every error message
/// cannot drift apart.
const PROJECTION_CHOICES: &str =
    "projection must be 'none', 'weighted-nmf', 'weighted-nmf-kl' or 'svd'";

/// Resolve the optional Phase-3 projection, or `None`. `"weighted-nmf"` = Frobenius NMF,
/// `"weighted-nmf-kl"` = KL NMF (count data), `"svd"` = CF-weighted PCA, `"none"`/`""` off.
fn parse_projection(
    projection: &str,
    projection_dim: usize,
    projection_max_iter: usize,
) -> PyResult<Option<ProjectionSpec>> {
    let kind = match projection {
        "none" | "" => return Ok(None),
        "weighted-nmf" => ProjectionKind::Nmf {
            kl: false,
            max_iter: projection_max_iter,
        },
        "weighted-nmf-kl" => ProjectionKind::Nmf {
            kl: true,
            max_iter: projection_max_iter,
        },
        "svd" => ProjectionKind::Svd,
        _ => return Err(PyValueError::new_err(PROJECTION_CHOICES)),
    };
    if projection_dim == 0 {
        return Err(PyValueError::new_err(
            "projection_dim must be > 0 for a projection",
        ));
    }
    if projection_max_iter == 0 && matches!(kind, ProjectionKind::Nmf { .. }) {
        return Err(PyValueError::new_err(
            "projection_max_iter must be > 0 for a 'weighted-nmf' projection",
        ));
    }
    Ok(Some(ProjectionSpec {
        rank: projection_dim,
        kind,
    }))
}

/// NMF is defined only for nonnegative data; reject signed input rather than silently shifting it
/// (a shift changes angles and the cosine geometry).
fn require_nonnegative<R: Real>(flat: &[R]) -> PyResult<()> {
    if flat.iter().any(|&v| v < R::zero()) {
        return Err(PyValueError::new_err(
            "projection='weighted-nmf' requires nonnegative data (X >= 0) — NMF is undefined for \
             signed values. For signed embeddings use method='vmf'/'spherical-kmeans' or reduce with \
             PCA / TruncatedSVD first.",
        ));
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn dispatch_kind<R: Real, C: ClusterFeature<R>>(
    feats: &[C],
    kind: Kind,
    k: usize,
    max_iter: usize,
    seed: u64,
    auto_k_max: usize,
) -> Dispatch {
    match kind {
        Kind::Parametric(method) => {
            let fit = fit_head(feats, k, method, max_iter, seed, auto_k_max);
            let proba = fit.resp.map(|r| {
                let kk = r.first().map_or(0, |row| row.len());
                let flat = r
                    .iter()
                    .flat_map(|row| row.iter().map(|v| v.to_f64().unwrap()))
                    .collect();
                (flat, kk)
            });
            // A head that named its own centres has already answered the "where are the centres"
            // question; deriving them again from the labels would average the clusters and hand
            // `predict` a partition the fit never produced.
            let rule = fit.centers.map(|c| {
                let labels = c.iter().map(|&(l, _)| l as i64).collect();
                let rows = c
                    .iter()
                    .flat_map(|(_, row)| row.iter().map(|v| v.to_f64().unwrap_or(0.0)))
                    .collect();
                match method {
                    // The fuzzy head labels by the same argmin as any centroid head, but its soft
                    // output is a membership rather than a posterior, so the rule carries the
                    // fuzzifier that turns a distance into one.
                    Method::FuzzyCMeans { fuzzifier } => PointRule::Fuzzy {
                        labels,
                        rows,
                        m: fuzzifier,
                    },
                    // The hyperbolic head's centres are points of `H^d` and its assignment is a
                    // Minkowski argmax, so a Euclidean nearest-centre rule would answer from a
                    // different Voronoi diagram than the one the fit produced.
                    Method::Hyperbolic => PointRule::Lorentz { labels, rows },
                    _ => PointRule::Centers { labels, rows },
                }
            });
            Dispatch {
                labels: fit.labels.into_iter().map(|l| l as i64).collect(),
                proba,
                mixture: fit.mixture,
                rule,
            }
        }
        Kind::Hdbscan {
            min_samples,
            min_cluster_size,
            graph_degree,
        } => Dispatch::hard(
            hdbscan_with(feats, min_samples, min_cluster_size, graph_degree, seed).labels,
        ),
        Kind::DcDist {
            objective,
            min_samples,
            graph_degree,
        } => Dispatch::hard(
            dc_clustering(feats, k, objective, min_samples, graph_degree, seed)
                .labels
                .into_iter()
                .map(|l| l as i64)
                .collect(),
        ),
        Kind::ScaleSpace => Dispatch::hard(
            scale_space(feats, 0, max_iter)
                .labels
                .into_iter()
                .map(|l| l as i64)
                .collect(),
        ),
    }
}

/// What one head dispatch produced, in this layer's own types.
struct Dispatch {
    labels: Vec<i64>,
    /// Flattened `n_leaves × k` responsibilities and `k`, for the probabilistic heads.
    proba: Option<(Vec<f64>, usize)>,
    mixture: Option<Mixture>,
    /// The head's own point rule, for a head whose centres are not the cluster means. `None` leaves
    /// the rule to be derived from the labels, which is what every other head wants.
    rule: Option<PointRule>,
}

impl Dispatch {
    fn hard(labels: Vec<i64>) -> Self {
        Self {
            labels,
            proba: None,
            mixture: None,
            rule: None,
        }
    }
}

/// What one Phase-3 labelling pass produced.
struct Labelling {
    labels: Vec<i64>,
    /// Flattened `n_leaves × k` responsibilities and `k`, for the probabilistic heads.
    proba: Option<(Vec<f64>, usize)>,
    /// The shared NMF parts (`r×d`) and the relative reconstruction error, when a projection ran.
    parts: Option<(Vec<Vec<f64>>, f64)>,
    /// The head's point-level density, for the generative heads.
    mixture: Option<Mixture>,
    /// The point rule the head itself defines, when it defines one: a *linear* projection's rule in
    /// code space (built here because this is the only place that holds the coded leaves), or a
    /// head's own centres where those are not the cluster means. `None` leaves the rule to be derived
    /// from the tree's own leaf statistics, which is what every other head wants.
    rule: Option<PointRule>,
}

/// Label leaf features, optionally projecting them to `nmf_dim`-dimensional CF-weighted NMF codes
/// first (for nonnegative data). The head then clusters the codes; labels stay per-leaf so `predict`
/// (point → leaf → label) is unchanged.
fn label_features_proba<R: Real, C: ClusterFeature<R>>(
    feats: &[C],
    kind: Kind,
    k: usize,
    max_iter: usize,
    seed: u64,
    auto_k_max: usize,
    nmf_dim: Option<ProjectionSpec>,
) -> Labelling {
    match nmf_dim {
        Some(spec) => {
            let p = crate::clustering::nmf::project(feats, spec, seed);
            // The head clustered *codes*, so its density lives in code space. A linear projection can
            // carry a raw row there and keep the head's own point rule; an NMF cannot, and falls back
            // to the microcluster route.
            let d = dispatch_kind(&p.coded, kind, k, max_iter, seed, auto_k_max);
            let (labels, proba) = (d.labels, d.proba);
            let rule = projected_rule(&p, &labels, kind, d.mixture, d.rule);
            let parts = p
                .components
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&v| v.to_f64().unwrap_or(f64::NAN))
                        .collect()
                })
                .collect();
            Labelling {
                labels,
                proba,
                parts: Some((parts, p.reconstruction_err.to_f64().unwrap_or(f64::NAN))),
                mixture: None,
                rule,
            }
        }
        None => {
            let d = dispatch_kind(feats, kind, k, max_iter, seed, auto_k_max);
            Labelling {
                labels: d.labels,
                proba: d.proba,
                parts: None,
                mixture: d.mixture,
                rule: d.rule,
            }
        }
    }
}

/// k-means++ restarts for the constrained head (mirrors the unconstrained `kmeans` default).
const COP_N_INIT: usize = 4;

/// Run COP-KMeans over leaf features (already-translated leaf-index constraints) → `i64` labels.
fn label_features_constrained<R: Real, C: ClusterFeature<R>>(
    feats: &[C],
    k: usize,
    must: &[(usize, usize)],
    cannot: &[(usize, usize)],
    max_iter: usize,
    seed: u64,
) -> Result<Vec<i64>, ConstraintError> {
    cop_kmeans(feats, k, must, cannot, max_iter, COP_N_INIT, seed)
        .map(|v| v.into_iter().map(|c| c as i64).collect())
}

/// A human-readable, actionable message for each constrained-clustering failure mode.
fn constraint_msg(e: ConstraintError) -> String {
    match e {
        ConstraintError::Contradiction => {
            "constraints are contradictory: a must-linked group is also cannot-linked".to_string()
        }
        ConstraintError::Infeasible => "constraints are infeasible at this n_clusters: increase \
             n_clusters or relax cannot-links"
            .to_string(),
    }
}

/// `(centers_flat, a, b, dim)` returned by the introspection helpers as `f64`, regardless of the
/// tree's element type (`a`/`b` are weights+radii for leaves, radii+weights for clusters).
type F64Stats = (Vec<f64>, Vec<f64>, Vec<f64>, usize);

/// Number of cluster rows to materialise for the macro accessors: `max(label) + 1` over non-noise
/// labels, so `cluster_centers_[label]` is addressable by the value `predict` returns (an empty
/// component, if any, yields a zero row rather than shifting indices).
fn cluster_count_for_centers(labels: &[i64]) -> usize {
    labels
        .iter()
        .filter(|&&l| l >= 0)
        .max()
        .map_or(0, |&m| m as usize + 1)
}

/// The point-level model a finalized head left behind — [`Rule`] made concrete, with the parameters
/// attached. `None` on the estimator means the head has no such model (or a projection replaced the
/// space the rows live in), and the microcluster route is the defined labelling.
#[derive(serde::Serialize, serde::Deserialize)]
enum PointRule {
    /// `(label, centre)` per non-empty cluster; `rows` is flat `labels.len() × dim`.
    Centers { labels: Vec<i64>, rows: Vec<f64> },
    /// The same centres, plus the fuzzifier `m` that turns a distance into a membership. The label
    /// is still the nearest centre — `u_j ∝ d_j^{−1/(m−1)}` is decreasing in `d_j`, so the argmax
    /// and the argmin agree — but the soft output is a partition of unity, not a posterior. Separate
    /// from [`PointRule::Centers`] because `m` is meaningless for every other centroid head, and an
    /// `Option<f64>` they all have to ignore is a worse way to say that.
    Fuzzy {
        labels: Vec<i64>,
        rows: Vec<f64>,
        m: f64,
    },
    /// The fitted mixture; the label is its maximum-posterior component.
    Posterior(Mixture),
    /// Encode the row through a linear projection — `(x − centre)·basisᵀ` — then apply `inner` in
    /// code space. Only `projection="svd"` reaches this: a PCA *is* a matrix, so a raw row costs
    /// `O(d·r)` to place in the space the head clustered. NMF codes are the solution of a per-row
    /// nonnegative least squares, so that path keeps the microcluster route — which on 20-newsgroups
    /// costs 0.062 ARI purely by answering with each row's leaf's label instead of the row's own.
    Projected {
        centre: Vec<f64>,
        basis: Vec<Vec<f64>>,
        inner: Box<PointRule>,
    },
    /// `(label, centre)` per non-empty cluster, on the Lorentz sheet. The label is `argmax_c ⟨x,c⟩_L`
    /// after the row is put on the sheet — a Voronoi diagram of `H^d`, which the Euclidean argmin over
    /// the same rows does not reproduce.
    Lorentz { labels: Vec<i64>, rows: Vec<f64> },
}

/// Fuzzy c-means memberships of one raw row against `k` centres laid out flat as `k × dim`.
///
/// `u_j = 1 / Σ_r (d_j / d_r)^{1/(m−1)}` with `d = ‖x − c‖²`, written as a ratio to the smallest
/// distance so a row sitting on a centre does not raise `1/d` to a large power and overflow. A row
/// exactly on one or more centres splits its membership over exactly those, which is the constrained
/// minimum and the only finite answer — the same singleton rule the leaf-level head uses.
fn fuzzy_memberships<R: Real>(x: &[R], rows: &[f64], dim: usize, m: f64, u: &mut [f64]) {
    let mut dmin = f64::INFINITY;
    for (c, v) in u.iter_mut().enumerate() {
        *v = rows[c * dim..(c + 1) * dim]
            .iter()
            .enumerate()
            .map(|(j, &mu)| {
                let t = x[j].to_f64().unwrap_or(0.0) - mu;
                t * t
            })
            .sum();
        dmin = dmin.min(*v);
    }
    if dmin <= 0.0 {
        let hits = u.iter().filter(|&&d| d <= 0.0).count() as f64;
        for v in u.iter_mut() {
            *v = if *v <= 0.0 { 1.0 / hits } else { 0.0 };
        }
        return;
    }
    let p = 1.0 / (m - 1.0);
    let mut sum = 0.0;
    for v in u.iter_mut() {
        *v = (dmin / *v).powf(p);
        sum += *v;
    }
    for v in u.iter_mut() {
        *v /= sum;
    }
}

/// `(x − centre)·basisᵀ` — one row through a linear projection, in `f64` whatever the tree's dtype
/// (the basis and the code-space model are `f64`, so encoding in `f32` would place the row in a
/// slightly different space from the one the head clustered).
fn encode_row<R: Real>(centre: &[f64], basis: &[Vec<f64>], x: &[R]) -> Vec<f64> {
    basis
        .iter()
        .map(|v| {
            x.iter()
                .zip(centre)
                .zip(v)
                .map(|((&xi, &c), &vi)| (xi.to_f64().unwrap_or(0.0) - c) * vi)
                .sum()
        })
        .collect()
}

/// A [`PointRule::Centers`] from pooled per-cluster statistics: one row per non-empty cluster,
/// re-normalized to the sphere for the heads whose argmin is a cosine argmax. Empty clusters are
/// dropped rather than emitted at the origin, where they would attract every point near it.
fn centers_rule(centers: &[f64], weights: &[f64], dim: usize, unit: bool) -> Option<PointRule> {
    let mut labels = Vec::new();
    let mut rows = Vec::new();
    for (c, &w) in weights.iter().enumerate() {
        if w <= 0.0 {
            continue;
        }
        let row = &centers[c * dim..(c + 1) * dim];
        let scale = if unit {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 { 1.0 / norm } else { 1.0 }
        } else {
            1.0
        };
        labels.push(c as i64);
        rows.extend(row.iter().map(|v| v * scale));
    }
    (!labels.is_empty()).then_some(PointRule::Centers { labels, rows })
}

/// The head's point rule *in code space*, wrapped in the encoder that takes a raw row there.
///
/// `None` unless the projection is a linear map (only `svd` is) and the head has a point model —
/// otherwise the microcluster route stays the defined labelling, as it is for every NMF projection.
fn projected_rule<R: Real>(
    proj: &Projection<R>,
    labels: &[i64],
    kind: Kind,
    mixture: Option<Mixture>,
    head_rule: Option<PointRule>,
) -> Option<PointRule> {
    let centre = proj.centre.as_ref()?;
    let Kind::Parametric(method) = kind else {
        return None;
    };
    // The head's own centres are already in code space — the projection is what it clustered.
    let inner = match (head_rule, assignment_rule(method)) {
        (Some(rule), _) => rule,
        (None, Rule::Posterior) => PointRule::Posterior(mixture?),
        // A projection replaces the space the rows live in, and a code vector has no time-like
        // coordinate — the sheet is a property of the input space, not of the codes.
        (None, Rule::Lorentz) => return None,
        (None, Rule::Microcluster) => return None,
        (None, Rule::Centroid { unit }) => {
            let k = cluster_count_for_centers(labels);
            let (centers, _radii, weights, dim) = compute_cluster_stats(&proj.coded, labels, k);
            centers_rule(&centers, &weights, dim, unit)?
        }
    };
    Some(PointRule::Projected {
        centre: centre.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect(),
        basis: proj
            .components
            .iter()
            .map(|row| row.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect())
            .collect(),
        inner: Box::new(inner),
    })
}

impl PointRule {
    /// Label one dense row. `scratch` is a reusable buffer for the posterior path.
    fn label_of<R: Real>(&self, x: &[R], scratch: &mut Vec<f64>) -> i64 {
        match self {
            PointRule::Centers { labels, rows } | PointRule::Fuzzy { labels, rows, .. } => {
                let dim = x.len();
                let mut best = labels[0];
                let mut bd = f64::INFINITY;
                for (c, &id) in labels.iter().enumerate() {
                    let mut d = 0.0;
                    for (j, &m) in rows[c * dim..(c + 1) * dim].iter().enumerate() {
                        let t = x[j].to_f64().unwrap_or(0.0) - m;
                        d += t * t;
                    }
                    if d < bd {
                        bd = d;
                        best = id;
                    }
                }
                best
            }
            PointRule::Lorentz { labels, rows } => {
                let dim = x.len();
                let point: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
                let p = project_to_sheet(&point);
                let mut best = labels[0];
                let mut bv = f64::NEG_INFINITY;
                for (c, &id) in labels.iter().enumerate() {
                    let v = lorentz_dot(&p, &rows[c * dim..(c + 1) * dim]);
                    if v > bv {
                        bv = v;
                        best = id;
                    }
                }
                best
            }
            PointRule::Posterior(m) => m.assign_into(x, scratch) as i64,
            PointRule::Projected {
                centre,
                basis,
                inner,
            } => inner.label_of(&encode_row(centre, basis, x), scratch),
        }
    }

    /// Label each of `n` dense rows — the partition the head *is*, rather than a tree descent.
    fn label_rows<R: Real>(&self, flat: &[R], n: usize, dim: usize) -> Vec<i64> {
        map_rows(n, |i| {
            let mut scratch = Vec::new();
            self.label_of(&flat[i * dim..(i + 1) * dim], &mut scratch)
        })
    }

    /// [`PointRule::label_rows`] for CSR input: each row is expanded into a reused dense buffer, so
    /// the dense `n × dim` matrix is never materialized (serial — the shared buffer precludes the
    /// parallel path).
    fn label_csr(&self, data: &[f64], indices: &[i64], indptr: &[i64], dim: usize) -> Vec<i64> {
        if let PointRule::Projected {
            centre,
            basis,
            inner,
        } = self
        {
            // `(x − x̄)Vᵀ = xVᵀ − x̄Vᵀ`: the second term is one constant vector, and the first touches
            // only the non-zeros. That keeps the projected sparse path at `O(nnz·r)` instead of the
            // `O(n·d·r)` a densify-then-encode would cost — 60× fewer multiplies on 20-newsgroups.
            let offset: Vec<f64> = basis
                .iter()
                .map(|v| -v.iter().zip(centre).map(|(&a, &b)| a * b).sum::<f64>())
                .collect();
            let mut scratch = Vec::new();
            let mut out = Vec::with_capacity(indptr.len().saturating_sub(1));
            for w in indptr.windows(2) {
                let (lo, hi) = (w[0] as usize, w[1] as usize);
                let mut code = offset.clone();
                for k in lo..hi {
                    let (c, v) = (indices[k] as usize, data[k]);
                    for (z, b) in code.iter_mut().zip(basis) {
                        *z += v * b[c];
                    }
                }
                out.push(inner.label_of(&code, &mut scratch));
            }
            return out;
        }
        let mut buf = vec![0.0f64; dim];
        let mut scratch = Vec::new();
        let mut out = Vec::with_capacity(indptr.len().saturating_sub(1));
        for w in indptr.windows(2) {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            for k in lo..hi {
                buf[indices[k] as usize] = data[k];
            }
            out.push(self.label_of(&buf, &mut scratch));
            for k in lo..hi {
                buf[indices[k] as usize] = 0.0;
            }
        }
        out
    }

    /// Per-row posterior `p(c | x)` flattened `n × k`, or `None` for a centroid rule, which has no
    /// calibrated posterior to report.
    fn proba_rows<R: Real>(&self, flat: &[R], n: usize, dim: usize) -> Option<(Vec<f64>, usize)> {
        if let PointRule::Projected {
            centre,
            basis,
            inner,
        } = self
        {
            // Encode once into a flat code matrix, then let the code-space rule score it — so
            // `predict_proba(X).argmax(1) == predict(X)` holds on this path too.
            let r = basis.len();
            let mut codes = Vec::with_capacity(n * r);
            for i in 0..n {
                codes.extend(encode_row(centre, basis, &flat[i * dim..(i + 1) * dim]));
            }
            return inner.proba_rows(&codes, n, r);
        }
        if let PointRule::Fuzzy { labels, rows, m } = self {
            let k = labels.len();
            let mut out = Vec::with_capacity(n * k);
            let mut u = vec![0.0; k];
            for i in 0..n {
                let x = &flat[i * dim..(i + 1) * dim];
                fuzzy_memberships(x, rows, dim, *m, &mut u);
                out.extend_from_slice(&u);
            }
            return Some((out, k));
        }
        let PointRule::Posterior(m) = self else {
            return None;
        };
        let k = m.n_components();
        let mut out = Vec::with_capacity(n * k);
        let mut scratch = Vec::with_capacity(k);
        for i in 0..n {
            m.responsibilities(&flat[i * dim..(i + 1) * dim], &mut scratch);
            out.extend_from_slice(&scratch);
        }
        Some((out, k))
    }
}

/// Per-leaf (microcluster) statistics as `f64`, regardless of the tree's element type: flat
/// row-major `centers` (`n_leaves × dim`), `weights` (effective point mass), and `radii` — the RMS
/// distance from the centroid, `sqrt(ssd / weight)`.
fn compute_leaf_stats<R: Real, C: ClusterFeature<R>>(feats: &[C]) -> F64Stats {
    let dim = feats.first().map_or(0, |c| c.dim());
    let mut centers = Vec::with_capacity(feats.len() * dim);
    let mut weights = Vec::with_capacity(feats.len());
    let mut radii = Vec::with_capacity(feats.len());
    for c in feats {
        for &m in c.mean() {
            centers.push(m.to_f64().unwrap());
        }
        let w = c.weight().to_f64().unwrap();
        let ssd = c.ssd().to_f64().unwrap();
        radii.push(if w > 0.0 { (ssd / w).sqrt() } else { 0.0 });
        weights.push(w);
    }
    (centers, weights, radii, dim)
}

/// Pooled per-cluster statistics over labelled leaves (`k` rows): mass-weighted `centers`, RMS
/// `radii`, and total `weights`. Noise leaves (`label < 0`, HDBSCAN) are skipped. The radius pools
/// within-leaf scatter and the leaf's displacement from the cluster centroid (König–Huygens), so it
/// is the exact RMS spread of the cluster's points around its centroid.
fn compute_cluster_stats<R: Real, C: ClusterFeature<R>>(
    feats: &[C],
    labels: &[i64],
    k: usize,
) -> F64Stats {
    let dim = feats.first().map_or(0, |c| c.dim());
    let mut weights = vec![0.0f64; k];
    let mut csum = vec![0.0f64; k * dim];
    let mut within = vec![0.0f64; k];
    for (li, c) in feats.iter().enumerate() {
        let lab = labels[li];
        if lab < 0 {
            continue;
        }
        let cl = lab as usize;
        let w = c.weight().to_f64().unwrap();
        weights[cl] += w;
        within[cl] += c.ssd().to_f64().unwrap();
        for (j, &m) in c.mean().iter().enumerate() {
            csum[cl * dim + j] += w * m.to_f64().unwrap();
        }
    }
    let mut centers = vec![0.0f64; k * dim];
    for cl in 0..k {
        if weights[cl] > 0.0 {
            for j in 0..dim {
                centers[cl * dim + j] = csum[cl * dim + j] / weights[cl];
            }
        }
    }
    let mut radii = vec![0.0f64; k];
    for (li, c) in feats.iter().enumerate() {
        let lab = labels[li];
        if lab < 0 {
            continue;
        }
        let cl = lab as usize;
        let w = c.weight().to_f64().unwrap();
        let mut d2 = 0.0;
        for (j, &m) in c.mean().iter().enumerate() {
            let diff = m.to_f64().unwrap() - centers[cl * dim + j];
            d2 += diff * diff;
        }
        radii[cl] += w * d2;
    }
    for cl in 0..k {
        radii[cl] = if weights[cl] > 0.0 {
            ((within[cl] + radii[cl]) / weights[cl]).sqrt()
        } else {
            0.0
        };
    }
    (centers, radii, weights, dim)
}

/// What an outlier score divides the centroid deviation by.
///
/// The two are calibrated against each other rather than merely both being "a z-score": with an
/// isotropic pooled covariance `Σ = (R²/d)·I`, `Whitened` returns exactly what `Radius` does, so the
/// refinement changes an answer only where the cluster is actually anisotropic.
enum OutlierScale<'a> {
    /// One RMS radius per cluster — the trace of the pooled covariance, so a cluster's short axis is
    /// judged by the length of its long one.
    Radius(&'a [f64]),
    /// Lower Cholesky factor of each cluster's pooled covariance; `None` where it has no spread to
    /// factor (a cluster of one point).
    Whitened(&'a [Option<Vec<Vec<f64>>>]),
}

/// Relative ridge on a pooled covariance, shared with the GMM head's `VAR_FLOOR_REL`: no axis is
/// treated as more than `1e6` times tighter than the cluster's mean axis. Without it a cluster whose
/// points are constant along some direction is singular, the Cholesky fails, and — worse, if it does
/// not fail — the score of every row is dominated by a direction carrying no information.
const OUTLIER_VAR_FLOOR_REL: f64 = 1e-6;

/// Cholesky factor of each cluster's pooled covariance, given the leaf `centers` that
/// [`compute_cluster_stats`] produced.
///
/// Parallel-axis theorem over the leaves: `Σ = Σ_l w_l (Σ_l + δ_l δ_lᵀ) / W`, with `δ_l = μ_l − c`.
/// The trace of that is the scalar `cluster_radii_` squared, exactly — so the whitened score is a
/// strict refinement of the scalar one, not a second differently-calibrated number.
///
/// The off-diagonal terms are what make it worth the `O(k d³)`: a cluster can be elongated along a
/// direction that is not a coordinate axis, and a per-dimension variance cannot see that. Only
/// `feature="full"` carries within-leaf cross-covariances, but the between-leaf outer product is
/// full-rank for every feature, so a sheared cluster is anisotropic here whatever the leaf model.
fn compute_cluster_chol<R: Real, C: ClusterFeature<R>>(
    feats: &[C],
    labels: &[i64],
    k: usize,
    centers: &[f64],
) -> Vec<Option<Vec<Vec<f64>>>> {
    let dim = feats.first().map_or(0, |c| c.dim());
    let mut cov = vec![vec![vec![0.0f64; dim]; dim]; k];
    let mut weights = vec![0.0f64; k];
    let mut delta = vec![0.0f64; dim];
    for (li, c) in feats.iter().enumerate() {
        let lab = labels[li];
        if lab < 0 {
            continue;
        }
        let cl = lab as usize;
        let w = c.weight().to_f64().unwrap();
        weights[cl] += w;
        for (j, &m) in c.mean().iter().enumerate() {
            delta[j] = m.to_f64().unwrap() - centers[cl * dim + j];
        }
        let leaf = c.cov_dense();
        let target = &mut cov[cl];
        for i in 0..dim {
            for j in 0..dim {
                target[i][j] += w * (leaf[i][j].to_f64().unwrap() + delta[i] * delta[j]);
            }
        }
    }
    cov.into_iter()
        .zip(weights)
        .map(|(mut m, w)| {
            if w <= 0.0 || dim == 0 {
                return None;
            }
            for row in m.iter_mut() {
                for v in row.iter_mut() {
                    *v /= w;
                }
            }
            let ridge = OUTLIER_VAR_FLOOR_REL * (0..dim).map(|i| m[i][i]).sum::<f64>() / dim as f64;
            if ridge <= 0.0 {
                return None;
            }
            for (i, row) in m.iter_mut().enumerate() {
                row[i] += ridge;
            }
            cholesky_lower(&m)
        })
        .collect()
}

/// The three internal validity indices over the labelled leaves, as
/// `(calinski_harabasz, davies_bouldin, medoid_silhouette)`.
///
/// Noise leaves (`label < 0`, HDBSCAN) are dropped rather than pooled into a cluster of their own:
/// noise is not a cluster, and scoring it as one would make every index a function of how much of
/// the data the head declined to label.
fn compute_validity<R: Real, C: ClusterFeature<R>>(
    feats: &[C],
    labels: &[i64],
    k: usize,
) -> (f64, f64, f64) {
    let mut kept: Vec<C> = Vec::with_capacity(feats.len());
    let mut kept_labels: Vec<usize> = Vec::with_capacity(feats.len());
    for (f, &l) in feats.iter().zip(labels) {
        if l >= 0 {
            kept.push(f.clone());
            kept_labels.push(l as usize);
        }
    }
    (
        crate::validity::calinski_harabasz(&kept, &kept_labels, k),
        crate::validity::davies_bouldin(&kept, &kept_labels, k),
        crate::validity::medoid_silhouette(&kept, &kept_labels, k),
    )
}

/// `(points, weights, offset, reference_cost, total_sensitivity, n_leaves, radii)` as Python
/// sees it.
type CoresetPy<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    f64,
    usize,
    Bound<'py, PyArray1<f64>>,
);

/// What a coreset export carries across the binding: the sample, and the three numbers its
/// guarantee is stated in terms of. A struct rather than a seven-tuple because the three scalars
/// are not interchangeable and a positional mix-up between them would be silent.
struct CoresetOut {
    points: Vec<f64>,
    weights: Vec<f64>,
    radii: Vec<f64>,
    dim: usize,
    offset: f64,
    reference_cost: f64,
    total_sensitivity: f64,
    n_leaves: usize,
}

/// The rows the compute reads: the caller's own numpy buffer where it can be used as it stands, a
/// converted copy where it cannot.
///
/// Everything downstream reads one flat row-major `&[R]`, which is exactly the memory a C-contiguous
/// array of the tree's element type already holds — so a copy is needed only when the array is
/// strided, its dtype is the other float width, or the head prepares rows ([`RowPrep`]) and must not
/// write into the caller's array. Copying unconditionally doubled the peak: measured on
/// `200 000 × 784` `f64`, a 1 196 MB input drove resident memory from 1 237 MB to 2 485 MB.
enum Rows<'py, R: Element> {
    Borrowed(PyReadonlyArray2<'py, R>),
    Owned(Vec<R>),
}

impl<R: Real + Element> Rows<'_, R> {
    fn as_slice(&self) -> &[R] {
        match self {
            // Only ever constructed after `as_slice` succeeded on this array, whose read borrow is
            // held for as long as `self` is: reaching the panic means that invariant broke.
            Self::Borrowed(a) => a
                .as_slice()
                .expect("Rows::Borrowed holds a contiguous array"),
            Self::Owned(v) => v,
        }
    }
}

/// Present a (non-empty) 2-D array as a flat row-major `n × dim` buffer prepared for the head;
/// returns `(rows, n, dim)`. Generic over the element type so `f32` inputs are clustered in `f32`
/// (no `f64` upcast).
fn to_rows<R: Real + Element>(
    data: PyReadonlyArray2<'_, R>,
    prep: RowPrep,
) -> PyResult<(Rows<'_, R>, usize, usize)> {
    let (n, dim) = {
        let arr = data.as_array();
        (arr.shape()[0], arr.shape()[1])
    };
    if n == 0 || dim == 0 {
        return Err(PyValueError::new_err("data must be a non-empty 2-D array"));
    }
    // Validate at the boundary: a NaN/Inf would silently corrupt the tree (mean/scatter become NaN
    // and every downstream label is garbage), so reject it loudly here.
    fn finite<R: Real>(s: &[R]) -> PyResult<()> {
        if s.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "data contains NaN or infinite values",
            ));
        }
        Ok(())
    }
    // `PyReadonlyArray::as_slice` succeeds on any *contiguous* array, and a column-major one is
    // contiguous — its buffer is then the transpose of what a row-major read expects, so borrowing
    // it would hand the tree a scrambled dataset with no error anywhere. Row-major is the layout
    // this whole module indexes by, so the borrow is gated on it explicitly rather than on
    // contiguity; `ndarray`'s standard layout is exactly that condition.
    if prep == RowPrep::None && data.as_array().is_standard_layout() {
        if let Ok(s) = data.as_slice() {
            finite(s)?;
            return Ok((Rows::Borrowed(data), n, dim));
        }
    }
    let arr = data.as_array();
    // A C-contiguous array exposes its backing slice, so the copy is a single memcpy and the
    // finiteness check one tight auto-vectorized pass — strided `arr[[i, j]]` indexing is neither.
    let mut flat: Vec<R> = match arr.as_slice() {
        Some(s) => s.to_vec(),
        None => arr.iter().copied().collect(), // non-contiguous (e.g. a transposed view)
    };
    finite(&flat)?;
    prepare_rows(&mut flat, n, dim, prep);
    Ok((Rows::Owned(flat), n, dim))
}

/// Map each row index `0..n` to one value, in parallel above a size threshold (with the `parallel`
/// feature). Results are collected in index order, so the output is identical to the serial path:
/// the per-row work is read-only and there is no floating-point reduction, so labels never change.
fn map_rows<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        const PAR_MIN: usize = 4096;
        if n >= PAR_MIN {
            return (0..n).into_par_iter().map(f).collect();
        }
    }
    (0..n).map(f).collect()
}

/// Which manifold a row is put on before it reaches the tree.
///
/// Two mutually exclusive preparations, so one value rather than two booleans of which only one may
/// ever be set. Both are idempotent, which is what lets `predict` re-apply the same preparation to a
/// row the fit never saw.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RowPrep {
    /// The row enters as it is.
    None,
    /// Onto `S^{d-1}` — `normalize=True`, and every directional head regardless of the flag.
    Unit,
    /// Onto the Lorentz sheet, by recomputing the time-like coordinate — `method="hyperbolic"`.
    Sheet,
}

/// How much of the dataset one `stream` call is handing over.
///
/// The distinction exists for `canonical_order`, which is only meaningful over the whole dataset:
/// sorting a chunk canonically would produce an order that is canonical for the wrong set, and
/// leave the result as dependent on the chunk boundaries as it ever was on the row order. A named
/// pair rather than a bare `bool`, because `stream(data, true)` at the call site says nothing about
/// what is true.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arrival {
    /// Every row at once — `fit`, `fit_predict`.
    Whole,
    /// One chunk of an open-ended stream — `partial_fit`.
    Chunk,
}

/// Apply a [`RowPrep`] to a flat `n × dim` buffer in place.
fn prepare_rows<R: Real>(flat: &mut [R], n: usize, dim: usize, prep: RowPrep) {
    match prep {
        RowPrep::None => {}
        RowPrep::Unit => normalize_rows(flat, n, dim),
        RowPrep::Sheet => {
            for i in 0..n {
                let row = &mut flat[i * dim..(i + 1) * dim];
                let sq = row[1..].iter().fold(R::zero(), |a, &v| a + v * v);
                row[0] = (R::one() + sq).sqrt();
            }
        }
    }
}

/// L2-normalise each row in place (zero rows are left unchanged). With `normalize=True` this maps
/// embeddings onto the unit sphere so direction (cosine) structure is what the tree clusters; on the
/// sphere squared-Euclidean and cosine distance are monotonically equivalent (`d² = 2 − 2·cosθ`),
/// so the existing Euclidean CF-tree clusters by angle without a separate cosine code path.
fn normalize_rows<R: Real>(flat: &mut [R], n: usize, dim: usize) {
    for i in 0..n {
        let row = &mut flat[i * dim..(i + 1) * dim];
        let mut s = R::zero();
        for &v in row.iter() {
            s = s + v * v;
        }
        let norm = s.sqrt();
        if norm > R::zero() {
            for v in row.iter_mut() {
                *v = *v / norm;
            }
        }
    }
}

/// Absorption criterion chosen at runtime, so the binding keeps a single tree type instead of one
/// per (feature × absorber) combination (the routing distance is the separate [`RouteKind`]).
///
/// The BIRCH grid D0–D4 and R, plus this crate's mass-invariant χ² gate. Every variant except
/// `Chi2` reads `threshold` in its own units — D1 is an L1 distance, the rest are squared — so the
/// same number means different things across them and a threshold tuned for one does not transfer.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum AbsorbKind<R> {
    Euclidean,
    Manhattan,
    Average,
    Diameter,
    Ward,
    Radius,
    Chi2(MahalanobisChi2<R>),
    Subspace(SubspaceChi2<R>),
}

impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for AbsorbKind<R> {
    fn point(&self, cf: &C, x: &[R]) -> R {
        match self {
            AbsorbKind::Euclidean => CentroidEuclidean.point(cf, x),
            AbsorbKind::Manhattan => CentroidManhattan.point(cf, x),
            AbsorbKind::Average => AverageIntercluster.point(cf, x),
            AbsorbKind::Diameter => AverageIntracluster.point(cf, x),
            AbsorbKind::Ward => VarianceIncrease.point(cf, x),
            AbsorbKind::Radius => Radius.point(cf, x),
            AbsorbKind::Chi2(m) => m.point(cf, x),
            AbsorbKind::Subspace(m) => m.point(cf, x),
        }
    }
    fn between(&self, a: &C, b: &C) -> R {
        match self {
            AbsorbKind::Euclidean => CentroidEuclidean.between(a, b),
            AbsorbKind::Manhattan => CentroidManhattan.between(a, b),
            AbsorbKind::Average => AverageIntercluster.between(a, b),
            AbsorbKind::Diameter => AverageIntracluster.between(a, b),
            AbsorbKind::Ward => VarianceIncrease.between(a, b),
            AbsorbKind::Radius => Radius.between(a, b),
            AbsorbKind::Chi2(m) => m.between(a, b),
            AbsorbKind::Subspace(m) => m.between(a, b),
        }
    }
}

/// Routing / inter-cluster distance chosen at runtime (point → leaf, and tree navigation).
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum RouteKind {
    Euclidean,
    Manhattan,
    Ward,
    Average,
}

impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for RouteKind {
    fn point(&self, cf: &C, x: &[R]) -> R {
        match self {
            RouteKind::Euclidean => CentroidEuclidean.point(cf, x),
            RouteKind::Manhattan => CentroidManhattan.point(cf, x),
            RouteKind::Ward => VarianceIncrease.point(cf, x),
            RouteKind::Average => AverageIntercluster.point(cf, x),
        }
    }
    fn between(&self, a: &C, b: &C) -> R {
        match self {
            RouteKind::Euclidean => CentroidEuclidean.between(a, b),
            RouteKind::Manhattan => CentroidManhattan.between(a, b),
            RouteKind::Ward => VarianceIncrease.between(a, b),
            RouteKind::Average => AverageIntercluster.between(a, b),
        }
    }
}

/// Map the `distance` keyword to a routing measure. (`radius` is an absorption-only criterion, not
/// a routing distance, so it is intentionally not offered here.)
fn parse_route(distance: &str) -> PyResult<RouteKind> {
    match distance {
        "euclidean" => Ok(RouteKind::Euclidean),
        "manhattan" => Ok(RouteKind::Manhattan),
        "ward" => Ok(RouteKind::Ward),
        "average" => Ok(RouteKind::Average),
        _ => Err(PyValueError::new_err(
            "distance must be 'euclidean', 'manhattan', 'ward' or 'average'",
        )),
    }
}

// ── one-shot function ─────────────────────────────────────────────────────────────────────────

/// Build the CF-tree — sequentially (default) or via parallel shard+merge when `n_jobs > 1` and the
/// `parallel` feature is on. The sequential path is the byte-identical default.
#[allow(clippy::too_many_arguments)]
fn build_tree<R: Real, C: ClusterFeature<R>>(
    dim: usize,
    branching: usize,
    leaf_cap: usize,
    threshold: R,
    max_leaves: usize,
    route: RouteKind,
    absorb: AbsorbKind<R>,
    flat: &[R],
    n: usize,
    n_jobs: usize,
    balance: Option<R>,
    canonical_order: bool,
) -> CFTree<R, C, RouteKind, AbsorbKind<R>> {
    let order = canonical_order.then(|| canonical_permutation(flat, n, dim));
    // Under `canonical_order` the shard count comes from the data, not from `n_jobs`: the shards are
    // the partition, so taking them from a thread count would leave the guarantee holding only until
    // somebody re-tuned `n_jobs`.
    let shards = match order {
        Some(_) => canonical_shards(n),
        None => n_jobs,
    };
    if shards > 1 {
        return CFTree::build_sharded(
            dim,
            branching,
            leaf_cap,
            threshold,
            max_leaves,
            route,
            absorb,
            flat,
            n,
            shards,
            balance,
            order.as_deref(),
        );
    }
    let mut tree = CFTree::new(
        dim, branching, leaf_cap, threshold, max_leaves, route, absorb,
    );
    tree.set_balance(balance);
    for rank in 0..n {
        let i = order.as_ref().map_or(rank, |o| o[rank] as usize);
        tree.insert(&flat[i * dim..(i + 1) * dim]);
    }
    tree
}

#[allow(clippy::too_many_arguments)]
fn cluster<R: Real, C: ClusterFeature<R>>(
    flat: &[R],
    n: usize,
    dim: usize,
    k: usize,
    kind: Kind,
    route: RouteKind,
    absorb: AbsorbKind<R>,
    threshold: R,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    max_iter: usize,
    seed: u64,
    auto_k_max: usize,
    n_jobs: usize,
    nmf_dim: Option<ProjectionSpec>,
    refine: usize,
    balance: Option<R>,
    leaf_refit: usize,
    canonical_order: bool,
) -> (Vec<i64>, usize) {
    let mut tree = build_tree::<R, C>(
        dim,
        branching,
        leaf_cap,
        threshold,
        max_leaves,
        route,
        absorb,
        flat,
        n,
        n_jobs,
        balance,
        canonical_order,
    );
    for _ in 0..leaf_refit {
        tree.refit_leaves(flat, n);
    }
    let leaves = tree.num_leaves();
    let labels = match kind {
        Kind::Parametric(method) => match nmf_dim {
            Some(_) => {
                // Reduce leaf centroids to codes and cluster those. A linear projection (`svd`) hands
                // back a point rule, so each row is labelled by its own code; an NMF cannot, and the
                // row inherits its leaf's label.
                let out = label_features_proba(
                    tree.leaf_features(),
                    kind,
                    k,
                    max_iter,
                    seed,
                    auto_k_max,
                    nmf_dim,
                );
                match out.rule {
                    Some(rule) => rule.label_rows(flat, n, dim),
                    None => map_rows(n, |i| {
                        out.labels[tree.nearest_entry(&flat[i * dim..(i + 1) * dim])]
                    }),
                }
            }
            None => {
                let mut model = Model::fit(tree, k, method, max_iter, seed, auto_k_max);
                model.refine(flat, n, dim, refine);
                map_rows(n, |i| model.predict(&flat[i * dim..(i + 1) * dim]) as i64)
            }
        },
        Kind::Hdbscan {
            min_samples,
            min_cluster_size,
            graph_degree,
        } => {
            let res = hdbscan_with(
                tree.leaf_features(),
                min_samples,
                min_cluster_size,
                graph_degree,
                seed,
            );
            map_rows(n, |i| {
                res.labels[tree.nearest_entry(&flat[i * dim..(i + 1) * dim])]
            })
        }
        Kind::DcDist {
            objective,
            min_samples,
            graph_degree,
        } => {
            let res = dc_clustering(
                tree.leaf_features(),
                k,
                objective,
                min_samples,
                graph_degree,
                seed,
            );
            map_rows(n, |i| {
                res.labels[tree.nearest_entry(&flat[i * dim..(i + 1) * dim])] as i64
            })
        }
        Kind::ScaleSpace => {
            let res = scale_space(tree.leaf_features(), 0, max_iter);
            map_rows(n, |i| {
                res.labels[tree.nearest_entry(&flat[i * dim..(i + 1) * dim])] as i64
            })
        }
    };
    (labels, leaves)
}

/// Build the tree and label every row for a single element type `R`, with the GIL released during
/// compute. `threshold` arrives as `f64` and is narrowed to `R`.
#[allow(clippy::too_many_arguments)]
fn run_oneshot<R: Real + Element>(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, R>,
    n_clusters: usize,
    feature: &str,
    kind: Kind,
    distance: &str,
    absorb: &str,
    chi2_p: f64,
    chi2_scale: f64,
    threshold: f64,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    max_iter: usize,
    seed: u64,
    n_jobs: usize,
    normalize: bool,
    nmf_dim: Option<ProjectionSpec>,
    refine: usize,
    balance: Option<f64>,
    auto_k_max: usize,
    leaf_refit: usize,
    canonical_order: bool,
) -> PyResult<(Vec<i64>, usize)> {
    // Directional heads cluster points on the unit sphere and the hyperbolic head on the Lorentz
    // sheet, so they always operate on prepared rows regardless of the caller's `normalize` flag.
    let prep = match kind {
        Kind::Parametric(Method::Hyperbolic) => RowPrep::Sheet,
        Kind::Parametric(Method::SphericalKMeans | Method::Movmf | Method::Watson) => RowPrep::Unit,
        _ if normalize => RowPrep::Unit,
        _ => RowPrep::None,
    };
    let (rows, n, dim) = to_rows(data, prep)?;
    let flat = rows.as_slice();
    if matches!(nmf_dim.map(|s| s.kind), Some(ProjectionKind::Nmf { .. })) {
        require_nonnegative(flat)?;
    }
    let route = parse_route(distance)?;
    let balance = balance.and_then(R::from_f64);
    py.detach(|| {
        let (gate, thr) = resolve_gate::<R>(absorb, dim, chi2_p, chi2_scale, threshold)?;
        match feature {
            "spherical" => Ok(cluster::<R, Spherical<R>>(
                flat,
                n,
                dim,
                n_clusters,
                kind,
                route,
                gate,
                thr,
                branching,
                leaf_cap,
                max_leaves,
                max_iter,
                seed,
                auto_k_max,
                n_jobs,
                nmf_dim,
                refine,
                balance,
                leaf_refit,
                canonical_order,
            )),
            "diagonal" => Ok(cluster::<R, Diagonal<R>>(
                flat,
                n,
                dim,
                n_clusters,
                kind,
                route,
                gate,
                thr,
                branching,
                leaf_cap,
                max_leaves,
                max_iter,
                seed,
                auto_k_max,
                n_jobs,
                nmf_dim,
                refine,
                balance,
                leaf_refit,
                canonical_order,
            )),
            "full" => Ok(cluster::<R, Full<R>>(
                flat,
                n,
                dim,
                n_clusters,
                kind,
                route,
                gate,
                thr,
                branching,
                leaf_cap,
                max_leaves,
                max_iter,
                seed,
                auto_k_max,
                n_jobs,
                nmf_dim,
                refine,
                balance,
                leaf_refit,
                canonical_order,
            )),
            "fd" => Ok(cluster::<R, FdSketch<R>>(
                flat,
                n,
                dim,
                n_clusters,
                kind,
                route,
                gate,
                thr,
                branching,
                leaf_cap,
                max_leaves,
                max_iter,
                seed,
                auto_k_max,
                n_jobs,
                nmf_dim,
                refine,
                balance,
                leaf_refit,
                canonical_order,
            )),
            _ => Err("feature must be 'spherical', 'diagonal', 'full' or 'fd'"),
        }
    })
    .map_err(PyValueError::new_err)
}

/// Cluster the rows of a 2-D float32 or float64 array; returns one int64 label per row (`-1` =
/// noise, produced only by `method="hdbscan"`). `float32` input is clustered in `f32` (half the
/// memory, no upcast). With `n_clusters=0` and `method="gmm"`/`"gmm-full"` the component count is
/// selected automatically by BIC.
///
/// `absorb` selects the CF-tree's absorption criterion — the full BIRCH grid plus this crate's own
/// gate: `"euclidean"` (D0, the default), `"manhattan"` (D1), `"average"` (D2, inter-cluster),
/// `"diameter"` (D3, intra-cluster of the merged cell), `"ward"` (D4, variance increase),
/// `"radius"` (R, the merged cell's mean squared radius), and `"chi2"`, a mass-invariant
/// Mahalanobis-χ² gate at level `chi2_p` with within-cluster variance `chi2_scale` (required for
/// `chi2`). `threshold` is read in the chosen criterion's own units — L1 for `"manhattan"`, squared
/// for the rest, a χ²_dim quantile for `"chi2"` — so it does not transfer between them.
///
/// `refine` runs BIRCH's Phase 4 — that many Lloyd sweeps over the raw rows, warm-started from the
/// Phase-3 centres — for the centroid heads (`"kmeans"`, `"spherical-kmeans"`); other heads ignore
/// it, having no centre model to sweep. It trades a second pass over the data for a lower
/// within-cluster sum of squares, which is not the same thing as a better partition: on `covtype`
/// scikit-learn's k-means already reaches the lower objective and the worse ARI. Default `0` (off).
#[pyfunction]
#[pyo3(signature = (
    data, n_clusters = 8, feature = "diagonal", method = "gmm", threshold = 0.0,
    branching = 32, leaf_cap = 32, max_leaves = 2000, max_iter = 100,
    min_samples = 5, min_cluster_size = 5, seed = 0, distance = "euclidean",
    absorb = "euclidean", chi2_p = 0.95, chi2_scale = 0.0, n_jobs = 1, normalize = false,
    resolution = 1.0, covariance_weight = 0.0, tangent_weight = 0.0, tangent_rank = 2,
    projection = "none", projection_dim = 64, projection_max_iter = 100, refine = 0, rank = 2,
    graph_degree = 0, balance = None, auto_k_max = 0, fuzzifier = 2.0, leaf_refit = 0,
    canonical_order = false
))]
#[allow(clippy::too_many_arguments)]
fn fit_predict<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    n_clusters: usize,
    feature: &str,
    method: &str,
    threshold: f64,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    max_iter: usize,
    min_samples: usize,
    min_cluster_size: usize,
    seed: u64,
    distance: &str,
    absorb: &str,
    chi2_p: f64,
    chi2_scale: f64,
    n_jobs: usize,
    normalize: bool,
    resolution: f64,
    covariance_weight: f64,
    tangent_weight: f64,
    tangent_rank: usize,
    projection: &str,
    projection_dim: usize,
    projection_max_iter: usize,
    refine: usize,
    rank: usize,
    graph_degree: usize,
    balance: Option<f64>,
    auto_k_max: usize,
    fuzzifier: f64,
    leaf_refit: usize,
    canonical_order: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let kind = parse_method(
        method,
        min_samples,
        min_cluster_size,
        resolution,
        covariance_weight,
        tangent_weight,
        tangent_rank,
        rank,
        graph_degree,
        fuzzifier,
    )?;
    let nmf_dim = parse_projection(projection, projection_dim, projection_max_iter)?;
    let (labels, leaves) = if let Ok(a) = data.extract::<PyReadonlyArray2<'py, f64>>() {
        run_oneshot::<f64>(
            py,
            a,
            n_clusters,
            feature,
            kind,
            distance,
            absorb,
            chi2_p,
            chi2_scale,
            threshold,
            branching,
            leaf_cap,
            max_leaves,
            max_iter,
            seed,
            n_jobs,
            normalize,
            nmf_dim,
            refine,
            balance,
            auto_k_max,
            leaf_refit,
            canonical_order,
        )?
    } else if let Ok(a) = data.extract::<PyReadonlyArray2<'py, f32>>() {
        run_oneshot::<f32>(
            py,
            a,
            n_clusters,
            feature,
            kind,
            distance,
            absorb,
            chi2_p,
            chi2_scale,
            threshold,
            branching,
            leaf_cap,
            max_leaves,
            max_iter,
            seed,
            n_jobs,
            normalize,
            nmf_dim,
            refine,
            balance,
            auto_k_max,
            leaf_refit,
            canonical_order,
        )?
    } else {
        return Err(PyValueError::new_err(
            "data must be a 2-D float32 or float64 array",
        ));
    };
    if kind.consumes_k() {
        warn_leaf_budget(py, leaves, n_clusters, max_leaves)?;
    }
    warn_no_compression(py, leaves, labels.len(), max_leaves)?;
    warn_isotropic_gaussian(py, method, feature, leaves, labels.len())?;
    Ok(labels.into_pyarray(py))
}

// ── streaming estimator ───────────────────────────────────────────────────────────────────────

type BetulaTree<R, C> = CFTree<R, C, RouteKind, AbsorbKind<R>>;

/// The `absorb` values this binding accepts, as one string so the parser and every error message
/// cannot drift apart.
const ABSORB_CHOICES: &str = "absorb must be 'euclidean', 'manhattan', 'average', 'diameter', \
                              'ward', 'radius', 'chi2' or 'subspace'";

/// Resolve the absorption gate and effective threshold for element type `R` (shared by the one-shot
/// path and the streaming estimator). χ² uses the user-supplied within-cluster scale `chi2_scale`;
/// euclidean keeps the user's squared-distance threshold (so the default path is unchanged).
fn resolve_gate<R: Real>(
    absorb: &str,
    dim: usize,
    chi2_p: f64,
    chi2_scale: f64,
    threshold: f64,
) -> Result<(AbsorbKind<R>, R), &'static str> {
    // Every geometric criterion takes the caller's threshold verbatim, in its own units; only χ²
    // substitutes one, because its scale is a χ²_dim quantile rather than a distance.
    let thr = R::from_f64(threshold).unwrap();
    match absorb {
        "euclidean" => Ok((AbsorbKind::Euclidean, thr)),
        "manhattan" => Ok((AbsorbKind::Manhattan, thr)),
        "average" => Ok((AbsorbKind::Average, thr)),
        "diameter" => Ok((AbsorbKind::Diameter, thr)),
        "ward" => Ok((AbsorbKind::Ward, thr)),
        "radius" => Ok((AbsorbKind::Radius, thr)),
        "chi2" => {
            if chi2_scale <= 0.0 {
                return Err(
                    "absorb='chi2' requires chi2_scale > 0 (the within-cluster variance scale)",
                );
            }
            let s0 = R::from_f64(chi2_scale).unwrap();
            let kappa = R::from_usize(dim + 2).unwrap();
            let q = R::from_f64(chi2_quantile(dim, chi2_p)).unwrap();
            Ok((AbsorbKind::Chi2(MahalanobisChi2::new(s0, kappa)), q))
        }
        // Same units and the same prior as `chi2`; it differs only in reading the leaf's own basis
        // where the feature model carries one, so it shares the argument validation verbatim.
        "subspace" => {
            if chi2_scale <= 0.0 {
                return Err(
                    "absorb='subspace' requires chi2_scale > 0 (the within-cluster variance scale)",
                );
            }
            let s0 = R::from_f64(chi2_scale).unwrap();
            let kappa = R::from_usize(dim + 2).unwrap();
            let q = R::from_f64(chi2_quantile(dim, chi2_p)).unwrap();
            Ok((AbsorbKind::Subspace(SubspaceChi2::new(s0, kappa)), q))
        }
        _ => Err(ABSORB_CHOICES),
    }
}

/// A CF-tree specialised to one covariance model, generic over the element type `R` so the
/// streaming estimator can hold an `f64` *or* an `f32` tree (`f32` halves the resident tree memory
/// on high-dimensional embeddings). The variant is picked at first fit.
#[derive(serde::Serialize, serde::Deserialize)]
enum TreeState<R: Real> {
    Spherical(BetulaTree<R, Spherical<R>>),
    Diagonal(BetulaTree<R, Diagonal<R>>),
    Full(BetulaTree<R, Full<R>>),
    Fd(BetulaTree<R, FdSketch<R>>),
}

impl<R: Real> TreeState<R> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        feature: &str,
        dim: usize,
        branching: usize,
        leaf_cap: usize,
        threshold: R,
        max_leaves: usize,
        route: RouteKind,
        gate: AbsorbKind<R>,
        huber_k: Option<R>,
        balance: Option<R>,
    ) -> Result<Self, &'static str> {
        macro_rules! tree {
            () => {{
                let mut t =
                    CFTree::new(dim, branching, leaf_cap, threshold, max_leaves, route, gate);
                t.set_huber_k(huber_k);
                t.set_balance(balance);
                t
            }};
        }
        match feature {
            "spherical" => Ok(TreeState::Spherical(tree!())),
            "diagonal" => Ok(TreeState::Diagonal(tree!())),
            "full" => Ok(TreeState::Full(tree!())),
            "fd" => Ok(TreeState::Fd(tree!())),
            _ => Err("feature must be 'spherical', 'diagonal', 'full' or 'fd'"),
        }
    }

    fn insert(&mut self, row: &[R]) {
        match self {
            TreeState::Spherical(t) => t.insert(row),
            TreeState::Diagonal(t) => t.insert(row),
            TreeState::Full(t) => t.insert(row),
            TreeState::Fd(t) => t.insert(row),
        }
    }

    fn num_leaves(&self) -> usize {
        match self {
            TreeState::Spherical(t) => t.num_leaves(),
            TreeState::Diagonal(t) => t.num_leaves(),
            TreeState::Full(t) => t.num_leaves(),
            TreeState::Fd(t) => t.num_leaves(),
        }
    }

    fn refit_leaves(&mut self, flat: &[R], n: usize) {
        match self {
            TreeState::Spherical(t) => t.refit_leaves(flat, n),
            TreeState::Diagonal(t) => t.refit_leaves(flat, n),
            TreeState::Full(t) => t.refit_leaves(flat, n),
            TreeState::Fd(t) => t.refit_leaves(flat, n),
        }
    }

    fn rebuilds(&self) -> usize {
        match self {
            TreeState::Spherical(t) => t.rebuilds(),
            TreeState::Diagonal(t) => t.rebuilds(),
            TreeState::Full(t) => t.rebuilds(),
            TreeState::Fd(t) => t.rebuilds(),
        }
    }

    fn threshold(&self) -> f64 {
        match self {
            TreeState::Spherical(t) => t.threshold().to_f64().unwrap(),
            TreeState::Diagonal(t) => t.threshold().to_f64().unwrap(),
            TreeState::Full(t) => t.threshold().to_f64().unwrap(),
            TreeState::Fd(t) => t.threshold().to_f64().unwrap(),
        }
    }

    fn nearest_entry(&self, row: &[R]) -> usize {
        match self {
            TreeState::Spherical(t) => t.nearest_entry(row),
            TreeState::Diagonal(t) => t.nearest_entry(row),
            TreeState::Full(t) => t.nearest_entry(row),
            TreeState::Fd(t) => t.nearest_entry(row),
        }
    }

    fn label_proba(
        &self,
        kind: Kind,
        k: usize,
        max_iter: usize,
        seed: u64,
        auto_k_max: usize,
        nmf_dim: Option<ProjectionSpec>,
    ) -> Labelling {
        match self {
            TreeState::Spherical(t) => label_features_proba(
                t.leaf_features(),
                kind,
                k,
                max_iter,
                seed,
                auto_k_max,
                nmf_dim,
            ),
            TreeState::Diagonal(t) => label_features_proba(
                t.leaf_features(),
                kind,
                k,
                max_iter,
                seed,
                auto_k_max,
                nmf_dim,
            ),
            TreeState::Full(t) => label_features_proba(
                t.leaf_features(),
                kind,
                k,
                max_iter,
                seed,
                auto_k_max,
                nmf_dim,
            ),
            TreeState::Fd(t) => label_features_proba(
                t.leaf_features(),
                kind,
                k,
                max_iter,
                seed,
                auto_k_max,
                nmf_dim,
            ),
        }
    }

    /// COP-KMeans over the leaves with leaf-index constraints; one cluster label per row's leaf.
    fn label_constrained(
        &self,
        k: usize,
        must: &[(usize, usize)],
        cannot: &[(usize, usize)],
        max_iter: usize,
        seed: u64,
    ) -> Result<Vec<i64>, ConstraintError> {
        match self {
            TreeState::Spherical(t) => {
                label_features_constrained(t.leaf_features(), k, must, cannot, max_iter, seed)
            }
            TreeState::Diagonal(t) => {
                label_features_constrained(t.leaf_features(), k, must, cannot, max_iter, seed)
            }
            TreeState::Full(t) => {
                label_features_constrained(t.leaf_features(), k, must, cannot, max_iter, seed)
            }
            TreeState::Fd(t) => {
                label_features_constrained(t.leaf_features(), k, must, cannot, max_iter, seed)
            }
        }
    }

    fn decay(&mut self, factor: R) {
        match self {
            TreeState::Spherical(t) => t.decay(factor),
            TreeState::Diagonal(t) => t.decay(factor),
            TreeState::Full(t) => t.decay(factor),
            TreeState::Fd(t) => t.decay(factor),
        }
    }

    /// Stream `n` rows of `flat` into the tree, lazily creating it (and resolving the gate) on the
    /// first call, applying EWMA decay first. `slot` is the estimator's per-dtype tree.
    #[allow(clippy::too_many_arguments)]
    fn stream_chunk(
        slot: &mut Option<Self>,
        cfg: &StreamCfg<'_>,
        flat: &[R],
        n: usize,
        dim: usize,
        order: Option<&[u32]>,
    ) -> Result<(), &'static str> {
        if cfg.decay < 1.0 {
            if let Some(tree) = slot.as_mut() {
                tree.decay(R::from_f64(cfg.decay).unwrap());
            }
        }
        if slot.is_none() {
            let (gate, thr) =
                resolve_gate::<R>(cfg.absorb, dim, cfg.chi2_p, cfg.chi2_scale, cfg.threshold)?;
            *slot = Some(Self::new(
                cfg.feature,
                dim,
                cfg.branching,
                cfg.leaf_cap,
                thr,
                cfg.max_leaves,
                cfg.route,
                gate,
                cfg.huber_k.map(|k| R::from_f64(k).unwrap()),
                cfg.balance.and_then(R::from_f64),
            )?);
        }
        let tree = slot.as_mut().unwrap();
        for rank in 0..n {
            let i = order.map_or(rank, |o| o[rank] as usize);
            tree.insert(&flat[i * dim..(i + 1) * dim]);
        }
        Ok(())
    }

    /// Route `n` rows of `flat` to their nearest leaf and read the cached labels.
    fn route(&self, labels: &[i64], flat: &[R], n: usize, dim: usize) -> Vec<i64> {
        map_rows(n, |i| {
            labels[self.nearest_entry(&flat[i * dim..(i + 1) * dim])]
        })
    }

    /// Stream CSR rows into the tree by expanding each into a reused dense buffer (so the dense
    /// `n × dim` matrix is never materialized). Caller has validated the CSR arrays. Generic over `R`.
    fn stream_chunk_csr(
        slot: &mut Option<Self>,
        cfg: &StreamCfg<'_>,
        data: &[R],
        indices: &[i64],
        indptr: &[i64],
        dim: usize,
        order: Option<&[u32]>,
    ) -> Result<(), &'static str> {
        if cfg.decay < 1.0 {
            if let Some(tree) = slot.as_mut() {
                tree.decay(R::from_f64(cfg.decay).unwrap());
            }
        }
        if slot.is_none() {
            let (gate, thr) =
                resolve_gate::<R>(cfg.absorb, dim, cfg.chi2_p, cfg.chi2_scale, cfg.threshold)?;
            *slot = Some(Self::new(
                cfg.feature,
                dim,
                cfg.branching,
                cfg.leaf_cap,
                thr,
                cfg.max_leaves,
                cfg.route,
                gate,
                cfg.huber_k.map(|k| R::from_f64(k).unwrap()),
                cfg.balance.and_then(R::from_f64),
            )?);
        }
        let tree = slot.as_mut().unwrap();
        let mut buf = vec![R::zero(); dim];
        let rows = indptr.len().saturating_sub(1);
        for rank in 0..rows {
            let i = order.map_or(rank, |o| o[rank] as usize);
            let (lo, hi) = (indptr[i] as usize, indptr[i + 1] as usize);
            for k in lo..hi {
                buf[indices[k] as usize] = data[k];
            }
            tree.insert(&buf);
            for k in lo..hi {
                buf[indices[k] as usize] = R::zero();
            }
        }
        Ok(())
    }

    /// Route CSR rows (expanded into a reused buffer) to their nearest leaf labels (serial — the
    /// shared buffer precludes the parallel path; predict is cold relative to the build).
    fn route_csr(
        &self,
        labels: &[i64],
        data: &[R],
        indices: &[i64],
        indptr: &[i64],
        dim: usize,
    ) -> Vec<i64> {
        let mut buf = vec![R::zero(); dim];
        let mut out = Vec::with_capacity(indptr.len().saturating_sub(1));
        for w in indptr.windows(2) {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            for k in lo..hi {
                buf[indices[k] as usize] = data[k];
            }
            out.push(labels[self.nearest_entry(&buf)]);
            for k in lo..hi {
                buf[indices[k] as usize] = R::zero();
            }
        }
        out
    }

    /// Per-leaf (microcluster) `(centers, weights, radii, dim)` in `f64`.
    fn leaf_stats(&self) -> F64Stats {
        match self {
            TreeState::Spherical(t) => compute_leaf_stats(t.leaf_features()),
            TreeState::Diagonal(t) => compute_leaf_stats(t.leaf_features()),
            TreeState::Full(t) => compute_leaf_stats(t.leaf_features()),
            TreeState::Fd(t) => compute_leaf_stats(t.leaf_features()),
        }
    }

    /// Pooled per-cluster `(centers, radii, weights, dim)` for `k` clusters, given the leaf labels.
    fn cluster_stats(&self, labels: &[i64], k: usize) -> F64Stats {
        match self {
            TreeState::Spherical(t) => compute_cluster_stats(t.leaf_features(), labels, k),
            TreeState::Diagonal(t) => compute_cluster_stats(t.leaf_features(), labels, k),
            TreeState::Full(t) => compute_cluster_stats(t.leaf_features(), labels, k),
            TreeState::Fd(t) => compute_cluster_stats(t.leaf_features(), labels, k),
        }
    }

    /// Cholesky factor of each cluster's pooled covariance.
    fn cluster_chol(
        &self,
        labels: &[i64],
        k: usize,
        centers: &[f64],
    ) -> Vec<Option<Vec<Vec<f64>>>> {
        match self {
            TreeState::Spherical(t) => compute_cluster_chol(t.leaf_features(), labels, k, centers),
            TreeState::Diagonal(t) => compute_cluster_chol(t.leaf_features(), labels, k, centers),
            TreeState::Full(t) => compute_cluster_chol(t.leaf_features(), labels, k, centers),
            TreeState::Fd(t) => compute_cluster_chol(t.leaf_features(), labels, k, centers),
        }
    }

    /// Point dimension of the leaf summary, or `0` before anything has been inserted.
    fn leaf_dim(&self) -> usize {
        fn go<R: Real, C: ClusterFeature<R>>(f: &[C]) -> usize {
            f.first().map_or(0, |c| c.dim())
        }
        match self {
            TreeState::Spherical(t) => go(t.leaf_features()),
            TreeState::Diagonal(t) => go(t.leaf_features()),
            TreeState::Full(t) => go(t.leaf_features()),
            TreeState::Fd(t) => go(t.leaf_features()),
        }
    }

    /// Gaussian-kernel MMD between the leaf surrogate and a raw sample, row-major `M × dim`.
    fn summary_mmd(&self, sample: &[R], bandwidth: Option<R>) -> f64 {
        fn go<R: Real, C: ClusterFeature<R>>(f: &[C], s: &[R], h: Option<R>) -> f64 {
            crate::fidelity::summary_mmd(f, s, h)
                .to_f64()
                .unwrap_or(f64::NAN)
        }
        match self {
            TreeState::Spherical(t) => go(t.leaf_features(), sample, bandwidth),
            TreeState::Diagonal(t) => go(t.leaf_features(), sample, bandwidth),
            TreeState::Full(t) => go(t.leaf_features(), sample, bandwidth),
            TreeState::Fd(t) => go(t.leaf_features(), sample, bandwidth),
        }
    }

    /// `(calinski_harabasz, davies_bouldin, medoid_silhouette)` over the labelled leaves.
    fn validity(&self, labels: &[i64], k: usize) -> (f64, f64, f64) {
        match self {
            TreeState::Spherical(t) => compute_validity(t.leaf_features(), labels, k),
            TreeState::Diagonal(t) => compute_validity(t.leaf_features(), labels, k),
            TreeState::Full(t) => compute_validity(t.leaf_features(), labels, k),
            TreeState::Fd(t) => compute_validity(t.leaf_features(), labels, k),
        }
    }

    /// Sensitivity-sampling coreset over the leaf summary. Returns the sampled means flattened
    /// row-major, their weights, the point dimension, and the three scalars the guarantee is stated
    /// in terms of.
    fn coreset(&self, k: usize, size: usize, seed: u64) -> CoresetOut {
        fn build<R: Real, C: ClusterFeature<R>>(
            feats: &[C],
            k: usize,
            size: usize,
            seed: u64,
        ) -> CoresetOut {
            let cs = crate::coreset::sensitivity_coreset(feats, k, size, seed);
            let dim = feats.first().map_or(0, |f| f.dim());
            let mut points = Vec::with_capacity(cs.points.len() * dim);
            for p in &cs.points {
                points.extend(p.iter().map(|v| v.to_f64().unwrap_or(0.0)));
            }
            // The retained leaves' own RMS radii, not a property of the sampling weights: a
            // sampled leaf still summarises exactly the points it absorbed.
            let radii = cs
                .indices
                .iter()
                .map(|&i| {
                    let w = feats[i].weight().to_f64().unwrap_or(0.0);
                    if w > 0.0 {
                        (feats[i].ssd().to_f64().unwrap_or(0.0) / w).max(0.0).sqrt()
                    } else {
                        0.0
                    }
                })
                .collect();
            CoresetOut {
                points,
                weights: cs
                    .weights
                    .iter()
                    .map(|w| w.to_f64().unwrap_or(0.0))
                    .collect(),
                radii,
                dim,
                offset: cs.offset.to_f64().unwrap_or(0.0),
                reference_cost: cs.reference_cost.to_f64().unwrap_or(0.0),
                total_sensitivity: cs.total_sensitivity,
                n_leaves: cs.n_leaves,
            }
        }
        match self {
            TreeState::Spherical(t) => build(t.leaf_features(), k, size, seed),
            TreeState::Diagonal(t) => build(t.leaf_features(), k, size, seed),
            TreeState::Full(t) => build(t.leaf_features(), k, size, seed),
            TreeState::Fd(t) => build(t.leaf_features(), k, size, seed),
        }
    }

    /// OPTICS reachability plot over the leaf microclusters.
    fn reachability(&self, min_samples: usize, graph_degree: usize, seed: u64) -> Reachability {
        match self {
            TreeState::Spherical(t) => optics(t.leaf_features(), min_samples, graph_degree, seed),
            TreeState::Diagonal(t) => optics(t.leaf_features(), min_samples, graph_degree, seed),
            TreeState::Full(t) => optics(t.leaf_features(), min_samples, graph_degree, seed),
            TreeState::Fd(t) => optics(t.leaf_features(), min_samples, graph_degree, seed),
        }
    }

    /// Mapper topological-skeleton graph over the leaf microclusters.
    fn mapper(&self, p: &MapperParams) -> MapperGraph {
        match self {
            TreeState::Spherical(t) => mapper(t.leaf_features(), p),
            TreeState::Diagonal(t) => mapper(t.leaf_features(), p),
            TreeState::Full(t) => mapper(t.leaf_features(), p),
            TreeState::Fd(t) => mapper(t.leaf_features(), p),
        }
    }

    /// For each row: the deviation from its assigned cluster centroid, normalized by `scale`. Points
    /// routed to a noise microcluster score `+inf`.
    fn outlier_scores(
        &self,
        labels: &[i64],
        centers: &[f64],
        scale: OutlierScale<'_>,
        flat: &[R],
        n: usize,
        dim: usize,
    ) -> Vec<f64> {
        map_rows(n, |i| {
            let x = &flat[i * dim..(i + 1) * dim];
            let lab = labels[self.nearest_entry(x)];
            if lab < 0 {
                return f64::INFINITY;
            }
            let cl = lab as usize;
            match scale {
                OutlierScale::Radius(radii) => {
                    let mut d2 = 0.0;
                    for (j, &xj) in x.iter().enumerate() {
                        let diff = xj.to_f64().unwrap() - centers[cl * dim + j];
                        d2 += diff * diff;
                    }
                    let d = d2.sqrt();
                    let r = radii[cl];
                    if r > 0.0 { d / r } else { d }
                }
                OutlierScale::Whitened(chols) => {
                    let delta: Vec<f64> = x
                        .iter()
                        .enumerate()
                        .map(|(j, &xj)| xj.to_f64().unwrap() - centers[cl * dim + j])
                        .collect();
                    // A cluster of one point has nothing to whiten against, and the ridge cannot
                    // rescue it (it is relative to a trace of zero). Fall back to the raw distance,
                    // the same answer the scalar path gives for a zero radius.
                    match &chols[cl] {
                        None => delta.iter().map(|v| v * v).sum::<f64>().sqrt(),
                        Some(l) => (mahalanobis_sq_from_chol(l, &delta) / dim as f64).sqrt(),
                    }
                }
            }
        })
    }

    /// For each row: the index of its nearest leaf (microcluster) within [`Self::leaf_stats`] order.
    fn assign_microclusters(&self, flat: &[R], n: usize, dim: usize) -> Vec<i64> {
        map_rows(n, |i| {
            self.nearest_entry(&flat[i * dim..(i + 1) * dim]) as i64
        })
    }
}

/// Validate CSR arrays at the untrusted boundary before the `O(nnz)` row expansion. Delegates to the
/// pure-Rust [`crate::sparse::validate_csr`] (matched lengths, well-formed `indptr`, in-range indices,
/// finite values, and an `n_features` upper bound so a hostile caller can't force an unbounded
/// allocation), mapping its message to a Python `ValueError`.
fn validate_csr(data: &[f64], indices: &[i64], indptr: &[i64], n_features: usize) -> PyResult<()> {
    crate::sparse::validate_csr(data, indices, indptr, n_features).map_err(PyValueError::new_err)
}

/// Extract an `(m, 2)` integer constraint array as row-index pairs (validates the second axis).
fn pairs_from(arr: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<(i64, i64)>> {
    let a = arr.as_array();
    if a.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "constraint arrays must have shape (m, 2)",
        ));
    }
    Ok(a.outer_iter().map(|r| (r[0], r[1])).collect())
}

/// Translate point-level constraints to leaf-index constraints and run COP-KMeans. Each constrained
/// row is routed to its leaf; a same-leaf must-link is trivially satisfied (dropped), a same-leaf
/// cannot-link is infeasible at the current granularity (the two points were compressed into one
/// microcluster) and is reported with an actionable message.
#[allow(clippy::too_many_arguments)]
fn constrained_labels<R: Real>(
    tree: &TreeState<R>,
    flat: &[R],
    n: usize,
    dim: usize,
    must: &[(i64, i64)],
    cannot: &[(i64, i64)],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> PyResult<Vec<i64>> {
    let leaf_of = |idx: i64| -> PyResult<usize> {
        if idx < 0 || idx as usize >= n {
            return Err(PyValueError::new_err(format!(
                "constraint row index {idx} is out of range for {n} samples"
            )));
        }
        let i = idx as usize;
        Ok(tree.nearest_entry(&flat[i * dim..(i + 1) * dim]))
    };
    let mut leaf_must: Vec<(usize, usize)> = Vec::with_capacity(must.len());
    for &(a, b) in must {
        let (la, lb) = (leaf_of(a)?, leaf_of(b)?);
        if la != lb {
            leaf_must.push((la.min(lb), la.max(lb)));
        }
    }
    let mut leaf_cannot: Vec<(usize, usize)> = Vec::with_capacity(cannot.len());
    for &(a, b) in cannot {
        let (la, lb) = (leaf_of(a)?, leaf_of(b)?);
        if la == lb {
            return Err(PyValueError::new_err(format!(
                "cannot-link ({a}, {b}) is infeasible: both points fall in the same microcluster at \
                 the current threshold; lower `threshold` to keep them separable"
            )));
        }
        leaf_cannot.push((la.min(lb), la.max(lb)));
    }
    leaf_must.sort_unstable();
    leaf_must.dedup();
    leaf_cannot.sort_unstable();
    leaf_cannot.dedup();
    tree.label_constrained(k, &leaf_must, &leaf_cannot, max_iter, seed)
        .map_err(|e| PyValueError::new_err(constraint_msg(e)))
}

struct StreamCfg<'a> {
    feature: &'a str,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    route: RouteKind,
    absorb: &'a str,
    chi2_p: f64,
    chi2_scale: f64,
    threshold: f64,
    decay: f64,
    huber_k: Option<f64>,
    balance: Option<f64>,
}

/// Stateful BETULA estimator. `partial_fit` streams data into a memory-bounded CF-tree; `fit`
/// (re)builds from one array; `predict` labels new points via their nearest leaf. The covariance
/// model and dimensionality are locked in at the first `partial_fit` / `fit`.
fn default_resolution() -> f64 {
    1.0
}

fn default_covariance_weight() -> f64 {
    0.0
}

fn default_tangent_rank() -> usize {
    2
}

fn default_projection_max_iter() -> usize {
    100
}

fn default_rank() -> usize {
    2
}

fn default_fuzzifier() -> f64 {
    2.0
}

#[pyclass(name = "Betula", module = "betula_cluster._core")]
#[derive(serde::Serialize, serde::Deserialize)]
struct Betula {
    feature: String,
    kind: Kind,
    route: RouteKind,
    // Raw constructor params kept verbatim for scikit-learn `get_params` / `set_params`.
    method: String,
    distance: String,
    min_samples: usize,
    min_cluster_size: usize,
    n_clusters: usize,
    threshold: f64,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    max_iter: usize,
    seed: u64,
    absorb: String,
    chi2_p: f64,
    chi2_scale: f64,
    decay: f64,
    #[serde(default)]
    normalize: bool,
    /// Huber/winsorization radius in per-dimension std units; `None` disables robust insertion.
    #[serde(default)]
    huber_k: Option<f64>,
    /// Per-leaf mass cap as a multiple of `n / max_leaves`; `None` is the purely geometric budget.
    #[serde(default)]
    balance: Option<f64>,
    /// Leiden resolution `γ` (only used by `method="leiden"` / `"leiden-cpm"`); kept for `get_params`.
    #[serde(default = "default_resolution")]
    resolution: f64,
    /// Covariance-aware Leiden weight `β` (log-Euclidean shape term, `method="leiden"` / `"leiden-cpm"`
    /// with `feature="full"`); kept for `get_params`.
    #[serde(default = "default_covariance_weight")]
    covariance_weight: f64,
    /// Tangent-aware Leiden weight `γ` (Grassmann subspace term); kept for `get_params`.
    #[serde(default = "default_covariance_weight")]
    tangent_weight: f64,
    /// Rank `r` of the local tangent subspaces compared by `tangent_weight`; kept for `get_params`.
    #[serde(default = "default_tangent_rank")]
    tangent_rank: usize,
    /// Subspace rank `q` of the subspace heads (`method="mppca"` / `"mfa"`); kept for `get_params`.
    #[serde(default = "default_rank")]
    rank: usize,
    /// Fuzzifier `m > 1` of the fuzzy c-means head (`method="fuzzy-cmeans"`); kept for `get_params`.
    /// `#[serde(default = ...)]` is what lets a model persisted before this field existed still load.
    #[serde(default = "default_fuzzifier")]
    fuzzifier: f64,
    /// Out-degree of the `method="hdbscan"` proximity graph; `0` = exact complete graph.
    #[serde(default)]
    graph_degree: usize,
    /// Ceiling the automatic selectors search under at `n_clusters=0`; `0` takes the crate default.
    /// `#[serde(default)]` is what lets a model persisted before this field existed load as `0`.
    #[serde(default)]
    auto_k_max: usize,
    /// Phase-3 CF-weighted NMF reduction dim (`projection="weighted-nmf"`); `None` = no projection.
    #[serde(default)]
    nmf_dim: Option<usize>,
    /// Use the KL-divergence NMF variant (`projection="weighted-nmf-kl"`, for count data) over Frobenius.
    #[serde(default)]
    nmf_kl: bool,
    /// NMF solver sweeps, independent of the head's `max_iter`; kept for `get_params`.
    #[serde(default = "default_projection_max_iter")]
    nmf_max_iter: usize,
    /// The projection of rank `nmf_dim` is a CF-weighted PCA rather than an NMF.
    #[serde(default)]
    projection_svd: bool,
    dim: usize,
    // The estimator holds an f64 *or* an f32 tree (chosen by the first input's dtype) — at most one
    // is ever `Some`. f32 halves the resident tree memory on high-d embeddings.
    state64: Option<TreeState<f64>>,
    state32: Option<TreeState<f32>>,
    labels: Option<Vec<i64>>,
    /// Per-leaf GMM soft responsibilities (flattened `n_leaves × k`, and `k`) set at finalize for the
    /// GMM heads; `None` for other heads. Backs `microcluster_proba_` / `predict_proba`.
    #[serde(default)]
    proba: Option<(Vec<f64>, usize)>,
    /// The NMF parts `H` (`r×d`) shared by every leaf, set at finalize when a projection ran.
    #[serde(default)]
    nmf_components: Option<Vec<Vec<f64>>>,
    /// Relative reconstruction error of the projection, `‖X̃ − W H‖_F / ‖X̃‖_F`.
    #[serde(default)]
    nmf_reconstruction_err: Option<f64>,
    /// The head's own model of a point, set at finalize. Backs `predict` / `predict_proba`; `None`
    /// keeps the microcluster route (see [`Model`]).
    #[serde(default)]
    rule: Option<PointRule>,
    /// BIRCH Phase-4 Lloyd sweeps over the raw rows after the leaf clustering; `0` disables it.
    #[serde(default)]
    refine: usize,
    /// Lloyd passes at the *micro-cluster* level, before the head runs; `0` disables it. Needs the
    /// rows, so `fit` / `fit_predict` honour it and a `partial_fit` stream cannot.
    #[serde(default)]
    leaf_refit: usize,
    /// Sort the rows by a key derived from the data before inserting them, so the summary is a
    /// function of the multiset rather than of the arrival sequence. Needs all the rows at once, so
    /// like `leaf_refit` a `partial_fit` stream cannot honour it.
    #[serde(default)]
    canonical_order: bool,
}

/// A 2-D array as flat row-major rows, casting from the other float dtype if needed (lossless
/// `f32→f64`; the deliberate `f64→f32` narrowing matches the f32 tree). Matching dtype is the one
/// case that can borrow, so it goes through [`to_rows`]; a cast has to build the buffer anyway.
fn flat_as<'py, R: Real + Element>(
    data: &Bound<'py, PyAny>,
    prep: RowPrep,
) -> PyResult<(Rows<'py, R>, usize, usize)> {
    if let Ok(a) = data.extract::<PyReadonlyArray2<'py, R>>() {
        return to_rows(a, prep);
    }
    let (mut flat, n, dim) = if let Ok(a) = data.extract::<PyReadonlyArray2<f64>>() {
        cast_flat::<f64, R>(&a)?
    } else if let Ok(a) = data.extract::<PyReadonlyArray2<f32>>() {
        cast_flat::<f32, R>(&a)?
    } else {
        return Err(PyValueError::new_err(
            "data must be a 2-D float32 or float64 array",
        ));
    };
    prepare_rows(&mut flat, n, dim, prep);
    Ok((Rows::Owned(flat), n, dim))
}

fn cast_flat<S: Real + Element, R: Real>(
    data: &PyReadonlyArray2<'_, S>,
) -> PyResult<(Vec<R>, usize, usize)> {
    let arr = data.as_array();
    let (n, dim) = (arr.shape()[0], arr.shape()[1]);
    if n == 0 || dim == 0 {
        return Err(PyValueError::new_err("data must be a non-empty 2-D array"));
    }
    // Contiguous fast path (memcpy-able source slice + vectorizable finiteness check), with the
    // per-element dtype cast folded into the collect; falls back to a strided scan for views.
    let cast = |s: &[S]| -> PyResult<Vec<R>> {
        if s.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "data contains NaN or infinite values",
            ));
        }
        Ok(s.iter()
            .map(|v| R::from_f64(v.to_f64().unwrap()).unwrap())
            .collect())
    };
    let flat = match arr.as_slice() {
        Some(s) => cast(s)?,
        None => cast(&arr.iter().copied().collect::<Vec<S>>())?,
    };
    Ok((flat, n, dim))
}

impl Betula {
    fn reset(&mut self) {
        self.state64 = None;
        self.state32 = None;
        self.labels = None;
        self.proba = None;
        self.rule = None;
        self.dim = 0;
    }

    /// Stream a chunk into the matching-dtype tree (dtype is the existing tree's, or the input's at
    /// first fit). Invalidates the cached labels. The config is built inline (not via a `&self`
    /// method) so the borrow checker keeps `&self.feature` disjoint from `&mut self.state*`.
    /// The permutation to insert `n` rows through, or `None` for arrival order.
    ///
    /// `Arrival::Chunk` always gets `None`: a `partial_fit` stream never has the whole dataset, so
    /// the guarantee `canonical_order` offers cannot be given and pretending otherwise would order
    /// each chunk canonically within itself and leave the answer dependent on where the chunks were
    /// cut. The estimator documents the asymmetry, exactly as it does for `leaf_refit`.
    fn insertion_order<R: Real>(
        &self,
        flat: &[R],
        n: usize,
        dim: usize,
        arrival: Arrival,
    ) -> Option<Vec<u32>> {
        (self.canonical_order && arrival == Arrival::Whole)
            .then(|| canonical_permutation(flat, n, dim))
    }

    fn stream(&mut self, data: &Bound<'_, PyAny>, arrival: Arrival) -> PyResult<()> {
        let use_f32 = match (&self.state64, &self.state32) {
            (Some(_), _) => false,
            (_, Some(_)) => true,
            (None, None) => data.extract::<PyReadonlyArray2<f64>>().is_err(),
        };
        let cfg = StreamCfg {
            feature: &self.feature,
            branching: self.branching,
            leaf_cap: self.leaf_cap,
            max_leaves: self.max_leaves,
            route: self.route,
            absorb: &self.absorb,
            chi2_p: self.chi2_p,
            chi2_scale: self.chi2_scale,
            threshold: self.threshold,
            decay: self.decay,
            huber_k: self.huber_k,
            balance: self.balance,
        };
        if use_f32 {
            let (src, n, dim) = flat_as::<f32>(data, self.row_prep())?;
            let flat = src.as_slice();
            if self.dim != 0 && self.dim != dim {
                return Err(PyValueError::new_err(
                    "dimension mismatch with previously fitted data",
                ));
            }
            let order = self.insertion_order(flat, n, dim, arrival);
            TreeState::stream_chunk(&mut self.state32, &cfg, flat, n, dim, order.as_deref())
                .map_err(PyValueError::new_err)?;
            self.dim = dim;
        } else {
            let (src, n, dim) = flat_as::<f64>(data, self.row_prep())?;
            let flat = src.as_slice();
            if self.dim != 0 && self.dim != dim {
                return Err(PyValueError::new_err(
                    "dimension mismatch with previously fitted data",
                ));
            }
            let order = self.insertion_order(flat, n, dim, arrival);
            TreeState::stream_chunk(&mut self.state64, &cfg, flat, n, dim, order.as_deref())
                .map_err(PyValueError::new_err)?;
            self.dim = dim;
        }
        self.labels = None;
        self.proba = None;
        Ok(())
    }

    /// Stream CSR rows into an `f64` tree (sparse input is `f64`-only). Mirrors [`Betula::stream`]
    /// but expands rows on the fly, so the dense matrix is never materialized.
    fn stream_csr(
        &mut self,
        data: &[f64],
        indices: &[i64],
        indptr: &[i64],
        n_features: usize,
        arrival: Arrival,
    ) -> PyResult<()> {
        if self.state32.is_some() {
            return Err(PyValueError::new_err(
                "sparse (CSR) input is float64-only; this estimator was already fit on float32 data",
            ));
        }
        // The sheet projection writes `x_0 = √(1 + ‖s‖²) ≥ 1` into every row, so column 0 stops being
        // sparse. Refused rather than run on rows the head's own convention was never applied to.
        if self.row_prep() == RowPrep::Sheet {
            return Err(PyValueError::new_err(
                "method='hyperbolic' needs dense rows: the sheet projection fills column 0 in every row",
            ));
        }
        self.check_dim(n_features)?;
        validate_csr(data, indices, indptr, n_features)?;
        let cfg = StreamCfg {
            feature: &self.feature,
            branching: self.branching,
            leaf_cap: self.leaf_cap,
            max_leaves: self.max_leaves,
            route: self.route,
            absorb: &self.absorb,
            chi2_p: self.chi2_p,
            chi2_scale: self.chi2_scale,
            threshold: self.threshold,
            decay: self.decay,
            huber_k: self.huber_k,
            balance: self.balance,
        };
        // Same rule as the dense path: only a call that holds the whole matrix can order it.
        let order = (self.canonical_order && arrival == Arrival::Whole)
            .then(|| canonical_permutation_csr(data, indices, indptr, n_features))
            .transpose()
            .map_err(PyValueError::new_err)?;
        TreeState::stream_chunk_csr(
            &mut self.state64,
            &cfg,
            data,
            indices,
            indptr,
            n_features,
            order.as_deref(),
        )
        .map_err(PyValueError::new_err)?;
        self.dim = n_features;
        self.labels = None;
        self.proba = None;
        Ok(())
    }

    /// Label CSR rows by their nearest leaf (requires a finalized clustering).
    fn route_csr_labels(
        &self,
        data: &[f64],
        indices: &[i64],
        indptr: &[i64],
        n_features: usize,
    ) -> PyResult<Vec<i64>> {
        let labels = self.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err("call fit() / fit_predict() / partial_fit() (finalize) first")
        })?;
        self.check_dim(n_features)?;
        validate_csr(data, indices, indptr, n_features)?;
        let t = self
            .state64
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no fitted float64 tree for sparse predict"))?;
        match self.rule.as_ref() {
            Some(rule) => Ok(rule.label_csr(data, indices, indptr, n_features)),
            None => Ok(t.route_csr(labels, data, indices, indptr, n_features)),
        }
    }

    /// The row preparation this estimator's head needs, from the one place that knows both the
    /// user's `normalize` flag and the method it was configured with.
    ///
    /// `self.normalize` is left as the caller set it — the wrapper's `get_params` has to round-trip
    /// through `clone()` unchanged — so the forcing lives here rather than in the stored field.
    fn row_prep(&self) -> RowPrep {
        if self.method == "hyperbolic" {
            RowPrep::Sheet
        } else if self.normalize {
            RowPrep::Unit
        } else {
            RowPrep::None
        }
    }

    fn check_dim(&self, dim: usize) -> PyResult<()> {
        if self.dim != 0 && self.dim != dim {
            return Err(PyValueError::new_err(
                "dimension mismatch with previously fitted data",
            ));
        }
        Ok(())
    }

    /// The Phase-3 projection this estimator was configured with.
    ///
    /// The three stored fields are the wire format — a schema-versioned CBOR layout that older saved
    /// models still have to load — so the sum type is rebuilt here rather than stored. Everything
    /// above this function sees only [`ProjectionSpec`].
    fn projection_spec(&self) -> Option<ProjectionSpec> {
        let rank = self.nmf_dim?;
        let kind = if self.projection_svd {
            ProjectionKind::Svd
        } else {
            ProjectionKind::Nmf {
                kl: self.nmf_kl,
                max_iter: self.nmf_max_iter,
            }
        };
        Some(ProjectionSpec { rank, kind })
    }

    /// The point model the finalized head defines, or `None` when it has none, or when a projection
    /// replaced the feature space the rows live in — in both cases the microcluster route is the only
    /// defined labelling. Constrained runs never take this path: COP-KMeans labels satisfy pairwise
    /// constraints that a Voronoi rule is free to violate.
    fn point_rule(&self, mixture: Option<Mixture>) -> Option<PointRule> {
        let Kind::Parametric(method) = self.kind else {
            return None;
        };
        if self.nmf_dim.is_some() {
            return None;
        }
        let unit = match assignment_rule(method) {
            Rule::Centroid { unit } => unit,
            Rule::Posterior => return mixture.map(PointRule::Posterior),
            // The hyperbolic head always names its own centres, so its rule arrives from the head
            // itself. Deriving one here would average each cluster's leaves in the ambient
            // coordinates, and that mean is not the Lorentzian centroid.
            Rule::Lorentz | Rule::Microcluster => return None,
        };
        // `cluster_stats_any` yields radii *before* weights (`F64Stats` swaps the two between the
        // leaf and the cluster helper); reading them in leaf order drops every zero-radius cluster.
        let (centers, _radii, weights, dim) = self.cluster_stats_any().ok()?;
        centers_rule(&centers, &weights, dim, unit)
    }

    /// `leaf_refit` Lloyd passes at the *micro-cluster* level, between the tree build and the head:
    /// each pass re-routes every row against the finished tree and rebuilds the leaf CFs from the
    /// rows they won. See [`CFTree::refit_leaves`] for why that is not the same summary.
    ///
    /// Like [`Self::refine_rule`], only the in-memory entry points can run it: a `partial_fit`
    /// stream keeps a tree, not the rows. It runs in the tree's own dtype, because it rewrites the
    /// tree's own features rather than a `f64` centre rule.
    fn refit_tree(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.leaf_refit == 0 {
            return Ok(());
        }
        let (prep, passes) = (self.row_prep(), self.leaf_refit);
        if let Some(t) = self.state64.as_mut() {
            let (src, n, _) = flat_as::<f64>(data, prep)?;
            let flat = src.as_slice();
            py.detach(|| {
                for _ in 0..passes {
                    t.refit_leaves(flat, n);
                }
            });
        } else if let Some(t) = self.state32.as_mut() {
            let (src, n, _) = flat_as::<f32>(data, prep)?;
            let flat = src.as_slice();
            py.detach(|| {
                for _ in 0..passes {
                    t.refit_leaves(flat, n);
                }
            });
        }
        self.labels = None;
        self.proba = None;
        Ok(())
    }

    /// BIRCH Phase 4 on the finalized centre rule: `refine` Lloyd sweeps over the rows just fitted,
    /// warm-started from the Phase-3 centres. A no-op unless the head left a centre rule behind.
    ///
    /// Only the in-memory entry points can run it. `partial_fit` accumulates a tree, not the data,
    /// so there is no `X` left to sweep; the CSR paths would have to densify the matrix they exist to
    /// avoid. Both keep the summary centres, and the docs say so.
    ///
    /// The sweep runs in `f64` whatever the tree's dtype, because the rule's centres are `f64` and
    /// [`PointRule::label_of`] already scores every row in `f64` — refining in `f32` would optimize a
    /// different objective from the one that then labels the points.
    fn refine_rule(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let Kind::Parametric(method) = self.kind else {
            return Ok(());
        };
        let Rule::Centroid { unit } = assignment_rule(method) else {
            return Ok(());
        };
        // A Lloyd sweep is the k-means update, so it refines a k-means centre and destroys a medoid.
        if !refinable(method) {
            return Ok(());
        }
        if self.refine == 0 || !matches!(self.rule, Some(PointRule::Centers { .. })) {
            return Ok(());
        }
        let (src, n, dim) = flat_as::<f64>(data, self.row_prep())?;
        let flat = src.as_slice();
        self.check_dim(dim)?;
        let iters = self.refine;
        if let Some(PointRule::Centers { rows, .. }) = self.rule.as_mut() {
            py.detach(|| refine_centers(rows, flat, n, dim, unit, iters));
        }
        Ok(())
    }

    /// Cluster the current leaf features (whichever dtype tree exists) and cache the labels.
    ///
    /// Takes `py` only to raise the leaf-budget warning: this is the one place every estimator
    /// entry point (`fit`, `fit_predict`, `partial_fit(None)`, and both CSR paths) funnels through,
    /// so the rule lives here rather than being repeated at each of them.
    fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        let (kind, k, mi, seed, akm, nmf) = (
            self.kind,
            self.n_clusters,
            self.max_iter,
            self.seed,
            self.auto_k_max,
            self.projection_spec(),
        );
        let result = if let Some(t) = &self.state64 {
            Some(t.label_proba(kind, k, mi, seed, akm, nmf))
        } else {
            self.state32
                .as_ref()
                .map(|t| t.label_proba(kind, k, mi, seed, akm, nmf))
        };
        match result {
            Some(out) => {
                self.labels = Some(out.labels);
                self.proba = out.proba;
                let (components, err) = out.parts.unzip();
                self.nmf_components = components;
                self.nmf_reconstruction_err = err;
                self.rule = out.rule.or_else(|| self.point_rule(out.mixture));
            }
            None => {
                self.labels = None;
                self.proba = None;
                self.nmf_components = None;
                self.nmf_reconstruction_err = None;
                self.rule = None;
            }
        }
        if kind.consumes_k() {
            warn_leaf_budget(py, self.n_leaves_(), k, self.max_leaves)?;
        }
        if let Some(labels) = &self.labels {
            let mut seen: Vec<i64> = labels.iter().copied().filter(|&l| l >= 0).collect();
            seen.sort_unstable();
            seen.dedup();
            warn_auto_k_saturated(py, kind, k, seen.len(), self.n_leaves_(), akm)?;
        }
        // The estimator does not carry a point count, but the leaf weights sum to one — and for the
        // unweighted `fit(X)` path that sum *is* `n`, which is what the compression test needs.
        let seen = self
            .leaf_stats_any()
            .map_or(0.0, |(_, w, _, _)| w.iter().sum::<f64>());
        let method = self.method.clone();
        let feature = self.feature.clone();
        warn_isotropic_gaussian(py, &method, &feature, self.n_leaves_(), seen as usize)?;
        Ok(())
    }

    /// Cluster the leaves under pairwise constraints (COP-KMeans). `data` is the just-streamed array,
    /// re-read only to route the constrained rows to their leaves. Sets `labels` (no GMM posterior).
    fn finalize_constrained(
        &mut self,
        data: &Bound<'_, PyAny>,
        must: &[(i64, i64)],
        cannot: &[(i64, i64)],
    ) -> PyResult<()> {
        let (k, mi, seed, norm) = (self.n_clusters, self.max_iter, self.seed, self.row_prep());
        let labels = if let Some(t) = self.state64.as_ref() {
            let (src, n, dim) = flat_as::<f64>(data, norm)?;
            let flat = src.as_slice();
            constrained_labels(t, flat, n, dim, must, cannot, k, mi, seed)?
        } else if let Some(t) = self.state32.as_ref() {
            let (src, n, dim) = flat_as::<f32>(data, norm)?;
            let flat = src.as_slice();
            constrained_labels(t, flat, n, dim, must, cannot, k, mi, seed)?
        } else {
            return Err(PyValueError::new_err("no data was fitted"));
        };
        self.labels = Some(labels);
        self.proba = None;
        Ok(())
    }

    /// Route the rows of `data` to their leaves, with the GIL released during compute.
    fn route_data<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let labels = self.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "call fit(), fit_predict(), or partial_fit() (no args, to finalize) before predict()",
            )
        })?;
        let rule = self.rule.as_ref();
        if let Some(t) = &self.state64 {
            let (src, n, dim) = flat_as::<f64>(data, self.row_prep())?;
            let flat = src.as_slice();
            self.check_dim(dim)?;
            Ok(py
                .detach(|| match rule {
                    Some(r) => r.label_rows(flat, n, dim),
                    None => t.route(labels, flat, n, dim),
                })
                .into_pyarray(py))
        } else if let Some(t) = &self.state32 {
            let (src, n, dim) = flat_as::<f32>(data, self.row_prep())?;
            let flat = src.as_slice();
            self.check_dim(dim)?;
            Ok(py
                .detach(|| match rule {
                    Some(r) => r.label_rows(flat, n, dim),
                    None => t.route(labels, flat, n, dim),
                })
                .into_pyarray(py))
        } else {
            Err(PyValueError::new_err(
                "call fit() or fit_predict() before predict()",
            ))
        }
    }

    /// Point posteriors for one dtype: the fitted mixture's own if the head has one, else the row's
    /// microcluster responsibilities (all a projected fit can offer, since the head clustered codes).
    fn proba_of<R: Real + Element>(
        &self,
        py: Python<'_>,
        tree: &TreeState<R>,
        flat: &[R],
        n: usize,
        dim: usize,
    ) -> PyResult<(Vec<f64>, usize)> {
        if let Some(rule) = self.rule.as_ref() {
            if let Some(out) = py.detach(|| rule.proba_rows(flat, n, dim)) {
                return Ok(out);
            }
        }
        let (leaf, k) = self.proba.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "predict_proba is only available after fit with method='gmm', 'gmm-full', 'mppca', 'mfa', 'vmf', 'watson', 'gmm-toeplitz', 'gmm-toeplitz-full', 'gmm-toeplitz-gs' or 'fuzzy-cmeans'",
            )
        })?;
        let idx = py.detach(|| tree.assign_microclusters(flat, n, dim));
        let mut out = Vec::with_capacity(n * k);
        for &i in &idx {
            let lo = i as usize * k;
            out.extend_from_slice(&leaf[lo..lo + k]);
        }
        Ok((out, *k))
    }

    /// Per-leaf stats from whichever dtype tree exists; errors if no data has been fitted.
    fn leaf_stats_any(&self) -> PyResult<F64Stats> {
        if let Some(t) = &self.state64 {
            Ok(t.leaf_stats())
        } else if let Some(t) = &self.state32 {
            Ok(t.leaf_stats())
        } else {
            Err(PyValueError::new_err(
                "call fit() or partial_fit() before inspecting microclusters",
            ))
        }
    }

    /// Pooled per-cluster stats; errors if the clustering has not been finalized.
    fn cluster_stats_any(&self) -> PyResult<F64Stats> {
        let labels = self.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "finalize first (fit / fit_predict / partial_fit with no args) before inspecting clusters",
            )
        })?;
        let k = cluster_count_for_centers(labels);
        if let Some(t) = &self.state64 {
            Ok(t.cluster_stats(labels, k))
        } else if let Some(t) = &self.state32 {
            Ok(t.cluster_stats(labels, k))
        } else {
            Err(PyValueError::new_err("not fitted"))
        }
    }

    /// Sensitivity-sampling coreset over the tree's leaves. Needs only a built tree, not a
    /// finalized head: the guarantee is over candidate solutions, so it does not depend on which
    /// one this estimator happens to have fitted.
    fn coreset_any(&self, k: usize, size: usize, seed: u64) -> PyResult<CoresetOut> {
        if let Some(t) = &self.state64 {
            Ok(t.coreset(k, size, seed))
        } else if let Some(t) = &self.state32 {
            Ok(t.coreset(k, size, seed))
        } else {
            Err(PyValueError::new_err("not fitted"))
        }
    }

    /// MMD of the leaf summary against `sample`. Needs only a built tree, not a finalized head:
    /// the number is a property of the summary, not of any clustering of it.
    fn summary_mmd_any(&self, flat: &[f64], dim: usize, bandwidth: Option<f64>) -> PyResult<f64> {
        let want = if let Some(t) = &self.state64 {
            t.leaf_dim()
        } else if let Some(t) = &self.state32 {
            t.leaf_dim()
        } else {
            return Err(PyValueError::new_err("not fitted"));
        };
        if dim != want {
            return Err(PyValueError::new_err(format!(
                "sample has {dim} columns but the summary is {want}-dimensional"
            )));
        }
        if let Some(t) = &self.state64 {
            Ok(t.summary_mmd(flat, bandwidth))
        } else if let Some(t) = &self.state32 {
            let narrowed: Vec<f32> = flat.iter().map(|&v| v as f32).collect();
            Ok(t.summary_mmd(&narrowed, bandwidth.map(|h| h as f32)))
        } else {
            Err(PyValueError::new_err("not fitted"))
        }
    }

    /// The three internal validity indices; errors if the clustering has not been finalized.
    fn validity_any(&self) -> PyResult<(f64, f64, f64)> {
        let labels = self.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "finalize first (fit / fit_predict / partial_fit with no args) before scoring clusters",
            )
        })?;
        let k = cluster_count_for_centers(labels);
        if let Some(t) = &self.state64 {
            Ok(t.validity(labels, k))
        } else if let Some(t) = &self.state32 {
            Ok(t.validity(labels, k))
        } else {
            Err(PyValueError::new_err("not fitted"))
        }
    }
}

#[pymethods]
impl Betula {
    #[new]
    #[pyo3(signature = (
        n_clusters = 8, feature = "diagonal", method = "gmm", threshold = 0.0,
        branching = 32, leaf_cap = 32, max_leaves = 2000, max_iter = 100,
        min_samples = 5, min_cluster_size = 5, seed = 0,
        distance = "euclidean", absorb = "euclidean", chi2_p = 0.95, chi2_scale = 0.0, decay = 1.0,
        normalize = false, huber_k = None, resolution = 1.0, covariance_weight = 0.0,
        tangent_weight = 0.0, tangent_rank = 2, projection = "none", projection_dim = 64,
        projection_max_iter = 100, refine = 0, rank = 2, graph_degree = 0, balance = None,
        auto_k_max = 0, fuzzifier = 2.0, leaf_refit = 0, canonical_order = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        feature: &str,
        method: &str,
        threshold: f64,
        branching: usize,
        leaf_cap: usize,
        max_leaves: usize,
        max_iter: usize,
        min_samples: usize,
        min_cluster_size: usize,
        seed: u64,
        distance: &str,
        absorb: &str,
        chi2_p: f64,
        chi2_scale: f64,
        decay: f64,
        normalize: bool,
        huber_k: Option<f64>,
        resolution: f64,
        covariance_weight: f64,
        tangent_weight: f64,
        tangent_rank: usize,
        projection: &str,
        projection_dim: usize,
        projection_max_iter: usize,
        refine: usize,
        rank: usize,
        graph_degree: usize,
        balance: Option<f64>,
        auto_k_max: usize,
        fuzzifier: f64,
        leaf_refit: usize,
        canonical_order: bool,
    ) -> PyResult<Self> {
        let kind = parse_method(
            method,
            min_samples,
            min_cluster_size,
            resolution,
            covariance_weight,
            tangent_weight,
            tangent_rank,
            rank,
            graph_degree,
            fuzzifier,
        )?;
        let proj = parse_projection(projection, projection_dim, projection_max_iter)?;
        let nmf_dim = proj.map(|p| p.rank);
        let nmf_kl = matches!(
            proj.map(|p| p.kind),
            Some(ProjectionKind::Nmf { kl: true, .. })
        );
        let projection_svd = matches!(proj.map(|p| p.kind), Some(ProjectionKind::Svd));
        let nmf_max_iter = match proj.map(|p| p.kind) {
            Some(ProjectionKind::Nmf { max_iter, .. }) => max_iter,
            _ => default_projection_max_iter(),
        };
        let route = parse_route(distance)?;
        if !matches!(feature, "spherical" | "diagonal" | "full" | "fd") {
            return Err(PyValueError::new_err(
                "feature must be 'spherical', 'diagonal', 'full' or 'fd'",
            ));
        }
        // The estimator rejects a bad `absorb` at construction, before any data has fixed `dim`, but
        // the accepted names and the χ²-scale rule belong to `resolve_gate`. Ask it with a
        // placeholder `dim` and drop the gate it builds, rather than restating them here where the
        // two lists had already drifted apart.
        resolve_gate::<f64>(absorb, 1, chi2_p, chi2_scale, threshold)
            .map_err(PyValueError::new_err)?;
        if let Some(k) = huber_k {
            if k <= 0.0 || k.is_nan() {
                return Err(PyValueError::new_err(
                    "huber_k must be > 0 (per-dimension std multiplier), or None to disable",
                ));
            }
        }
        if let Some(b) = balance {
            if b <= 0.0 || b.is_nan() {
                return Err(PyValueError::new_err(
                    "balance must be > 0 (multiple of the n / max_leaves ideal), or None to disable",
                ));
            }
        }
        Ok(Self {
            feature: feature.to_string(),
            kind,
            route,
            method: method.to_string(),
            distance: distance.to_string(),
            min_samples,
            min_cluster_size,
            n_clusters,
            threshold,
            branching,
            leaf_cap,
            max_leaves,
            max_iter,
            auto_k_max,
            seed,
            absorb: absorb.to_string(),
            chi2_p,
            chi2_scale,
            decay,
            normalize,
            huber_k,
            balance,
            resolution,
            covariance_weight,
            tangent_weight,
            tangent_rank,
            rank,
            fuzzifier,
            graph_degree,
            nmf_dim,
            nmf_kl,
            nmf_max_iter,
            projection_svd,
            dim: 0,
            state64: None,
            state32: None,
            labels: None,
            proba: None,
            nmf_components: None,
            nmf_reconstruction_err: None,
            rule: None,
            refine,
            leaf_refit,
            canonical_order,
        })
    }

    /// Stream a chunk of rows into the tree (`data` given) without re-clustering, or run the global
    /// clustering over everything accumulated so far (`data=None`). Mirrors scikit-learn's Birch:
    /// `partial_fit(X)` adds data, a final `partial_fit()` finalizes. Returns `self`.
    #[pyo3(signature = (data = None))]
    fn partial_fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        match data {
            Some(data) => slf.stream(data, Arrival::Chunk)?,
            None => {
                if slf.state64.is_none() && slf.state32.is_none() {
                    return Err(PyValueError::new_err(
                        "partial_fit() with no data before any rows were added",
                    ));
                }
                let py = slf.py();
                slf.finalize(py)?;
            }
        }
        Ok(slf)
    }

    /// Reset, build the tree from `data`, and cluster its leaves. Returns `self`.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.reset();
        slf.stream(data, Arrival::Whole)?;
        let py = slf.py();
        slf.refit_tree(py, data)?;
        slf.finalize(py)?;
        slf.refine_rule(py, data)?;
        Ok(slf)
    }

    /// Label new rows by their nearest leaf (requires a prior `fit` / `fit_predict`).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.route_data(py, data)
    }

    /// Per-row posterior `p(c | x)` — `(n, k)`, columns aligned with `predict`, so
    /// `predict_proba(X).argmax(1) == predict(X)`. A mixture head scores the point itself; when a
    /// projection replaced the space the head clustered, the row's microcluster responsibilities are
    /// the closest defined answer. Raises for a head with no posterior at all.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (flat, k) = if let Some(t) = &self.state64 {
            let (src, n, dim) = flat_as::<f64>(data, self.row_prep())?;
            let rows = src.as_slice();
            self.check_dim(dim)?;
            self.proba_of(py, t, rows, n, dim)?
        } else if let Some(t) = &self.state32 {
            let (src, n, dim) = flat_as::<f32>(data, self.row_prep())?;
            let rows = src.as_slice();
            self.check_dim(dim)?;
            self.proba_of(py, t, rows, n, dim)?
        } else {
            return Err(PyValueError::new_err(
                "call fit() or fit_predict() before predict_proba()",
            ));
        };
        let rows = flat.len().checked_div(k).unwrap_or(0);
        Ok(Array2::from_shape_vec((rows, k), flat)
            .expect("proba length is rows*k")
            .into_pyarray(py))
    }

    /// Reset, fit on `data`, and return the training labels in one call.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.reset();
        self.stream(data, Arrival::Whole)?;
        self.refit_tree(py, data)?;
        self.finalize(py)?;
        self.refine_rule(py, data)?;
        self.route_data(py, data)
    }

    /// Reset, build the tree from a dense `data`, and cluster its leaves under pairwise constraints
    /// (`must_link` / `cannot_link` are `(m, 2)` int arrays of *row* indices into `data`). COP-KMeans
    /// only; the wrapper enforces `method="kmeans"`. Returns `self`.
    fn fit_constrained<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
        must_link: PyReadonlyArray2<'py, i64>,
        cannot_link: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if slf.n_clusters == 0 {
            return Err(PyValueError::new_err(
                "constrained clustering requires n_clusters >= 1 (auto-k is not supported)",
            ));
        }
        if !matches!(slf.kind, Kind::Parametric(Method::KMeans)) {
            return Err(PyValueError::new_err(
                "constraints are only supported with method='kmeans'",
            ));
        }
        let must = pairs_from(&must_link)?;
        let cannot = pairs_from(&cannot_link)?;
        slf.reset();
        slf.stream(data, Arrival::Whole)?;
        slf.finalize_constrained(data, &must, &cannot)?;
        Ok(slf)
    }

    // ── sparse CSR entry points (the `betula_cluster.Betula` wrapper routes scipy.sparse here) ──
    fn partial_fit_csr(
        &mut self,
        data: PyReadonlyArray1<'_, f64>,
        indices: PyReadonlyArray1<'_, i64>,
        indptr: PyReadonlyArray1<'_, i64>,
        n_features: usize,
    ) -> PyResult<()> {
        self.stream_csr(
            data.as_slice()?,
            indices.as_slice()?,
            indptr.as_slice()?,
            n_features,
            Arrival::Chunk,
        )
    }

    fn fit_csr(
        &mut self,
        py: Python<'_>,
        data: PyReadonlyArray1<'_, f64>,
        indices: PyReadonlyArray1<'_, i64>,
        indptr: PyReadonlyArray1<'_, i64>,
        n_features: usize,
    ) -> PyResult<()> {
        self.reset();
        self.stream_csr(
            data.as_slice()?,
            indices.as_slice()?,
            indptr.as_slice()?,
            n_features,
            Arrival::Whole,
        )?;
        self.finalize(py)
    }

    fn fit_predict_csr<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray1<'_, f64>,
        indices: PyReadonlyArray1<'_, i64>,
        indptr: PyReadonlyArray1<'_, i64>,
        n_features: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (d, idx, ip) = (data.as_slice()?, indices.as_slice()?, indptr.as_slice()?);
        self.reset();
        self.stream_csr(d, idx, ip, n_features, Arrival::Whole)?;
        self.finalize(py)?;
        Ok(self
            .route_csr_labels(d, idx, ip, n_features)?
            .into_pyarray(py))
    }

    fn predict_csr<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'_, f64>,
        indices: PyReadonlyArray1<'_, i64>,
        indptr: PyReadonlyArray1<'_, i64>,
        n_features: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let labels = self.route_csr_labels(
            data.as_slice()?,
            indices.as_slice()?,
            indptr.as_slice()?,
            n_features,
        )?;
        Ok(labels.into_pyarray(py))
    }

    /// Number of clusters found (distinct non-noise labels); `0` before fitting.
    #[getter]
    fn n_clusters_(&self) -> usize {
        match &self.labels {
            Some(l) => {
                let mut v: Vec<i64> = l.iter().copied().filter(|&x| x >= 0).collect();
                v.sort_unstable();
                v.dedup();
                v.len()
            }
            None => 0,
        }
    }

    /// Number of leaf micro-clusters currently in the tree.
    #[getter]
    fn n_leaves_(&self) -> usize {
        self.state64
            .as_ref()
            .map(|t| t.num_leaves())
            .or_else(|| self.state32.as_ref().map(|t| t.num_leaves()))
            .unwrap_or(0)
    }

    /// Number of times the CF-tree rebuilt (threshold-grew) under the leaf bound — high values mean
    /// the tree thrashed; raise `max_leaves` or `threshold`.
    #[getter]
    fn n_rebuilds_(&self) -> usize {
        self.state64
            .as_ref()
            .map(|t| t.rebuilds())
            .or_else(|| self.state32.as_ref().map(|t| t.rebuilds()))
            .unwrap_or(0)
    }

    /// Current absorption threshold of the CF-tree (grows as it rebuilds).
    #[getter]
    fn threshold_(&self) -> f64 {
        self.state64
            .as_ref()
            .map(|t| t.threshold())
            .or_else(|| self.state32.as_ref().map(|t| t.threshold()))
            .unwrap_or(0.0)
    }

    /// Microcluster (leaf) centroids — `(n_microclusters, dim)`.
    #[getter]
    fn microcluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (centers, _w, _r, dim) = self.leaf_stats_any()?;
        let rows = centers.len().checked_div(dim).unwrap_or(0);
        Ok(Array2::from_shape_vec((rows, dim), centers)
            .expect("centers length is rows*dim")
            .into_pyarray(py))
    }

    /// Microcluster effective point mass — `(n_microclusters,)`.
    #[getter]
    fn microcluster_weights_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.leaf_stats_any()?.1.into_pyarray(py))
    }

    /// Microcluster RMS radius `sqrt(ssd / weight)` — `(n_microclusters,)`.
    #[getter]
    fn microcluster_radii_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.leaf_stats_any()?.2.into_pyarray(py))
    }

    /// Per-microcluster GMM soft responsibilities — `(n_microclusters, k)`. Only the GMM heads have a
    /// posterior; raises otherwise. This is a *leaf-level* quantity: `predict_proba` scores the point
    /// itself, and only falls back to these rows when a projection replaced the head's feature space.
    #[getter]
    fn microcluster_proba_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (flat, k) = self.proba.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "predict_proba posterior is only available after fit with method='gmm', 'gmm-full', 'mppca', 'mfa', 'vmf', 'watson', 'gmm-toeplitz', 'gmm-toeplitz-full' or 'gmm-toeplitz-gs'",
            )
        })?;
        let rows = flat.len().checked_div(*k).unwrap_or(0);
        Ok(Array2::from_shape_vec((rows, *k), flat.clone())
            .expect("proba length is n_leaves*k")
            .into_pyarray(py))
    }

    /// The NMF parts `H` — `(rank, dim)`, rows unit-L2 and ordered by descending energy. Every leaf
    /// code is a nonnegative combination of these, so a row reads directly as a "topic" over features.
    /// Requires `projection="weighted-nmf"` / `"weighted-nmf-kl"`.
    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let h = self.nmf_components.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "components_ is only available after fit with a projection ('weighted-nmf', 'weighted-nmf-kl' or 'svd')",
            )
        })?;
        let (r, d) = (h.len(), h.first().map_or(0, |row| row.len()));
        Ok(
            Array2::from_shape_vec((r, d), h.iter().flatten().copied().collect())
                .expect("components are r rows of d")
                .into_pyarray(py),
        )
    }

    /// Relative reconstruction error of the projection, `‖X̃ − W H‖_F / ‖X̃‖_F` over the leaf centroid
    /// matrix — how much of the compressed data the `rank` parts actually explain. Requires a projection.
    #[getter]
    fn reconstruction_err_(&self) -> PyResult<f64> {
        self.nmf_reconstruction_err.ok_or_else(|| {
            PyValueError::new_err(
                "reconstruction_err_ is only available after fit with a projection ('weighted-nmf', 'weighted-nmf-kl' or 'svd')",
            )
        })
    }

    /// Macro-cluster centroids — `(n_clusters, dim)`; requires a finalized clustering. These are the
    /// Phase-3 summary centroids, the mass-weighted mean of each cluster's leaves. `refine` moves the
    /// centres `predict` scores against, not these: a Phase-4 sweep optimizes over raw points, so
    /// pooling its result back into the summary would make the paired `cluster_radii_` /
    /// `cluster_sizes_` describe a partition neither of them was computed under.
    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (centers, _r, _w, dim) = self.cluster_stats_any()?;
        let rows = centers.len().checked_div(dim).unwrap_or(0);
        Ok(Array2::from_shape_vec((rows, dim), centers)
            .expect("centers length is rows*dim")
            .into_pyarray(py))
    }

    /// Macro-cluster RMS radius — `(n_clusters,)`; requires a finalized clustering.
    #[getter]
    fn cluster_radii_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.cluster_stats_any()?.1.into_pyarray(py))
    }

    /// Macro-cluster total point mass — `(n_clusters,)`; requires a finalized clustering.
    #[getter]
    fn cluster_sizes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.cluster_stats_any()?.2.into_pyarray(py))
    }

    /// `(calinski_harabasz, davies_bouldin, medoid_silhouette)` over the leaf summary; requires a
    /// finalized clustering.
    fn validity_(&self) -> PyResult<(f64, f64, f64)> {
        self.validity_any()
    }

    /// Gaussian-kernel MMD between the leaf summary and `sample`; requires a built tree only.
    #[pyo3(signature = (sample, bandwidth=None))]
    fn summary_mmd_(
        &self,
        sample: PyReadonlyArray2<'_, f64>,
        bandwidth: Option<f64>,
    ) -> PyResult<f64> {
        let (src, _n, dim) = to_rows(sample, RowPrep::None)?;
        self.summary_mmd_any(src.as_slice(), dim, bandwidth)
    }

    /// `(points, weights, offset, reference_cost, total_sensitivity, n_leaves, radii)` for a
    /// sensitivity-sampling coreset of `size` leaves aimed at `k` centres.
    #[pyo3(signature = (k, size, seed))]
    fn export_coreset_<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        size: usize,
        seed: u64,
    ) -> PyResult<CoresetPy<'py>> {
        let out = self.coreset_any(k, size, seed)?;
        let rows = out.points.len().checked_div(out.dim).unwrap_or(0);
        let points = Array2::from_shape_vec((rows, out.dim), out.points)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            points.into_pyarray(py),
            PyArray1::from_vec(py, out.weights),
            out.offset,
            out.reference_cost,
            out.total_sensitivity,
            out.n_leaves,
            PyArray1::from_vec(py, out.radii),
        ))
    }

    /// Per-row outlier score: deviation from the assigned cluster centroid, normalized either by the
    /// cluster's scalar RMS radius (`metric="radius"`) or by its pooled per-dimension variance
    /// (`metric="mahalanobis"`). Rows routed to HDBSCAN noise score `+inf`.
    #[pyo3(signature = (data, metric="radius"))]
    fn outlier_scores<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        metric: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let labels = self.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "finalize first (fit / fit_predict / partial_fit with no args) before outlier_scores()",
            )
        })?;
        let whiten = match metric {
            "radius" => false,
            "mahalanobis" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "metric must be 'radius' or 'mahalanobis', got '{other}'"
                )));
            }
        };
        let k = cluster_count_for_centers(labels);
        if let Some(t) = &self.state64 {
            let (centers, radii, _w, _d) = t.cluster_stats(labels, k);
            let chols = if whiten {
                t.cluster_chol(labels, k, &centers)
            } else {
                Vec::new()
            };
            let scale = if whiten {
                OutlierScale::Whitened(&chols)
            } else {
                OutlierScale::Radius(&radii)
            };
            let (src, n, dim) = flat_as::<f64>(data, self.row_prep())?;
            let flat = src.as_slice();
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.outlier_scores(labels, &centers, scale, flat, n, dim))
                .into_pyarray(py))
        } else if let Some(t) = &self.state32 {
            let (centers, radii, _w, _d) = t.cluster_stats(labels, k);
            let chols = if whiten {
                t.cluster_chol(labels, k, &centers)
            } else {
                Vec::new()
            };
            let scale = if whiten {
                OutlierScale::Whitened(&chols)
            } else {
                OutlierScale::Radius(&radii)
            };
            let (src, n, dim) = flat_as::<f32>(data, self.row_prep())?;
            let flat = src.as_slice();
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.outlier_scores(labels, &centers, scale, flat, n, dim))
                .into_pyarray(py))
        } else {
            Err(PyValueError::new_err("call fit() before outlier_scores()"))
        }
    }

    /// Per-row nearest microcluster (leaf) index, aligned with `microcluster_centers_`.
    fn assign_microclusters<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        if let Some(t) = &self.state64 {
            let (src, n, dim) = flat_as::<f64>(data, self.row_prep())?;
            let flat = src.as_slice();
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.assign_microclusters(flat, n, dim))
                .into_pyarray(py))
        } else if let Some(t) = &self.state32 {
            let (src, n, dim) = flat_as::<f32>(data, self.row_prep())?;
            let flat = src.as_slice();
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.assign_microclusters(flat, n, dim))
                .into_pyarray(py))
        } else {
            Err(PyValueError::new_err(
                "call fit() or partial_fit() before assign_microclusters()",
            ))
        }
    }

    /// OPTICS reachability plot over the leaf microclusters, as a dict of arrays.
    ///
    /// Returns `order` (leaf indices in sweep order), `reachability` (aligned with `order`, the
    /// first entry `inf`), `core_distances` and `weights` (both in leaf indexing). `min_samples`
    /// and `graph_degree` mean what they do for `method="hdbscan"`, and the plot is a readout of
    /// the same mutual-reachability spanning tree that head builds its hierarchy from.
    #[pyo3(signature = (min_samples = 5, graph_degree = 0))]
    fn reachability<'py>(
        &self,
        py: Python<'py>,
        min_samples: usize,
        graph_degree: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let plot = match (&self.state64, &self.state32) {
            (Some(t), _) => py.detach(|| t.reachability(min_samples, graph_degree, self.seed)),
            (_, Some(t)) => py.detach(|| t.reachability(min_samples, graph_degree, self.seed)),
            _ => {
                return Err(PyValueError::new_err(
                    "call fit() or partial_fit() before reachability()",
                ));
            }
        };
        let d = pyo3::types::PyDict::new(py);
        let order: Vec<i64> = plot.order.iter().map(|&i| i as i64).collect();
        d.set_item("order", Array1::from(order).into_pyarray(py))?;
        d.set_item(
            "reachability",
            Array1::from(plot.reachability).into_pyarray(py),
        )?;
        d.set_item("core_distances", Array1::from(plot.core).into_pyarray(py))?;
        d.set_item("weights", Array1::from(plot.mass).into_pyarray(py))?;
        Ok(d)
    }

    /// Mapper topological-skeleton graph over the leaf microclusters, as a dict of arrays.
    ///
    /// `lens` selects the filter function (`density` / `radius` / `l2norm` / `coordinate` /
    /// `eccentricity`); `resolution` × `gain` set the overlapping cover; `link_scale` the within-bin
    /// single-linkage scale (× the median NN gap); nodes lighter than `min_node_mass` are dropped. Returns node
    /// members / mass / bin / lens / centroids, weighted `edges`, `branch_points` and `bridges`.
    #[pyo3(signature = (lens = "density", resolution = 10, gain = 0.3, link_scale = 1.0,
                        min_node_mass = 0.0, density_k = 5, coordinate = 0,
                        link = "centroid"))]
    #[allow(clippy::too_many_arguments)]
    fn mapper<'py>(
        &self,
        py: Python<'py>,
        lens: &str,
        resolution: usize,
        gain: f64,
        link_scale: f64,
        min_node_mass: f64,
        density_k: usize,
        coordinate: usize,
        link: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let lens = match lens {
            "density" => Lens::Density { k: density_k },
            "radius" => Lens::Radius,
            "l2norm" | "l2" => Lens::L2Norm,
            "coordinate" | "coord" => {
                if self.dim != 0 && coordinate >= self.dim {
                    return Err(PyValueError::new_err(
                        "coordinate index out of range for the fitted dimensionality",
                    ));
                }
                Lens::Coordinate(coordinate)
            }
            "eccentricity" | "ecc" => Lens::Eccentricity,
            _ => {
                return Err(PyValueError::new_err(
                    "lens must be 'density', 'radius', 'l2norm', 'coordinate' or 'eccentricity'",
                ));
            }
        };
        let link = match link {
            "centroid" => Link::Centroid,
            "bhattacharyya" => Link::Bhattacharyya,
            _ => {
                return Err(PyValueError::new_err(
                    "link must be 'centroid' or 'bhattacharyya'",
                ));
            }
        };
        let p = MapperParams {
            lens,
            resolution,
            gain,
            link_scale,
            min_node_mass,
            link,
        };
        let g = match (&self.state64, &self.state32) {
            (Some(t), _) => py.detach(|| t.mapper(&p)),
            (_, Some(t)) => py.detach(|| t.mapper(&p)),
            _ => {
                return Err(PyValueError::new_err(
                    "call fit() or partial_fit() before mapper()",
                ));
            }
        };

        let n_nodes = g.nodes.len();
        let dim = g
            .nodes
            .first()
            .map(|n| n.centroid.len())
            .unwrap_or(self.dim);
        let members: Vec<Vec<i64>> = g
            .nodes
            .iter()
            .map(|n| n.members.iter().map(|&i| i as i64).collect())
            .collect();
        let mass: Vec<f64> = g.nodes.iter().map(|n| n.mass).collect();
        let bin: Vec<i64> = g.nodes.iter().map(|n| n.bin as i64).collect();
        let lens_val: Vec<f64> = g.nodes.iter().map(|n| n.lens_value).collect();
        let mut centroids = vec![0.0f64; n_nodes * dim];
        for (r, node) in g.nodes.iter().enumerate() {
            centroids[r * dim..r * dim + node.centroid.len()].copy_from_slice(&node.centroid);
        }
        let mut edges = vec![0i64; g.edges.len() * 3];
        for (r, &(a, b, w)) in g.edges.iter().enumerate() {
            edges[r * 3] = a as i64;
            edges[r * 3 + 1] = b as i64;
            edges[r * 3 + 2] = w as i64;
        }
        let branch_points: Vec<i64> = g.branch_points.iter().map(|&i| i as i64).collect();
        let bridges: Vec<i64> = g.bridges.iter().map(|&i| i as i64).collect();

        let d = pyo3::types::PyDict::new(py);
        d.set_item("node_members", members)?;
        d.set_item("node_mass", mass.into_pyarray(py))?;
        d.set_item("node_bin", bin.into_pyarray(py))?;
        d.set_item("node_lens", lens_val.into_pyarray(py))?;
        d.set_item(
            "node_centroids",
            Array2::from_shape_vec((n_nodes, dim), centroids)
                .expect("centroids length is n_nodes*dim")
                .into_pyarray(py),
        )?;
        d.set_item(
            "edges",
            Array2::from_shape_vec((g.edges.len(), 3), edges)
                .expect("edges length is n_edges*3")
                .into_pyarray(py),
        )?;
        d.set_item("branch_points", branch_points.into_pyarray(py))?;
        d.set_item("bridges", bridges.into_pyarray(py))?;
        d.set_item("edge_overlap", g.edge_overlap.clone().into_pyarray(py))?;
        // 0-D persistence diagrams of the nerve, both filtrations: (k, 2) births/deaths with `inf` in
        // the death column for essential (connected-component) classes. Cheap (O(E log E)) over the graph.
        for (key, filt) in [
            (
                "persistence_overlap",
                crate::topology::Filtration::EdgeOverlap,
            ),
            ("persistence_lens", crate::topology::Filtration::Lens),
        ] {
            let diag = g.persistence_diagram(filt);
            let k = diag.points.len();
            let mut flat = vec![0.0f64; k * 2];
            for (r, &(b, dth)) in diag.points.iter().enumerate() {
                flat[r * 2] = b;
                flat[r * 2 + 1] = dth;
            }
            d.set_item(
                key,
                Array2::from_shape_vec((k, 2), flat)
                    .expect("persistence points length is k*2")
                    .into_pyarray(py),
            )?;
        }
        Ok(d)
    }

    /// Construction parameters as a dict. Internal: the `betula_cluster.Betula` Python wrapper reads
    /// this to recover the parameter set after `load`, exposing the scikit-learn `get_params` itself.
    #[pyo3(signature = (deep = true))]
    fn get_params<'py>(
        &self,
        py: Python<'py>,
        deep: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let _ = deep; // no nested estimators
        let d = pyo3::types::PyDict::new(py);
        d.set_item("n_clusters", self.n_clusters)?;
        d.set_item("feature", &self.feature)?;
        d.set_item("method", &self.method)?;
        d.set_item("threshold", self.threshold)?;
        d.set_item("branching", self.branching)?;
        d.set_item("leaf_cap", self.leaf_cap)?;
        d.set_item("max_leaves", self.max_leaves)?;
        d.set_item("max_iter", self.max_iter)?;
        d.set_item("min_samples", self.min_samples)?;
        d.set_item("min_cluster_size", self.min_cluster_size)?;
        d.set_item("seed", self.seed)?;
        d.set_item("distance", &self.distance)?;
        d.set_item("absorb", &self.absorb)?;
        d.set_item("chi2_p", self.chi2_p)?;
        d.set_item("chi2_scale", self.chi2_scale)?;
        d.set_item("decay", self.decay)?;
        d.set_item("normalize", self.normalize)?;
        d.set_item("huber_k", self.huber_k)?;
        d.set_item("balance", self.balance)?;
        d.set_item("resolution", self.resolution)?;
        d.set_item("covariance_weight", self.covariance_weight)?;
        d.set_item("tangent_weight", self.tangent_weight)?;
        d.set_item("tangent_rank", self.tangent_rank)?;
        d.set_item("rank", self.rank)?;
        d.set_item("fuzzifier", self.fuzzifier)?;
        d.set_item("graph_degree", self.graph_degree)?;
        d.set_item("auto_k_max", self.auto_k_max)?;
        d.set_item("refine", self.refine)?;
        d.set_item("leaf_refit", self.leaf_refit)?;
        d.set_item("canonical_order", self.canonical_order)?;
        Ok(d)
    }

    /// Save the (fitted or partial) estimator to a file — bincode, version-tagged.
    fn save(&self, path: &str) -> PyResult<()> {
        let bytes = encode(self)?;
        std::fs::write(path, bytes).map_err(|e| PyValueError::new_err(format!("write failed: {e}")))
    }

    /// Load an estimator previously written with [`Betula::save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes =
            std::fs::read(path).map_err(|e| PyValueError::new_err(format!("read failed: {e}")))?;
        decode(&bytes)
    }

    /// Pickle support: serialize the estimator state to bytes.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &encode(self)?))
    }

    /// Pickle support: restore the estimator state from bytes.
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        *self = decode(state)?;
        Ok(())
    }

    /// Pickle support: reconstruct via the default constructor, then `__setstate__`.
    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::empty(py)
    }
}

/// On-disk schema version; bump on any breaking change to the serialized layout.
const SCHEMA_VERSION: u32 = 2;

/// Serialize an estimator with its schema version prepended (CBOR via `ciborium`, a compact,
/// maintained serde format).
fn encode(est: &Betula) -> PyResult<Vec<u8>> {
    let mut buf = Vec::new();
    ciborium::into_writer(&(SCHEMA_VERSION, est), &mut buf)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {e}")))?;
    Ok(buf)
}

/// Deserialize an estimator, rejecting an unknown schema version.
fn decode(bytes: &[u8]) -> PyResult<Betula> {
    let (version, est): (u32, Betula) = ciborium::from_reader(bytes)
        .map_err(|e| PyValueError::new_err(format!("deserialize failed: {e}")))?;
    if version != SCHEMA_VERSION {
        return Err(PyValueError::new_err(format!(
            "unsupported model version {version} (this build expects {SCHEMA_VERSION})"
        )));
    }
    Ok(est)
}

/// Streaming **windowed** clusterer: a CF-tree per time frame, and window queries answered by
/// summing frames rather than by subtracting snapshots (see `crate::window` for the measurement
/// that rules the subtraction out). Spherical micro-clusters, `f64`.
#[pyclass(name = "WindowStream", module = "betula_cluster._core")]
struct PyWindowStream {
    frame_width: f64,
    capacity: usize,
    max_micros: usize,
    threshold: f64,
    max_leaves: usize,
    branching: usize,
    leaf_cap: usize,
    seed: u64,
    inner: Option<WindowStream<f64, Spherical<f64>>>,
}

#[pymethods]
impl PyWindowStream {
    #[new]
    #[pyo3(signature = (
        frame_width = 1.0, capacity = 64, max_micros = 200, threshold = 0.0,
        max_leaves = 2000, branching = 32, leaf_cap = 32, seed = 0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        frame_width: f64,
        capacity: usize,
        max_micros: usize,
        threshold: f64,
        max_leaves: usize,
        branching: usize,
        leaf_cap: usize,
        seed: u64,
    ) -> Self {
        Self {
            frame_width,
            capacity,
            max_micros,
            threshold,
            max_leaves,
            branching,
            leaf_cap,
            seed,
            inner: None,
        }
    }

    /// Construction params as a dict (read by the Python wrapper's scikit-learn `get_params`).
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("frame_width", self.frame_width)?;
        d.set_item("capacity", self.capacity)?;
        d.set_item("max_micros", self.max_micros)?;
        d.set_item("threshold", self.threshold)?;
        d.set_item("max_leaves", self.max_leaves)?;
        d.set_item("branching", self.branching)?;
        d.set_item("leaf_cap", self.leaf_cap)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Stream a chunk of points with their timestamps (one per row).
    fn partial_fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        times: Vec<f64>,
    ) -> PyResult<()> {
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        if times.len() != n {
            return Err(PyValueError::new_err(
                "times must carry one timestamp per row of data",
            ));
        }
        if let Some(ws) = &self.inner {
            if ws.dim() != dim {
                return Err(PyValueError::new_err(
                    "dimension mismatch with previously streamed data",
                ));
            }
        }
        if self.inner.is_none() {
            self.inner = Some(WindowStream::new(
                dim,
                self.branching,
                self.leaf_cap,
                self.threshold,
                self.max_leaves,
                self.frame_width,
                self.capacity,
                self.max_micros,
                self.seed,
            ));
        }
        let ws = self.inner.as_mut().unwrap();
        py.detach(|| {
            for (i, &t) in times.iter().enumerate() {
                ws.insert(&flat[i * dim..(i + 1) * dim], t);
            }
        });
        Ok(())
    }

    /// Close the frame currently being filled. A no-op if nothing is open.
    fn close_frame(&mut self) {
        if let Some(ws) = &mut self.inner {
            ws.close_frame();
        }
    }

    /// Number of closed frames retained.
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.as_ref().map_or(0, |ws| ws.frames().len())
    }

    /// `(t_min, t_max, weight)` per closed frame, oldest first.
    fn frame_spans(&self) -> Vec<(f64, f64, f64)> {
        self.inner.as_ref().map_or_else(Vec::new, |ws| {
            ws.frames()
                .iter()
                .map(|f| (f.span.min, f.span.max, f.span.weight))
                .collect()
        })
    }

    /// `(weight, mean, ssd)` of the closed frames reaching into `[t0, t1]`.
    fn window_moments(&self, t0: f64, t1: f64) -> (f64, Vec<f64>, f64) {
        match &self.inner {
            Some(ws) => {
                let m = ws.window_moments(t0, t1);
                (m.weight, m.mean, m.ssd)
            }
            None => (0.0, Vec::new(), 0.0),
        }
    }

    /// Cluster the window into `k` groups: `(centers, cluster_masses, inertia)`.
    /// `None` when the window holds fewer than `k` micro-clusters.
    fn cluster_window(
        &self,
        py: Python<'_>,
        t0: f64,
        t1: f64,
        k: usize,
        max_iter: usize,
    ) -> Option<(Vec<Vec<f64>>, Vec<f64>, f64)> {
        let ws = self.inner.as_ref()?;
        let micros = ws.window(t0, t1);
        let km = py.detach(|| ws.cluster_window(t0, t1, k, max_iter))?;
        let mut masses = vec![0.0; km.centers.len()];
        for (cf, &l) in micros.iter().zip(&km.labels) {
            masses[l] += cf.weight();
        }
        Some((km.centers, masses, km.inertia))
    }
}

/// A [`DriftReport`] as a Python dict. Both streaming clusterers report the same four fields, so the
/// mapping lives once: `alarms` (change reports since construction), `last_alarm` (stream time of the
/// most recent, `None` if there has been none), `distance` (mean routing distance over the adaptive
/// window, in micro-cluster radii) and `window` (points the window holds).
fn drift_dict(py: Python<'_>, r: DriftReport) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("alarms", r.alarms)?;
    d.set_item("last_alarm", r.last_alarm)?;
    d.set_item("distance", r.distance)?;
    d.set_item("window", r.window)?;
    Ok(d)
}

/// Streaming **DenStream** density clusterer over spherical fading micro-clusters (`f64`). Kept
/// separate from `Betula` because it is a different model: a flat set of decaying micro-clusters,
/// not a CF-tree. Built lazily on the first `partial_fit` (dimensionality fixed then).
#[pyclass(name = "DenStream", module = "betula_cluster._core")]
struct PyDenStream {
    eps: f64,
    decay: f64,
    beta: f64,
    mu: f64,
    inner: Option<DenStream<f64, Spherical<f64>>>,
}

#[pymethods]
impl PyDenStream {
    #[new]
    #[pyo3(signature = (eps, decay = 0.25, beta = 0.2, mu = 10.0))]
    fn new(eps: f64, decay: f64, beta: f64, mu: f64) -> Self {
        Self {
            eps,
            decay,
            beta,
            mu,
            inner: None,
        }
    }

    /// Construction params as a dict (read by the Python wrapper's scikit-learn `get_params`).
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("eps", self.eps)?;
        d.set_item("decay", self.decay)?;
        d.set_item("beta", self.beta)?;
        d.set_item("mu", self.mu)?;
        Ok(d)
    }

    fn dim_check(&self, dim: usize) -> PyResult<()> {
        match &self.inner {
            Some(ds) if ds.dim() != dim => Err(PyValueError::new_err(
                "dimension mismatch with previously streamed data",
            )),
            _ => Ok(()),
        }
    }

    /// Stream a chunk (2-D float32/float64) of points into the fading micro-clusters.
    fn partial_fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        self.dim_check(dim)?;
        if self.inner.is_none() {
            self.inner = Some(
                DenStream::new(dim, self.eps, self.decay, self.beta, self.mu)
                    .map_err(PyValueError::new_err)?,
            );
        }
        let ds = self.inner.as_mut().unwrap();
        py.detach(|| {
            for i in 0..n {
                ds.insert(&flat[i * dim..(i + 1) * dim]);
            }
        });
        Ok(())
    }

    /// Run the offline step (connected components of potential micro-clusters → labels).
    fn cluster(&mut self, py: Python<'_>) -> PyResult<()> {
        let ds = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("call partial_fit() or fit() before cluster()"))?;
        py.detach(|| ds.cluster());
        Ok(())
    }

    /// Reset, stream `data`, and run the offline clustering.
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner = None;
        self.partial_fit(py, data)?;
        if let Some(ds) = self.inner.as_mut() {
            py.detach(|| ds.cluster());
        }
        Ok(())
    }

    /// Label `data` rows by their nearest potential micro-cluster (`-1` = noise).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ds = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("call fit() (or partial_fit() + cluster()) first")
        })?;
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        self.dim_check(dim)?;
        let labels = py.detach(|| map_rows(n, |i| ds.predict(&flat[i * dim..(i + 1) * dim])));
        Ok(labels.into_pyarray(py))
    }

    /// Reset, stream + cluster `data`, and return its labels.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.fit(py, data)?;
        self.predict(py, data)
    }

    #[getter]
    fn n_clusters_(&self) -> usize {
        self.inner.as_ref().map_or(0, |d| d.n_clusters())
    }

    #[getter]
    fn n_microclusters_(&self) -> usize {
        self.inner.as_ref().map_or(0, |d| d.potential_count())
    }

    /// Drift diagnostic: how far incoming points are landing from the micro-clusters the stream has
    /// built, and whether that has changed. Reported, never acted on. Zeros before the first
    /// `partial_fit`.
    #[getter]
    fn drift_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        drift_dict(
            py,
            self.inner.as_ref().map_or(
                DriftReport {
                    alarms: 0,
                    last_alarm: None,
                    distance: 0.0,
                    window: 0,
                },
                |d| d.drift(),
            ),
        )
    }

    #[getter]
    fn microcluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (centers, _w, _r, dim) = self.stats()?;
        let rows = centers.len().checked_div(dim).unwrap_or(0);
        Ok(Array2::from_shape_vec((rows, dim), centers)
            .expect("centers length is rows*dim")
            .into_pyarray(py))
    }

    #[getter]
    fn microcluster_weights_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.stats()?.1.into_pyarray(py))
    }

    #[getter]
    fn microcluster_radii_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.stats()?.2.into_pyarray(py))
    }
}

impl PyDenStream {
    fn stats(&self) -> PyResult<F64Stats> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call partial_fit() or fit() first"))?
            .potential_stats())
    }
}

/// Streaming **DBSTREAM** clusterer (Hahsler & Bolaños, 2016): fading micro-clusters connected by
/// *shared density* (overlap mass), recovering arbitrarily-shaped clusters and resisting false
/// bridges between close-but-disconnected regions. Spherical micro-clusters, `float64`; built lazily
/// on the first `partial_fit`.
#[pyclass(name = "DbStream", module = "betula_cluster._core")]
struct PyDbStream {
    r: f64,
    decay: f64,
    alpha: f64,
    min_weight: f64,
    inner: Option<DbStream<f64, Spherical<f64>>>,
}

#[pymethods]
impl PyDbStream {
    #[new]
    #[pyo3(signature = (r = 1.0, decay = 0.01, alpha = 0.1, min_weight = 2.0))]
    fn new(r: f64, decay: f64, alpha: f64, min_weight: f64) -> Self {
        Self {
            r,
            decay,
            alpha,
            min_weight,
            inner: None,
        }
    }

    /// Construction params as a dict (read by the Python wrapper's scikit-learn `get_params`).
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("r", self.r)?;
        d.set_item("decay", self.decay)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("min_weight", self.min_weight)?;
        Ok(d)
    }

    fn dim_check(&self, dim: usize) -> PyResult<()> {
        match &self.inner {
            Some(ds) if ds.dim() != dim => Err(PyValueError::new_err(
                "dimension mismatch with previously streamed data",
            )),
            _ => Ok(()),
        }
    }

    /// Stream a chunk (2-D float32/float64) of points into the fading micro-clusters.
    fn partial_fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        self.dim_check(dim)?;
        if self.inner.is_none() {
            self.inner = Some(
                DbStream::new(dim, self.r, self.decay, self.alpha, self.min_weight)
                    .map_err(PyValueError::new_err)?,
            );
        }
        let ds = self.inner.as_mut().unwrap();
        py.detach(|| {
            for i in 0..n {
                ds.insert(&flat[i * dim..(i + 1) * dim]);
            }
        });
        Ok(())
    }

    /// Run the offline step (connected components of the shared-density graph → labels).
    fn cluster(&mut self, py: Python<'_>) -> PyResult<()> {
        let ds = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("call partial_fit() or fit() before cluster()"))?;
        py.detach(|| ds.cluster());
        Ok(())
    }

    /// Reset, stream `data`, and run the offline clustering.
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner = None;
        self.partial_fit(py, data)?;
        if let Some(ds) = self.inner.as_mut() {
            py.detach(|| ds.cluster());
        }
        Ok(())
    }

    /// Label `data` rows by their nearest micro-cluster within `r` (`-1` = noise).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ds = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("call fit() (or partial_fit() + cluster()) first")
        })?;
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        self.dim_check(dim)?;
        let labels = py.detach(|| map_rows(n, |i| ds.predict(&flat[i * dim..(i + 1) * dim])));
        Ok(labels.into_pyarray(py))
    }

    /// Reset, stream + cluster `data`, and return its labels.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.fit(py, data)?;
        self.predict(py, data)
    }

    #[getter]
    fn n_clusters_(&self) -> usize {
        self.inner.as_ref().map_or(0, |d| d.n_clusters())
    }

    #[getter]
    fn n_microclusters_(&self) -> usize {
        self.inner.as_ref().map_or(0, |d| d.micro_count())
    }

    /// Drift diagnostic: how far incoming points are landing from the micro-clusters the stream has
    /// built, and whether that has changed. Reported, never acted on. Zeros before the first
    /// `partial_fit`.
    #[getter]
    fn drift_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        drift_dict(
            py,
            self.inner.as_ref().map_or(
                DriftReport {
                    alarms: 0,
                    last_alarm: None,
                    distance: 0.0,
                    window: 0,
                },
                |d| d.drift(),
            ),
        )
    }

    #[getter]
    fn microcluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (centers, _w, _r, dim) = self.stats()?;
        let rows = centers.len().checked_div(dim).unwrap_or(0);
        Ok(Array2::from_shape_vec((rows, dim), centers)
            .expect("centers length is rows*dim")
            .into_pyarray(py))
    }

    #[getter]
    fn microcluster_weights_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.stats()?.1.into_pyarray(py))
    }

    #[getter]
    fn microcluster_radii_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.stats()?.2.into_pyarray(py))
    }
}

impl PyDbStream {
    fn stats(&self) -> PyResult<F64Stats> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call partial_fit() or fit() first"))?
            .micro_stats())
    }
}

/// Which of the three k-prototypes blocks a column belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Numeric,
    Categorical,
    Directional,
}

/// Resolve the per-column block assignment: validate indices, reject overlaps, and derive the three
/// block widths. The split order within a block is ascending column order.
///
/// A numeric block is always required — it is what the two other block weights are priced against —
/// and at least one of the other two must be present, or this is `Betula(method='kmeans')` with extra
/// steps. A one-column directional block is rejected outright: a "direction" in one dimension is a
/// sign, and a sign is a category.
fn block_mask(
    categorical: &[usize],
    directional: &[usize],
    dim: usize,
) -> PyResult<(Vec<Block>, usize, usize, usize)> {
    let mut kind = vec![Block::Numeric; dim];
    for (name, cols, block) in [
        ("categorical", categorical, Block::Categorical),
        ("directional", directional, Block::Directional),
    ] {
        for &c in cols {
            if c >= dim {
                return Err(PyValueError::new_err(format!(
                    "{name} column index {c} is out of range for {dim} features"
                )));
            }
            if kind[c] != Block::Numeric {
                return Err(PyValueError::new_err(format!(
                    "column {c} is listed in more than one block"
                )));
            }
            kind[c] = block;
        }
    }
    let count = |b: Block| kind.iter().filter(|&&k| k == b).count();
    let (n_cat, n_dir) = (count(Block::Categorical), count(Block::Directional));
    let n_num = dim - n_cat - n_dir;
    if n_num == 0 {
        return Err(PyValueError::new_err(
            "KPrototypes needs at least one numeric column (pure k-modes is not supported)",
        ));
    }
    if n_cat == 0 && n_dir == 0 {
        return Err(PyValueError::new_err(
            "KPrototypes needs at least one categorical or directional column (otherwise use Betula(method='kmeans'))",
        ));
    }
    if n_dir == 1 {
        return Err(PyValueError::new_err(
            "a directional block needs at least two columns: a direction in one dimension is a sign, which belongs in `categorical`",
        ));
    }
    Ok((kind, n_num, n_cat, n_dir))
}

/// Split a row-major dense matrix into the three blocks by column kind. Categorical columns must hold
/// non-negative integer codes (finiteness is already validated upstream), and the directional block is
/// L2-normalised **per row here, once** — the head takes its unit vectors as given rather than
/// re-checking them at every distance call. A directional row that is all zeros has no direction and
/// is left at zero, which makes its term equal for every prototype: it stops voting instead of voting
/// arbitrarily, the same convention as `normalize=True`.
fn split_mixed(
    flat: &[f64],
    n: usize,
    dim: usize,
    kind: &[Block],
) -> PyResult<(Vec<f64>, Vec<usize>, Vec<f64>)> {
    let count = |b: Block| kind.iter().filter(|&&k| k == b).count();
    let (n_cat, n_dir) = (count(Block::Categorical), count(Block::Directional));
    let mut num = Vec::with_capacity(n * (dim - n_cat - n_dir));
    let mut cat = Vec::with_capacity(n * n_cat);
    let mut dir = Vec::with_capacity(n * n_dir);
    for i in 0..n {
        for (j, &block) in kind.iter().enumerate() {
            let v = flat[i * dim + j];
            match block {
                Block::Numeric => num.push(v),
                Block::Directional => dir.push(v),
                Block::Categorical => {
                    if v < 0.0 || v.fract() != 0.0 {
                        return Err(PyValueError::new_err(
                            "categorical columns must hold non-negative integer codes",
                        ));
                    }
                    cat.push(v as usize);
                }
            }
        }
        if n_dir > 0 {
            let row = &mut dir[i * n_dir..(i + 1) * n_dir];
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in row {
                    *v /= norm;
                }
            }
        }
    }
    Ok((num, cat, dir))
}

/// Per-dimension numeric dispersion as `(mean standard deviation, mean variance)` — the two summaries
/// the block-weight defaults are priced from.
fn numeric_dispersion(num: &[f64], n: usize, n_num: usize) -> (f64, f64) {
    if n == 0 || n_num == 0 {
        return (0.0, 0.0);
    }
    let mut mean = vec![0.0; n_num];
    for i in 0..n {
        for (j, m) in mean.iter_mut().enumerate() {
            *m += num[i * n_num + j];
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut var = vec![0.0; n_num];
    for i in 0..n {
        for (j, v) in var.iter_mut().enumerate() {
            let d = num[i * n_num + j] - mean[j];
            *v += d * d;
        }
    }
    let scale = 1.0 / n as f64 / n_num as f64;
    (
        var.iter().map(|v| (v / n as f64).sqrt()).sum::<f64>() / n_num as f64,
        var.iter().sum::<f64>() * scale,
    )
}

/// Huang's heuristic default for `γ_cat`: half the mean per-dimension numeric standard deviation
/// (falling back to 1.0 when the numeric attributes are degenerate, so the categorical term still
/// matters).
fn default_gamma(num: &[f64], n: usize, n_num: usize) -> f64 {
    let avg_std = numeric_dispersion(num, n, n_num).0;
    if avg_std > 0.0 { 0.5 * avg_std } else { 1.0 }
}

/// Default for `γ_dir`: the mean per-dimension numeric **variance**, so that one unit of
/// `‖u − c‖²` — which runs over `[0, 4]` whatever the data — costs one numeric variance. There is no
/// published heuristic for this weight; this is a scale-matching convention, not a result, and it
/// falls back to 1.0 on degenerate numeric attributes for the same reason `γ_cat` does.
fn default_gamma_dir(num: &[f64], n: usize, n_num: usize) -> f64 {
    let avg_var = numeric_dispersion(num, n, n_num).1;
    if avg_var > 0.0 { avg_var } else { 1.0 }
}

/// A fitted k-prototypes model: mixed micro-clusters, each one's cluster label, and the split metadata.
struct KpModel {
    micros: Vec<MixedCf<f64>>,
    micro_labels: Vec<usize>,
    weights: BlockWeights<f64>,
    n_clusters: usize,
    dim: usize,
    kind: Vec<Block>,
    n_num: usize,
    n_cat: usize,
    n_dir: usize,
}

impl KpModel {
    /// Per-cluster prototypes: numeric centroids (`rows × n_num`), modes (`rows × n_cat`) and unit
    /// directions (`rows × n_dir`), built by merging the micro-clusters of each label.
    fn protos(&self) -> (Vec<f64>, Vec<i64>, Vec<f64>, usize) {
        let schema = self
            .micros
            .first()
            .map(MixedCf::schema)
            .unwrap_or(MixedSchema {
                numeric: self.n_num,
                cardinalities: vec![0; self.n_cat],
                directional: self.n_dir,
            });
        let rows = self.micro_labels.iter().copied().max().map_or(0, |m| m + 1);
        let mut acc: Vec<MixedCf<f64>> = (0..rows).map(|_| MixedCf::new(&schema)).collect();
        for (mi, &lab) in self.micro_labels.iter().enumerate() {
            acc[lab].merge(&self.micros[mi]);
        }
        let mut cent = Vec::with_capacity(rows * self.n_num);
        let mut modes = Vec::with_capacity(rows * self.n_cat);
        let mut dirs = Vec::with_capacity(rows * self.n_dir);
        for a in &acc {
            cent.extend_from_slice(a.numeric_mean());
            modes.extend(a.mode().iter().map(|&c| c as i64));
            dirs.extend_from_slice(a.direction());
        }
        (cent, modes, dirs, rows)
    }
}

// ──────────────────────── Bregman geometry: a second estimator ────────────────────────

/// Which Bregman geometry the tree and heads work in. Names match the `divergence=` keyword.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DivKind {
    Euclidean,
    Kl,
    ItakuraSaito,
    Logistic,
}

impl DivKind {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "euclidean" => Ok(DivKind::Euclidean),
            "kl" => Ok(DivKind::Kl),
            "itakura-saito" => Ok(DivKind::ItakuraSaito),
            "logistic" => Ok(DivKind::Logistic),
            other => Err(PyValueError::new_err(format!(
                "unknown divergence {other:?}: expected \"euclidean\", \"kl\", \
                 \"itakura-saito\" or \"logistic\""
            ))),
        }
    }

    /// The domain `φ` is finite on, as `(low, high, low_open, high_open)`; `None` = all of `ℝ`.
    /// `BregmanCf::push` only `debug_assert!`s this, so a release build would return `NaN` instead
    /// of failing — the check has to happen here, before any value reaches Rust's hot path.
    fn domain(self) -> Option<(f64, f64, bool, bool)> {
        match self {
            DivKind::Euclidean => None,
            DivKind::Kl | DivKind::ItakuraSaito => Some((0.0, f64::INFINITY, true, false)),
            DivKind::Logistic => Some((0.0, 1.0, true, true)),
        }
    }

    fn validate(self, flat: &[f64], dim: usize) -> PyResult<()> {
        let Some((lo, hi, lo_open, hi_open)) = self.domain() else {
            return Ok(());
        };
        for (i, &v) in flat.iter().enumerate() {
            let ok = v.is_finite()
                && (if lo_open { v > lo } else { v >= lo })
                && (if hi_open { v < hi } else { v <= hi });
            if !ok {
                let bound = if hi.is_finite() {
                    format!("in ({lo}, {hi})")
                } else {
                    format!("> {lo}")
                };
                return Err(PyValueError::new_err(format!(
                    "divergence requires every value {bound}, but row {} column {} is {v}",
                    i / dim,
                    i % dim
                )));
            }
        }
        Ok(())
    }
}

/// Which head runs over the Bregman leaves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BregmanHead {
    KMeans,
    Ward,
    Mixture,
}

impl BregmanHead {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "kmeans" => Ok(BregmanHead::KMeans),
            "ward" => Ok(BregmanHead::Ward),
            "mixture" => Ok(BregmanHead::Mixture),
            other => Err(PyValueError::new_err(format!(
                "unknown method {other:?}: expected \"kmeans\", \"ward\" or \"mixture\""
            ))),
        }
    }
}

/// Build a Bregman CF-tree and label every row, monomorphised over one divergence.
#[allow(clippy::too_many_arguments)]
fn bregman_run<B: BregmanDivergence<f64>>(
    flat: &[f64],
    n: usize,
    dim: usize,
    k: usize,
    head: BregmanHead,
    threshold: f64,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    max_iter: usize,
    n_init: usize,
    beta: f64,
    seed: u64,
) -> (Vec<i64>, usize) {
    let mut tree: CFTree<f64, BregmanCf<f64, B>, BregmanCentroid<B>, BregmanIncrease<B>> =
        CFTree::new(
            dim,
            branching,
            leaf_cap,
            threshold,
            max_leaves,
            BregmanCentroid::<B>::new(),
            BregmanIncrease::<B>::new(),
        );
    for i in 0..n {
        tree.insert(&flat[i * dim..(i + 1) * dim]);
    }
    let leaves = tree.num_leaves();
    let micro: Vec<usize> = match head {
        BregmanHead::KMeans => {
            bregman_kmeans::<f64, B>(tree.leaf_features(), k, max_iter, n_init, seed).labels
        }
        BregmanHead::Ward => bregman_agglomerative::<f64, B>(tree.leaf_features(), k).labels,
        BregmanHead::Mixture => {
            bregman_em::<f64, B>(tree.leaf_features(), k, beta, max_iter, seed).labels
        }
    };
    let labels = map_rows(n, |i| {
        micro[tree.nearest_entry(&flat[i * dim..(i + 1) * dim])] as i64
    });
    (labels, leaves)
}

/// Clusterer for data whose natural geometry is a **Bregman divergence** rather than squared
/// Euclidean: KL on the simplex, Itakura–Saito on spectra, logistic loss on probabilities.
///
/// A separate estimator rather than a `divergence=` keyword on [`Betula`], because the combinations
/// that would be legal to write there are not legal to run: a Gaussian head reading a Bregman
/// information as a variance, a χ² gate applying a variance prior to a quantity that is not one.
/// See `docs/adr/004-bregman-public-api.md`.
#[pyclass(name = "BregmanBetula", module = "betula_cluster._core")]
struct PyBregmanBetula {
    n_clusters: usize,
    divergence: String,
    method: String,
    beta: f64,
    threshold: f64,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
    labels: Option<Vec<i64>>,
    leaves: usize,
}

#[pymethods]
impl PyBregmanBetula {
    #[new]
    #[pyo3(signature = (
        n_clusters = 8, divergence = "kl".to_string(), method = "kmeans".to_string(), beta = 1.0,
        threshold = 0.0, branching = 50, leaf_cap = 50, max_leaves = 2048, max_iter = 100,
        n_init = 4, seed = 0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        divergence: String,
        method: String,
        beta: f64,
        threshold: f64,
        branching: usize,
        leaf_cap: usize,
        max_leaves: usize,
        max_iter: usize,
        n_init: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_clusters,
            divergence,
            method,
            beta,
            threshold,
            branching,
            leaf_cap,
            max_leaves,
            max_iter,
            n_init,
            seed,
            labels: None,
            leaves: 0,
        }
    }

    /// Construction params as a dict (read by the Python wrapper's scikit-learn `get_params`).
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("n_clusters", self.n_clusters)?;
        d.set_item("divergence", self.divergence.clone())?;
        d.set_item("method", self.method.clone())?;
        d.set_item("beta", self.beta)?;
        d.set_item("threshold", self.threshold)?;
        d.set_item("branching", self.branching)?;
        d.set_item("leaf_cap", self.leaf_cap)?;
        d.set_item("max_leaves", self.max_leaves)?;
        d.set_item("max_iter", self.max_iter)?;
        d.set_item("n_init", self.n_init)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Leaf count of the fitted tree.
    #[getter]
    fn n_leaves(&self) -> usize {
        self.leaves
    }

    /// Fit and return the training-row labels.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let div = DivKind::parse(&self.divergence)?;
        let head = BregmanHead::parse(&self.method)?;
        if self.n_clusters < 1 {
            return Err(PyValueError::new_err("n_clusters must be >= 1"));
        }
        if !(self.beta > 0.0 && self.beta.is_finite()) {
            return Err(PyValueError::new_err("beta must be positive and finite"));
        }
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        if n < self.n_clusters {
            return Err(PyValueError::new_err(
                "need at least n_clusters rows to fit",
            ));
        }
        div.validate(flat, dim)?;
        let (k, thr, br, lc, ml, mi, ni, beta, seed) = (
            self.n_clusters,
            self.threshold,
            self.branching,
            self.leaf_cap,
            self.max_leaves,
            self.max_iter,
            self.n_init,
            self.beta,
            self.seed,
        );
        let (labels, leaves) = py.detach(|| match div {
            DivKind::Euclidean => bregman_run::<SquaredEuclidean>(
                flat, n, dim, k, head, thr, br, lc, ml, mi, ni, beta, seed,
            ),
            DivKind::Kl => bregman_run::<KullbackLeibler>(
                flat, n, dim, k, head, thr, br, lc, ml, mi, ni, beta, seed,
            ),
            DivKind::ItakuraSaito => bregman_run::<ItakuraSaito>(
                flat, n, dim, k, head, thr, br, lc, ml, mi, ni, beta, seed,
            ),
            DivKind::Logistic => {
                bregman_run::<Logistic>(flat, n, dim, k, head, thr, br, lc, ml, mi, ni, beta, seed)
            }
        });
        self.leaves = leaves;
        self.labels = Some(labels.clone());
        Ok(labels.into_pyarray(py))
    }

    /// Training-row labels from the last fit.
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let l = self
            .labels
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call fit_predict() first"))?;
        Ok(l.clone().into_pyarray(py))
    }
}

/// k-prototypes clusterer for **mixed numeric + categorical + directional** data (Huang, 1997,
/// extended by a third block). `categorical` lists the integer-coded categorical column indices and
/// `directional` the columns that together form one direction (L2-normalised per row here); the
/// remaining columns are numeric. Rows are summarised into bounded mixed micro-clusters by a flat
/// leader pass, then k-prototypes clusters those. `f64`.
#[pyclass(name = "KPrototypes", module = "betula_cluster._core")]
struct PyKPrototypes {
    n_clusters: usize,
    categorical: Vec<usize>,
    directional: Vec<usize>,
    gamma: Option<f64>,
    gamma_dir: Option<f64>,
    threshold: f64,
    max_leaves: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
    model: Option<KpModel>,
}

#[pymethods]
impl PyKPrototypes {
    #[new]
    #[pyo3(signature = (
        n_clusters = 8, categorical = Vec::new(), directional = Vec::new(), gamma = None,
        gamma_dir = None, threshold = 0.0, max_leaves = 2048, max_iter = 100, n_init = 4, seed = 0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        categorical: Vec<usize>,
        directional: Vec<usize>,
        gamma: Option<f64>,
        gamma_dir: Option<f64>,
        threshold: f64,
        max_leaves: usize,
        max_iter: usize,
        n_init: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_clusters,
            categorical,
            directional,
            gamma,
            gamma_dir,
            threshold,
            max_leaves,
            max_iter,
            n_init,
            seed,
            model: None,
        }
    }

    /// Construction params as a dict (read by the Python wrapper's scikit-learn `get_params`).
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("n_clusters", self.n_clusters)?;
        d.set_item("categorical", self.categorical.clone())?;
        d.set_item("directional", self.directional.clone())?;
        d.set_item("gamma", self.gamma)?;
        d.set_item("gamma_dir", self.gamma_dir)?;
        d.set_item("threshold", self.threshold)?;
        d.set_item("max_leaves", self.max_leaves)?;
        d.set_item("max_iter", self.max_iter)?;
        d.set_item("n_init", self.n_init)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Summarise `data` into mixed micro-clusters and cluster them. Returns `self`.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let model = slf.build(data)?;
        slf.model = Some(model);
        Ok(slf)
    }

    /// Fit and return the training-row labels in one call.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.model = Some(self.build(data)?);
        self.predict(py, data)
    }

    /// Label `data` rows by their nearest mixed micro-cluster.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call fit() or fit_predict() first"))?;
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        if dim != m.dim {
            return Err(PyValueError::new_err(
                "dimension mismatch with previously fitted data",
            ));
        }
        let (num, cat, dir) = split_mixed(flat, n, dim, &m.kind)?;
        let labels = py.detach(|| {
            map_rows(n, |i| {
                let xn = &num[i * m.n_num..(i + 1) * m.n_num];
                let xc = &cat[i * m.n_cat..(i + 1) * m.n_cat];
                let xd = &dir[i * m.n_dir..(i + 1) * m.n_dir];
                m.micro_labels[nearest_micro(&m.micros, xn, xc, xd, m.weights)] as i64
            })
        });
        Ok(labels.into_pyarray(py))
    }

    #[getter]
    fn n_clusters_(&self) -> usize {
        self.model.as_ref().map_or(0, |m| m.n_clusters)
    }

    /// Numeric cluster centroids — `(n_clusters, n_numeric)` in categorical-stripped column order.
    #[getter]
    fn cluster_centroids_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call fit() or fit_predict() first"))?;
        let (cent, _modes, _dirs, rows) = m.protos();
        Ok(Array2::from_shape_vec((rows, m.n_num), cent)
            .expect("centroids length is rows*n_num")
            .into_pyarray(py))
    }

    /// Categorical cluster modes — `(n_clusters, n_categorical)` integer codes.
    #[getter]
    fn cluster_modes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call fit() or fit_predict() first"))?;
        let (_cent, modes, _dirs, rows) = m.protos();
        Ok(Array2::from_shape_vec((rows, m.n_cat), modes)
            .expect("modes length is rows*n_cat")
            .into_pyarray(py))
    }

    /// Directional cluster prototypes — `(n_clusters, n_directional)` unit vectors, each the
    /// normalised resultant `R/‖R‖` of its cluster. A cluster whose directions cancelled to (near)
    /// zero has no direction and its row is all zeros.
    #[getter]
    fn cluster_directions_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call fit() or fit_predict() first"))?;
        let (_cent, _modes, dirs, rows) = m.protos();
        Ok(Array2::from_shape_vec((rows, m.n_dir), dirs)
            .expect("directions length is rows*n_dir")
            .into_pyarray(py))
    }
}

impl PyKPrototypes {
    /// Shared fit core: split, summarise, and cluster (compute runs with the GIL released).
    fn build(&self, data: &Bound<'_, PyAny>) -> PyResult<KpModel> {
        if self.n_clusters == 0 {
            return Err(PyValueError::new_err(
                "KPrototypes requires n_clusters >= 1",
            ));
        }
        let (src, n, dim) = flat_as::<f64>(data, RowPrep::None)?;
        let flat = src.as_slice();
        let (kind, n_num, n_cat, n_dir) = block_mask(&self.categorical, &self.directional, dim)?;
        let (num, cat, dir) = split_mixed(flat, n, dim, &kind)?;
        let (k, thr, ml, mi, ni, seed, gpar, gdpar) = (
            self.n_clusters,
            self.threshold,
            self.max_leaves,
            self.max_iter,
            self.n_init,
            self.seed,
            self.gamma,
            self.gamma_dir,
        );
        let py = data.py();
        let model = py.detach(|| {
            let mut cards = vec![0usize; n_cat];
            for i in 0..n {
                for (j, card) in cards.iter_mut().enumerate() {
                    *card = (*card).max(cat[i * n_cat + j] + 1);
                }
            }
            let schema = MixedSchema {
                numeric: n_num,
                cardinalities: cards,
                directional: n_dir,
            };
            let weights = BlockWeights {
                categorical: gpar.unwrap_or_else(|| default_gamma(&num, n, n_num)),
                directional: gdpar.unwrap_or_else(|| default_gamma_dir(&num, n, n_num)),
            };
            let rows = MixedRows {
                numeric: &num,
                categorical: &cat,
                directional: &dir,
                n,
            };
            let micros = summarize_mixed(rows, &schema, weights, thr, ml);
            let micro_labels = kprototypes(&micros, k, weights, mi, ni, seed);
            let mut distinct = micro_labels.clone();
            distinct.sort_unstable();
            distinct.dedup();
            KpModel {
                micros,
                micro_labels,
                weights,
                n_clusters: distinct.len(),
                dim,
                kind,
                n_num,
                n_cat,
                n_dir,
            }
        });
        warn_leaf_budget(py, model.micros.len(), k, ml)?;
        Ok(model)
    }
}

/// Streaming **KLL** quantile sketch (rank-error). Standalone `betula-sketch` primitive.
#[pyclass(name = "KllSketch", module = "betula_cluster._core")]
struct PyKllSketch {
    inner: crate::sketch::KllSketch,
}

#[pymethods]
impl PyKllSketch {
    #[new]
    #[pyo3(signature = (k = 200, seed = 0))]
    fn new(k: usize, seed: u64) -> Self {
        Self {
            inner: crate::sketch::KllSketch::new(k, seed),
        }
    }

    /// Add one value.
    fn update(&mut self, x: f64) {
        self.inner.update(x);
    }

    /// Add every value of a 1-D array.
    fn update_many(&mut self, py: Python<'_>, data: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let v = data.as_array().to_vec();
        py.detach(|| {
            for x in v {
                self.inner.update(x);
            }
        });
        Ok(())
    }

    /// Merge another KLL sketch into this one.
    fn merge(&mut self, other: PyRef<'_, PyKllSketch>) {
        self.inner.merge(&other.inner);
    }

    /// Estimated `q`-quantile (`q ∈ [0, 1]`).
    fn quantile(&self, q: f64) -> f64 {
        self.inner.quantile(q)
    }

    /// Estimated quantiles for an array of `q` values.
    fn quantiles<'py>(
        &self,
        py: Python<'py>,
        qs: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qs = qs.as_array().to_vec();
        Ok(self.inner.quantiles(&qs).into_pyarray(py))
    }

    /// Estimated number of values `≤ value`.
    fn rank(&self, value: f64) -> u64 {
        self.inner.rank(value)
    }

    /// Total number of values added.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }
    /// Smallest value seen (`NaN` if empty).
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }
    /// Largest value seen (`NaN` if empty).
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }
}

/// Streaming **DDSketch** quantile sketch (relative-error). Standalone `betula-sketch` primitive.
#[pyclass(name = "DdSketch", module = "betula_cluster._core")]
struct PyDdSketch {
    inner: crate::sketch::DdSketch,
}

#[pymethods]
impl PyDdSketch {
    #[new]
    #[pyo3(signature = (alpha = 0.01, max_bins = 2048))]
    fn new(alpha: f64, max_bins: usize) -> PyResult<Self> {
        Ok(Self {
            inner: crate::sketch::DdSketch::new(alpha, max_bins).map_err(PyValueError::new_err)?,
        })
    }

    /// Add one value.
    fn update(&mut self, x: f64) {
        self.inner.update(x);
    }

    /// Add every value of a 1-D array.
    fn update_many(&mut self, py: Python<'_>, data: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let v = data.as_array().to_vec();
        py.detach(|| {
            for x in v {
                self.inner.update(x);
            }
        });
        Ok(())
    }

    /// Merge another DDSketch into this one.
    fn merge(&mut self, other: PyRef<'_, PyDdSketch>) -> PyResult<()> {
        self.inner
            .merge(&other.inner)
            .map_err(PyValueError::new_err)
    }

    /// Estimated `q`-quantile (`q ∈ [0, 1]`).
    fn quantile(&self, q: f64) -> f64 {
        self.inner.quantile(q)
    }

    /// Estimated quantiles for an array of `q` values.
    fn quantiles<'py>(
        &self,
        py: Python<'py>,
        qs: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qs = qs.as_array().to_vec();
        Ok(self.inner.quantiles(&qs).into_pyarray(py))
    }

    /// Total number of values added.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }
    /// Relative accuracy `α` (smaller ⇒ tighter quantiles, more buckets).
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }
    /// Smallest value seen (`NaN` if empty).
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }
    /// Largest value seen (`NaN` if empty).
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }
}

/// Compiled core (`betula_cluster._core`); the public API is re-exported by the `betula_cluster`
/// Python package (which also carries the type stubs and `py.typed` marker).
/// Map a method name to a parametric Phase-3 head (the sparse path has no posterior / HDBSCAN).
///
/// `kmedoids` is absent deliberately. Its centre is a single micro-cluster, and this path's leader
/// summary makes that centre useless as a row rule for the reason [`SparseCentroids::pooled`] states:
/// a row's distance to one micro-cluster is dominated by that micro-cluster's norm, not by the
/// overlap that knows the topic. Measured on the four-block corpus at `max_leaves = 300`, the head
/// reads ARI 0.017 by its own medoid rule and 0.002 by the micro-cluster route, against 1.000 for
/// `kmeans` here and 1.000 for `kmedoids` itself through `Betula.fit`, which builds a real CF-tree
/// from the same CSR. Pooling its labels into cluster means would score well and answer a partition
/// the fit never produced, which is the defect this path was fixed for in 0.7.0.
///
/// `fuzzy-cmeans` is absent for a duller reason: this entry point carries no `fuzzifier` keyword, so
/// the head could only ever run at one hard-coded `m`. Its centre is a weighted mean of many
/// micro-clusters and is not subject to the norm domination above, so `Betula::fit` on the same CSR
/// runs it without reservation.
fn parse_parametric(method: &str) -> PyResult<Method> {
    match method {
        "kmeans" => Ok(Method::KMeans),
        "xmeans" => Ok(Method::XMeans),
        "gmm" => Ok(Method::Gmm),
        "gmm-full" => Ok(Method::GmmFull),
        "ward" => Ok(Method::Ward),
        "average" => Ok(Method::Agglomerative {
            linkage: Linkage::Average,
        }),
        "weighted" => Ok(Method::Agglomerative {
            linkage: Linkage::Weighted,
        }),
        "centroid" => Ok(Method::Agglomerative {
            linkage: Linkage::Centroid,
        }),
        "median" => Ok(Method::Agglomerative {
            linkage: Linkage::Median,
        }),
        "spherical-kmeans" => Ok(Method::SphericalKMeans),
        "vmf" => Ok(Method::Movmf),
        _ => Err(PyValueError::new_err(
            "method must be 'kmeans', 'xmeans', 'gmm', 'gmm-full', 'ward', 'average', 'weighted', \
             'centroid', 'median', 'spherical-kmeans' or 'vmf' for sparse input",
        )),
    }
}

/// The `O(nnz)`-per-row labelling a head leaves behind for sparse input, when it has one.
///
/// The two arms are the two kinds of point model a Phase-3 head defines: a Voronoi partition around
/// `k` centres, and a density. Neither is the microcluster route, which is what remains when the head
/// has no point model at all.
enum SparseRule<'a> {
    /// Nearest pooled cluster centroid, with the cluster label each centroid stands for.
    Centres(SparseCentroids, Vec<i64>),
    /// Maximum-posterior component of the head's own fitted density.
    Density(SparseAssigner<'a>),
}

impl SparseRule<'_> {
    fn label_of(&self, idx: &[usize], val: &[f64], x_sq: f64) -> i64 {
        match self {
            SparseRule::Centres(centroids, ids) => ids[centroids.nearest(idx, val, x_sq)],
            SparseRule::Density(density) => density.label_of(idx, val, x_sq) as i64,
        }
    }
}

/// One-shot `O(nnz)` clustering of a CSR matrix (`data` / `indices` / `indptr`, `n_features`). Rows
/// are summarised into spherical micro-clusters touching only the non-zeros (flat leader pass, bounded
/// by `max_leaves`), the micro-clusters are clustered by a parametric head, and each row is labelled by
/// the head's own point rule — its nearest cluster centroid for the centre-based heads, its
/// maximum-posterior component for the density heads whose kernel splits over the support of `x`, and
/// otherwise the label of the micro-cluster the summarisation put it in. See `sparse.rs` for the
/// numerical trade-off of the sparse-native path.
///
/// `projection="svd"` makes this the one-call reduce-then-cluster pipeline for text: the leaf summary
/// is reduced to `projection_dim` CF-weighted principal directions, the head clusters the codes, and
/// each row is labelled by its **own** code — encoded from its non-zeros in `O(nnz·r)`, so the raw
/// high-dimensional geometry is never clustered directly. `"weighted-nmf"` reduces the same way but
/// keeps the micro-cluster route, its codes being a per-row nonnegative least squares rather than a
/// matrix product.
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, n_features, n_clusters = 8, method = "kmeans",
    threshold = 0.0, max_leaves = 2048, max_iter = 100, seed = 0,
    projection = "none", projection_dim = 64, projection_max_iter = 100, auto_k_max = 0
))]
#[allow(clippy::too_many_arguments)]
fn fit_predict_sparse<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    indptr: PyReadonlyArray1<'py, i64>,
    n_features: usize,
    n_clusters: usize,
    method: &str,
    threshold: f64,
    max_leaves: usize,
    max_iter: usize,
    seed: u64,
    projection: &str,
    projection_dim: usize,
    projection_max_iter: usize,
    auto_k_max: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let m = parse_parametric(method)?;
    let spec = parse_projection(projection, projection_dim, projection_max_iter)?;
    let data = data.as_slice()?;
    if matches!(spec.map(|s| s.kind), Some(ProjectionKind::Nmf { .. })) {
        require_nonnegative(data)?;
    }
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    validate_csr(data, indices, indptr, n_features)?;
    if indptr.len() < 2 {
        return Err(PyValueError::new_err("data must have at least one row"));
    }
    // The directional heads cluster on the unit sphere, so they get L2-normalized rows here exactly
    // as they do on the dense path — that rule lived only in the estimator, and this entry point is
    // a module function that never passes through it. Left unnormalized, a leader's norm multiplies
    // the fitted `κ_c` and the head's labels stop agreeing with its own mixture.
    let normalized = matches!(m, Method::SphericalKMeans | Method::Movmf)
        .then(|| normalize_csr_rows(data, indptr));
    let data: &[f64] = normalized.as_deref().unwrap_or(data);
    let (labels, leaves) = py.detach(|| {
        let (micros, of_row) = summarize_sparse(
            data,
            indices,
            indptr,
            n_features,
            threshold,
            max_leaves.max(1),
        );
        let leaves = micros.len();
        let kind = Kind::Parametric(m);
        let out = label_features_proba(&micros, kind, n_clusters, max_iter, seed, auto_k_max, spec);
        // A linear projection labels each row from its own code, touching only the non-zeros.
        let labels = match &out.rule {
            Some(rule) => rule.label_csr(data, indices, indptr, n_features),
            None => {
                let micro_labels = out.labels;
                // Otherwise the head's own rule decides, in whichever `O(nnz)` form it has. A
                // centre-based head owns a Voronoi partition, so the row goes to its nearest
                // *cluster* centroid — what the dense path has labelled with since 0.6.0. A posterior
                // head owns a density; the diagonal and von Mises-Fisher kernels split into a
                // per-component constant plus an `O(nnz)` correction, and the rest do not.
                let sparse_rule = match (spec, assignment_rule(m)) {
                    (None, Rule::Centroid { unit }) => {
                        SparseCentroids::pooled(&micros, &micro_labels, unit)
                            .map(|(c, ids)| SparseRule::Centres(c, ids))
                    }
                    (None, Rule::Posterior) => out
                        .mixture
                        .as_ref()
                        .and_then(Mixture::sparse_assigner)
                        .map(SparseRule::Density),
                    _ => None,
                };
                match sparse_rule {
                    Some(rule) => map_rows(indptr.len() - 1, |r| {
                        let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
                        let val = &data[lo..hi];
                        let idx: Vec<usize> = indices[lo..hi].iter().map(|&c| c as usize).collect();
                        let x_sq: f64 = val.iter().map(|v| v * v).sum();
                        rule.label_of(&idx, val, x_sq)
                    }),
                    // No `O(nnz)` point rule: the row keeps the label of the micro-cluster it was
                    // summarised into. That is the sparse counterpart of the dense path's
                    // point-to-leaf route, it agrees with the summary by construction, and it costs
                    // nothing — the alternative, re-deriving the nearest micro-cluster by centroid
                    // distance, is `O(nnz·L)` and answers a different question badly (see
                    // `SparseCentroids::pooled`).
                    None => of_row.iter().map(|&i| micro_labels[i]).collect(),
                }
            }
        };
        (labels, leaves)
    });
    if Kind::Parametric(m).consumes_k() {
        warn_leaf_budget(py, leaves, n_clusters, max_leaves)?;
    }
    warn_no_compression(py, leaves, labels.len(), max_leaves)?;
    Ok(labels.into_pyarray(py))
}

/// Rows of `means`, checked against the declared component count and dimension.
fn mixture_means(
    means: &PyReadonlyArray2<'_, f64>,
    k: usize,
    dim: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let shape = means.shape();
    if shape[0] != k || shape[1] != dim {
        return Err(PyValueError::new_err(format!(
            "means must be ({k}, {dim}), got ({}, {})",
            shape[0], shape[1]
        )));
    }
    // Logical order, not buffer order: `as_slice` would also accept a column-major array and hand
    // back its transpose, which is the same trap `to_rows` gates against.
    let flat: Vec<f64> = means.as_array().iter().copied().collect();
    Ok(flat.chunks_exact(dim).map(<[f64]>::to_vec).collect())
}

/// Covariances as either `(k, dim)` per-coordinate variances or `(k, dim, dim)` dense matrices.
///
/// Returned as owned matrices plus the diagonal form, because [`Spread`] borrows and the caller
/// needs somewhere for the data to live; only the branch that is actually used is populated.
type Covariances = (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>);

fn mixture_covs(covs: &PyReadonlyArrayDyn<'_, f64>, k: usize, dim: usize) -> PyResult<Covariances> {
    let shape = covs.shape();
    let flat: Vec<f64> = covs.as_array().iter().copied().collect();
    match *shape {
        [rows, cols] if rows == k && cols == dim => Ok((
            flat.chunks_exact(dim).map(<[f64]>::to_vec).collect(),
            Vec::new(),
        )),
        [rows, r, c] if rows == k && r == dim && c == dim => Ok((
            Vec::new(),
            flat.chunks_exact(dim * dim)
                .map(|m| m.chunks_exact(dim).map(<[f64]>::to_vec).collect())
                .collect(),
        )),
        _ => Err(PyValueError::new_err(format!(
            "covariances must be ({k}, {dim}) diagonal or ({k}, {dim}, {dim}) full, got {shape:?}"
        ))),
    }
}

fn spreads<'a>(diag: &'a [Vec<f64>], full: &'a [Vec<Vec<f64>>]) -> Vec<Spread<'a, f64>> {
    if full.is_empty() {
        diag.iter()
            .map(|v| Spread::Diagonal(v.as_slice()))
            .collect()
    } else {
        full.iter().map(|m| Spread::Full(m.as_slice())).collect()
    }
}

/// Mixture-Wasserstein `MW2` between two fitted Gaussian mixtures.
///
/// Takes the parameters rather than an estimator so that a mixture fitted *elsewhere* — sklearn's
/// `GaussianMixture`, an ELKI run, the same model at an earlier timestamp — can be compared without
/// either side being converted into the other's object.
#[pyfunction]
#[pyo3(name = "mixture_w2")]
fn mixture_w2_py(
    weights_a: PyReadonlyArray1<'_, f64>,
    means_a: PyReadonlyArray2<'_, f64>,
    covariances_a: PyReadonlyArrayDyn<'_, f64>,
    weights_b: PyReadonlyArray1<'_, f64>,
    means_b: PyReadonlyArray2<'_, f64>,
    covariances_b: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<f64> {
    let wa = weights_a.as_slice()?;
    let wb = weights_b.as_slice()?;
    if wa.is_empty() || wb.is_empty() {
        return Err(PyValueError::new_err(
            "a mixture needs at least one component",
        ));
    }
    let dim = means_a.shape()[1];
    if means_b.shape()[1] != dim {
        return Err(PyValueError::new_err(format!(
            "the two mixtures live in different dimensions: {dim} and {}",
            means_b.shape()[1]
        )));
    }
    let ma = mixture_means(&means_a, wa.len(), dim)?;
    let mb = mixture_means(&means_b, wb.len(), dim)?;
    let (da, fa) = mixture_covs(&covariances_a, wa.len(), dim)?;
    let (db, fb) = mixture_covs(&covariances_b, wb.len(), dim)?;
    let sa = spreads(&da, &fa);
    let sb = spreads(&db, &fb);
    mixture_w2(
        GaussianMixture {
            weights: wa,
            means: &ma,
            covs: &sa,
        },
        GaussianMixture {
            weights: wb,
            means: &mb,
            covs: &sb,
        },
    )
    .ok_or_else(|| PyValueError::new_err("neither mixture may carry only non-positive weights"))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_predict, m)?)?;
    m.add_function(wrap_pyfunction!(fit_predict_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(mixture_w2_py, m)?)?;
    m.add_class::<Betula>()?;
    m.add_class::<PyBregmanBetula>()?;
    m.add_class::<PyDenStream>()?;
    m.add_class::<PyWindowStream>()?;
    m.add_class::<PyDbStream>()?;
    m.add_class::<PyKPrototypes>()?;
    m.add_class::<PyKllSketch>()?;
    m.add_class::<PyDdSketch>()?;
    m.add(
        "__doc__",
        "Fast, numerically stable BETULA clustering (Rust core).",
    )?;
    Ok(())
}
