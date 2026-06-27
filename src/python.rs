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

use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::clustering::hdbscan::hdbscan;
use crate::clustering::{
    cop_kmeans, kprototypes, nearest_micro, summarize_mixed, ConstraintError, MixedCf,
};
use crate::distance::{
    AverageIntercluster, CFDistance, CentroidEuclidean, CentroidManhattan, MahalanobisChi2,
    VarianceIncrease,
};
use crate::feature::{ClusterFeature, Diagonal, FdSketch, Full, Spherical};
use crate::model::{cluster_leaves, cluster_leaves_proba, Method, Model};
use crate::sparse::{nearest_sparse, summarize_sparse};
use crate::stats::chi2_quantile;
use crate::stream::{DbStream, DenStream};
use crate::topology::{mapper, Lens, MapperGraph, MapperParams};
use crate::tree::CFTree;
use crate::types::Real;

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum Kind {
    Parametric(Method),
    Hdbscan {
        min_samples: usize,
        min_cluster_size: usize,
    },
}

/// Map the `method` keyword (+ HDBSCAN params) to an internal [`Kind`].
fn parse_method(method: &str, min_samples: usize, min_cluster_size: usize) -> PyResult<Kind> {
    match method {
        "kmeans" => Ok(Kind::Parametric(Method::KMeans)),
        "gmm" => Ok(Kind::Parametric(Method::Gmm)),
        "gmm-full" => Ok(Kind::Parametric(Method::GmmFull)),
        "ward" => Ok(Kind::Parametric(Method::Ward)),
        "hdbscan" => Ok(Kind::Hdbscan {
            min_samples,
            min_cluster_size,
        }),
        _ => Err(PyValueError::new_err(
            "method must be 'kmeans', 'gmm', 'gmm-full', 'ward' or 'hdbscan'",
        )),
    }
}

/// Label leaf features with the configured head and, for the GMM heads, also return the per-leaf
/// soft responsibility matrix flattened `(resp, k)` (`None` otherwise) so `predict_proba` can read a
/// true posterior without recomputing the E-step. HDBSCAN keeps `-1` for noise; parametric labels are
/// cast to `i64`. Generic over the element type so it serves both the `f64` and `f32` trees.
#[allow(clippy::type_complexity)]
fn label_features_proba<R: Real, C: ClusterFeature<R>>(
    feats: &[C],
    kind: Kind,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<i64>, Option<(Vec<f64>, usize)>) {
    match kind {
        Kind::Parametric(method) => {
            let (labels, proba) = cluster_leaves_proba(feats, k, method, max_iter, seed);
            (labels.into_iter().map(|l| l as i64).collect(), proba)
        }
        Kind::Hdbscan {
            min_samples,
            min_cluster_size,
        } => (hdbscan(feats, min_samples, min_cluster_size).labels, None),
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

/// Copy the rows of a (non-empty) 2-D array into a flat row-major buffer; returns `(flat, n, dim)`.
/// Generic over the element type so `f32` inputs are clustered in `f32` (no `f64` upcast).
fn to_flat<R: Real + Element>(data: &PyReadonlyArray2<'_, R>) -> PyResult<(Vec<R>, usize, usize)> {
    let arr = data.as_array();
    let n = arr.shape()[0];
    let dim = arr.shape()[1];
    if n == 0 || dim == 0 {
        return Err(PyValueError::new_err("data must be a non-empty 2-D array"));
    }
    // A C-contiguous array exposes its backing slice, so the copy is a single memcpy and the
    // finiteness check one tight auto-vectorized pass — strided `arr[[i, j]]` indexing is neither.
    let flat: Vec<R> = match arr.as_slice() {
        Some(s) => s.to_vec(),
        None => arr.iter().copied().collect(), // non-contiguous (e.g. a transposed view)
    };
    // Validate at the boundary: a NaN/Inf would silently corrupt the tree (mean/scatter become NaN
    // and every downstream label is garbage), so reject it loudly here.
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "data contains NaN or infinite values",
        ));
    }
    Ok((flat, n, dim))
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
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
enum AbsorbKind<R> {
    Euclidean,
    Chi2(MahalanobisChi2<R>),
}

impl<R: Real, C: ClusterFeature<R>> CFDistance<R, C> for AbsorbKind<R> {
    fn point(&self, cf: &C, x: &[R]) -> R {
        match self {
            AbsorbKind::Euclidean => CentroidEuclidean.point(cf, x),
            AbsorbKind::Chi2(m) => m.point(cf, x),
        }
    }
    fn between(&self, a: &C, b: &C) -> R {
        match self {
            AbsorbKind::Euclidean => CentroidEuclidean.between(a, b),
            AbsorbKind::Chi2(m) => m.between(a, b),
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
) -> CFTree<R, C, RouteKind, AbsorbKind<R>> {
    #[cfg(feature = "parallel")]
    if n_jobs > 1 {
        return CFTree::build_parallel(
            dim, branching, leaf_cap, threshold, max_leaves, route, absorb, flat, n, n_jobs,
        );
    }
    let _ = n_jobs;
    let mut tree = CFTree::new(
        dim, branching, leaf_cap, threshold, max_leaves, route, absorb,
    );
    for i in 0..n {
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
    n_jobs: usize,
) -> Vec<i64> {
    let tree = build_tree::<R, C>(
        dim, branching, leaf_cap, threshold, max_leaves, route, absorb, flat, n, n_jobs,
    );
    match kind {
        Kind::Parametric(method) => {
            let model = Model::fit(tree, k, method, max_iter, seed);
            map_rows(n, |i| model.predict(&flat[i * dim..(i + 1) * dim]) as i64)
        }
        Kind::Hdbscan {
            min_samples,
            min_cluster_size,
        } => {
            let res = hdbscan(tree.leaf_features(), min_samples, min_cluster_size);
            map_rows(n, |i| {
                res.labels[tree.nearest_entry(&flat[i * dim..(i + 1) * dim])]
            })
        }
    }
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
) -> PyResult<Vec<i64>> {
    let (mut flat, n, dim) = to_flat(&data)?;
    if normalize {
        normalize_rows(&mut flat, n, dim);
    }
    let route = parse_route(distance)?;
    py.detach(|| {
        // Resolve the absorption gate. χ² uses a user-supplied within-cluster variance scale `s₀`
        // (auto-estimating it from the data picks up between-cluster spread and makes the gate too
        // loose), and a χ²-quantile threshold; euclidean keeps the user's squared-distance
        // threshold unchanged (the default path is computationally identical to before).
        let (gate, thr) = match absorb {
            "euclidean" => (AbsorbKind::Euclidean, R::from_f64(threshold).unwrap()),
            "chi2" => {
                if chi2_scale <= 0.0 {
                    return Err(
                        "absorb='chi2' requires chi2_scale > 0 (the within-cluster variance scale)",
                    );
                }
                let s0 = R::from_f64(chi2_scale).unwrap();
                let kappa = R::from_usize(dim + 2).unwrap();
                let q = R::from_f64(chi2_quantile(dim, chi2_p)).unwrap();
                (AbsorbKind::Chi2(MahalanobisChi2::new(s0, kappa)), q)
            }
            _ => return Err("absorb must be 'euclidean' or 'chi2'"),
        };
        match feature {
            "spherical" => Ok(cluster::<R, Spherical<R>>(
                &flat, n, dim, n_clusters, kind, route, gate, thr, branching, leaf_cap, max_leaves,
                max_iter, seed, n_jobs,
            )),
            "diagonal" => Ok(cluster::<R, Diagonal<R>>(
                &flat, n, dim, n_clusters, kind, route, gate, thr, branching, leaf_cap, max_leaves,
                max_iter, seed, n_jobs,
            )),
            "full" => Ok(cluster::<R, Full<R>>(
                &flat, n, dim, n_clusters, kind, route, gate, thr, branching, leaf_cap, max_leaves,
                max_iter, seed, n_jobs,
            )),
            "fd" => Ok(cluster::<R, FdSketch<R>>(
                &flat, n, dim, n_clusters, kind, route, gate, thr, branching, leaf_cap, max_leaves,
                max_iter, seed, n_jobs,
            )),
            _ => Err("feature must be 'spherical', 'diagonal', 'full' or 'fd'"),
        }
    })
    .map_err(PyValueError::new_err)
}

/// Cluster the rows of a 2-D float32 or float64 array; returns one int64 label per row (`-1` =
/// noise, produced only by `method="hdbscan"`). `float32` input is clustered in `f32` (half the
/// memory, no upcast). With `n_clusters=0` and `method="gmm"`/`"gmm-full"` the component count is
/// selected automatically by BIC. `absorb="chi2"` switches the CF-tree's absorption to a
/// mass-invariant Mahalanobis-χ² gate at level `chi2_p` with within-cluster variance `chi2_scale`
/// (required for `chi2`; `absorb="euclidean"` is the default and unchanged).
#[pyfunction]
#[pyo3(signature = (
    data, n_clusters = 8, feature = "diagonal", method = "gmm", threshold = 0.0,
    branching = 32, leaf_cap = 32, max_leaves = 2000, max_iter = 100,
    min_samples = 5, min_cluster_size = 5, seed = 0, distance = "euclidean",
    absorb = "euclidean", chi2_p = 0.95, chi2_scale = 0.0, n_jobs = 1, normalize = false
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
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let kind = parse_method(method, min_samples, min_cluster_size)?;
    let labels = if let Ok(a) = data.extract::<PyReadonlyArray2<'py, f64>>() {
        run_oneshot::<f64>(
            py, a, n_clusters, feature, kind, distance, absorb, chi2_p, chi2_scale, threshold,
            branching, leaf_cap, max_leaves, max_iter, seed, n_jobs, normalize,
        )?
    } else if let Ok(a) = data.extract::<PyReadonlyArray2<'py, f32>>() {
        run_oneshot::<f32>(
            py, a, n_clusters, feature, kind, distance, absorb, chi2_p, chi2_scale, threshold,
            branching, leaf_cap, max_leaves, max_iter, seed, n_jobs, normalize,
        )?
    } else {
        return Err(PyValueError::new_err(
            "data must be a 2-D float32 or float64 array",
        ));
    };
    Ok(labels.into_pyarray(py))
}

// ── streaming estimator ───────────────────────────────────────────────────────────────────────

type BetulaTree<R, C> = CFTree<R, C, RouteKind, AbsorbKind<R>>;

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
    match absorb {
        "euclidean" => Ok((AbsorbKind::Euclidean, R::from_f64(threshold).unwrap())),
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
        _ => Err("absorb must be 'euclidean' or 'chi2'"),
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
    ) -> Result<Self, &'static str> {
        macro_rules! tree {
            () => {{
                let mut t =
                    CFTree::new(dim, branching, leaf_cap, threshold, max_leaves, route, gate);
                t.set_huber_k(huber_k);
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

    #[allow(clippy::type_complexity)]
    fn label_proba(
        &self,
        kind: Kind,
        k: usize,
        max_iter: usize,
        seed: u64,
    ) -> (Vec<i64>, Option<(Vec<f64>, usize)>) {
        match self {
            TreeState::Spherical(t) => {
                label_features_proba(t.leaf_features(), kind, k, max_iter, seed)
            }
            TreeState::Diagonal(t) => {
                label_features_proba(t.leaf_features(), kind, k, max_iter, seed)
            }
            TreeState::Full(t) => label_features_proba(t.leaf_features(), kind, k, max_iter, seed),
            TreeState::Fd(t) => label_features_proba(t.leaf_features(), kind, k, max_iter, seed),
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
            )?);
        }
        let tree = slot.as_mut().unwrap();
        for i in 0..n {
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
            )?);
        }
        let tree = slot.as_mut().unwrap();
        let mut buf = vec![R::zero(); dim];
        for w in indptr.windows(2) {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
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

    /// Mapper topological-skeleton graph over the leaf microclusters.
    fn mapper(&self, p: &MapperParams) -> MapperGraph {
        match self {
            TreeState::Spherical(t) => mapper(t.leaf_features(), p),
            TreeState::Diagonal(t) => mapper(t.leaf_features(), p),
            TreeState::Full(t) => mapper(t.leaf_features(), p),
            TreeState::Fd(t) => mapper(t.leaf_features(), p),
        }
    }

    /// For each row: distance to its assigned cluster centroid divided by the cluster's RMS radius
    /// (a Mahalanobis-like z-score). Points routed to a noise microcluster score `+inf`.
    fn outlier_scores(
        &self,
        labels: &[i64],
        centers: &[f64],
        radii: &[f64],
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
            let mut d2 = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                let diff = xj.to_f64().unwrap() - centers[cl * dim + j];
                d2 += diff * diff;
            }
            let d = d2.sqrt();
            let r = radii[cl];
            if r > 0.0 {
                d / r
            } else {
                d
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

/// Tree-construction config passed to [`TreeState::stream_chunk`] (groups the estimator's settings).
/// Validate CSR arrays so the row-expansion never indexes out of bounds: matched `data`/`indices`
/// lengths, an `indptr` that starts at 0, is non-decreasing, and ends at `nnz`, in-range column
/// indices, and finite values.
fn validate_csr(data: &[f64], indices: &[i64], indptr: &[i64], n_features: usize) -> PyResult<()> {
    if n_features == 0 {
        return Err(PyValueError::new_err("n_features must be > 0"));
    }
    if data.len() != indices.len() {
        return Err(PyValueError::new_err(
            "CSR data and indices must have equal length",
        ));
    }
    if indptr.first() != Some(&0) || *indptr.last().unwrap_or(&-1) as usize != data.len() {
        return Err(PyValueError::new_err(
            "CSR indptr must start at 0 and end at nnz",
        ));
    }
    if indptr.windows(2).any(|w| w[1] < w[0]) {
        return Err(PyValueError::new_err("CSR indptr must be non-decreasing"));
    }
    if indices.iter().any(|&c| c < 0 || c as usize >= n_features) {
        return Err(PyValueError::new_err("CSR column index out of range"));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "data contains NaN or infinite values",
        ));
    }
    Ok(())
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
}

/// Stateful BETULA estimator. `partial_fit` streams data into a memory-bounded CF-tree; `fit`
/// (re)builds from one array; `predict` labels new points via their nearest leaf. The covariance
/// model and dimensionality are locked in at the first `partial_fit` / `fit`.
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
}

/// Copy a 2-D array into a flat row-major `Vec<R>`, casting from the other float dtype if needed
/// (lossless `f32→f64`; the deliberate `f64→f32` narrowing matches the f32 tree).
fn flat_as<R: Real + Element>(
    data: &Bound<'_, PyAny>,
    normalize: bool,
) -> PyResult<(Vec<R>, usize, usize)> {
    let (mut flat, n, dim) = if let Ok(a) = data.extract::<PyReadonlyArray2<R>>() {
        to_flat(&a)?
    } else if let Ok(a) = data.extract::<PyReadonlyArray2<f64>>() {
        cast_flat::<f64, R>(&a)?
    } else if let Ok(a) = data.extract::<PyReadonlyArray2<f32>>() {
        cast_flat::<f32, R>(&a)?
    } else {
        return Err(PyValueError::new_err(
            "data must be a 2-D float32 or float64 array",
        ));
    };
    if normalize {
        normalize_rows(&mut flat, n, dim);
    }
    Ok((flat, n, dim))
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
        self.dim = 0;
    }

    /// Stream a chunk into the matching-dtype tree (dtype is the existing tree's, or the input's at
    /// first fit). Invalidates the cached labels. The config is built inline (not via a `&self`
    /// method) so the borrow checker keeps `&self.feature` disjoint from `&mut self.state*`.
    fn stream(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
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
        };
        if use_f32 {
            let (flat, n, dim) = flat_as::<f32>(data, self.normalize)?;
            if self.dim != 0 && self.dim != dim {
                return Err(PyValueError::new_err(
                    "dimension mismatch with previously fitted data",
                ));
            }
            TreeState::stream_chunk(&mut self.state32, &cfg, &flat, n, dim)
                .map_err(PyValueError::new_err)?;
            self.dim = dim;
        } else {
            let (flat, n, dim) = flat_as::<f64>(data, self.normalize)?;
            if self.dim != 0 && self.dim != dim {
                return Err(PyValueError::new_err(
                    "dimension mismatch with previously fitted data",
                ));
            }
            TreeState::stream_chunk(&mut self.state64, &cfg, &flat, n, dim)
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
    ) -> PyResult<()> {
        if self.state32.is_some() {
            return Err(PyValueError::new_err(
                "sparse (CSR) input is float64-only; this estimator was already fit on float32 data",
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
        };
        TreeState::stream_chunk_csr(&mut self.state64, &cfg, data, indices, indptr, n_features)
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
        Ok(t.route_csr(labels, data, indices, indptr, n_features))
    }

    fn check_dim(&self, dim: usize) -> PyResult<()> {
        if self.dim != 0 && self.dim != dim {
            return Err(PyValueError::new_err(
                "dimension mismatch with previously fitted data",
            ));
        }
        Ok(())
    }

    /// Cluster the current leaf features (whichever dtype tree exists) and cache the labels.
    fn finalize(&mut self) {
        let (kind, k, mi, seed) = (self.kind, self.n_clusters, self.max_iter, self.seed);
        let result = if let Some(t) = &self.state64 {
            Some(t.label_proba(kind, k, mi, seed))
        } else {
            self.state32
                .as_ref()
                .map(|t| t.label_proba(kind, k, mi, seed))
        };
        match result {
            Some((labels, proba)) => {
                self.labels = Some(labels);
                self.proba = proba;
            }
            None => {
                self.labels = None;
                self.proba = None;
            }
        }
    }

    /// Cluster the leaves under pairwise constraints (COP-KMeans). `data` is the just-streamed array,
    /// re-read only to route the constrained rows to their leaves. Sets `labels` (no GMM posterior).
    fn finalize_constrained(
        &mut self,
        data: &Bound<'_, PyAny>,
        must: &[(i64, i64)],
        cannot: &[(i64, i64)],
    ) -> PyResult<()> {
        let (k, mi, seed, norm) = (self.n_clusters, self.max_iter, self.seed, self.normalize);
        let labels = if let Some(t) = self.state64.as_ref() {
            let (flat, n, dim) = flat_as::<f64>(data, norm)?;
            constrained_labels(t, &flat, n, dim, must, cannot, k, mi, seed)?
        } else if let Some(t) = self.state32.as_ref() {
            let (flat, n, dim) = flat_as::<f32>(data, norm)?;
            constrained_labels(t, &flat, n, dim, must, cannot, k, mi, seed)?
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
        if let Some(t) = &self.state64 {
            let (flat, n, dim) = flat_as::<f64>(data, self.normalize)?;
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.route(labels, &flat, n, dim))
                .into_pyarray(py))
        } else if let Some(t) = &self.state32 {
            let (flat, n, dim) = flat_as::<f32>(data, self.normalize)?;
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.route(labels, &flat, n, dim))
                .into_pyarray(py))
        } else {
            Err(PyValueError::new_err(
                "call fit() or fit_predict() before predict()",
            ))
        }
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
}

#[pymethods]
impl Betula {
    #[new]
    #[pyo3(signature = (
        n_clusters = 8, feature = "diagonal", method = "gmm", threshold = 0.0,
        branching = 32, leaf_cap = 32, max_leaves = 2000, max_iter = 100,
        min_samples = 5, min_cluster_size = 5, seed = 0,
        distance = "euclidean", absorb = "euclidean", chi2_p = 0.95, chi2_scale = 0.0, decay = 1.0,
        normalize = false, huber_k = None
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
    ) -> PyResult<Self> {
        let kind = parse_method(method, min_samples, min_cluster_size)?;
        let route = parse_route(distance)?;
        if !matches!(feature, "spherical" | "diagonal" | "full" | "fd") {
            return Err(PyValueError::new_err(
                "feature must be 'spherical', 'diagonal', 'full' or 'fd'",
            ));
        }
        if !matches!(absorb, "euclidean" | "chi2") {
            return Err(PyValueError::new_err(
                "absorb must be 'euclidean' or 'chi2'",
            ));
        }
        if absorb == "chi2" && chi2_scale <= 0.0 {
            return Err(PyValueError::new_err(
                "absorb='chi2' requires chi2_scale > 0 (the within-cluster variance scale)",
            ));
        }
        if let Some(k) = huber_k {
            if k <= 0.0 || k.is_nan() {
                return Err(PyValueError::new_err(
                    "huber_k must be > 0 (per-dimension std multiplier), or None to disable",
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
            seed,
            absorb: absorb.to_string(),
            chi2_p,
            chi2_scale,
            decay,
            normalize,
            huber_k,
            dim: 0,
            state64: None,
            state32: None,
            labels: None,
            proba: None,
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
            Some(data) => slf.stream(data)?,
            None => {
                if slf.state64.is_none() && slf.state32.is_none() {
                    return Err(PyValueError::new_err(
                        "partial_fit() with no data before any rows were added",
                    ));
                }
                slf.finalize();
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
        slf.stream(data)?;
        slf.finalize();
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

    /// Reset, fit on `data`, and return the training labels in one call.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.reset();
        self.stream(data)?;
        self.finalize();
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
        slf.stream(data)?;
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
        )
    }

    fn fit_csr(
        &mut self,
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
        )?;
        self.finalize();
        Ok(())
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
        self.stream_csr(d, idx, ip, n_features)?;
        self.finalize();
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
    /// posterior; raises otherwise. Backs `predict_proba` (route a point to its leaf, read its row).
    #[getter]
    fn microcluster_proba_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (flat, k) = self.proba.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "predict_proba posterior is only available after fit with method='gmm' or 'gmm-full'",
            )
        })?;
        let rows = flat.len().checked_div(*k).unwrap_or(0);
        Ok(Array2::from_shape_vec((rows, *k), flat.clone())
            .expect("proba length is n_leaves*k")
            .into_pyarray(py))
    }

    /// Macro-cluster centroids — `(n_clusters, dim)`; requires a finalized clustering.
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

    /// Per-row outlier score: distance to the assigned cluster centroid divided by that cluster's
    /// RMS radius (a Mahalanobis-like z-score). Rows routed to HDBSCAN noise score `+inf`.
    fn outlier_scores<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let labels = self.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "finalize first (fit / fit_predict / partial_fit with no args) before outlier_scores()",
            )
        })?;
        let k = cluster_count_for_centers(labels);
        if let Some(t) = &self.state64 {
            let (centers, radii, _w, _d) = t.cluster_stats(labels, k);
            let (flat, n, dim) = flat_as::<f64>(data, self.normalize)?;
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.outlier_scores(labels, &centers, &radii, &flat, n, dim))
                .into_pyarray(py))
        } else if let Some(t) = &self.state32 {
            let (centers, radii, _w, _d) = t.cluster_stats(labels, k);
            let (flat, n, dim) = flat_as::<f32>(data, self.normalize)?;
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.outlier_scores(labels, &centers, &radii, &flat, n, dim))
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
            let (flat, n, dim) = flat_as::<f64>(data, self.normalize)?;
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.assign_microclusters(&flat, n, dim))
                .into_pyarray(py))
        } else if let Some(t) = &self.state32 {
            let (flat, n, dim) = flat_as::<f32>(data, self.normalize)?;
            self.check_dim(dim)?;
            Ok(py
                .detach(|| t.assign_microclusters(&flat, n, dim))
                .into_pyarray(py))
        } else {
            Err(PyValueError::new_err(
                "call fit() or partial_fit() before assign_microclusters()",
            ))
        }
    }

    /// Mapper topological-skeleton graph over the leaf microclusters, as a dict of arrays.
    ///
    /// `lens` selects the filter function (`density` / `radius` / `l2norm` / `coordinate` /
    /// `eccentricity`); `resolution` × `gain` set the overlapping cover; `link_scale` the within-bin
    /// single-linkage scale (× the median NN gap); nodes lighter than `min_node_mass` are dropped. Returns node
    /// members / mass / bin / lens / centroids, weighted `edges`, `branch_points` and `bridges`.
    #[pyo3(signature = (lens = "density", resolution = 10, gain = 0.3, link_scale = 1.0,
                        min_node_mass = 0.0, density_k = 5, coordinate = 0))]
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
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let lens =
            match lens {
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
                _ => return Err(PyValueError::new_err(
                    "lens must be 'density', 'radius', 'l2norm', 'coordinate' or 'eccentricity'",
                )),
            };
        let p = MapperParams {
            lens,
            resolution,
            gain,
            link_scale,
            min_node_mass,
        };
        let g = match (&self.state64, &self.state32) {
            (Some(t), _) => py.detach(|| t.mapper(&p)),
            (_, Some(t)) => py.detach(|| t.mapper(&p)),
            _ => {
                return Err(PyValueError::new_err(
                    "call fit() or partial_fit() before mapper()",
                ))
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
        let (flat, n, dim) = flat_as::<f64>(data, false)?;
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
        let (flat, n, dim) = flat_as::<f64>(data, false)?;
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
        let (flat, n, dim) = flat_as::<f64>(data, false)?;
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
        let (flat, n, dim) = flat_as::<f64>(data, false)?;
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

/// Resolve the categorical column mask: validate indices, derive the ascending categorical column
/// order (the split order) and the numeric/categorical counts. Requires both kinds to be present.
fn cat_mask(categorical: &[usize], dim: usize) -> PyResult<(Vec<bool>, Vec<usize>, usize, usize)> {
    let mut is_cat = vec![false; dim];
    for &c in categorical {
        if c >= dim {
            return Err(PyValueError::new_err(format!(
                "categorical column index {c} is out of range for {dim} features"
            )));
        }
        is_cat[c] = true;
    }
    let cat_cols: Vec<usize> = (0..dim).filter(|&j| is_cat[j]).collect();
    let n_cat = cat_cols.len();
    let n_num = dim - n_cat;
    if n_cat == 0 {
        return Err(PyValueError::new_err(
            "KPrototypes needs at least one categorical column (otherwise use Betula(method='kmeans'))",
        ));
    }
    if n_num == 0 {
        return Err(PyValueError::new_err(
            "KPrototypes needs at least one numeric column (pure k-modes is not supported)",
        ));
    }
    Ok((is_cat, cat_cols, n_num, n_cat))
}

/// Split a row-major dense matrix into numeric values and integer category codes by column kind.
/// Categorical columns must hold non-negative integer codes (finiteness is already validated upstream).
fn split_mixed(
    flat: &[f64],
    n: usize,
    dim: usize,
    is_cat: &[bool],
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let n_cat = is_cat.iter().filter(|&&c| c).count();
    let mut num = Vec::with_capacity(n * (dim - n_cat));
    let mut cat = Vec::with_capacity(n * n_cat);
    for i in 0..n {
        for (j, &cat_col) in is_cat.iter().enumerate() {
            let v = flat[i * dim + j];
            if cat_col {
                if v < 0.0 || v.fract() != 0.0 {
                    return Err(PyValueError::new_err(
                        "categorical columns must hold non-negative integer codes",
                    ));
                }
                cat.push(v as usize);
            } else {
                num.push(v);
            }
        }
    }
    Ok((num, cat))
}

/// Huang's heuristic default for `γ`: half the mean per-dimension numeric standard deviation (falling
/// back to 1.0 when the numeric attributes are degenerate, so the categorical term still matters).
fn default_gamma(num: &[f64], n: usize, n_num: usize) -> f64 {
    if n == 0 || n_num == 0 {
        return 1.0;
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
    let avg_std = var.iter().map(|v| (v / n as f64).sqrt()).sum::<f64>() / n_num as f64;
    if avg_std > 0.0 {
        0.5 * avg_std
    } else {
        1.0
    }
}

/// A fitted k-prototypes model: mixed micro-clusters, each one's cluster label, and the split metadata.
struct KpModel {
    micros: Vec<MixedCf<f64>>,
    micro_labels: Vec<usize>,
    gamma: f64,
    n_clusters: usize,
    dim: usize,
    cat_cols: Vec<usize>,
    n_num: usize,
    n_cat: usize,
}

impl KpModel {
    /// Per-cluster prototypes: numeric centroids (`rows × n_num`) and modes (`rows × n_cat`), built by
    /// merging the micro-clusters of each label.
    fn protos(&self) -> (Vec<f64>, Vec<i64>, usize) {
        let cards = self
            .micros
            .first()
            .map(|m| m.cardinalities())
            .unwrap_or_default();
        let rows = self.micro_labels.iter().copied().max().map_or(0, |m| m + 1);
        let mut acc: Vec<MixedCf<f64>> = (0..rows)
            .map(|_| MixedCf::new(self.n_num, &cards))
            .collect();
        for (mi, &lab) in self.micro_labels.iter().enumerate() {
            acc[lab].merge(&self.micros[mi]);
        }
        let mut cent = Vec::with_capacity(rows * self.n_num);
        let mut modes = Vec::with_capacity(rows * self.n_cat);
        for a in &acc {
            cent.extend_from_slice(a.numeric_mean());
            modes.extend(a.mode().iter().map(|&c| c as i64));
        }
        (cent, modes, rows)
    }
}

/// k-prototypes clusterer for **mixed numeric + categorical** data (Huang, 1997). `categorical` lists
/// the integer-coded categorical column indices; the remaining columns are numeric. Rows are summarised
/// into bounded mixed micro-clusters by a flat leader pass, then k-prototypes clusters those. `f64`.
#[pyclass(name = "KPrototypes", module = "betula_cluster._core")]
struct PyKPrototypes {
    n_clusters: usize,
    categorical: Vec<usize>,
    gamma: Option<f64>,
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
        n_clusters = 8, categorical = Vec::new(), gamma = None, threshold = 0.0,
        max_leaves = 2048, max_iter = 100, n_init = 4, seed = 0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        categorical: Vec<usize>,
        gamma: Option<f64>,
        threshold: f64,
        max_leaves: usize,
        max_iter: usize,
        n_init: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_clusters,
            categorical,
            gamma,
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
        d.set_item("gamma", self.gamma)?;
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
        let (flat, n, dim) = flat_as::<f64>(data, false)?;
        if dim != m.dim {
            return Err(PyValueError::new_err(
                "dimension mismatch with previously fitted data",
            ));
        }
        let mut is_cat = vec![false; dim];
        for &c in &m.cat_cols {
            is_cat[c] = true;
        }
        let (num, cat) = split_mixed(&flat, n, dim, &is_cat)?;
        let labels = py.detach(|| {
            map_rows(n, |i| {
                let xn = &num[i * m.n_num..(i + 1) * m.n_num];
                let xc = &cat[i * m.n_cat..(i + 1) * m.n_cat];
                m.micro_labels[nearest_micro(&m.micros, xn, xc, m.gamma)] as i64
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
        let (cent, _modes, rows) = m.protos();
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
        let (_cent, modes, rows) = m.protos();
        Ok(Array2::from_shape_vec((rows, m.n_cat), modes)
            .expect("modes length is rows*n_cat")
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
        let (flat, n, dim) = flat_as::<f64>(data, false)?;
        let (is_cat, cat_cols, n_num, n_cat) = cat_mask(&self.categorical, dim)?;
        let (num, cat) = split_mixed(&flat, n, dim, &is_cat)?;
        let (k, thr, ml, mi, ni, seed, gpar) = (
            self.n_clusters,
            self.threshold,
            self.max_leaves,
            self.max_iter,
            self.n_init,
            self.seed,
            self.gamma,
        );
        let py = data.py();
        Ok(py.detach(|| {
            let mut cards = vec![0usize; n_cat];
            for i in 0..n {
                for (j, card) in cards.iter_mut().enumerate() {
                    *card = (*card).max(cat[i * n_cat + j] + 1);
                }
            }
            let gamma = gpar.unwrap_or_else(|| default_gamma(&num, n, n_num));
            let micros = summarize_mixed(&num, &cat, n, n_num, &cards, gamma, thr, ml);
            let micro_labels = kprototypes(&micros, k, gamma, mi, ni, seed);
            let mut distinct = micro_labels.clone();
            distinct.sort_unstable();
            distinct.dedup();
            KpModel {
                micros,
                micro_labels,
                gamma,
                n_clusters: distinct.len(),
                dim,
                cat_cols,
                n_num,
                n_cat,
            }
        }))
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

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }
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

    fn update(&mut self, x: f64) {
        self.inner.update(x);
    }

    fn update_many(&mut self, py: Python<'_>, data: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let v = data.as_array().to_vec();
        py.detach(|| {
            for x in v {
                self.inner.update(x);
            }
        });
        Ok(())
    }

    fn merge(&mut self, other: PyRef<'_, PyDdSketch>) -> PyResult<()> {
        self.inner
            .merge(&other.inner)
            .map_err(PyValueError::new_err)
    }

    fn quantile(&self, q: f64) -> f64 {
        self.inner.quantile(q)
    }

    fn quantiles<'py>(
        &self,
        py: Python<'py>,
        qs: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qs = qs.as_array().to_vec();
        Ok(self.inner.quantiles(&qs).into_pyarray(py))
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }
}

/// Compiled core (`betula_cluster._core`); the public API is re-exported by the `betula_cluster`
/// Python package (which also carries the type stubs and `py.typed` marker).
/// Map a method name to a parametric Phase-3 head (the sparse path has no posterior / HDBSCAN).
fn parse_parametric(method: &str) -> PyResult<Method> {
    match method {
        "kmeans" => Ok(Method::KMeans),
        "gmm" => Ok(Method::Gmm),
        "gmm-full" => Ok(Method::GmmFull),
        "ward" => Ok(Method::Ward),
        _ => Err(PyValueError::new_err(
            "method must be 'kmeans', 'gmm', 'gmm-full' or 'ward' for sparse input",
        )),
    }
}

/// One-shot `O(nnz)` clustering of a CSR matrix (`data` / `indices` / `indptr`, `n_features`). Rows
/// are summarised into spherical micro-clusters touching only the non-zeros (flat leader pass, bounded
/// by `max_leaves`), the micro-clusters are clustered by a parametric head, and each row is labelled by
/// its nearest micro-cluster. See `sparse.rs` for the numerical trade-off of the sparse-native path.
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, n_features, n_clusters = 8, method = "kmeans",
    threshold = 0.0, max_leaves = 2048, max_iter = 100, seed = 0
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
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let m = parse_parametric(method)?;
    let data = data.as_slice()?;
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    validate_csr(data, indices, indptr, n_features)?;
    if indptr.len() < 2 {
        return Err(PyValueError::new_err("data must have at least one row"));
    }
    let labels = py.detach(|| {
        let micros = summarize_sparse(
            data,
            indices,
            indptr,
            n_features,
            threshold,
            max_leaves.max(1),
        );
        let micro_labels = cluster_leaves(&micros, n_clusters, m, max_iter, seed);
        let means: Vec<Vec<f64>> = micros.iter().map(|c| c.mean().to_vec()).collect();
        let musq: Vec<f64> = means
            .iter()
            .map(|mu| mu.iter().map(|v| v * v).sum())
            .collect();
        map_rows(indptr.len() - 1, |r| {
            let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
            let val = &data[lo..hi];
            let idx: Vec<usize> = indices[lo..hi].iter().map(|&c| c as usize).collect();
            let x_sq: f64 = val.iter().map(|v| v * v).sum();
            micro_labels[nearest_sparse(&means, &musq, &idx, val, x_sq)] as i64
        })
    });
    Ok(labels.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_predict, m)?)?;
    m.add_function(wrap_pyfunction!(fit_predict_sparse, m)?)?;
    m.add_class::<Betula>()?;
    m.add_class::<PyDenStream>()?;
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
