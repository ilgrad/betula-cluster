//! betula-cluster: numerically stable, fast clustering on BETULA CF-trees.
//!
//! See `DESIGN.md` for the full architecture and the verified mathematical foundation.
//! The core stores clustering features as `(n, μ, S)` (weight, mean, sum of squared
//! deviations) and updates/merges them without catastrophic cancellation; covariance is
//! positive semi-definite by construction.
//!
//! # What is public
//!
//! The modules listed below are the crate's contract, and semantic versioning is a promise about
//! them: [`tree`], [`feature`], [`distance`], [`bregman`], [`model`], [`clustering`], [`types`],
//! [`sparse`], [`order`], [`coreset`], [`stream`], [`window`], [`validity`]. A minimal fit reads:
//!
//! ```no_run
//! use betula_cluster::{distance::CentroidEuclidean, feature::Spherical};
//! use betula_cluster::{model::{Method, Model}, tree::CFTree};
//! # let points: Vec<Vec<f64>> = Vec::new();
//! let mut tree: CFTree<f64, Spherical<f64>, _, _> =
//!     CFTree::new(2, 32, 32, 0.0, 2000, CentroidEuclidean, CentroidEuclidean);
//! for p in &points {
//!     tree.try_insert(p)?;
//! }
//! let model = Model::fit(tree, 4, Method::Gmm, 100, 0, 0, 0);
//! # Ok::<(), betula_cluster::types::ShapeError>(())
//! ```
//!
//! The rest — `adwin`, `assign`, `fidelity`, `kernels`, `linalg`, `mixture`, `sketch`, `stats`,
//! `topology`, `wasserstein` — is `#[doc(hidden)]`: reachable, because the benchmarks in this
//! repository call into it and because hiding is cheaper to reverse than deleting, but **not part
//! of the contract**. Those modules may change shape in any release. What they implement is
//! reachable through the Python package, which is the supported way to use them: `mapper()` is
//! `topology`, `KllSketch` / `DdSketch` are `sketch`, the mixture-Wasserstein drift distance is
//! `wasserstein`, and `tree_report()`'s fidelity number is `fidelity`.

pub mod bregman;
pub mod clustering;
pub mod coreset;
pub mod distance;
pub mod feature;
pub mod model;
pub mod order;
pub mod sparse;
pub mod stream;
pub mod tree;
pub mod types;
pub mod validity;
pub mod window;

#[doc(hidden)]
pub mod adwin;
#[doc(hidden)]
pub mod assign;
#[doc(hidden)]
pub mod fidelity;
#[doc(hidden)]
pub mod kernels;
#[doc(hidden)]
pub mod linalg;
#[doc(hidden)]
pub mod mixture;
#[doc(hidden)]
pub mod sketch;
#[doc(hidden)]
pub mod stats;
#[doc(hidden)]
pub mod topology;
#[doc(hidden)]
pub mod wasserstein;

#[cfg(feature = "python")]
mod python;
