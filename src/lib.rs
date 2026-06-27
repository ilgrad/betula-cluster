//! betula-cluster: numerically stable, fast clustering on BETULA CF-trees.
//!
//! See `DESIGN.md` for the full architecture and the verified mathematical foundation.
//! The core stores clustering features as `(n, μ, S)` (weight, mean, sum of squared
//! deviations) and updates/merges them without catastrophic cancellation; covariance is
//! positive semi-definite by construction.

pub mod clustering;
pub mod distance;
pub mod feature;
pub mod kernels;
pub mod linalg;
pub mod model;
pub mod sketch;
pub mod sparse;
pub mod stats;
pub mod stream;
pub mod topology;
pub mod tree;
pub mod types;

#[cfg(feature = "python")]
mod python;
