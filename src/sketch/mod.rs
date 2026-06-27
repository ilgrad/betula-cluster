//! Streaming quantile sketches (`betula-sketch`): compact, mergeable summaries that answer
//! quantile / rank queries over a stream in bounded memory.
//!
//! - [`KllSketch`] — **rank-error** guarantee `≈ ε·n` (`ε = O(1/k)`), uniform across the
//!   distribution (Karnin–Lang–Liberty).
//! - [`DdSketch`] — **relative-error** guarantee `α`, ideal for skewed / positive / long-tailed data
//!   such as latencies (Masson–Rim–Lee).
//!
//! Both work on `f64`, support `merge`, and track exact min/max. Standalone today; a natural future
//! step is a sketch per microcluster for robust per-cluster quantiles.

mod ddsketch;
mod kll;

pub use ddsketch::DdSketch;
pub use kll::KllSketch;
