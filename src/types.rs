//! Numeric scalar type used throughout the crate.

/// Real scalar (`f32` or `f64`) with the operations the clustering math needs.
///
/// `Send + Sync + 'static` keep features usable across rayon worker threads; `Sum` and
/// `FromPrimitive` cover reductions and `usize -> R` conversions.
pub trait Real:
    num_traits::Float
    + num_traits::FromPrimitive
    + std::iter::Sum
    + std::fmt::Debug
    + Send
    + Sync
    + 'static
{
}

impl<T> Real for T where
    T: num_traits::Float
        + num_traits::FromPrimitive
        + std::iter::Sum
        + std::fmt::Debug
        + Send
        + Sync
        + 'static
{
}
