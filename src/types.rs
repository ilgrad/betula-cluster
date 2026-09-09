//! Numeric scalar type used throughout the crate, and the one shape error its point APIs return.

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

/// A point whose length is not the dimension the structure was built for.
///
/// The checked entry points ([`crate::tree::CFTree::try_insert`] and its siblings) return this
/// rather than reading whatever the shorter slice happens to contain: the SIMD kernels take
/// `a.len().min(b.len())` elements, so a row one column short is silently clustered on its prefix
/// and a row one column long silently drops its last column. Both answer a question the caller did
/// not ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeError {
    /// The dimension the structure was built for.
    pub expected: usize,
    /// The length of the point that was offered.
    pub got: usize,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "point has {} coordinates, this structure is {}-dimensional",
            self.got, self.expected
        )
    }
}

impl std::error::Error for ShapeError {}

impl ShapeError {
    /// `Ok(())` when `got == expected`.
    pub(crate) fn check(expected: usize, got: usize) -> Result<(), Self> {
        if got == expected {
            Ok(())
        } else {
            Err(Self { expected, got })
        }
    }
}
