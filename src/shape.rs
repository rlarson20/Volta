//! Newtypes for tensor shapes, strides, and axis indices.
//!
//! `Shape` replaces ad hoc `Vec<usize>`/`&[usize]` shape parameters across the
//! crate, and `Axis` replaces bare `usize` dimension indices. `Strides`
//! carries the row-major stride computation that used to live as a free
//! function on `RawTensor`.

use crate::error::{Result, VoltaError};

/// Largest element count a tensor may have (guards against runaway allocations).
pub const MAX_NUMEL: usize = 100_000_000;

/// Pads `dims` with leading `1`s (right-aligning it to `len`) for broadcasting.
fn left_pad_ones(dims: &[usize], len: usize) -> impl Iterator<Item = usize> + '_ {
    std::iter::repeat_n(1, len - dims.len()).chain(dims.iter().copied())
}

/// The dimensions of a tensor, e.g. `[batch, channels, height, width]`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(Vec<usize>);

/// Row-major memory strides for a [`Shape`].
///
/// For shape `[3, 4, 5]`, strides are `[20, 5, 1]`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Strides(Vec<usize>);

/// An index into a [`Shape`]'s dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Axis(pub usize);

impl Shape {
    /// Builds a `Shape` from a list of dimensions.
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::InvalidParameter` if any dimension is `0`, if the
    /// product of `dims` (the element count) would overflow `usize`, or if it
    /// exceeds [`MAX_NUMEL`].
    pub fn from_dims(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.contains(&0) {
            return Err(VoltaError::InvalidParameter(format!(
                "shape {dims:?} has a zero dimension"
            )));
        }
        let numel = dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                VoltaError::InvalidParameter(format!(
                    "shape {dims:?} overflows usize while computing element count"
                ))
            })?;
        if numel > MAX_NUMEL {
            return Err(VoltaError::InvalidParameter(format!(
                "shape {dims:?} has {numel} elements > {MAX_NUMEL} - check for memory issues"
            )));
        }
        Ok(Self(dims))
    }

    /// Builds a `Shape` for a size an op computed from tensors that already exist.
    ///
    /// An error here means the op's shape arithmetic is wrong, not that the caller
    /// passed bad input, so it panics instead of returning a `Result`.
    ///
    /// # Panics
    ///
    /// If [`Shape::from_dims`] rejects `dims`.
    #[track_caller]
    pub(crate) fn derived(dims: impl Into<Vec<usize>>) -> Self {
        Self::from_dims(dims).expect("op computed an invalid output shape")
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements (product of all dimensions).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Computes row-major strides for this shape.
    #[must_use]
    pub fn strides(&self) -> Strides {
        Strides::from_shape(&self.0)
    }

    /// Computes the broadcast result of two shapes.
    ///
    /// Shapes are aligned from the right (trailing dimensions); for each
    /// aligned pair of dimensions both must be equal, or one must be `1`.
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::BroadcastError` if the shapes cannot be
    /// broadcast together.
    pub fn broadcast(&self, other: &Self) -> Result<Self> {
        let len = self.0.len().max(other.0.len());
        let mut result = Vec::with_capacity(len);
        for (a, b) in left_pad_ones(&self.0, len).zip(left_pad_ones(&other.0, len)) {
            let dim = match (a, b) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                _ => return Err(VoltaError::BroadcastError(self.0.clone(), other.0.clone())),
            };
            result.push(dim);
        }
        Self::from_dims(result)
    }

    /// Returns the size of the given axis.
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::DimensionOutOfBounds` if `axis` is not a valid
    /// dimension of this shape.
    pub fn dim(&self, axis: Axis) -> Result<usize> {
        self.0
            .get(axis.0)
            .copied()
            .ok_or_else(|| VoltaError::DimensionOutOfBounds {
                dim: axis.0,
                shape: self.0.clone(),
            })
    }

    /// Size of dimension `i`, for a caller that has already bounded `i` by the rank
    /// (a rank-bounded loop, or after a rank check).
    ///
    /// # Panics
    ///
    /// If `i >= rank`, naming the caller. Use [`Self::dim`] when `i` comes from a user.
    #[track_caller]
    #[must_use]
    pub(crate) fn dim_at(&self, i: usize) -> usize {
        crate::index::fetch(&self.0, i)
    }

    /// The dims as an array, for a caller that has already checked the rank is `N`.
    ///
    /// # Panics
    ///
    /// If the rank is not `N`, naming the caller.
    #[track_caller]
    #[must_use]
    pub(crate) fn fixed<const N: usize>(&self) -> [usize; N] {
        fixed_dims(&self.0)
    }

    /// The dims in order, as values.
    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = usize> + ExactSizeIterator + '_ {
        self.0.iter().copied()
    }

    /// Product of the dims before position `i` (the "outer" size of a reduction over `i`).
    ///
    /// # Panics
    ///
    /// If `i > rank`.
    #[track_caller]
    #[must_use]
    pub(crate) fn numel_before(&self, i: usize) -> usize {
        crate::index::span(&self.0, 0..i).iter().product()
    }

    /// Product of the dims from position `i` on. `numel_from(rank)` is `1`.
    ///
    /// # Panics
    ///
    /// If `i > rank`.
    #[track_caller]
    #[must_use]
    pub(crate) fn numel_from(&self, i: usize) -> usize {
        crate::index::span(&self.0, i..self.0.len())
            .iter()
            .product()
    }

    /// This shape with a size-1 axis inserted at position `i` (keepdim reductions).
    ///
    /// # Panics
    ///
    /// If `i > rank`.
    #[track_caller]
    #[must_use]
    pub(crate) fn unsqueeze(&self, i: usize) -> Self {
        assert!(
            i <= self.0.len(),
            "unsqueeze at {i} past rank {}",
            self.0.len()
        );
        let mut dims = self.0.clone();
        dims.insert(i, 1);
        Self(dims)
    }

    /// This shape with dimension `i` set to `size` (a reduction's keepdim output).
    ///
    /// # Panics
    ///
    /// If `i >= rank`.
    #[track_caller]
    #[must_use]
    pub(crate) fn with_dim(&self, i: usize, size: usize) -> Self {
        let mut dims = self.0.clone();
        *crate::index::slot(&mut dims, i) = size;
        Self::derived(dims)
    }

    /// This shape with dimension `i` removed (a reduction's non-keepdim output).
    ///
    /// # Panics
    ///
    /// If `i >= rank`.
    #[track_caller]
    #[must_use]
    pub(crate) fn squeeze(&self, i: usize) -> Self {
        assert!(
            i < self.0.len(),
            "squeeze at {i} past rank {}",
            self.0.len()
        );
        let mut dims = self.0.clone();
        dims.remove(i);
        Self(dims)
    }

    /// Named, explicit demotion to a raw `&[usize]` view of the dimensions.
    ///
    /// Deliberately not exposed via `Deref`/`AsRef` — every place that steps
    /// outside the `Shape` abstraction must spell out `.truncate_rank()`, so
    /// it stays grep-able.
    #[must_use]
    pub fn truncate_rank(&self) -> &[usize] {
        &self.0
    }
}

// Comparing a shape with literal dims (`assert_eq!(t.shape(), [2, 3])`) is equality,
// not a demotion: it hands no `&[usize]` to a shape-blind API, so it needs no
// `truncate_rank()`.
impl PartialEq<[usize]> for Shape {
    fn eq(&self, other: &[usize]) -> bool {
        self.0 == other
    }
}

impl<const N: usize> PartialEq<[usize; N]> for Shape {
    fn eq(&self, other: &[usize; N]) -> bool {
        self.0 == other
    }
}

impl PartialEq<Vec<usize>> for Shape {
    fn eq(&self, other: &Vec<usize>) -> bool {
        &self.0 == other
    }
}

impl PartialEq<Shape> for Vec<usize> {
    fn eq(&self, other: &Shape) -> bool {
        *self == other.0
    }
}

impl Strides {
    /// Computes row-major strides for `shape`.
    ///
    /// Each stride is the product of the dimensions to its right, so the last is `1`.
    /// Takes a slice rather than a [`Shape`] because op code usually holds raw dims.
    pub(crate) fn from_shape(shape: &[usize]) -> Self {
        let mut running = 1usize;
        let mut strides: Vec<usize> = shape
            .iter()
            .rev()
            .map(|&dim| {
                let stride = running;
                running *= dim;
                stride
            })
            .collect();
        strides.reverse();
        Self(strides)
    }

    /// Named, explicit demotion to a raw `&[usize]` view of the strides.
    #[must_use]
    pub fn truncate_rank(&self) -> &[usize] {
        &self.0
    }
}

/// The dims of a raw shape whose rank the caller already knows is `N`, as an array.
///
/// Replaces the positional `shape.get(i).copied()` + size-1 default reads (DECISIONS.md,
/// "Indexing policy"): a missing dim is a rank bug upstream, not a size-1 axis.
///
/// # Panics
///
/// If `dims.len() != N`, naming the caller.
#[track_caller]
pub(crate) fn fixed_dims<const N: usize>(dims: &[usize]) -> [usize; N] {
    <[usize; N]>::try_from(dims)
        .unwrap_or_else(|_| panic!("expected a rank-{N} shape, got {dims:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_dims_reads_exact_rank() {
        assert_eq!(fixed_dims::<2>(&[3, 4]), [3, 4]);
    }

    #[test]
    #[should_panic(expected = "expected a rank-2 shape, got [3]")]
    fn fixed_dims_rejects_wrong_rank() {
        let _ = fixed_dims::<2>(&[3]);
    }

    #[test]
    fn strides_of_3x4x5_are_row_major() {
        let strides = Strides::from_shape(&[3, 4, 5]);
        assert_eq!(strides.truncate_rank(), &[20, 5, 1]);
    }

    #[test]
    fn strides_of_rank_0_is_empty() {
        let strides = Strides::from_shape(&[]);
        assert_eq!(strides.truncate_rank(), &[] as &[usize]);
    }

    #[test]
    fn strides_of_single_dim() {
        let strides = Strides::from_shape(&[7]);
        assert_eq!(strides.truncate_rank(), &[1]);
    }

    #[test]
    fn from_dims_reports_rank_and_numel() {
        let shape = Shape::from_dims(vec![2, 3]).unwrap();
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.numel(), 6);
    }

    #[test]
    fn from_dims_scalar_shape() {
        let shape = Shape::from_dims(Vec::<usize>::new()).unwrap();
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.numel(), 1);
    }

    #[test]
    fn from_dims_overflow_errs() {
        let result = Shape::from_dims(vec![usize::MAX, 2]);
        assert!(matches!(result, Err(VoltaError::InvalidParameter(_))));
    }

    #[test]
    fn from_dims_zero_dim_errs() {
        let result = Shape::from_dims(vec![3, 0, 2]);
        assert!(matches!(result, Err(VoltaError::InvalidParameter(_))));
    }

    #[test]
    fn from_dims_at_max_numel_is_ok() {
        assert!(Shape::from_dims(vec![MAX_NUMEL]).is_ok());
    }

    #[test]
    fn from_dims_over_max_numel_errs() {
        let result = Shape::from_dims(vec![MAX_NUMEL + 1]);
        assert!(matches!(result, Err(VoltaError::InvalidParameter(_))));
    }

    #[test]
    #[should_panic(expected = "op computed an invalid output shape")]
    fn derived_panics_on_zero_dim() {
        let _ = Shape::derived(vec![2, 0]);
    }

    #[test]
    fn shape_strides_delegate_to_row_major() {
        let shape = Shape::from_dims(vec![3, 4, 5]).unwrap();
        assert_eq!(shape.strides().truncate_rank(), &[20, 5, 1]);
    }

    #[test]
    fn broadcast_aligns_from_the_right() {
        let a = Shape::from_dims(vec![3, 1]).unwrap();
        let b = Shape::from_dims(vec![1, 4]).unwrap();
        let result = a.broadcast(&b).unwrap();
        assert_eq!(result, Shape::from_dims(vec![3, 4]).unwrap());
    }

    #[test]
    fn broadcast_extends_rank() {
        let a = Shape::from_dims(vec![5, 3, 1]).unwrap();
        let b = Shape::from_dims(vec![1, 4]).unwrap();
        let result = a.broadcast(&b).unwrap();
        assert_eq!(result, Shape::from_dims(vec![5, 3, 4]).unwrap());
    }

    #[test]
    fn broadcast_incompatible_dims_errs() {
        let a = Shape::from_dims(vec![3, 4]).unwrap();
        let b = Shape::from_dims(vec![2, 4]).unwrap();
        let result = a.broadcast(&b);
        assert!(matches!(result, Err(VoltaError::BroadcastError(_, _))));
    }

    #[test]
    fn dim_in_bounds() {
        let shape = Shape::from_dims(vec![3, 4, 5]).unwrap();
        assert_eq!(shape.dim(Axis(1)).unwrap(), 4);
    }

    #[test]
    fn dim_out_of_bounds_errs() {
        let shape = Shape::from_dims(vec![3, 4]).unwrap();
        let result = shape.dim(Axis(5));
        assert!(matches!(
            result,
            Err(VoltaError::DimensionOutOfBounds { dim: 5, .. })
        ));
    }

    #[test]
    fn truncate_rank_is_raw_slice() {
        let shape = Shape::from_dims(vec![3, 4]).unwrap();
        assert_eq!(shape.truncate_rank(), &[3, 4]);
    }

    #[test]
    fn shapes_with_equal_dims_are_equal() {
        assert_eq!(
            Shape::from_dims(vec![1, 2]).unwrap(),
            Shape::from_dims(vec![1, 2]).unwrap()
        );
        assert_ne!(
            Shape::from_dims(vec![1, 2]).unwrap(),
            Shape::from_dims(vec![2, 1]).unwrap()
        );
    }

    #[test]
    fn axis_is_copy_and_comparable() {
        let a = Axis(2);
        let b = a;
        assert_eq!(a, b);
        assert_ne!(Axis(1), Axis(2));
    }
}
