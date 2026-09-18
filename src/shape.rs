//! Newtypes for tensor shapes, strides, and axis indices.
//!
//! `Shape` replaces ad hoc `Vec<usize>`/`&[usize]` shape parameters across the
//! crate, and `Axis` replaces bare `usize` dimension indices. `Strides`
//! carries the row-major stride computation that used to live as a free
//! function on `RawTensor`.

use crate::error::{Result, VoltaError};

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
    /// Returns `VoltaError::InvalidParameter` if the product of `dims`
    /// (the element count) would overflow `usize`.
    pub fn from_dims(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        dims.iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                VoltaError::InvalidParameter(format!(
                    "shape {dims:?} overflows usize while computing element count"
                ))
            })?;
        Ok(Self(dims))
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
        self.0.get(axis.0).copied().ok_or_else(|| VoltaError::DimensionOutOfBounds {
            dim: axis.0,
            shape: self.0.clone(),
        })
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

impl Strides {
    /// Computes row-major strides for `shape`.
    fn from_shape(shape: &[usize]) -> Self {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            if let Some(&stride_val) = strides.get(i + 1)
                && let Some(&shape_val) = shape.get(i + 1)
                && let Some(slot) = strides.get_mut(i)
            {
                *slot = stride_val * shape_val;
            }
        }
        Self(strides)
    }

    /// Named, explicit demotion to a raw `&[usize]` view of the strides.
    #[must_use]
    pub fn truncate_rank(&self) -> &[usize] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
