// Operation enums and trait implementations
pub mod binary;
pub mod gpu_ops;
pub mod matmul;
pub mod movement;
pub mod reduce;
pub mod ternary;
pub mod unary;

// Re-export operation types
pub use binary::BinaryOp;
pub use movement::MovementOp;
pub use reduce::ReduceOp;
pub use ternary::TernaryOp;
pub use unary::UnaryOp;

// ===== LOAD OPERATIONS =====
// DONT KNOW WHERE LOADS GO
//other methods to add:
//to_device LoadOp?

impl RawTensor {
    /// Create empty (zero-filled) tensor (fallible)
    ///
    /// # Errors
    ///
    /// Same as [`RawTensor::try_zeros`]
    pub fn try_empty(shape: &[usize]) -> crate::Result<Tensor> {
        Self::try_zeros(shape)
    }

    /// Create empty (zero-filled) tensor
    ///
    /// # Panics
    ///
    /// If `shape` is invalid. Use [`Self::try_empty`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn empty(shape: &[usize]) -> Tensor {
        Self::try_empty(shape).unwrap_or_else(|e| panic!("empty: {e}"))
    }

    /// Create tensor filled with constant value (fallible)
    ///
    /// # Errors
    ///
    /// Same as [`RawTensor::try_zeros`]
    pub fn try_constant(value: f32, shape: &[usize]) -> crate::Result<Tensor> {
        let size = Self::checked_numel(shape)?;
        Self::try_new(vec![value; size], shape, false)
    }

    /// Create tensor filled with constant value
    ///
    /// # Panics
    ///
    /// If `shape` is invalid. Use [`Self::try_constant`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn constant(value: f32, shape: &[usize]) -> Tensor {
        Self::try_constant(value, shape).unwrap_or_else(|e| panic!("constant: {e}"))
    }

    /// Create tensor from existing Vec (fallible)
    ///
    /// # Errors
    ///
    /// Same as [`RawTensor::try_new`]
    pub fn try_from_vec(data: Vec<f32>, shape: &[usize]) -> crate::Result<Tensor> {
        Self::try_new(data, shape, false)
    }

    /// Create tensor from existing Vec
    ///
    /// # Panics
    ///
    /// If `shape` is invalid or `data.len()` != `shape.product()`. Use
    /// [`Self::try_from_vec`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Self::try_from_vec(data, shape).unwrap_or_else(|e| panic!("from_vec: {e}"))
    }

    /// Ensure tensor is contiguous in memory
    ///
    /// Currently all tensors are contiguous. This would be needed
    /// if we implement views/strides that share memory.
    #[must_use]
    pub fn contiguous(self_t: &Tensor) -> Tensor {
        let s = self_t.borrow();
        Self::from_shape(s.data.to_vec(), s.shape.clone(), s.requires_grad)
    }
}

// Import core types for operation implementations
use crate::{RawTensor, Tensor};
