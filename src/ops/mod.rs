// Operation enums and trait implementations
/// Elementwise binary ops.
pub mod binary;
/// Elementwise comparisons (Bool results).
pub mod compare;
pub mod gpu_ops;
/// Fused losses with their own gradient (binary cross entropy).
pub mod loss;
/// Matrix multiplication.
pub mod matmul;
/// Movement ops (reshape, permute, expand, pad, shrink, stride).
pub mod movement;
/// Reductions (sum, max, mean, and per-axis forms).
pub mod reduce;
/// Ternary ops (`mul_add`, `where_cond`).
pub mod ternary;
/// Elementwise unary ops.
pub mod unary;

// Re-export operation types
pub use binary::BinaryOp;
pub use compare::CmpOp;
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

    /// A CPU leaf tensor (no grad) of any [`Element`](crate::storage::Element) type; its
    /// dtype follows `T` (fallible).
    ///
    /// Separate from [`Self::from_vec`] on purpose: an unsuffixed float literal is `f64` in
    /// Rust, so a generic `from_vec(vec![1.0])` would build an f64 tensor unasked.
    ///
    /// Every op, layer, and optimizer computes in f32, f64, f16 or bf16 on the CPU (the GPU
    /// in f32 only); half types compute in f32 and round each output. Other dtypes can be
    /// built, inspected, saved, and cast. An integer or bool tensor cannot require
    /// gradients (`set_requires_grad(true)` panics).
    ///
    /// # Errors
    ///
    /// Same as [`RawTensor::try_new`]
    pub fn try_from_typed<T: crate::storage::Element>(
        data: Vec<T>,
        shape: &[usize],
    ) -> crate::Result<Tensor> {
        Self::try_new_typed(data, shape, false)
    }

    /// Panicking form of [`Self::try_from_typed`].
    ///
    /// # Panics
    ///
    /// If `shape` is invalid or `data.len()` != `shape.product()`.
    #[must_use]
    #[track_caller]
    pub fn from_typed<T: crate::storage::Element>(data: Vec<T>, shape: &[usize]) -> Tensor {
        Self::try_from_typed(data, shape).unwrap_or_else(|e| panic!("from_typed: {e}"))
    }

    /// Ensure tensor is contiguous in memory.
    ///
    /// Every tensor is contiguous today, so this is a differentiable identity: a
    /// [`reshape`](Self::reshape) to the tensor's own shape. The result is a new graph node
    /// that shares the input's buffer (copy-on-write), keeps its device and dtype, and
    /// passes the gradient back unchanged.
    ///
    /// **When storage gains strided views, this must become its own op.** A same-shape
    /// `reshape` would then stay a view, and `contiguous` would silently stop materializing.
    /// Give it a `MovementOp` that copies the data densely, with an identity backward, and
    /// send the choke points named in DECISIONS.md ("Matmul layout: contiguous-only")
    /// through it. Don't reuse `reshape` there.
    #[must_use]
    pub fn contiguous(self_t: &Tensor) -> Tensor {
        let shape = self_t.borrow().shape.clone();
        Self::reshape(self_t, shape.truncate_rank())
    }
}

// Import core types for operation implementations
use crate::{RawTensor, Tensor};
