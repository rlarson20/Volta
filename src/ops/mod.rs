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
    /// Create empty (zero-filled) tensor
    #[must_use]
    pub fn empty(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }
    /// Create tensor filled with constant value
    #[must_use]
    pub fn constant(value: f32, shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![value; size], shape, false)
    }
    /// Create tensor from existing Vec
    #[must_use]
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Self::new(data, shape, false)
    }

    /// Ensure tensor is contiguous in memory
    ///
    /// Currently all tensors are contiguous. This would be needed
    /// if we implement views/strides that share memory.
    #[must_use]
    pub fn contiguous(self_t: &Tensor) -> Tensor {
        let s = self_t.borrow();
        Self::new(s.data.to_vec(), &s.shape, s.requires_grad)
    }
}

// Import core types for operation implementations
use crate::{RawTensor, Tensor};
