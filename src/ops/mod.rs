// Operation enums and trait implementations
pub mod binary;
pub mod matmul;
pub mod movement;
pub mod reduce;
pub mod ternary;
pub mod unary;

// Re-export operation types
pub use binary::{BinaryGradFn, BinaryOp};
pub use matmul::MatMulGradFn;
pub use movement::{MovementGradFn, MovementOp};
pub use reduce::{MaxReduceGradFn, MeanGradFn, ReduceOp, SumGradFn};
pub use ternary::{MulAccGradFn, TernaryOp, WhereGradFn};
pub use unary::{UnaryGradFn, UnaryOp};

// ===== LOAD OPERATIONS =====
// DONT KNOW WHERE LOADS GO
//other methods to add:
//to_device LoadOp?

impl RawTensor {
    /// Create empty (zero-filled) tensor
    pub fn empty(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }
    /// Create tensor filled with constant value
    pub fn constant(value: f32, shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![value; size], shape, false)
    }
    /// Create tensor from existing Vec
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Self::new(data, shape, false)
    }

    /// Ensure tensor is contiguous in memory
    ///
    /// Currently all tensors are contiguous. This would be needed
    /// if we implement views/strides that share memory.
    pub fn contiguous(self_t: &Tensor) -> Tensor {
        let s = self_t.borrow();
        Self::new(s.data.clone(), &s.shape, s.requires_grad)
    }
}

// Import core types for operation implementations
use crate::{RawTensor, Tensor};
