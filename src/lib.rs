//! # Torch From Scratch / Volta
//!
//! A minimal automatic differentiation library implementing PyTorch-like tensor operations
//! from scratch in pure Rust. This library provides:
//! - Dynamic computation graphs for automatic differentiation
//! - Broadcasting support for tensor operations
//! - Common neural network operations (matmul, activations, etc.)
//! - Numerical gradient checking for validation
//!
//! ## Architecture
//!
//! The library uses reference-counted interior mutability (`Rc<RefCell<RawTensor>>`) to build
//! dynamic computation graphs. Each tensor operation creates new tensors and stores gradient
//! functions that know how to backpropagate through that operation.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// Type alias for a reference-counted, interior-mutable tensor.
///
/// We use `Rc<RefCell<RawTensor>>` to allow multiple references to the same tensor
/// (needed for computation graphs) while still allowing mutation (for gradient accumulation).
///
/// **Note for production**: This is single-threaded only. For multi-threading,
/// replace with `Arc<Mutex<RawTensor>>`.
pub type Tensor = Rc<RefCell<RawTensor>>;

// ===== OPERATION ENUMS =====
// These enums categorize all operations our tensor library supports.
// Each category has a corresponding gradient implementation.

/// Unary operations: single input, single output
///
/// Each operation has a corresponding derivative:
/// - Neg: d(-x)/dx = -1
/// - Recip: d(1/x)/dx = -1/x²
/// - Sqrt: d(√x)/dx = 1/(2√x)
/// - Exp: d(eˣ)/dx = eˣ
/// - Log: d(ln(x))/dx = 1/x
/// - Exp2: d(2ˣ)/dx = 2ˣ·ln(2)
/// - Log2: d(log₂(x))/dx = 1/(x·ln(2))
/// - Sin: d(sin(x))/dx = cos(x)
/// - Cos: d(cos(x))/dx = -sin(x)
/// - Tanh: d(tanh(x))/dx = 1 - tanh²(x)
/// - Sigmoid: d(σ(x))/dx = σ(x)·(1-σ(x))
/// - ReLU: d(max(0,x))/dx = x > 0 ? 1 : 0
#[derive(Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Recip,
    Sqrt,
    Exp2,
    Exp,
    Log2,
    Log,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    ReLU,
}

/// Binary operations: two inputs, one output
///
/// Broadcasting is automatically handled for compatible shapes.
/// Non-differentiable operations (Mod, Cmplt) return tensors with requires_grad=false.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,   // x + y
    Sub,   // x - y
    Mul,   // x * y (element-wise)
    Div,   // x / y (element-wise)
    Max,   // max(x, y) (element-wise)
    Mod,   // x % y (non-differentiable)
    Cmplt, // x < y ? 1 : 0 (non-differentiable)
}

/// Reduction operations: reduce tensor to scalar
///
/// These operations collapse all dimensions and require special gradient handling
/// since the output shape differs from the input.
#[derive(Clone, Copy)]
pub enum ReduceOp {
    Sum,  // Σ(x) - gradient broadcasts ones
    Max,  // max(x) - gradient goes only to max element
    Mean, // mean(x) - gradient broadcasts 1/n
}

/// Ternary operations: three inputs, one output
#[derive(Clone, Copy)]
pub enum TernaryOp {
    MulAcc, // x*y + z (fused multiply-accumulate)
    Where,  // condition ? x : y (masked selection)
}

/// Movement operations: reshape/reorder data without changing values
///
/// These operations don't modify data values, only how they're indexed.
/// Gradients must "undo" these operations during backpropagation.
#[derive(Clone)]
pub enum MovementOp {
    Reshape { new_shape: Vec<usize> }, // Change shape, preserve order
    Permute { axes: Vec<usize> },      // Transpose/reorder axes
    Expand { new_shape: Vec<usize> },  // Broadcast to larger shape
    Pad { padding: Vec<(usize, usize)> }, // Add zeros around edges
    Shrink { ranges: Vec<(usize, usize)> }, // Extract subregion
    Stride { strides: Vec<usize> },    // Subsample with stride
}

/// Load operations: tensor creation without computation graph
///
/// These are "leaf" operations that don't have gradients to backpropagate.
pub enum LoadOp {
    Empty,      // Allocate uninitialized
    Rand,       // Random uniform [0,1)
    Const,      // Filled with constant
    From,       // From Vec<f32>
    Contiguous, // Ensure contiguous memory
    Custom,     // User-defined
}

// ===== GRADIENT FUNCTION TRAIT =====

/// Trait for gradient computation functions.
///
/// Each operation type implements this to define how gradients flow backward.
/// The `backward` method takes:
/// - `out_grad`: gradient of loss w.r.t. this operation's output
/// - `parents`: the input tensors to this operation
///
/// Returns: vector of gradients w.r.t. each parent (Some if requires_grad, None otherwise)
pub trait GradFn {
    /// Compute gradients for parent tensors given output gradient
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>>;
    /// Clone this gradient function (needed for Rc/RefCell)
    fn clone_box(&self) -> Box<dyn GradFn>;
}

// ===== GRADIENT FUNCTION IMPLEMENTATIONS =====

/// Gradient function for unary operations
///
/// Stores which operation was performed so backward can apply the correct derivative.
struct UnaryGradFn {
    op: UnaryOp,
}

impl GradFn for UnaryGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x = parents[0].borrow();

        // Apply chain rule: ∂L/∂x = ∂L/∂y · ∂y/∂x
        // where y = f(x) is the unary operation
        let grad_data: Vec<f32> = match self.op {
            UnaryOp::Neg => out_grad.data.iter().map(|&g| -g).collect(),
            UnaryOp::Recip => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| -g / (x * x))
                .collect(),
            UnaryOp::Sqrt => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| g / (2.0 * x.sqrt()))
                .collect(),
            UnaryOp::Exp => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| g * x.exp())
                .collect(),
            UnaryOp::Log => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| g / x)
                .collect(),
            UnaryOp::Exp2 => {
                let ln2 = std::f32::consts::LN_2;
                out_grad
                    .data
                    .iter()
                    .zip(&x.data)
                    .map(|(&g, &x)| g * 2_f32.powf(x) * ln2)
                    .collect()
            }
            UnaryOp::Log2 => {
                let ln2 = std::f32::consts::LN_2;
                out_grad
                    .data
                    .iter()
                    .zip(&x.data)
                    .map(|(&g, &x)| g / (x * ln2))
                    .collect()
            }
            UnaryOp::Sin => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| g * x.cos())
                .collect(),
            UnaryOp::Cos => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| -g * x.sin())
                .collect(),
            UnaryOp::Tanh => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| {
                    let t = x.tanh();
                    g * (1.0 - t * t)
                })
                .collect(),
            UnaryOp::Sigmoid => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| {
                    let s = 1.0 / (1.0 + (-x).exp());
                    g * s * (1.0 - s)
                })
                .collect(),
            UnaryOp::ReLU => out_grad
                .data
                .iter()
                .zip(&x.data)
                .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                .collect(),
        };
        vec![Some(RawTensor::new(grad_data, &x.shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(UnaryGradFn { op: self.op })
    }
}

/// Gradient function for binary operations
///
/// Handles broadcasting during backward pass - gradients must be summed
/// over dimensions that were broadcast in the forward pass.
struct BinaryGradFn {
    op: BinaryOp,
}

impl GradFn for BinaryGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x_val = parents[0].borrow();
        let y_val = parents[1].borrow();

        let (grad_x, grad_y) = match self.op {
            BinaryOp::Add => {
                // ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
                // But must sum over broadcast dimensions
                let gx = if x_val.requires_grad {
                    let summed = RawTensor::sum_over_broadcast_dims(
                        &out_grad.data,
                        &out_grad.shape,
                        &x_val.shape,
                    );
                    Some(RawTensor::new(summed, &x_val.shape, false))
                } else {
                    None
                };
                let gy = if y_val.requires_grad {
                    let summed = RawTensor::sum_over_broadcast_dims(
                        &out_grad.data,
                        &out_grad.shape,
                        &y_val.shape,
                    );
                    Some(RawTensor::new(summed, &y_val.shape, false))
                } else {
                    None
                };
                (gx, gy)
            }
            BinaryOp::Sub => {
                // ∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1
                let gx = if x_val.requires_grad {
                    let summed = RawTensor::sum_over_broadcast_dims(
                        &out_grad.data,
                        &out_grad.shape,
                        &x_val.shape,
                    );
                    Some(RawTensor::new(summed, &x_val.shape, false))
                } else {
                    None
                };
                let gy = if y_val.requires_grad {
                    let neg_grad: Vec<f32> = out_grad.data.iter().map(|&g| -g).collect();
                    let summed = RawTensor::sum_over_broadcast_dims(
                        &neg_grad,
                        &out_grad.shape,
                        &y_val.shape,
                    );
                    Some(RawTensor::new(summed, &y_val.shape, false))
                } else {
                    None
                };
                (gx, gy)
            }
            BinaryOp::Mul => {
                // ∂(x*y)/∂x = y, ∂(x*y)/∂y = x
                let gx = if x_val.requires_grad {
                    // Broadcast y to out_grad shape for multiplication
                    let y_bc = RawTensor::broadcast_to(&y_val.data, &y_val.shape, &out_grad.shape);
                    let grad: Vec<f32> = out_grad
                        .data
                        .iter()
                        .zip(&y_bc)
                        .map(|(&g, &y)| g * y)
                        .collect();
                    let summed =
                        RawTensor::sum_over_broadcast_dims(&grad, &out_grad.shape, &x_val.shape);
                    Some(RawTensor::new(summed, &x_val.shape, false))
                } else {
                    None
                };
                let gy = if y_val.requires_grad {
                    let x_bc = RawTensor::broadcast_to(&x_val.data, &x_val.shape, &out_grad.shape);
                    let grad: Vec<f32> = out_grad
                        .data
                        .iter()
                        .zip(&x_bc)
                        .map(|(&g, &x)| g * x)
                        .collect();
                    let summed =
                        RawTensor::sum_over_broadcast_dims(&grad, &out_grad.shape, &y_val.shape);
                    Some(RawTensor::new(summed, &y_val.shape, false))
                } else {
                    None
                };
                (gx, gy)
            }
            BinaryOp::Div => {
                // ∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y²
                let gx = if x_val.requires_grad {
                    let y_bc = RawTensor::broadcast_to(&y_val.data, &y_val.shape, &out_grad.shape);
                    let grad: Vec<f32> = out_grad
                        .data
                        .iter()
                        .zip(&y_bc)
                        .map(|(&g, &y)| g / y)
                        .collect();
                    let summed =
                        RawTensor::sum_over_broadcast_dims(&grad, &out_grad.shape, &x_val.shape);
                    Some(RawTensor::new(summed, &x_val.shape, false))
                } else {
                    None
                };
                let gy = if y_val.requires_grad {
                    let x_bc = RawTensor::broadcast_to(&x_val.data, &x_val.shape, &out_grad.shape);
                    let y_bc = RawTensor::broadcast_to(&y_val.data, &y_val.shape, &out_grad.shape);
                    let grad: Vec<f32> = out_grad
                        .data
                        .iter()
                        .zip(&x_bc)
                        .zip(&y_bc)
                        .map(|((&g, &x), &y)| -g * x / (y * y))
                        .collect();
                    let summed =
                        RawTensor::sum_over_broadcast_dims(&grad, &out_grad.shape, &y_val.shape);
                    Some(RawTensor::new(summed, &y_val.shape, false))
                } else {
                    None
                };
                (gx, gy)
            }
            BinaryOp::Max => {
                // Gradient flows to whichever input was larger
                let gx = if x_val.requires_grad {
                    let x_bc = RawTensor::broadcast_to(&x_val.data, &x_val.shape, &out_grad.shape);
                    let y_bc = RawTensor::broadcast_to(&y_val.data, &y_val.shape, &out_grad.shape);
                    let grad: Vec<f32> = out_grad
                        .data
                        .iter()
                        .zip(&x_bc)
                        .zip(&y_bc)
                        .map(|((&g, &x), &y)| if x >= y { g } else { 0.0 })
                        .collect();
                    let summed =
                        RawTensor::sum_over_broadcast_dims(&grad, &out_grad.shape, &x_val.shape);
                    Some(RawTensor::new(summed, &x_val.shape, false))
                } else {
                    None
                };
                let gy = if y_val.requires_grad {
                    let x_bc = RawTensor::broadcast_to(&x_val.data, &x_val.shape, &out_grad.shape);
                    let y_bc = RawTensor::broadcast_to(&y_val.data, &y_val.shape, &out_grad.shape);
                    let grad: Vec<f32> = out_grad
                        .data
                        .iter()
                        .zip(&x_bc)
                        .zip(&y_bc)
                        .map(|((&g, &x), &y)| if y > x { g } else { 0.0 })
                        .collect();
                    let summed =
                        RawTensor::sum_over_broadcast_dims(&grad, &out_grad.shape, &y_val.shape);
                    Some(RawTensor::new(summed, &y_val.shape, false))
                } else {
                    None
                };
                (gx, gy)
            }
            BinaryOp::Mod | BinaryOp::Cmplt => {
                // Non-differentiable
                (None, None)
            }
        };

        vec![grad_x, grad_y]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(BinaryGradFn { op: self.op })
    }
}

/// Gradient function for Sum reduction
///
/// Sum reduction collapses to scalar, so gradient broadcasts back to original shape.
struct SumGradFn {
    input_shape: Vec<usize>,
}

impl GradFn for SumGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        let grad_val: f32 = out_grad.data[0];
        vec![Some(RawTensor::new(
            vec![grad_val; size],
            &self.input_shape,
            false,
        ))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(SumGradFn {
            input_shape: self.input_shape.clone(),
        })
    }
}

/// Gradient function for Max reduction
///
/// Only the maximum element receives gradient; all others get zero.
struct MaxReduceGradFn {
    input_shape: Vec<usize>,
    max_index: usize, // Linear index of the maximum element
}

impl GradFn for MaxReduceGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        let mut grad_data = vec![0.0; size];
        grad_data[self.max_index] = out_grad.data[0];
        vec![Some(RawTensor::new(grad_data, &self.input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MaxReduceGradFn {
            input_shape: self.input_shape.clone(),
            max_index: self.max_index,
        })
    }
}

/// Gradient function for Mean reduction
///
/// Each element gets gradient / num_elements.
struct MeanGradFn {
    input_shape: Vec<usize>,
}

impl GradFn for MeanGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        let grad_val = out_grad.data[0] / (size as f32);
        vec![Some(RawTensor::new(
            vec![grad_val; size],
            &self.input_shape,
            false,
        ))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MeanGradFn {
            input_shape: self.input_shape.clone(),
        })
    }
}

/// Gradient for sum_dim: broadcast ones back to input shape
struct SumDimGradFn {
    input_shape: Vec<usize>,
    dim: usize,
    keepdim: bool,
}

impl GradFn for SumDimGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = &out_grad.data;
        let grad_shape = &out_grad.shape;

        // If keepdim=false, we need to unsqueeze the dimension back
        let mut expanded_shape = grad_shape.clone();
        if !self.keepdim {
            expanded_shape.insert(self.dim, 1);
        }

        // Broadcast gradient back to input shape
        // Each output element contributed to by input_shape[dim] elements
        let size: usize = self.input_shape.iter().product();
        let mut result = vec![0.0; size];

        let _strides = RawTensor::compute_strides(&self.input_shape);
        let grad_strides = RawTensor::compute_strides(&expanded_shape);

        for i in 0..size {
            // Get input coordinates
            let mut coords = vec![0; self.input_shape.len()];
            let mut rem = i;
            for (d, &dim_sz) in self.input_shape.iter().enumerate().rev() {
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            // Map to gradient coordinates (zero out the summed dimension)
            let mut grad_coords = coords.clone();
            grad_coords[self.dim] = 0;

            let grad_idx: usize = grad_coords
                .iter()
                .zip(&grad_strides)
                .map(|(c, s)| c * s)
                .sum();
            result[i] = grad_data[grad_idx];
        }

        vec![Some(RawTensor::new(result, &self.input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(SumDimGradFn {
            input_shape: self.input_shape.clone(),
            dim: self.dim,
            keepdim: self.keepdim,
        })
    }
}

/// Gradient for max_dim: sparse gradient to max elements only
struct MaxDimGradFn {
    input_shape: Vec<usize>,
    max_indices: Vec<usize>, // linear indices of max elements
    dim: usize,
    keepdim: bool,
}

impl GradFn for MaxDimGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = &out_grad.data;
        let grad_shape = &out_grad.shape;

        let mut expanded_shape = grad_shape.clone();
        if !self.keepdim {
            expanded_shape.insert(self.dim, 1);
        }

        let size: usize = self.input_shape.iter().product();
        let mut result = vec![0.0; size];

        // Only max elements receive gradient
        let grad_strides = RawTensor::compute_strides(&expanded_shape);

        for (out_idx, &max_lin_idx) in self.max_indices.iter().enumerate() {
            // Convert output index to coordinates in expanded shape
            let mut grad_coords = vec![0; expanded_shape.len()];
            let mut rem = out_idx;
            for (d, &dim_sz) in expanded_shape.iter().enumerate().rev() {
                grad_coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            let grad_idx: usize = grad_coords
                .iter()
                .zip(&grad_strides)
                .map(|(c, s)| c * s)
                .sum();
            result[max_lin_idx] = grad_data[grad_idx];
        }

        vec![Some(RawTensor::new(result, &self.input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MaxDimGradFn {
            input_shape: self.input_shape.clone(),
            max_indices: self.max_indices.clone(),
            dim: self.dim,
            keepdim: self.keepdim,
        })
    }
}

/// Gradient function for MulAcc (fused multiply-add)
///
/// z = x*y + w has gradients:
/// - ∂z/∂x = y
/// - ∂z/∂y = x
/// - ∂z/∂w = 1
struct MulAccGradFn;
impl GradFn for MulAccGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x_val = parents[0].borrow();
        let y_val = parents[1].borrow();

        let grad_x = if x_val.requires_grad {
            let data = out_grad
                .data
                .iter()
                .zip(&y_val.data)
                .map(|(&g, &y)| g * y)
                .collect();
            Some(RawTensor::new(data, &out_grad.shape, false))
        } else {
            None
        };

        let grad_y = if y_val.requires_grad {
            let data = out_grad
                .data
                .iter()
                .zip(&x_val.data)
                .map(|(&g, &x)| g * x)
                .collect();
            Some(RawTensor::new(data, &out_grad.shape, false))
        } else {
            None
        };

        let grad_z = if parents[2].borrow().requires_grad {
            Some(RawTensor::new(
                out_grad.data.clone(),
                &out_grad.shape,
                false,
            ))
        } else {
            None
        };

        vec![grad_x, grad_y, grad_z]
    }
    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MulAccGradFn)
    }
}

/// Gradient function for Where (conditional selection)
///
/// Gradient flows through the branch that was selected.
/// The condition tensor itself is not differentiable.
struct WhereGradFn {
    condition: Vec<f32>,
}
impl GradFn for WhereGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x_requires = parents[0].borrow().requires_grad;
        let y_requires = parents[1].borrow().requires_grad;

        let grad_x = if x_requires {
            let data = out_grad
                .data
                .iter()
                .zip(&self.condition)
                .map(|(&g, &c)| if c != 0.0 { g } else { 0.0 })
                .collect();
            Some(RawTensor::new(data, &out_grad.shape, false))
        } else {
            None
        };

        let grad_y = if y_requires {
            let data = out_grad
                .data
                .iter()
                .zip(&self.condition)
                .map(|(&g, &c)| if c == 0.0 { g } else { 0.0 })
                .collect();
            Some(RawTensor::new(data, &out_grad.shape, false))
        } else {
            None
        };

        vec![grad_x, grad_y]
    }
    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(WhereGradFn {
            condition: self.condition.clone(),
        })
    }
}

/// Gradient function for matrix multiplication
///
/// For z = x @ y:
/// - ∂L/∂x = ∂L/∂z @ y^T
/// - ∂L/∂y = x^T @ ∂L/∂z
///
/// Handles multiple cases: 2D×2D, 2D×1D, 1D×2D, 1D×1D (dot product)
struct MatMulGradFn;

impl GradFn for MatMulGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x = parents[0].borrow();
        let y = parents[1].borrow();

        // For z = x @ y where x: (m,n), y: (n,p), z: (m,p)
        // ∂L/∂x = ∂L/∂z @ y^T  -> (m,p) @ (p,n) = (m,n)
        // ∂L/∂y = x^T @ ∂L/∂z  -> (n,m) @ (m,p) = (n,p)

        let grad_x = if x.requires_grad {
            match (x.shape.len(), y.shape.len()) {
                (2, 2) => {
                    // Standard 2D: ∂L/∂x = out_grad @ y^T
                    let y_t = RawTensor::transpose_2d(&y.data, &y.shape);
                    let grad_data = RawTensor::matmul_raw(
                        &out_grad.data,
                        &y_t,
                        out_grad.shape[0],
                        out_grad.shape[1],
                        y.shape[0],
                    );
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂x = ∂L/∂z[:,None] @ v[None,:] = outer(out_grad, v)
                    let m = x.shape[0];
                    let n = x.shape[1];
                    let mut grad_data = vec![0.0; m * n];
                    for i in 0..m {
                        let gz_i = out_grad.data[i];
                        for j in 0..n {
                            grad_data[i * n + j] = gz_i * y.data[j];
                        }
                    }
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (1, 2) => {
                    // Vector-matrix: (n,) @ (n,p) -> (p,)
                    // ∂L/∂x: out_grad is (p,), y is (n,p)
                    // grad_x = out_grad @ y^T -> (p,) @ (p,n) -> (n,)
                    let mut grad_data = vec![0.0; x.shape[0]];
                    for i in 0..x.shape[0] {
                        for j in 0..y.shape[1] {
                            grad_data[i] += out_grad.data[j] * y.data[i * y.shape[1] + j];
                        }
                    }
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    // ∂L/∂x = out_grad * y
                    let og = out_grad.data[0];
                    let grad_data: Vec<f32> = y.data.iter().map(|&v| og * v).collect();
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                _ => None,
            }
        } else {
            None
        };

        let grad_y = if y.requires_grad {
            match (x.shape.len(), y.shape.len()) {
                (2, 2) => {
                    let x_t = RawTensor::transpose_2d(&x.data, &x.shape);
                    let grad_data = RawTensor::matmul_raw(
                        &x_t,
                        &out_grad.data,
                        x.shape[1],
                        x.shape[0],
                        out_grad.shape[1],
                    );
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂v = X^T @ ∂L/∂z -> (n,)
                    let m = x.shape[0];
                    let n = x.shape[1];
                    let mut grad_data = vec![0.0; n];
                    for j in 0..n {
                        let mut sum = 0.0;
                        for i in 0..m {
                            sum += x.data[i * n + j] * out_grad.data[i];
                        }
                        grad_data[j] = sum;
                    }
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (1, 2) => {
                    // grad_y = x^T @ out_grad -> (n,1) @ (1,p) -> (n,p)
                    let mut grad_data = vec![0.0; y.shape[0] * y.shape[1]];
                    for i in 0..y.shape[0] {
                        for j in 0..y.shape[1] {
                            grad_data[i * y.shape[1] + j] = x.data[i] * out_grad.data[j];
                        }
                    }
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    let og = out_grad.data[0];
                    let grad_data: Vec<f32> = x.data.iter().map(|&u| og * u).collect();
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                _ => None,
            }
        } else {
            None
        };

        vec![grad_x, grad_y]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MatMulGradFn)
    }
}

/// Unified gradient function for all movement operations
///
/// Movement ops don't change data values, only how they're indexed.
/// During backward, we need to "undo" the movement to restore the original shape.
#[derive(Clone)]
struct MovementGradFn {
    op: MovementOp,
    original_shape: Vec<usize>,
}

impl GradFn for MovementGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_tensor = match &self.op {
            MovementOp::Reshape { .. } => {
                // Reshape back to original shape
                RawTensor::new(out_grad.data.clone(), &self.original_shape, false)
            }
            MovementOp::Permute { axes } => {
                // Invert the permutation to restore original order
                let mut inverse_axes = vec![0; axes.len()];
                for (i, &ax) in axes.iter().enumerate() {
                    inverse_axes[ax] = i;
                }
                // Permute gradient back
                let grad_t = RawTensor::new(out_grad.data.clone(), &out_grad.shape, false);
                let result = RawTensor::permute_impl(&grad_t, &inverse_axes);
                return vec![Some(result)];
            }
            MovementOp::Expand { new_shape } => {
                // Sum gradient over dimensions that were expanded (broadcast)
                let mut grad_data = vec![0.0; self.original_shape.iter().product()];
                let old_strides = RawTensor::compute_strides(&self.original_shape);
                let new_strides = RawTensor::compute_strides(new_shape);

                for i in 0..out_grad.data.len() {
                    let mut old_idx = 0;
                    let mut rem = i;
                    for j in (0..new_shape.len()).rev() {
                        let coord = rem / new_strides[j];
                        rem %= new_strides[j];
                        // If this dimension was size 1, don't advance the index
                        if self.original_shape[j] != 1 {
                            old_idx += coord * old_strides[j];
                        }
                    }
                    grad_data[old_idx] += out_grad.data[i];
                }
                RawTensor::new(grad_data, &self.original_shape, false)
            }
            MovementOp::Pad { padding } => {
                // Remove padding from gradient (extract center region)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides = RawTensor::compute_strides(&self.original_shape);
                let new_strides = RawTensor::compute_strides(&out_grad.shape);

                fn unpad_recursive(
                    result: &mut [f32],
                    grad: &[f32],
                    dim: usize,
                    old_shape: &[usize],
                    _new_shape: &[usize],
                    padding: &[(usize, usize)],
                    old_offset: usize,
                    new_offset: usize,
                    old_strides: &[usize],
                    new_strides: &[usize],
                ) {
                    if dim == old_shape.len() {
                        result[old_offset] = grad[new_offset];
                        return;
                    }

                    for i in 0..old_shape[dim] {
                        let new_i = i + padding[dim].0;
                        unpad_recursive(
                            result,
                            grad,
                            dim + 1,
                            old_shape,
                            _new_shape,
                            padding,
                            old_offset + i * old_strides[dim],
                            new_offset + new_i * new_strides[dim],
                            old_strides,
                            new_strides,
                        );
                    }
                }

                unpad_recursive(
                    &mut result,
                    &out_grad.data,
                    0,
                    &self.original_shape,
                    &out_grad.shape,
                    padding,
                    0,
                    0,
                    &old_strides,
                    &new_strides,
                );
                RawTensor::new(result, &self.original_shape, false)
            }
            MovementOp::Shrink { ranges } => {
                // Pad gradient back to original size (inverse of shrink)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides = RawTensor::compute_strides(&self.original_shape);
                let new_strides = RawTensor::compute_strides(&out_grad.shape);

                fn unshrink_recursive(
                    result: &mut [f32],
                    grad: &[f32],
                    dim: usize,
                    old_shape: &[usize],
                    new_shape: &[usize],
                    ranges: &[(usize, usize)],
                    old_offset: usize,
                    new_offset: usize,
                    old_strides: &[usize],
                    new_strides: &[usize],
                ) {
                    if dim == old_shape.len() {
                        result[old_offset] = grad[new_offset];
                        return;
                    }

                    for i in 0..new_shape[dim] {
                        let old_i = i + ranges[dim].0; //Offset by range start
                        unshrink_recursive(
                            result,
                            grad,
                            dim + 1,
                            old_shape,
                            new_shape,
                            ranges,
                            old_offset + old_i * old_strides[dim],
                            new_offset + i * new_strides[dim],
                            old_strides,
                            new_strides,
                        );
                    }
                }

                unshrink_recursive(
                    &mut result,
                    &out_grad.data,
                    0,
                    &self.original_shape,
                    &out_grad.shape,
                    ranges,
                    0,
                    0,
                    &old_strides,
                    &new_strides,
                );
                RawTensor::new(result, &self.original_shape, false)
            }
            MovementOp::Stride { strides } => {
                // Upsample gradient (inverse of stride/downsampling)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides_mem = RawTensor::compute_strides(&self.original_shape);
                let new_strides_mem = RawTensor::compute_strides(&out_grad.shape);

                fn unstride_recursive(
                    result: &mut [f32],
                    grad: &[f32],
                    dim: usize,
                    old_shape: &[usize],
                    new_shape: &[usize],
                    strides: &[usize],
                    old_offset: usize,
                    new_offset: usize,
                    old_strides: &[usize],
                    new_strides: &[usize],
                ) {
                    if dim == old_shape.len() {
                        result[old_offset] = grad[new_offset];
                        return;
                    }

                    for i in 0..new_shape[dim] {
                        let old_i = i * strides[dim];
                        if old_i < old_shape[dim] {
                            unstride_recursive(
                                result,
                                grad,
                                dim + 1,
                                old_shape,
                                new_shape,
                                strides,
                                old_offset + old_i * old_strides[dim],
                                new_offset + i * new_strides[dim],
                                old_strides,
                                new_strides,
                            );
                        }
                    }
                }

                unstride_recursive(
                    &mut result,
                    &out_grad.data,
                    0,
                    &self.original_shape,
                    &out_grad.shape,
                    strides,
                    0,
                    0,
                    &old_strides_mem,
                    &new_strides_mem,
                );
                RawTensor::new(result, &self.original_shape, false)
            }
        };

        vec![Some(grad_tensor)]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MovementGradFn {
            op: self.op.clone(),
            original_shape: self.original_shape.clone(),
        })
    }
}

// ===== DEVICE ENUM =====

/// Compute device for tensor operations
///
/// Currently only CPU is implemented. GPU would require integration
/// with CUDA/OpenCL, and Metal would be for Apple Silicon.
#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    GPU,
    //TODO: possible Metal variant
}

// ===== RAW TENSOR STRUCTURE =====

/// The core tensor structure containing data and gradient tracking
///
/// This is wrapped in `Rc<RefCell<>>` to create the public `Tensor` type.
/// Fields:
/// - `data`: flat Vec<f32> of actual values (row-major order)
/// - `shape`: dimensions, e.g. [batch, channels, height, width]
/// - `grad`: accumulated gradient (Some if requires_grad, None otherwise)
/// - `requires_grad`: whether to track gradients for this tensor
/// - `grad_fn`: function to compute parent gradients during backward
/// - `parents`: input tensors that this tensor depends on
/// - `device`: where computation happens (CPU/GPU)
pub struct RawTensor {
    pub data: Vec<f32>,         // flat data vec, len = prod shape dims
    pub shape: Vec<usize>,      //tensor dims, eg [B,C,H,W]
    pub grad: Option<Vec<f32>>, //grad w.r.t tensor data, None if req_grad == false
    pub requires_grad: bool,
    pub grad_fn: Option<Box<dyn GradFn>>, //func to compute grad, if result of op
    pub parents: Vec<Tensor>,             //refs to parent tensor on graph
    pub device: Device,                   //cpu/gpu
}

impl Clone for RawTensor {
    fn clone(&self) -> Self {
        RawTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.as_ref().map(|gf| gf.clone_box()),
            parents: self.parents.clone(),
            device: self.device.clone(),
        }
    }
}

impl std::fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .field("device", &self.device)
            .finish()
    }
}

// ===== TENSOR CONSTRUCTORS =====
impl RawTensor {
    /// Create a new tensor from data and shape
    ///
    /// # Arguments
    /// * `data` - Flat vector of values (length must equal product of shape dimensions)
    /// * `shape` - Dimensions of the tensor
    /// * `requires_grad` - Whether to track gradients for backpropagation
    ///
    /// # Panics
    /// Panics if data.len() != shape.product()
    pub fn new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );
        let raw = RawTensor {
            data,
            shape: shape.to_vec(),
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![],
            device: Device::CPU,
        };
        Rc::new(RefCell::new(raw))
    }
    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }
    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape, false)
    }
    /// Create a tensor with random values uniformly distributed in [0, 1)
    pub fn rand(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        Self::new(data, shape, false)
    }
    /// Create a tensor with values from standard normal distribution N(0, 1)
    pub fn randn(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Self::new(data, shape, false)
    }
}

//basic util
impl RawTensor {
    /* methods for props: tensor.shape(), tensor.num_elements(), etc */
}

// ===== UNARY OPERATIONS =====
impl RawTensor {
    /// Apply a unary operation element-wise
    ///
    /// This is the unified implementation for all unary ops.
    /// Creates a new tensor and sets up gradient tracking if needed.
    pub fn unary_op(t: &Tensor, op: UnaryOp) -> Tensor {
        let (data, shape, req) = {
            let s = t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let result = data
            .iter()
            .map(|&x| match op {
                UnaryOp::Neg => -x,
                UnaryOp::Recip => 1.0 / x,
                UnaryOp::Sqrt => x.sqrt(),
                UnaryOp::Exp2 => 2_f32.powf(x),
                UnaryOp::Exp => x.exp(),
                UnaryOp::Log2 => x.log2(),
                UnaryOp::Log => x.ln(),
                UnaryOp::Sin => x.sin(),
                UnaryOp::Cos => x.cos(),
                UnaryOp::Tanh => x.tanh(),
                UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                UnaryOp::ReLU => x.max(0.0),
            })
            .collect();

        let out = Self::new(result, &shape, req);

        // Set up backpropagation if this tensor requires gradients
        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(UnaryGradFn { op }));
        }
        out
    }
    // Convenience methods for each unary operation
    pub fn neg(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Neg)
    }
    pub fn recip(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Recip)
    }
    pub fn sqrt(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Sqrt)
    }
    pub fn exp2(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Exp2)
    }
    pub fn log2(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Log2)
    }
    pub fn exp(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Exp)
    }
    pub fn log(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Log)
    }
    pub fn sin(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Sin)
    }
    pub fn cos(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Cos)
    }
    pub fn tanh(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Tanh)
    }
    pub fn sigmoid(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::Sigmoid)
    }
    pub fn relu(t: &Tensor) -> Tensor {
        Self::unary_op(t, UnaryOp::ReLU)
    }
}

// ===== BINARY OPERATIONS =====
impl RawTensor {
    /// Compute broadcast shape following NumPy broadcasting rules
    ///
    /// Rules:
    /// 1. Align shapes from the right (trailing dimensions)
    /// 2. For each dimension, both must be equal OR one must be 1
    /// 3. Output dimension is the maximum of the two
    ///
    /// Examples:
    /// - (3, 1) + (1, 4) -> (3, 4)
    /// - (5, 3, 1) + (1, 4) -> (5, 3, 4)
    fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Vec<usize> {
        let max_len = shape_a.len().max(shape_b.len());
        let mut result = vec![1; max_len];

        // Align from right (trailing dimensions)
        for i in 0..max_len {
            let a_dim = if i < shape_a.len() {
                shape_a[shape_a.len() - 1 - i]
            } else {
                1
            };
            let b_dim = if i < shape_b.len() {
                shape_b[shape_b.len() - 1 - i]
            } else {
                1
            };

            if a_dim == b_dim {
                result[max_len - 1 - i] = a_dim;
            } else if a_dim == 1 {
                result[max_len - 1 - i] = b_dim;
            } else if b_dim == 1 {
                result[max_len - 1 - i] = a_dim;
            } else {
                panic!(
                    "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                    shape_a, shape_b, i
                );
            }
        }
        result
    }

    /// Broadcast data from one shape to another
    ///
    /// This repeats values along dimensions where from_shape is 1
    /// and to_shape is larger.
    fn broadcast_to(data: &[f32], from_shape: &[usize], to_shape: &[usize]) -> Vec<f32> {
        if from_shape == to_shape {
            return data.to_vec();
        }

        let to_size: usize = to_shape.iter().product();
        let mut result = vec![0.0; to_size];

        // Pad from_shape with leading 1s to match rank
        let mut padded_from = vec![1; to_shape.len()];
        let offset = to_shape.len() - from_shape.len();
        padded_from[offset..].copy_from_slice(from_shape);
        let from_strides_padded = Self::compute_strides(&padded_from);
        let to_strides = Self::compute_strides(to_shape);

        // For each output position, compute corresponding input position
        for i in 0..to_size {
            let mut from_idx = 0;
            let mut remainder = i;
            //calc coords based on to_shape and map
            for dim in 0..to_shape.len() {
                let coord = remainder / to_strides[dim];
                remainder %= to_strides[dim];
                //if dim broadcast (size was 1) use coord 0 for from_idx
                if padded_from[dim] != 1 {
                    from_idx += coord * from_strides_padded[dim];
                }
            }
            result[i] = data[from_idx];
        }
        result
    }

    /// Sum gradient over dimensions that were broadcast
    ///
    /// During backward pass, if a dimension was broadcast from size 1 to size N,
    /// we need to sum the gradients over that dimension to get the gradient
    /// for the original size-1 dimension.
    fn sum_over_broadcast_dims(
        grad: &[f32],
        grad_shape: &[usize],
        target_shape: &[usize],
    ) -> Vec<f32> {
        if grad_shape == target_shape {
            return grad.to_vec();
        }

        // Pad target_shape with leading 1s to match ranks
        let mut padded_target = vec![1; grad_shape.len()];
        let offset = grad_shape.len() - target_shape.len();
        padded_target[offset..].copy_from_slice(target_shape);

        // Find dimensions that need summing (where target was 1, grad is >1)
        let mut sum_axes = Vec::new();
        for (i, (&g, &t)) in grad_shape.iter().zip(&padded_target).enumerate() {
            if t == 1 && g > 1 {
                sum_axes.push(i);
            }
        }

        // Also sum over leading dimensions if target has fewer dims
        for i in 0..offset {
            if !sum_axes.contains(&i) {
                sum_axes.push(i);
            }
        }

        if sum_axes.is_empty() {
            return grad.to_vec();
        }
        let mut result = vec![0.0; target_shape.iter().product()];
        let target_strides = Self::compute_strides(target_shape);
        let grad_strides = Self::compute_strides(grad_shape);

        // For each gradient element, sum into appropriate result position
        for (i, &grad_val) in grad.iter().enumerate() {
            let mut target_idx = 0;
            let mut remainder = i;

            // convert lin index i in grad_shape to coords and map to target
            for dim in 0..grad_shape.len() {
                let coord = remainder / grad_strides[dim];
                remainder %= grad_strides[dim];

                // Map to target coordinate (skip if was broadcast)
                if dim >= offset && padded_target[dim] != 1 {
                    let target_dim_idx = dim - offset;
                    target_idx += coord * target_strides[target_dim_idx];
                }
            }
            if target_idx < result.len() {
                result[target_idx] += grad_val;
            }
        }
        result
    }

    /// Apply a binary operation with broadcasting
    ///
    /// Steps:
    /// 1. Compute broadcast shape
    /// 2. Broadcast both inputs to that shape
    /// 3. Apply operation element-wise
    /// 4. Set up gradient tracking
    pub fn binary_op(self_t: &Tensor, other: &Tensor, op: BinaryOp) -> Tensor {
        let (data_a, shape_a, req_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_b, shape_b, req_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone(), o.requires_grad)
        };

        // Compute broadcast shape
        let out_shape = Self::broadcast_shape(&shape_a, &shape_b);

        // Broadcast inputs to output shape
        let bc_data_a = Self::broadcast_to(&data_a, &shape_a, &out_shape);
        let bc_data_b = Self::broadcast_to(&data_b, &shape_b, &out_shape);

        let result_data: Vec<f32> = bc_data_a
            .iter()
            .zip(&bc_data_b)
            .map(|(a, b)| match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => a / b,
                BinaryOp::Max => a.max(*b),
                BinaryOp::Mod => a % b,
                BinaryOp::Cmplt => {
                    if a < b {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect();

        // Mod and Cmplt are non-differentiable
        let requires_grad = match op {
            BinaryOp::Mod | BinaryOp::Cmplt => false,
            _ => req_a || req_b,
        };

        let out = Self::new(result_data, &out_shape, requires_grad);

        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(BinaryGradFn { op }));
        }
        out
    }

    // Convenience methods for binary operations
    pub fn add(self_t: &Tensor, other: &Tensor) -> Tensor {
        Self::binary_op(self_t, other, BinaryOp::Add)
    }
    pub fn sub(self_t: &Tensor, other: &Tensor) -> Tensor {
        Self::binary_op(self_t, other, BinaryOp::Sub)
    }
    pub fn elem_mul(self_t: &Tensor, other: &Tensor) -> Tensor {
        Self::binary_op(self_t, other, BinaryOp::Mul)
    }
    pub fn div(self_t: &Tensor, other: &Tensor) -> Tensor {
        Self::binary_op(self_t, other, BinaryOp::Div)
    }
    pub fn max_elem(self_t: &Tensor, other: &Tensor) -> Tensor {
        Self::binary_op(self_t, other, BinaryOp::Max)
    }
    pub fn modulo(self_t: &Tensor, other: &Tensor) -> Tensor {
        Self::binary_op(self_t, other, BinaryOp::Mod)
    }
    pub fn cmplt(self_t: &Tensor, other: &Tensor) -> Tensor {
        Self::binary_op(self_t, other, BinaryOp::Cmplt)
    }
}

// ===== REDUCE OPERATIONS =====

impl RawTensor {
    /// Apply a reduction operation that collapses tensor to scalar
    ///
    /// All reduction ops produce a shape [1] output.
    pub fn reduce_op(self_t: &Tensor, op: ReduceOp) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        let (result_val, grad_fn): (f32, Box<dyn GradFn>) = match op {
            ReduceOp::Sum => {
                let sum: f32 = data.iter().sum();
                (
                    sum,
                    Box::new(SumGradFn {
                        input_shape: shape.clone(),
                    }),
                )
            }
            ReduceOp::Max => {
                let (max_val, max_idx) = data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, &val)| (val, idx))
                    .unwrap();
                (
                    max_val,
                    Box::new(MaxReduceGradFn {
                        input_shape: shape.clone(),
                        max_index: max_idx,
                    }),
                )
            }
            ReduceOp::Mean => {
                let sum: f32 = data.iter().sum();
                let mean_val = sum / (data.len() as f32);
                (
                    mean_val,
                    Box::new(MeanGradFn {
                        input_shape: shape.clone(),
                    }),
                )
            }
        };

        let out = Self::new(vec![result_val], &[1], req_grad);

        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(grad_fn);
        }
        out
    }

    pub fn sum(self_t: &Tensor) -> Tensor {
        Self::reduce_op(self_t, ReduceOp::Sum)
    }
    pub fn max_reduce(self_t: &Tensor) -> Tensor {
        Self::reduce_op(self_t, ReduceOp::Max)
    }
    pub fn mean(self_t: &Tensor) -> Tensor {
        Self::reduce_op(self_t, ReduceOp::Mean)
    }
}

// ===== SOFTMAX & AXIS REDUCTIONS =====
impl RawTensor {
    /// Sum along a specific axis
    ///
    /// # Arguments
    /// * `dim` - Axis to reduce (0-indexed)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Examples
    /// let x = Tensor::new(vec![1,2,3,4,5,6], &[2,3], true);
    /// x.sum_dim(1, false) // -> [6, 15] shape [2]
    /// x.sum_dim(1, true)  // -> [[6], [15]] shape [2,1]
    pub fn sum_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            assert!(
                dim < s.shape.len(),
                "dim {} out of bounds for shape {:?}",
                dim,
                s.shape
            );
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        let _dim_size = shape[dim];
        let mut out_shape = shape.clone();
        out_shape[dim] = 1; // intermediate shape before squeeze
        let out_size: usize = out_shape.iter().product();
        let mut result = vec![0.0; out_size];

        // Compute strides for indexing
        let _strides = Self::compute_strides(&shape);
        let out_strides = Self::compute_strides(&out_shape);

        // Sum over the target dimension
        for i in 0..data.len() {
            // Convert linear index to coordinates
            let mut coords = vec![0; shape.len()];
            let mut rem = i;
            for (d, &dim_sz) in shape.iter().enumerate().rev() {
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            // Zero out the target dimension for output indexing
            let mut out_coords = coords.clone();
            out_coords[dim] = 0;

            // Convert output coords to linear index
            let out_idx: usize = out_coords
                .iter()
                .zip(&out_strides)
                .map(|(c, s)| c * s)
                .sum();
            result[out_idx] += data[i];
        }

        // Squeeze dimension if keepdim=false
        let final_shape = if keepdim {
            out_shape
        } else {
            out_shape
                .iter()
                .enumerate()
                .filter(|(d, _)| *d != dim)
                .map(|(_, &sz)| sz)
                .collect()
        };

        let out = Self::new(result, &final_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(SumDimGradFn {
                input_shape: shape,
                dim,
                keepdim,
            }));
        }
        out
    }

    /// Max along a specific axis
    ///
    /// Returns maximum value along dimension and stores indices for backward pass.
    pub fn max_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            assert!(
                dim < s.shape.len(),
                "dim {} out of bounds for shape {:?}",
                dim,
                s.shape
            );
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        let _dim_size = shape[dim];
        let mut out_shape = shape.clone();
        out_shape[dim] = 1;
        let out_size: usize = out_shape.iter().product();

        let mut result = vec![f32::NEG_INFINITY; out_size];
        let mut max_indices = vec![0; out_size]; // track which index won

        let _strides = Self::compute_strides(&shape);
        let out_strides = Self::compute_strides(&out_shape);

        for i in 0..data.len() {
            let mut coords = vec![0; shape.len()];
            let mut rem = i;
            for (d, &dim_sz) in shape.iter().enumerate().rev() {
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            let mut out_coords = coords.clone();
            out_coords[dim] = 0;
            let out_idx: usize = out_coords
                .iter()
                .zip(&out_strides)
                .map(|(c, s)| c * s)
                .sum();

            if data[i] > result[out_idx] {
                result[out_idx] = data[i];
                max_indices[out_idx] = i; // store linear index of max element
            }
        }

        let final_shape = if keepdim {
            out_shape.clone()
        } else {
            out_shape
                .iter()
                .enumerate()
                .filter(|(d, _)| *d != dim)
                .map(|(_, &sz)| sz)
                .collect()
        };

        let out = Self::new(result, &final_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MaxDimGradFn {
                input_shape: shape,
                max_indices,
                dim,
                keepdim,
            }));
        }
        out
    }

    pub fn softmax(self_t: &Tensor, dim: usize) -> Tensor {
        let max = Self::max_dim(self_t, dim, true);
        let shifted = self_t.sub(&max);
        let exp_x = shifted.exp();
        let sum_exp = Self::sum_dim(&exp_x, dim, true);
        exp_x.div(&sum_exp)
    }
}

// ===== LOSS FUNCTIONS =====
impl RawTensor {
    pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        let diff = pred.sub(target);
        let squared = diff.elem_mul(&diff);
        squared.mean()
    }

    pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
        let softmax = Self::softmax(logits, 1);
        let log_probs = softmax.log();
        // -sum(targets * log_probs, dim=1).mean()
        let prod = targets.elem_mul(&log_probs);
        let sum = Self::sum_dim(&prod, 1, false);
        sum.neg().mean()
    }
}

// ===== TERNARY OPERATIONS =====

impl RawTensor {
    /// Apply ternary operations (3 inputs, 1 output)
    pub fn ternary_op(x: &Tensor, y: &Tensor, z: &Tensor, op: TernaryOp) -> Tensor {
        let (data_x, shape_x, req_x) = {
            let s = x.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_y, shape_y, req_y) = {
            let s = y.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_z, shape_z, req_z) = {
            let s = z.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        assert_eq!(shape_x, shape_y, "Ternary op requires matching shapes");
        assert_eq!(shape_x, shape_z, "Ternary op requires matching shapes");

        let (result_data, grad_fn, requires_grad): (Vec<f32>, Box<dyn GradFn>, bool) = match op {
            TernaryOp::MulAcc => {
                // x * y + z
                let data = data_x
                    .iter()
                    .zip(&data_y)
                    .zip(&data_z)
                    .map(|((a, b), c)| a * b + c)
                    .collect();
                (data, Box::new(MulAccGradFn), req_x || req_y || req_z)
            }
            TernaryOp::Where => {
                // x is condition, y is true branch, z is false branch
                // x[i] != 0 ? y[i] : z[i]
                // Conditional selection: x ? y : z
                // x is condition (nonzero = true), y is true branch, z is false branch

                let data = data_x
                    .iter()
                    .zip(&data_y)
                    .zip(&data_z)
                    .map(|((c, a), b)| if *c != 0.0 { *a } else { *b })
                    .collect();
                (
                    data,
                    Box::new(WhereGradFn {
                        condition: data_x.clone(),
                    }),
                    req_y || req_z,
                )
            }
        };

        let out = Self::new(result_data, &shape_x, requires_grad);

        if out.borrow().requires_grad {
            let parents = match op {
                TernaryOp::MulAcc => vec![x.clone(), y.clone(), z.clone()],
                TernaryOp::Where => vec![y.clone(), z.clone()], // condition is not a parent
            };
            out.borrow_mut().parents = parents;
            out.borrow_mut().grad_fn = Some(grad_fn);
        }
        out
    }

    pub fn mulacc(x: &Tensor, y: &Tensor, z: &Tensor) -> Tensor {
        Self::ternary_op(x, y, z, TernaryOp::MulAcc)
    }
    pub fn where_op(cond: &Tensor, x: &Tensor, y: &Tensor) -> Tensor {
        Self::ternary_op(cond, x, y, TernaryOp::Where)
    }
}

// ===== MOVEMENT OPERATIONS =====
impl RawTensor {
    /// Reshape tensor to new shape (same number of elements)
    pub fn reshape(self_t: &Tensor, new_shape: &[usize]) -> Tensor {
        let (data, old_shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        let old_size: usize = old_shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "Cannot reshape: size mismatch");

        let out = Self::new(data, new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Reshape {
                    new_shape: new_shape.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Internal permute implementation (no gradient tracking)
    ///
    /// Reorders axes according to the permutation specified by `axes`.
    /// For example, axes=[1,0] transposes a 2D matrix.
    fn permute_impl(self_t: &Tensor, axes: &[usize]) -> Tensor {
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };
        assert_eq!(axes.len(), shape.len(), "Axes length must match rank");

        let new_shape: Vec<usize> = axes.iter().map(|&i| shape[i]).collect();
        let old_strides = Self::compute_strides(&shape);
        let _new_strides: Vec<usize> = axes.iter().map(|&i| old_strides[i]).collect();

        let mut new_data = vec![0.0; data.len()];

        // Helper: convert linear index to coordinates
        fn index_to_coords(idx: usize, shape: &[usize]) -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            let mut remaining = idx;
            for i in (0..shape.len()).rev() {
                coords[i] = remaining % shape[i];
                remaining /= shape[i];
            }
            coords
        }

        // Helper: convert coordinates to linear index using strides
        fn coords_to_index(coords: &[usize], strides: &[usize]) -> usize {
            coords.iter().zip(strides).map(|(c, s)| c * s).sum()
        }

        for (new_idx, val) in new_data.iter_mut().enumerate() {
            let new_coords = index_to_coords(new_idx, &new_shape);
            // Map new coordinates back to old coordinates
            let mut old_coords = vec![0; axes.len()];
            for (i, &ax) in axes.iter().enumerate() {
                old_coords[ax] = new_coords[i];
            }
            let old_idx = coords_to_index(&old_coords, &old_strides);
            *val = data[old_idx];
        }
        Self::new(new_data, &new_shape, false)
    }

    /// Permute (reorder) tensor axes
    ///
    /// # Arguments
    /// * `axes` - New ordering of axes (must be a valid permutation of 0..rank)
    pub fn permute(self_t: &Tensor, axes: &[usize]) -> Tensor {
        let req_grad = self_t.borrow().requires_grad;
        let old_shape = self_t.borrow().shape.clone();

        // Verify axes is a valid permutation
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        for (i, &ax) in sorted_axes.iter().enumerate() {
            assert_eq!(i, ax, "Invalid permutation axes");
        }

        let out = Self::permute_impl(self_t, axes);
        out.borrow_mut().requires_grad = req_grad;

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Permute {
                    axes: axes.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Expand (broadcast) tensor to larger shape
    ///
    /// Dimensions can only be expanded from size 1 to size N.
    /// Rank must remain the same.
    pub fn expand(self_t: &Tensor, new_shape: &[usize]) -> Tensor {
        let (data, old_shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        assert_eq!(old_shape.len(), new_shape.len(), "Expand: rank must match");

        // Verify broadcasting rules: old dims must be 1 or equal to new
        for (old_d, new_d) in old_shape.iter().zip(new_shape) {
            assert!(
                *old_d == 1 || old_d == new_d,
                "Cannot expand dimension {} to {}",
                old_d,
                new_d
            );
        }

        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_size];

        // Broadcast by repeating values
        let old_strides = Self::compute_strides(&old_shape);
        let new_strides = Self::compute_strides(new_shape);

        for i in 0..new_size {
            let mut old_idx = 0;
            let mut rem = i;
            for j in (0..new_shape.len()).rev() {
                let coord = rem / new_strides[j];
                rem %= new_strides[j];
                if old_shape[j] != 1 {
                    old_idx += coord * old_strides[j];
                }
            }
            result[i] = data[old_idx];
        }

        let out = Self::new(result, new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Expand {
                    new_shape: new_shape.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Pad tensor with zeros
    ///
    /// # Arguments
    /// * `padding` - For each dimension, (left_pad, right_pad)
    pub fn pad(self_t: &Tensor, padding: &[(usize, usize)]) -> Tensor {
        let (data, old_shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        assert_eq!(
            padding.len(),
            old_shape.len(),
            "Padding length must match rank"
        );

        let new_shape: Vec<usize> = old_shape
            .iter()
            .zip(padding)
            .map(|(d, (l, r))| d + l + r)
            .collect();

        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_size];

        // Copy old data into padded positions
        let old_strides = Self::compute_strides(&old_shape);
        let new_strides = Self::compute_strides(&new_shape);

        fn pad_recursive(
            result: &mut [f32],
            data: &[f32],
            dim: usize,
            old_shape: &[usize],
            _new_shape: &[usize],
            padding: &[(usize, usize)],
            old_offset: usize,
            new_offset: usize,
            old_strides: &[usize],
            new_strides: &[usize],
        ) {
            if dim == old_shape.len() {
                result[new_offset] = data[old_offset];
                return;
            }

            for i in 0..old_shape[dim] {
                let new_i = i + padding[dim].0;
                pad_recursive(
                    result,
                    data,
                    dim + 1,
                    old_shape,
                    _new_shape,
                    padding,
                    old_offset + i * old_strides[dim],
                    new_offset + new_i * new_strides[dim],
                    old_strides,
                    new_strides,
                );
            }
        }

        pad_recursive(
            &mut result,
            &data,
            0,
            &old_shape,
            &new_shape,
            padding,
            0,
            0,
            &old_strides,
            &new_strides,
        );

        let out = Self::new(result, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Pad {
                    padding: padding.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Extract a sub-region of the tensor
    ///
    /// # Arguments
    /// * `ranges` - For each dimension, (start, end) indices
    pub fn shrink(self_t: &Tensor, ranges: &[(usize, usize)]) -> Tensor {
        let (data, old_shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        assert_eq!(
            ranges.len(),
            old_shape.len(),
            "Ranges length must match rank"
        );

        let new_shape: Vec<usize> = ranges.iter().map(|(start, end)| end - start).collect();
        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_size];

        let old_strides = Self::compute_strides(&old_shape);
        let new_strides = Self::compute_strides(&new_shape);

        fn shrink_recursive(
            result: &mut [f32],
            data: &[f32],
            dim: usize,
            old_shape: &[usize],
            new_shape: &[usize],
            ranges: &[(usize, usize)],
            old_offset: usize,
            new_offset: usize,
            old_strides: &[usize],
            new_strides: &[usize],
        ) {
            if dim == old_shape.len() {
                result[new_offset] = data[old_offset];
                return;
            }

            for i in 0..new_shape[dim] {
                let old_i = i + ranges[dim].0;
                shrink_recursive(
                    result,
                    data,
                    dim + 1,
                    old_shape,
                    new_shape,
                    ranges,
                    old_offset + old_i * old_strides[dim],
                    new_offset + i * new_strides[dim],
                    old_strides,
                    new_strides,
                );
            }
        }

        shrink_recursive(
            &mut result,
            &data,
            0,
            &old_shape,
            &new_shape,
            ranges,
            0,
            0,
            &old_strides,
            &new_strides,
        );

        let out = Self::new(result, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Shrink {
                    ranges: ranges.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Subsample tensor with specified strides
    ///
    /// Similar to slicing with step: array[::2] takes every other element
    pub fn stride_op(self_t: &Tensor, strides: &[usize]) -> Tensor {
        let (data, old_shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        assert_eq!(
            strides.len(),
            old_shape.len(),
            "Strides length must match rank"
        );

        let new_shape: Vec<usize> = old_shape
            .iter()
            .zip(strides)
            .map(|(d, s)| d.div_ceil(*s))
            .collect();

        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_size];

        let old_strides_mem = Self::compute_strides(&old_shape);
        let new_strides_mem = Self::compute_strides(&new_shape);

        fn stride_recursive(
            result: &mut [f32],
            data: &[f32],
            dim: usize,
            old_shape: &[usize],
            new_shape: &[usize],
            strides: &[usize],
            old_offset: usize,
            new_offset: usize,
            old_strides: &[usize],
            new_strides: &[usize],
        ) {
            if dim == old_shape.len() {
                result[new_offset] = data[old_offset];
                return;
            }

            for i in 0..new_shape[dim] {
                let old_i = i * strides[dim];
                stride_recursive(
                    result,
                    data,
                    dim + 1,
                    old_shape,
                    new_shape,
                    strides,
                    old_offset + old_i * old_strides[dim],
                    new_offset + i * new_strides[dim],
                    old_strides,
                    new_strides,
                );
            }
        }

        stride_recursive(
            &mut result,
            &data,
            0,
            &old_shape,
            &new_shape,
            strides,
            0,
            0,
            &old_strides_mem,
            &new_strides_mem,
        );

        let out = Self::new(result, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Stride {
                    strides: strides.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Compute memory strides for row-major layout
    ///
    /// For shape [3, 4, 5], strides are [20, 5, 1]
    /// This tells us how many elements to skip to move one step in each dimension.
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

// ===== LOAD OPERATIONS =====
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

// ===== MATRIX MULTIPLICATION =====

impl RawTensor {
    /// Transpose a 2D matrix
    ///
    /// For shape [m, n], produces shape [n, m]
    fn transpose_2d(data: &[f32], shape: &[usize]) -> Vec<f32> {
        assert_eq!(shape.len(), 2, "Transpose expects 2D shape");
        let (m, n) = (shape[0], shape[1]);
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = data[i * n + j];
            }
        }
        result
    }

    // Helper: raw matmul computation
    /// Raw matrix multiplication: (m,k) @ (k,n) -> (m,n)
    /// Uses naive O(mnk) algorithm. For production, use optimized BLAS.
    fn matmul_raw(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            unsafe extern "C" {
                fn cblas_sgemm(
                    layout: i32,
                    trans_a: i32,
                    trans_b: i32,
                    m: i32,
                    n: i32,
                    k: i32,
                    alpha: f32,
                    a: *const f32,
                    lda: i32,
                    b: *const f32,
                    ldb: i32,
                    beta: f32,
                    c: *mut f32,
                    ldc: i32,
                );
            }

            let mut result = vec![0.0; m * n];
            unsafe {
                cblas_sgemm(
                    101, // CblasRowMajor
                    111, // CblasNoTrans
                    111, // CblasNoTrans
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    n as i32,
                    0.0,
                    result.as_mut_ptr(),
                    n as i32,
                );
            }
            result
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }
            result
        }
    }

    /// Matrix multiplication with multiple cases
    ///
    /// Supports:
    /// - (m,n) @ (n,p) -> (m,p)  [standard matmul]
    /// - (m,n) @ (n,) -> (m,)    [matrix-vector]
    /// - (n,) @ (n,p) -> (p,)    [vector-matrix]
    /// - (n,) @ (n,) -> scalar   [dot product]
    pub fn matmul(self_t: &Tensor, other: &Tensor) -> Tensor {
        let (data_a, shape_a, req_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_b, shape_b, req_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone(), o.requires_grad)
        };

        // Handle different cases
        match (shape_a.len(), shape_b.len()) {
            (2, 2) => {
                // Standard 2D matmul: (m,n) @ (n,p) -> (m,p)
                let (m, n) = (shape_a[0], shape_a[1]);
                let (n2, p) = (shape_b[0], shape_b[1]);
                assert_eq!(
                    n, n2,
                    "Matmul dimension mismatch: ({},{}) @ ({},{})",
                    m, n, n2, p
                );

                let result_data = Self::matmul_raw(&data_a, &data_b, m, n, p);
                let out = Self::new(result_data, &[m, p], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            (2, 1) => {
                // Matrix-vector: (m,n) @ (n,) -> (m,)
                let (m, n) = (shape_a[0], shape_a[1]);
                let n2 = shape_b[0];
                assert_eq!(n, n2, "Matmul dimension mismatch: ({},{}) @ ({})", m, n, n2);

                let mut result_data = vec![0.0; m];
                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += data_a[i * n + j] * data_b[j];
                    }
                    result_data[i] = sum;
                }

                let out = Self::new(result_data, &[m], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            (1, 2) => {
                // Vector-matrix: (n,) @ (n,p) -> (p,)
                let n = shape_a[0];
                let (n2, p) = (shape_b[0], shape_b[1]);
                assert_eq!(n, n2, "Matmul dimension mismatch: ({}) @ ({},{})", n, n2, p);

                let mut result_data = vec![0.0; p];
                for j in 0..p {
                    let mut sum = 0.0;
                    for i in 0..n {
                        sum += data_a[i] * data_b[i * p + j];
                    }
                    result_data[j] = sum;
                }

                let out = Self::new(result_data, &[p], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            (1, 1) => {
                // Dot product: (n,) @ (n,) -> scalar
                let n = shape_a[0];
                let n2 = shape_b[0];
                assert_eq!(n, n2, "Dot product dimension mismatch: ({}) @ ({})", n, n2);

                let sum: f32 = data_a.iter().zip(&data_b).map(|(a, b)| a * b).sum();
                let out = Self::new(vec![sum], &[1], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            _ => panic!(
                "Matmul not supported for shapes: {:?} @ {:?}",
                shape_a, shape_b
            ),
        }
    }

    /// Transpose a 2D tensor
    pub fn transpose(self_t: &Tensor) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        assert_eq!(shape.len(), 2, "Transpose expects 2D tensor");

        let transposed_data = Self::transpose_2d(&data, &shape);
        let new_shape = vec![shape[1], shape[0]];

        let out = Self::new(transposed_data, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            // Use movement grad with inverse permutation [1,0]
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Permute { axes: vec![1, 0] },
                // original shape is the input shape; movement grad uses it to map back
                original_shape: vec![shape[0], shape[1]],
            }));
        }
        out
    }
}

// ===== BACKPROPAGATION =====

impl RawTensor {
    /// Run backpropagation starting from this tensor
    ///
    /// This implements reverse-mode automatic differentiation:
    /// 1. Initialize this tensor's gradient to 1 (assumes it's a scalar loss)
    /// 2. Traverse the computation graph backwards (topological sort via DFS)
    /// 3. For each node, call its grad_fn to compute parent gradients
    /// 4. Accumulate gradients in each parent tensor
    ///
    /// Uses a HashSet to track visited nodes and avoid recomputation.
    pub fn backward(tensor_ref: &Tensor) {
        let tensor = tensor_ref.borrow();
        assert!(
            tensor.requires_grad,
            "Called backward on a tensor that doesn't require grad"
        );
        drop(tensor);
        // Initialize gradient if not already set
        {
            let mut tensor = tensor_ref.borrow_mut();
            if tensor.grad.is_none() {
                let grad_size = if tensor.shape.len() == 1 && tensor.shape[0] == 1 {
                    1
                } else {
                    tensor.data.len()
                };
                tensor.grad = Some(vec![1.0; grad_size]);
            }
        }

        // DFS-based topological traversal
        let mut stack = vec![tensor_ref.clone()];
        let mut visited = HashSet::new();

        while let Some(tensor) = stack.pop() {
            // Use raw pointer for HashSet (Rc doesn't impl Hash)
            if !visited.insert(tensor.as_ptr()) {
                continue;
            }
            let (grad_fn, parents, grad_data, shape) = {
                let t = tensor.borrow();
                (
                    t.grad_fn.as_ref().map(|gf| gf.clone_box()),
                    t.parents.clone(),
                    t.grad.clone(),
                    t.shape.clone(),
                )
            };
            // If this node has a gradient function, backpropagate
            if let Some(grad_fn) = grad_fn
                && let Some(grad_out_data) = grad_data
            {
                let grad_out = RawTensor {
                    data: grad_out_data,
                    shape,
                    grad: None,
                    requires_grad: false,
                    grad_fn: None,
                    parents: vec![],
                    device: Device::CPU,
                };
                // Compute gradients for parent tensors
                let parent_grads = grad_fn.backward(&grad_out, &parents);

                // Accumulate gradients in parents
                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
                    if let Some(g) = parent_grad {
                        let mut parent = parent_ref.borrow_mut();
                        let g_data = g.borrow().data.clone();

                        // Initialize or accumulate gradient
                        if parent.grad.is_none() {
                            parent.grad = Some(g_data)
                        } else {
                            let existing = parent.grad.as_mut().unwrap();
                            for (accum, &new) in existing.iter_mut().zip(g_data.iter()) {
                                *accum += new;
                            }
                        }
                        // Add to stack if it has grad_fn (not a leaf)
                        if parent.grad_fn.is_some() {
                            drop(parent);
                            stack.push(parent_ref.clone());
                        }
                    }
                }
            }
        }
    }
}

// ===== NUMERICAL GRADIENT CHECKING =====

impl RawTensor {
    /// Check gradients numerically using finite differences
    ///
    /// For each parameter, we compute:
    ///
    /// Analytical gradient: What our backward() computes
    /// Numerical gradient: (f(x+ε) - f(x-ε)) / (2ε)
    ///
    /// The central difference formula is more accurate than forward difference.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor whose gradients to check
    /// * `loss_fn` - Function that computes a scalar loss from the tensor
    /// * `epsilon` - Step size for finite differences (typically 1e-5 to 1e-2)
    /// * `tolerance` - Maximum acceptable relative error (typically 1e-3 to 1e-2)
    ///
    /// # Returns
    /// (max_error, mean_error, passed)
    pub fn check_gradients<F>(
        tensor: &Tensor,
        loss_fn: F,
        epsilon: f32,
        tolerance: f32,
    ) -> (f32, f32, bool)
    where
        F: Fn(&Tensor) -> Tensor,
    {
        // Compute analytical gradient
        let loss = loss_fn(tensor);
        loss.backward();

        let analytical_grad = tensor.grad().expect("Tensor must have gradient");
        let mut numerical_grad = vec![0.0; analytical_grad.len()];

        let original_data = tensor.borrow().data.clone();
        let original_shape = tensor.borrow().shape.clone();
        let requires_grad = tensor.borrow().requires_grad;

        // Compute numerical gradient for each element
        for i in 0..original_data.len() {
            //f(x + epsilon)
            let mut data_plus = original_data.clone();
            data_plus[i] += epsilon;
            let tensor_plus = RawTensor::new(data_plus, &original_shape, requires_grad);
            let loss_plus = loss_fn(&tensor_plus);
            let val_plus = loss_plus.borrow().data[0];

            let mut data_minus = original_data.clone();
            data_minus[i] -= epsilon;
            let tensor_minus = RawTensor::new(data_minus, &original_shape, requires_grad);
            let loss_minus = loss_fn(&tensor_minus);
            let val_minus = loss_minus.borrow().data[0];
            //central diff
            numerical_grad[i] = (val_plus - val_minus) / (2.0 * epsilon);
        }

        // Compute errors
        let mut max_error: f32 = 0.0;
        let mut total_error: f32 = 0.0;

        for (i, (&analytical, &numerical)) in
            analytical_grad.iter().zip(&numerical_grad).enumerate()
        {
            let error = (analytical - numerical).abs();
            let relative_error = if numerical.abs() > 1e-8 {
                error / numerical.abs()
            } else {
                error
            };

            max_error = max_error.max(relative_error);
            total_error += relative_error;

            if relative_error > tolerance {
                eprintln!(
                    "Gradient mismatch at index {}: analytical={:.6e}, numerical={:.6e}, error={:.6e}",
                    i, analytical, numerical, relative_error
                );
            }
        }

        let mean_error = total_error / analytical_grad.len() as f32;
        let passed = max_error < tolerance;

        (max_error, mean_error, passed)
    }

    /// Simplified gradient checker with default parameters
    ///
    /// Uses epsilon=1e-2 and tolerance=1e-3, which work well for most cases.
    pub fn check_gradients_simple<F>(tensor: &Tensor, loss_fn: F) -> bool
    where
        F: Fn(&Tensor) -> Tensor,
    {
        let (max_err, mean_err, passed) = Self::check_gradients(
            tensor, loss_fn, 1e-2, // epsilon
            1e-3, // tolerance
        );

        if !passed {
            eprintln!(
                "Gradient check FAILED: max_error={:.6e}, mean_error={:.6e}",
                max_err, mean_err
            );
        }

        passed
    }
}

// ===== TRAIT-BASED API =====

/// Public trait for tensor operations
///
/// This provides a more ergonomic API: `tensor.add(&other)` instead of `RawTensor::add(&tensor, &other)`
pub trait TensorOps {
    //Binary ops
    fn add(&self, other: &Tensor) -> Tensor;
    fn sub(&self, other: &Tensor) -> Tensor;
    fn elem_mul(&self, other: &Tensor) -> Tensor;
    fn div(&self, other: &Tensor) -> Tensor;
    fn max_elem(&self, other: &Tensor) -> Tensor;
    fn modulo(&self, other: &Tensor) -> Tensor;
    fn cmplt(&self, other: &Tensor) -> Tensor;

    // Unary ops
    fn neg(&self) -> Tensor;
    fn recip(&self) -> Tensor;
    fn sqrt(&self) -> Tensor;
    fn exp2(&self) -> Tensor;
    fn log2(&self) -> Tensor;
    fn exp(&self) -> Tensor;
    fn log(&self) -> Tensor;
    fn sin(&self) -> Tensor;
    fn cos(&self) -> Tensor;
    fn tanh(&self) -> Tensor;
    fn sigmoid(&self) -> Tensor;
    fn relu(&self) -> Tensor;

    //Reduce ops
    fn sum(&self) -> Tensor;
    fn max_reduce(&self) -> Tensor;
    fn mean(&self) -> Tensor;

    //Ternary ops
    fn mulacc(&self, y: &Tensor, z: &Tensor) -> Tensor;
    fn where_op(&self, x: &Tensor, y: &Tensor) -> Tensor;

    // Movement ops
    fn reshape(&self, new_shape: &[usize]) -> Tensor;
    fn permute(&self, axes: &[usize]) -> Tensor;
    fn expand(&self, new_shape: &[usize]) -> Tensor;
    fn pad(&self, padding: &[(usize, usize)]) -> Tensor;
    fn shrink(&self, ranges: &[(usize, usize)]) -> Tensor;
    fn stride_op(&self, strides: &[usize]) -> Tensor;

    //Matmul
    fn matmul(&self, other: &Tensor) -> Tensor;
    fn transpose(&self) -> Tensor;

    //Gradient ops
    fn backward(&self);
    fn grad(&self) -> Option<Vec<f32>>;

    // Axis reductions
    fn sum_dim(&self, dim: usize, keepdim: bool) -> Tensor;
    fn max_dim(&self, dim: usize, keepdim: bool) -> Tensor;

    // Softmax
    fn softmax(&self, dim: usize) -> Tensor;
}

impl TensorOps for Tensor {
    fn add(&self, other: &Tensor) -> Tensor {
        RawTensor::add(self, other)
    }
    fn sub(&self, other: &Tensor) -> Tensor {
        RawTensor::sub(self, other)
    }
    fn elem_mul(&self, other: &Tensor) -> Tensor {
        RawTensor::elem_mul(self, other)
    }
    fn div(&self, other: &Tensor) -> Tensor {
        RawTensor::div(self, other)
    }
    fn max_elem(&self, other: &Tensor) -> Tensor {
        RawTensor::max_elem(self, other)
    }
    fn modulo(&self, other: &Tensor) -> Tensor {
        RawTensor::modulo(self, other)
    }
    fn cmplt(&self, other: &Tensor) -> Tensor {
        RawTensor::cmplt(self, other)
    }

    fn neg(&self) -> Tensor {
        RawTensor::neg(self)
    }
    fn recip(&self) -> Tensor {
        RawTensor::recip(self)
    }
    fn sqrt(&self) -> Tensor {
        RawTensor::sqrt(self)
    }
    fn exp2(&self) -> Tensor {
        RawTensor::exp2(self)
    }
    fn log2(&self) -> Tensor {
        RawTensor::log2(self)
    }
    fn exp(&self) -> Tensor {
        RawTensor::exp(self)
    }
    fn log(&self) -> Tensor {
        RawTensor::log(self)
    }
    fn sin(&self) -> Tensor {
        RawTensor::sin(self)
    }
    fn cos(&self) -> Tensor {
        RawTensor::cos(self)
    }
    fn tanh(&self) -> Tensor {
        RawTensor::tanh(self)
    }
    fn sigmoid(&self) -> Tensor {
        RawTensor::sigmoid(self)
    }
    fn relu(&self) -> Tensor {
        RawTensor::relu(self)
    }

    fn sum(&self) -> Tensor {
        RawTensor::sum(self)
    }
    fn max_reduce(&self) -> Tensor {
        RawTensor::max_reduce(self)
    }
    fn mean(&self) -> Tensor {
        RawTensor::mean(self)
    }

    fn mulacc(&self, y: &Tensor, z: &Tensor) -> Tensor {
        RawTensor::mulacc(self, y, z)
    }
    fn where_op(&self, x: &Tensor, y: &Tensor) -> Tensor {
        RawTensor::where_op(self, x, y)
    }

    fn reshape(&self, new_shape: &[usize]) -> Tensor {
        RawTensor::reshape(self, new_shape)
    }
    fn permute(&self, axes: &[usize]) -> Tensor {
        RawTensor::permute(self, axes)
    }
    fn expand(&self, new_shape: &[usize]) -> Tensor {
        RawTensor::expand(self, new_shape)
    }
    fn pad(&self, padding: &[(usize, usize)]) -> Tensor {
        RawTensor::pad(self, padding)
    }
    fn shrink(&self, ranges: &[(usize, usize)]) -> Tensor {
        RawTensor::shrink(self, ranges)
    }
    fn stride_op(&self, strides: &[usize]) -> Tensor {
        RawTensor::stride_op(self, strides)
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        RawTensor::matmul(self, other)
    }
    fn transpose(&self) -> Tensor {
        RawTensor::transpose(self)
    }

    fn backward(&self) {
        RawTensor::backward(self)
    }
    fn grad(&self) -> Option<Vec<f32>> {
        self.borrow().grad.clone()
    }
    fn sum_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        RawTensor::sum_dim(self, dim, keepdim)
    }
    fn max_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        RawTensor::max_dim(self, dim, keepdim)
    }
    fn softmax(&self, dim: usize) -> Tensor {
        RawTensor::softmax(self, dim)
    }
}

// ===== Modules =====

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.borrow_mut().grad = None;
        }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

impl Sequential {
    // Helper constructor for easier testing and building
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

// Usage example
#[allow(dead_code)]
impl Sequential {
    fn build_mlp(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Sequential {
        Sequential::new(vec![
            Box::new(Linear::new(input_dim, hidden_dim, true)),
            Box::new(ReLU),
            Box::new(Linear::new(hidden_dim, output_dim, true)),
        ])
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![] // No learnable params
    }
}

// ===== OPTIMIZERS =====

/// Stochastic Gradient Descent optimizer with optional momentum
///
/// Update rule:
/// - Without momentum: θ ← θ - lr·∇θ
/// - With momentum: v ← β·v - lr·∇θ, θ ← θ + v
///
/// Momentum helps accelerate convergence and dampen oscillations.
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer
    ///
    /// # Arguments
    /// * `params` - List of parameters to optimize
    /// * `lr` - Learning rate (typical: 0.01 to 0.1)
    /// * `momentum` - Momentum coefficient (typical: 0.9, or 0.0 for no momentum)
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let velocity = if momentum > 0.0 {
            params
                .iter()
                .map(|p| vec![0.0; p.borrow().data.len()])
                .collect()
        } else {
            vec![]
        };

        SGD {
            params,
            lr,
            momentum,
            velocity,
        }
    }

    /// Zero all parameter gradients
    ///
    /// Must be called before each backward pass to avoid gradient accumulation
    /// across multiple batches.
    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }
}

impl SGD {
    /// Perform one optimization step
    ///
    /// Updates all parameters using their accumulated gradients.
    pub fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            if let Some(grad) = &p.grad.clone() {
                if self.momentum > 0.0 {
                    // Update velocity: v = momentum·v - lr·grad
                    for (v, &g) in self.velocity[i].iter_mut().zip(grad.iter()) {
                        *v = self.momentum * *v - self.lr * g;
                    }
                    // Update parameters: θ = θ + v
                    for (d, &v) in p.data.iter_mut().zip(&self.velocity[i]) {
                        *d += v;
                    }
                } else {
                    // Simple SGD: θ = θ - lr·grad
                    for (d, &g) in p.data.iter_mut().zip(grad.iter()) {
                        *d -= self.lr * g;
                    }
                }
            }
        }
    }
}

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    m: Vec<Vec<f32>>, // 1st moment
    v: Vec<Vec<f32>>, // 2nd moment
    t: usize,         // timestep
}

impl Adam {
    pub fn step(&mut self) {
        self.t += 1;
        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            if let Some(grad) = &p.grad {
                // Update biased moments
                for j in 0..grad.len() {
                    self.m[i][j] = self.betas.0 * self.m[i][j] + (1.0 - self.betas.0) * grad[j];
                    self.v[i][j] =
                        self.betas.1 * self.v[i][j] + (1.0 - self.betas.1) * grad[j].powi(2);
                }

                // Bias correction
                let m_hat_scale = 1.0 / (1.0 - self.betas.0.powi(self.t as i32));
                let v_hat_scale = 1.0 / (1.0 - self.betas.1.powi(self.t as i32));

                // Update parameters
                for j in 0..p.data.len() {
                    let m_hat = self.m[i][j] * m_hat_scale;
                    let v_hat = self.v[i][j] * v_hat_scale;
                    p.data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }
    }
}
impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32, betas: (f32, f32), eps: f32) -> Self {
        let m = params
            .iter()
            .map(|p| vec![0.0; p.borrow().data.len()])
            .collect();
        let v = params
            .iter()
            .map(|p| vec![0.0; p.borrow().data.len()])
            .collect();
        Adam {
            params,
            lr,
            betas,
            eps,
            m,
            v,
            t: 0,
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }
}

//TODO: implement Muon

// ===== NEURAL NETWORK LAYERS =====

/// Fully-connected (dense/linear) layer
///
/// Computes: y = xW + b
/// where x is (batch, in_features), W is (in_features, out_features), b is (out_features)
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone())
        }
        params
    }
}

impl Linear {
    /// Create a new linear layer with random initialization
    ///
    /// Uses randn (normal distribution) for weights. For better initialization,
    /// consider using Xavier/He initialization.
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let w = RawTensor::xavier_uniform(&[in_features, out_features]);
        w.borrow_mut().requires_grad = true;
        let b = if use_bias {
            let b = RawTensor::zeros(&[out_features]);
            b.borrow_mut().requires_grad = true;
            Some(b)
        } else {
            None
        };
        Linear { weight: w, bias: b }
    }

    /// Forward pass through the layer
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = x.matmul(&self.weight);
        if let Some(b) = &self.bias {
            out.add(b)
        } else {
            out
        }
    }
}

impl RawTensor {
    /// Xavier uniform initialization
    ///
    /// Samples weights uniformly from [-limit, limit] where
    /// limit = sqrt(6 / (fan_in + fan_out))
    ///
    /// This helps maintain gradient variance across layers.
    pub fn xavier_uniform(shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let fan_out = shape[1];
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let data: Vec<f32> = (0..fan_in * fan_out)
            .map(|_| rand::rng().random_range(-limit..limit))
            .collect();
        Self::new(data, shape, false)
    }
}

// ===== DataLoaders =====

pub struct DataLoader {
    data: Vec<f32>,
    targets: Vec<f32>,
    data_shape: Vec<usize>, // per-sample shape
    target_shape: Vec<usize>,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current: usize,
}

impl DataLoader {
    pub fn new(
        data: Vec<f32>,
        targets: Vec<f32>,
        data_shape: &[usize],   // e.g., [28, 28] for MNIST
        target_shape: &[usize], // e.g., [10] for one-hot
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        let num_samples = data.len() / data_shape.iter().product::<usize>();
        let mut indices: Vec<usize> = (0..num_samples).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::rng());
        }

        DataLoader {
            data,
            targets,
            data_shape: data_shape.to_vec(),
            target_shape: target_shape.to_vec(),
            batch_size,
            shuffle,
            indices,
            current: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            self.indices.shuffle(&mut rand::rng());
        }
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];
        let actual_batch = batch_indices.len();

        let sample_size: usize = self.data_shape.iter().product();
        let target_size: usize = self.target_shape.iter().product();

        // Gather batch
        let mut batch_data = Vec::with_capacity(actual_batch * sample_size);
        let mut batch_targets = Vec::with_capacity(actual_batch * target_size);

        for &idx in batch_indices {
            let data_start = idx * sample_size;
            let target_start = idx * target_size;

            batch_data.extend_from_slice(&self.data[data_start..data_start + sample_size]);
            batch_targets
                .extend_from_slice(&self.targets[target_start..target_start + target_size]);
        }

        self.current = end;

        let mut batch_shape = vec![actual_batch];
        batch_shape.extend_from_slice(&self.data_shape);

        let mut target_batch_shape = vec![actual_batch];
        target_batch_shape.extend_from_slice(&self.target_shape);

        Some((
            RawTensor::new(batch_data, &batch_shape, false),
            RawTensor::new(batch_targets, &target_batch_shape, false),
        ))
    }
}

// ===== TESTS =====
//
// The test suite validates:
// - Basic operations (add, mul, etc.)
// - Gradient correctness (chain rule, broadcasting)
// - Complex scenarios (neural networks, matmul variants)
// - Numerical gradient checking (validates all gradients)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_add_backward() {
        let a = RawTensor::new(vec![2.0], &[1], true);
        let b = RawTensor::new(vec![3.0], &[1], true);
        let c = a.add(&b);
        c.backward();

        assert_eq!(a.grad(), Some(vec![1.0]));
        assert_eq!(b.grad(), Some(vec![1.0]));
    }

    #[test]
    fn test_multiply_backward() {
        let a = RawTensor::new(vec![3.0], &[1], true);
        let b = RawTensor::new(vec![4.0], &[1], true);
        let c = a.elem_mul(&b);
        c.backward();

        assert_eq!(a.grad(), Some(vec![4.0]));
        assert_eq!(b.grad(), Some(vec![3.0]));
    }

    #[test]
    fn test_chain_rule() {
        let a = RawTensor::new(vec![2.0], &[1], true);
        let b = RawTensor::new(vec![3.0], &[1], true);
        let c = a.add(&b);
        let d = c.elem_mul(&a);
        d.backward();

        assert_eq!(a.grad(), Some(vec![7.0]));
        assert_eq!(b.grad(), Some(vec![2.0]));
    }

    #[test]
    fn test_sum_backward() {
        let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let loss = a.sum();
        loss.backward();

        assert_eq!(a.grad(), Some(vec![1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_multidim_ops() {
        let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = RawTensor::new(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], true);
        let c = a.elem_mul(&b);
        let loss = c.sum();
        loss.backward();

        assert_eq!(a.grad(), Some(vec![0.5, 0.5, 0.5, 0.5]));
        assert_eq!(b.grad(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }
}

#[cfg(test)]
mod unary_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_neg_forward_backward() {
        let x = RawTensor::new(vec![2.0, -3.0], &[2], true);
        let y = x.neg();

        // Forward
        assert_eq!(y.borrow().data, vec![-2.0, 3.0]);

        // Backward: ∂(-x)/∂x = -1
        y.backward();
        assert_eq!(x.grad(), Some(vec![-1.0, -1.0]));
    }

    #[test]
    fn test_sqrt_chain() {
        let x = RawTensor::new(vec![4.0], &[1], true);
        let y = x.sqrt(); // y = 2.0
        let z = y.elem_mul(&y); // z = 4.0
        z.backward();

        // ∂z/∂x = ∂z/∂y * ∂y/∂x = 2y * 1/(2√x) = 2*2 * 1/4 = 1.0
        assert_relative_eq!(x.grad().unwrap()[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_exp2_log2_inverse() {
        let x = RawTensor::new(vec![2.0], &[1], true);
        let y = x.exp2().log2(); // should recover x
        y.backward();

        assert_relative_eq!(y.borrow().data[0], 2.0, epsilon = 1e-6);
        // Chain rule: ∂(log2(2^x))/∂x = 1
        assert_relative_eq!(x.grad().unwrap()[0], 1.0, epsilon = 1e-6);
    }
}

#[cfg(test)]
mod binary_tests {
    use super::*;

    #[test]
    fn test_div_backward() {
        let x = RawTensor::new(vec![6.0], &[1], true);
        let y = RawTensor::new(vec![2.0], &[1], true);
        let z = x.div(&y); // z = 3.0
        z.backward();

        // ∂(x/y)/∂x = 1/y = 0.5
        assert_eq!(x.grad(), Some(vec![0.5]));
        // ∂(x/y)/∂y = -x/y² = -6/4 = -1.5
        assert_eq!(y.grad(), Some(vec![-1.5]));
    }

    #[test]
    fn test_max_backward() {
        let x = RawTensor::new(vec![3.0, 1.0], &[2], true);
        let y = RawTensor::new(vec![2.0, 4.0], &[2], true);
        let z = x.max_elem(&y);
        let loss = z.sum();
        loss.backward();

        // max picks [3.0, 4.0], so grads flow to x[0] and y[1]
        assert_eq!(x.grad(), Some(vec![1.0, 0.0]));
        assert_eq!(y.grad(), Some(vec![0.0, 1.0]));
    }
}

#[cfg(test)]
mod reduce_tests {
    use super::*;

    #[test]
    fn test_reduce_max_backward() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0], &[3], true);
        let y = x.max_reduce(); // finds 5.0 at index 1
        y.backward();

        // Only max element gets gradient
        assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0]));
    }
}

#[cfg(test)]
mod ternary_tests {
    use super::*;

    #[test]
    fn test_mulacc_backward() {
        // z = x*y + w
        let x = RawTensor::new(vec![2.0], &[1], true);
        let y = RawTensor::new(vec![3.0], &[1], true);
        let w = RawTensor::new(vec![1.0], &[1], true);
        let z = x.mulacc(&y, &w); // z = 7.0
        z.backward();

        assert_eq!(x.grad(), Some(vec![3.0])); // ∂z/∂x = y
        assert_eq!(y.grad(), Some(vec![2.0])); // ∂z/∂y = x
        assert_eq!(w.grad(), Some(vec![1.0])); // ∂z/∂w = 1
    }

    #[test]
    fn test_where_backward() {
        let cond = RawTensor::new(vec![1.0, 0.0], &[2], false);
        let x = RawTensor::new(vec![10.0, 20.0], &[2], true);
        let y = RawTensor::new(vec![30.0, 40.0], &[2], true);
        let z = cond.where_op(&x, &y); // picks [10.0, 40.0]
        z.backward();

        assert_eq!(x.grad(), Some(vec![1.0, 0.0])); // grad flows where cond=1
        assert_eq!(y.grad(), Some(vec![0.0, 1.0])); // grad flows where cond=0
    }
}

#[cfg(test)]
mod movement_tests {
    use super::*;

    #[test]
    fn test_reshape_backward() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let y = x.reshape(&[2, 2]);
        let loss = y.sum();
        loss.backward();

        // Gradient reshapes back to [4]
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_permute_backward() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let y = x.permute(&[1, 0]); // transpose
        let loss = y.sum();
        loss.backward();

        // Gradient permutes back
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
    }
}

#[cfg(test)]
mod misc_tests {
    use super::*;
    // ===== NEURAL NETWORK LAYER TEST =====

    #[test]
    fn test_linear_layer() {
        // Simple linear layer: y = xW + b
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], true);
        let w = RawTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2], true);
        let b = RawTensor::new(vec![0.1, 0.2], &[1, 2], true);

        let y = x.matmul(&w); // [1,3] @ [3,2] = [1,2]
        let out = y.add(&b);
        let loss = out.sum();

        loss.backward();

        // All should have gradients
        assert!(x.grad().is_some());
        assert!(w.grad().is_some());
        assert!(b.grad().is_some());

        // b gradient should be ones (direct path from sum)
        assert_eq!(b.grad(), Some(vec![1.0, 1.0]));
    }

    // ===== BROADCASTING TESTS =====

    #[test]
    fn test_broadcast_shape() {
        // (3, 1) and (1, 4) -> (3, 4)
        let shape = RawTensor::broadcast_shape(&[3, 1], &[1, 4]);
        assert_eq!(shape, vec![3, 4]);

        // (5, 3, 1) and (1, 4) -> (5, 3, 4)
        let shape = RawTensor::broadcast_shape(&[5, 3, 1], &[1, 4]);
        assert_eq!(shape, vec![5, 3, 4]);

        // (1,) and (3, 4) -> (3, 4)
        let shape = RawTensor::broadcast_shape(&[1], &[3, 4]);
        assert_eq!(shape, vec![3, 4]);

        // (3, 4) and (4,) -> (3, 4)
        let shape = RawTensor::broadcast_shape(&[3, 4], &[4]);
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast")]
    fn test_broadcast_incompatible() {
        RawTensor::broadcast_shape(&[3, 2], &[4, 3]);
    }

    #[test]
    fn test_broadcast_add_scalar() {
        // (2, 3) + scalar -> (2, 3)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let scalar = RawTensor::new(vec![10.0], &[1], true);
        let y = x.add(&scalar);

        assert_eq!(y.borrow().shape, vec![2, 3]);
        assert_eq!(y.borrow().data, vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

        y.backward();

        // x gradient: all ones
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        // scalar gradient: sum of all gradients = 6.0
        assert_eq!(scalar.grad(), Some(vec![6.0]));
    }

    #[test]
    fn test_broadcast_mul_vector() {
        // (2, 3) * (3,) -> (2, 3)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let v = RawTensor::new(vec![2.0, 3.0, 4.0], &[3], true);
        let y = x.elem_mul(&v);

        assert_eq!(y.borrow().shape, vec![2, 3]);
        assert_eq!(y.borrow().data, vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);

        y.backward();

        // x gradient: broadcast v
        assert_eq!(x.grad(), Some(vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]));
        // v gradient: sum over rows
        assert_eq!(v.grad(), Some(vec![5.0, 7.0, 9.0])); // [1+4, 2+5, 3+6]
    }

    #[test]
    fn test_broadcast_add_matrix() {
        // (2, 1) + (1, 3) -> (2, 3)
        let x = RawTensor::new(vec![1.0, 2.0], &[2, 1], true);
        let y = RawTensor::new(vec![10.0, 20.0, 30.0], &[1, 3], true);
        let z = x.add(&y);

        assert_eq!(z.borrow().shape, vec![2, 3]);
        assert_eq!(z.borrow().data, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);

        z.backward();

        // x gradient: sum over columns -> [3.0, 3.0]
        assert_eq!(x.grad(), Some(vec![3.0, 3.0]));
        // y gradient: sum over rows -> [2.0, 2.0, 2.0]
        assert_eq!(y.grad(), Some(vec![2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_broadcast_batch_bias() {
        // Simulate batch with bias: (batch=3, features=2) + (features=2,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
        let bias = RawTensor::new(vec![0.5, 1.0], &[2], true);
        let y = x.add(&bias);

        assert_eq!(y.borrow().shape, vec![3, 2]);
        assert_eq!(y.borrow().data, vec![1.5, 3.0, 3.5, 5.0, 5.5, 7.0]);

        let loss = y.sum();
        loss.backward();

        // x gradient: all ones
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        // bias gradient: sum over batch -> [3.0, 3.0]
        assert_eq!(bias.grad(), Some(vec![3.0, 3.0]));
    }

    #[test]
    fn test_broadcast_div() {
        // (2, 3) / (1, 3) -> (2, 3)
        let x = RawTensor::new(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0], &[2, 3], true);
        let y = RawTensor::new(vec![2.0, 2.0, 2.0], &[1, 3], true);
        let z = x.div(&y);

        assert_eq!(z.borrow().shape, vec![2, 3]);
        assert_eq!(z.borrow().data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        z.backward();

        // x gradient: 1/y broadcast
        assert_eq!(x.grad(), Some(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]));
        // y gradient: sum(-x/y²) over rows
        // -2/4=-0.5, -4/4=-1.0, -6/4=-1.5 (row 1)
        // -8/4=-2.0, -10/4=-2.5, -12/4=-3.0 (row 2)
        // sum: [-2.5, -3.5, -4.5]
        assert_eq!(y.grad(), Some(vec![-2.5, -3.5, -4.5]));
    }

    #[test]
    fn test_broadcast_3d() {
        // (1, 2, 3) + (2, 1) -> (1, 2, 3) but will broadcast to match
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], true);
        let y = RawTensor::new(vec![10.0, 20.0], &[2, 1], true);
        let z = x.add(&y);

        assert_eq!(z.borrow().shape, vec![1, 2, 3]);
        // Row 0: [1,2,3] + 10 = [11,12,13]
        // Row 1: [4,5,6] + 20 = [24,25,26]
        assert_eq!(z.borrow().data, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

        z.backward();

        // x gradient: all ones
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        // y gradient: sum over last dimension -> [3.0, 3.0]
        assert_eq!(y.grad(), Some(vec![3.0, 3.0]));
    }

    #[test]
    fn test_broadcast_max() {
        // (2, 3) max (3,) -> (2, 3)
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], true);
        let y = RawTensor::new(vec![2.0, 3.0, 4.0], &[3], true);
        let z = x.max_elem(&y);

        assert_eq!(z.borrow().shape, vec![2, 3]);
        // [max(1,2), max(5,3), max(3,4)] = [2,5,4]
        // [max(4,2), max(2,3), max(6,4)] = [4,3,6]
        assert_eq!(z.borrow().data, vec![2.0, 5.0, 4.0, 4.0, 3.0, 6.0]);

        z.backward();

        // Gradient flows to max elements
        // x: [0, 1, 0, 1, 0, 1] (x wins at indices 1, 3, 5)
        assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]));
        // y: sum over rows where y wins
        // y[0] wins at (0,0): 1
        // y[1] wins at (1,1): 1
        // y[2] wins at (0,2): 1
        // Total: [1, 1, 1]
        assert_eq!(y.grad(), Some(vec![1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_broadcast_bias_add() {
        // Common pattern: batch matmul + bias
        // (batch=2, in=3) @ (3, 4) + (4,) -> (2, 4)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let w = RawTensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            &[3, 4],
            true,
        );
        let b = RawTensor::new(vec![0.01, 0.02, 0.03, 0.04], &[4], true);

        let y = x.matmul(&w);
        let z = y.add(&b); // Broadcasting happens here
        let loss = z.sum();

        loss.backward();

        // All should have gradients
        assert!(x.grad().is_some());
        assert!(w.grad().is_some());
        assert!(b.grad().is_some());

        // Bias gradient should be [batch_size, batch_size, ...]
        // Sum over batch dimension -> [2, 2, 2, 2]
        assert_eq!(b.grad(), Some(vec![2.0, 2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_matmul_matrix_vector_backward() {
        // (m,n) @ (n,) -> (m,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
        let v = RawTensor::new(vec![0.5, -1.0], &[2], true);

        // z = X @ v
        let z = x.matmul(&v);
        // loss = sum(z) => ∂L/∂z = 1
        let loss = z.sum();
        loss.backward();

        // ∂L/∂X = outer(ones(m), v) = repeat v on each row
        assert_eq!(x.grad(), Some(vec![0.5, -1.0, 0.5, -1.0, 0.5, -1.0]));
        // ∂L/∂v = X^T @ ones(m) = column sums of X
        // sums: col0 = 1+3+5 = 9, col1 = 2+4+6 = 12
        assert_eq!(v.grad(), Some(vec![9.0, 12.0]));
    }

    #[test]
    fn test_dot_backward() {
        // (n,) @ (n,) -> scalar
        let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let b = RawTensor::new(vec![4.0, 5.0, 6.0], &[3], true);

        // loss = a · b = 1*4 + 2*5 + 3*6 = 32
        let loss = a.matmul(&b);
        loss.backward();

        // ∂L/∂a = b
        assert_eq!(a.grad(), Some(vec![4.0, 5.0, 6.0]));
        // ∂L/∂b = a
        assert_eq!(b.grad(), Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_gradcheck_matrix_vector_matmul() {
        // Check gradients numerically for X in (m,n) @ (n,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let v = RawTensor::new(vec![0.3, -0.7], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.matmul(&v).sum());
        assert!(passed, "Matrix-vector matmul gradient check failed");
    }

    #[test]
    fn test_broadcast_sub() {
        // Test that sub also broadcasts correctly
        let x = RawTensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], true);
        let y = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let z = x.sub(&y);

        assert_eq!(z.borrow().shape, vec![2, 2]);
        assert_eq!(z.borrow().data, vec![4.0, 4.0, 6.0, 6.0]);

        z.backward();

        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
        // Sub gradient for y is negative and summed
        assert_eq!(y.grad(), Some(vec![-2.0, -2.0]));
    }

    // ===== NUMERICAL GRADIENT CHECKING TESTS =====

    #[test]
    fn test_gradcheck_unary_ops() {
        // Test sqrt gradient
        let x = RawTensor::new(vec![4.0, 9.0, 16.0], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sqrt();
            y.sum()
        });
        assert!(passed, "Sqrt gradient check failed");

        // Test sin gradient
        let x = RawTensor::new(vec![0.5, 1.0, 1.5], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sin();
            y.sum()
        });
        assert!(passed, "Sin gradient check failed");

        // Test sigmoid gradient
        let x = RawTensor::new(vec![0.0, 1.0, -1.0], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sigmoid();
            y.sum()
        });
        assert!(passed, "Sigmoid gradient check failed");
    }

    #[test]
    fn test_gradcheck_binary_ops() {
        // Test add gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let y = RawTensor::new(vec![4.0, 5.0, 6.0], &[3], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.add(&y);
            z.sum()
        });
        assert!(passed, "Add gradient check failed");

        // Test mul gradient
        let x = RawTensor::new(vec![2.0, 3.0], &[2], true);
        let y = RawTensor::new(vec![4.0, 5.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.elem_mul(&y);
            z.sum()
        });
        assert!(passed, "Mul gradient check failed");

        // Test div gradient
        let x = RawTensor::new(vec![6.0, 8.0], &[2], true);
        let y = RawTensor::new(vec![2.0, 4.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.div(&y);
            z.sum()
        });
        assert!(passed, "Div gradient check failed");
    }

    #[test]
    fn test_gradcheck_matmul() {
        // Test matmul gradient for first operand
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let w = RawTensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            &[3, 3],
            false,
        );
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.matmul(&w);
            y.sum()
        });
        assert!(passed, "Matmul gradient check failed");
    }

    #[test]
    fn test_gradcheck_broadcast() {
        // Test broadcasting gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let y = RawTensor::new(vec![0.5], &[1], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.elem_mul(&y);
            z.sum()
        });
        assert!(passed, "Broadcast gradient check failed");

        // Test with matrix broadcast
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let y = RawTensor::new(vec![0.5, 1.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.add(&y);
            z.sum()
        });
        assert!(passed, "Matrix broadcast gradient check failed");
    }

    #[test]
    fn test_gradcheck_movement_ops() {
        // Test reshape gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.reshape(&[2, 2]);
            y.sum()
        });
        assert!(passed, "Reshape gradient check failed");

        // Test permute gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.permute(&[1, 0]);
            y.sum()
        });
        assert!(passed, "Permute gradient check failed");

        // Test pad gradient
        let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.pad(&[(1, 1)]);
            y.sum()
        });
        assert!(passed, "Pad gradient check failed");

        // Test shrink gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.shrink(&[(1, 3)]);
            y.sum()
        });
        assert!(passed, "Shrink gradient check failed");
    }

    #[test]
    fn test_gradcheck_reduce_ops() {
        // Test mean gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.mean());
        assert!(passed, "Mean gradient check failed");

        // Test max gradient (more challenging due to discontinuity)
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.max_reduce());
        assert!(passed, "Max gradient check failed");
    }

    #[test]
    fn test_gradcheck_ternary_ops() {
        // Test mulacc gradient
        let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let y = RawTensor::new(vec![3.0, 4.0], &[2], false);
        let z = RawTensor::new(vec![0.5, 1.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let out = t.mulacc(&y, &z);
            out.sum()
        });
        assert!(passed, "MulAcc gradient check failed");
    }

    #[test]
    fn test_gradcheck_complex_chain() {
        // Test complex computation graph
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let w = RawTensor::new(vec![0.5, 1.0, 1.5], &[3], false);

        let passed = RawTensor::check_gradients_simple(&x, |t| {
            // y = sigmoid(x * w)
            let prod = t.elem_mul(&w);
            let y = prod.sigmoid();
            y.sum()
        });
        assert!(passed, "Complex chain gradient check failed");
    }

    #[test]
    fn test_gradcheck_neural_network_layer() {
        // Test full linear layer: y = xW + b
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], true);
        let w = RawTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2], false);
        let b = RawTensor::new(vec![0.1, 0.2], &[2], false);

        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.matmul(&w);
            let z = y.add(&b);
            z.sum()
        });
        assert!(passed, "Neural network layer gradient check failed");
    }

    #[test]
    fn test_gradcheck_with_tolerance() {
        // Test with custom epsilon and tolerance
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);

        let (max_err, mean_err, passed) = RawTensor::check_gradients(
            &x,
            |t| {
                let y = t.relu();
                y.sum()
            },
            1e-5, // smaller epsilon
            1e-2, // larger tolerance (ReLU has discontinuity at 0)
        );

        assert!(passed, "Custom tolerance gradient check failed");
        println!(
            "ReLU gradcheck: max_err={:.6e}, mean_err={:.6e}",
            max_err, mean_err
        );
    }

    #[test]
    fn test_gradcheck_multidim() {
        // Test with 2D tensors
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sqrt();
            let z = y.elem_mul(t);
            z.sum()
        });
        assert!(passed, "Multidim gradient check failed");
    }

    #[test]
    fn test_gradcheck_expand() {
        let x = RawTensor::new(vec![1.0, 2.0], &[2, 1], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.expand(&[2, 3]);
            y.sum()
        });
        assert!(passed, "Expand gradient check failed");
    }

    #[test]
    fn test_gradcheck_pad() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.pad(&[(1, 1)]);
            y.sum()
        });
        assert!(passed, "Pad gradient check failed");
    }

    #[test]
    fn test_gradcheck_shrink() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.shrink(&[(1, 4)]);
            y.sum()
        });
        assert!(passed, "Shrink gradient check failed");
    }

    #[test]
    fn test_gradcheck_stride() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.stride_op(&[2]);
            y.sum()
        });
        assert!(passed, "Stride gradient check failed");
    }

    #[test]
    fn test_gradcheck_matmul_vec() {
        // vec-mat: (n,) @ (n,p) -> (p,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let w = RawTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.matmul(&w);
            y.sum()
        });
        assert!(passed, "Vec-mat matmul gradient check failed");
    }
    #[test]
    fn test_broadcast_3d_fix() {
        // (2,1) broadcasted with (1,2,3) -> (1,2,3)
        let x = RawTensor::new(vec![10.0, 20.0], &[2, 1], true);
        let y = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], true);
        let z = x.add(&y);

        assert_eq!(z.borrow().shape, vec![1, 2, 3]);
        // Row 0: [1,2,3] + 10 = [11,12,13]
        // Row 1: [4,5,6] + 20 = [24,25,26]
        assert_eq!(z.borrow().data, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

        z.backward();
        assert_eq!(x.grad(), Some(vec![3.0, 3.0])); // sum over last dim
        assert_eq!(y.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_broadcast_batch_channels() {
        // Typical conv bias: (B,C,H,W) + (C,1,1) -> (B,C,H,W)
        let x = RawTensor::new((0..16).map(|i| i as f32).collect(), &[2, 2, 2, 2], true);
        let bias = RawTensor::new(vec![0.1, 0.2], &[2, 1, 1], true);
        let z = x.add(&bias);

        assert_eq!(z.borrow().shape, vec![2, 2, 2, 2]);
        let loss = z.sum();
        loss.backward();

        // Bias grad should sum over B,H,W -> [8.0, 8.0]
        assert_eq!(bias.grad(), Some(vec![8.0, 8.0]));
    }

    #[test]
    fn test_gradcheck_broadcast_3d() {
        let x = RawTensor::new(vec![1.0, 2.0], &[2, 1], true);
        let y = RawTensor::new(vec![0.5; 6], &[1, 2, 3], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.add(&y).sum());
        assert!(passed, "3D broadcast gradcheck failed");
    }
    #[test]
    fn test_sequential_forward() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 4, true)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 2, true)),
        ]);

        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], true);
        let y = model.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2]);

        let loss = y.sum();
        loss.backward();

        // All layer params should have gradients
        for param in model.parameters() {
            assert!(param.grad().is_some(), "Missing gradient");
        }
    }

    #[test]
    fn test_sequential_zero_grad() {
        let mut model = Sequential::new(vec![Box::new(Linear::new(2, 3, true))]);

        let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], true);
        model.forward(&x).sum().backward();

        // Params have grads
        assert!(model.parameters()[0].grad().is_some());

        model.zero_grad();

        // Grads cleared
        for p in model.parameters() {
            assert!(p.grad().is_none());
        }
    }
    #[test]
    fn test_adam_converges_faster() {
        // Synthetic dataset: learn XOR-like function
        // Inputs: 4 samples, 2 features
        let x_data = vec![
            0.0, 0.0, // -> 0
            0.0, 1.0, // -> 1
            1.0, 0.0, // -> 1
            1.0, 1.0, // -> 0
        ];
        let x = RawTensor::new(x_data, &[4, 2], false);

        let y_data = vec![0.0, 1.0, 1.0, 0.0];
        let y = RawTensor::new(y_data, &[4], false);

        // Model: 2 -> 4 -> 1
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4, true)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 1, true)),
        ]);

        let params = model.parameters();
        let mut opt = Adam::new(params, 0.1, (0.9, 0.999), 1e-8);

        let mut losses = vec![];
        for epoch in 0..150 {
            opt.zero_grad();

            let pred = model.forward(&x).reshape(&[4]);
            let loss = RawTensor::mse_loss(&pred, &y);
            loss.backward();
            opt.step();

            losses.push(loss.borrow().data[0]);

            if epoch % 10 == 0 {
                println!("Epoch {}: loss={:.6}", epoch, losses[epoch]);
            }
        }

        // Should converge to <0.12 in 150 epochs
        assert!(
            losses[149] < 0.12,
            "Adam failed to converge: final loss={:.6}",
            losses[149]
        );

        // Loss should be monotonically decreasing (with some tolerance)
        let mid_loss = losses[75];
        let final_loss = losses[149];
        assert!(
            final_loss < mid_loss * 0.5,
            "Loss not decreasing fast enough"
        );
    }
    #[test]
    fn test_adam_vs_sgd() {
        // Same setup, train two models
        fn train_model(use_adam: bool) -> f32 {
            let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
            let x = RawTensor::new(x_data, &[4, 2], false);
            let y_data = vec![0.0, 1.0, 1.0, 0.0];
            let y = RawTensor::new(y_data, &[4], false);

            let model = Sequential::new(vec![
                Box::new(Linear::new(2, 8, true)),
                Box::new(ReLU),
                Box::new(Linear::new(8, 1, true)),
            ]);

            let params = model.parameters();

            if use_adam {
                let mut opt = Adam::new(params, 0.05, (0.9, 0.999), 1e-8);
                for _ in 0..50 {
                    opt.zero_grad();
                    let pred = model.forward(&x).reshape(&[4]);
                    let loss = RawTensor::mse_loss(&pred, &y);
                    loss.backward();
                    opt.step();
                }
            } else {
                let mut opt = SGD::new(params, 0.01, 0.0);
                for _ in 0..50 {
                    opt.zero_grad();
                    let pred = model.forward(&x).reshape(&[4]);
                    let loss = RawTensor::mse_loss(&pred, &y);
                    loss.backward();
                    opt.step();
                }
            }

            // Return final loss
            let pred = model.forward(&x).reshape(&[4]);
            RawTensor::mse_loss(&pred, &y).borrow().data[0]
        }

        let adam_loss = train_model(true);
        let sgd_loss = train_model(false);

        println!(
            "Adam final loss: {:.6}, SGD final loss: {:.6}",
            adam_loss, sgd_loss
        );

        // Adam should be significantly better
        assert!(adam_loss < sgd_loss * 0.75, "Adam not outperforming SGD");
    }
    #[test]
    fn test_dataloader_iteration() {
        // 8 samples, 2 features each
        let data = (0..16).map(|i| i as f32).collect();
        let targets = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let mut loader = DataLoader::new(
            data,
            targets,
            &[2],  // 2 features per sample
            &[1],  // 1 target per sample
            3,     // batch_size
            false, // no shuffle for deterministic test
        );

        // First batch: samples 0,1,2
        let (x, y) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![3, 2]);
        assert_eq!(x.borrow().data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(y.borrow().shape, vec![3, 1]);

        // Second batch: samples 3,4,5
        let (x, _y) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![3, 2]);

        // Third batch: samples 6,7 (partial)
        let (x, _y) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![2, 2]);

        // Done
        assert!(loader.next().is_none());

        // Reset
        loader.reset();
        let (x, _) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![3, 2]);
    }

    #[test]
    fn test_dataloader_in_training_loop() {
        let data = vec![0.0; 40]; // 10 samples, 4 features
        let targets = vec![1.0; 10];

        let model = Sequential::new(vec![Box::new(Linear::new(4, 2, true))]);

        let mut opt = SGD::new(model.parameters(), 0.1, 0.0);

        for epoch in 0..2 {
            let loader = DataLoader::new(data.clone(), targets.clone(), &[4], &[1], 3, false);

            for (batch_x, _batch_y) in loader {
                opt.zero_grad();
                let pred = model.forward(&batch_x);
                // Dummy loss
                let loss = pred.sum();
                loss.backward();
                opt.step();
            }

            println!("Epoch {} complete", epoch);
        }
    }
    #[test]
    fn bench_matmul_speedup() {
        use std::time::Instant;

        let a = vec![1.0; 256 * 256];
        let b = vec![1.0; 256 * 256];

        let start = Instant::now();
        let _ = RawTensor::matmul_raw(&a, &b, 256, 256, 256);
        let duration = start.elapsed();

        println!("256x256 matmul: {:?}", duration);

        #[cfg(feature = "accelerate")]
        assert!(
            duration.as_millis() < 10,
            "BLAS should be <10ms, got {:?}",
            duration
        );

        #[cfg(not(feature = "accelerate"))]
        assert!(
            duration.as_millis() > 50,
            "Naive should be >50ms, got {:?}",
            duration
        );
    }
}

#[cfg(test)]
mod axis_reduce_tests {
    use super::*;

    #[test]
    fn test_sum_dim_basic() {
        // [2,3] sum along dim=1 -> [2]
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = RawTensor::sum_dim(&x, 1, false);

        assert_eq!(y.borrow().shape, vec![2]);
        assert_eq!(y.borrow().data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_sum_dim_keepdim() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = RawTensor::sum_dim(&x, 0, true);

        assert_eq!(y.borrow().shape, vec![1, 2]);
        assert_eq!(y.borrow().data, vec![4.0, 6.0]); // [1+3, 2+4]
    }

    #[test]
    fn test_sum_dim_backward() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = RawTensor::sum_dim(&x, 1, false); // [6, 15]
        y.backward();

        // Gradient broadcasts back: each element contributed once
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_max_dim_basic() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], false);
        let y = RawTensor::max_dim(&x, 1, false);

        assert_eq!(y.borrow().shape, vec![2]);
        assert_eq!(y.borrow().data, vec![5.0, 8.0]); // max of each row
    }

    #[test]
    fn test_max_dim_backward() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], true);
        let y = RawTensor::max_dim(&x, 1, false);
        y.backward();

        // Only max elements get gradient
        assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_gradcheck_sum_dim() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let passed =
            RawTensor::check_gradients_simple(&x, |t| RawTensor::sum_dim(t, 0, false).sum());
        assert!(passed, "sum_dim gradient check failed");
    }

    #[test]
    fn test_gradcheck_max_dim() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], true);
        let passed =
            RawTensor::check_gradients_simple(&x, |t| RawTensor::max_dim(t, 1, false).sum());
        assert!(passed, "max_dim gradient check failed");
    }

    #[test]
    fn test_softmax_forward() {
        // Test softmax computation
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = RawTensor::softmax(&x, 1);

        // Each row should sum to 1.0
        let row0_sum: f32 = y.borrow().data[0..3].iter().sum();
        let row1_sum: f32 = y.borrow().data[3..6].iter().sum();

        approx::assert_relative_eq!(row0_sum, 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(row1_sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gradcheck_softmax() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| RawTensor::softmax(t, 1).sum());
        assert!(passed, "Softmax gradient check failed");
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Simple 2-class, 2-sample batch
        let logits = RawTensor::new(vec![2.0, 1.0, 0.5, 2.5], &[2, 2], true);
        let targets = RawTensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], false);

        let loss = RawTensor::cross_entropy_loss(&logits, &targets);
        loss.backward();

        // Loss should be positive scalar
        assert_eq!(loss.borrow().shape, vec![1]);
        assert!(loss.borrow().data[0] > 0.0);

        // Gradients should exist and have correct shape
        assert_eq!(logits.grad().unwrap().len(), 4);
    }
}
