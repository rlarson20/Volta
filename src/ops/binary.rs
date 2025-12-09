use crate::autograd::GradFn;
use crate::device::Device;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Binary operations: two inputs, one output
///
/// Broadcasting is automatically handled for compatible shapes.
/// Non-differentiable operations (Mod, Cmplt) return tensors with `requires_grad=false`.
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

impl RawTensor {
    #[cfg(feature = "gpu")]
    fn try_gpu_binary_result(
        self_t: &Tensor,
        other: &Tensor,
        op: BinaryOp,
    ) -> Option<(Vec<usize>, Storage, Device)> {
        // Only these ops have GPU kernels at the moment.
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {}
            _ => return None,
        }

        let (shape_a, device_a, storage_a) = {
            let a = self_t.borrow();
            (a.shape.clone(), a.device.clone(), a.data.clone())
        };
        let (shape_b, device_b, storage_b) = {
            let b = other.borrow();
            (b.shape.clone(), b.device.clone(), b.data.clone())
        };

        // Both tensors must be on the same GPU device.
        if !device_a.is_gpu() || !device_b.is_gpu() || device_a != device_b {
            return None;
        }

        // Broadcasting on GPU is not yet implemented; fall back to CPU when
        // shapes differ.
        if shape_a != shape_b {
            return None;
        }

        let storage = match op {
            BinaryOp::Add => RawTensor::gpu_add(&storage_a, &storage_b)?,
            BinaryOp::Sub => RawTensor::gpu_sub(&storage_a, &storage_b)?,
            BinaryOp::Mul => RawTensor::gpu_mul(&storage_a, &storage_b)?,
            BinaryOp::Div => RawTensor::gpu_div(&storage_a, &storage_b)?,
            _ => unreachable!(),
        };

        Some((shape_a, storage, device_a))
    }
}

/// Gradient function for binary operations
///
/// Handles broadcasting during backward pass - gradients must be summed
/// over dimensions that were broadcast in the forward pass.
pub struct BinaryGradFn {
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

// ===== BINARY OPERATIONS =====
impl RawTensor {
    /// Compute broadcast shape following `NumPy` broadcasting rules
    ///
    /// Rules:
    /// 1. Align shapes from the right (trailing dimensions)
    /// 2. For each dimension, both must be equal OR one must be 1
    /// 3. Output dimension is the maximum of the two
    ///
    /// Examples:
    /// - (3, 1) + (1, 4) -> (3, 4)
    /// - (5, 3, 1) + (1, 4) -> (5, 3, 4)
    pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Vec<usize> {
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
    /// This repeats values along dimensions where `from_shape` is 1
    /// and `to_shape` is larger.
    pub(crate) fn broadcast_to(data: &[f32], from_shape: &[usize], to_shape: &[usize]) -> Vec<f32> {
        if from_shape == to_shape {
            return data.to_vec();
        }

        let to_size: usize = to_shape.iter().product();
        const MAX_ALLOC: usize = 100_000_000;
        assert!(
            to_size <= MAX_ALLOC,
            "Broadcast would create tensor with {} elements (max: {}). Check shapes {:?} -> {:?}",
            to_size,
            MAX_ALLOC,
            from_shape,
            to_shape
        );
        let mut result = vec![0.0; to_size];

        // Pad from_shape with leading 1s to match rank
        let mut padded_from = vec![1; to_shape.len()];
        let offset = to_shape.len() - from_shape.len();
        padded_from[offset..].copy_from_slice(from_shape);
        let from_strides_padded = Self::compute_strides(&padded_from);
        let to_strides = Self::compute_strides(to_shape);

        // For each output position, compute corresponding input position
        #[allow(clippy::needless_range_loop)]
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
    pub(crate) fn sum_over_broadcast_dims(
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

        // Mod and Cmplt are non-differentiable
        let requires_grad = match op {
            BinaryOp::Mod | BinaryOp::Cmplt => false,
            _ => req_a || req_b,
        };

        // If both operands are already on the same GPU and we have a matching
        // kernel, try to perform the operation there and fall back to CPU
        // otherwise.
        #[cfg(feature = "gpu")]
        {
            if let Some((shape, storage, device)) = Self::try_gpu_binary_result(self_t, other, op) {
                let out = Rc::new(RefCell::new(RawTensor {
                    data: storage,
                    shape,
                    grad: None,
                    requires_grad,
                    grad_fn: None,
                    parents: vec![self_t.clone(), other.clone()],
                    device,
                }));
                if requires_grad {
                    out.borrow_mut().grad_fn = Some(Box::new(BinaryGradFn { op }));
                }
                return out;
            }
        }

        // CPU path: compute broadcast shape and perform the operation on host data.
        let out_shape = Self::broadcast_shape(&shape_a, &shape_b);

        // Broadcast inputs to output shape
        // Check for reasonable sizes to prevent memory issues
        let out_size: usize = out_shape.iter().product();
        assert!(out_size > 0, "Invalid broadcast result size");

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
