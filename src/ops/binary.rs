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
        if !matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Max
                | BinaryOp::Mod
                | BinaryOp::Cmplt
        ) {
            return None;
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
            BinaryOp::Max => RawTensor::gpu_max(&storage_a, &storage_b)?,
            BinaryOp::Mod => RawTensor::gpu_mod(&storage_a, &storage_b)?,
            BinaryOp::Cmplt => RawTensor::gpu_cmplt(&storage_a, &storage_b)?,
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
        let x_ref = parents.first().cloned().unwrap();
        let y_ref = parents.get(1).cloned().unwrap();
        let x_val = x_ref.borrow();
        let y_val = y_ref.borrow();

        // Check GPU path - use legacy path for same-shape, broadcast path for different shapes
        #[cfg(feature = "gpu")]
        {
            if out_grad.device.is_gpu() && x_val.device.is_gpu() && y_val.device.is_gpu() {
                // For same-shape case, use legacy path (more tested)
                if out_grad.shape == x_val.shape && out_grad.shape == y_val.shape {
                    if let Some((kernel_a, kernel_b)) = binary_backward_kernel_names(self.op) {
                        let gx = if x_val.requires_grad {
                            RawTensor::gpu_binary_backward_a(
                                &out_grad.data,
                                &x_val.data,
                                &y_val.data,
                                kernel_a,
                            )
                            .map(|storage| {
                                RawTensor::new_with_storage(
                                    storage,
                                    &x_val.shape,
                                    x_val.device.clone(),
                                    false,
                                )
                            })
                        } else {
                            None
                        };

                        let gy = if y_val.requires_grad {
                            RawTensor::gpu_binary_backward_b(
                                &out_grad.data,
                                &x_val.data,
                                &y_val.data,
                                kernel_b,
                            )
                            .map(|storage| {
                                RawTensor::new_with_storage(
                                    storage,
                                    &y_val.shape,
                                    y_val.device.clone(),
                                    false,
                                )
                            })
                        } else {
                            None
                        };

                        // If GPU path succeeded, return early
                        if (x_val.requires_grad && gx.is_some())
                            || (y_val.requires_grad && gy.is_some())
                        {
                            return vec![gx, gy];
                        }
                    }
                } else if let Some(broadcast_kernel) =
                    binary_backward_broadcast_kernel_name(self.op)
                {
                    // For different shapes (broadcasting), try RACE-FREE broadcast path
                    if let Some((grad_a_storage, grad_b_storage)) =
                        RawTensor::gpu_binary_backward_broadcast_safe(
                            &out_grad.data,
                            &x_val.data,
                            &y_val.data,
                            broadcast_kernel,
                            &out_grad.shape,
                            &x_val.shape,
                            &y_val.shape,
                        )
                    {
                        let gx = if x_val.requires_grad {
                            Some(RawTensor::new_with_storage(
                                grad_a_storage,
                                &x_val.shape,
                                x_val.device.clone(),
                                false,
                            ))
                        } else {
                            None
                        };

                        let gy = if y_val.requires_grad {
                            Some(RawTensor::new_with_storage(
                                grad_b_storage,
                                &y_val.shape,
                                y_val.device.clone(),
                                false,
                            ))
                        } else {
                            None
                        };

                        return vec![gx, gy];
                    }
                    // If all GPU paths failed, fall through to CPU path
                }
            }
        }

        // CPU fallback
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

// Map a `BinaryOp` to the corresponding GPU backward kernel names, if supported.
// Returns (kernel_for_grad_a, kernel_for_grad_b)
#[cfg(feature = "gpu")]
fn binary_backward_kernel_names(op: BinaryOp) -> Option<(&'static str, &'static str)> {
    match op {
        BinaryOp::Add => Some(("add_backward_a", "add_backward_b")),
        BinaryOp::Sub => Some(("sub_backward_a", "sub_backward_b")),
        BinaryOp::Mul => Some(("mul_backward_a", "mul_backward_b")),
        BinaryOp::Div => Some(("div_backward_a", "div_backward_b")),
        BinaryOp::Max => Some(("max_backward_a", "max_backward_b")),
        BinaryOp::Mod | BinaryOp::Cmplt => None, // Non-differentiable
    }
}

/// Get the broadcast kernel name for a binary operation
fn binary_backward_broadcast_kernel_name(op: BinaryOp) -> Option<&'static str> {
    match op {
        BinaryOp::Add => Some("add"),
        BinaryOp::Sub => Some("sub"),
        BinaryOp::Mul => Some("mul"),
        BinaryOp::Div => Some("div"),
        BinaryOp::Max => Some("max"),
        BinaryOp::Mod | BinaryOp::Cmplt => None, // Non-differentiable
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
    /// # Panics
    /// broadcast failures
    #[must_use]
    pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Vec<usize> {
        let max_len = shape_a.len().max(shape_b.len());
        let mut result = vec![1; max_len];

        // Align from right (trailing dimensions)
        for i in 0..max_len {
            let a_dim = if i < shape_a.len() && !shape_a.is_empty() {
                let idx = shape_a.len() - 1 - i;
                shape_a.get(idx).copied().unwrap_or(1)
            } else {
                1
            };
            let b_dim = if i < shape_b.len() && !shape_b.is_empty() {
                let idx = shape_b.len() - 1 - i;
                shape_b.get(idx).copied().unwrap_or(1)
            } else {
                1
            };

            let result_idx = max_len - 1 - i;
            if let Some(slot) = result.get_mut(result_idx) {
                if a_dim == b_dim {
                    *slot = a_dim;
                } else if a_dim == 1 {
                    *slot = b_dim;
                } else if b_dim == 1 {
                    *slot = a_dim;
                } else {
                    panic!("Cannot broadcast shapes {shape_a:?} and {shape_b:?} at dimension {i}");
                }
            }
        }
        result
    }

    /// Broadcast data from one shape to another
    ///
    /// This repeats values along dimensions where `from_shape` is 1
    /// and `to_shape` is larger.
    pub(crate) fn broadcast_to(data: &[f32], from_shape: &[usize], to_shape: &[usize]) -> Vec<f32> {
        const MAX_ALLOC: usize = 100_000_000;

        if from_shape == to_shape {
            return data.to_vec();
        }

        let to_size: usize = to_shape.iter().product();
        assert!(
            to_size <= MAX_ALLOC,
            "Broadcast would create tensor with {to_size} elements (max: {MAX_ALLOC}). Check shapes {from_shape:?} -> {to_shape:?}"
        );
        let mut result = vec![0.0; to_size];

        // Pad from_shape with leading 1s to match rank
        let mut padded_from = vec![1; to_shape.len()];
        let offset = to_shape.len() - from_shape.len();
        if let Some(dest) = padded_from.get_mut(offset..) {
            dest.copy_from_slice(from_shape);
        }
        let from_strides_padded = Self::compute_strides(&padded_from);
        let to_strides = Self::compute_strides(to_shape);

        // For each output position, compute corresponding input position
        #[allow(clippy::needless_range_loop)]
        for i in 0..to_size {
            let mut from_idx = 0;
            let mut remainder = i;
            //calc coords based on to_shape and map
            for dim in 0..to_shape.len() {
                let stride = to_strides.get(dim).copied().unwrap_or(1);
                let coord = remainder / stride;
                remainder %= stride;
                //if dim broadcast (size was 1) use coord 0 for from_idx
                if padded_from.get(dim).copied().unwrap_or(1) != 1 {
                    let from_stride = from_strides_padded.get(dim).copied().unwrap_or(1);
                    from_idx += coord * from_stride;
                }
            }
            if let Some(&src_val) = data.get(from_idx)
                && let Some(slot) = result.get_mut(i)
            {
                *slot = src_val;
            }
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
        if let Some(dest) = padded_target.get_mut(offset..) {
            dest.copy_from_slice(target_shape);
        }

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
                let grad_stride = grad_strides.get(dim).copied().unwrap_or(1);
                let coord = remainder / grad_stride;
                remainder %= grad_stride;

                // Map to target coordinate (skip if was broadcast)
                if dim >= offset && padded_target.get(dim).copied().unwrap_or(1) != 1 {
                    let target_dim_idx = dim - offset;
                    let target_stride = target_strides.get(target_dim_idx).copied().unwrap_or(1);
                    target_idx += coord * target_stride;
                }
            }
            if target_idx < result.len()
                && let Some(slot) = result.get_mut(target_idx)
            {
                *slot += grad_val;
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
    /// # Panics
    /// broadcast failure
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
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Max => {
                req_a || req_b
            }
        };

        // If both operands are already on the same GPU and we have a matching
        // kernel, try to perform the operation there and fall back to CPU
        // otherwise.
        #[cfg(feature = "gpu")]
        {
            if RawTensor::common_gpu_device(&[self_t, other]).is_some()
                && let Some((shape, storage, device)) =
                    Self::try_gpu_binary_result(self_t, other, op)
            {
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
