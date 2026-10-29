use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::index::{fetch, slot};
use crate::macros::define_ops;
use crate::shape::{Shape, Strides};
use crate::storage::Storage;
use crate::{RawTensor, Result, Tensor, VoltaError};

// One row per op. A differentiable row gives the forward, the per-element backward
// `(∂L/∂a, ∂L/∂b)` (before the reduction over broadcast dims, which `BinaryGradFn`
// does once for every op), and the GPU pipelines. A `non_differentiable` row has no
// backward, so its `edge()` grad fn is trivial and never attached. See `define_ops!`.
define_ops! {
    /// Binary operations: two inputs, one output
    ///
    /// Broadcasting is automatically handled for compatible shapes.
    /// Non-differentiable operations (Mod, Cmplt) return tensors with `requires_grad=false`.
    binary BinaryOp {
        differentiable {
            /// x + y
            Add => |a, b| a + b,
                |g, _a, _b| (g, g),
                gpu(add, add_backward_a, add_backward_b, add_broadcast_pass1, add_broadcast_pass2);
            /// x - y
            Sub => |a, b| a - b,
                |g, _a, _b| (g, -g),
                gpu(sub, sub_backward_a, sub_backward_b, sub_broadcast_pass1, sub_broadcast_pass2);
            /// x * y (element-wise)
            Mul => |a, b| a * b,
                |g, a, b| (g * b, g * a),
                gpu(mul, mul_backward_a, mul_backward_b, mul_broadcast_pass1, mul_broadcast_pass2);
            /// x / y (element-wise): ∂/∂x = 1/y, ∂/∂y = -x/y²
            Div => |a, b| a / b,
                |g, a, b| (g / b, -g * a / (b * b)),
                gpu(div, div_backward_a, div_backward_b, div_broadcast_pass1, div_broadcast_pass2);
            /// max(x, y) (element-wise): the gradient goes to the larger input, and to `x` on ties
            Max => |a, b| a.max(b),
                |g, a, b| (if a >= b { g } else { 0.0 }, if b > a { g } else { 0.0 }),
                gpu(max, max_backward_a, max_backward_b, max_broadcast_pass1, max_broadcast_pass2);
        }
        non_differentiable {
            /// x % y
            Mod => |a, b| a % b, gpu(mod_op);
            /// x < y ? 1 : 0
            Cmplt => |a, b| if a < b { 1.0 } else { 0.0 }, gpu(cmplt);
        }
    }
}

/// Gradient function for binary operations
///
/// Handles broadcasting during backward pass - gradients must be summed
/// over dimensions that were broadcast in the forward pass.
#[derive(Clone)]
pub(crate) struct BinaryGradFn {
    op: BinaryOp,
}

impl BinaryGradFn {
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let [x_ref, y_ref] = parents else {
            panic!("binary ops require 2 parents, got {}", parents.len());
        };
        let x = x_ref.borrow();
        let y = y_ref.borrow();

        if let Some(grads) = RawTensor::gpu_binary_backward(self.op, out_grad, &x, &y) {
            return grads;
        }

        // CPU path: the table gives the per-element (∂a, ∂b) at the broadcast shape;
        // each is then summed back down to its operand's shape.
        let x_bc = RawTensor::broadcast_to(&x.data.to_host_f32(), &x.shape, &out_grad.shape);
        let y_bc = RawTensor::broadcast_to(&y.data.to_host_f32(), &y.shape, &out_grad.shape);
        let (grad_x, grad_y): (Vec<f32>, Vec<f32>) = out_grad
            .data
            .to_host_f32()
            .iter()
            .zip(&x_bc)
            .zip(&y_bc)
            .map(|((&g, &a), &b)| self.op.backward(g, a, b))
            .unzip();

        let reduce_to = |full: &[f32], operand: &RawTensor| {
            let summed = RawTensor::sum_over_broadcast_dims(full, &out_grad.shape, &operand.shape);
            RawTensor::new(summed, &operand.shape, false)
        };
        vec![
            x.requires_grad.then(|| reduce_to(&grad_x, &x)),
            y.requires_grad.then(|| reduce_to(&grad_y, &y)),
        ]
    }

    pub(crate) fn is_trivial(&self) -> bool {
        self.op.is_trivial()
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
        match Self::try_broadcast_shape(shape_a, shape_b) {
            Ok(shape) => shape,
            Err(VoltaError::BroadcastError(a, b)) => {
                // Find the first incompatible dimension for better error message
                let max_len = a.len().max(b.len());
                for i in 0..max_len {
                    let a_dim = if i < a.len() && !a.is_empty() {
                        fetch(&a, a.len() - 1 - i)
                    } else {
                        1
                    };
                    let b_dim = if i < b.len() && !b.is_empty() {
                        fetch(&b, b.len() - 1 - i)
                    } else {
                        1
                    };
                    if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
                        let dim = max_len - 1 - i;
                        panic!(
                            "Cannot broadcast shapes {shape_a:?} and {shape_b:?} at dimension {dim}"
                        );
                    }
                }
                panic!("Cannot broadcast shapes {shape_a:?} and {shape_b:?}");
            }
            Err(e) => panic!("Broadcast error: {e}"),
        }
    }

    /// Compute the broadcast result shape (fallible)
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::BroadcastError` if shapes cannot be broadcast together, or
    /// `VoltaError::InvalidParameter` if an operand or the result is not a valid
    /// [`Shape`] (zero dimension, or more than `MAX_NUMEL` elements)
    pub fn try_broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
        let a = Shape::from_dims(shape_a)?;
        let b = Shape::from_dims(shape_b)?;
        Ok(a.broadcast(&b)?.truncate_rank().to_vec())
    }

    /// Broadcast data from one shape to another
    ///
    /// This repeats values along dimensions where `from_shape` is 1
    /// and `to_shape` is larger.
    pub(crate) fn broadcast_to(data: &[f32], from_shape: &[usize], to_shape: &[usize]) -> Vec<f32> {
        if from_shape == to_shape {
            return data.to_vec();
        }

        let to_size = Shape::from_dims(to_shape)
            .unwrap_or_else(|e| {
                panic!("Broadcast: invalid target shape {to_shape:?} from {from_shape:?}: {e}")
            })
            .numel();
        let mut result = vec![0.0; to_size];

        // Pad from_shape with leading 1s to match rank
        let mut padded_from = vec![1; to_shape.len()];
        let offset = to_shape.len() - from_shape.len();
        padded_from
            .get_mut(offset..)
            .expect("offset = rank difference, so it is within the padded rank")
            .copy_from_slice(from_shape);
        let from_strides_padded = Strides::from_shape(&padded_from);
        let from_strides_padded = from_strides_padded.truncate_rank();
        let to_strides = Strides::from_shape(to_shape);
        let to_strides = to_strides.truncate_rank();

        // For each output position, compute corresponding input position
        for i in 0..to_size {
            let mut from_idx = 0;
            let mut remainder = i;
            //calc coords based on to_shape and map
            for dim in 0..to_shape.len() {
                let stride = fetch(to_strides, dim);
                let coord = remainder / stride;
                remainder %= stride;
                //if dim broadcast (size was 1) use coord 0 for from_idx
                if fetch(&padded_from, dim) != 1 {
                    let from_stride = fetch(from_strides_padded, dim);
                    from_idx += coord * from_stride;
                }
            }
            *slot(&mut result, i) = fetch(data, from_idx);
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
        padded_target
            .get_mut(offset..)
            .expect("offset = rank difference, so it is within the padded rank")
            .copy_from_slice(target_shape);

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
        let target_strides = Strides::from_shape(target_shape);
        let target_strides = target_strides.truncate_rank();
        let grad_strides = Strides::from_shape(grad_shape);
        let grad_strides = grad_strides.truncate_rank();

        // For each gradient element, sum into appropriate result position
        for (i, &grad_val) in grad.iter().enumerate() {
            let mut target_idx = 0;
            let mut remainder = i;

            // convert lin index i in grad_shape to coords and map to target
            for dim in 0..grad_shape.len() {
                let grad_stride = fetch(grad_strides, dim);
                let coord = remainder / grad_stride;
                remainder %= grad_stride;

                // Map to target coordinate (skip if was broadcast)
                if dim >= offset && fetch(&padded_target, dim) != 1 {
                    let target_dim_idx = dim - offset;
                    let target_stride = fetch(target_strides, target_dim_idx);
                    target_idx += coord * target_stride;
                }
            }
            *slot(&mut result, target_idx) += grad_val;
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
    #[must_use]
    pub fn binary_op(self_t: &Tensor, other: &Tensor, op: BinaryOp) -> Tensor {
        // If both operands are on the same GPU with equal shapes, run the kernel there;
        // otherwise (or if it fails) fall through to the CPU path.
        if let Some((shape, storage, device)) = Self::gpu_binary_forward(self_t, other, op) {
            return edge(
                [self_t.clone(), other.clone()],
                storage,
                Shape::derived(shape),
                device,
                GradFn::Binary(BinaryGradFn { op }),
            );
        }

        let (data_a, shape_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };
        let (data_b, shape_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone())
        };

        // CPU path: compute broadcast shape and perform the operation on host data.
        let out_shape = Self::broadcast_shape(&shape_a, &shape_b);

        // Broadcast inputs to output shape
        // Check for reasonable sizes to prevent memory issues
        let out_size: usize = out_shape.iter().product();
        assert!(out_size > 0, "Invalid broadcast result size");

        let bc_data_a = Self::broadcast_to(&data_a.to_host_f32(), &shape_a, &out_shape);
        let bc_data_b = Self::broadcast_to(&data_b.to_host_f32(), &shape_b, &out_shape);

        let result_data: Vec<f32> = bc_data_a
            .iter()
            .zip(&bc_data_b)
            .map(|(&a, &b)| op.forward(a, b))
            .collect();

        edge(
            [self_t.clone(), other.clone()],
            Storage::cpu(result_data),
            Shape::derived(out_shape),
            Device::CPU,
            GradFn::Binary(BinaryGradFn { op }),
        )
    }

    // Convenience methods for binary operations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{check_gradients, manual_seed, randn};

    /// Differentiable ops only: the gradcheck below needs a backward to check.
    fn differentiable() -> impl Iterator<Item = BinaryOp> {
        BinaryOp::ALL.iter().copied().filter(|op| !op.is_trivial())
    }

    /// Random operands in each op's domain, away from Div's pole and from Max ties
    /// (the finite-difference step must not straddle `a == b`).
    fn operands(op: BinaryOp, shape_a: &[usize], shape_b: &[usize]) -> (Tensor, Tensor) {
        for seed in 0..64_u64 {
            manual_seed(0xB1A5 + seed);
            let a = randn(shape_a);
            let b = randn(shape_b);
            let av: Vec<f32> = a.data().clone();
            let bv: Vec<f32> = match op {
                BinaryOp::Div => b.data().iter().map(|v| 0.5 + v.abs()).collect(),
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Max
                | BinaryOp::Mod
                | BinaryOp::Cmplt => b.data().clone(),
            };
            let (a, b) = (
                RawTensor::new(av, shape_a, true),
                RawTensor::new(bv, shape_b, true),
            );
            if op != BinaryOp::Max {
                return (a, b);
            }
            let out_shape = RawTensor::broadcast_shape(shape_a, shape_b);
            let a_bc = RawTensor::broadcast_to(&a.data(), shape_a, &out_shape);
            let b_bc = RawTensor::broadcast_to(&b.data(), shape_b, &out_shape);
            if a_bc.iter().zip(&b_bc).all(|(x, y)| (x - y).abs() > 0.05) {
                return (a, b);
            }
        }
        panic!("no tie-free Max operands for {shape_a:?} x {shape_b:?}");
    }

    #[test]
    fn binary_gradcheck_every_op_equal_broadcast_and_unequal_rank() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[2, 3], &[2, 3]),
            (&[2, 3], &[3]),
            (&[2, 1], &[1, 3]),
            (&[4, 2, 3], &[2, 1]),
        ];
        for op in differentiable() {
            for &(shape_a, shape_b) in cases {
                let (a, b) = operands(op, shape_a, shape_b);
                let out_shape = RawTensor::broadcast_shape(shape_a, shape_b);
                // Weighted sum, so every output element contributes a different gradient.
                manual_seed(0x17);
                let w = randn(&out_shape);

                let (err_a, _, ok_a) = check_gradients(
                    &a,
                    |t| RawTensor::binary_op(t, &b, op).elem_mul(&w).sum(),
                    1e-2,
                    1e-2,
                );
                assert!(
                    ok_a,
                    "{op:?} {shape_a:?}x{shape_b:?}: d/da max_error={err_a:.3e}"
                );

                let (err_b, _, ok_b) = check_gradients(
                    &b,
                    |t| RawTensor::binary_op(&a, t, op).elem_mul(&w).sum(),
                    1e-2,
                    1e-2,
                );
                assert!(
                    ok_b,
                    "{op:?} {shape_a:?}x{shape_b:?}: d/db max_error={err_b:.3e}"
                );
            }
        }
    }

    #[test]
    fn is_trivial_exactly_for_mod_and_cmplt() {
        for &op in BinaryOp::ALL {
            assert_eq!(
                op.is_trivial(),
                matches!(op, BinaryOp::Mod | BinaryOp::Cmplt),
                "{op:?}"
            );
        }
    }

    #[test]
    fn non_differentiable_ops_attach_no_grad_fn() {
        let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let b = RawTensor::new(vec![2.0, 2.0, 2.0], &[3], true);
        for op in [BinaryOp::Mod, BinaryOp::Cmplt] {
            assert!(!RawTensor::binary_op(&a, &b, op).requires_grad(), "{op:?}");
        }
    }

    #[test]
    #[should_panic(expected = "non-differentiable")]
    fn backward_of_non_differentiable_op_panics() {
        let _ = BinaryOp::Mod.backward(1.0, 1.0, 2.0);
    }

    #[test]
    fn all_lists_each_variant_once() {
        assert_eq!(BinaryOp::ALL.len(), 7);
        for (i, a) in BinaryOp::ALL.iter().enumerate() {
            for b in BinaryOp::ALL.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn max_tie_sends_gradient_to_first_operand() {
        // Pins the documented tie rule (x on ties), so a table edit can't change it silently.
        assert_eq!(BinaryOp::Max.backward(1.0, 3.0, 3.0), (1.0, 0.0));
    }
}
