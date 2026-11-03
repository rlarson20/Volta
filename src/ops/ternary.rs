use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::device_policy::kernel_failed;
use crate::dtype::DType;
use crate::real::{Real, compute_dtype, dispatch_bits, dispatch_real, promote_operands};
use crate::shape::Shape;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

/// Ternary operations: three inputs, one output
#[derive(Clone, Copy, Debug)]
pub enum TernaryOp {
    /// `x * y + z`, fused.
    MulAcc,
    /// `condition ? x : y`, selected elementwise.
    Where,
}

/// Gradient function for `MulAcc` (fused multiply-add)
///
/// z = x*y + w has gradients:
/// - ∂z/∂x = y
/// - ∂z/∂y = x
/// - ∂z/∂w = 1
#[derive(Clone)]
pub(crate) struct MulAccGradFn;
impl MulAccGradFn {
    /// Gradients for `x`, `y`, and the optional addend, reduced to each one's shape.
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // SAFETY: MulAcc operation always creates 2 or 3 parents (x, y, optional w)
        debug_assert!(
            parents.len() >= 2,
            "MulAcc grad requires at least 2 parents"
        );
        let x_ref = parents.first().cloned().unwrap();
        let y_ref = parents.get(1).cloned().unwrap();
        let x_val = x_ref.borrow();
        let y_val = y_ref.borrow();

        let dtype = out_grad.data.dtype();
        let product = |other: &Storage| {
            dispatch_real!(dtype, T => Storage::from_compute(dtype,
                out_grad
                    .data
                    .host::<T>()
                    .iter()
                    .zip(other.host::<T>().iter())
                    .map(|(&g, &o)| g * o)
                    .collect::<Vec<T>>(),
            ))
        };
        let wrap = |data: Storage| {
            RawTensor::new_with_storage(data, out_grad.shape.clone(), Device::CPU, false)
        };
        let grad_x = x_val.requires_grad.then(|| wrap(product(&y_val.data)));
        let grad_y = y_val.requires_grad.then(|| wrap(product(&x_val.data)));
        let grad_z = parents
            .get(2)
            .is_some_and(|p| p.borrow().requires_grad)
            // A host copy, not `out_grad.data.clone()`: on the GPU that would be GPU storage
            // under a CPU label. This CPU gradient is the gap `accumulate_grad` handles.
            .then(|| {
                wrap(dispatch_real!(dtype, T => {
                    Storage::from_compute(dtype, out_grad.data.host::<T>().to_vec())
                }))
            });

        vec![grad_x, grad_y, grad_z]
    }
}

/// Gradient function for Where (conditional selection)
///
/// Gradients flow through the branch that was selected, and must be reduced
/// along any dimensions that were broadcast.
#[derive(Clone)]
pub(crate) struct WhereGradFn {
    /// Condition broadcast to the output's shape: `true` selects the "true" branch.
    mask: Vec<bool>,
    /// Original shape of the "true" branch.
    true_shape: Shape,
    /// Original shape of the "false" branch.
    false_shape: Shape,
}

impl WhereGradFn {
    /// Routes the gradient to the selected branch and reduces over broadcast dims.
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // SAFETY: Where operation always creates exactly 2 parents (true_branch, false_branch)
        debug_assert!(parents.len() >= 2, "Where grad requires 2 parents");
        let true_ref = parents.first().cloned().unwrap();
        let false_ref = parents.get(1).cloned().unwrap();
        let true_parent = true_ref.borrow();
        let false_parent = false_ref.borrow();
        let needs_true = true_parent.requires_grad;
        let needs_false = false_parent.requires_grad;
        drop(true_parent);
        drop(false_parent);

        // The gradient goes to the branch each element came from, then is summed back
        // over the dims that branch was broadcast along.
        let branch_grad = |take: bool, shape: &Shape| {
            let grad = dispatch_real!(out_grad.data.dtype(), T => {
                let data: Vec<T> = out_grad
                    .data
                    .host::<T>()
                    .iter()
                    .zip(&self.mask)
                    .map(|(&g, &m)| if m == take { g } else { T::ZERO })
                    .collect();
                Storage::from_compute(out_grad.data.dtype(), RawTensor::sum_over_broadcast_dims(
                    &data,
                    out_grad.shape.truncate_rank(),
                    shape.truncate_rank(),
                ))
            });
            RawTensor::new_with_storage(grad, shape.clone(), Device::CPU, false)
        };
        let grad_true = needs_true.then(|| branch_grad(true, &self.true_shape));
        let grad_false = needs_false.then(|| branch_grad(false, &self.false_shape));

        vec![grad_true, grad_false]
    }
}

// ===== TERNARY OPERATIONS =====

impl RawTensor {
    /// Apply ternary operations (3 inputs, 1 output)
    /// # Panics
    /// assert shape match for `x`, `y`
    #[must_use]
    pub fn ternary_op(x: &Tensor, y: &Tensor, z: &Tensor, op: TernaryOp) -> Tensor {
        match op {
            TernaryOp::MulAcc => {
                // Mixed operands are cast to one dtype first (`DTypePolicy`), as addcmul does.
                let (promoted, operands) =
                    promote_operands(|| format!("{op:?}"), &[x, y, z], false);
                let [x, y, z] = operands.as_slice() else {
                    unreachable!("three operands in, three out")
                };
                let dtype = compute_dtype(|| format!("{op:?}"), &[promoted]);
                let (data_x, shape_x, dev_x) = {
                    let s = x.borrow();
                    (s.data.clone(), s.shape.clone(), s.device.clone())
                };
                let (data_y, shape_y, dev_y) = {
                    let s = y.borrow();
                    (s.data.clone(), s.shape.clone(), s.device.clone())
                };
                let (data_z, shape_z, dev_z) = {
                    let s = z.borrow();
                    (s.data.clone(), s.shape.clone(), s.device.clone())
                };

                assert_eq!(
                    shape_x, shape_y,
                    "MulAcc requires matching shapes for x and y"
                );
                assert_eq!(
                    shape_x, shape_z,
                    "MulAcc requires matching shapes for x and z"
                );

                // Fast path: when all three inputs live on the same GPU device and
                // shapes match exactly, evaluate z = x * y + w entirely on GPU.
                // A `None` from either kernel is a failure, not a reason to use the CPU.
                if dev_x.is_gpu() && dev_x == dev_y && dev_x == dev_z {
                    let sum = Self::gpu_mul(&data_x, &data_y)
                        .and_then(|prod| Self::gpu_add(&prod, &data_z))
                        .unwrap_or_else(|| kernel_failed("MulAcc"));
                    return edge(
                        [x.clone(), y.clone(), z.clone()],
                        sum,
                        shape_x,
                        dev_x,
                        GradFn::MulAcc(MulAccGradFn),
                    );
                }

                // CPU path (CPU inputs; mixed devices panic in `edge()`).
                let result = dispatch_real!(dtype, T => Storage::from_compute(dtype,
                    data_x
                        .host::<T>()
                        .iter()
                        .zip(data_y.host::<T>().iter())
                        .zip(data_z.host::<T>().iter())
                        .map(|((&a, &b), &c)| a * b + c)
                        .collect::<Vec<T>>(),
                ));
                edge(
                    [x.clone(), y.clone(), z.clone()],
                    result,
                    shape_x,
                    Device::CPU,
                    GradFn::MulAcc(MulAccGradFn),
                )
            }
            TernaryOp::Where => {
                // x = condition (Bool), y = true branch, z = false branch. The branches share
                // a dtype, any dtype: selecting copies bits (`real::Bits`). The condition is
                // not a graph input, so `edge()` can't check its device: do it here.
                assert!(
                    x.dtype() == DType::Bool,
                    "Where: the condition must be Bool, got {}; compare (`lt`, `elem_eq`, ..) \
                     or cast with `to_dtype(DType::Bool)`",
                    x.dtype()
                );
                assert!(
                    ![x, y, z].iter().any(|t| t.device().is_gpu()),
                    "Where: its Bool condition cannot live on the GPU until ADDENDUM §11 P8; \
                     select on the CPU"
                );
                // The branches are cast to one dtype (`DTypePolicy`); the condition stays Bool.
                let (dtype, branches) = promote_operands(|| format!("{op:?}"), &[y, z], false);
                let [y, z] = branches.as_slice() else {
                    unreachable!("two branches in, two out")
                };
                let (cond_data, cond_shape) = {
                    let s = x.borrow();
                    (s.data.clone(), s.shape.clone())
                };
                let (true_data, true_shape) = {
                    let s = y.borrow();
                    (s.data.clone(), s.shape.clone())
                };
                let (false_data, false_shape) = {
                    let s = z.borrow();
                    (s.data.clone(), s.shape.clone())
                };

                let tmp_shape =
                    Self::broadcast_shape(cond_shape.truncate_rank(), true_shape.truncate_rank());
                let out_shape = Self::broadcast_shape(&tmp_shape, false_shape.truncate_rank());

                let mask = Self::broadcast_to(
                    cond_data
                        .as_host::<bool>()
                        .expect("Where: Bool condition checked"),
                    cond_shape.truncate_rank(),
                    &out_shape,
                );
                let result = dispatch_bits!(dtype, B => {
                    let true_bc =
                        Self::broadcast_to(true_data.bits::<B>(), true_shape.truncate_rank(), &out_shape);
                    let false_bc =
                        Self::broadcast_to(false_data.bits::<B>(), false_shape.truncate_rank(), &out_shape);
                    let picked: Vec<B> = mask
                        .iter()
                        .zip(&true_bc)
                        .zip(&false_bc)
                        .map(|((&m, &t), &f)| if m { t } else { f })
                        .collect();
                    Storage::from_bits(dtype, &picked)
                });

                // Note: the condition (x) is never a parent — gradients only flow
                // through whichever branch was selected, never through the mask.
                edge(
                    [y.clone(), z.clone()],
                    result,
                    Shape::derived(out_shape),
                    Device::CPU,
                    GradFn::Where(WhereGradFn {
                        mask,
                        true_shape,
                        false_shape,
                    }),
                )
            }
        }
    }
}
