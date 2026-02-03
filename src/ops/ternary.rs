use crate::autograd::GradFn;
use crate::{RawTensor, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Ternary operations: three inputs, one output
#[derive(Clone, Copy)]
pub enum TernaryOp {
    MulAcc, // x*y + z (fused multiply-accumulate)
    Where,  // condition ? x : y (masked selection)
}

/// Gradient function for `MulAcc` (fused multiply-add)
///
/// z = x*y + w has gradients:
/// - ∂z/∂x = y
/// - ∂z/∂y = x
/// - ∂z/∂w = 1
pub struct MulAccGradFn;
impl GradFn for MulAccGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // SAFETY: MulAcc operation always creates 2 or 3 parents (x, y, optional w)
        debug_assert!(
            parents.len() >= 2,
            "MulAcc grad requires at least 2 parents"
        );
        let x_ref = parents.first().cloned().unwrap();
        let y_ref = parents.get(1).cloned().unwrap();
        let x_val = x_ref.borrow();
        let y_val = y_ref.borrow();

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

        let grad_z = if parents.get(2).is_some_and(|p| p.borrow().requires_grad) {
            Some(RawTensor::new(
                out_grad.data.to_vec(),
                &out_grad.shape,
                false,
            ))
        } else {
            None
        };

        vec![grad_x, grad_y, grad_z]
    }
    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(Self)
    }
}

/// Gradient function for Where (conditional selection)
///
/// Gradients flow through the branch that was selected, and must be reduced
/// along any dimensions that were broadcast.
pub struct WhereGradFn {
    condition: Vec<f32>,     // broadcasted condition (matches out_grad shape)
    true_shape: Vec<usize>,  // original shape of the "true" branch
    false_shape: Vec<usize>, // original shape of the "false" branch
}

impl GradFn for WhereGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
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

        #[allow(
            clippy::useless_let_if_seq,
            reason = "will handle idiomatically after i finish the rest of these"
        )]
        let mut grad_true = None;
        if needs_true {
            let mut data = vec![0.0; out_grad.data.len()];
            for (i, (&g, &c)) in out_grad.data.iter().zip(&self.condition).enumerate() {
                if c != 0.0
                    && let Some(slot) = data.get_mut(i)
                {
                    *slot = g;
                }
            }
            let reduced =
                RawTensor::sum_over_broadcast_dims(&data, &out_grad.shape, &self.true_shape);
            grad_true = Some(RawTensor::new(reduced, &self.true_shape, false));
        }

        #[allow(
            clippy::useless_let_if_seq,
            reason = "will handle idiomatically after i finish the rest of these"
        )]
        let mut grad_false = None;
        if needs_false {
            let mut data = vec![0.0; out_grad.data.len()];
            for (i, (&g, &c)) in out_grad.data.iter().zip(&self.condition).enumerate() {
                if c == 0.0
                    && let Some(slot) = data.get_mut(i)
                {
                    *slot = g;
                }
            }
            let reduced =
                RawTensor::sum_over_broadcast_dims(&data, &out_grad.shape, &self.false_shape);
            grad_false = Some(RawTensor::new(reduced, &self.false_shape, false));
        }

        vec![grad_true, grad_false]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(Self {
            condition: self.condition.clone(),
            true_shape: self.true_shape.clone(),
            false_shape: self.false_shape.clone(),
        })
    }
}

// ===== TERNARY OPERATIONS =====

impl RawTensor {
    /// Apply ternary operations (3 inputs, 1 output)
    /// # Panics
    /// assert shape match for `x`, `y`
    pub fn ternary_op(x: &Tensor, y: &Tensor, z: &Tensor, op: TernaryOp) -> Tensor {
        match op {
            TernaryOp::MulAcc => {
                let (data_x, shape_x, req_x, dev_x) = {
                    let s = x.borrow();
                    (
                        s.data.clone(),
                        s.shape.clone(),
                        s.requires_grad,
                        s.device.clone(),
                    )
                };
                let (data_y, shape_y, req_y, dev_y) = {
                    let s = y.borrow();
                    (
                        s.data.clone(),
                        s.shape.clone(),
                        s.requires_grad,
                        s.device.clone(),
                    )
                };
                let (data_z, shape_z, req_z, dev_z) = {
                    let s = z.borrow();
                    (
                        s.data.clone(),
                        s.shape.clone(),
                        s.requires_grad,
                        s.device.clone(),
                    )
                };

                assert_eq!(
                    shape_x, shape_y,
                    "MulAcc requires matching shapes for x and y"
                );
                assert_eq!(
                    shape_x, shape_z,
                    "MulAcc requires matching shapes for x and z"
                );

                let requires_grad = req_x || req_y || req_z;

                // Fast path: when all three inputs live on the same GPU device and
                // shapes match exactly, evaluate z = x * y + w entirely on GPU.
                #[cfg(feature = "gpu")]
                {
                    if dev_x.is_gpu() && dev_x == dev_y && dev_x == dev_z {
                        if let Some(prod) = Self::gpu_mul(&data_x, &data_y) {
                            if let Some(sum) = Self::gpu_add(&prod, &data_z) {
                                let out = Rc::new(RefCell::new(Self {
                                    data: sum,
                                    shape: shape_x,
                                    grad: None,
                                    requires_grad,
                                    grad_fn: None,
                                    parents: vec![x.clone(), y.clone(), z.clone()],
                                    device: dev_x,
                                }));
                                if requires_grad {
                                    out.borrow_mut().grad_fn = Some(Box::new(MulAccGradFn));
                                }
                                return out;
                            }
                            eprintln!(
                                "Warning: GPU add for MulAcc failed; falling back to CPU path"
                            );
                        } else {
                            eprintln!(
                                "Warning: GPU mul for MulAcc failed; falling back to CPU path"
                            );
                        }
                    }
                }

                // CPU fallback.
                let result_data = data_x
                    .iter()
                    .zip(&data_y)
                    .zip(&data_z)
                    .map(|((a, b), c)| a * b + c)
                    .collect::<Vec<_>>();
                let out = Self::new(result_data, &shape_x, requires_grad);
                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![x.clone(), y.clone(), z.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MulAccGradFn));
                }
                out
            }
            TernaryOp::Where => {
                // x = condition, y = true branch, z = false branch
                let (cond_data, cond_shape) = {
                    let s = x.borrow();
                    (s.data.clone(), s.shape.clone())
                };
                let (true_data, true_shape, true_req) = {
                    let s = y.borrow();
                    (s.data.clone(), s.shape.clone(), s.requires_grad)
                };
                let (false_data, false_shape, false_req) = {
                    let s = z.borrow();
                    (s.data.clone(), s.shape.clone(), s.requires_grad)
                };

                let tmp_shape = Self::broadcast_shape(&cond_shape, &true_shape);
                let out_shape = Self::broadcast_shape(&tmp_shape, &false_shape);

                let cond_bc = Self::broadcast_to(&cond_data, &cond_shape, &out_shape);
                let true_bc = Self::broadcast_to(&true_data, &true_shape, &out_shape);
                let false_bc = Self::broadcast_to(&false_data, &false_shape, &out_shape);

                let result_data = cond_bc
                    .iter()
                    .zip(&true_bc)
                    .zip(&false_bc)
                    .map(|((&c, &t), &f)| if c == 0.0 { f } else { t })
                    .collect::<Vec<_>>();

                let out = Self::new(result_data, &out_shape, true_req || false_req);
                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![y.clone(), z.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(WhereGradFn {
                        condition: cond_bc,
                        true_shape,
                        false_shape,
                    }));
                }
                out
            }
        }
    }

    pub fn mulacc(x: &Tensor, y: &Tensor, z: &Tensor) -> Tensor {
        Self::ternary_op(x, y, z, TernaryOp::MulAcc)
    }
    pub fn where_op(cond: &Tensor, x: &Tensor, y: &Tensor) -> Tensor {
        Self::ternary_op(cond, x, y, TernaryOp::Where)
    }
}
