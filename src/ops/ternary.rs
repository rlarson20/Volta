use crate::autograd::GradFn;
use crate::{RawTensor, Tensor};

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
/// Gradients flow through the branch that was selected, and must be reduced
/// along any dimensions that were broadcast.
pub struct WhereGradFn {
    condition: Vec<f32>,     // broadcasted condition (matches out_grad shape)
    true_shape: Vec<usize>,  // original shape of the "true" branch
    false_shape: Vec<usize>, // original shape of the "false" branch
}

impl GradFn for WhereGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let true_parent = parents[0].borrow();
        let false_parent = parents[1].borrow();
        let needs_true = true_parent.requires_grad;
        let needs_false = false_parent.requires_grad;
        drop(true_parent);
        drop(false_parent);

        let mut grad_true = None;
        if needs_true {
            let mut data = vec![0.0; out_grad.data.len()];
            for (i, (&g, &c)) in out_grad.data.iter().zip(&self.condition).enumerate() {
                if c != 0.0 {
                    data[i] = g;
                }
            }
            let reduced =
                RawTensor::sum_over_broadcast_dims(&data, &out_grad.shape, &self.true_shape);
            grad_true = Some(RawTensor::new(reduced, &self.true_shape, false));
        }

        let mut grad_false = None;
        if needs_false {
            let mut data = vec![0.0; out_grad.data.len()];
            for (i, (&g, &c)) in out_grad.data.iter().zip(&self.condition).enumerate() {
                if c == 0.0 {
                    data[i] = g;
                }
            }
            let reduced =
                RawTensor::sum_over_broadcast_dims(&data, &out_grad.shape, &self.false_shape);
            grad_false = Some(RawTensor::new(reduced, &self.false_shape, false));
        }

        vec![grad_true, grad_false]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(WhereGradFn {
            condition: self.condition.clone(),
            true_shape: self.true_shape.clone(),
            false_shape: self.false_shape.clone(),
        })
    }
}

// ===== TERNARY OPERATIONS =====

impl RawTensor {
    /// Apply ternary operations (3 inputs, 1 output)
    pub fn ternary_op(x: &Tensor, y: &Tensor, z: &Tensor, op: TernaryOp) -> Tensor {
        match op {
            TernaryOp::MulAcc => {
                let (data_x, shape_x, req_x) = {
                    let s = x.borrow();
                    (s.data.clone(), s.shape.clone(), s.requires_grad)
                };
                let (data_y, _, req_y) = {
                    let s = y.borrow();
                    (s.data.clone(), s.shape.clone(), s.requires_grad)
                };
                let (data_z, _, req_z) = {
                    let s = z.borrow();
                    (s.data.clone(), s.shape.clone(), s.requires_grad)
                };
                assert_eq!(shape_x, y.borrow().shape, "MulAcc requires matching shapes");
                assert_eq!(shape_x, z.borrow().shape, "MulAcc requires matching shapes");

                let result_data = data_x
                    .iter()
                    .zip(&data_y)
                    .zip(&data_z)
                    .map(|((a, b), c)| a * b + c)
                    .collect::<Vec<_>>();

                let out = Self::new(result_data, &shape_x, req_x || req_y || req_z);
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
                    .map(|((&c, &t), &f)| if c != 0.0 { t } else { f })
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
