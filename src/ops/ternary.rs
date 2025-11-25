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
/// Gradient flows through the branch that was selected.
/// The condition tensor itself is not differentiable.
pub struct WhereGradFn {
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
