use crate::autograd::GradFn;
use crate::{RawTensor, Tensor};

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

/// Gradient function for unary operations
///
/// Stores which operation was performed so backward can apply the correct derivative.
pub struct UnaryGradFn {
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
