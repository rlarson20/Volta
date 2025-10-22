use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub type Tensor = Rc<RefCell<RawTensor>>;
//using Rc<RefCell<Tensor>> is simple approach for dyn graph
//for thread-safety, use Arc<Mutex<Tensor>>
//starting single-threaded

// ===== OPS =====

#[derive(Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Recip,
    Sqrt,
    Exp2,
    Log2,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    ReLU,
}

#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Mod,
    Cmplt,
}

#[derive(Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
    Mean,
}

#[derive(Clone, Copy)]
pub enum TernaryOp {
    MulAcc,
    Where,
}

#[derive(Clone)]
pub enum MovementOp {
    Reshape { new_shape: Vec<usize> },
    Permute { axes: Vec<usize> },
    Expand { new_shape: Vec<usize> },
    Pad { padding: Vec<(usize, usize)> },
    Shrink { ranges: Vec<(usize, usize)> },
    Stride { strides: Vec<usize> },
}

pub enum LoadOp {
    Empty,
    Rand,
    Const,
    From,
    Contiguous,
    Custom,
}

// ===== GRADFN =====

pub trait GradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>>;
    fn clone_box(&self) -> Box<dyn GradFn>;
}

// ===== GRADFN IMPLEMENTATIONS =====

struct UnaryGradFn {
    op: UnaryOp,
}

impl GradFn for UnaryGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x = parents[0].borrow();
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

struct BinaryGradFn {
    op: BinaryOp,
}

impl GradFn for BinaryGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x_val = parents[0].borrow();
        let y_val = parents[1].borrow();

        let (grad_x, grad_y) = match self.op {
            BinaryOp::Add => {
                let gx = if x_val.requires_grad {
                    // Sum gradient over broadcast dimensions
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

struct MaxReduceGradFn {
    input_shape: Vec<usize>,
    max_index: usize,
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

// Unified movement op gradient
#[derive(Clone)]
struct MovementGradFn {
    op: MovementOp,
    original_shape: Vec<usize>,
}

impl GradFn for MovementGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_tensor = match &self.op {
            MovementOp::Reshape { .. } => {
                RawTensor::new(out_grad.data.clone(), &self.original_shape, false)
            }
            MovementOp::Permute { axes } => {
                // Compute inverse permutation
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
                // Sum over expanded dimensions
                let mut grad_data = vec![0.0; self.original_shape.iter().product()];
                let old_strides = RawTensor::compute_strides(&self.original_shape);
                let new_strides = RawTensor::compute_strides(new_shape);

                for i in 0..out_grad.data.len() {
                    let mut old_idx = 0;
                    let mut rem = i;
                    for j in (0..new_shape.len()).rev() {
                        let coord = rem / new_strides[j];
                        rem %= new_strides[j];
                        if self.original_shape[j] != 1 {
                            old_idx += coord * old_strides[j];
                        }
                    }
                    grad_data[old_idx] += out_grad.data[i];
                }
                RawTensor::new(grad_data, &self.original_shape, false)
            }
            MovementOp::Pad { padding } => {
                // Strip padding from gradient
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
                // Pad gradient back to original size
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
                        let old_i = i + ranges[dim].0;
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
                // Upsample gradient
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

#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    GPU,
    //TODO: possible Metal variant
}

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

//CONSTRUCTORS
impl RawTensor {
    //from data and shape
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
    pub fn zeros(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }
    pub fn ones(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape, false)
    }
    pub fn rand(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        Self::new(data, shape, false)
    }

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

//UnaryOps
impl RawTensor {
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
                UnaryOp::Log2 => x.log2(),
                UnaryOp::Sin => x.sin(),
                UnaryOp::Cos => x.cos(),
                UnaryOp::Tanh => x.tanh(),
                UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                UnaryOp::ReLU => x.max(0.0),
            })
            .collect();
        let out = Self::new(result, &shape, req);
        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(UnaryGradFn { op }));
        }
        out
    }
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

//BinaryOps
impl RawTensor {
    // Helper: compute broadcast shape following numpy rules
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

    // FIXME!!!
    // Helper: broadcast data to target shape
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

        for i in 0..to_size {
            let mut from_idx = 0;
            let mut remainder = i;
            //calc coords based on to_shape and map
            for (dim, &_dim_size) in to_shape.iter().enumerate() {
                let stride = to_shape[dim + 1..].iter().product::<usize>();
                let coord = remainder / stride;
                remainder %= stride;
                //if dim broadcast (size was 1) use coord 0 for from_idx
                if padded_from[dim] != 1 {
                    from_idx += coord * from_strides_padded[dim];
                }
            }
            result[i] = data[from_idx];
        }
        result
    }

    // Helper: sum gradient over broadcast dimensions
    fn sum_over_broadcast_dims(
        grad: &[f32],
        grad_shape: &[usize],
        target_shape: &[usize],
    ) -> Vec<f32> {
        if grad_shape == target_shape {
            return grad.to_vec();
        }

        // Pad target_shape with leading 1s
        let mut padded_target = vec![1; grad_shape.len()];
        let offset = grad_shape.len() - target_shape.len();
        padded_target[offset..].copy_from_slice(target_shape);

        // Find dimensions that were broadcast (where target was 1)
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
        for (i, &grad_val) in grad.iter().enumerate() {
            let mut target_idx = 0;
            let mut remainder = i;

            //convert lin index i in grad_shape to coord
            for (dim, &_dim_size) in grad_shape.iter().enumerate() {
                let stride = grad_shape[dim + 1..].iter().product::<usize>();
                let coord = remainder / stride;
                remainder %= stride;

                //if wasnt bc, add to index
                let target_dim_idx = dim as i32 - offset as i32;
                if target_dim_idx >= 0 {
                    let target_dim_idx = target_dim_idx as usize;
                    if padded_target[dim] != 1 {
                        target_idx += coord * target_strides[target_dim_idx];
                    }
                }
            }
            if target_idx < result.len() {
                result[target_idx] += grad_val;
            }
        }
        result
    }

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

// ===== REDUCE OPS =====

impl RawTensor {
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

//TernaryOps
impl RawTensor {
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

// ===== MOVEMENT OPS =====
impl RawTensor {
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

        fn index_to_coords(idx: usize, shape: &[usize]) -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            let mut remaining = idx;
            for i in (0..shape.len()).rev() {
                coords[i] = remaining % shape[i];
                remaining /= shape[i];
            }
            coords
        }

        fn coords_to_index(coords: &[usize], strides: &[usize]) -> usize {
            coords.iter().zip(strides).map(|(c, s)| c * s).sum()
        }

        for (new_idx, val) in new_data.iter_mut().enumerate() {
            let new_coords = index_to_coords(new_idx, &new_shape);
            let mut old_coords = vec![0; axes.len()];
            for (i, &ax) in axes.iter().enumerate() {
                old_coords[ax] = new_coords[i];
            }
            let old_idx = coords_to_index(&old_coords, &old_strides);
            *val = data[old_idx];
        }
        Self::new(new_data, &new_shape, false)
    }

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

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

// ===== LOAD OPS =====
//other methods to add:
//to_device LoadOp?

impl RawTensor {
    pub fn empty(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }

    pub fn constant(value: f32, shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![value; size], shape, false)
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Self::new(data, shape, false)
    }

    pub fn contiguous(self_t: &Tensor) -> Tensor {
        // For now, all tensors are contiguous
        // Later: handle views/strides
        let s = self_t.borrow();
        Self::new(s.data.clone(), &s.shape, s.requires_grad)
    }
}

// ===== MATMUL =====

impl RawTensor {
    // Helper: transpose 2D matrix stored as flat vec
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
    // a: (m, k), b: (k, n) -> result: (m, n)
    fn matmul_raw(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

// ===== BACKWARD =====

impl RawTensor {
    pub fn backward(tensor_ref: &Tensor) {
        let tensor = tensor_ref.borrow();
        assert!(
            tensor.requires_grad,
            "Called backward on a tensor that doesn't require grad"
        );
        drop(tensor);

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

        let mut stack = vec![tensor_ref.clone()];
        let mut visited = HashSet::new();

        while let Some(tensor) = stack.pop() {
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
                let parent_grads = grad_fn.backward(&grad_out, &parents);

                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
                    if let Some(g) = parent_grad {
                        let mut parent = parent_ref.borrow_mut();
                        let g_data = g.borrow().data.clone();
                        if parent.grad.is_none() {
                            parent.grad = Some(g_data)
                        } else {
                            let existing = parent.grad.as_mut().unwrap();
                            for (accum, &new) in existing.iter_mut().zip(g_data.iter()) {
                                *accum += new;
                            }
                        }
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
    /// For a scalar function f(x), the gradient is approximated as:
    /// ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / (2ε)
    ///
    /// This is more accurate than forward difference: (f(x+ε) - f(x)) / ε
    ///
    /// Returns: (max_error, mean_error, passed)
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

    /// Simplified gradient checker with default params
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

// ===== TRAIT API =====

pub trait TensorOps {
    fn add(&self, other: &Tensor) -> Tensor;
    fn sub(&self, other: &Tensor) -> Tensor;
    fn elem_mul(&self, other: &Tensor) -> Tensor;
    fn div(&self, other: &Tensor) -> Tensor;
    fn max_elem(&self, other: &Tensor) -> Tensor;
    fn modulo(&self, other: &Tensor) -> Tensor;
    fn cmplt(&self, other: &Tensor) -> Tensor;

    fn neg(&self) -> Tensor;
    fn recip(&self) -> Tensor;
    fn sqrt(&self) -> Tensor;
    fn exp2(&self) -> Tensor;
    fn log2(&self) -> Tensor;
    fn sin(&self) -> Tensor;
    fn cos(&self) -> Tensor;
    fn tanh(&self) -> Tensor;
    fn sigmoid(&self) -> Tensor;
    fn relu(&self) -> Tensor;

    fn sum(&self) -> Tensor;
    fn max_reduce(&self) -> Tensor;
    fn mean(&self) -> Tensor;

    fn mulacc(&self, y: &Tensor, z: &Tensor) -> Tensor;
    fn where_op(&self, x: &Tensor, y: &Tensor) -> Tensor;

    fn reshape(&self, new_shape: &[usize]) -> Tensor;
    fn permute(&self, axes: &[usize]) -> Tensor;
    fn expand(&self, new_shape: &[usize]) -> Tensor;
    fn pad(&self, padding: &[(usize, usize)]) -> Tensor;
    fn shrink(&self, ranges: &[(usize, usize)]) -> Tensor;
    fn stride_op(&self, strides: &[usize]) -> Tensor;

    fn matmul(&self, other: &Tensor) -> Tensor;
    fn transpose(&self) -> Tensor;

    fn backward(&self);
    fn grad(&self) -> Option<Vec<f32>>;
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
}

// ===== Optimizers =====

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    pub fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            if let Some(grad) = &p.grad.clone() {
                if self.momentum > 0.0 {
                    for (v, &g) in self.velocity[i].iter_mut().zip(grad.iter()) {
                        *v = self.momentum * *v - self.lr * g;
                    }
                    for (d, &v) in p.data.iter_mut().zip(&self.velocity[i]) {
                        *d += v;
                    }
                } else {
                    for (d, &g) in p.data.iter_mut().zip(grad.iter()) {
                        *d -= self.lr * g;
                    }
                }
            }
        }
    }
}

impl SGD {
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

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }
}

// ===== NN layers =====

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let w = RawTensor::randn(&[in_features, out_features]);
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

// ===== TESTS =====

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

    #[test] //failing rn, adds 10 to the second half when should be adding 20
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
}
