use crate::autograd::GradFn;
use crate::{RawTensor, Tensor};

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

/// Gradient function for Sum reduction
///
/// Sum reduction collapses to scalar, so gradient broadcasts back to original shape.
pub struct SumGradFn {
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
pub struct MaxReduceGradFn {
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
/// Each element gets gradient / `num_elements`.
pub struct MeanGradFn {
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
