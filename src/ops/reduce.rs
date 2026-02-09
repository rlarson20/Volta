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
        let grad_val: f32 = out_grad.data.first().copied().unwrap_or(0.0);

        // Check if we can do GPU backward
        #[cfg(feature = "gpu")]
        {
            if out_grad.device.is_gpu()
                && let Some(storage) = crate::RawTensor::gpu_sum_backward(grad_val, size)
            {
                return vec![Some(RawTensor::new_with_storage(
                    storage,
                    &self.input_shape,
                    out_grad.device.clone(),
                    false,
                ))];
            }
        }

        // CPU fallback
        vec![Some(RawTensor::new(
            vec![grad_val; size],
            &self.input_shape,
            false,
        ))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(Self {
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
        let grad_val: f32 = out_grad.data.first().copied().unwrap_or(0.0);

        // Check if we can do GPU backward
        #[cfg(feature = "gpu")]
        {
            if out_grad.device.is_gpu()
                && let Some(storage) =
                    crate::RawTensor::gpu_max_backward(grad_val, size, self.max_index)
            {
                return vec![Some(RawTensor::new_with_storage(
                    storage,
                    &self.input_shape,
                    out_grad.device.clone(),
                    false,
                ))];
            }
        }

        // CPU fallback
        let mut grad_data = vec![0.0; size];
        if let Some(slot) = grad_data.get_mut(self.max_index) {
            *slot = grad_val;
        }
        vec![Some(RawTensor::new(grad_data, &self.input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(Self {
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
        let grad_val: f32 = out_grad.data.first().copied().unwrap_or(0.0);

        // Check if we can do GPU backward
        #[cfg(feature = "gpu")]
        {
            if out_grad.device.is_gpu()
                && let Some(storage) = crate::RawTensor::gpu_mean_backward(grad_val, size)
            {
                return vec![Some(RawTensor::new_with_storage(
                    storage,
                    &self.input_shape,
                    out_grad.device.clone(),
                    false,
                ))];
            }
        }

        // CPU fallback
        let grad_val_cpu = grad_val / (size as f32);
        vec![Some(RawTensor::new(
            vec![grad_val_cpu; size],
            &self.input_shape,
            false,
        ))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(Self {
            input_shape: self.input_shape.clone(),
        })
    }
}

// ===== REDUCE OPERATIONS =====

impl RawTensor {
    /// Apply a reduction operation that collapses tensor to scalar
    ///
    /// All reduction ops produce a shape \[1\] output.
    /// # Panics
    /// unwrap map
    pub fn reduce_op(self_t: &Tensor, op: ReduceOp) -> Tensor {
        let (data, shape, req_grad, device) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        let (result_val, grad_fn): (f32, Box<dyn GradFn>) = match op {
            ReduceOp::Sum => {
                // Try GPU first if available
                let sum: f32 = if device.is_gpu() {
                    #[cfg(feature = "gpu")]
                    #[allow(clippy::option_if_let_else, reason = "can't figure out right now")]
                    {
                        if let Some(result) = Self::gpu_sum_reduce(&data) {
                            result
                        } else {
                            data.iter().sum()
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        data.iter().sum()
                    }
                } else {
                    data.iter().sum()
                };
                (sum, Box::new(SumGradFn { input_shape: shape }))
            }
            ReduceOp::Max => {
                // Try GPU first if available
                let (max_val, max_idx) = if device.is_gpu() {
                    #[cfg(feature = "gpu")]
                    {
                        #[allow(clippy::option_if_let_else, reason = "can't figure out rn")]
                        if let Some(result) = Self::gpu_max_reduce(&data) {
                            result
                        } else {
                            let (max_val, max_idx) = data
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(idx, &val)| (val, idx))
                                .unwrap();
                            (max_val, max_idx)
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        let (max_val, max_idx) = data
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, &val)| (val, idx))
                            .unwrap();
                        (max_val, max_idx)
                    }
                } else {
                    let (max_val, max_idx) = data
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, &val)| (val, idx))
                        .unwrap();
                    (max_val, max_idx)
                };
                (
                    max_val,
                    Box::new(MaxReduceGradFn {
                        input_shape: shape,
                        max_index: max_idx,
                    }),
                )
            }
            ReduceOp::Mean => {
                // Try GPU first if available
                let mean_val: f32 = if device.is_gpu() {
                    #[cfg(feature = "gpu")]
                    {
                        #[allow(clippy::option_if_let_else, reason = "can't figure out rn")]
                        if let Some(result) = Self::gpu_mean_reduce(&data) {
                            result
                        } else {
                            let sum: f32 = data.iter().sum();
                            sum / (data.len() as f32)
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        let sum: f32 = data.iter().sum();
                        sum / (data.len() as f32)
                    }
                } else {
                    let sum: f32 = data.iter().sum();
                    sum / (data.len() as f32)
                };
                (mean_val, Box::new(MeanGradFn { input_shape: shape }))
            }
        };

        // Start with a CPU scalar and then place it on the same logical device
        // as the input tensor. The reduction computation may have happened on GPU.
        let out = Self::new(vec![result_val], &[1], req_grad);
        {
            let mut ob = out.borrow_mut();
            ob.data = ob.data.to_device(&device);
            ob.device = device;
        }

        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(grad_fn);
        }
        out
    }
}
