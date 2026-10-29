use crate::autograd::{GradFn, edge};
use crate::index::{fetch, slot};
use crate::shape::Shape;
use crate::storage::Storage;
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
#[derive(Clone)]
pub(crate) struct SumGradFn {
    input_shape: Shape,
}

impl SumGradFn {
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        let grad_val: f32 = fetch(&out_grad.data.to_host_f32(), 0);

        // Check if we can do GPU backward
        #[cfg(feature = "gpu")]
        {
            if out_grad.device.is_gpu()
                && let Some(storage) = crate::RawTensor::gpu_sum_backward(grad_val, size)
            {
                return vec![Some(RawTensor::new_with_storage(
                    storage,
                    self.input_shape.clone(),
                    out_grad.device.clone(),
                    false,
                ))];
            }
        }

        // CPU fallback
        vec![Some(RawTensor::new(
            vec![grad_val; size],
            self.input_shape.truncate_rank(),
            false,
        ))]
    }
}

/// Gradient function for Max reduction
///
/// Only the maximum element receives gradient; all others get zero.
#[derive(Clone)]
pub(crate) struct MaxReduceGradFn {
    input_shape: Shape,
    max_index: usize, // Linear index of the maximum element
}

impl MaxReduceGradFn {
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        let grad_val: f32 = fetch(&out_grad.data.to_host_f32(), 0);

        // Check if we can do GPU backward
        #[cfg(feature = "gpu")]
        {
            if out_grad.device.is_gpu()
                && let Some(storage) =
                    crate::RawTensor::gpu_max_backward(grad_val, size, self.max_index)
            {
                return vec![Some(RawTensor::new_with_storage(
                    storage,
                    self.input_shape.clone(),
                    out_grad.device.clone(),
                    false,
                ))];
            }
        }

        // CPU fallback
        let mut grad_data = vec![0.0; size];
        *slot(&mut grad_data, self.max_index) = grad_val;
        vec![Some(RawTensor::from_shape(
            grad_data,
            self.input_shape.clone(),
            false,
        ))]
    }
}

/// Gradient function for Mean reduction
///
/// Each element gets gradient / `num_elements`.
#[derive(Clone)]
pub(crate) struct MeanGradFn {
    input_shape: Shape,
}

impl MeanGradFn {
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        let grad_val: f32 = fetch(&out_grad.data.to_host_f32(), 0);

        // Check if we can do GPU backward
        #[cfg(feature = "gpu")]
        {
            if out_grad.device.is_gpu()
                && let Some(storage) = crate::RawTensor::gpu_mean_backward(grad_val, size)
            {
                return vec![Some(RawTensor::new_with_storage(
                    storage,
                    self.input_shape.clone(),
                    out_grad.device.clone(),
                    false,
                ))];
            }
        }

        // CPU fallback
        let grad_val_cpu = grad_val / (size as f32);
        vec![Some(RawTensor::new(
            vec![grad_val_cpu; size],
            self.input_shape.truncate_rank(),
            false,
        ))]
    }
}

// ===== REDUCE OPERATIONS =====

impl RawTensor {
    /// Apply a reduction operation that collapses tensor to scalar
    ///
    /// All reduction ops produce a shape \[1\] output.
    /// # Panics
    /// unwrap map
    #[must_use]
    pub fn reduce_op(self_t: &Tensor, op: ReduceOp) -> Tensor {
        let (data, shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        let (result_val, grad_fn): (f32, GradFn) = match op {
            ReduceOp::Sum => {
                // Try GPU first if available
                let sum: f32 = if device.is_gpu() {
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(result) = Self::gpu_sum_reduce(&data) {
                            result
                        } else {
                            data.to_host_f32().iter().sum()
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        data.to_host_f32().iter().sum()
                    }
                } else {
                    data.to_host_f32().iter().sum()
                };
                (sum, GradFn::Sum(SumGradFn { input_shape: shape }))
            }
            ReduceOp::Max => {
                // Try GPU first if available
                let (max_val, max_idx) = if device.is_gpu() {
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(result) = Self::gpu_max_reduce(&data) {
                            result
                        } else {
                            let (max_val, max_idx) = data
                                .to_host_f32()
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
                            .to_host_f32()
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, &val)| (val, idx))
                            .unwrap();
                        (max_val, max_idx)
                    }
                } else {
                    let (max_val, max_idx) = data
                        .to_host_f32()
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, &val)| (val, idx))
                        .unwrap();
                    (max_val, max_idx)
                };
                (
                    max_val,
                    GradFn::MaxReduce(MaxReduceGradFn {
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
                        if let Some(result) = Self::gpu_mean_reduce(&data) {
                            result
                        } else {
                            let sum: f32 = data.to_host_f32().iter().sum();
                            sum / (data.len() as f32)
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        let sum: f32 = data.to_host_f32().iter().sum();
                        sum / (data.len() as f32)
                    }
                } else {
                    let sum: f32 = data.to_host_f32().iter().sum();
                    sum / (data.len() as f32)
                };
                (mean_val, GradFn::Mean(MeanGradFn { input_shape: shape }))
            }
        };

        // Start with a CPU scalar and then place it on the same logical device
        // as the input tensor. The reduction computation may have happened on GPU.
        let storage = Storage::cpu(vec![result_val]).to_device(&device);

        edge(
            [self_t.clone()],
            storage,
            Shape::derived(vec![1]),
            device,
            grad_fn,
        )
    }
}
