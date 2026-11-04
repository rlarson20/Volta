use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::device_policy::kernel_failed;
use crate::index::{fetch, slot};
use crate::real::{Real, compute_dtype, dispatch_real};
use crate::shape::Shape;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

/// Reduction operations: reduce tensor to scalar
///
/// These operations collapse all dimensions and require special gradient handling
/// since the output shape differs from the input.
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    /// Sum of all elements; the gradient broadcasts ones.
    Sum,
    /// Maximum element; the gradient goes only to the maximum.
    Max,
    /// Mean of all elements; the gradient broadcasts `1/n`.
    Mean,
}

/// Gradient function for Sum reduction
///
/// Sum reduction collapses to scalar, so gradient broadcasts back to original shape.
#[derive(Clone)]
pub(crate) struct SumGradFn {
    /// Shape of the reduced input.
    input_shape: Shape,
}

impl SumGradFn {
    /// Broadcasts the scalar gradient to the input's shape.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        // GPU path for a GPU gradient; a `None` from the kernel is a failure.
        if out_grad.device.is_gpu() {
            let grad_val: f32 = fetch(&out_grad.data.host::<f32>(), 0);
            let storage = crate::RawTensor::gpu_sum_backward(out_grad.data.dtype(), grad_val, size)
                .unwrap_or_else(|| kernel_failed("Sum"));
            return vec![Some(RawTensor::new_with_storage(
                storage,
                self.input_shape.clone(),
                out_grad.device.clone(),
                false,
            ))];
        }

        // CPU path, in the gradient's dtype.
        let grad = dispatch_real!(out_grad.data.dtype(), T => {
            Storage::from_compute(out_grad.data.dtype(), vec![fetch(&out_grad.data.host::<T>(), 0); size])
        });
        vec![Some(RawTensor::new_with_storage(
            grad,
            self.input_shape.clone(),
            Device::CPU,
            false,
        ))]
    }
}

/// Gradient function for Max reduction
///
/// Only the maximum element receives gradient; all others get zero.
#[derive(Clone)]
pub(crate) struct MaxReduceGradFn {
    /// Shape of the reduced input.
    input_shape: Shape,
    /// Linear index of the maximum element.
    max_index: usize,
}

impl MaxReduceGradFn {
    /// Routes the scalar gradient to `max_index`; every other element gets zero.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        // GPU path for a GPU gradient; a `None` from the kernel is a failure.
        if out_grad.device.is_gpu() {
            let grad_val: f32 = fetch(&out_grad.data.host::<f32>(), 0);
            let storage = crate::RawTensor::gpu_max_backward(
                out_grad.data.dtype(),
                grad_val,
                size,
                self.max_index,
            )
            .unwrap_or_else(|| kernel_failed("MaxReduce"));
            return vec![Some(RawTensor::new_with_storage(
                storage,
                self.input_shape.clone(),
                out_grad.device.clone(),
                false,
            ))];
        }

        // CPU path, in the gradient's dtype.
        let grad = dispatch_real!(out_grad.data.dtype(), T => {
            let mut grad_data = vec![T::ZERO; size];
            *slot(&mut grad_data, self.max_index) = fetch(&out_grad.data.host::<T>(), 0);
            Storage::from_compute(out_grad.data.dtype(), grad_data)
        });
        vec![Some(RawTensor::new_with_storage(
            grad,
            self.input_shape.clone(),
            Device::CPU,
            false,
        ))]
    }
}

/// Gradient function for Mean reduction
///
/// Each element gets gradient / `num_elements`.
#[derive(Clone)]
pub(crate) struct MeanGradFn {
    /// Shape of the reduced input.
    input_shape: Shape,
}

impl MeanGradFn {
    /// Broadcasts the scalar gradient divided by the element count.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let size: usize = self.input_shape.iter().product();
        // GPU path for a GPU gradient; a `None` from the kernel is a failure.
        if out_grad.device.is_gpu() {
            let grad_val: f32 = fetch(&out_grad.data.host::<f32>(), 0);
            let storage =
                crate::RawTensor::gpu_mean_backward(out_grad.data.dtype(), grad_val, size)
                    .unwrap_or_else(|| kernel_failed("Mean"));
            return vec![Some(RawTensor::new_with_storage(
                storage,
                self.input_shape.clone(),
                out_grad.device.clone(),
                false,
            ))];
        }

        // CPU path, in the gradient's dtype.
        let grad = dispatch_real!(out_grad.data.dtype(), T => {
            let g = fetch(&out_grad.data.host::<T>(), 0) / T::from_usize(size);
            Storage::from_compute(out_grad.data.dtype(), vec![g; size])
        });
        vec![Some(RawTensor::new_with_storage(
            grad,
            self.input_shape.clone(),
            Device::CPU,
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
        let dtype = compute_dtype(|| format!("{op:?}"), &[self_t.dtype()]);
        let (data, shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        // On the GPU, a `None` from the kernel is a failure. The kernels accumulate in f32
        // and return a host scalar, rounded here to the input's dtype (as the CPU rounds a
        // half reduction once) and put back on the input's device: not a fallback, because
        // the reduction itself ran there.
        let on_gpu = |v: f32| Storage::from_compute(dtype, vec![v]).to_device(&device);
        let (storage, grad_fn): (Storage, GradFn) = match op {
            ReduceOp::Sum => {
                let sum = if device.is_gpu() {
                    on_gpu(Self::gpu_sum_reduce(&data).unwrap_or_else(|| kernel_failed("Sum")))
                } else {
                    dispatch_real!(dtype, T => {
                        Storage::from_compute(dtype, vec![data.host::<T>().iter().copied().sum::<T>()])
                    })
                };
                (sum, GradFn::Sum(SumGradFn { input_shape: shape }))
            }
            ReduceOp::Max => {
                let (max, max_idx) = if device.is_gpu() {
                    let (v, i) =
                        Self::gpu_max_reduce(&data).unwrap_or_else(|| kernel_failed("MaxReduce"));
                    (on_gpu(v), i)
                } else {
                    dispatch_real!(dtype, T => {
                        let (max_idx, &max_val) = data
                            .host::<T>()
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("MaxReduce: a NaN input has no maximum (S3: PyTorch propagates NaN)"))
                            .expect("MaxReduce: a tensor has at least one element");
                        (Storage::from_compute(dtype, vec![max_val]), max_idx)
                    })
                };
                (
                    max,
                    GradFn::MaxReduce(MaxReduceGradFn {
                        input_shape: shape,
                        max_index: max_idx,
                    }),
                )
            }
            ReduceOp::Mean => {
                let mean = if device.is_gpu() {
                    on_gpu(Self::gpu_mean_reduce(&data).unwrap_or_else(|| kernel_failed("Mean")))
                } else {
                    dispatch_real!(dtype, T => {
                        let sum: T = data.host::<T>().iter().copied().sum();
                        Storage::from_compute(dtype, vec![sum / T::from_usize(data.len())])
                    })
                };
                (mean, GradFn::Mean(MeanGradFn { input_shape: shape }))
            }
        };

        edge(
            [self_t.clone()],
            storage,
            Shape::derived(vec![1]),
            device,
            grad_fn,
        )
    }
}
