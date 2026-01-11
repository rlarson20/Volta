//! GPU-accelerated tensor operations
//!
//! This module provides GPU implementations of tensor operations.
//! These are called automatically when tensors are on GPU storage.

use crate::RawTensor;
use crate::gpu::GpuKernels;
use crate::storage::Storage;
#[cfg(feature = "gpu")]
use std::cell::RefCell;
#[cfg(feature = "gpu")]
use std::sync::Arc;

impl RawTensor {
    /// GPU-accelerated element-wise addition
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_add(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "add")?;

        // Return GPU storage without eager CPU transfer.
        // CPU access requires explicit to_vec() call.
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated element-wise subtraction
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_sub(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "sub")?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated element-wise multiplication
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mul(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "mul")?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated element-wise division
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_div(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "div")?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated element-wise maximum
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_max(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "max")?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated element-wise modulo
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mod(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "mod")?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated element-wise less-than comparison
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_cmplt(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "cmplt")?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated sum reduction - returns the scalar sum
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_sum_reduce(data: &Storage) -> Option<f32> {
        let buf = data.gpu_buffer()?;
        GpuKernels::sum(buf)
    }

    /// GPU-accelerated max reduction - returns (max_value, max_index)
    /// Uses GPU to find max value, then CPU to find the index
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_max_reduce(data: &Storage) -> Option<(f32, usize)> {
        let buf = data.gpu_buffer()?;
        let max_val = GpuKernels::max(buf)?;

        // Find the index of the max value on CPU (using cached data)
        let cpu_data = data.to_vec();
        let max_idx = cpu_data
            .iter()
            .enumerate()
            .find(|&(_, v)| (v - max_val).abs() < 1e-6)
            .map(|(i, _)| i)
            .unwrap_or(0);

        Some((max_val, max_idx))
    }

    /// GPU-accelerated mean reduction - returns the scalar mean
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mean_reduce(data: &Storage) -> Option<f32> {
        let buf = data.gpu_buffer()?;
        GpuKernels::mean(buf)
    }

    /// GPU-accelerated matrix multiplication
    #[allow(dead_code)]
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_matmul(
        a: &Storage,
        b: &Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::matmul(buf_a, buf_b, m, k, n)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated unary operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_unary(input: &Storage, op: &str) -> Option<Storage> {
        let buf = input.gpu_buffer()?;

        let result = GpuKernels::unary_op(buf, op)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated unary backward operation
    ///
    /// Computes gradient for unary operations: grad = out_grad * df/dx
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_unary_backward(out_grad: &Storage, x: &Storage, op: &str) -> Option<Storage> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_x = x.gpu_buffer()?;

        let result = GpuKernels::unary_backward(buf_out, buf_x, op)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated binary backward operation for first input (a)
    ///
    /// Computes gradient with respect to first input for binary operations
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_backward_a(
        out_grad: &Storage,
        a: &Storage,
        b: &Storage,
        op: &str,
    ) -> Option<Storage> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_backward_a(buf_out, buf_a, buf_b, op)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated binary backward operation for second input (b)
    ///
    /// Computes gradient with respect to second input for binary operations
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_backward_b(
        out_grad: &Storage,
        a: &Storage,
        b: &Storage,
        op: &str,
    ) -> Option<Storage> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_backward_b(buf_out, buf_a, buf_b, op)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated binary backward operation with broadcasting support
    ///
    /// Computes gradients for both inputs simultaneously, handling broadcasting
    /// by reducing gradients over broadcasted dimensions.
    ///
    /// # Returns
    /// A tuple of (gradient wrt a, gradient wrt b)
    #[cfg(feature = "gpu")]
    #[allow(dead_code)] // Keep as fallback, safe version is now used
    pub(crate) fn gpu_binary_backward_broadcast(
        out_grad: &Storage,
        a: &Storage,
        b: &Storage,
        op: &str,
        out_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Option<(Storage, Storage)> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let (a_grad, b_grad) = GpuKernels::binary_backward_broadcast(
            buf_out, buf_a, buf_b, op, out_shape, a_shape, b_shape,
        )?;

        Some((
            Storage::Gpu {
                buffer: Arc::new(a_grad),
                dtype: crate::dtype::DType::F32,
                cpu_cache: RefCell::new(None),
            },
            Storage::Gpu {
                buffer: Arc::new(b_grad),
                dtype: crate::dtype::DType::F32,
                cpu_cache: RefCell::new(None),
            },
        ))
    }

    /// GPU-accelerated binary backward with RACE-FREE broadcasting support
    ///
    /// This uses a two-pass algorithm to avoid race conditions in gradient reduction.
    ///
    /// # Returns
    /// A tuple of (gradient wrt a, gradient wrt b)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_backward_broadcast_safe(
        out_grad: &Storage,
        a: &Storage,
        b: &Storage,
        op: &str,
        out_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Option<(Storage, Storage)> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let (a_grad, b_grad) = GpuKernels::binary_backward_broadcast_safe(
            buf_out, buf_a, buf_b, op, out_shape, a_shape, b_shape,
        )?;

        Some((
            Storage::Gpu {
                buffer: Arc::new(a_grad),
                dtype: crate::dtype::DType::F32,
                cpu_cache: RefCell::new(None),
            },
            Storage::Gpu {
                buffer: Arc::new(b_grad),
                dtype: crate::dtype::DType::F32,
                cpu_cache: RefCell::new(None),
            },
        ))
    }

    /// GPU-accelerated permute operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_permute(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        axes: &[usize],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::permute(buf, old_shape, new_shape, axes)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated expand (broadcast) operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_expand(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::expand(buf, old_shape, new_shape)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated pad operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_pad(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::pad(buf, old_shape, new_shape, padding)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated shrink operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_shrink(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        ranges: &[(usize, usize)],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::shrink(buf, old_shape, new_shape, ranges)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated stride operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_stride(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        strides: &[usize],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::stride(buf, old_shape, new_shape, strides)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated matmul backward operation for gradient with respect to A
    ///
    /// Computes dA = grad @ B^T
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_matmul_backward_a(
        grad: &Storage,
        b: &Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<Storage> {
        let buf_grad = grad.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::matmul_backward_a(buf_grad, buf_b, m, k, n)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated matmul backward operation for gradient with respect to B
    ///
    /// Computes dB = A^T @ grad
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_matmul_backward_b(
        a: &Storage,
        grad: &Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_grad = grad.gpu_buffer()?;

        let result = GpuKernels::matmul_backward_b(buf_a, buf_grad, m, k, n)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated sum backward operation
    ///
    /// Broadcasts scalar gradient to all elements
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_sum_backward(grad: f32, input_size: usize) -> Option<Storage> {
        let result = GpuKernels::sum_backward(grad, input_size)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated mean backward operation
    ///
    /// Broadcasts scalar gradient / count to all elements
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mean_backward(grad: f32, input_size: usize) -> Option<Storage> {
        let result = GpuKernels::mean_backward(grad, input_size)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated max backward operation
    ///
    /// Sparse gradient - only max element receives gradient
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_max_backward(
        grad: f32,
        input_size: usize,
        max_index: usize,
    ) -> Option<Storage> {
        let result = GpuKernels::max_backward(grad, input_size, max_index)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated optimizer step operation
    ///
    /// Updates parameters in-place using gradients and optimizer state.
    /// Supports SGD, SGD with momentum, and Adam optimizers.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_optimizer_step(
        params: &Storage,
        grads: &Storage,
        state1: &Storage,
        state2: &Storage,
        opt_params: &crate::gpu::OptimizerStepParams,
    ) -> Option<()> {
        let buf_params = params.gpu_buffer()?;
        let buf_grads = grads.gpu_buffer()?;
        let buf_state1 = state1.gpu_buffer()?;
        let buf_state2 = state2.gpu_buffer()?;
        GpuKernels::optimizer_step(buf_params, buf_grads, buf_state1, buf_state2, opt_params)
    }

    /// GPU-accelerated im2col transformation for convolution
    ///
    /// Transforms 4D input (B, C, H, W) into 2D matrix (B*H_out*W_out, C*K_h*K_w).
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gpu_im2col(
        input: &Storage,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        h_out: usize,
        w_out: usize,
    ) -> Option<Storage> {
        let buf_input = input.gpu_buffer()?;
        let result = GpuKernels::im2col(
            buf_input, batch_size, channels, height, width, kernel_h, kernel_w, stride_h, stride_w,
            h_out, w_out,
        )?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    // ===== MOVEMENT BACKWARD OPERATIONS =====

    /// GPU-accelerated permute backward operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_permute_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        axes: &[usize],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::permute_backward(buf, old_shape, new_shape, axes)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated expand backward operation (sum over broadcast dimensions)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_expand_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::expand_backward(buf, old_shape, new_shape)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated pad backward operation (extract center region)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_pad_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::pad_backward(buf, old_shape, new_shape, padding)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated shrink backward operation (pad back to original size)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_shrink_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        ranges: &[(usize, usize)],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::shrink_backward(buf, old_shape, new_shape, ranges)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }

    /// GPU-accelerated stride backward operation (upsample gradient)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_stride_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        strides: &[usize],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::stride_backward(buf, old_shape, new_shape, strides)?;
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            dtype: crate::dtype::DType::F32,
            cpu_cache: RefCell::new(None),
        })
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::gpu::is_gpu_available;
    use crate::storage::Storage;

    #[test]
    fn gpu_add_returns_gpu_storage_without_eager_transfer() {
        if !is_gpu_available() {
            // Skip on machines without a usable GPU.
            return;
        }

        let a = Storage::gpu(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Storage::gpu(vec![5.0, 6.0, 7.0, 8.0]);

        let out = RawTensor::gpu_add(&a, &b).expect("gpu_add failed");

        // Must be GPU-backed storage.
        assert!(out.is_gpu());

        // cpu_cache is NOT eagerly populated (lazy transfer).
        // to_vec() fetches from GPU buffer when cache is None.
        assert_eq!(out.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }
}
