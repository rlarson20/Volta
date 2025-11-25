//! GPU-accelerated tensor operations
//!
//! This module provides GPU implementations of tensor operations.
//! These are called automatically when tensors are on GPU storage.

#[cfg(feature = "gpu")]
use crate::gpu::{GpuBuffer, GpuKernels};
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

impl RawTensor {
    /// GPU-accelerated element-wise addition
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_add(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "add")?;

        Some(Storage::Gpu {
            buffer: std::sync::Arc::new(result),
            cpu_cache: None,
        })
    }

    /// GPU-accelerated element-wise subtraction
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_sub(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "sub")?;

        Some(Storage::Gpu {
            buffer: std::sync::Arc::new(result),
            cpu_cache: None,
        })
    }

    /// GPU-accelerated element-wise multiplication
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mul(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "mul")?;

        Some(Storage::Gpu {
            buffer: std::sync::Arc::new(result),
            cpu_cache: None,
        })
    }

    /// GPU-accelerated element-wise division
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_div(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "div")?;

        Some(Storage::Gpu {
            buffer: std::sync::Arc::new(result),
            cpu_cache: None,
        })
    }

    /// GPU-accelerated matrix multiplication
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
            buffer: std::sync::Arc::new(result),
            cpu_cache: None,
        })
    }

    /// GPU-accelerated unary operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_unary(input: &Storage, op: &str) -> Option<Storage> {
        let buf = input.gpu_buffer()?;

        let result = GpuKernels::unary_op(buf, op)?;

        Some(Storage::Gpu {
            buffer: std::sync::Arc::new(result),
            cpu_cache: None,
        })
    }
}
