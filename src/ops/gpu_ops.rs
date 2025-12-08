//! GPU-accelerated tensor operations
//!
//! This module provides GPU implementations of tensor operations.
//! These are called automatically when tensors are on GPU storage.

use crate::gpu::{GpuBuffer, GpuKernels};
use crate::storage::Storage;
use crate::{RawTensor, Tensor};
#[cfg(feature = "gpu")]
use std::sync::Arc;

impl RawTensor {
    /// GPU-accelerated element-wise addition
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_add(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "add")?;

        // Populate cpu_cache so later CPU-style access (as_slice / iter) is safe.
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
        })
    }

    /// GPU-accelerated element-wise subtraction
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_sub(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "sub")?;
        let cpu_cache = Some(result.to_vec());
        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
        })
    }

    /// GPU-accelerated element-wise multiplication
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mul(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "mul")?;
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
        })
    }

    /// GPU-accelerated element-wise division
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_div(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "div")?;
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
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
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
        })
    }

    /// GPU-accelerated unary operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_unary(input: &Storage, op: &str) -> Option<Storage> {
        let buf = input.gpu_buffer()?;

        let result = GpuKernels::unary_op(buf, op)?;
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
        })
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::gpu::is_gpu_available;
    use crate::storage::Storage;

    #[test]
    fn gpu_add_populates_cpu_cache_and_roundtrips() {
        if !is_gpu_available() {
            // Skip on machines without a usable GPU.
            return;
        }

        let a = Storage::gpu(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Storage::gpu(vec![5.0, 6.0, 7.0, 8.0]);

        let out = RawTensor::gpu_add(&a, &b).expect("gpu_add failed");

        // Must be GPU-backed storage.
        assert!(out.is_gpu());

        // cpu_cache must be populated so CPU-style access is safe.
        // as_slice internally uses the cache.
        assert_eq!(out.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

        // to_vec should also work and match expectations.
        assert_eq!(out.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }
}
