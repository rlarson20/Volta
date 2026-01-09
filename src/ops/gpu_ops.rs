//! GPU-accelerated tensor operations
//!
//! This module provides GPU implementations of tensor operations.
//! These are called automatically when tensors are on GPU storage.

use crate::RawTensor;
use crate::gpu::GpuKernels;
use crate::storage::Storage;
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

    /// GPU-accelerated element-wise maximum
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_max(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "max")?;
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
        })
    }

    /// GPU-accelerated element-wise modulo
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mod(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "mod")?;
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
        })
    }

    /// GPU-accelerated element-wise less-than comparison
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_cmplt(a: &Storage, b: &Storage) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, "cmplt")?;
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
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
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
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
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
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
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
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
        let cpu_cache = Some(result.to_vec());

        Some(Storage::Gpu {
            buffer: Arc::new(result),
            cpu_cache,
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
