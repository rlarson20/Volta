//! GPU backend for tensor operations using wgpu
//!
//! This module provides GPU acceleration for tensor operations.
//! The key components are:
//! - `GpuContext`: Manages the GPU device and command queue
//! - `GpuBuffer`: Holds tensor data on the GPU
//! - Various compute shaders for tensor operations

mod buffer;
mod context;
#[deny(missing_docs, clippy::missing_docs_in_private_items)]
mod kernels;
mod pool;
pub(crate) mod shader_template;
/// Pool of `MAP_READ` staging buffers for GPU-to-CPU transfers.
pub(crate) mod staging_pool;

pub use crate::gpu_counters::GpuCounters;
pub use buffer::GpuBuffer;
pub(crate) use buffer::check_fits;
pub(crate) use context::GpuContext;
#[cfg(test)]
pub(crate) use context::set_f16_emulation;
pub(crate) use context::{BinaryBackwardPipelines, ComputePipelines};
pub(crate) use kernels::{AxisView, GpuKernels, MAX_GATHER_RANK, MaxResult, OptimizerStepParams};

use std::sync::OnceLock;

/// Global GPU context, initialized lazily on first use; `None` if no adapter was found.
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// Get the global GPU context, initializing it if necessary
/// Returns None if GPU is not available
pub(crate) fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CONTEXT
        .get_or_init(|| match GpuContext::new() {
            Ok(ctx) => {
                log::info!("GPU initialized: {}", ctx.device_name());
                Some(ctx)
            }
            Err(e) => {
                log::warn!("GPU initialization failed: {e}. No GPU is available; to_device(GPU) will panic.");
                None
            }
        })
        .as_ref()
}

/// Check if GPU is available
#[must_use]
pub fn is_gpu_available() -> bool {
    get_gpu_context().is_some()
}

/// The largest tensor, in bytes on the GPU, that this machine's GPU can hold, or `None`
/// without a GPU.
///
/// It is the adapter's `max_storage_buffer_binding_size` (every kernel binds a tensor as
/// one storage buffer), which Volta requests in place of wgpu's 128 MiB default (#306).
/// An f32 or bool element takes 4 bytes, and f16 and bf16 take 2, padded to a whole 4-byte
/// word. [`crate::Tensor::try_to_device`] returns an error for a larger tensor before
/// allocating anything; a kernel whose output would be larger panics.
#[must_use]
pub fn max_tensor_bytes() -> Option<u64> {
    get_gpu_context().map(GpuContext::max_binding_bytes)
}

/// Force GPU synchronization - wait for all pending commands to complete.
///
/// Call this when you need a clean sync point, such as:
/// - Between benchmark iterations to ensure accurate timing
/// - Before reading GPU results back to CPU
/// - After a burst of GPU operations to prevent command queue buildup
///
/// Returns true if sync completed successfully, false if it timed out.
/// A timeout doesn't necessarily mean the device is lost - the GPU may
/// just be under heavy load.
///
/// If GPU is not available, this returns true (no-op success).
///
/// # Example
///
/// ```no_run
/// use volta::{Device, randn};
/// use volta::gpu::gpu_sync;
///
/// let tensor = randn(&[64, 64]).to_device(Device::gpu().expect("a GPU"));
/// // Run many GPU operations
/// for _ in 0..100 {
///     let _ = tensor.relu();
/// }
///
/// // Ensure all work completes before timing next section
/// if !gpu_sync() {
///     eprintln!("Warning: GPU sync timed out");
/// }
/// ```
#[must_use]
pub fn gpu_sync() -> bool {
    get_gpu_context().is_none_or(context::GpuContext::sync) // No GPU = success
}

/// Get the current number of pending GPU submissions (diagnostic)
///
/// This returns the GPU submissions not yet known to be complete, plus uploads not yet
/// flushed: the back-pressure window, which stays below [`gpu_sync_threshold`]. Useful for
/// debugging GPU command queue issues.
///
/// Returns 0 if GPU is not available.
///
/// # Example
///
/// ```no_run
/// use volta::{Device, randn};
/// use volta::gpu::gpu_pending_count;
///
/// let t = randn(&[64, 64]).to_device(Device::gpu().expect("a GPU"));
/// let _ = t.relu();
/// println!("Pending ops: {}", gpu_pending_count());
/// ```
#[must_use]
pub fn gpu_pending_count() -> u32 {
    get_gpu_context().map_or(0, context::GpuContext::pending_count)
}

/// How often the GPU backend has submitted work, drained the queue, and copied data
/// between host and GPU since the process started (see [`GpuCounters`]).
///
/// # Example
///
/// ```no_run
/// use volta::{Device, randn};
/// use volta::gpu::gpu_counters;
///
/// let x = randn(&[64, 64]).to_device(Device::gpu().expect("a GPU"));
/// let before = gpu_counters();
/// let _ = x.relu().relu();
/// let used = gpu_counters().since(&before);
/// assert_eq!(used.readbacks, 0);
/// ```
#[must_use]
pub fn gpu_counters() -> GpuCounters {
    crate::gpu_counters::snapshot()
}

/// Get the GPU back-pressure threshold (diagnostic)
///
/// Returns the most submissions kept in flight: past it, a new submission waits for the
/// oldest to finish (not for the whole queue). Returns 0 if GPU is not available.
#[must_use]
pub fn gpu_sync_threshold() -> u32 {
    get_gpu_context().map_or(0, context::GpuContext::sync_threshold)
}

/// Clear GPU buffer pools to release memory
///
/// Call this between benchmark groups or when GPU memory needs to be reclaimed.
/// This clears:
/// - Buffer pool (reusable GPU buffers)
/// - Staging pool (GPU→CPU transfer buffers)
///
/// The function syncs the GPU before clearing to ensure all pending operations
/// complete before buffers are released.
///
/// Returns true if cleanup succeeded, false if GPU unavailable.
///
/// # Example
///
/// ```no_run
/// use volta::gpu::gpu_cleanup;
///
/// // After a batch of GPU operations
/// gpu_cleanup();
/// println!("GPU memory released");
/// ```
#[must_use]
pub fn gpu_cleanup() -> bool {
    get_gpu_context().is_some_and(|ctx| {
        ctx.sync();
        ctx.buffer_pool().clear();
        ctx.staging_pool().clear();
        true
    })
}

/// Force GPU memory compaction to release accumulated buffers
///
/// Call this between heavy benchmark groups to ensure GPU memory is actually
/// released by the driver, not just marked for cleanup.
///
/// On macOS Metal, buffer drops don't immediately reclaim memory—this function
/// forces the driver to complete pending work and run its internal GC by:
/// 1. Clearing buffer pools (drops `wgpu::Buffer` instances)
/// 2. Submitting an empty command buffer to flush the command queue
/// 3. Polling the device to completion
/// 4. Brief sleep to allow Metal's internal command queue to drain
/// 5. Final poll to pick up any deferred cleanup
///
/// Returns true if compaction succeeded, false if GPU unavailable.
///
/// # Example
///
/// ```no_run
/// use volta::gpu::gpu_compact;
///
/// // After a heavy benchmark group
/// gpu_compact();
/// println!("GPU memory compacted and released");
/// ```
#[must_use]
pub fn gpu_compact() -> bool {
    if let Some(ctx) = get_gpu_context() {
        // Step 1: Clear buffer pools (drops wgpu::Buffer instances)
        ctx.buffer_pool().clear();
        ctx.staging_pool().clear();

        // Step 2: Submit empty command buffer to flush command queue
        let encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compaction Encoder"),
            });
        let flush = ctx.queue().submit(Some(encoder.finish()));

        // Step 3: Poll until that submission (and everything before it) finishes; our own
        // index, not `None`, so another thread's in-flight submit can't race it (#437).
        let _ = ctx.device().poll(wgpu::PollType::Wait {
            submission_index: Some(flush.clone()),
            timeout: Some(std::time::Duration::from_secs(2)),
        });

        // Step 4: Brief sleep to allow Metal's internal command queue to drain
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Step 5: Poll again to pick up any deferred cleanup
        let _ = ctx.device().poll(wgpu::PollType::Wait {
            submission_index: Some(flush),
            timeout: Some(std::time::Duration::from_millis(500)),
        });

        true
    } else {
        false
    }
}

/// Get current buffer pool statistics for debugging
///
/// Returns a tuple of (`buffer_pool_count`, `staging_pool_count`) representing
/// the number of buffers currently held in each pool.
///
/// Returns None if GPU is not available.
///
/// # Example
///
/// ```no_run
/// use volta::gpu::gpu_pool_stats;
///
/// if let Some((buffers, staging)) = gpu_pool_stats() {
///     println!("Pooled buffers: {}, Staging buffers: {}", buffers, staging);
/// }
/// ```
#[must_use]
pub fn gpu_pool_stats() -> Option<(usize, usize)> {
    get_gpu_context().map(|ctx| {
        let buffer_stats = ctx.buffer_pool().stats();
        let staging_stats = ctx.staging_pool().stats();
        (buffer_stats.total_pooled, staging_stats.total_pooled)
    })
}

/// Invalidate CPU caches for all provided GPU tensors
///
/// This function releases CPU-side copies of GPU data, reducing memory usage.
/// The GPU data remains intact and will be re-cached on next CPU access if needed.
///
/// This is particularly useful in benchmarks to prevent memory accumulation
/// when setup tensors persist through multiple benchmark iterations.
///
/// # Arguments
/// * `tensors` - Slice of tensor references to process
///
/// # Example
///
/// ```no_run
/// use volta::{Device, gpu::invalidate_all_tensor_caches, randn};
///
/// let gpu = Device::gpu().expect("a GPU");
/// let a = randn(&[64, 64]).to_device(gpu.clone());
/// let b = randn(&[64, 64]).to_device(gpu);
///
/// // ... run benchmarks that may populate CPU caches ...
///
/// // Release CPU-side copies to free memory
/// invalidate_all_tensor_caches(&[&a, &b]);
/// ```
pub fn invalidate_all_tensor_caches(tensors: &[&crate::Tensor]) {
    for tensor in tensors {
        tensor.invalidate_gpu_cache();
    }
}
