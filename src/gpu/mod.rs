//! GPU backend for tensor operations using wgpu
//!
//! This module provides GPU acceleration for tensor operations.
//! The key components are:
//! - `GpuContext`: Manages the GPU device and command queue
//! - `GpuBuffer`: Holds tensor data on the GPU
//! - Various compute shaders for tensor operations

mod buffer;
mod context;
pub mod early_warning;
mod kernels;
pub mod monitor;
mod pool;
pub mod staging_pool;
pub mod system_monitor;

pub use buffer::GpuBuffer;
pub use context::{GpuContext, GpuSyncError};
pub use early_warning::{EarlyWarningSystem, HealthStatus as EarlyWarningHealthStatus, TrendData};
pub use kernels::{GpuKernels, MovementParams, OptimizerStepParams};
pub use monitor::{ResourceStatus, check_system_resources, get_process_memory_mb};
pub use pool::{BufferPool, BufferPoolConfig};
pub use staging_pool::{StagingBufferPool, StagingPoolStats};
pub use system_monitor::{MonitorStats, SystemMonitor};

use std::sync::OnceLock;

// Global GPU context - initialized lazily on first use
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// Get the global GPU context, initializing it if necessary
/// Returns None if GPU is not available
pub fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CONTEXT
        .get_or_init(|| match GpuContext::new() {
            Ok(ctx) => {
                println!("GPU initialized: {}", ctx.device_name());
                Some(ctx)
            }
            Err(e) => {
                eprintln!("GPU initialization failed: {}. Falling back to CPU.", e);
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
/// ```ignore
/// use volta::{gpu_sync, RawTensor, TensorOps, Device};
///
/// // Run many GPU operations
/// for _ in 0..100 {
///     tensor.relu();
/// }
///
/// // Ensure all work completes before timing next section
/// if !gpu_sync() {
///     eprintln!("Warning: GPU sync timed out");
/// }
/// ```
#[must_use]
pub fn gpu_sync() -> bool {
    get_gpu_context().map(|ctx| ctx.sync()).unwrap_or(true) // No GPU = success
}

/// Get the current number of pending GPU submissions (diagnostic)
///
/// This returns the number of GPU submissions that have been queued but
/// not yet synchronized. Useful for debugging GPU command queue issues.
///
/// Returns 0 if GPU is not available.
///
/// # Example
///
/// ```ignore
/// use volta::{gpu_pending_count, RawTensor, TensorOps, Device};
///
/// let t = tensor.to_device(Device::gpu().unwrap());
/// let _ = t.relu();
/// println!("Pending ops: {}", gpu_pending_count());
/// ```
#[must_use]
pub fn gpu_pending_count() -> u32 {
    get_gpu_context()
        .map(|ctx| ctx.pending_count())
        .unwrap_or(0)
}

/// Get the GPU sync threshold (diagnostic)
///
/// Returns the number of pending submissions that triggers automatic
/// synchronization. Returns 0 if GPU is not available.
#[must_use]
pub fn gpu_sync_threshold() -> u32 {
    get_gpu_context()
        .map(|ctx| ctx.sync_threshold())
        .unwrap_or(0)
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
/// ```ignore
/// use volta::gpu_cleanup;
///
/// // After a batch of GPU operations
/// gpu_cleanup();
/// println!("GPU memory released");
/// ```
#[must_use]
pub fn gpu_cleanup() -> bool {
    if let Some(ctx) = get_gpu_context() {
        ctx.sync(); // Ensure all work complete before clearing
        ctx.buffer_pool().clear();
        ctx.staging_pool().clear();
        true
    } else {
        false
    }
}

/// Force GPU memory compaction to release accumulated buffers
///
/// Call this between heavy benchmark groups to ensure GPU memory is actually
/// released by the driver, not just marked for cleanup.
///
/// On macOS Metal, buffer drops don't immediately reclaim memory—this function
/// forces the driver to complete pending work and run its internal GC by:
/// 1. Clearing buffer pools (drops wgpu::Buffer instances)
/// 2. Submitting an empty command buffer to flush the command queue
/// 3. Polling the device to completion
/// 4. Brief sleep to allow Metal's internal command queue to drain
/// 5. Final poll to pick up any deferred cleanup
///
/// Returns true if compaction succeeded, false if GPU unavailable.
///
/// # Example
///
/// ```ignore
/// use volta::gpu_compact;
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
        ctx.queue().submit(Some(encoder.finish()));

        // Step 3: Poll device to completion to ensure all pending work finishes
        let _ = ctx.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(2)),
        });

        // Step 4: Brief sleep to allow Metal's internal command queue to drain
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Step 5: Poll again to pick up any deferred cleanup
        let _ = ctx.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(500)),
        });

        true
    } else {
        false
    }
}

/// Get current buffer pool statistics for debugging
///
/// Returns a tuple of (buffer_pool_count, staging_pool_count) representing
/// the number of buffers currently held in each pool.
///
/// Returns None if GPU is not available.
///
/// # Example
///
/// ```ignore
/// use volta::gpu_pool_stats;
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
/// ```ignore
/// use volta::{Device, TensorOps, Tensor, gpu::invalidate_all_tensor_caches};
///
/// let a = tensor.to_device(Device::gpu().unwrap());
/// let b = tensor.to_device(Device::gpu().unwrap());
///
/// // ... run benchmarks that may populate CPU caches ...
///
/// // Release CPU-side copies to free memory
/// invalidate_all_tensor_caches(&[&a, &b]);
/// ```
pub fn invalidate_all_tensor_caches(tensors: &[&crate::Tensor]) {
    for tensor in tensors {
        tensor.borrow_mut().invalidate_gpu_cache();
    }
}
