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
pub fn gpu_pending_count() -> u32 {
    get_gpu_context()
        .map(|ctx| ctx.pending_count())
        .unwrap_or(0)
}

/// Get the GPU sync threshold (diagnostic)
///
/// Returns the number of pending submissions that triggers automatic
/// synchronization. Returns 0 if GPU is not available.
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
pub fn gpu_pool_stats() -> Option<(usize, usize)> {
    get_gpu_context().map(|ctx| {
        let buffer_stats = ctx.buffer_pool().stats();
        let staging_stats = ctx.staging_pool().stats();
        (buffer_stats.total_pooled, staging_stats.total_pooled)
    })
}
