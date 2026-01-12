//! GPU backend for tensor operations using wgpu
//!
//! This module provides GPU acceleration for tensor operations.
//! The key components are:
//! - `GpuContext`: Manages the GPU device and command queue
//! - `GpuBuffer`: Holds tensor data on the GPU
//! - Various compute shaders for tensor operations

mod buffer;
mod context;
mod kernels;
mod pool;

pub use buffer::GpuBuffer;
pub use context::GpuContext;
pub use kernels::{GpuKernels, MovementParams, OptimizerStepParams};
pub use pool::{BufferPool, BufferPoolConfig};

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
/// If GPU is not available, this is a no-op.
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
/// gpu_sync();
/// ```
pub fn gpu_sync() {
    if let Some(ctx) = get_gpu_context() {
        ctx.sync();
    }
}
