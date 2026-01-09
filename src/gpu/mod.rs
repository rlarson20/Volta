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

pub use buffer::GpuBuffer;
pub use context::GpuContext;
pub use kernels::{GpuKernels, MovementParams, OptimizerStepParams};

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
