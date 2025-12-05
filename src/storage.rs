//! Tensor storage abstraction
//!
//! This module provides a unified interface for tensor data storage
//! that can be backed by either CPU memory or GPU buffers.

#[cfg(feature = "gpu")]
use crate::gpu::{GpuBuffer, is_gpu_available};

use crate::device::Device;

/// Storage backend for tensor data
///
/// This enum allows tensors to store their data either on CPU (as a Vec<f32>)
/// or on GPU (as a `GpuBuffer`). Operations automatically handle the right backend.
#[derive(Clone)]
pub enum Storage {
    /// CPU storage - data lives in main memory
    Cpu(Vec<f32>),

    /// GPU storage - data lives in GPU memory
    #[cfg(feature = "gpu")]
    Gpu {
        /// The GPU buffer (wrapped in Arc for cheap cloning)
        buffer: std::sync::Arc<GpuBuffer>,
        /// Cached CPU copy (for operations that need CPU access)
        /// This is lazily populated when needed
        cpu_cache: Option<Vec<f32>>,
    },
}

impl Storage {
    /// Create new CPU storage from data
    pub fn cpu(data: Vec<f32>) -> Self {
        Storage::Cpu(data)
    }

    /// Create new GPU storage from data (falls back to CPU if GPU unavailable)
    #[cfg(feature = "gpu")]
    pub fn gpu(data: Vec<f32>) -> Self {
        if let Some(buffer) = GpuBuffer::from_slice(&data) {
            Storage::Gpu {
                buffer: std::sync::Arc::new(buffer),
                cpu_cache: Some(data), // Keep original data as cache
            }
        } else {
            // Fall back to CPU
            Storage::Cpu(data)
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn gpu(data: Vec<f32>) -> Self {
        Storage::Cpu(data)
    }

    /// Get data as a slice (may trigger GPU->CPU transfer)
    pub fn as_slice(&self) -> &[f32] {
        match self {
            Storage::Cpu(data) => data,
            #[cfg(feature = "gpu")]
            Storage::Gpu {
                buffer: _,
                cpu_cache,
            } => {
                // If we have a cache, use it
                // Otherwise, we'd need interior mutability to populate it
                // For now, panic - the user should call to_vec() first
                cpu_cache
                    .as_ref()
                    .expect("GPU buffer needs to_vec() call before slice access")
            }
        }
    }

    /// Get data as a mutable slice (only works for CPU storage)
    pub fn as_mut_slice(&mut self) -> Option<&mut [f32]> {
        match self {
            Storage::Cpu(data) => Some(data),
            #[cfg(feature = "gpu")]
            Storage::Gpu { .. } => None, // Can't mutate GPU data directly
        }
    }

    /// Convert to Vec<f32> (triggers GPU->CPU transfer if needed)
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            Storage::Cpu(data) => data.clone(),
            #[cfg(feature = "gpu")]
            Storage::Gpu { buffer, cpu_cache } => {
                cpu_cache.clone().unwrap_or_else(|| buffer.to_vec())
            }
        }
    }

    /// Get the length
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(data) => data.len(),
            #[cfg(feature = "gpu")]
            Storage::Gpu { buffer, .. } => buffer.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this is GPU storage
    pub fn is_gpu(&self) -> bool {
        match self {
            Storage::Cpu(_) => false,
            #[cfg(feature = "gpu")]
            Storage::Gpu { .. } => true,
        }
    }

    /// Move to a specific device
    pub fn to_device(&self, device: &Device) -> Self {
        match device {
            Device::CPU => Storage::Cpu(self.to_vec()),
            Device::GPU(_) | Device::Metal(_) => {
                #[cfg(feature = "gpu")]
                {
                    if is_gpu_available() {
                        Storage::gpu(self.to_vec())
                    } else {
                        eprintln!("Warning: GPU requested but not available, using CPU");
                        Storage::Cpu(self.to_vec())
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    eprintln!("Warning: GPU feature not enabled, using CPU");
                    Storage::Cpu(self.to_vec())
                }
            }
        }
    }

    /// Get the GPU buffer if this is GPU storage
    #[cfg(feature = "gpu")]
    pub fn gpu_buffer(&self) -> Option<&GpuBuffer> {
        match self {
            Storage::Gpu { buffer, .. } => Some(buffer.as_ref()),
            _ => None,
        }
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Cpu(data) => write!(f, "Storage::Cpu({} elements)", data.len()),
            #[cfg(feature = "gpu")]
            Storage::Gpu { buffer, .. } => write!(f, "Storage::Gpu({} elements)", buffer.len()),
        }
    }
}
