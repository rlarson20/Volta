//! Tensor storage abstraction
//!
//! This module provides a unified interface for tensor data storage
//! that can be backed by either CPU memory or GPU buffers.

use crate::device::Device;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuBuffer, is_gpu_available};
#[cfg(feature = "gpu")]
use std::cell::RefCell;
use std::ops::{
    Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

/// Storage backend for tensor data
///
/// This enum allows tensors to store their data either on CPU (as a `Vec<f32>`)
/// or on GPU (as a `GpuBuffer`). Operations automatically handle the right backend.
pub enum Storage {
    /// CPU storage - data lives in main memory
    Cpu(Vec<f32>),

    /// GPU storage - data lives in GPU memory
    #[cfg(feature = "gpu")]
    Gpu {
        /// The GPU buffer (wrapped in Arc for cheap cloning)
        buffer: std::sync::Arc<GpuBuffer>,
        /// Cached CPU copy (for operations that need CPU access)
        /// Uses RefCell for lazy population - populated on first CPU access
        cpu_cache: RefCell<Option<Vec<f32>>>,
    },
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(data) => Storage::Cpu(data.clone()),
            #[cfg(feature = "gpu")]
            Storage::Gpu { buffer, cpu_cache } => Storage::Gpu {
                buffer: buffer.clone(),
                cpu_cache: RefCell::new(cpu_cache.borrow().clone()),
            },
        }
    }
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
                cpu_cache: RefCell::new(Some(data)), // Keep original data as cache
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

    /// Create new zero-filled storage on the specified device
    pub fn new_zeros(len: usize, device: &Device) -> Self {
        match device {
            Device::CPU => Storage::Cpu(vec![0.0; len]),
            Device::GPU(_) => {
                #[cfg(feature = "gpu")]
                {
                    if is_gpu_available() {
                        // Create GPU zeros
                        if let Some(buffer) = GpuBuffer::zeros(len) {
                            return Storage::Gpu {
                                buffer: std::sync::Arc::new(buffer),
                                cpu_cache: RefCell::new(None),
                            };
                        }
                    }
                    // Fall back to CPU
                    Storage::Cpu(vec![0.0; len])
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Storage::Cpu(vec![0.0; len])
                }
            }
        }
    }

    /// Get data as a slice (triggers GPU->CPU transfer on first access)
    ///
    /// For GPU storage, this lazily populates the CPU cache on first access.
    /// Subsequent calls return the cached data without GPU transfer.
    pub fn as_slice(&self) -> &[f32] {
        match self {
            Storage::Cpu(data) => data,
            #[cfg(feature = "gpu")]
            Storage::Gpu { buffer, cpu_cache } => {
                // Ensure cache is populated (lazy transfer from GPU)
                {
                    let mut cache = cpu_cache.borrow_mut();
                    if cache.is_none() {
                        *cache = Some(buffer.to_vec());
                    }
                }
                // SAFETY: cache is now populated and never cleared.
                // We use as_ptr() to get a reference that outlives the RefCell borrow.
                // This is safe because:
                // 1. The cache is Some(Vec) after the block above
                // 2. We never clear the cache once populated
                // 3. The Vec inside is stable (won't move/reallocate)
                unsafe { (*cpu_cache.as_ptr()).as_ref().unwrap().as_slice() }
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

    /// Convert to `Vec<f32>` (triggers GPU->CPU transfer if needed)
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            Storage::Cpu(data) => data.clone(),
            #[cfg(feature = "gpu")]
            Storage::Gpu { buffer, cpu_cache } => {
                // Check if cache exists, if not fetch from GPU
                let cache = cpu_cache.borrow();
                if let Some(ref data) = *cache {
                    data.clone()
                } else {
                    drop(cache);
                    // Populate cache for future use
                    let data = buffer.to_vec();
                    *cpu_cache.borrow_mut() = Some(data.clone());
                    data
                }
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
            Device::GPU(_) => {
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

impl Storage {
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.as_slice().iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f32> {
        self.as_mut_slice()
            .expect("Mutable iteration not supported for GPU storage")
            .iter_mut()
    }
}

impl Deref for Storage {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for Storage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
            .expect("Mutable access not supported for GPU storage")
    }
}

impl Index<usize> for Storage {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for Storage {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self
            .as_mut_slice()
            .expect("Mutable access not supported for GPU storage")[index]
    }
}

impl Index<Range<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.as_slice()[range]
    }
}

impl Index<RangeFrom<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        &self.as_slice()[range]
    }
}

impl Index<RangeTo<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        &self.as_slice()[range]
    }
}

impl Index<RangeToInclusive<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeToInclusive<usize>) -> &Self::Output {
        &self.as_slice()[range]
    }
}

impl Index<RangeInclusive<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        &self.as_slice()[range]
    }
}

impl Index<RangeFull> for Storage {
    type Output = [f32];
    fn index(&self, _range: RangeFull) -> &Self::Output {
        self.as_slice()
    }
}

impl<'a> IntoIterator for &'a Storage {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut Storage {
    type Item = &'a mut f32;
    type IntoIter = std::slice::IterMut<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice()
            .expect("Mutable iteration not supported for GPU storage")
            .iter_mut()
    }
}

impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<Vec<f32>> for Storage {
    fn eq(&self, other: &Vec<f32>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<Storage> for Vec<f32> {
    fn eq(&self, other: &Storage) -> bool {
        self.as_slice() == other.as_slice()
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
