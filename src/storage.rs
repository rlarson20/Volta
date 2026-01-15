//! Tensor storage abstraction
//!
//! This module provides a unified interface for tensor data storage
//! that can be backed by either CPU memory or GPU buffers, with support
//! for multiple data types (f16, bf16, f32, f64, etc.).

use crate::device::Device;
use crate::dtype::DType;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuBuffer, is_gpu_available};
use bytemuck::{cast_slice, cast_slice_mut};
use half::{bf16, f16};

#[cfg(feature = "gpu")]
use std::cell::RefCell;
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

/// Storage backend for tensor data
///
/// This enum allows tensors to store their data either on CPU (as raw bytes with dtype)
/// or on GPU (as a `GpuBuffer`). Operations automatically handle the right backend.
pub enum Storage {
    /// CPU storage - data lives in main memory as raw bytes
    Cpu { data: Vec<u8>, dtype: DType },

    /// GPU storage - data lives in GPU memory
    #[cfg(feature = "gpu")]
    Gpu {
        /// The GPU buffer (wrapped in Arc for cheap cloning)
        buffer: std::sync::Arc<GpuBuffer>,
        /// Data type
        dtype: DType,
        /// Cached CPU copy (for operations that need CPU access)
        /// Uses RefCell for lazy population - populated on first CPU access
        cpu_cache: RefCell<Option<Vec<u8>>>,
    },
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu { data, dtype } => Storage::Cpu {
                data: data.clone(),
                dtype: *dtype,
            },
            #[cfg(feature = "gpu")]
            Storage::Gpu {
                buffer,
                dtype,
                cpu_cache: _,
            } => Storage::Gpu {
                buffer: buffer.clone(),
                dtype: *dtype,
                cpu_cache: RefCell::new(None),
            },
        }
    }
}

impl Storage {
    // ========== Constructors ==========

    /// Create new CPU storage from f32 data (default dtype)
    pub fn cpu(data: Vec<f32>) -> Self {
        let bytes: Vec<u8> = cast_slice(&data).to_vec();
        Storage::Cpu {
            data: bytes,
            dtype: DType::F32,
        }
    }

    /// Create new CPU storage from f32 data with explicit dtype
    pub fn cpu_f32(data: Vec<f32>) -> Self {
        Self::cpu(data)
    }

    /// Create new CPU storage from f64 data
    pub fn cpu_f64(data: Vec<f64>) -> Self {
        let bytes: Vec<u8> = cast_slice(&data).to_vec();
        Storage::Cpu {
            data: bytes,
            dtype: DType::F64,
        }
    }

    /// Create new CPU storage from f16 data
    pub fn cpu_f16(data: Vec<f16>) -> Self {
        let bytes: Vec<u8> = cast_slice(&data).to_vec();
        Storage::Cpu {
            data: bytes,
            dtype: DType::F16,
        }
    }

    /// Create new CPU storage from bf16 data
    pub fn cpu_bf16(data: Vec<bf16>) -> Self {
        let bytes: Vec<u8> = cast_slice(&data).to_vec();
        Storage::Cpu {
            data: bytes,
            dtype: DType::BF16,
        }
    }

    /// Create new CPU storage from raw bytes with a specific dtype
    pub fn from_bytes(data: Vec<u8>, dtype: DType) -> Self {
        assert!(
            data.len().is_multiple_of(dtype.size_of()),
            "Byte length {} is not divisible by dtype size {}",
            data.len(),
            dtype.size_of()
        );
        Storage::Cpu { data, dtype }
    }

    /// Create new GPU storage from f32 data (falls back to CPU if GPU unavailable)
    ///
    /// The CPU cache is initialized lazily - data is only copied back from GPU
    /// when accessed via `as_f32_slice()` or similar methods. This avoids
    /// duplicating data in CPU memory unnecessarily.
    #[cfg(feature = "gpu")]
    pub fn gpu(data: Vec<f32>) -> Self {
        if let Some(buffer) = GpuBuffer::from_slice(&data) {
            Storage::Gpu {
                buffer: std::sync::Arc::new(buffer),
                dtype: DType::F32,
                cpu_cache: RefCell::new(None), // Lazy initialization on first access
            }
        } else {
            Self::cpu(data)
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn gpu(data: Vec<f32>) -> Self {
        Self::cpu(data)
    }

    // ========== Dtype Access ==========

    /// Get the data type of this storage
    pub fn dtype(&self) -> DType {
        match self {
            Storage::Cpu { dtype, .. } => *dtype,
            #[cfg(feature = "gpu")]
            Storage::Gpu { dtype, .. } => *dtype,
        }
    }

    /// Create new zero-filled storage on the specified device
    pub fn new_zeros(len: usize, device: &Device) -> Self {
        match device {
            Device::CPU => Storage::cpu(vec![0.0; len]),
            Device::GPU(_) => {
                #[cfg(feature = "gpu")]
                {
                    if is_gpu_available() {
                        // Create GPU zeros
                        if let Some(buffer) = GpuBuffer::zeros(len) {
                            return Storage::Gpu {
                                buffer: std::sync::Arc::new(buffer),
                                dtype: DType::F32,
                                cpu_cache: RefCell::new(None),
                            };
                        }
                    }
                    // Fall back to CPU
                    Storage::cpu(vec![0.0; len])
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Storage::cpu(vec![0.0; len])
                }
            }
        }
    }

    // ========== F32 Access (Backward Compatible) ==========

    /// Get data as f32 slice. Only valid if dtype is F32.
    /// Panics if dtype is not F32.
    pub fn as_f32_slice(&self) -> &[f32] {
        match self {
            Storage::Cpu { data, dtype } => {
                assert_eq!(*dtype, DType::F32, "Storage dtype is {:?}, not F32", dtype);
                cast_slice(data)
            }
            #[cfg(feature = "gpu")]
            Storage::Gpu {
                buffer,
                dtype,
                cpu_cache,
            } => {
                assert_eq!(*dtype, DType::F32, "Storage dtype is {:?}, not F32", dtype);
                // Ensure cache is populated (lazy transfer from GPU)
                {
                    let mut cache = cpu_cache.borrow_mut();
                    if cache.is_none() {
                        let f32_data = buffer.to_vec();
                        let bytes: Vec<u8> = cast_slice(&f32_data).to_vec();
                        *cache = Some(bytes);
                    }
                }
                // SAFETY: cache is now populated and never cleared.
                // We use as_ptr() to get a reference that outlives the RefCell borrow.
                // This is safe because:
                // 1. The cache is Some(Vec<u8>) after the block above
                // 2. We never clear the cache once populated
                // 3. The Vec inside is stable (won't move/reallocate)
                // 4. We cast bytes to &[f32] using cast_slice
                unsafe {
                    let bytes = (*cpu_cache.as_ptr()).as_ref().unwrap().as_slice();
                    cast_slice(bytes)
                }
            }
        }
    }

    /// Get data as mutable f32 slice. Only valid if dtype is F32.
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Storage::Cpu { data, dtype } => {
                if *dtype == DType::F32 {
                    Some(cast_slice_mut(data))
                } else {
                    None
                }
            }
            #[cfg(feature = "gpu")]
            Storage::Gpu { .. } => None,
        }
    }

    /// Backward compatible: get data as slice (assumes F32)
    pub fn as_slice(&self) -> &[f32] {
        self.as_f32_slice()
    }

    /// Backward compatible: get data as mutable slice (assumes F32)
    pub fn as_mut_slice(&mut self) -> Option<&mut [f32]> {
        self.as_f32_slice_mut()
    }

    // ========== Other Dtype Access ==========

    /// Get data as f64 slice. Only valid if dtype is F64.
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            Storage::Cpu { data, dtype } if *dtype == DType::F64 => Some(cast_slice(data)),
            #[cfg(feature = "gpu")]
            Storage::Gpu {
                buffer,
                dtype,
                cpu_cache,
            } if *dtype == DType::F64 => {
                // Ensure cache is populated (lazy transfer from GPU)
                {
                    let mut cache = cpu_cache.borrow_mut();
                    if cache.is_none() {
                        let f32_data = buffer.to_vec();
                        let bytes: Vec<u8> = cast_slice(&f32_data).to_vec();
                        *cache = Some(bytes);
                    }
                }
                // SAFETY: Similar to as_f32_slice(), cache is populated and stable
                unsafe {
                    let bytes = (*cpu_cache.as_ptr()).as_ref().unwrap().as_slice();
                    Some(cast_slice(bytes))
                }
            }
            _ => None,
        }
    }

    /// Get data as f16 slice. Only valid if dtype is F16.
    pub fn as_f16_slice(&self) -> Option<&[f16]> {
        match self {
            Storage::Cpu { data, dtype } if *dtype == DType::F16 => Some(cast_slice(data)),
            #[cfg(feature = "gpu")]
            Storage::Gpu {
                buffer,
                dtype,
                cpu_cache,
            } if *dtype == DType::F16 => {
                // Ensure cache is populated (lazy transfer from GPU)
                {
                    let mut cache = cpu_cache.borrow_mut();
                    if cache.is_none() {
                        let f32_data = buffer.to_vec();
                        let bytes: Vec<u8> = cast_slice(&f32_data).to_vec();
                        *cache = Some(bytes);
                    }
                }
                // SAFETY: Similar to as_f32_slice(), cache is populated and stable
                unsafe {
                    let bytes = (*cpu_cache.as_ptr()).as_ref().unwrap().as_slice();
                    Some(cast_slice(bytes))
                }
            }
            _ => None,
        }
    }

    /// Get data as bf16 slice. Only valid if dtype is BF16.
    pub fn as_bf16_slice(&self) -> Option<&[bf16]> {
        match self {
            Storage::Cpu { data, dtype } if *dtype == DType::BF16 => Some(cast_slice(data)),
            #[cfg(feature = "gpu")]
            Storage::Gpu {
                buffer,
                dtype,
                cpu_cache,
            } if *dtype == DType::BF16 => {
                // Ensure cache is populated (lazy transfer from GPU)
                {
                    let mut cache = cpu_cache.borrow_mut();
                    if cache.is_none() {
                        let f32_data = buffer.to_vec();
                        let bytes: Vec<u8> = cast_slice(&f32_data).to_vec();
                        *cache = Some(bytes);
                    }
                }
                // SAFETY: Similar to as_f32_slice(), cache is populated and stable
                unsafe {
                    let bytes = (*cpu_cache.as_ptr()).as_ref().unwrap().as_slice();
                    Some(cast_slice(bytes))
                }
            }
            _ => None,
        }
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Storage::Cpu { data, .. } => data,
            #[cfg(feature = "gpu")]
            Storage::Gpu {
                buffer,
                cpu_cache,
                dtype: _,
            } => {
                // Ensure cache is populated (lazy transfer from GPU)
                {
                    let mut cache = cpu_cache.borrow_mut();
                    if cache.is_none() {
                        let f32_data = buffer.to_vec();
                        let bytes: Vec<u8> = cast_slice(&f32_data).to_vec();
                        *cache = Some(bytes);
                    }
                }
                // SAFETY: Similar to as_f32_slice(), cache is populated and stable
                unsafe { (*cpu_cache.as_ptr()).as_ref().unwrap().as_slice() }
            }
        }
    }

    // ========== Conversion ==========

    /// Convert to `Vec<f32>` (always works, may involve conversion)
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.dtype() {
            DType::F32 => self.as_f32_slice().to_vec(),
            DType::F64 => self
                .as_f64_slice()
                .unwrap()
                .iter()
                .map(|&x| x as f32)
                .collect(),
            DType::F16 => self
                .as_f16_slice()
                .unwrap()
                .iter()
                .map(|x| x.to_f32())
                .collect(),
            DType::BF16 => self
                .as_bf16_slice()
                .unwrap()
                .iter()
                .map(|x| x.to_f32())
                .collect(),
            DType::I32 => {
                let bytes = self.as_bytes();
                let ints: &[i32] = cast_slice(bytes);
                ints.iter().map(|&x| x as f32).collect()
            }
            DType::I64 => {
                let bytes = self.as_bytes();
                let ints: &[i64] = cast_slice(bytes);
                ints.iter().map(|&x| x as f32).collect()
            }
            DType::U8 => {
                let bytes = self.as_bytes();
                bytes.iter().map(|&x| x as f32).collect()
            }
            DType::Bool => {
                let bytes = self.as_bytes();
                bytes
                    .iter()
                    .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                    .collect()
            }
        }
    }

    /// Backward compatible: convert to `Vec<f32>`
    pub fn to_vec(&self) -> Vec<f32> {
        self.to_f32_vec()
    }

    /// Convert storage to a different dtype
    pub fn to_dtype(&self, target: DType) -> Storage {
        if self.dtype() == target {
            return self.clone();
        }

        // First convert to f32 as intermediate
        let f32_data = self.to_f32_vec();

        // Then convert to target dtype
        match target {
            DType::F32 => Storage::cpu(f32_data),
            DType::F64 => {
                let data: Vec<f64> = f32_data.iter().map(|&x| x as f64).collect();
                Storage::cpu_f64(data)
            }
            DType::F16 => {
                let data: Vec<f16> = f32_data.iter().map(|&x| f16::from_f32(x)).collect();
                Storage::cpu_f16(data)
            }
            DType::BF16 => {
                let data: Vec<bf16> = f32_data.iter().map(|&x| bf16::from_f32(x)).collect();
                Storage::cpu_bf16(data)
            }
            DType::I32 => {
                let data: Vec<i32> = f32_data.iter().map(|&x| x as i32).collect();
                let bytes: Vec<u8> = cast_slice(&data).to_vec();
                Storage::Cpu {
                    data: bytes,
                    dtype: DType::I32,
                }
            }
            DType::I64 => {
                let data: Vec<i64> = f32_data.iter().map(|&x| x as i64).collect();
                let bytes: Vec<u8> = cast_slice(&data).to_vec();
                Storage::Cpu {
                    data: bytes,
                    dtype: DType::I64,
                }
            }
            DType::U8 => {
                let data: Vec<u8> = f32_data
                    .iter()
                    .map(|&x| x.clamp(0.0, 255.0) as u8)
                    .collect();
                Storage::Cpu {
                    data,
                    dtype: DType::U8,
                }
            }
            DType::Bool => {
                let data: Vec<u8> = f32_data
                    .iter()
                    .map(|&x| if x != 0.0 { 1 } else { 0 })
                    .collect();
                Storage::Cpu {
                    data,
                    dtype: DType::Bool,
                }
            }
        }
    }

    // ========== Length and Properties ==========

    /// Get the number of elements
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu { data, dtype } => data.len() / dtype.size_of(),
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
            Storage::Cpu { .. } => false,
            #[cfg(feature = "gpu")]
            Storage::Gpu { .. } => true,
        }
    }

    /// Move to a specific device
    pub fn to_device(&self, device: &Device) -> Self {
        match device {
            Device::CPU => {
                let data = self.as_bytes().to_vec();
                Storage::Cpu {
                    data,
                    dtype: self.dtype(),
                }
            }
            Device::GPU(_) => {
                #[cfg(feature = "gpu")]
                {
                    if is_gpu_available() {
                        // Convert to f32 for GPU (GPU ops are f32 only for now)
                        Storage::gpu(self.to_f32_vec())
                    } else {
                        eprintln!("Warning: GPU requested but not available, using CPU");
                        self.clone()
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    eprintln!("Warning: GPU feature not enabled, using CPU");
                    self.clone()
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

    /// Invalidate the CPU cache for GPU storage
    ///
    /// This releases the CPU-side copy of GPU data, reducing memory usage.
    /// The cache will be repopulated on the next CPU access if needed.
    ///
    /// This is useful for releasing memory after GPU operations are complete
    /// and the CPU copy is no longer needed. Call this between benchmark groups
    /// or after training steps to reduce memory pressure.
    ///
    /// For CPU storage, this is a no-op.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let storage = Storage::gpu(vec![1.0, 2.0, 3.0]);
    /// let _ = storage.as_f32_slice(); // Populates CPU cache
    /// storage.invalidate_cpu_cache(); // Releases CPU copy
    /// ```
    #[cfg(feature = "gpu")]
    pub fn invalidate_cpu_cache(&self) {
        if let Storage::Gpu { cpu_cache, .. } = self {
            cpu_cache.borrow_mut().take();
        }
    }
}

// ========== Iterator Support (F32 only for backward compat) ==========

impl Storage {
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.as_f32_slice().iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f32> {
        self.as_f32_slice_mut()
            .expect("Mutable iteration requires F32 CPU storage")
            .iter_mut()
    }
}

// ========== Deref implementations (F32 only for backward compat) ==========

impl std::ops::Deref for Storage {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.as_f32_slice()
    }
}

impl std::ops::DerefMut for Storage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_f32_slice_mut()
            .expect("Mutable deref requires F32 CPU storage")
    }
}

// ========== Index implementations (F32 only for backward compat) ==========

impl Index<usize> for Storage {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_f32_slice()[index]
    }
}

impl IndexMut<usize> for Storage {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self
            .as_f32_slice_mut()
            .expect("Mutable access requires F32 CPU storage")[index]
    }
}

impl Index<Range<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.as_f32_slice()[range]
    }
}

impl Index<RangeFrom<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        &self.as_f32_slice()[range]
    }
}

impl Index<RangeTo<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        &self.as_f32_slice()[range]
    }
}

impl Index<RangeToInclusive<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeToInclusive<usize>) -> &Self::Output {
        &self.as_f32_slice()[range]
    }
}

impl Index<RangeInclusive<usize>> for Storage {
    type Output = [f32];
    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        &self.as_f32_slice()[range]
    }
}

impl Index<RangeFull> for Storage {
    type Output = [f32];
    fn index(&self, _range: RangeFull) -> &Self::Output {
        self.as_f32_slice()
    }
}

impl<'a> IntoIterator for &'a Storage {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_f32_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut Storage {
    type Item = &'a mut f32;
    type IntoIter = std::slice::IterMut<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_f32_slice_mut()
            .expect("Mutable iteration requires F32 CPU storage")
            .iter_mut()
    }
}

// ========== Comparison ==========

impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        if self.dtype() != other.dtype() {
            return false;
        }
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<Vec<f32>> for Storage {
    fn eq(&self, other: &Vec<f32>) -> bool {
        if self.dtype() != DType::F32 {
            return false;
        }
        self.as_f32_slice() == other.as_slice()
    }
}

impl PartialEq<Storage> for Vec<f32> {
    fn eq(&self, other: &Storage) -> bool {
        other == self
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Cpu { data, dtype } => {
                write!(
                    f,
                    "Storage::Cpu({} elements, {})",
                    data.len() / dtype.size_of(),
                    dtype
                )
            }
            #[cfg(feature = "gpu")]
            Storage::Gpu { buffer, dtype, .. } => {
                write!(f, "Storage::Gpu({} elements, {})", buffer.len(), dtype)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = Storage::cpu(data.clone());

        assert_eq!(storage.dtype(), DType::F32);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.as_f32_slice(), &data);
        assert_eq!(storage[0], 1.0);
        assert_eq!(storage[3], 4.0);
    }

    #[test]
    fn test_storage_f16() {
        let data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let storage = Storage::cpu_f16(data.clone());

        assert_eq!(storage.dtype(), DType::F16);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.as_f16_slice().unwrap(), &data);

        // Test conversion to f32
        let f32_vec = storage.to_f32_vec();
        assert_eq!(f32_vec.len(), 4);
        assert!((f32_vec[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_storage_conversion() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = Storage::cpu(data);

        // Convert to f16
        let f16_storage = storage.to_dtype(DType::F16);
        assert_eq!(f16_storage.dtype(), DType::F16);
        assert_eq!(f16_storage.len(), 4);

        // Convert back to f32
        let f32_storage = f16_storage.to_dtype(DType::F32);
        assert_eq!(f32_storage.dtype(), DType::F32);
        let recovered = f32_storage.to_f32_vec();
        assert!((recovered[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_storage_from_bytes() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = cast_slice(&data).to_vec();

        let storage = Storage::from_bytes(bytes, DType::F32);
        assert_eq!(storage.dtype(), DType::F32);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.as_f32_slice(), &data);
    }
}
