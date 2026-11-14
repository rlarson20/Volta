//! GPU buffer management
//!
//! `GpuBuffer` wraps a wgpu buffer and provides methods for
//! transferring data between CPU and GPU.
//!
//! # Element types (ADDENDUM §11 P8)
//!
//! A buffer holds F32, F16, BF16 or Bool elements ([`GpuBuffer::dtype`]). F32 is four
//! bytes per element; F16 and BF16 are two, packed two per 32-bit word low half first
//! (the CPU's byte layout, so a transfer is a byte copy) and padded to a whole word;
//! Bool is one `u32` (0 or 1) per element, so a shader can write any element without
//! touching its neighbors. `shader_template` expands each shader for the element type.
//!
//! # Buffer Pooling
//!
//! `GpuBuffers` are automatically returned to a pool when dropped, enabling
//! efficient reuse across operations. This prevents memory exhaustion during
//! repeated GPU operations like benchmarking or training loops.

use super::get_gpu_context;
use super::pool::BufferPool;
use crate::dtype::DType;
use crate::storage::HostBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

/// A buffer that lives on the GPU
///
/// This is analogous to a `Vec<T>` but the data lives in GPU memory.
/// We need to explicitly copy data to/from the CPU.
///
/// When dropped, the buffer is returned to the pool for reuse rather than
/// being immediately deallocated. This significantly improves performance
/// for repeated operations.
pub struct GpuBuffer {
    /// The actual GPU buffer (Option to allow taking on Drop)
    buffer: Option<wgpu::Buffer>,
    /// Size in elements
    len: usize,
    /// Element type: F32, F16, BF16 or Bool
    dtype: DType,
    /// Reference to the pool for returning the buffer on drop
    pool: Option<Arc<BufferPool>>,
}

/// Bytes `len` elements of `dtype` occupy on the GPU (16-bit types padded to a word).
///
/// # Panics
/// If `dtype` has no GPU representation.
fn gpu_bytes(dtype: DType, len: usize) -> usize {
    match dtype {
        DType::F32 | DType::Bool => len * 4,
        DType::F16 | DType::BF16 => len.div_ceil(2) * 4,
        other @ (DType::F64 | DType::I32 | DType::I64 | DType::U8) => {
            panic!("GPU buffers hold f32, f16, bf16 or bool, not {other}")
        }
    }
}

impl GpuBuffer {
    /// A pooled or fresh storage buffer of at least `byte_size` bytes.
    fn allocate(byte_size: usize, label: &str) -> Option<wgpu::Buffer> {
        let ctx = get_gpu_context()?;
        if let Some(buffer) = ctx.buffer_pool().acquire(byte_size) {
            return Some(buffer);
        }
        // Allocate with the bucket size (power of 2) for consistent pooling
        let alloc_size = BufferPool::allocation_size(byte_size);
        Some(ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: alloc_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// A buffer of `len` `dtype` elements holding `bytes` (already in the GPU layout,
    /// `gpu_bytes(dtype, len)` long).
    fn upload(dtype: DType, len: usize, bytes: &[u8], label: &str) -> Option<Self> {
        let ctx = get_gpu_context()?;
        debug_assert_eq!(bytes.len(), gpu_bytes(dtype, len));
        let buffer = Self::allocate(bytes.len(), label)?;
        ctx.queue().write_buffer(&buffer, 0, bytes);
        ctx.increment_pending();
        ctx.maybe_sync();
        Some(Self {
            buffer: Some(buffer),
            len,
            dtype,
            pool: Some(ctx.buffer_pool_arc()),
        })
    }

    /// Create a new f32 GPU buffer from CPU data
    ///
    /// This copies the data from CPU to GPU memory.
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Option<Self> {
        Self::upload(
            DType::F32,
            data.len(),
            bytemuck::cast_slice(data),
            "Tensor Buffer",
        )
    }

    /// A GPU copy of `host` (F32, F16, BF16 or Bool).
    ///
    /// # Panics
    /// If `host` has another dtype (callers check: `Storage::to_device`).
    #[must_use]
    pub(crate) fn from_host(host: &HostBuf) -> Option<Self> {
        let dtype = host.dtype();
        let len = host.len();
        match host {
            HostBuf::Bool(b) => {
                let words: Vec<u32> = b.iter().map(|&v| u32::from(v)).collect();
                Self::upload(
                    dtype,
                    len,
                    bytemuck::cast_slice(&words),
                    "Tensor Buffer (bool)",
                )
            }
            HostBuf::F32(_) => Self::upload(dtype, len, host.as_bytes(), "Tensor Buffer"),
            HostBuf::F16(_) | HostBuf::BF16(_) => {
                let mut bytes = host.as_bytes().to_vec();
                bytes.resize(gpu_bytes(dtype, len), 0);
                Self::upload(dtype, len, &bytes, "Tensor Buffer (16-bit)")
            }
            HostBuf::F64(_) | HostBuf::I32(_) | HostBuf::I64(_) | HostBuf::U8(_) => {
                panic!("GPU buffers hold f32, f16, bf16 or bool, not {dtype}")
            }
        }
    }

    /// Create a zero-initialized f32 GPU buffer
    ///
    /// Useful for output buffers that will be written by shaders.
    #[must_use]
    pub fn zeros(len: usize) -> Option<Self> {
        Self::zeros_of(DType::F32, len)
    }

    /// A zero-initialized buffer of `len` `dtype` elements (all-zero bits are 0.0 or
    /// `false` in every GPU element type).
    #[must_use]
    pub(crate) fn zeros_of(dtype: DType, len: usize) -> Option<Self> {
        let bytes = vec![0u8; gpu_bytes(dtype, len)];
        Self::upload(dtype, len, &bytes, "Tensor Buffer (zeros)")
    }

    /// The element type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The buffer's bytes in the GPU layout (`gpu_bytes` long).
    fn download(&self) -> Vec<u8> {
        let ctx = get_gpu_context().expect("GPU context should exist if buffer exists");
        let buffer = self.buffer();
        let byte_size = gpu_bytes(self.dtype, self.len) as u64;

        // Try to get a staging buffer from the pool
        let staging_buffer = ctx.staging_pool().acquire(byte_size).unwrap_or_else(|| {
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Buffer Encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, byte_size);
        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();

        // Map the staging buffer and wait for the copy (and everything before it).
        let buffer_slice = staging_buffer.slice(..byte_size);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("map_async callback receiver dropped");
        });
        ctx.device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("Failed to poll device");
        receiver
            .recv()
            .expect("map_async result channel closed")
            .expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let bytes = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        ctx.staging_pool().release(staging_buffer, byte_size);
        bytes
    }

    /// Copy f32 data from GPU back to CPU
    ///
    /// This is a relatively expensive operation - try to minimize transfers!
    /// # Panics
    /// Panics if GPU context does not exist, or the buffer is not F32 (use
    /// `to_host` for any element type).
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        assert_eq!(
            self.dtype,
            DType::F32,
            "GpuBuffer::to_vec: an f32 read of a {} buffer",
            self.dtype
        );
        bytemuck::pod_collect_to_vec(&self.download())
    }

    /// Copy the buffer to the host in its own element type.
    ///
    /// # Panics
    /// If the GPU context is gone, or a Bool word is not 0 or 1 (a kernel bug).
    #[must_use]
    pub(crate) fn to_host(&self) -> HostBuf {
        let bytes = self.download();
        let host = match self.dtype {
            DType::Bool => {
                let words: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes);
                let bytes: Vec<u8> = words
                    .iter()
                    .map(|&w| {
                        assert!(w <= 1, "GPU bool buffer holds {w}, not 0 or 1");
                        u8::from(w == 1)
                    })
                    .collect();
                crate::storage::Storage::try_from_bytes(&bytes, DType::Bool)
            }
            dtype @ (DType::F16
            | DType::BF16
            | DType::F32
            | DType::F64
            | DType::I32
            | DType::I64
            | DType::U8) => {
                let n = self.len * dtype.size_of();
                crate::storage::Storage::try_from_bytes(
                    bytes.get(..n).expect("download covers the elements"),
                    dtype,
                )
            }
        };
        match host.expect("valid bytes for the buffer's dtype") {
            crate::storage::Storage::Cpu(h) => h,
            #[cfg(feature = "gpu")]
            crate::storage::Storage::Gpu(_) => unreachable!("try_from_bytes builds CPU storage"),
        }
    }

    /// Get the underlying wgpu buffer (for use in compute passes)
    ///
    /// # Panics
    /// If the wgpu buffer was taken out of this `GpuBuffer`. Only `Drop` does that, so a live
    /// buffer never panics here.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        debug_assert!(self.buffer.is_some(), "Buffer should not be taken");
        self.buffer.as_ref().expect("Buffer should not be taken")
    }

    /// Get the number of elements
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy a region of this buffer to a new GPU buffer of the same element type
    ///
    /// # Arguments
    /// * `offset` - Starting element offset in this buffer
    /// * `len` - Number of elements to copy
    /// # Panics
    /// Panics if buffer is taken, or (16-bit types) `offset` is odd: a copy starts on a
    /// 4-byte boundary, which is a whole word of two elements.
    #[must_use]
    pub fn copy_region(&self, offset: usize, len: usize) -> Option<Self> {
        let ctx = get_gpu_context()?;
        let src_buffer = self.buffer();
        let byte_size = gpu_bytes(self.dtype, len);
        let byte_offset = match self.dtype {
            DType::F16 | DType::BF16 => {
                assert!(
                    offset.is_multiple_of(2),
                    "copy_region: odd offset {offset} into a 16-bit buffer"
                );
                offset * 2
            }
            DType::F32 | DType::Bool | DType::F64 | DType::I32 | DType::I64 | DType::U8 => {
                offset * 4
            }
        };
        let new_buffer = Self::allocate(byte_size, "Copied Buffer Region")?;

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Buffer Region Encoder"),
            });
        encoder.copy_buffer_to_buffer(
            src_buffer,
            byte_offset as u64,
            &new_buffer,
            0,
            byte_size as u64,
        );
        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(Self {
            buffer: Some(new_buffer),
            len,
            dtype: self.dtype,
            pool: Some(ctx.buffer_pool_arc()),
        })
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Return the buffer to the pool for reuse
        if let Some(buffer) = self.buffer.take()
            && let Some(pool) = &self.pool
        {
            // Try to return to pool; if pool is full, buffer is dropped normally
            pool.release(buffer, gpu_bytes(self.dtype, self.len));
        }
    }
}
