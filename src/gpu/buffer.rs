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
use std::num::NonZeroU64;
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

/// `Ok` if `len` elements of `dtype` fit in one storage binding on this GPU
/// ([`super::GpuContext::max_binding_bytes`]), the largest tensor a kernel can bind.
///
/// # Errors
/// A message naming the tensor's size and the limit if it does not fit, or saying that no
/// GPU is available. The caller wraps it in its own `VoltaError`.
pub(crate) fn check_fits(dtype: DType, len: usize) -> std::result::Result<(), String> {
    let ctx = get_gpu_context().ok_or_else(|| "no GPU is available".to_string())?;
    let limit = ctx.max_binding_bytes();
    let bytes = gpu_bytes_checked(dtype, len);
    if bytes.is_some_and(|b| b as u64 <= limit) {
        return Ok(());
    }
    Err(format!(
        "a tensor of {len} {dtype} elements ({} bytes on the GPU) exceeds {}'s \
         {limit}-byte storage binding limit (`max_storage_buffer_binding_size`)",
        bytes.map_or_else(|| "over usize::MAX".to_string(), |b| b.to_string()),
        ctx.device_name(),
    ))
}

/// [`gpu_bytes`], or `None` if the byte count overflows `usize`.
fn gpu_bytes_checked(dtype: DType, len: usize) -> Option<usize> {
    match dtype {
        DType::F32 | DType::Bool => len.checked_mul(4),
        DType::F16 | DType::BF16 => len.div_ceil(2).checked_mul(4),
        DType::F64 | DType::I32 | DType::I64 | DType::U8 => Some(gpu_bytes(dtype, len)),
    }
}

impl GpuBuffer {
    /// A pooled or fresh storage buffer of at least `byte_size` bytes.
    ///
    /// # Panics
    /// If `byte_size` is over this GPU's storage binding limit (#306): no kernel could bind
    /// the buffer. `to_device` checks an upload first and returns an error instead
    /// ([`check_fits`]); this catches a kernel output that outgrows its inputs.
    #[track_caller]
    fn allocate(byte_size: usize, label: &str) -> Option<wgpu::Buffer> {
        let ctx = get_gpu_context()?;
        let limit = ctx.max_binding_bytes();
        assert!(
            byte_size as u64 <= limit,
            "GPU {label}: {byte_size} bytes exceed this GPU's {limit}-byte storage binding \
             limit (`max_storage_buffer_binding_size`)"
        );
        if let Some(buffer) = ctx.buffer_pool().acquire(byte_size) {
            return Some(buffer);
        }
        // A pooled size rounds up to its power-of-two bucket; a size too large to pool is
        // allocated exactly, so it never costs twice its bytes (#306).
        let alloc_size = ctx.buffer_pool().allocation_size_for(byte_size);
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
    ///
    /// # Panics
    /// If `bytes` is not `gpu_bytes(dtype, len)` long, in every build profile and before a
    /// GPU is looked up: a short upload would leave the tail of the buffer as whatever the
    /// pool held, and a long one would be cut off (#173).
    #[track_caller]
    fn upload(dtype: DType, len: usize, bytes: &[u8], label: &str) -> Option<Self> {
        assert_eq!(
            bytes.len(),
            gpu_bytes(dtype, len),
            "GPU upload of {len} {dtype} elements: got {} bytes",
            bytes.len()
        );
        let ctx = get_gpu_context()?;
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
    ///
    /// Costs a host allocation and an upload of `gpu_bytes(dtype, len)` bytes. For a kernel
    /// output that the kernel writes in full, use [`Self::uninit_of`].
    #[must_use]
    pub(crate) fn zeros_of(dtype: DType, len: usize) -> Option<Self> {
        let bytes = vec![0u8; gpu_bytes(dtype, len)];
        Self::upload(dtype, len, &bytes, "Tensor Buffer (zeros)")
    }

    /// A buffer of `len` `dtype` elements whose contents are unspecified (#310): a pooled
    /// buffer still holds a dropped tensor's bytes, a fresh one is zero (wgpu clears new
    /// buffers). Nothing is uploaded.
    ///
    /// Only for a kernel output whose shader writes every element `0..len` on every path.
    /// An output a kernel writes in part, accumulates into, or reads before writing takes
    /// [`Self::zeros_of`].
    #[must_use]
    pub(crate) fn uninit_of(dtype: DType, len: usize) -> Option<Self> {
        let ctx = get_gpu_context()?;
        let buffer = Self::allocate(gpu_bytes(dtype, len), "Tensor Buffer (output)")?;
        Some(Self {
            buffer: Some(buffer),
            len,
            dtype,
            pool: Some(ctx.buffer_pool_arc()),
        })
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
        let copy = ctx.queue().submit(Some(encoder.finish()));
        // Counted, but no `maybe_sync`: the blocking poll below waits for this submit and
        // everything before it. It does not reset the counter, so the next `maybe_sync` may
        // sync an already idle queue once, which costs a poll and nothing else.
        ctx.increment_pending();

        // Map the staging buffer and wait for the copy (and everything before it: a queue
        // completes in order). Waiting on this submission, not on `None` (the device's last
        // one), matters when threads share the device: `None` can name another thread's
        // submission before the queue has registered it, which trips an assert in wgpu-core
        // (#437).
        let buffer_slice = staging_buffer.slice(..byte_size);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("map_async callback receiver dropped");
        });
        ctx.device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(copy),
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("Failed to poll device");
        receiver
            .recv()
            .expect("map_async result channel closed")
            .expect("Failed to map buffer");

        let data = buffer_slice
            .get_mapped_range()
            .expect("Failed to get mapped range");
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
            crate::storage::Storage::Gpu(_) => unreachable!("try_from_bytes builds CPU storage"),
        }
    }

    /// Get the underlying wgpu buffer (for use in compute passes). Internal (#326): a caller
    /// holding the handle could `destroy()` it, and the pool would then hand the destroyed
    /// buffer to an unrelated allocation.
    ///
    /// # Panics
    /// If the wgpu buffer was taken out of this `GpuBuffer`. Only `Drop` does that, so a live
    /// buffer never panics here.
    #[must_use]
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        debug_assert!(self.buffer.is_some(), "Buffer should not be taken");
        self.buffer.as_ref().expect("Buffer should not be taken")
    }

    /// The binding a kernel uses for this buffer: bytes `0..gpu_bytes(dtype, len)`, not the
    /// whole allocation. A pooled buffer is rounded up to a power of two, and the slack past
    /// `len` holds a dropped tensor's bytes; binding only the tensor's bytes keeps a shader
    /// that bounds its loop with `arrayLength` inside them (#306, G13), and lets a tensor
    /// whose bucket is over the binding limit still bind.
    #[must_use]
    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer(),
            offset: 0,
            // An empty buffer has no bytes to bind; wgpu takes `None` as the whole (empty)
            // buffer, which is what binding it did before.
            size: NonZeroU64::new(gpu_bytes(self.dtype, self.len) as u64),
        })
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
    /// Internal, like [`Self::buffer`] (#325, #326): every caller passes a region of a buffer
    /// it built, so an out-of-range region is a bug and panics.
    ///
    /// # Arguments
    /// * `offset` - Starting element offset in this buffer
    /// * `len` - Number of elements to copy
    /// # Panics
    /// Panics if `offset + len` is past this buffer's length (a pooled buffer's slack past
    /// `len` holds a dropped tensor's data, #325), if the buffer is taken, or (16-bit types)
    /// if `offset` is odd: a copy starts on a 4-byte boundary, which is a whole word of two
    /// elements.
    #[must_use]
    #[track_caller]
    pub(crate) fn copy_region(&self, offset: usize, len: usize) -> Option<Self> {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "copy_region: elements {offset}..{offset}+{len} are past the buffer's {} elements \
             (index-mapping bug)",
            self.len
        );
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

#[cfg(test)]
mod upload_tests {
    use super::*;

    /// #173: a byte length that disagrees with `len` panics in release builds too, and
    /// with or without a GPU (the check runs before the context is looked up).
    #[test]
    #[should_panic(expected = "GPU upload of 3 f32 elements: got 8 bytes")]
    fn upload_rejects_a_wrong_byte_length() {
        let _ = GpuBuffer::upload(DType::F32, 3, &[0u8; 8], "test");
    }
}

#[cfg(test)]
mod copy_region_tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// The panic message of `f`, which must panic.
    fn panic_message(f: impl FnOnce()) -> String {
        let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(ToString::to_string))
            .expect("a string panic payload")
    }

    /// #325: a region past the buffer's length panics with a Volta message instead of
    /// reading the pooled slack (a dropped tensor's data) or failing inside wgpu. Covers a
    /// region that ends past the length but inside the size class (`0..8` of 5 elements,
    /// `3..8`) and one past the size class (`0..9`). Skipped without a GPU.
    #[test]
    fn copy_region_rejects_a_region_past_the_length() {
        if !crate::gpu::is_gpu_available() {
            return;
        }
        let buf = GpuBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("GPU buffer");
        for (offset, len) in [(0, 8), (3, 5), (0, 9), (usize::MAX, 2)] {
            let msg = panic_message(|| {
                let _ = buf.copy_region(offset, len);
            });
            assert!(
                msg.contains("past the buffer's 5 elements"),
                "copy_region({offset}, {len}): {msg}"
            );
        }
        // An in-range region still copies.
        let tail = buf.copy_region(3, 2).expect("in-range copy");
        assert_eq!(tail.to_vec(), vec![4.0, 5.0]);
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    /// #306: `check_fits` accepts a tensor at the binding limit and refuses one element
    /// past it, and a length whose byte count overflows, without allocating either.
    /// Skipped without a GPU.
    #[test]
    fn check_fits_is_the_binding_limit() {
        let Some(ctx) = get_gpu_context() else {
            return;
        };
        let limit = usize::try_from(ctx.max_binding_bytes()).expect("limit fits usize");
        let at = limit / 4;
        assert!(check_fits(DType::F32, at).is_ok());
        let msg = check_fits(DType::F32, at + 1).expect_err("one element past the limit");
        assert!(msg.contains("storage binding limit"), "{msg}");
        // Two 16-bit elements per word: 2 * (limit / 4) of them fill the limit exactly.
        assert!(check_fits(DType::F16, 2 * at).is_ok());
        assert!(check_fits(DType::BF16, 2 * at + 1).is_err());
        let msg = check_fits(DType::F32, usize::MAX).expect_err("overflowing byte count");
        assert!(msg.contains("over usize::MAX"), "{msg}");
    }

    /// #306 / G13: a buffer binds its tensor's bytes, not its pooled allocation. Five f32
    /// elements sit in a 32-byte bucket; the binding is 20 bytes. Skipped without a GPU.
    #[test]
    fn binding_covers_only_the_tensor() {
        if get_gpu_context().is_none() {
            return;
        }
        for (dtype, len, bytes) in [
            (DType::F32, 5, 20),
            (DType::F16, 5, 12),
            (DType::Bool, 3, 12),
        ] {
            let buf = GpuBuffer::uninit_of(dtype, len).expect("GPU buffer");
            let wgpu::BindingResource::Buffer(b) = buf.binding() else {
                panic!("a buffer binding")
            };
            assert_eq!(b.offset, 0);
            assert_eq!(b.size.map(NonZeroU64::get), Some(bytes), "{dtype} x {len}");
            assert!(buf.buffer().size() >= bytes);
        }
    }
}
