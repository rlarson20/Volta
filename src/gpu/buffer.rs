//! GPU buffer management
//!
//! `GpuBuffer` wraps a wgpu buffer and provides methods for
//! transferring data between CPU and GPU.
//!
//! # Buffer Pooling
//!
//! `GpuBuffers` are automatically returned to a pool when dropped, enabling
//! efficient reuse across operations. This prevents memory exhaustion during
//! repeated GPU operations like benchmarking or training loops.

use super::get_gpu_context;
use super::pool::BufferPool;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

/// A buffer that lives on the GPU
///
/// This is analogous to a `Vec<f32>` but the data lives in GPU memory.
/// We need to explicitly copy data to/from the CPU.
///
/// When dropped, the buffer is returned to the pool for reuse rather than
/// being immediately deallocated. This significantly improves performance
/// for repeated operations.
pub struct GpuBuffer {
    /// The actual GPU buffer (Option to allow taking on Drop)
    buffer: Option<wgpu::Buffer>,
    /// Size in number of f32 elements
    len: usize,
    /// Reference to the pool for returning the buffer on drop
    pool: Option<Arc<BufferPool>>,
}

impl GpuBuffer {
    /// Create a new GPU buffer from CPU data
    ///
    /// This copies the data from CPU to GPU memory.
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Option<Self> {
        let ctx = get_gpu_context()?;
        let byte_size = std::mem::size_of_val(data);

        // Try to get a buffer from the pool first
        if let Some(buffer) = ctx.buffer_pool().acquire(byte_size) {
            // Write data to the pooled buffer
            ctx.queue()
                .write_buffer(&buffer, 0, bytemuck::cast_slice(data));

            return Some(GpuBuffer {
                buffer: Some(buffer),
                len: data.len(),
                pool: Some(ctx.buffer_pool_arc()),
            });
        }

        // Allocate with the bucket size (power of 2) for consistent pooling
        let alloc_size = BufferPool::allocation_size(byte_size);

        // Create a buffer with the STORAGE usage (for compute shaders)
        // and COPY_SRC/COPY_DST for data transfer
        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Buffer"),
            size: alloc_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write the actual data
        ctx.queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));

        Some(GpuBuffer {
            buffer: Some(buffer),
            len: data.len(),
            pool: Some(ctx.buffer_pool_arc()),
        })
    }

    /// Create an empty (zeroed) GPU buffer of a given size
    ///
    /// First attempts to acquire a buffer from the pool for reuse.
    /// If none available, allocates a new buffer with power-of-2 size
    /// for efficient pooling.
    #[must_use]
    pub fn zeros(len: usize) -> Option<Self> {
        let ctx = get_gpu_context()?;
        let byte_size = len * std::mem::size_of::<f32>();

        // Try to get a buffer from the pool first
        if let Some(buffer) = ctx.buffer_pool().acquire(byte_size) {
            // Zero-initialize the pooled buffer (only the portion we'll use)
            let zeros = vec![0.0f32; len];
            ctx.queue()
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&zeros));

            return Some(GpuBuffer {
                buffer: Some(buffer),
                len,
                pool: Some(ctx.buffer_pool_arc()),
            });
        }

        // Allocate with the bucket size (power of 2) for consistent pooling
        let alloc_size = BufferPool::allocation_size(byte_size);

        // Allocate a new buffer
        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Buffer (zeros)"),
            size: alloc_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero-initialize by writing zeros (only the portion we'll use)
        let zeros = vec![0.0f32; len];
        ctx.queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&zeros));

        Some(GpuBuffer {
            buffer: Some(buffer),
            len,
            pool: Some(ctx.buffer_pool_arc()),
        })
    }

    /// Copy data from GPU back to CPU
    ///
    /// This is a relatively expensive operation - try to minimize transfers!
    /// # Panics
    /// Panics if GPU context does not exist
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        let ctx = get_gpu_context().expect("GPU context should exist if buffer exists");
        let buffer = self.buffer.as_ref().expect("Buffer should not be taken");

        let byte_size = (self.len * std::mem::size_of::<f32>()) as u64;

        // Try to acquire staging buffer from pool first, create new if pool full
        // GPU buffers with STORAGE usage can't be mapped directly, so we need
        // a staging buffer with MAP_READ usage for GPU→CPU transfers
        let staging_buffer = ctx.staging_pool().acquire(byte_size).unwrap_or_else(|| {
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // Create a command encoder and copy data
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            (self.len * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the copy command
        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        // Note: We don't call maybe_sync() here because we're about to
        // block on device.poll() anyway to read the buffer back

        // Map the staging buffer and read the data
        let buffer_slice = staging_buffer.slice(..);

        // This is async, but we can synchronously block because the GPU feature pulls in pollster.

        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("map_async callback receiver dropped");
        });

        // Wait for the GPU to complete the mapping operation.
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
        // Read the data

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Unmap before returning to pool
        drop(data);
        staging_buffer.unmap();

        // Return staging buffer to pool for reuse
        // (if pool is full, buffer will be dropped and freed)
        ctx.staging_pool().release(staging_buffer, byte_size);

        result
    }

    /// Get the underlying wgpu buffer (for use in compute passes)
    /// # Panics
    /// Panics if buffer is taken
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("Buffer should not be taken")
    }

    /// Get the number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy a region of this buffer to a new GPU buffer
    ///
    /// # Arguments
    /// * `offset` - Starting element offset in this buffer
    /// * `len` - Number of elements to copy
    /// # Panics
    /// Panics if buffer is taken
    #[must_use]
    pub fn copy_region(&self, offset: usize, len: usize) -> Option<Self> {
        let ctx = get_gpu_context()?;
        let src_buffer = self.buffer.as_ref().expect("Buffer should not be taken");
        let byte_size = len * std::mem::size_of::<f32>();

        // Try to get a buffer from the pool first
        let new_buffer = if let Some(pooled) = ctx.buffer_pool().acquire(byte_size) {
            pooled
        } else {
            // Allocate with the bucket size (power of 2) for consistent pooling
            let alloc_size = BufferPool::allocation_size(byte_size);
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Copied Buffer Region"),
                size: alloc_size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Buffer Region Encoder"),
            });

        let byte_offset = (offset * std::mem::size_of::<f32>()) as u64;

        encoder.copy_buffer_to_buffer(src_buffer, byte_offset, &new_buffer, 0, byte_size as u64);

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(GpuBuffer {
            buffer: Some(new_buffer),
            len,
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
            let byte_size = self.len * std::mem::size_of::<f32>();
            // Try to return to pool; if pool is full, buffer is dropped normally
            pool.release(buffer, byte_size);
        }
        // If no pool reference or pool rejected, buffer drops normally
    }
}

// We need this trait for wgpu buffer initialization
// use wgpu::util::DeviceExt; //I think we still need this but I'm getting errors with it rn
