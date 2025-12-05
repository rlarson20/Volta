//! GPU buffer management
//!
//! GpuBuffer wraps a wgpu buffer and provides methods for
//! transferring data between CPU and GPU.

use super::get_gpu_context;
use std::sync::mpsc;
use std::time::Duration;

/// A buffer that lives on the GPU
///
/// This is analogous to a Vec<f32> but the data lives in GPU memory.
/// We need to explicitly copy data to/from the CPU.
pub struct GpuBuffer {
    /// The actual GPU buffer
    buffer: wgpu::Buffer,
    /// Size in number of f32 elements
    len: usize,
}

impl GpuBuffer {
    /// Create a new GPU buffer from CPU data
    ///
    /// This copies the data from CPU to GPU memory.
    pub fn from_slice(data: &[f32]) -> Option<Self> {
        let ctx = get_gpu_context()?;

        // Create a buffer with the STORAGE usage (for compute shaders)
        // and COPY_SRC/COPY_DST for data transfer
        let buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Some(GpuBuffer {
            buffer,
            len: data.len(),
        })
    }

    /// Create an empty (zeroed) GPU buffer of a given size
    pub fn zeros(len: usize) -> Option<Self> {
        let ctx = get_gpu_context()?;

        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Buffer (zeros)"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero-initialize by writing zeros
        let zeros = vec![0.0f32; len];
        ctx.queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&zeros));

        Some(GpuBuffer { buffer, len })
    }

    /// Copy data from GPU back to CPU
    ///
    /// This is a relatively expensive operation - try to minimize transfers!
    pub fn to_vec(&self) -> Vec<f32> {
        let ctx = get_gpu_context().expect("GPU context should exist if buffer exists");

        // Create a staging buffer for reading
        // GPU buffers with STORAGE usage can't be mapped directly
        let staging_buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a command encoder and copy data
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging_buffer,
            0,
            (self.len * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the copy command
        ctx.queue().submit(Some(encoder.finish()));

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

        // Unmap before returning
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Get the underlying wgpu buffer (for use in compute passes)
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// We need this trait for wgpu buffer initialization
use wgpu::util::DeviceExt;
