//! GPU kernel execution
//!
//! This module contains the logic for dispatching compute shaders.
//! Each operation creates a command buffer, binds the appropriate
//! pipeline and buffers, and submits work to the GPU.

use super::{GpuBuffer, get_gpu_context};

/// Parameters for matrix dimensions (used in matmul shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatMulParams {
    pub m: u32,        // Rows of A / rows of C
    pub k: u32,        // Cols of A / rows of B
    pub n: u32,        // Cols of B / cols of C
    pub _padding: u32, // Align to 16 bytes
}

/// Parameters for reduction operations
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReduceParams {
    pub input_size: u32,
    pub _padding: [u32; 3],
}

/// High-level interface for GPU kernel execution
pub struct GpuKernels;

impl GpuKernels {
    /// Execute an element-wise binary operation
    ///
    /// # Arguments
    /// * `a` - First input buffer
    /// * `b` - Second input buffer (must be same size as a)
    /// * `op` - Which operation to perform ("add", "sub", "mul", "div")
    ///
    /// # Returns
    /// A new buffer containing the result
    pub fn binary_op(a: &GpuBuffer, b: &GpuBuffer, op: &str) -> Option<GpuBuffer> {
        assert_eq!(a.len(), b.len(), "Buffer sizes must match for binary ops");

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(a.len())?;

        // Select the appropriate pipeline
        let pipeline = match op {
            "add" => &ctx.pipelines().add,
            "sub" => &ctx.pipelines().sub,
            "mul" => &ctx.pipelines().mul,
            "div" => &ctx.pipelines().div,
            _ => panic!("Unknown binary op: {}", op),
        };

        // Create bind group - this connects our buffers to the shader
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Op Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.buffer().as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Binary Op Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Op Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups
            // Each workgroup processes 256 elements (defined in shader)
            let workgroup_count = (a.len() as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit to GPU
        ctx.queue().submit(Some(encoder.finish()));

        Some(result)
    }

    /// Execute a unary operation
    pub fn unary_op(input: &GpuBuffer, op: &str) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(input.len())?;

        let pipeline = match op {
            "neg" => &ctx.pipelines().neg,
            "exp" => &ctx.pipelines().exp,
            "log" => &ctx.pipelines().log,
            "relu" => &ctx.pipelines().relu,
            "sigmoid" => &ctx.pipelines().sigmoid,
            "tanh" => &ctx.pipelines().tanh,
            "sqrt" => &ctx.pipelines().sqrt,
            _ => panic!("Unknown unary op: {}", op),
        };

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Unary Op Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Unary Op Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Unary Op Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (input.len() as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));

        Some(result)
    }

    /// Matrix multiplication: C = A @ B
    ///
    /// # Arguments
    /// * `a` - Matrix A with shape (m, k)
    /// * `b` - Matrix B with shape (k, n)
    /// * `m` - Number of rows in A
    /// * `k` - Number of columns in A / rows in B
    /// * `n` - Number of columns in B
    pub fn matmul(a: &GpuBuffer, b: &GpuBuffer, m: usize, k: usize, n: usize) -> Option<GpuBuffer> {
        assert_eq!(a.len(), m * k, "A buffer size doesn't match dimensions");
        assert_eq!(b.len(), k * n, "B buffer size doesn't match dimensions");

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(m * n)?;

        // Create uniform buffer for dimensions
        let params = MatMulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = &ctx.pipelines().matmul;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MatMul Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch one workgroup per 16x16 tile of the output
            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (m as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));

        Some(result)
    }

    /// Sum all elements in a buffer
    pub fn sum(input: &GpuBuffer) -> Option<f32> {
        // For simplicity, we'll do reduction on CPU for now
        // A proper GPU reduction is more complex (requires multiple passes)
        let data = input.to_vec();
        Some(data.iter().sum())
    }
}

use wgpu::util::DeviceExt;
