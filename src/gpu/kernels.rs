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

/// Parameters for reduction backward operations
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReduceBackwardParams {
    pub input_size: u32,
    pub op: u32, // 0 = sum, 1 = mean, 2 = max
    pub grad_value: f32,
    pub _padding: u32,
}

/// Parameters for optimizer step operations
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OptimizerStepParams {
    pub op: u32,           // 0=SGD, 1=SGD+momentum, 2=Adam
    pub lr: f32,           // Learning rate
    pub beta1: f32,        // Adam beta1 or momentum coefficient
    pub beta2: f32,        // Adam beta2
    pub t: f32,            // Timestep for bias correction (Adam)
    pub eps: f32,          // Adam epsilon
    pub weight_decay: f32, // L2 regularization
    pub padding: f32,
}

/// Parameters for movement operations
/// Must match the `MovementParams` struct in movement.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MovementParams {
    pub old_shape: [u32; 4],
    pub new_shape: [u32; 4],
    pub op_params: [u32; 4], // Operation-specific (permute axes, strides, etc.)
    pub rank: u32,
    pub padding2: u32, // Packed padding for dims 2,3
    pub padding: [u32; 2],
}

/// Parameters for im2col operation
/// Must match the `Im2colParams` struct in im2col.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Im2colParams {
    pub batch_size: u32,
    pub channels: u32,
    pub height: u32,
    pub width: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub _padding: [u32; 2],
}

/// Parameters for direct convolution operation
/// Must match the `DirectConvParams` struct in `direct_conv.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectConvParams {
    pub batch_size: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub height: u32,
    pub width: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub _padding: u32,
}

/// Parameters for binary backward operations with broadcasting
/// Must match the `BinaryBackwardParams` struct in `binary_backward.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryBackwardParams {
    pub out_shape: [u32; 4],
    pub a_shape: [u32; 4],
    pub b_shape: [u32; 4],
    pub out_rank: u32,
    pub a_rank: u32,
    pub b_rank: u32,
    pub _padding: u32,
}

/// Run a reduction operation that returns a scalar f32 value
fn reduce_scalar(input: &GpuBuffer, pipeline: &wgpu::ComputePipeline) -> Option<f32> {
    let ctx = get_gpu_context()?;
    let result_buffer = GpuBuffer::zeros(1)?;

    // Create uniform buffer with input size (pad to 32 bytes for wgpu min uniform buffer size)
    // Shader only reads first 4 u32s, but buffer must be at least 32 bytes
    let params = [input.len() as u32, 0, 0, 0];
    let mut padded_params = [0u32; 8];
    padded_params[..4].copy_from_slice(&params);
    let params_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&padded_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    // Create bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reduce Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create command encoder and dispatch
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reduce Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reduce Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
    ctx.increment_pending();
    ctx.maybe_sync();

    // Read back the result
    result_buffer.to_vec().first().copied()
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
    /// # Panics
    /// Mismatched buffer sizes
    #[must_use]
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
            "max" => &ctx.pipelines().max,
            "mod" => &ctx.pipelines().mod_op,
            "cmplt" => &ctx.pipelines().cmplt,
            _ => panic!("Unknown binary op: {op}"),
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
            let workgroup_count = (a.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit to GPU with throttling
        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Execute a unary operation
    /// # Panics
    /// Unknown unary op
    #[must_use]
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
            "recip" => &ctx.pipelines().recip,
            "exp2" => &ctx.pipelines().exp2,
            "log2" => &ctx.pipelines().log2,
            "sin" => &ctx.pipelines().sin,
            "cos" => &ctx.pipelines().cos,
            _ => panic!("Unknown unary op: {op}"),
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

            let workgroup_count = (input.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Execute a unary backward operation
    ///
    /// For y = f(x), computes grad = `out_grad` * df/dx
    ///
    /// # Arguments
    /// * `out_grad` - Gradient of the output (upstream gradient)
    /// * `x` - Input tensor from the forward pass
    /// * `op` - Which operation to perform (`"exp_backward"`, `"relu_backward"`, etc.)
    ///
    /// # Returns
    /// A new buffer containing the gradient with respect to x
    /// # Panics
    /// `out_grad` and `x` size mismatch
    #[must_use]
    pub fn unary_backward(out_grad: &GpuBuffer, x: &GpuBuffer, op: &str) -> Option<GpuBuffer> {
        assert_eq!(
            out_grad.len(),
            x.len(),
            "out_grad and x must have same size for unary backward"
        );

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(out_grad.len())?;

        let pipeline = match op {
            "neg_backward" => &ctx.pipelines().neg_backward,
            "exp_backward" => &ctx.pipelines().exp_backward,
            "log_backward" => &ctx.pipelines().log_backward,
            "relu_backward" => &ctx.pipelines().relu_backward,
            "sigmoid_backward" => &ctx.pipelines().sigmoid_backward,
            "tanh_backward" => &ctx.pipelines().tanh_backward,
            "sqrt_backward" => &ctx.pipelines().sqrt_backward,
            "recip_backward" => &ctx.pipelines().recip_backward,
            "exp2_backward" => &ctx.pipelines().exp2_backward,
            "log2_backward" => &ctx.pipelines().log2_backward,
            "sin_backward" => &ctx.pipelines().sin_backward,
            "cos_backward" => &ctx.pipelines().cos_backward,
            _ => panic!("Unknown unary backward op: {op}"),
        };

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Unary Backward Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Unary Backward Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Unary Backward Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (out_grad.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Execute a binary backward operation for gradient wrt first input (a)
    ///
    /// # Arguments
    /// * `out_grad` - Gradient of the output (upstream gradient)
    /// * `a` - First input from forward pass
    /// * `b` - Second input from forward pass (needed for mul/div/max)
    /// * `op` - Which operation (`"add_backward_a"`, `"mul_backward_a"`, etc.)
    ///
    /// # Returns
    /// A new buffer containing the gradient with respect to a
    /// # Panics
    /// `out_grad` and `a` shape mismatch
    #[must_use]
    pub fn binary_backward_a(
        out_grad: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        op: &str,
    ) -> Option<GpuBuffer> {
        assert_eq!(
            out_grad.len(),
            a.len(),
            "out_grad and a must have same size for binary backward"
        );
        assert_eq!(
            out_grad.len(),
            b.len(),
            "out_grad and b must have same size for binary backward"
        );

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(out_grad.len())?;

        let pipeline = match op {
            "add_backward_a" => &ctx.pipelines().add_backward_a,
            "sub_backward_a" => &ctx.pipelines().sub_backward_a,
            "mul_backward_a" => &ctx.pipelines().mul_backward_a,
            "div_backward_a" => &ctx.pipelines().div_backward_a,
            "max_backward_a" => &ctx.pipelines().max_backward_a,
            _ => panic!("Unknown binary backward op for a: {op}"),
        };

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Backward A Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Binary Backward A Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Backward A Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (out_grad.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Execute a binary backward operation for gradient wrt second input (b)
    ///
    /// # Arguments
    /// * `out_grad` - Gradient of the output (upstream gradient)
    /// * `a` - First input from forward pass (needed for mul/div/max)
    /// * `b` - Second input from forward pass
    /// * `op` - Which operation (`"add_backward_b"`, `"mul_backward_b"`, etc.)
    ///
    /// # Returns
    /// A new buffer containing the gradient with respect to b
    /// # Panics
    /// `out_grad` and `b` must have same shape
    #[must_use]
    pub fn binary_backward_b(
        out_grad: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        op: &str,
    ) -> Option<GpuBuffer> {
        assert_eq!(
            out_grad.len(),
            b.len(),
            "out_grad and b must have same size for binary backward"
        );
        assert_eq!(
            out_grad.len(),
            a.len(),
            "out_grad and a must have same size for binary backward"
        );

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(out_grad.len())?;

        let pipeline = match op {
            "add_backward_b" => &ctx.pipelines().add_backward_b,
            "sub_backward_b" => &ctx.pipelines().sub_backward_b,
            "mul_backward_b" => &ctx.pipelines().mul_backward_b,
            "div_backward_b" => &ctx.pipelines().div_backward_b,
            "max_backward_b" => &ctx.pipelines().max_backward_b,
            _ => panic!("Unknown binary backward op for b: {op}"),
        };

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Backward B Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Binary Backward B Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Backward B Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (out_grad.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Execute a binary backward operation with broadcasting support
    ///
    /// This computes gradients for both inputs simultaneously, handling broadcasting
    /// by reducing gradients over broadcasted dimensions.
    ///
    /// # Arguments
    /// * `out_grad` - Gradient of the output (upstream gradient)
    /// * `a` - First input from forward pass
    /// * `b` - Second input from forward pass
    /// * `op` - Which operation ("add", "sub", "mul", "div", "max")
    /// * `out_shape` - Shape of the output tensor
    /// * `a_shape` - Shape of the first input tensor
    /// * `b_shape` - Shape of the second input tensor
    ///
    /// # Returns
    /// A tuple of (gradient wrt a, gradient wrt b)
    /// # Panics
    /// Unknown binary backward op
    #[must_use]
    pub fn binary_backward_broadcast(
        out_grad: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        op: &str,
        out_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Option<(GpuBuffer, GpuBuffer)> {
        let ctx = get_gpu_context()?;

        // Create unified result buffer (concatenated a_grad and b_grad)
        // The shader uses atomic adds, so we need to zero-initialize
        let result_grad = GpuBuffer::zeros(a.len() + b.len())?;

        // Prepare shape parameters (pad to 4D)
        let mut out_shape_padded = [1u32; 4];
        let mut a_shape_padded = [1u32; 4];
        let mut b_shape_padded = [1u32; 4];

        let out_rank = out_shape.len() as u32;
        let a_rank = a_shape.len() as u32;
        let b_rank = b_shape.len() as u32;

        // Pad shapes to match (align to right for NumPy-style broadcasting)
        let out_offset = 4usize.saturating_sub(out_shape.len());
        let a_offset = 4usize.saturating_sub(a_shape.len());
        let b_offset = 4usize.saturating_sub(b_shape.len());

        for (i, &dim) in out_shape.iter().enumerate() {
            let idx = out_offset + i;
            if let Some(slot) = out_shape_padded.get_mut(idx) {
                *slot = dim as u32;
            }
        }
        for (i, &dim) in a_shape.iter().enumerate() {
            let idx = a_offset + i;
            if let Some(slot) = a_shape_padded.get_mut(idx) {
                *slot = dim as u32;
            }
        }
        for (i, &dim) in b_shape.iter().enumerate() {
            let idx = b_offset + i;
            if let Some(slot) = b_shape_padded.get_mut(idx) {
                *slot = dim as u32;
            }
        }

        let params = BinaryBackwardParams {
            out_shape: out_shape_padded,
            a_shape: a_shape_padded,
            b_shape: b_shape_padded,
            out_rank,
            a_rank,
            b_rank,
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Binary Backward Broadcast Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = match op {
            "add" => &ctx.pipelines().add_broadcast,
            "sub" => &ctx.pipelines().sub_broadcast,
            "mul" => &ctx.pipelines().mul_broadcast,
            "div" => &ctx.pipelines().div_broadcast,
            "max" => &ctx.pipelines().max_broadcast,
            _ => panic!("Unknown binary backward op: {op}"),
        };

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Backward Broadcast Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Binary Backward Broadcast Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Backward Broadcast Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (out_grad.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        // Split the concatenated result into two separate buffers
        let a_grad = result_grad.copy_region(0, a.len())?;
        let b_grad = result_grad.copy_region(a.len(), b.len())?;

        Some((a_grad, b_grad))
    }

    /// Execute a binary backward operation with RACE-FREE broadcasting support
    ///
    /// This uses a two-pass algorithm to avoid race conditions in gradient reduction:
    /// - Pass 1: Each thread scatters (`target_idx_a`, `target_idx_b`, `grad_value`) to temp buffer
    /// - Pass 2: Each thread reduces contributions for its gradient position
    ///
    /// # Arguments
    /// * `out_grad` - Gradient of the output (upstream gradient)
    /// * `a` - First input from forward pass
    /// * `b` - Second input from forward pass
    /// * `op` - Which operation ("add", "sub", "mul", "div", "max")
    /// * `out_shape` - Shape of the output tensor
    /// * `a_shape` - Shape of the first input tensor
    /// * `b_shape` - Shape of the second input tensor
    ///
    /// # Returns
    /// A tuple of (gradient wrt a, gradient wrt b)
    /// # Panics
    /// Unknown binary backward op
    #[must_use]
    pub fn binary_backward_broadcast_safe(
        out_grad: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        op: &str,
        out_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Option<(GpuBuffer, GpuBuffer)> {
        let ctx = get_gpu_context()?;

        // Calculate temporary buffer size
        // For add/sub: 3 floats per output (idx_a, idx_b, val)
        // For mul/div: 3 floats per output (idx_a, idx_b, val) - values read in pass2
        // For max: 5 floats per output (idx_a, idx_b, val, a_val, b_val)
        let temp_size = if op == "max" {
            out_grad.len() * 5
        } else {
            out_grad.len() * 3
        };
        let temp_buffer = GpuBuffer::zeros(temp_size)?;

        // Create unified result buffer (concatenated a_grad and b_grad)
        let result_grad = GpuBuffer::zeros(a.len() + b.len())?;

        // Prepare shape parameters (pad to 4D)
        let mut out_shape_padded = [1u32; 4];
        let mut a_shape_padded = [1u32; 4];
        let mut b_shape_padded = [1u32; 4];

        let out_rank = out_shape.len() as u32;
        let a_rank = a_shape.len() as u32;
        let b_rank = b_shape.len() as u32;

        // Pad shapes to match (align to right for NumPy-style broadcasting)
        let out_offset = 4usize.saturating_sub(out_shape.len());
        let a_offset = 4usize.saturating_sub(a_shape.len());
        let b_offset = 4usize.saturating_sub(b_shape.len());

        for (i, &dim) in out_shape.iter().enumerate() {
            let idx = out_offset + i;
            if let Some(slot) = out_shape_padded.get_mut(idx) {
                *slot = dim as u32;
            }
        }
        for (i, &dim) in a_shape.iter().enumerate() {
            let idx = a_offset + i;
            if let Some(slot) = a_shape_padded.get_mut(idx) {
                *slot = dim as u32;
            }
        }
        for (i, &dim) in b_shape.iter().enumerate() {
            let idx = b_offset + i;
            if let Some(slot) = b_shape_padded.get_mut(idx) {
                *slot = dim as u32;
            }
        }

        let params = BinaryBackwardParams {
            out_shape: out_shape_padded,
            a_shape: a_shape_padded,
            b_shape: b_shape_padded,
            out_rank,
            a_rank,
            b_rank,
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Binary Backward Safe Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Select pipelines for both passes
        let (pass1_pipeline, pass2_pipeline) = match op {
            "add" => (
                &ctx.pipelines().add_broadcast_pass1,
                &ctx.pipelines().add_broadcast_pass2,
            ),
            "sub" => (
                &ctx.pipelines().sub_broadcast_pass1,
                &ctx.pipelines().sub_broadcast_pass2,
            ),
            "mul" => (
                &ctx.pipelines().mul_broadcast_pass1,
                &ctx.pipelines().mul_broadcast_pass2,
            ),
            "div" => (
                &ctx.pipelines().div_broadcast_pass1,
                &ctx.pipelines().div_broadcast_pass2,
            ),
            "max" => (
                &ctx.pipelines().max_broadcast_pass1,
                &ctx.pipelines().max_broadcast_pass2,
            ),
            _ => panic!("Unknown binary backward op: {op}"),
        };

        // ============ PASS 1: Scatter ============
        let pass1_bind_group_layout = pass1_pipeline.get_bind_group_layout(0);
        let pass1_bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Backward Safe Pass1 Bind Group"),
            layout: &pass1_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Binary Backward Safe Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Backward Safe Pass1"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pass1_pipeline);
            compute_pass.set_bind_group(0, &pass1_bind_group, &[]);

            let workgroup_count = (out_grad.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // ============ PASS 2: Reduce ============
        let pass2_bind_group_layout = pass2_pipeline.get_bind_group_layout(0);
        let pass2_bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Backward Safe Pass2 Bind Group"),
            layout: &pass2_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Backward Safe Pass2"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pass2_pipeline);
            compute_pass.set_bind_group(0, &pass2_bind_group, &[]);

            let max_size = (a.len().max(b.len()) as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(max_size, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        // Split the concatenated result into two separate buffers
        let a_grad = result_grad.copy_region(0, a.len())?;
        let b_grad = result_grad.copy_region(a.len(), b.len())?;

        Some((a_grad, b_grad))
    }

    /// Matrix multiplication: C = A @ B
    ///
    /// # Arguments
    /// * `a` - Matrix A with shape (m, k)
    /// * `b` - Matrix B with shape (k, n)
    /// * `m` - Number of rows in A
    /// * `k` - Number of columns in A / rows in B
    /// * `n` - Number of columns in B
    /// # Panics
    /// Buffer size doesn't match dims
    #[must_use]
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
            let workgroups_x = (n as u32).div_ceil(16);
            let workgroups_y = (m as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Matrix multiplication backward pass: gradient with respect to A
    ///
    /// For C = A @ B where A: [M, K], B: [K, N], C: [M, N]:
    /// dC/dA = grad @ B^T  ->  [M, N] @ [N, K] = [M, K]
    ///
    /// # Arguments
    /// * `grad` - Output gradient with shape (m, n)
    /// * `b` - Matrix B from forward pass with shape (k, n)
    /// * `m` - Number of rows in grad / rows in dA
    /// * `k` - Number of columns in grad / rows in B
    /// * `n` - Number of columns in B / columns in grad
    /// # Panics
    /// Grad buffer size doesn't match dimensions
    #[must_use]
    pub fn matmul_backward_a(
        grad: &GpuBuffer,
        b: &GpuBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<GpuBuffer> {
        assert_eq!(
            grad.len(),
            m * n,
            "Grad buffer size doesn't match dimensions"
        );
        assert_eq!(b.len(), k * n, "B buffer size doesn't match dimensions");

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(m * k)?;

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
                label: Some("MatMul Backward A Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = &ctx.pipelines().matmul_backward_a;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Backward A Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad.buffer().as_entire_binding(),
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
                label: Some("MatMul Backward A Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Backward A Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Output is (m, k), dispatch one workgroup per 16x16 tile
            let workgroups_x = (k as u32).div_ceil(16);
            let workgroups_y = (m as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Matrix multiplication backward pass: gradient with respect to B
    ///
    /// For C = A @ B where A: [M, K], B: [K, N], C: [M, N]:
    /// dC/dB = A^T @ grad  ->  [K, M] @ [M, N] = [K, N]
    ///
    /// # Arguments
    /// * `a` - Matrix A from forward pass with shape (m, k)
    /// * `grad` - Output gradient with shape (m, n)
    /// * `m` - Number of rows in A / rows in grad
    /// * `k` - Number of columns in A / rows in dB
    /// * `n` - Number of columns in grad / columns in B
    /// # Panics
    /// `A` buffer size doesn't match dimensions
    #[must_use]
    pub fn matmul_backward_b(
        a: &GpuBuffer,
        grad: &GpuBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<GpuBuffer> {
        assert_eq!(a.len(), m * k, "A buffer size doesn't match dimensions");
        assert_eq!(
            grad.len(),
            m * n,
            "Grad buffer size doesn't match dimensions"
        );

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(k * n)?;

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
                label: Some("MatMul Backward B Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = &ctx.pipelines().matmul_backward_b;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Backward B Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer().as_entire_binding(),
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
                label: Some("MatMul Backward B Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Backward B Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Output is (k, n), dispatch one workgroup per 16x16 tile
            let workgroups_x = (n as u32).div_ceil(16);
            let workgroups_y = (k as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Sum all elements in a buffer
    #[must_use]
    pub fn sum(input: &GpuBuffer) -> Option<f32> {
        let ctx = get_gpu_context()?;
        reduce_scalar(input, &ctx.pipelines().sum_reduce)
    }

    /// Find the maximum value in a buffer
    #[must_use]
    pub fn max(input: &GpuBuffer) -> Option<f32> {
        let ctx = get_gpu_context()?;
        reduce_scalar(input, &ctx.pipelines().max_reduce)
    }

    /// Compute the mean of all elements in a buffer
    #[must_use]
    pub fn mean(input: &GpuBuffer) -> Option<f32> {
        let ctx = get_gpu_context()?;
        reduce_scalar(input, &ctx.pipelines().mean_reduce)
    }

    /// Sum backward: broadcast scalar gradient to all elements
    #[must_use]
    pub fn sum_backward(grad: f32, input_size: usize) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(input_size)?;

        let params = ReduceBackwardParams {
            input_size: input_size as u32,
            op: 0, // sum
            grad_value: grad,
            _padding: 0,
        };

        let grad_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Grad Buffer"),
                contents: bytemuck::bytes_of(&grad),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let max_indices_buffer =
            ctx.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Max Indices Buffer"),
                    contents: bytemuck::cast_slice(&[0u32; 1]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reduce Backward Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines().sum_backward;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sum Backward Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: max_indices_buffer.as_entire_binding(),
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
                label: Some("Sum Backward Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sum Backward Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (input_size as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Mean backward: broadcast scalar gradient / count to all elements
    #[must_use]
    pub fn mean_backward(grad: f32, input_size: usize) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(input_size)?;

        let params = ReduceBackwardParams {
            input_size: input_size as u32,
            op: 1, // mean
            grad_value: grad,
            _padding: 0,
        };

        let grad_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Grad Buffer"),
                contents: bytemuck::bytes_of(&grad),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let max_indices_buffer =
            ctx.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Max Indices Buffer"),
                    contents: bytemuck::cast_slice(&[0u32; 1]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reduce Backward Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines().mean_backward;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mean Backward Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: max_indices_buffer.as_entire_binding(),
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
                label: Some("Mean Backward Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mean Backward Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (input_size as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Max backward: sparse gradient - only max element receives gradient
    #[must_use]
    pub fn max_backward(grad: f32, input_size: usize, max_index: usize) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros(input_size)?;

        let params = ReduceBackwardParams {
            input_size: input_size as u32,
            op: 2, // max
            grad_value: grad,
            _padding: 0,
        };

        let grad_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Grad Buffer"),
                contents: bytemuck::bytes_of(&grad),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create max indices buffer
        let mut indices = vec![0u32; input_size.div_ceil(256)];
        if let Some(slot) = indices.get_mut(0) {
            *slot = max_index as u32;
        }
        let max_indices_buffer =
            ctx.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Max Indices Buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reduce Backward Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines().max_backward;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Max Backward Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: max_indices_buffer.as_entire_binding(),
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
                label: Some("Max Backward Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Max Backward Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (input_size as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Permute tensor axes
    #[must_use]
    pub fn permute(
        input: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        axes: &[usize],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = new_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(axes),
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(input, result, &params, &ctx.pipelines().permute)
    }

    /// Expand (broadcast) tensor to larger shape
    #[must_use]
    pub fn expand(
        input: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = new_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        // For expand, op_params represents which dimensions were broadcast (size 1 -> N)
        let mut broadcast_flags = [0u32; 4];
        for (i, (&old, &new)) in old_shape.iter().zip(new_shape).take(4).enumerate() {
            if let Some(slot) = broadcast_flags.get_mut(i) {
                *slot = u32::from(old == 1 && new > 1);
            }
        }

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: broadcast_flags,
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(input, result, &params, &ctx.pipelines().expand)
    }

    /// Pad tensor with zeros
    #[must_use]
    pub fn pad(
        input: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = new_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        // Pack padding into op_params for all dimensions (up to 4)
        // op_params: [left0, right0, left1, right1] for dims 0,1
        // padding2: [left2 | right2] for dim 2 (16 bits each)
        // _padding: [left3 | right3] for dim 3 (16 bits each)
        let mut pad_params = [0u32; 4];
        for (i, &(left, right)) in padding.iter().enumerate().take(2) {
            let idx = i * 2;
            if let Some(slot) = pad_params.get_mut(idx) {
                *slot = left as u32;
            }
            if let Some(slot) = pad_params.get_mut(idx + 1) {
                *slot = right as u32;
            }
        }

        // Pack dim 2 into padding2 (left in upper 16 bits, right in lower 16 bits)
        let padding2 = if let Some(&(left2, right2)) = padding.get(2) {
            (left2 as u32) << 16 | (right2 as u32)
        } else {
            0
        };

        // Pack dim 3 into _padding[0] (left in upper 16 bits, right in lower 16 bits)
        let padding3 = if let Some(&(left3, right3)) = padding.get(3) {
            (left3 as u32) << 16 | (right3 as u32)
        } else {
            0
        };

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_params,
            rank: old_shape.len() as u32,
            padding2,
            padding: [padding3, 0],
        };

        movement_op(input, result, &params, &ctx.pipelines().pad)
    }

    /// Shrink tensor to subregion
    #[must_use]
    pub fn shrink(
        input: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        ranges: &[(usize, usize)],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = new_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        // op_params contains range starts - convert to usize first
        let starts: Vec<usize> = ranges.iter().map(|(start, _)| *start).collect();

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(&starts),
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(input, result, &params, &ctx.pipelines().shrink)
    }

    /// Stride (subsample) tensor
    #[must_use]
    pub fn stride(
        input: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        strides: &[usize],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = new_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(strides),
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(input, result, &params, &ctx.pipelines().stride)
    }

    // ===== MOVEMENT BACKWARD OPERATIONS =====

    /// Permute backward: apply inverse permutation to gradient
    #[must_use]
    pub fn permute_backward(
        out_grad: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        axes: &[usize],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = old_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(axes),
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(out_grad, result, &params, &ctx.pipelines().permute_backward)
    }

    /// Expand backward: sum gradients over broadcast dimensions
    #[must_use]
    pub fn expand_backward(
        out_grad: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = old_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: [0, 0, 0, 0],
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(out_grad, result, &params, &ctx.pipelines().expand_backward)
    }

    /// Pad backward: extract center region from gradient (remove padding)
    #[must_use]
    pub fn pad_backward(
        out_grad: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = old_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        // Pack padding into params (same as forward pad operation)
        let mut op_params = [0u32; 4];
        let mut padding2 = 0u32;
        #[allow(clippy::used_underscore_binding)] //TEMP!!!
        let mut _padding = [0u32; 2];

        for (i, &(left, right)) in padding.iter().enumerate().take(4) {
            match i {
                0 => {
                    op_params[0] = left as u32;
                    op_params[1] = right as u32;
                }
                1 => {
                    op_params[2] = left as u32;
                    op_params[3] = right as u32;
                }
                2 => {
                    padding2 = ((left as u32) << 16) | (right as u32);
                }
                3 => {
                    _padding[0] = ((left as u32) << 16) | (right as u32);
                }
                _ => {}
            }
        }

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params,
            rank: old_shape.len() as u32,
            padding2,
            padding: _padding,
        };

        movement_op(out_grad, result, &params, &ctx.pipelines().pad_backward)
    }

    /// Shrink backward: pad gradient back to original size
    #[must_use]
    pub fn shrink_backward(
        out_grad: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        ranges: &[(usize, usize)],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = old_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        // Pack range starts into op_params
        let mut range_starts = [0usize; 4];
        for (i, &(start, _)) in ranges.iter().enumerate().take(4) {
            if let Some(slot) = range_starts.get_mut(i) {
                *slot = start;
            }
        }

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(&range_starts),
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(out_grad, result, &params, &ctx.pipelines().shrink_backward)
    }

    /// Stride backward: upsample gradient (inverse of striding/downsampling)
    #[must_use]
    pub fn stride_backward(
        out_grad: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        strides: &[usize],
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = old_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros(output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(strides),
            rank: old_shape.len() as u32,
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(out_grad, result, &params, &ctx.pipelines().stride_backward)
    }

    /// Optimizer step: update parameters using gradients and optimizer state
    ///
    /// Supports SGD, SGD with momentum, and Adam updates on GPU.
    ///
    /// # Arguments
    /// * `params` - Parameters to update (read/write)
    /// * `grads` - Gradients for each parameter (read)
    /// * `state1` - First state buffer: velocity (SGD) or m (Adam) (read/write)
    /// * `state2` - Second state buffer: v (Adam only), unused for SGD (read/write)
    /// * `opt_params` - Optimizer hyperparameters
    #[must_use]
    pub fn optimizer_step(
        params: &GpuBuffer,
        grads: &GpuBuffer,
        state1: &GpuBuffer,
        state2: &GpuBuffer,
        opt_params: &OptimizerStepParams,
    ) -> Option<()> {
        let ctx = get_gpu_context()?;

        // Create uniform buffer with optimizer parameters
        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Optimizer Params"),
                contents: bytemuck::bytes_of(opt_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines().optimizer_step;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Optimizer Step Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grads.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state1.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state2.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Optimizer Step Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Optimizer Step Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (params.len() as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(())
    }

    /// Image-to-column transformation for GPU convolution
    ///
    /// Transforms 4D input (B, C, H, W) into 2D matrix (B*`H_out`*`W_out`, C*`K_h`*`K_w`)
    /// where each row represents a flattened receptive field for one output position.
    ///
    /// # Arguments
    /// * `input` - 4D input tensor (batch, channels, height, width)
    /// * `batch_size` - Batch size (B)
    /// * `channels` - Number of input channels (C)
    /// * `height` - Input height (H)
    /// * `width` - Input width (W)
    /// * `kernel_h` - Kernel height
    /// * `kernel_w` - Kernel width
    /// * `stride_h` - Vertical stride
    /// * `stride_w` - Horizontal stride
    /// * `h_out` - Output height
    /// * `w_out` - Output width
    ///
    /// # Returns
    /// A 2D matrix of shape (B*`H_out`*`W_out`, C*`K_h`*`K_w`)
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn im2col(
        input: &GpuBuffer,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        h_out: usize,
        w_out: usize,
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;

        // Output matrix size: (B*H_out*W_out, C*K_h*K_w)
        let rows = batch_size * h_out * w_out;
        let cols = channels * kernel_h * kernel_w;
        let output_size = rows * cols;
        let result = GpuBuffer::zeros(output_size)?;

        // Create uniform buffer with im2col parameters
        let params = Im2colParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            kernel_h: kernel_h as u32,
            kernel_w: kernel_w as u32,
            stride_h: stride_h as u32,
            stride_w: stride_w as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,
            _padding: [0, 0],
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Im2col Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines().im2col;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Im2col Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Im2col Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Im2col Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Each workgroup handles one output row (one output position)
            let workgroup_count = (rows as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Direct convolution without im2col intermediate
    /// Each thread computes one output pixel
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn direct_conv(
        input: &GpuBuffer,
        weight: &GpuBuffer,
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        h_out: usize,
        w_out: usize,
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;

        // Output size: [B, out_channels, h_out, w_out]
        let output_size = batch_size * out_channels * h_out * w_out;
        let result = GpuBuffer::zeros(output_size)?;

        // Create uniform buffer with direct conv parameters
        let params = DirectConvParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            height: height as u32,
            width: width as u32,
            kernel_h: kernel_h as u32,
            kernel_w: kernel_w as u32,
            stride_h: stride_h as u32,
            stride_w: stride_w as u32,
            pad_h: pad_h as u32,
            pad_w: pad_w as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Direct Conv Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines().direct_conv;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Direct Conv Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer().as_entire_binding(),
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
                label: Some("Direct Conv Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Direct Conv Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup dimensions: (out_channels, batch * h_out * w_out, 1)
            // Each thread handles one output pixel
            let workgroups_x = (out_channels as u32).div_ceil(16);
            let workgroups_y = ((batch_size * h_out * w_out) as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }
}

/// Helper to pad a slice to exactly 4 u32 values
fn pad_to_u32_4(values: &[usize]) -> [u32; 4] {
    let mut result = [0u32; 4];
    for (i, &v) in values.iter().take(4).enumerate() {
        if let Some(slot) = result.get_mut(i) {
            *slot = v as u32;
        }
    }
    result
}

/// Run a movement operation with parameters
fn movement_op(
    input: &GpuBuffer,
    result: GpuBuffer,
    params: &MovementParams,
    pipeline: &wgpu::ComputePipeline,
) -> Option<GpuBuffer> {
    let ctx = get_gpu_context()?;

    // Create uniform buffer with parameters
    let params_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Movement Params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Movement Bind Group"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Movement Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Movement Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Only multiply actual dimensions (based on rank), not the padded zeros
        let output_size = (0..params.rank as usize)
            .filter_map(|i| params.new_shape.get(i))
            .product::<u32>();
        let workgroup_count = output_size.div_ceil(256);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
    ctx.increment_pending();
    ctx.maybe_sync();

    // Return the result buffer (now filled by GPU computation)
    Some(result)
}

use wgpu::util::DeviceExt;
