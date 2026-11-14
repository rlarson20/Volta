//! GPU kernel execution
//!
//! This module contains the logic for dispatching compute shaders.
//! Each operation creates a command buffer, binds the appropriate
//! pipeline and buffers, and submits work to the GPU.
use super::shader_template::Elem;
use super::{GpuBuffer, get_gpu_context};
use crate::dtype::DType;
use crate::index::slot;
use crate::ops::CmpOp;
use crate::ops::{BinaryOp, UnaryOp};
use crate::real::Real;
use crate::spec::PadValue;

/// Parameters for matrix dimensions (used in matmul shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatMulParams {
    /// Rows of `A` and of `C`.
    pub m: u32,
    /// Columns of `A` and rows of `B`.
    pub k: u32,
    /// Columns of `B` and of `C`.
    pub n: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Parameters for one pass of a reduction (`reduce.wgsl`)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReduceParams {
    /// Number of values this pass reads.
    pub n: u32,
    /// Each output of the pass is divided by this: 1.0, except the last pass of a mean.
    pub divisor: f32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: [u32; 2],
}

/// What [`GpuKernels::max`] leaves on the device.
pub struct MaxResult {
    /// The maximum, one element in the input's dtype.
    pub max: GpuBuffer,
    /// The maximum again, one f32 (NaN if the input held a NaN), for `max_backward`.
    pub value: GpuBuffer,
    /// How many elements equal the maximum (every NaN, for a NaN maximum), one `u32`.
    pub count: GpuBuffer,
}

/// Parameters for reduction backward operations
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReduceBackwardParams {
    /// Number of elements in the reduction input, which is the size of the gradient produced.
    pub input_size: u32,
    /// Reduction being differentiated: 0 = sum, 1 = mean (a max has its own entry point).
    pub op: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: [u32; 2],
}

/// Parameters for optimizer step operations
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OptimizerStepParams {
    /// Update rule: 0 = SGD, 1 = SGD with momentum, 2 = Adam.
    pub op: u32,
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient for op 1, or Adam's first-moment decay for op 2.
    pub beta1: f32,
    /// Adam's second-moment decay; unused by the SGD ops.
    pub beta2: f32,
    /// Step count for Adam's bias correction (`1 - beta^t`), held as an `f32`.
    pub t: f32,
    /// Adam's epsilon, added to the denominator.
    pub eps: f32,
    /// L2 coefficient: when nonzero, `weight_decay * param` is added to the gradient before the update.
    pub weight_decay: f32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub padding: f32,
}

/// Parameters for the Muon kernels
/// Must match the `MuonParams` struct in muon.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MuonParams {
    /// Element count
    n: u32,
    /// Momentum: 1 for the Nesterov update
    flag: u32,
    /// Momentum: coefficient; normalize: epsilon; axpby: scale of `x`
    alpha: f32,
    /// Axpby: scale of `y`
    beta: f32,
}

/// Parameters for movement operations
/// Must match the `MovementParams` struct in movement.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MovementParams {
    /// Shape of the input; only the first `rank` entries are read.
    pub old_shape: [u32; 4],
    /// Shape of the output; only the first `rank` entries are read.
    pub new_shape: [u32; 4],
    /// Per-operation values. Permute: the axis order. Shrink: the range start per axis. Stride: the stride per axis. Pad: `[left0, right0, left1, right1]` for dims 0 and 1.
    pub op_params: [u32; 4],
    /// Number of valid dimensions in the shapes, at most 4.
    pub rank: u32,
    /// Pad only: the left and right padding of dim 2, packed as `left << 16 | right` (16 bits each). Zero for other operations.
    pub padding2: u32,
    /// `padding[0]` packs the pad amounts of dim 3 the same way as `padding2`; `padding[1]` only keeps the struct 16-byte aligned. Zero for operations other than pad.
    pub padding: [u32; 2],
}

/// Parameters for im2col operation
/// Must match the `Im2colParams` struct in im2col.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Im2colParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Image channels `C`.
    pub channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: [u32; 2],
}

/// Parameters for col2im operation
/// Must match the `Col2imParams` struct in col2im.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Col2imParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Image channels `C`.
    pub channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: [u32; 2],
}

/// Parameters for direct convolution operation
/// Must match the `DirectConvParams` struct in `direct_conv.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectConvParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Input channels `C_in`.
    pub in_channels: u32,
    /// Output channels `C_out`.
    pub out_channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Zero padding added to each of the top and bottom of the input.
    pub pad_h: u32,
    /// Zero padding added to each of the left and right of the input.
    pub pad_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Parameters for convolution backward input operation
/// Must match the `ConvBackwardInputParams` struct in `conv_backward_input.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvBackwardInputParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Input channels `C_in`.
    pub in_channels: u32,
    /// Output channels `C_out`.
    pub out_channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Zero padding added to each of the top and bottom of the input.
    pub pad_h: u32,
    /// Zero padding added to each of the left and right of the input.
    pub pad_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Parameters for iGEMM convolution operation
/// Must match the `IGEMMParams` struct in `igemm.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IGEMMParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Input channels `C_in`.
    pub in_channels: u32,
    /// Output channels `C_out`.
    pub out_channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Zero padding added to each of the top and bottom of the input.
    pub pad_h: u32,
    /// Zero padding added to each of the left and right of the input.
    pub pad_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    // Tile configuration
    /// Output channels per tile.
    pub oc_tile: u32,
    /// Input channels per tile.
    pub ic_tile: u32,
    /// Spatial positions per tile.
    pub s_tile: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Parameters for convolution backward weight operation
/// Must match the `ConvBackwardWeightParams` struct in `conv_backward_weight.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvBackwardWeightParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Input channels `C_in`.
    pub in_channels: u32,
    /// Output channels `C_out`.
    pub out_channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Zero padding added to each of the top and bottom of the input.
    pub pad_h: u32,
    /// Zero padding added to each of the left and right of the input.
    pub pad_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Parameters for iGEMM convolution backward input operation
/// Must match the `IGEMMBackwardInputParams` struct in `igemm_backward_input.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IGEMMBackwardInputParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Input channels `C_in`.
    pub in_channels: u32,
    /// Output channels `C_out`.
    pub out_channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Zero padding added to each of the top and bottom of the input.
    pub pad_h: u32,
    /// Zero padding added to each of the left and right of the input.
    pub pad_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Parameters for iGEMM convolution backward weight operation
/// Must match the `IGEMMBackwardWeightParams` struct in `igemm_backward_weight.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IGEMMBackwardWeightParams {
    /// Batch size `N`.
    pub batch_size: u32,
    /// Input channels `C_in`.
    pub in_channels: u32,
    /// Output channels `C_out`.
    pub out_channels: u32,
    /// Input height `H`.
    pub height: u32,
    /// Input width `W`.
    pub width: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride along the height.
    pub stride_h: u32,
    /// Stride along the width.
    pub stride_w: u32,
    /// Zero padding added to each of the top and bottom of the input.
    pub pad_h: u32,
    /// Zero padding added to each of the left and right of the input.
    pub pad_w: u32,
    /// Output height, computed on the host.
    pub h_out: u32,
    /// Output width, computed on the host.
    pub w_out: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Parameters for binary backward operations with broadcasting
/// Must match the `BinaryBackwardParams` struct in `binary_backward.wgsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryBackwardParams {
    /// Broadcast output shape; only the first `out_rank` entries are read.
    pub out_shape: [u32; 4],
    /// Shape of the first input; only the first `a_rank` entries are read.
    pub a_shape: [u32; 4],
    /// Shape of the second input; only the first `b_rank` entries are read.
    pub b_shape: [u32; 4],
    /// Rank of the output.
    pub out_rank: u32,
    /// Rank of the first input.
    pub a_rank: u32,
    /// Rank of the second input.
    pub b_rank: u32,
    /// Pads the struct to a multiple of 16 bytes, as a WGSL uniform requires; always zero.
    pub _padding: u32,
}

/// Which full-tensor reduction [`reduce_to_scalar`] runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reduction {
    /// Sum of the elements.
    Sum,
    /// Sum divided by the element count (the division is part of the last pass).
    Mean,
    /// Sum of the squares of the elements.
    SumSquares,
    /// Maximum, with the number of elements equal to it.
    Max,
}

/// Elements one workgroup of `reduce.wgsl` consumes per pass (256 threads x 4).
const REDUCE_PER_GROUP: usize = 1024;

/// Reduce `input` to one f32 on the device, in as many passes as it takes
/// (`ceil(n / 1024)` partials per pass, until one is left), all in one command buffer.
///
/// Returns the f32 result as a one-element buffer, and for [`Reduction::Max`] the number of
/// elements equal to the maximum (every NaN, when it is NaN) as a one-element buffer of
/// `u32` bits (typed Bool, one word per element, like the `found` flag of `amp_unscale`).
/// Nothing is read back.
///
/// # Panics
/// If `input` is empty, or has too many elements for a pass's offsets to fit in a `u32`.
fn reduce_to_scalar(input: &GpuBuffer, kind: Reduction) -> Option<(GpuBuffer, Option<GpuBuffer>)> {
    let ctx = get_gpu_context()?;
    let pipelines = ctx.pipelines_for(input.dtype());
    let (first, next) = match kind {
        Reduction::Sum | Reduction::Mean => (&pipelines.sum_first, &pipelines.sum_next),
        Reduction::SumSquares => (&pipelines.sumsq_first, &pipelines.sum_next),
        Reduction::Max => (&pipelines.max_first, &pipelines.max_next),
    };
    let total = input.len();
    assert!(total > 0, "GPU reduction of an empty buffer");
    // A workgroup's first offset, `group * 1024`, is below `n + 1024`.
    pack_u32(total + REDUCE_PER_GROUP);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reduce Encoder"),
        });
    let mut n = total;
    // The previous pass's partial values and (for max) counts; `None` on the first pass.
    let mut previous: Option<(GpuBuffer, Option<GpuBuffer>)> = None;
    // Earlier passes' partials. Dropping a `GpuBuffer` returns it to the shared pool at once,
    // where another thread can take it and write to it before this encoder is submitted, so
    // they are held until after the submit (#244).
    let mut retired: Vec<(GpuBuffer, Option<GpuBuffer>)> = Vec::new();
    loop {
        let groups = n.div_ceil(REDUCE_PER_GROUP);
        let out = GpuBuffer::zeros(groups)?;
        let out_count = match kind {
            Reduction::Max => Some(GpuBuffer::zeros_of(DType::Bool, groups)?),
            Reduction::Sum | Reduction::Mean | Reduction::SumSquares => None,
        };
        let divisor = if kind == Reduction::Mean && groups == 1 {
            <f32 as Real>::from_usize(total)
        } else {
            1.0
        };
        let params = ReduceParams {
            n: pack_u32(n),
            divisor,
            _padding: [0; 2],
        };
        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reduce Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Exactly the bindings the entry point uses (the pipelines use an auto layout).
        let pipeline = if previous.is_none() { first } else { next };
        let mut bindings: Vec<(u32, &wgpu::Buffer)> = Vec::with_capacity(5);
        match &previous {
            None => bindings.push((0, input.buffer())),
            Some((values, counts)) => {
                bindings.push((1, values.buffer()));
                if let Some(counts) = counts {
                    bindings.push((3, counts.buffer()));
                }
            }
        }
        bindings.push((2, out.buffer()));
        if let Some(count) = &out_count {
            bindings.push((4, count.buffer()));
        }
        bindings.push((5, &params_buffer));
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduce Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reduce Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            dispatch_flat(&mut pass, pack_u32(groups));
        }

        retired.extend(previous.replace((out, out_count)));
        if groups == 1 {
            break;
        }
        n = groups;
    }

    ctx.queue().submit(Some(encoder.finish()));
    drop(retired);
    ctx.increment_pending();
    ctx.maybe_sync();
    previous
}

/// What `max_backward` reads beside the upstream gradient: the saved input, and the
/// maximum with the number of elements equal to it, both from [`GpuKernels::max`].
struct MaxSaved<'a> {
    /// The reduced input, in the gradient's dtype.
    input: &'a GpuBuffer,
    /// The maximum, one f32 (NaN if the input held a NaN).
    value: &'a GpuBuffer,
    /// How many elements are a maximum, one `u32`.
    count: &'a GpuBuffer,
}

/// Broadcast a scalar gradient back to `input_size` elements: `op` 0 = sum, 1 = mean
/// (divided by the count), or, with `max` given, the max rule (the gradient divided by the
/// number of maximum elements, and the others zero).
///
/// The gradient is read on the device (`out_grad` at binding 0), never fetched to the host.
/// The result has `out_grad`'s dtype.
fn reduce_backward(
    op: u32,
    out_grad: &GpuBuffer,
    input_size: usize,
    max: Option<&MaxSaved>,
) -> Option<GpuBuffer> {
    let ctx = get_gpu_context()?;
    let dtype = out_grad.dtype();
    let result = GpuBuffer::zeros_of(dtype, input_size)?;
    let params = ReduceBackwardParams {
        input_size: pack_u32(input_size),
        op,
        _padding: [0; 2],
    };
    let params_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reduce Backward Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let pipelines = ctx.pipelines_for(dtype);
    let pipeline = match (op, max) {
        (_, Some(_)) => &pipelines.max_backward,
        (0, None) => &pipelines.sum_backward,
        (_, None) => &pipelines.mean_backward,
    };
    // Exactly the bindings the entry point uses (the pipelines use an auto layout).
    let mut bindings: Vec<(u32, &wgpu::Buffer)> = vec![
        (0, out_grad.buffer()),
        (2, result.buffer()),
        (3, &params_buffer),
    ];
    if let Some(saved) = max {
        assert_same_dtype("max_backward", &[out_grad, saved.input]);
        bindings.push((1, saved.count.buffer()));
        bindings.push((4, saved.input.buffer()));
        bindings.push((5, saved.value.buffer()));
    }
    let entries: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .map(|&(binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reduce Backward Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reduce Backward Encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reduce Backward Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_flat(&mut pass, pack_u32(input_size).div_ceil(256));
    }
    ctx.queue().submit(Some(encoder.finish()));
    ctx.increment_pending();
    ctx.maybe_sync();
    Some(result)
}

/// Panics unless every buffer in `buffers` has one dtype: a kernel's pipeline is chosen
/// by its first input's element type, so another type bound beside it would be read with
/// the wrong layout.
#[track_caller]
fn assert_same_dtype(kernel: &str, buffers: &[&GpuBuffer]) {
    let first = buffers.first().map(|b| b.dtype());
    for b in buffers {
        assert!(
            Some(b.dtype()) == first,
            "GPU {kernel}: a {} buffer beside a {:?} one (a kernel binds one element type)",
            b.dtype(),
            first
        );
    }
}

/// The `{ n, pad, pad, pad }` uniform of the cast, compare and where shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CountParams {
    /// Elements to process.
    n: u32,
    /// Padding to 16 bytes.
    pad: [u32; 3],
}

/// Run `pipeline` over `n` elements, `buffers` bound at 0, 1, .. and a [`CountParams`]
/// uniform at the next binding.
fn dispatch_counted(
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&GpuBuffer],
    n: usize,
) -> Option<()> {
    let ctx = get_gpu_context()?;
    let params = CountParams {
        n: pack_u32(n),
        pad: [0; 3],
    };
    let params_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Count Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let mut entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .zip(0u32..)
        .map(|(b, binding)| wgpu::BindGroupEntry {
            binding,
            resource: b.buffer().as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: u32::try_from(buffers.len()).expect("a few bindings"),
        resource: params_buffer.as_entire_binding(),
    });
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Counted Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Counted Encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Counted Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_flat(&mut pass, workgroups_for(n));
    }
    ctx.queue().submit(Some(encoder.finish()));
    ctx.increment_pending();
    ctx.maybe_sync();
    Some(())
}

/// High-level interface for GPU kernel execution
pub struct GpuKernels;

impl GpuKernels {
    /// `input` converted to `target` on the device: between F32, F16 and BF16 (rounded to
    /// nearest-even), and between those and Bool (nonzero, NaN included, is true).
    ///
    /// # Panics
    /// If `target` or the input's dtype is not one the GPU holds, or both are the same.
    #[must_use]
    pub(crate) fn cast(input: &GpuBuffer, target: DType) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let source = input.dtype();
        assert_ne!(source, target, "GPU cast to the same dtype");
        let (pipeline, n) = match (source, target) {
            (DType::Bool, t) => (
                ctx.cast_pipeline(Elem::F32, ctx.elem_for(t), "from_bool"),
                input.len(),
            ),
            (s, DType::Bool) => (
                ctx.cast_pipeline(ctx.elem_for(s), Elem::F32, "to_bool"),
                input.len(),
            ),
            (s, t) => (
                ctx.cast_pipeline(ctx.elem_for(s), ctx.elem_for(t), "cast_elements"),
                input.len(),
            ),
        };
        let out = GpuBuffer::zeros_of(target, n)?;
        dispatch_counted(&pipeline, &[input, &out], n)?;
        Some(out)
    }

    /// `grad * inv` in `grad`'s type, and whether any rounded result is inf or NaN
    /// (`GradScaler::unscale`). One `u32` is read back.
    #[must_use]
    pub(crate) fn amp_unscale(grad: &GpuBuffer, inv: f32) -> Option<(GpuBuffer, bool)> {
        let ctx = get_gpu_context()?;
        let out = GpuBuffer::zeros_of(grad.dtype(), grad.len())?;
        let found = GpuBuffer::zeros_of(DType::Bool, 1)?;
        let params = [pack_u32(grad.len()), inv.to_bits(), 0, 0];
        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Unscale Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let pipeline = &ctx.pipelines_for(grad.dtype()).amp_unscale;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Unscale Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: found.buffer().as_entire_binding(),
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
                label: Some("Unscale Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Unscale Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            dispatch_flat(&mut pass, workgroups_for(grad.len()));
        }
        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();
        let crate::storage::HostBuf::Bool(flag) = found.to_host() else {
            unreachable!("a Bool buffer downloads as Bool")
        };
        Some((out, *flag.first().expect("a one-element flag buffer")))
    }

    /// `a <op> b` for equal-length buffers of one float dtype, as a Bool buffer.
    ///
    /// # Panics
    /// If the lengths or dtypes differ.
    #[must_use]
    pub(crate) fn compare(a: &GpuBuffer, b: &GpuBuffer, op: CmpOp) -> Option<GpuBuffer> {
        assert_eq!(a.len(), b.len(), "GPU compare: operand lengths");
        assert_eq!(a.dtype(), b.dtype(), "GPU compare: operand dtypes");
        let ctx = get_gpu_context()?;
        let p = ctx.pipelines_for(a.dtype());
        let pipeline = match op {
            CmpOp::Lt => &p.cmp_lt,
            CmpOp::Le => &p.cmp_le,
            CmpOp::Gt => &p.cmp_gt,
            CmpOp::Ge => &p.cmp_ge,
            CmpOp::Eq => &p.cmp_eq,
            CmpOp::Ne => &p.cmp_ne,
        };
        let out = GpuBuffer::zeros_of(DType::Bool, a.len())?;
        dispatch_counted(pipeline, &[a, b, &out], a.len())?;
        Some(out)
    }

    /// `where(cond, x, y)` for equal lengths: `cond` Bool, `x` and `y` one float dtype.
    ///
    /// # Panics
    /// If the lengths or dtypes don't fit.
    #[must_use]
    pub(crate) fn where_select(
        cond: &GpuBuffer,
        x: &GpuBuffer,
        y: &GpuBuffer,
    ) -> Option<GpuBuffer> {
        assert_eq!(cond.dtype(), DType::Bool, "GPU where: Bool condition");
        assert_eq!(x.dtype(), y.dtype(), "GPU where: branch dtypes");
        assert!(
            cond.len() == x.len() && x.len() == y.len(),
            "GPU where: lengths"
        );
        let ctx = get_gpu_context()?;
        let out = GpuBuffer::zeros_of(x.dtype(), x.len())?;
        dispatch_counted(
            &ctx.pipelines_for(x.dtype()).where_select,
            &[cond, x, y, &out],
            x.len(),
        )?;
        Some(out)
    }

    /// Execute an element-wise binary operation
    ///
    /// # Arguments
    /// * `a` - First input buffer
    /// * `b` - Second input buffer (must be same size as a)
    /// * `op` - Which operation to perform
    ///
    /// # Returns
    /// A new buffer containing the result
    /// # Panics
    /// Mismatched buffer sizes
    #[must_use]
    pub fn binary_op(a: &GpuBuffer, b: &GpuBuffer, op: BinaryOp) -> Option<GpuBuffer> {
        assert_same_dtype("binary_op", &[a, b]);
        assert_eq!(a.len(), b.len(), "Buffer sizes must match for binary ops");

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros_of(a.dtype(), a.len())?;

        // Select the appropriate pipeline
        let pipeline = op.gpu_forward(ctx.pipelines_for(a.dtype()));

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
            let workgroup_count = pack_u32(a.len()).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
        }

        // Submit to GPU with throttling
        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Execute a unary operation
    ///
    /// Returns `None` if the op is `cpu_only` (no GPU kernel) or no GPU is available.
    #[must_use]
    pub fn unary_op(input: &GpuBuffer, op: UnaryOp) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let (pipeline, _) = op.gpu_pipelines(ctx.pipelines_for(input.dtype()))?;
        let result = GpuBuffer::zeros_of(input.dtype(), input.len())?;

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

            let workgroup_count = pack_u32(input.len()).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
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
    /// * `op` - Which operation's derivative to apply
    ///
    /// # Returns
    /// A new buffer containing the gradient with respect to x, or `None` if the op is
    /// `cpu_only` or no GPU is available.
    /// # Panics
    /// `out_grad` and `x` size mismatch
    #[must_use]
    pub fn unary_backward(out_grad: &GpuBuffer, x: &GpuBuffer, op: UnaryOp) -> Option<GpuBuffer> {
        assert_same_dtype("unary_backward", &[out_grad, x]);
        assert_eq!(
            out_grad.len(),
            x.len(),
            "out_grad and x must have same size for unary backward"
        );

        let ctx = get_gpu_context()?;
        let (_, pipeline) = op.gpu_pipelines(ctx.pipelines_for(out_grad.dtype()))?;
        let result = GpuBuffer::zeros_of(out_grad.dtype(), out_grad.len())?;

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

            let workgroup_count = pack_u32(out_grad.len()).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
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
    /// * `op` - Which operation; returns `None` if it is non-differentiable
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
        op: BinaryOp,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("binary_backward_a", &[out_grad, a, b]);
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
        let result = GpuBuffer::zeros_of(out_grad.dtype(), out_grad.len())?;

        let pipeline = op.gpu_backward(ctx.pipelines_for(out_grad.dtype()))?.a;

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

            let workgroup_count = pack_u32(out_grad.len()).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
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
    /// * `op` - Which operation; returns `None` if it is non-differentiable
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
        op: BinaryOp,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("binary_backward_b", &[out_grad, a, b]);
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
        let result = GpuBuffer::zeros_of(out_grad.dtype(), out_grad.len())?;

        let pipeline = op.gpu_backward(ctx.pipelines_for(out_grad.dtype()))?.b;

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

            let workgroup_count = pack_u32(out_grad.len()).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
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
    /// * `op` - Which operation; returns `None` if it is non-differentiable
    /// * `out_shape` - Shape of the output tensor
    /// * `a_shape` - Shape of the first input tensor
    /// * `b_shape` - Shape of the second input tensor
    ///
    /// # Returns
    /// A tuple of (gradient wrt a, gradient wrt b)
    #[must_use]
    pub fn binary_backward_broadcast_safe(
        out_grad: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        op: BinaryOp,
        out_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Option<(GpuBuffer, GpuBuffer)> {
        assert_same_dtype("binary_backward_broadcast_safe", &[out_grad, a, b]);
        let ctx = get_gpu_context()?;

        // Calculate temporary buffer size
        // For add/sub: 3 floats per output (idx_a, idx_b, val)
        // For mul/div: 3 floats per output (idx_a, idx_b, val) - values read in pass2
        // For max: 5 floats per output (idx_a, idx_b, val, a_val, b_val)
        let temp_size = if op == BinaryOp::Max {
            out_grad.len() * 5
        } else {
            out_grad.len() * 3
        };
        let temp_buffer = GpuBuffer::zeros_of(out_grad.dtype(), temp_size)?;

        // Create unified result buffer (concatenated a_grad and b_grad)
        let result_grad = GpuBuffer::zeros_of(out_grad.dtype(), a.len() + b.len())?;

        // Prepare shape parameters (pad to 4D)
        let mut out_shape_padded = [1u32; 4];
        let mut a_shape_padded = [1u32; 4];
        let mut b_shape_padded = [1u32; 4];

        let out_rank = pack_u32(out_shape.len());
        let a_rank = pack_u32(a_shape.len());
        let b_rank = pack_u32(b_shape.len());

        // Pad shapes to match (align to right for NumPy-style broadcasting)
        let out_offset = 4usize.saturating_sub(out_shape.len());
        let a_offset = 4usize.saturating_sub(a_shape.len());
        let b_offset = 4usize.saturating_sub(b_shape.len());

        for (i, &dim) in out_shape.iter().enumerate() {
            let idx = out_offset + i;
            *slot(&mut out_shape_padded, idx) = pack_u32(dim);
        }
        for (i, &dim) in a_shape.iter().enumerate() {
            let idx = a_offset + i;
            *slot(&mut a_shape_padded, idx) = pack_u32(dim);
        }
        for (i, &dim) in b_shape.iter().enumerate() {
            let idx = b_offset + i;
            *slot(&mut b_shape_padded, idx) = pack_u32(dim);
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
        let backward = op.gpu_backward(ctx.pipelines_for(out_grad.dtype()))?;
        let (pass1_pipeline, pass2_pipeline) = (backward.pass1, backward.pass2);

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

            let workgroup_count = pack_u32(out_grad.len()).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
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

            let max_size = pack_u32(a.len().max(b.len())).div_ceil(256);
            dispatch_flat(&mut compute_pass, max_size);
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
    /// * `spec` - Validated `m x k @ k x n` geometry; must not be batched
    /// # Panics
    /// If `spec` is batched, or a buffer size doesn't match dims
    #[must_use]
    pub(crate) fn matmul(
        a: &GpuBuffer,
        b: &GpuBuffer,
        spec: &crate::spec::MatmulSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("matmul", &[a, b]);
        assert!(
            !spec.is_batched(),
            "GPU matmul kernels are 2D only; batched products run on the CPU path"
        );
        let (m, k, n) = (spec.m, spec.k, spec.n);
        assert_eq!(a.len(), m * k, "A buffer size doesn't match dimensions");
        assert_eq!(b.len(), k * n, "B buffer size doesn't match dimensions");

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros_of(a.dtype(), m * n)?;

        // Create uniform buffer for dimensions
        let params = MatMulParams {
            m: pack_u32(m),
            k: pack_u32(k),
            n: pack_u32(n),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = &ctx.pipelines_for(a.dtype()).matmul;
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
            let workgroups_x = pack_u32(n).div_ceil(16);
            let workgroups_y = pack_u32(m).div_ceil(16);
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
    /// * `spec` - Validated `m x k @ k x n` geometry; must not be batched
    /// # Panics
    /// Grad buffer size doesn't match dimensions
    #[must_use]
    pub(crate) fn matmul_backward_a(
        grad: &GpuBuffer,
        b: &GpuBuffer,
        spec: &crate::spec::MatmulSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("matmul_backward_a", &[grad, b]);
        assert!(
            !spec.is_batched(),
            "GPU matmul kernels are 2D only; batched products run on the CPU path"
        );
        let (m, k, n) = (spec.m, spec.k, spec.n);
        assert_eq!(
            grad.len(),
            m * n,
            "Grad buffer size doesn't match dimensions"
        );
        assert_eq!(b.len(), k * n, "B buffer size doesn't match dimensions");

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros_of(grad.dtype(), m * k)?;

        // Create uniform buffer for dimensions
        let params = MatMulParams {
            m: pack_u32(m),
            k: pack_u32(k),
            n: pack_u32(n),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul Backward A Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = &ctx.pipelines_for(grad.dtype()).matmul_backward_a;
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
            let workgroups_x = pack_u32(k).div_ceil(16);
            let workgroups_y = pack_u32(m).div_ceil(16);
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
    /// * `spec` - Validated `m x k @ k x n` geometry; must not be batched
    /// # Panics
    /// `A` buffer size doesn't match dimensions
    #[must_use]
    pub(crate) fn matmul_backward_b(
        a: &GpuBuffer,
        grad: &GpuBuffer,
        spec: &crate::spec::MatmulSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("matmul_backward_b", &[a, grad]);
        assert!(
            !spec.is_batched(),
            "GPU matmul kernels are 2D only; batched products run on the CPU path"
        );
        let (m, k, n) = (spec.m, spec.k, spec.n);
        assert_eq!(a.len(), m * k, "A buffer size doesn't match dimensions");
        assert_eq!(
            grad.len(),
            m * n,
            "Grad buffer size doesn't match dimensions"
        );

        let ctx = get_gpu_context()?;
        let result = GpuBuffer::zeros_of(a.dtype(), k * n)?;

        // Create uniform buffer for dimensions
        let params = MatMulParams {
            m: pack_u32(m),
            k: pack_u32(k),
            n: pack_u32(n),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul Backward B Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = &ctx.pipelines_for(a.dtype()).matmul_backward_b;
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
            let workgroups_x = pack_u32(n).div_ceil(16);
            let workgroups_y = pack_u32(k).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Sum all elements of a buffer, as a one-element buffer of the input's dtype. The
    /// sum is never read back: a gradient that flows from it stays on the device.
    #[must_use]
    pub fn sum(input: &GpuBuffer) -> Option<GpuBuffer> {
        let (value, _) = reduce_to_scalar(input, Reduction::Sum)?;
        Self::round_to(value, input.dtype())
    }

    /// The maximum of a buffer, as `torch.max()` (NaN if any element is NaN), with what
    /// backward needs: a one-element buffer of the input's dtype, a copy of the maximum in
    /// f32, and the number of elements equal to it (every NaN, for a NaN maximum) as a
    /// one-element `u32` buffer. The last two are not tensors; they are only ever bound to
    /// `max_backward`. The rule is the single `combine` function in `reduce.wgsl`.
    #[must_use]
    pub fn max(input: &GpuBuffer) -> Option<MaxResult> {
        let (value, count) = reduce_to_scalar(input, Reduction::Max)?;
        // `round_to` hands an f32 result on as it is, so backward keeps its own copy.
        let kept = value.copy_region(0, 1)?;
        Some(MaxResult {
            max: Self::round_to(value, input.dtype())?,
            value: kept,
            count: count?,
        })
    }

    /// The mean of all elements of a buffer, as a one-element buffer of the input's dtype.
    #[must_use]
    pub fn mean(input: &GpuBuffer) -> Option<GpuBuffer> {
        let (value, _) = reduce_to_scalar(input, Reduction::Mean)?;
        Self::round_to(value, input.dtype())
    }

    /// The sum of squares of an f32 buffer, as a one-element f32 buffer (the Muon
    /// Frobenius norm).
    ///
    /// # Panics
    /// If `input` is not f32.
    #[must_use]
    pub(crate) fn sum_squares(input: &GpuBuffer) -> Option<GpuBuffer> {
        assert_eq!(input.dtype(), DType::F32, "sum_squares: an f32 buffer");
        Some(reduce_to_scalar(input, Reduction::SumSquares)?.0)
    }

    /// An f32 reduction result rounded once to `dtype`, as the CPU rounds a half
    /// reduction.
    fn round_to(value: GpuBuffer, dtype: DType) -> Option<GpuBuffer> {
        if dtype == DType::F32 {
            Some(value)
        } else {
            Self::cast(&value, dtype)
        }
    }

    /// Sum backward: broadcast the scalar gradient `out_grad` to `input_size` elements.
    #[must_use]
    pub fn sum_backward(out_grad: &GpuBuffer, input_size: usize) -> Option<GpuBuffer> {
        reduce_backward(0, out_grad, input_size, None)
    }

    /// Mean backward: broadcast the scalar gradient / count to all elements.
    #[must_use]
    pub fn mean_backward(out_grad: &GpuBuffer, input_size: usize) -> Option<GpuBuffer> {
        reduce_backward(1, out_grad, input_size, None)
    }

    /// Max backward: the scalar gradient `out_grad`, split evenly over the elements of
    /// `input` equal to the maximum (`value` and `count` from [`Self::max`]); the rest get
    /// zero.
    #[must_use]
    pub fn max_backward(
        out_grad: &GpuBuffer,
        input_size: usize,
        input: &GpuBuffer,
        value: &GpuBuffer,
        count: &GpuBuffer,
    ) -> Option<GpuBuffer> {
        reduce_backward(
            2,
            out_grad,
            input_size,
            Some(&MaxSaved {
                input,
                value,
                count,
            }),
        )
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
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(axes),
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            input,
            result,
            &params,
            &ctx.pipelines_for(input.dtype()).permute,
        )
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
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        // For expand, op_params represents which dimensions were broadcast (size 1 -> N)
        let mut broadcast_flags = [0u32; 4];
        for (i, (&old, &new)) in old_shape.iter().zip(new_shape).take(4).enumerate() {
            *slot(&mut broadcast_flags, i) = u32::from(old == 1 && new > 1);
        }

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: broadcast_flags,
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            input,
            result,
            &params,
            &ctx.pipelines_for(input.dtype()).expand,
        )
    }

    /// Pad tensor, filling the border with `value`
    #[must_use]
    pub(crate) fn pad(
        input: &GpuBuffer,
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
        value: PadValue,
    ) -> Option<GpuBuffer> {
        let ctx = get_gpu_context()?;
        let output_size = new_shape.iter().product::<usize>();
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        let (pad_params, padding2, padding3) = pack_pad_widths(padding);

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_params,
            rank: pack_u32(old_shape.len()),
            padding2,
            // padding[1] carries the fill value bits, read by `pad` in movement.wgsl
            padding: [padding3, value.get().to_bits()],
        };

        movement_op(
            input,
            result,
            &params,
            &ctx.pipelines_for(input.dtype()).pad,
        )
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
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        // op_params contains range starts - convert to usize first
        let starts: Vec<usize> = ranges.iter().map(|(start, _)| *start).collect();

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(&starts),
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            input,
            result,
            &params,
            &ctx.pipelines_for(input.dtype()).shrink,
        )
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
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(strides),
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            input,
            result,
            &params,
            &ctx.pipelines_for(input.dtype()).stride,
        )
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
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(axes),
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            out_grad,
            result,
            &params,
            &ctx.pipelines_for(out_grad.dtype()).permute_backward,
        )
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
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: [0, 0, 0, 0],
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            out_grad,
            result,
            &params,
            &ctx.pipelines_for(out_grad.dtype()).expand_backward,
        )
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
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let (op_params, padding2, padding3) = pack_pad_widths(padding);

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params,
            rank: pack_u32(old_shape.len()),
            padding2,
            padding: [padding3, 0],
        };

        movement_op(
            out_grad,
            result,
            &params,
            &ctx.pipelines_for(out_grad.dtype()).pad_backward,
        )
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
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        // Pack range starts into op_params
        let mut range_starts = [0usize; 4];
        for (i, &(start, _)) in ranges.iter().enumerate().take(4) {
            *slot(&mut range_starts, i) = start;
        }

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(&range_starts),
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            out_grad,
            result,
            &params,
            &ctx.pipelines_for(out_grad.dtype()).shrink_backward,
        )
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
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let params = MovementParams {
            old_shape: pad_to_u32_4(old_shape),
            new_shape: pad_to_u32_4(new_shape),
            op_params: pad_to_u32_4(strides),
            rank: pack_u32(old_shape.len()),
            padding2: 0,
            padding: [0, 0],
        };

        movement_op(
            out_grad,
            result,
            &params,
            &ctx.pipelines_for(out_grad.dtype()).stride_backward,
        )
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
        assert_same_dtype("optimizer_step", &[params, grads, state1, state2]);
        let ctx = get_gpu_context()?;

        // Create uniform buffer with optimizer parameters
        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Optimizer Params"),
                contents: bytemuck::bytes_of(opt_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(params.dtype()).optimizer_step;
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

            let workgroup_count = pack_u32(params.len()).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(())
    }

    /// Muon momentum: `v ← μ·v + g`, returning `update = μ·v + g` (Nesterov) or `v`.
    ///
    /// `velocity` is updated in place, like the parameter in [`Self::optimizer_step`], so
    /// the caller must have got it through the unsharing write path.
    ///
    /// # Panics
    /// If `velocity` and `grad` differ in length.
    #[must_use]
    pub(crate) fn muon_momentum(
        velocity: &GpuBuffer,
        grad: &GpuBuffer,
        momentum: f32,
        nesterov: bool,
    ) -> Option<GpuBuffer> {
        assert_eq!(velocity.len(), grad.len(), "Muon velocity/grad length");
        let ctx = get_gpu_context()?;
        let update = GpuBuffer::zeros(grad.len())?;
        let params = MuonParams {
            n: pack_u32(grad.len()),
            flag: u32::from(nesterov),
            alpha: momentum,
            beta: 0.0,
        };
        muon_dispatch(
            &ctx.pipelines().muon_momentum,
            &[(0, velocity), (1, grad), (2, &update)],
            &params,
            workgroups_for(grad.len()),
        )?;
        Some(update)
    }

    /// Approximately orthogonalize the `rows × cols` row-major matrix `g` with `steps`
    /// quintic Newton-Schulz iterations, on the device. `g` is not modified.
    ///
    /// The same computation as the CPU `Muon::newton_schulz`: Frobenius normalization, a
    /// tall matrix transposed so the Gram matrix is on the smaller side, then
    /// `X ← a·X + (b·A + c·A²)·X` with `A = X Xᵀ`, then the transpose back. `coeffs` is
    /// `[a, b, c]` and `eps` the normalization epsilon; the optimizer owns both, so the
    /// kernel and the CPU path cannot drift apart.
    ///
    /// # Panics
    /// If `g.len() != rows * cols`, or either dimension is 0.
    #[must_use]
    pub(crate) fn muon_newton_schulz(
        g: &GpuBuffer,
        rows: usize,
        cols: usize,
        steps: usize,
        coeffs: [f32; 3],
        eps: f32,
    ) -> Option<GpuBuffer> {
        assert!(
            rows > 0 && cols > 0,
            "Newton-Schulz needs a non-empty matrix"
        );
        assert_eq!(g.len(), rows * cols, "Newton-Schulz buffer vs dimensions");
        let [a, b, c] = coeffs;
        let transposed = rows > cols;
        let (m, n) = if transposed {
            (cols, rows)
        } else {
            (rows, cols)
        };

        let x0 = if transposed {
            Self::permute(g, &[rows, cols], &[cols, rows], &[1, 0])?
        } else {
            g.copy_region(0, g.len())?
        };

        // Frobenius normalization: the sum of squares stays on the device.
        let ctx = get_gpu_context()?;
        let norm_sq = Self::sum_squares(&x0)?;
        let len = pack_u32(x0.len());
        let mut x = GpuBuffer::zeros(x0.len())?;
        muon_dispatch(
            &ctx.pipelines().muon_normalize,
            &[(0, &x0), (1, &norm_sq), (2, &x)],
            &MuonParams {
                n: len,
                flag: 0,
                alpha: eps,
                beta: 0.0,
            },
            workgroups_for(x0.len()),
        )?;

        let spec = |lhs: &[usize], rhs: &[usize]| {
            use crate::shape::Shape;
            crate::spec::MatmulSpec::new(
                &Shape::from_dims(lhs).expect("valid Newton-Schulz dims"),
                &Shape::from_dims(rhs).expect("valid Newton-Schulz dims"),
            )
            .expect("valid Newton-Schulz matmul")
        };
        let gram_spec = spec(&[m, n], &[n, m]);
        let square_spec = spec(&[m, m], &[m, m]);
        let apply_spec = spec(&[m, m], &[m, n]);

        for _ in 0..steps {
            // A = X Xᵀ (m×m), B = b·A + c·A², X ← a·X + B·X
            let xt = Self::permute(&x, &[m, n], &[n, m], &[1, 0])?;
            let gram = Self::matmul(&x, &xt, &gram_spec)?;
            let gram_sq = Self::matmul(&gram, &gram, &square_spec)?;
            let poly = muon_axpby(&gram, b, &gram_sq, c)?;
            let poly_x = Self::matmul(&poly, &x, &apply_spec)?;
            x = muon_axpby(&x, a, &poly_x, 1.0)?;
        }

        if transposed {
            Self::permute(&x, &[m, n], &[n, m], &[1, 0])
        } else {
            Some(x)
        }
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
    #[must_use]
    pub(crate) fn im2col(input: &GpuBuffer, spec: &crate::spec::Window2dSpec) -> Option<GpuBuffer> {
        let crate::spec::Window2dSpec {
            batch: batch_size,
            channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_h: h_out,
            out_w: w_out,
            ..
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output matrix size: (B*H_out*W_out, C*K_h*K_w)
        let rows = batch_size * h_out * w_out;
        let cols = channels * kernel_h * kernel_w;
        let output_size = rows * cols;
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        // Create uniform buffer with im2col parameters
        let params = Im2colParams {
            batch_size: pack_u32(batch_size),
            channels: pack_u32(channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            _padding: [0, 0],
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Im2col Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(input.dtype()).im2col;
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
            let workgroup_count = pack_u32(rows).div_ceil(256);
            dispatch_flat(&mut compute_pass, workgroup_count);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Column-to-image transformation (col2im)
    ///
    /// Transforms a 2D matrix back to a 4D tensor by accumulating gradients.
    /// This is the inverse of im2col, used for computing input gradients.
    ///
    /// # Arguments
    /// * `col` - 2D col matrix of shape `(B*H_out*W_out, C*K_h*K_w)`
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
    /// A 4D tensor of shape (B, C, H, W)
    #[must_use]
    pub(crate) fn col2im(col: &GpuBuffer, spec: &crate::spec::Window2dSpec) -> Option<GpuBuffer> {
        let crate::spec::Window2dSpec {
            batch: batch_size,
            channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_h: h_out,
            out_w: w_out,
            ..
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output tensor size: (B, C, H, W)
        let output_size = batch_size * channels * height * width;
        let result = GpuBuffer::zeros_of(col.dtype(), output_size)?;

        // Create uniform buffer with col2im parameters
        let params = Col2imParams {
            batch_size: pack_u32(batch_size),
            channels: pack_u32(channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            _padding: [0, 0],
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Col2im Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(col.dtype()).col2im;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Col2im Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: col.buffer().as_entire_binding(),
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
                label: Some("Col2im Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Col2im Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup size is (16, 16, 1)
            // x dimension: channels, y dimension: batch * height * width
            let workgroup_x = pack_u32(channels).div_ceil(16);
            let workgroup_y = pack_u32(batch_size * height * width).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Direct convolution without im2col intermediate
    /// Each thread computes one output pixel
    #[must_use]
    pub(crate) fn direct_conv(
        input: &GpuBuffer,
        weight: &GpuBuffer,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("direct_conv", &[input, weight]);
        let crate::spec::Conv2dSpec {
            batch: batch_size,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: h_out,
            out_w: w_out,
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output size: [B, out_channels, h_out, w_out]
        let output_size = batch_size * out_channels * h_out * w_out;
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        // Create uniform buffer with direct conv parameters
        let params = DirectConvParams {
            batch_size: pack_u32(batch_size),
            in_channels: pack_u32(in_channels),
            out_channels: pack_u32(out_channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            pad_h: pack_u32(pad_h),
            pad_w: pack_u32(pad_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Direct Conv Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(input.dtype()).direct_conv;
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
            let workgroups_x = pack_u32(out_channels).div_ceil(16);
            let workgroups_y = pack_u32(batch_size * h_out * w_out).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// Implicit GEMM (iGEMM) convolution with tiled computation
    /// Balances memory usage and performance by avoiding im2col materialization
    /// while using tiling for better cache efficiency
    #[must_use]
    pub(crate) fn igemm(
        input: &GpuBuffer,
        weight: &GpuBuffer,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("igemm", &[input, weight]);
        let crate::spec::Conv2dSpec {
            batch: batch_size,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: h_out,
            out_w: w_out,
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output size: [B, out_channels, h_out, w_out]
        let output_size = batch_size * out_channels * h_out * w_out;
        let result = GpuBuffer::zeros_of(input.dtype(), output_size)?;

        // Tile configuration (can be tuned based on GPU architecture)
        let oc_tile: u32 = 16; // Output channels per tile
        let ic_tile: u32 = 16; // Input channels per tile
        let s_tile: u32 = 16; // Spatial positions per tile

        // Create uniform buffer with iGEMM parameters
        let params = IGEMMParams {
            batch_size: pack_u32(batch_size),
            in_channels: pack_u32(in_channels),
            out_channels: pack_u32(out_channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            pad_h: pack_u32(pad_h),
            pad_w: pack_u32(pad_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            oc_tile,
            ic_tile,
            s_tile,
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iGEMM Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(input.dtype()).igemm;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iGEMM Bind Group"),
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
                label: Some("iGEMM Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iGEMM Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate number of tiles
            let num_oc_tiles = pack_u32(out_channels).div_ceil(oc_tile);
            let total_spatial = pack_u32(batch_size * h_out * w_out);
            let num_spatial_tiles = total_spatial.div_ceil(s_tile);
            let num_tiles = num_oc_tiles * num_spatial_tiles;

            // Workgroup dimensions: (oc_tile, s_tile, num_tiles)
            // Each workgroup processes one tile
            compute_pass.dispatch_workgroups(oc_tile, s_tile, num_tiles);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// GPU-accelerated convolution backward: gradient with respect to input
    /// Computes: ∂L/∂X\[b,ic,ih,iw\] = Σ ∂L/∂Y\[b,oc,oh,ow\] * W\[oc,ic,kh,kw\]
    #[must_use]
    pub(crate) fn conv_backward_input(
        out_grad: &GpuBuffer,
        weight: &GpuBuffer,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("conv_backward_input", &[out_grad, weight]);
        let crate::spec::Conv2dSpec {
            batch: batch_size,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: h_out,
            out_w: w_out,
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output size: [B, in_channels, height, width]
        let output_size = batch_size * in_channels * height * width;
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let params = ConvBackwardInputParams {
            batch_size: pack_u32(batch_size),
            in_channels: pack_u32(in_channels),
            out_channels: pack_u32(out_channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            pad_h: pack_u32(pad_h),
            pad_w: pack_u32(pad_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Conv Backward Input Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(out_grad.dtype()).conv_backward_input;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conv Backward Input Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
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
                label: Some("Conv Backward Input Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Conv Backward Input Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup: (in_channels, batch * height * width, 1)
            let workgroups_x = pack_u32(in_channels).div_ceil(16);
            let workgroups_y = pack_u32(batch_size * height * width).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// GPU-accelerated convolution backward: gradient with respect to weights
    /// Computes: ∂L/∂W\[oc,ic,kh,kw\] = Σ X\[b,ic,oh*sh+kh-ph,ow*sw+kw-pw\] * ∂L/∂Y\[b,oc,oh,ow\]
    #[must_use]
    pub(crate) fn conv_backward_weight(
        out_grad: &GpuBuffer,
        input: &GpuBuffer,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("conv_backward_weight", &[out_grad, input]);
        let crate::spec::Conv2dSpec {
            batch: batch_size,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: h_out,
            out_w: w_out,
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output size: [out_channels, in_channels, kernel_h, kernel_w]
        let output_size = out_channels * in_channels * kernel_h * kernel_w;
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let params = ConvBackwardWeightParams {
            batch_size: pack_u32(batch_size),
            in_channels: pack_u32(in_channels),
            out_channels: pack_u32(out_channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            pad_h: pack_u32(pad_h),
            pad_w: pack_u32(pad_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Conv Backward Weight Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(out_grad.dtype()).conv_backward_weight;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conv Backward Weight Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer().as_entire_binding(),
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
                label: Some("Conv Backward Weight Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Conv Backward Weight Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup: (out_channels, in_channels * kernel_h * kernel_w, 1)
            let workgroups_x = pack_u32(out_channels).div_ceil(16);
            let workgroups_y = pack_u32(in_channels * kernel_h * kernel_w).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// iGEMM convolution backward: gradient with respect to input
    #[must_use]
    pub(crate) fn igemm_backward_input(
        out_grad: &GpuBuffer,
        weight: &GpuBuffer,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("igemm_backward_input", &[out_grad, weight]);
        let crate::spec::Conv2dSpec {
            batch: batch_size,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: h_out,
            out_w: w_out,
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output size: [B, in_channels, height, width]
        let output_size = batch_size * in_channels * height * width;
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let params = IGEMMBackwardInputParams {
            batch_size: pack_u32(batch_size),
            in_channels: pack_u32(in_channels),
            out_channels: pack_u32(out_channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            pad_h: pack_u32(pad_h),
            pad_w: pack_u32(pad_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iGEMM Backward Input Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(out_grad.dtype()).igemm_backward_input;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iGEMM Backward Input Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
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
                label: Some("iGEMM Backward Input Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iGEMM Backward Input Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup: (in_channels, batch * height * width, 1)
            let workgroups_x = pack_u32(in_channels).div_ceil(16);
            let workgroups_y = pack_u32(batch_size * height * width).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }

    /// iGEMM convolution backward: gradient with respect to weights
    #[must_use]
    pub(crate) fn igemm_backward_weight(
        out_grad: &GpuBuffer,
        input: &GpuBuffer,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<GpuBuffer> {
        assert_same_dtype("igemm_backward_weight", &[out_grad, input]);
        let crate::spec::Conv2dSpec {
            batch: batch_size,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h: height,
            in_w: width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: h_out,
            out_w: w_out,
        } = *spec;
        let ctx = get_gpu_context()?;

        // Output size: [out_channels, in_channels, kernel_h, kernel_w]
        let output_size = out_channels * in_channels * kernel_h * kernel_w;
        let result = GpuBuffer::zeros_of(out_grad.dtype(), output_size)?;

        let params = IGEMMBackwardWeightParams {
            batch_size: pack_u32(batch_size),
            in_channels: pack_u32(in_channels),
            out_channels: pack_u32(out_channels),
            height: pack_u32(height),
            width: pack_u32(width),
            kernel_h: pack_u32(kernel_h),
            kernel_w: pack_u32(kernel_w),
            stride_h: pack_u32(stride_h),
            stride_w: pack_u32(stride_w),
            pad_h: pack_u32(pad_h),
            pad_w: pack_u32(pad_w),
            h_out: pack_u32(h_out),
            w_out: pack_u32(w_out),
            _padding: 0,
        };

        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iGEMM Backward Weight Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = &ctx.pipelines_for(out_grad.dtype()).igemm_backward_weight;
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iGEMM Backward Weight Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer().as_entire_binding(),
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
                label: Some("iGEMM Backward Weight Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iGEMM Backward Weight Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup: (out_channels, in_channels * kernel_h * kernel_w, 1)
            let workgroups_x = pack_u32(out_channels).div_ceil(16);
            let workgroups_y = pack_u32(in_channels * kernel_h * kernel_w).div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
        ctx.increment_pending();
        ctx.maybe_sync();

        Some(result)
    }
}

/// Pack up to four movement-kernel values (dims, axes, strides, or offsets) into the
/// `[u32; 4]` uniform slots, zero-filling the rest.
///
/// # Panics
/// More than four values, or a value over `u32::MAX`. The movement ops route rank > 4 to
/// the CPU (`gpu_movement_supported`), so either is a dispatch bug, not data to truncate.
fn pad_to_u32_4(values: &[usize]) -> [u32; 4] {
    assert!(
        values.len() <= 4,
        "GPU movement kernels take rank <= 4, got {values:?}"
    );
    let mut result = [0u32; 4];
    for (out, &v) in result.iter_mut().zip(values) {
        *out = u32::try_from(v).unwrap_or_else(|_| panic!("GPU movement param {v} exceeds u32"));
    }
    result
}

/// Pack `pad`'s widths as `movement.wgsl` reads them: dims 0 and 1 as
/// `[left0, right0, left1, right1]`, and dims 2 and 3 each as `left << 16 | right` (the
/// second and third values).
///
/// # Panics
/// Rank > 4, or a dim-2 or dim-3 width over 16 bits. `gpu_pad_supported` routes both to the
/// CPU, so either is a dispatch bug; unchecked, the shift would silently corrupt the width.
fn pack_pad_widths(padding: &[(usize, usize)]) -> ([u32; 4], u32, u32) {
    assert!(
        padding.len() <= 4,
        "GPU pad takes rank <= 4, got {} dims",
        padding.len()
    );
    let half =
        |w: usize| {
            u32::from(u16::try_from(w).unwrap_or_else(|_| {
                panic!("GPU pad width {w} on dim 2 or 3 does not fit in 16 bits")
            }))
        };
    let packed = |dim: usize| {
        padding
            .get(dim)
            .map_or(0, |&(left, right)| half(left) << 16 | half(right))
    };
    let mut low = [0usize; 4];
    for (i, &(left, right)) in padding.iter().take(2).enumerate() {
        *slot(&mut low, 2 * i) = left;
        *slot(&mut low, 2 * i + 1) = right;
    }
    (pad_to_u32_4(&low), packed(2), packed(3))
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
        let output_size = crate::index::span(&params.new_shape, 0..params.rank as usize)
            .iter()
            .product::<u32>();
        let workgroup_count = output_size.div_ceil(256);
        dispatch_flat(&mut compute_pass, workgroup_count);
    }

    ctx.queue().submit(Some(encoder.finish()));
    ctx.increment_pending();
    ctx.maybe_sync();

    // Return the result buffer (now filled by GPU computation)
    Some(result)
}

/// Converts an index, extent, or length to the `u32` a kernel parameter holds, panicking if it
/// does not fit instead of truncating.
///
/// # Panics
///
/// If `v > u32::MAX`.
#[track_caller]
fn pack_u32(v: usize) -> u32 {
    u32::try_from(v)
        .unwrap_or_else(|_| panic!("kernel parameter {v} does not fit u32 (index-mapping bug)"))
}

/// Workgroups (256 threads each) that cover `len` elements; `dispatch_flat` spreads more
/// than 65535 of them over two dimensions.
fn workgroups_for(len: usize) -> u32 {
    pack_u32(len).div_ceil(256)
}

/// Dispatch `groups` workgroups of a 256-wide 1-D kernel. wgpu caps each dimension at
/// 65535 workgroups (16,776,960 invocations), so a larger launch spreads over `y`; the
/// kernel rebuilds its flat index with `flat_global_id` (`shader_template`), and extra
/// invocations past the end fail their bounds check like any trailing ones (audit A2).
///
/// # Panics
/// If `groups` needs more than 65535 rows (over 2^32 invocations), which `pack_u32` on
/// the element count already rules out.
fn dispatch_flat(pass: &mut wgpu::ComputePass<'_>, groups: u32) {
    const MAX: u32 = 65_535;
    let (x, y) = if groups <= MAX {
        (groups, 1)
    } else {
        (MAX, groups.div_ceil(MAX))
    };
    assert!(
        y <= MAX,
        "GPU dispatch of {groups} workgroups exceeds a 65535 x 65535 grid"
    );
    pass.dispatch_workgroups(x, y, 1);
}

/// `out = alpha·x + beta·y`, elementwise, into a new buffer.
fn muon_axpby(x: &GpuBuffer, alpha: f32, y: &GpuBuffer, beta: f32) -> Option<GpuBuffer> {
    assert_eq!(x.len(), y.len(), "axpby operand lengths");
    let ctx = get_gpu_context()?;
    let out = GpuBuffer::zeros(x.len())?;
    let params = MuonParams {
        n: pack_u32(x.len()),
        flag: 0,
        alpha,
        beta,
    };
    muon_dispatch(
        &ctx.pipelines().muon_axpby,
        &[(0, x), (1, y), (2, &out)],
        &params,
        workgroups_for(x.len()),
    )?;
    Some(out)
}

/// Bind `buffers` (binding index, buffer) and `params` to a `muon.wgsl` pipeline and
/// dispatch it.
fn muon_dispatch(
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &GpuBuffer)],
    params: &MuonParams,
    workgroups: u32,
) -> Option<()> {
    // Muon's kernels compute on f32 buffers; half parameters are cast around them
    // (`optim::muon`), so its scratch never rounds to 16 bits.
    for &(binding, buffer) in buffers {
        assert_eq!(
            buffer.dtype(),
            DType::F32,
            "Muon kernel: binding {binding} is a {} buffer, not f32",
            buffer.dtype()
        );
    }
    let ctx = get_gpu_context()?;
    let params_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Muon Params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let mut entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .map(|&(binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.buffer().as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: params_buffer.as_entire_binding(),
    });
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Muon Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Muon Encoder"),
        });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Muon Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        dispatch_flat(&mut compute_pass, workgroups);
    }
    ctx.queue().submit(Some(encoder.finish()));
    ctx.increment_pending();
    ctx.maybe_sync();
    Some(())
}

use wgpu::util::DeviceExt;

#[cfg(test)]
mod packing_tests {
    use super::{pack_pad_widths, pack_u32, pad_to_u32_4};

    #[test]
    fn pack_u32_converts_in_range_values_exactly() {
        assert_eq!(pack_u32(0), 0);
        assert_eq!(pack_u32(u32::MAX as usize), u32::MAX);
    }

    #[test]
    #[should_panic(expected = "does not fit u32")]
    fn pack_u32_over_u32_panics() {
        let _ = pack_u32(u32::MAX as usize + 1);
    }

    #[test]
    fn pad_widths_pack_as_the_shader_reads_them() {
        let (low, dim2, dim3) = pack_pad_widths(&[(1, 2), (3, 4), (5, 6), (65535, 7)]);
        assert_eq!(low, [1, 2, 3, 4]);
        assert_eq!(dim2, 5 << 16 | 6);
        assert_eq!(dim3, 65535 << 16 | 7);
        assert_eq!(pack_pad_widths(&[(1, 2)]), ([1, 2, 0, 0], 0, 0));
    }

    #[test]
    #[should_panic(expected = "GPU pad width 65536 on dim 2 or 3 does not fit in 16 bits")]
    fn pad_width_over_16_bits_panics() {
        let _ = pack_pad_widths(&[(0, 0), (0, 0), (0, 65536)]);
    }

    #[test]
    #[should_panic(expected = "GPU pad takes rank <= 4")]
    fn pad_rank_5_panics() {
        let _ = pack_pad_widths(&[(0, 0); 5]);
    }

    #[test]
    #[should_panic(expected = "GPU movement kernels take rank <= 4")]
    fn movement_rank_5_panics() {
        let _ = pad_to_u32_4(&[1, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic(expected = "exceeds u32")]
    fn movement_param_over_u32_panics() {
        let _ = pad_to_u32_4(&[usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit usize")]);
    }
}

#[cfg(test)]
mod matmul_tests {
    use super::{GpuBuffer, GpuKernels};
    use crate::gpu;
    use crate::shape::Shape;
    use crate::spec::MatmulSpec;

    fn spec(a: &[usize], b: &[usize]) -> MatmulSpec {
        MatmulSpec::new(&Shape::from_dims(a).unwrap(), &Shape::from_dims(b).unwrap()).unwrap()
    }

    #[test]
    fn test_gpu_matmul() {
        if !gpu::is_gpu_available() {
            return;
        }

        // 2x2 @ 2x2
        let a = GpuBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = GpuBuffer::from_slice(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        let c = GpuKernels::matmul(&a, &b, &spec(&[2, 2], &[2, 2])).unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_matmul_non_square_uses_the_right_dims() {
        if !gpu::is_gpu_available() {
            return;
        }

        // (2x3) @ (3x4): m, k, n are all different, so a swapped pair gives wrong values.
        let a = GpuBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b =
            GpuBuffer::from_slice(&[1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0])
                .unwrap();

        let c = GpuKernels::matmul(&a, &b, &spec(&[2, 3], &[3, 4])).unwrap();

        assert_eq!(c.to_vec(), vec![4.0, 5.0, 5.0, 9.0, 10.0, 11.0, 14.0, 21.0]);
    }

    #[test]
    fn test_gpu_matmul_rejects_a_batched_spec() {
        if !gpu::is_gpu_available() {
            return;
        }
        let a = GpuBuffer::from_slice(&[0.0; 12]).unwrap();
        let b = GpuBuffer::from_slice(&[0.0; 12]).unwrap();
        let batched = spec(&[2, 2, 3], &[2, 3, 2]);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GpuKernels::matmul(&a, &b, &batched)
        }));
        let Err(payload) = result else {
            panic!("a batched spec must not reach the 2D kernel");
        };
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("panic payload should be a string");
        assert!(msg.contains("2D only"), "{msg}");
    }
}

/// Direct `GpuKernels` checks against fixed expected values, run on `GpuBuffer`s: `Add`, `Sub`, `Mul`, and `Div` binary ops, and the `ReLU`, `Sqrt`, `Neg`, `Recip`, `Exp2`, `Log2`, `Sin`, and `Cos` unary ops. Each test skips when no GPU is available.
#[cfg(test)]
mod direct_kernel_tests {
    use super::{GpuBuffer, GpuKernels};
    use crate::gpu;
    use crate::ops::{BinaryOp, UnaryOp};

    // Helper to run comprehensive binary op tests
    fn run_binary_op_test(op: BinaryOp, expected: &[f32]) {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU test (no device)");
            return;
        }

        let a = GpuBuffer::from_slice(&[10.0, 20.0, 30.0, 40.0]).unwrap();
        let b = GpuBuffer::from_slice(&[2.0, 4.0, 5.0, 8.0]).unwrap();

        let c = GpuKernels::binary_op(&a, &b, op)
            .unwrap_or_else(|| panic!("Failed to run binary op {op:?}"));

        let result = c.to_vec();
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Op {op:?} failed at index {i}: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_gpu_binary_ops_comprehensive() {
        // Sub: 10-2, 20-4, 30-5, 40-8
        run_binary_op_test(BinaryOp::Sub, &[8.0, 16.0, 25.0, 32.0]);
        // Mul: 10*2, 20*4, 30*5, 40*8
        run_binary_op_test(BinaryOp::Mul, &[20.0, 80.0, 150.0, 320.0]);
        // Div: 10/2, 20/4, 30/5, 40/8
        run_binary_op_test(BinaryOp::Div, &[5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_gpu_unary_ops_comprehensive() {
        if !gpu::is_gpu_available() {
            return;
        }

        let input = [-1.0, 0.0, 1.0, 4.0];
        let buf = GpuBuffer::from_slice(&input).unwrap();

        // Test ReLU
        let res = GpuKernels::unary_op(&buf, UnaryOp::ReLU).unwrap().to_vec();
        assert_eq!(res, vec![0.0, 0.0, 1.0, 4.0]);

        // Test Sqrt (using positive inputs)
        let buf_pos = GpuBuffer::from_slice(&[4.0, 9.0, 16.0]).unwrap();
        let res_sqrt = GpuKernels::unary_op(&buf_pos, UnaryOp::Sqrt)
            .unwrap()
            .to_vec();
        assert_eq!(res_sqrt, vec![2.0, 3.0, 4.0]);

        // Test Neg
        let res_neg = GpuKernels::unary_op(&buf, UnaryOp::Neg).unwrap().to_vec();
        assert_eq!(res_neg, vec![1.0, -0.0, -1.0, -4.0]);

        // Test Recip (using positive inputs to avoid division by zero)
        let buf_recip = GpuBuffer::from_slice(&[1.0, 2.0, 4.0, 8.0]).unwrap();
        let res_recip = GpuKernels::unary_op(&buf_recip, UnaryOp::Recip)
            .unwrap()
            .to_vec();
        let recip_inputs = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(res_recip.len(), recip_inputs.len());
        for (i, (&val, &input)) in res_recip.iter().zip(&recip_inputs).enumerate() {
            let expected = 1.0 / input;
            assert!(
                (val - expected).abs() < 1e-5,
                "Recip failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Exp2
        let res_exp2 = GpuKernels::unary_op(&buf, UnaryOp::Exp2).unwrap().to_vec();
        let exp_vals = [-1.0, 0.0, 1.0, 4.0];
        assert_eq!(res_exp2.len(), exp_vals.len());
        for (i, (&val, &exponent)) in res_exp2.iter().zip(&exp_vals).enumerate() {
            let expected = 2_f32.powf(exponent);
            assert!(
                (val - expected).abs() < 1e-4,
                "Exp2 failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Log2 (using positive inputs)
        let buf_log2 = GpuBuffer::from_slice(&[1.0, 2.0, 4.0, 8.0]).unwrap();
        let res_log2 = GpuKernels::unary_op(&buf_log2, UnaryOp::Log2)
            .unwrap()
            .to_vec();
        let expected_vals = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(res_log2.len(), expected_vals.len());
        for (i, (&val, &expected)) in res_log2.iter().zip(&expected_vals).enumerate() {
            assert!(
                (val - expected).abs() < 1e-5,
                "Log2 failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Sin
        let res_sin = GpuKernels::unary_op(&buf, UnaryOp::Sin).unwrap().to_vec();
        let expected_vals = [-1.0_f32.sin(), 0.0_f32.sin(), 1.0_f32.sin(), 4.0_f32.sin()];
        assert_eq!(res_sin.len(), expected_vals.len());
        for (i, (&val, &expected)) in res_sin.iter().zip(&expected_vals).enumerate() {
            assert!(
                (val - expected).abs() < 1e-5,
                "Sin failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Cos
        let res_cos = GpuKernels::unary_op(&buf, UnaryOp::Cos).unwrap().to_vec();
        let expected_vals = [0.540_302_3, 1.0, 0.540_302_3, -0.653_643_6]; // cos([-1, 0, 1, 4])
        assert_eq!(res_cos.len(), expected_vals.len());
        for (i, (&val, &expected)) in res_cos.iter().zip(&expected_vals).enumerate() {
            assert!(
                (val - expected).abs() < 1e-5,
                "Cos failed at index {i}: got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gpu_add() {
        if !gpu::is_gpu_available() {
            return;
        }

        let a = GpuBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = GpuBuffer::from_slice(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        let c = GpuKernels::binary_op(&a, &b, BinaryOp::Add).unwrap();

        assert_eq!(c.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }
}

#[cfg(test)]
mod muon_tests {
    use super::{GpuBuffer, GpuKernels};
    use crate::Muon;
    use crate::gpu;
    use crate::index::fetch;

    const COEFFS: [f32; 3] = [3.4445, -4.7750, 2.0315];
    const EPS: f32 = 1e-7;

    /// A seeded random `rows × cols` matrix as a flat row-major vector.
    fn random_matrix(rows: usize, cols: usize) -> Vec<f32> {
        crate::tensor::randn(&[rows, cols]).borrow().data.to_vec()
    }

    /// The device Newton-Schulz of `g`, read back to the host.
    fn on_gpu(g: &[f32], rows: usize, cols: usize, steps: usize) -> Vec<f32> {
        let buf = GpuBuffer::from_slice(g).expect("GPU buffer");
        GpuKernels::muon_newton_schulz(&buf, rows, cols, steps, COEFFS, EPS)
            .expect("GPU Newton-Schulz")
            .to_vec()
    }

    #[test]
    fn newton_schulz_matches_the_cpu_for_wide_tall_square_and_untiled_shapes() {
        if !gpu::is_gpu_available() {
            return;
        }
        crate::tensor::manual_seed(0x6A11);
        // Measured worst difference on an Apple M2: about 1e-6 (summation order and fused
        // multiply-add differ), so 1e-5 leaves 10x of margin without hiding a real error.
        // 300x2 and 2x300 have more than 256 elements in a 2-row Gram matrix, so the
        // norm's strided loop runs more than once per thread; 20x37 and 37x20 are not
        // multiples of the 16x16 matmul tile.
        for (rows, cols) in [(20, 37), (37, 20), (64, 96), (16, 16), (2, 300), (300, 2)] {
            let g = random_matrix(rows, cols);
            let mut cpu = g.clone();
            Muon::newton_schulz(&mut cpu, rows, cols, 5);
            let gpu = on_gpu(&g, rows, cols, 5);
            assert_eq!(gpu.len(), cpu.len(), "{rows}x{cols}: length");
            let worst = cpu
                .iter()
                .zip(&gpu)
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 1e-5,
                "{rows}x{cols}: GPU differs from CPU by {worst}"
            );
        }
    }

    #[test]
    fn newton_schulz_does_not_modify_its_input() {
        if !gpu::is_gpu_available() {
            return;
        }
        crate::tensor::manual_seed(0x1D07);
        let g = random_matrix(5, 9);
        let buf = GpuBuffer::from_slice(&g).expect("GPU buffer");
        let _ = GpuKernels::muon_newton_schulz(&buf, 5, 9, 5, COEFFS, EPS).expect("GPU");
        assert_eq!(buf.to_vec(), g);
    }

    #[test]
    fn newton_schulz_leaves_a_zero_matrix_at_zero() {
        if !gpu::is_gpu_available() {
            return;
        }
        let out = on_gpu(&[0.0; 12], 3, 4, 5);
        for i in 0..12 {
            assert_eq!(fetch(&out, i), 0.0, "entry {i} must stay 0, with no NaN");
        }
    }

    #[test]
    fn momentum_updates_the_velocity_in_place_and_returns_the_update() {
        if !gpu::is_gpu_available() {
            return;
        }
        let v0 = [1.0f32, -2.0, 0.5, 4.0];
        let g = [0.25f32, 1.0, -3.0, 0.0];
        let mu = 0.9f32;
        for nesterov in [false, true] {
            let velocity = GpuBuffer::from_slice(&v0).expect("GPU buffer");
            let grad = GpuBuffer::from_slice(&g).expect("GPU buffer");
            let update = GpuKernels::muon_momentum(&velocity, &grad, mu, nesterov)
                .expect("GPU momentum")
                .to_vec();
            let (v_out, want_v): (Vec<f32>, Vec<f32>) = (
                velocity.to_vec(),
                v0.iter().zip(&g).map(|(v, g)| mu * v + g).collect(),
            );
            let want_update: Vec<f32> = if nesterov {
                want_v.iter().zip(&g).map(|(v, g)| mu * v + g).collect()
            } else {
                want_v.clone()
            };
            for i in 0..4 {
                assert!(
                    (fetch(&v_out, i) - fetch(&want_v, i)).abs() < 1e-6,
                    "v[{i}]"
                );
                assert!(
                    (fetch(&update, i) - fetch(&want_update, i)).abs() < 1e-6,
                    "update[{i}] nesterov={nesterov}"
                );
            }
        }
    }
}
