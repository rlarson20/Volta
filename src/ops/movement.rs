use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::device_policy::kernel_failed;
use crate::error::{Result, VoltaError};
use crate::index::{fetch, slot};
use crate::shape::{Axis, Shape, Strides};
use crate::spec::PadValue;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

// ===== INDEX MAPPING TRAITS =====

/// Defines how to map indices for forward movement operations
///
/// This trait encapsulates the index transformation logic for pad, shrink, and stride
/// operations during the forward pass.
trait ForwardIndexMapper {
    /// Get the iteration shape for the forward pass
    ///
    /// Different operations iterate over different shapes:
    /// - Pad: iterates over input (original) shape
    /// - Shrink: iterates over output (shrunk) shape
    /// - Stride: iterates over output (strided) shape
    fn iteration_shape<'a>(
        &self,
        input_shape: &'a [usize],
        output_shape: &'a [usize],
    ) -> &'a [usize];

    /// Map an iteration index to the corresponding input and output indices
    ///
    /// # Arguments
    /// * `dim` - Current dimension being processed
    /// * `iter_idx` - Index in the iteration shape for this dimension
    ///
    /// # Returns
    /// * (`input_idx`, `output_idx`) - The corresponding indices in the input and output tensors
    fn map_forward_indices(&self, dim: usize, iter_idx: usize) -> (usize, usize);
}

/// Mapper for pad operations: adds padding offset to index
struct PadMapper {
    padding: Vec<(usize, usize)>,
}

impl ForwardIndexMapper for PadMapper {
    fn iteration_shape<'a>(
        &self,
        input_shape: &'a [usize],
        _output_shape: &'a [usize],
    ) -> &'a [usize] {
        input_shape // Iterate over input (original) shape
    }

    fn map_forward_indices(&self, dim: usize, iter_idx: usize) -> (usize, usize) {
        // Forward: output_idx = input_idx + pad_left
        let pad_left = self.padding.get(dim).map_or(0, |p| p.0);
        (iter_idx, iter_idx + pad_left)
    }
}

/// Mapper for shrink operations: offsets by range start
struct ShrinkMapper {
    ranges: Vec<(usize, usize)>,
}

impl ForwardIndexMapper for ShrinkMapper {
    fn iteration_shape<'a>(
        &self,
        _input_shape: &'a [usize],
        output_shape: &'a [usize],
    ) -> &'a [usize] {
        output_shape // Iterate over output (shrunk) shape
    }

    fn map_forward_indices(&self, dim: usize, iter_idx: usize) -> (usize, usize) {
        // Forward: input_idx (original) = output_idx (shrunk) + range_start
        let range_start = self.ranges.get(dim).map_or(0, |r| r.0);
        (iter_idx + range_start, iter_idx)
    }
}

/// Mapper for stride operations: multiplies by stride
struct StrideMapper {
    strides: Vec<usize>,
}

impl ForwardIndexMapper for StrideMapper {
    fn iteration_shape<'a>(
        &self,
        _input_shape: &'a [usize],
        output_shape: &'a [usize],
    ) -> &'a [usize] {
        output_shape // Iterate over output (strided) shape
    }

    fn map_forward_indices(&self, dim: usize, iter_idx: usize) -> (usize, usize) {
        // Forward: input_idx (original) = output_idx (strided) * stride
        let stride = fetch(&self.strides, dim);
        (iter_idx * stride, iter_idx)
    }
}

/// Defines how to map indices for backward movement operations
///
/// This trait encapsulates the index transformation logic for pad, shrink, and stride
/// operations during the backward pass.
trait BackwardIndexMapper {
    /// Get the iteration shape for the backward pass
    fn iteration_shape<'a>(
        &self,
        input_shape: &'a [usize],
        output_shape: &'a [usize],
    ) -> &'a [usize];

    /// Map an iteration index to the corresponding input and output indices
    ///
    /// # Arguments
    /// * `dim` - Current dimension being processed
    /// * `iter_idx` - Index in the iteration shape for this dimension
    /// * `input_dim_size` - Size of the input tensor in this dimension
    /// * `output_dim_size` - Size of the output tensor in this dimension
    ///
    /// # Returns
    /// * `Some((input_idx, output_idx))` - Valid indices for both tensors
    /// * `None` - If the mapping is invalid and should be skipped
    fn map_backward_indices(
        &self,
        dim: usize,
        iter_idx: usize,
        input_dim_size: usize,
        output_dim_size: usize,
    ) -> Option<(usize, usize)>;
}

/// Backward mapper for pad operations
struct PadBackwardMapper {
    padding: Vec<(usize, usize)>,
}

impl BackwardIndexMapper for PadBackwardMapper {
    fn iteration_shape<'a>(
        &self,
        input_shape: &'a [usize],
        _output_shape: &'a [usize],
    ) -> &'a [usize] {
        input_shape // Iterate over input (unpadded) shape
    }

    fn map_backward_indices(
        &self,
        dim: usize,
        iter_idx: usize,
        _input_dim_size: usize,
        _output_dim_size: usize,
    ) -> Option<(usize, usize)> {
        // Backward: we iterate over input shape and map to padded output shape
        // input_idx = iter_idx, output_idx = iter_idx + pad_left
        let pad_left = self.padding.get(dim).map_or(0, |p| p.0);
        Some((iter_idx, iter_idx + pad_left))
    }
}

/// Backward mapper for shrink operations
struct ShrinkBackwardMapper {
    ranges: Vec<(usize, usize)>,
}

impl BackwardIndexMapper for ShrinkBackwardMapper {
    fn iteration_shape<'a>(
        &self,
        _input_shape: &'a [usize],
        output_shape: &'a [usize],
    ) -> &'a [usize] {
        output_shape // Iterate over output (shrunk) shape
    }

    fn map_backward_indices(
        &self,
        dim: usize,
        iter_idx: usize,
        _input_dim_size: usize,
        _output_dim_size: usize,
    ) -> Option<(usize, usize)> {
        // Backward: we iterate over output shape and map to larger input shape
        // output_idx = iter_idx, input_idx = iter_idx + range_start
        let range_start = self.ranges.get(dim).map_or(0, |r| r.0);
        Some((iter_idx + range_start, iter_idx))
    }
}

/// Backward mapper for stride operations
struct StrideBackwardMapper {
    strides: Vec<usize>,
}

impl BackwardIndexMapper for StrideBackwardMapper {
    fn iteration_shape<'a>(
        &self,
        _input_shape: &'a [usize],
        output_shape: &'a [usize],
    ) -> &'a [usize] {
        output_shape // Iterate over output (strided) shape
    }

    fn map_backward_indices(
        &self,
        dim: usize,
        iter_idx: usize,
        input_dim_size: usize,
        _output_dim_size: usize,
    ) -> Option<(usize, usize)> {
        // Backward: we iterate over output shape and map to larger input shape
        // output_idx = iter_idx, input_idx = iter_idx * stride
        let stride = fetch(&self.strides, dim);
        let input_idx = iter_idx * stride;
        if input_idx < input_dim_size {
            Some((input_idx, iter_idx))
        } else {
            None // Out of bounds
        }
    }
}

/// Context struct for movement operations
///
/// Holds all the shared parameters for recursive movement operations,
/// reducing the number of function parameters from 10 to 4.
struct MovementContext<'a> {
    input_shape: &'a [usize],
    output_shape: &'a [usize],
    input_strides: &'a [usize],
    output_strides: &'a [usize],
}

impl<'a> MovementContext<'a> {
    fn new(
        input_shape: &'a [usize],
        output_shape: &'a [usize],
        input_strides: &'a [usize],
        output_strides: &'a [usize],
    ) -> Self {
        Self {
            input_shape,
            output_shape,
            input_strides,
            output_strides,
        }
    }
}

/// Whether the GPU movement kernels cover a tensor of this rank. They pack shapes into
/// `[u32; 4]`, so a higher rank has no GPU kernel: a gap for the device policy, not a
/// kernel to run on silently truncated shapes (H5).
const fn gpu_movement_supported(rank: usize) -> bool {
    rank <= 4
}

/// Whether the GPU `pad` kernels cover this padding: rank <= 4, and dims 2 and 3 are packed
/// as two 16-bit halves of one `u32` each, so their pad widths must fit in 16 bits.
fn gpu_pad_supported(padding: &[(usize, usize)]) -> bool {
    let fits = |w: usize| u16::try_from(w).is_ok();
    gpu_movement_supported(padding.len())
        && padding
            .iter()
            .skip(2)
            .all(|&(left, right)| fits(left) && fits(right))
}

/// Unified recursive index transformation for backward movement operations
///
/// This function consolidates `unpad_recursive`, `unshrink_recursive`, and `unstride_recursive`.
/// It copies data from the output gradient back to the input gradient position.
///
/// # Arguments
/// * `result` - Output gradient buffer (accumulates values)
/// * `grad` - Input gradient buffer (source of values)
/// * `dim` - Current dimension being processed
/// * `input_shape` - Shape of the input tensor (original shape)
/// * `output_shape` - Shape of the output tensor (after operation)
/// * `mapper` - Index mapping strategy
/// * `input_offset` - Current linear offset in input buffer
/// * `output_offset` - Current linear offset in output buffer
/// * `input_strides` - Memory strides for input tensor
/// * `output_strides` - Memory strides for output tensor
fn apply_movement_backward<M: BackwardIndexMapper>(
    ctx: &MovementContext<'_>,
    result: &mut [f32],
    grad: &[f32],
    dim: usize,
    mapper: &M,
    input_offset: usize,
    output_offset: usize,
) {
    // Base case: reached the innermost dimension
    if dim == ctx.input_shape.len() {
        *slot(result, input_offset) = fetch(grad, output_offset);
        return;
    }

    // Get the iteration shape for this backward operation
    let iter_shape = mapper.iteration_shape(ctx.input_shape, ctx.output_shape);
    let dim_size = fetch(iter_shape, dim);
    let input_dim_size = fetch(ctx.input_shape, dim);
    let output_dim_size = fetch(ctx.output_shape, dim);

    // Iterate according to the mapper's iteration shape
    for iter_idx in 0..dim_size {
        // Map iteration index to input and output indices
        if let Some((input_idx, output_idx)) =
            mapper.map_backward_indices(dim, iter_idx, input_dim_size, output_dim_size)
        {
            let input_stride = fetch(ctx.input_strides, dim);
            let output_stride = fetch(ctx.output_strides, dim);

            apply_movement_backward(
                ctx,
                result,
                grad,
                dim + 1,
                mapper,
                input_offset + input_idx * input_stride,
                output_offset + output_idx * output_stride,
            );
        }
    }
}

/// Unified recursive index transformation for forward movement operations
///
/// This function consolidates `pad_recursive`, `shrink_recursive`, and `stride_recursive`.
/// It copies data from the input buffer to the output buffer with index transformation.
///
/// # Arguments
/// * `result` - Output buffer (destination)
/// * `data` - Input buffer (source)
/// * `dim` - Current dimension being processed
/// * `input_shape` - Shape of the input tensor
/// * `output_shape` - Shape of the output tensor
/// * `mapper` - Index mapping strategy
/// * `input_offset` - Current linear offset in input buffer
/// * `output_offset` - Current linear offset in output buffer
/// * `input_strides` - Memory strides for input tensor
/// * `output_strides` - Memory strides for output tensor
fn apply_movement_forward<M: ForwardIndexMapper>(
    ctx: &MovementContext<'_>,
    result: &mut [f32],
    data: &[f32],
    dim: usize,
    mapper: &M,
    input_offset: usize,
    output_offset: usize,
) {
    // Base case: reached the innermost dimension
    if dim == ctx.input_shape.len() {
        *slot(result, output_offset) = fetch(data, input_offset);
        return;
    }

    // Get the iteration shape for this operation
    let iter_shape = mapper.iteration_shape(ctx.input_shape, ctx.output_shape);
    let dim_size = fetch(iter_shape, dim);
    let input_dim_size = fetch(ctx.input_shape, dim);
    let output_dim_size = fetch(ctx.output_shape, dim);

    // Iterate through the dimension
    for iter_idx in 0..dim_size {
        // Map iteration index to input and output indices
        let (input_idx, output_idx) = mapper.map_forward_indices(dim, iter_idx);

        // Check bounds
        if input_idx < input_dim_size && output_idx < output_dim_size {
            let input_stride = fetch(ctx.input_strides, dim);
            let output_stride = fetch(ctx.output_strides, dim);

            apply_movement_forward(
                ctx,
                result,
                data,
                dim + 1,
                mapper,
                input_offset + input_idx * input_stride,
                output_offset + output_idx * output_stride,
            );
        }
    }
}

// ===== MOVEMENT OPERATIONS =====

/// Movement operations: reshape/reorder data without changing values
///
/// These operations don't modify data values, only how they're indexed.
/// Gradients must "undo" these operations during backpropagation.
#[derive(Clone)]
pub enum MovementOp {
    Reshape { new_shape: Vec<usize> }, // Change shape, preserve order
    Permute { axes: Vec<usize> },      // Transpose/reorder axes
    Expand { new_shape: Vec<usize> },  // Broadcast to larger shape
    Pad { padding: Vec<(usize, usize)> }, // Add zeros around edges
    Shrink { ranges: Vec<(usize, usize)> }, // Extract subregion
    Stride { strides: Vec<usize> },    // Subsample with stride
}

/// Unified gradient function for all movement operations
///
/// Movement ops don't change data values, only how they're indexed.
/// During backward, we need to "undo" the movement to restore the original shape.
#[derive(Clone)]
pub(crate) struct MovementGradFn {
    op: MovementOp,
    original_shape: Shape,
}

impl MovementGradFn {
    /// The op's name in device errors and fallback events, e.g. `"Movement(Permute)"`.
    pub(crate) fn op_name(&self) -> String {
        let op = match self.op {
            MovementOp::Reshape { .. } => "Reshape",
            MovementOp::Permute { .. } => "Permute",
            MovementOp::Expand { .. } => "Expand",
            MovementOp::Pad { .. } => "Pad",
            MovementOp::Shrink { .. } => "Shrink",
            MovementOp::Stride { .. } => "Stride",
        };
        format!("Movement({op})")
    }

    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let grad_tensor = match &self.op {
            MovementOp::Reshape { .. } => {
                // Reshape back to original shape: same data, same device, new shape.
                return vec![Some(RawTensor::new_with_storage(
                    out_grad.data.clone(),
                    self.original_shape.clone(),
                    out_grad.device.clone(),
                    false,
                ))];
            }
            MovementOp::Permute { axes } => {
                // Invert the permutation to restore original order
                let mut inverse_axes = vec![0; axes.len()];
                for (i, &ax) in axes.iter().enumerate() {
                    *slot(&mut inverse_axes, ax) = i;
                }

                // GPU path when the kernel covers this rank; a `None` from it is a failure.
                if out_grad.device.is_gpu() && gpu_movement_supported(self.original_shape.rank()) {
                    let grad_storage = RawTensor::gpu_permute_backward(
                        &out_grad.data,
                        self.original_shape.truncate_rank(),
                        out_grad.shape.truncate_rank(),
                        &inverse_axes,
                    )
                    .unwrap_or_else(|| kernel_failed("Movement(Permute)"));
                    return vec![Some(RawTensor::new_with_storage(
                        grad_storage,
                        self.original_shape.clone(),
                        out_grad.device.clone(),
                        false,
                    ))];
                }

                // CPU path
                let grad_t =
                    RawTensor::from_shape(out_grad.data.to_vec(), out_grad.shape.clone(), false);
                let (storage, shape, device) = RawTensor::permute_impl(&grad_t, &inverse_axes);
                return vec![Some(RawTensor::new_with_storage(
                    storage,
                    Shape::derived(shape),
                    device,
                    false,
                ))];
            }
            MovementOp::Expand { new_shape } => {
                // GPU path when the kernel covers this rank; a `None` from it is a failure.
                if out_grad.device.is_gpu() && gpu_movement_supported(self.original_shape.rank()) {
                    let grad_storage = RawTensor::gpu_expand_backward(
                        &out_grad.data,
                        self.original_shape.truncate_rank(),
                        new_shape,
                    )
                    .unwrap_or_else(|| kernel_failed("Movement(Expand)"));
                    return vec![Some(RawTensor::new_with_storage(
                        grad_storage,
                        self.original_shape.clone(),
                        out_grad.device.clone(),
                        false,
                    ))];
                }

                // CPU path: Sum gradient over dimensions that were expanded (broadcast)
                let mut grad_data = vec![0.0; self.original_shape.iter().product()];
                let old_strides = self.original_shape.strides();
                let old_strides = old_strides.truncate_rank();

                let g_host = out_grad.data.to_host_f32();
                for i in 0..out_grad.data.len() {
                    let mut old_idx = 0;
                    let mut rem = i;
                    for j in (0..new_shape.len()).rev() {
                        let dim_size = fetch(new_shape, j);
                        let coord = rem % dim_size;
                        rem /= dim_size;
                        // If this dimension was size 1, don't advance the index
                        if self.original_shape.dim_at(j) != 1 {
                            old_idx += coord * fetch(old_strides, j);
                        }
                    }
                    *slot(&mut grad_data, old_idx) += fetch(&g_host, i);
                }
                RawTensor::from_shape(grad_data, self.original_shape.clone(), false)
            }
            MovementOp::Pad { padding } => {
                // GPU path when the kernel covers this rank; a `None` from it is a failure.
                if out_grad.device.is_gpu() && gpu_pad_supported(padding) {
                    let grad_storage = RawTensor::gpu_pad_backward(
                        &out_grad.data,
                        self.original_shape.truncate_rank(),
                        out_grad.shape.truncate_rank(),
                        padding,
                    )
                    .unwrap_or_else(|| kernel_failed("Movement(Pad)"));
                    return vec![Some(RawTensor::new_with_storage(
                        grad_storage,
                        self.original_shape.clone(),
                        out_grad.device.clone(),
                        false,
                    ))];
                }

                // CPU path: Remove padding from gradient (extract center region)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides = self.original_shape.strides();
                let old_strides = old_strides.truncate_rank();
                let new_strides = out_grad.shape.strides();
                let new_strides = new_strides.truncate_rank();
                let mapper = PadBackwardMapper {
                    padding: padding.clone(),
                };
                let ctx = MovementContext::new(
                    self.original_shape.truncate_rank(),
                    out_grad.shape.truncate_rank(),
                    old_strides,
                    new_strides,
                );

                apply_movement_backward(
                    &ctx,
                    &mut result,
                    &out_grad.data.to_host_f32(),
                    0,
                    &mapper,
                    0,
                    0,
                );
                RawTensor::from_shape(result, self.original_shape.clone(), false)
            }
            MovementOp::Shrink { ranges } => {
                // GPU path when the kernel covers this rank; a `None` from it is a failure.
                if out_grad.device.is_gpu() && gpu_movement_supported(self.original_shape.rank()) {
                    let grad_storage = RawTensor::gpu_shrink_backward(
                        &out_grad.data,
                        self.original_shape.truncate_rank(),
                        out_grad.shape.truncate_rank(),
                        ranges,
                    )
                    .unwrap_or_else(|| kernel_failed("Movement(Shrink)"));
                    return vec![Some(RawTensor::new_with_storage(
                        grad_storage,
                        self.original_shape.clone(),
                        out_grad.device.clone(),
                        false,
                    ))];
                }

                // CPU path: Pad gradient back to original size (inverse of shrink)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides = self.original_shape.strides();
                let old_strides = old_strides.truncate_rank();
                let new_strides = out_grad.shape.strides();
                let new_strides = new_strides.truncate_rank();
                let mapper = ShrinkBackwardMapper {
                    ranges: ranges.clone(),
                };
                let ctx = MovementContext::new(
                    self.original_shape.truncate_rank(),
                    out_grad.shape.truncate_rank(),
                    old_strides,
                    new_strides,
                );

                apply_movement_backward(
                    &ctx,
                    &mut result,
                    &out_grad.data.to_host_f32(),
                    0,
                    &mapper,
                    0,
                    0,
                );
                RawTensor::from_shape(result, self.original_shape.clone(), false)
            }
            MovementOp::Stride { strides } => {
                // GPU path when the kernel covers this rank; a `None` from it is a failure.
                if out_grad.device.is_gpu() && gpu_movement_supported(self.original_shape.rank()) {
                    let grad_storage = RawTensor::gpu_stride_backward(
                        &out_grad.data,
                        self.original_shape.truncate_rank(),
                        out_grad.shape.truncate_rank(),
                        strides,
                    )
                    .unwrap_or_else(|| kernel_failed("Movement(Stride)"));
                    return vec![Some(RawTensor::new_with_storage(
                        grad_storage,
                        self.original_shape.clone(),
                        out_grad.device.clone(),
                        false,
                    ))];
                }

                // CPU path: Upsample gradient (inverse of stride/downsampling)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides_mem = self.original_shape.strides();
                let old_strides_mem = old_strides_mem.truncate_rank();
                let new_strides_mem = out_grad.shape.strides();
                let new_strides_mem = new_strides_mem.truncate_rank();
                let mapper = StrideBackwardMapper {
                    strides: strides.clone(),
                };
                let ctx = MovementContext::new(
                    self.original_shape.truncate_rank(),
                    out_grad.shape.truncate_rank(),
                    old_strides_mem,
                    new_strides_mem,
                );

                apply_movement_backward(
                    &ctx,
                    &mut result,
                    &out_grad.data.to_host_f32(),
                    0,
                    &mapper,
                    0,
                    0,
                );
                RawTensor::from_shape(result, self.original_shape.clone(), false)
            }
        };

        vec![Some(grad_tensor)]
    }
}

// ===== MOVEMENT OPERATIONS =====
impl RawTensor {
    /// The one place a movement op's autograd edge is built.
    ///
    /// Every movement op computes its output storage (on the GPU or the CPU), then ends in
    /// exactly one call here, so the `GradFn` it records can't differ between the two paths.
    fn movement_edge(
        self_t: &Tensor,
        storage: Storage,
        shape: Shape,
        device: Device,
        op: MovementOp,
        original_shape: Shape,
    ) -> Tensor {
        edge(
            [self_t.clone()],
            storage,
            shape,
            device,
            GradFn::Movement(MovementGradFn { op, original_shape }),
        )
    }

    /// Reshape tensor to new shape (same number of elements)
    ///
    /// # Panics
    ///
    /// If `new_shape` is invalid or has a different element count. Use
    /// [`Self::try_reshape`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn reshape(self_t: &Tensor, new_shape: &[usize]) -> Tensor {
        Self::try_reshape(self_t, new_shape).unwrap_or_else(|e| panic!("Reshape: {e}"))
    }

    /// Reshape tensor to new shape (fallible)
    ///
    /// # Errors
    ///
    /// `VoltaError::InvalidParameter` if `new_shape` has a zero dimension, overflows, or
    /// exceeds `MAX_NUMEL` (see [`Shape::from_dims`]), or has a different element count.
    pub fn try_reshape(self_t: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
        // Validate before `product()` or any allocation: `new_shape` may be user data.
        let target = Shape::from_dims(new_shape)?;
        let (data, old_shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        if old_shape.numel() != target.numel() {
            return Err(VoltaError::InvalidParameter(format!(
                "cannot reshape {:?} ({} elements) to {new_shape:?} ({} elements)",
                old_shape.truncate_rank(),
                old_shape.numel(),
                target.numel()
            )));
        }

        // `data` shares the input's buffer (copy-on-write storage); only the shape changes.
        Ok(Self::movement_edge(
            self_t,
            data,
            target,
            device,
            MovementOp::Reshape {
                new_shape: new_shape.to_vec(),
            },
            old_shape,
        ))
    }

    /// Internal permute implementation (no gradient tracking)
    ///
    /// Reorders axes according to the permutation specified by `axes`.
    /// For example, axes=[1,0] transposes a 2D matrix.
    fn permute_impl(self_t: &Tensor, axes: &[usize]) -> (Storage, Vec<usize>, Device) {
        // Helper: convert linear index to coordinates
        fn index_to_coords(idx: usize, shape: &[usize]) -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            let mut remaining = idx;
            for i in (0..shape.len()).rev() {
                let dim_size = fetch(shape, i);
                *slot(&mut coords, i) = remaining % dim_size;
                remaining /= dim_size;
            }
            coords
        }

        // Helper: convert coordinates to linear index using strides
        fn coords_to_index(coords: &[usize], strides: &[usize]) -> usize {
            coords.iter().zip(strides).map(|(c, s)| c * s).sum()
        }

        let (data, shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };
        assert_eq!(axes.len(), shape.rank(), "Axes length must match rank");

        let new_shape: Vec<usize> = axes.iter().map(|&i| shape.dim_at(i)).collect();

        // GPU path when the kernel covers this rank; a `None` from it is a failure.
        if device.is_gpu() && gpu_movement_supported(shape.rank()) {
            let storage = Self::gpu_permute(&data, shape.truncate_rank(), &new_shape, axes)
                .unwrap_or_else(|| kernel_failed("Movement(Permute)"));
            return (storage, new_shape, device);
        }

        // CPU path (`movement_edge` applies the device policy to a GPU input)
        let old_strides = shape.strides();
        let old_strides = old_strides.truncate_rank();
        let mut new_data = vec![0.0; data.len()];

        let cpu_data = data.to_vec();

        for (new_idx, val) in new_data.iter_mut().enumerate() {
            let new_coords = index_to_coords(new_idx, &new_shape);
            // Map new coordinates back to old coordinates
            let mut old_coords = vec![0; axes.len()];
            for (i, &ax) in axes.iter().enumerate() {
                *slot(&mut old_coords, ax) = fetch(&new_coords, i);
            }
            let old_idx = coords_to_index(&old_coords, old_strides);
            *val = fetch(&cpu_data, old_idx);
        }
        (Storage::cpu(new_data), new_shape, Device::CPU)
    }

    /// Permute (reorder) tensor axes
    ///
    /// # Arguments
    /// * `axes` - New ordering of axes (must be a valid permutation of 0..rank)
    /// # Panics
    /// invalid permutation axes
    #[must_use]
    pub fn permute(self_t: &Tensor, axes: &[Axis]) -> Tensor {
        let old_shape = self_t.borrow().shape.clone();
        let axes: Vec<usize> = axes.iter().map(|a| a.0).collect();

        // Verify axes is a valid permutation
        let mut sorted_axes = axes.clone();
        sorted_axes.sort_unstable();
        for (i, &ax) in sorted_axes.iter().enumerate() {
            assert_eq!(i, ax, "Invalid permutation axes");
        }

        let (storage, new_shape, device) = Self::permute_impl(self_t, &axes);

        Self::movement_edge(
            self_t,
            storage,
            Shape::derived(new_shape),
            device,
            MovementOp::Permute { axes },
            old_shape,
        )
    }

    /// Expand (broadcast) tensor to larger shape
    ///
    /// Dimensions can only be expanded from size 1 to size N.
    /// Rank must remain the same.
    ///
    /// # Panics
    ///
    /// If `new_shape` is invalid, has a different rank, or a dimension is neither `1`
    /// nor equal to the target. Use [`Self::try_expand`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn expand(self_t: &Tensor, new_shape: &[usize]) -> Tensor {
        Self::try_expand(self_t, new_shape).unwrap_or_else(|e| panic!("Expand: {e}"))
    }

    /// Expand (broadcast) tensor to larger shape (fallible)
    ///
    /// # Errors
    ///
    /// `VoltaError::InvalidParameter` if `new_shape` is invalid (see [`Shape::from_dims`])
    /// or has a different rank, `VoltaError::BroadcastError` if a dimension is neither
    /// `1` nor equal to the target.
    pub fn try_expand(self_t: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
        let out_shape = Shape::from_dims(new_shape)?;
        let (data, old_shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        if old_shape.rank() != new_shape.len() {
            return Err(VoltaError::InvalidParameter(format!(
                "expand: rank must match, got {:?} -> {new_shape:?}",
                old_shape.truncate_rank()
            )));
        }

        // Verify broadcasting rules: old dims must be 1 or equal to new
        for (old_d, &new_d) in old_shape.iter().zip(new_shape) {
            if old_d != 1 && old_d != new_d {
                return Err(VoltaError::BroadcastError(
                    old_shape.truncate_rank().to_vec(),
                    new_shape.to_vec(),
                ));
            }
        }

        let new_size = out_shape.numel();

        // GPU path when the kernel covers this case; a `None` from it is a failure.
        let gpu_storage =
            (device.is_gpu() && gpu_movement_supported(old_shape.rank())).then(|| {
                Self::gpu_expand(&data, old_shape.truncate_rank(), new_shape)
                    .unwrap_or_else(|| kernel_failed("Movement(Expand)"))
            });
        let (storage, out_device) = if let Some(storage) = gpu_storage {
            (storage, device)
        } else {
            let mut result = vec![0.0; new_size];
            let cpu_data = data.to_vec();

            // Broadcast by repeating values
            let old_strides = old_shape.strides();
            let old_strides = old_strides.truncate_rank();

            for i in 0..new_size {
                let mut old_idx = 0;
                let mut rem = i;
                for j in (0..new_shape.len()).rev() {
                    let dim_size = fetch(new_shape, j);
                    let coord = rem % dim_size;
                    rem /= dim_size;
                    if old_shape.dim_at(j) != 1 {
                        let stride = fetch(old_strides, j);
                        old_idx += coord * stride;
                    }
                }
                *slot(&mut result, i) = fetch(&cpu_data, old_idx);
            }
            (Storage::cpu(result), Device::CPU)
        };

        Ok(Self::movement_edge(
            self_t,
            storage,
            out_shape,
            out_device,
            MovementOp::Expand {
                new_shape: new_shape.to_vec(),
            },
            old_shape,
        ))
    }

    /// Pad a tensor, filling the new border with `value`.
    ///
    /// # Arguments
    /// * `padding` - For each dimension, (`left_pad`, `right_pad`)
    /// * `value` - Fill for the border. Convolution wants [`PadValue::ZERO`];
    ///   max pooling wants [`PadValue::NEG_INF`] (H1).
    /// # Panics
    /// Panics if `padding` does not match the rank or the padded shape is
    /// invalid. Use [`Self::try_pad_with`] to get an error instead.
    #[must_use]
    pub fn pad_with(self_t: &Tensor, padding: &[(usize, usize)], value: PadValue) -> Tensor {
        Self::try_pad_with(self_t, padding, value).unwrap_or_else(|e| panic!("Pad: {e}"))
    }

    /// Fallible version of [`Self::pad_with`].
    ///
    /// # Errors
    /// * `VoltaError::InvalidParameter` - `padding.len()` differs from the rank
    /// * Any `Shape::from_dims` error for the padded shape (zero dim, over the element cap)
    pub fn try_pad_with(
        self_t: &Tensor,
        padding: &[(usize, usize)],
        value: PadValue,
    ) -> Result<Tensor> {
        use crate::error::VoltaError;

        let (data, old_shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        if padding.len() != old_shape.rank() {
            return Err(VoltaError::InvalidParameter(format!(
                "Padding length ({}) must match tensor rank ({})",
                padding.len(),
                old_shape.rank()
            )));
        }

        let new_shape: Vec<usize> = old_shape
            .iter()
            .zip(padding)
            .map(|(d, (l, r))| d + l + r)
            .collect();

        let out_shape = Shape::from_dims(new_shape.clone())?;
        let new_size = out_shape.numel();

        // GPU path when the kernel covers this case; a `None` from it is a failure.
        let gpu_storage = (device.is_gpu() && gpu_pad_supported(padding)).then(|| {
            Self::gpu_pad(&data, old_shape.truncate_rank(), &new_shape, padding, value)
                .unwrap_or_else(|| kernel_failed("Movement(Pad)"))
        });
        let (storage, out_device) = if let Some(storage) = gpu_storage {
            (storage, device)
        } else {
            // CPU path. `PadMapper` only writes the interior, so the fill is the border.
            let mut result = vec![value.get(); new_size];
            let cpu_data = data.to_vec();

            // Copy old data into padded positions
            let old_strides = old_shape.strides();
            let old_strides = old_strides.truncate_rank();
            let new_strides = Strides::from_shape(&new_shape);
            let new_strides = new_strides.truncate_rank();
            let mapper = PadMapper {
                padding: padding.to_vec(),
            };
            let ctx = MovementContext::new(
                old_shape.truncate_rank(),
                &new_shape,
                old_strides,
                new_strides,
            );

            apply_movement_forward(&ctx, &mut result, &cpu_data, 0, &mapper, 0, 0);
            (Storage::cpu(result), Device::CPU)
        };

        Ok(Self::movement_edge(
            self_t,
            storage,
            out_shape,
            out_device,
            MovementOp::Pad {
                padding: padding.to_vec(),
            },
            old_shape,
        ))
    }

    /// Extract a sub-region of the tensor
    ///
    /// # Arguments
    /// * `ranges` - For each dimension, (start, end) indices
    /// # Panics
    /// Validate shrink ranges for bounds checking
    ///
    /// Ensures that:
    /// - Ranks match (`ranges.len() == shape.len()`)
    /// - Range order is valid (`start <= end` for each dimension)
    /// - End bounds are within dimension size (`end <= dimension_size`)
    /// - Resulting tensor is non-empty (at least one dimension has size > 0)
    fn validate_shrink_ranges(shape: &[usize], ranges: &[(usize, usize)]) -> Result<()> {
        use crate::error::VoltaError;

        // Validate rank match
        if ranges.len() != shape.len() {
            return Err(VoltaError::InvalidParameter(format!(
                "Shrink ranges length ({}) must match tensor rank ({})",
                ranges.len(),
                shape.len()
            )));
        }

        // Validate each range
        for (dim, (&(start, end), &dim_size)) in ranges.iter().zip(shape.iter()).enumerate() {
            // Check range order
            if start > end {
                return Err(VoltaError::InvalidParameter(format!(
                    "Invalid shrink range for dimension {dim}: start ({start}) exceeds end ({end})"
                )));
            }

            // Check end bound
            if end > dim_size {
                return Err(VoltaError::DimensionOutOfBounds {
                    dim,
                    shape: shape.to_vec(),
                });
            }

            // Check for empty range
            if start == end {
                return Err(VoltaError::InvalidParameter(format!(
                    "Invalid shrink range for dimension {dim}: start equals end ({start}), creates zero-sized dimension"
                )));
            }
        }

        Ok(())
    }

    /// Shrink tensor to specified ranges for each dimension
    ///
    /// Extracts a subregion of the tensor by specifying start and end indices
    /// for each dimension. The range `[start, end)` follows Python slicing conventions.
    ///
    /// # Arguments
    /// * `ranges` - Slice of `(start, end)` tuples for each dimension
    ///
    /// # Returns
    /// New tensor with shrunk dimensions
    ///
    /// # Panics
    /// - If `ranges` length doesn't match tensor rank
    /// - If any range has `start > end`
    /// - If any range has `end > dimension size`
    /// - If any range has `start == end` (zero-sized dimension)
    #[must_use]
    pub fn shrink(self_t: &Tensor, ranges: &[(usize, usize)]) -> Tensor {
        Self::try_shrink(self_t, ranges).expect("Invalid shrink ranges")
    }

    /// Try to shrink tensor to specified ranges for each dimension
    ///
    /// Extracts a subregion of the tensor by specifying start and end indices
    /// for each dimension. The range `[start, end)` follows Python slicing conventions.
    ///
    /// # Arguments
    /// * `ranges` - Slice of `(start, end)` tuples for each dimension
    ///
    /// # Returns
    /// * `Ok(Tensor)` - New tensor with shrunk dimensions
    /// * `Err(VoltaError)` - If validation fails
    ///
    /// # Errors
    /// * `VoltaError::InvalidParameter` - If:
    ///   - Ranges length doesn't match tensor rank
    ///   - Any range has `start > end`
    ///   - Any range has `start == end` (zero-sized dimension)
    /// * `VoltaError::DimensionOutOfBounds` - If any range has `end > dimension size`
    pub fn try_shrink(self_t: &Tensor, ranges: &[(usize, usize)]) -> Result<Tensor> {
        let (data, old_shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        // Validate ranges before any computation
        Self::validate_shrink_ranges(old_shape.truncate_rank(), ranges)?;

        let new_shape: Vec<usize> = ranges.iter().map(|(start, end)| end - start).collect();

        // GPU path when the kernel covers this case; a `None` from it is a failure.
        let gpu_storage =
            (device.is_gpu() && gpu_movement_supported(old_shape.rank())).then(|| {
                Self::gpu_shrink(&data, old_shape.truncate_rank(), &new_shape, ranges)
                    .unwrap_or_else(|| kernel_failed("Movement(Shrink)"))
            });
        let (storage, out_device) = if let Some(storage) = gpu_storage {
            (storage, device)
        } else {
            let new_size: usize = new_shape.iter().product();
            let mut result = vec![0.0; new_size];
            let cpu_data = data.to_vec();

            let old_strides = old_shape.strides();
            let old_strides = old_strides.truncate_rank();
            let new_strides = Strides::from_shape(&new_shape);
            let new_strides = new_strides.truncate_rank();
            let mapper = ShrinkMapper {
                ranges: ranges.to_vec(),
            };
            let ctx = MovementContext::new(
                old_shape.truncate_rank(),
                &new_shape,
                old_strides,
                new_strides,
            );

            apply_movement_forward(&ctx, &mut result, &cpu_data, 0, &mapper, 0, 0);
            (Storage::cpu(result), Device::CPU)
        };

        Ok(Self::movement_edge(
            self_t,
            storage,
            Shape::derived(new_shape),
            out_device,
            MovementOp::Shrink {
                ranges: ranges.to_vec(),
            },
            old_shape,
        ))
    }

    /// Subsample tensor with specified strides
    ///
    /// Similar to slicing with step: array\[`::2`\] takes every other element
    /// # Panics
    /// Strides length must match rank
    #[must_use]
    pub fn stride_op(self_t: &Tensor, strides: &[usize]) -> Tensor {
        let (data, old_shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        assert_eq!(
            strides.len(),
            old_shape.rank(),
            "Strides length must match rank"
        );

        let new_shape: Vec<usize> = old_shape
            .iter()
            .zip(strides)
            .map(|(d, s)| d.div_ceil(*s))
            .collect();

        // GPU path when the kernel covers this case; a `None` from it is a failure.
        let gpu_storage =
            (device.is_gpu() && gpu_movement_supported(old_shape.rank())).then(|| {
                Self::gpu_stride(&data, old_shape.truncate_rank(), &new_shape, strides)
                    .unwrap_or_else(|| kernel_failed("Movement(Stride)"))
            });
        let (storage, out_device) = if let Some(storage) = gpu_storage {
            (storage, device)
        } else {
            let new_size: usize = new_shape.iter().product();
            let mut result = vec![0.0; new_size];
            let cpu_data = data.to_vec();

            let old_strides_mem = old_shape.strides();
            let old_strides_mem = old_strides_mem.truncate_rank();
            let new_strides_mem = Strides::from_shape(&new_shape);
            let new_strides_mem = new_strides_mem.truncate_rank();
            let mapper = StrideMapper {
                strides: strides.to_vec(),
            };
            let ctx = MovementContext::new(
                old_shape.truncate_rank(),
                &new_shape,
                old_strides_mem,
                new_strides_mem,
            );

            apply_movement_forward(&ctx, &mut result, &cpu_data, 0, &mapper, 0, 0);
            (Storage::cpu(result), Device::CPU)
        };

        Self::movement_edge(
            self_t,
            storage,
            Shape::derived(new_shape),
            out_device,
            MovementOp::Stride {
                strides: strides.to_vec(),
            },
            old_shape,
        )
    }
}
