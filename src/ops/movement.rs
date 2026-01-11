use crate::autograd::GradFn;
use crate::{RawTensor, Tensor};

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
pub struct MovementGradFn {
    op: MovementOp,
    original_shape: Vec<usize>,
}

impl GradFn for MovementGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_tensor = match &self.op {
            MovementOp::Reshape { .. } => {
                // Reshape back to original shape
                RawTensor::new(out_grad.data.to_vec(), &self.original_shape, false)
            }
            MovementOp::Permute { axes } => {
                // Invert the permutation to restore original order
                let mut inverse_axes = vec![0; axes.len()];
                for (i, &ax) in axes.iter().enumerate() {
                    inverse_axes[ax] = i;
                }

                // Try GPU path
                #[cfg(feature = "gpu")]
                {
                    if out_grad.device.is_gpu()
                        && let Some(grad_storage) = RawTensor::gpu_permute_backward(
                            &out_grad.data,
                            &self.original_shape,
                            &out_grad.shape,
                            &inverse_axes,
                        )
                    {
                        return vec![Some(RawTensor::new_with_storage(
                            grad_storage,
                            &self.original_shape,
                            out_grad.device.clone(),
                            false,
                        ))];
                    }
                }

                // CPU fallback
                let grad_t = RawTensor::new(out_grad.data.to_vec(), &out_grad.shape, false);
                let result = RawTensor::permute_impl(&grad_t, &inverse_axes);
                return vec![Some(result)];
            }
            MovementOp::Expand { new_shape } => {
                // Try GPU path
                #[cfg(feature = "gpu")]
                {
                    if out_grad.device.is_gpu()
                        && let Some(grad_storage) = RawTensor::gpu_expand_backward(
                            &out_grad.data,
                            &self.original_shape,
                            new_shape,
                        )
                    {
                        return vec![Some(RawTensor::new_with_storage(
                            grad_storage,
                            &self.original_shape,
                            out_grad.device.clone(),
                            false,
                        ))];
                    }
                }

                // CPU fallback: Sum gradient over dimensions that were expanded (broadcast)
                let mut grad_data = vec![0.0; self.original_shape.iter().product()];
                let old_strides = RawTensor::compute_strides(&self.original_shape);
                let _new_strides = RawTensor::compute_strides(new_shape);

                for i in 0..out_grad.data.len() {
                    let mut old_idx = 0;
                    let mut rem = i;
                    for j in (0..new_shape.len()).rev() {
                        let coord = rem % new_shape[j];
                        rem /= new_shape[j];
                        // If this dimension was size 1, don't advance the index
                        if self.original_shape[j] != 1 {
                            old_idx += coord * old_strides[j];
                        }
                    }
                    grad_data[old_idx] += out_grad.data[i];
                }
                RawTensor::new(grad_data, &self.original_shape, false)
            }
            MovementOp::Pad { padding } => {
                // Try GPU path
                #[cfg(feature = "gpu")]
                {
                    if out_grad.device.is_gpu()
                        && let Some(grad_storage) = RawTensor::gpu_pad_backward(
                            &out_grad.data,
                            &self.original_shape,
                            &out_grad.shape,
                            padding,
                        )
                    {
                        return vec![Some(RawTensor::new_with_storage(
                            grad_storage,
                            &self.original_shape,
                            out_grad.device.clone(),
                            false,
                        ))];
                    }
                }

                // CPU fallback: Remove padding from gradient (extract center region)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides = RawTensor::compute_strides(&self.original_shape);
                let new_strides = RawTensor::compute_strides(&out_grad.shape);

                #[allow(clippy::too_many_arguments)]
                fn unpad_recursive(
                    result: &mut [f32],
                    grad: &[f32],
                    dim: usize,
                    old_shape: &[usize],
                    _new_shape: &[usize],
                    padding: &[(usize, usize)],
                    old_offset: usize,
                    new_offset: usize,
                    old_strides: &[usize],
                    new_strides: &[usize],
                ) {
                    if dim == old_shape.len() {
                        result[old_offset] = grad[new_offset];
                        return;
                    }

                    for i in 0..old_shape[dim] {
                        let new_i = i + padding[dim].0;
                        unpad_recursive(
                            result,
                            grad,
                            dim + 1,
                            old_shape,
                            _new_shape,
                            padding,
                            old_offset + i * old_strides[dim],
                            new_offset + new_i * new_strides[dim],
                            old_strides,
                            new_strides,
                        );
                    }
                }

                unpad_recursive(
                    &mut result,
                    &out_grad.data,
                    0,
                    &self.original_shape,
                    &out_grad.shape,
                    padding,
                    0,
                    0,
                    &old_strides,
                    &new_strides,
                );
                RawTensor::new(result, &self.original_shape, false)
            }
            MovementOp::Shrink { ranges } => {
                // Try GPU path
                #[cfg(feature = "gpu")]
                {
                    if out_grad.device.is_gpu()
                        && let Some(grad_storage) = RawTensor::gpu_shrink_backward(
                            &out_grad.data,
                            &self.original_shape,
                            &out_grad.shape,
                            ranges,
                        )
                    {
                        return vec![Some(RawTensor::new_with_storage(
                            grad_storage,
                            &self.original_shape,
                            out_grad.device.clone(),
                            false,
                        ))];
                    }
                }

                // CPU fallback: Pad gradient back to original size (inverse of shrink)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides = RawTensor::compute_strides(&self.original_shape);
                let new_strides = RawTensor::compute_strides(&out_grad.shape);

                #[allow(clippy::too_many_arguments)]
                fn unshrink_recursive(
                    result: &mut [f32],
                    grad: &[f32],
                    dim: usize,
                    old_shape: &[usize],
                    new_shape: &[usize],
                    ranges: &[(usize, usize)],
                    old_offset: usize,
                    new_offset: usize,
                    old_strides: &[usize],
                    new_strides: &[usize],
                ) {
                    if dim == old_shape.len() {
                        result[old_offset] = grad[new_offset];
                        return;
                    }

                    for i in 0..new_shape[dim] {
                        let old_i = i + ranges[dim].0; //Offset by range start
                        unshrink_recursive(
                            result,
                            grad,
                            dim + 1,
                            old_shape,
                            new_shape,
                            ranges,
                            old_offset + old_i * old_strides[dim],
                            new_offset + i * new_strides[dim],
                            old_strides,
                            new_strides,
                        );
                    }
                }

                unshrink_recursive(
                    &mut result,
                    &out_grad.data,
                    0,
                    &self.original_shape,
                    &out_grad.shape,
                    ranges,
                    0,
                    0,
                    &old_strides,
                    &new_strides,
                );
                RawTensor::new(result, &self.original_shape, false)
            }
            MovementOp::Stride { strides } => {
                // Try GPU path
                #[cfg(feature = "gpu")]
                {
                    if out_grad.device.is_gpu()
                        && let Some(grad_storage) = RawTensor::gpu_stride_backward(
                            &out_grad.data,
                            &self.original_shape,
                            &out_grad.shape,
                            strides,
                        )
                    {
                        return vec![Some(RawTensor::new_with_storage(
                            grad_storage,
                            &self.original_shape,
                            out_grad.device.clone(),
                            false,
                        ))];
                    }
                }

                // CPU fallback: Upsample gradient (inverse of stride/downsampling)
                let mut result = vec![0.0; self.original_shape.iter().product()];
                let old_strides_mem = RawTensor::compute_strides(&self.original_shape);
                let new_strides_mem = RawTensor::compute_strides(&out_grad.shape);

                #[allow(clippy::too_many_arguments)]
                fn unstride_recursive(
                    result: &mut [f32],
                    grad: &[f32],
                    dim: usize,
                    old_shape: &[usize],
                    new_shape: &[usize],
                    strides: &[usize],
                    old_offset: usize,
                    new_offset: usize,
                    old_strides: &[usize],
                    new_strides: &[usize],
                ) {
                    if dim == old_shape.len() {
                        result[old_offset] = grad[new_offset];
                        return;
                    }

                    for i in 0..new_shape[dim] {
                        let old_i = i * strides[dim];
                        if old_i < old_shape[dim] {
                            unstride_recursive(
                                result,
                                grad,
                                dim + 1,
                                old_shape,
                                new_shape,
                                strides,
                                old_offset + old_i * old_strides[dim],
                                new_offset + i * new_strides[dim],
                                old_strides,
                                new_strides,
                            );
                        }
                    }
                }

                unstride_recursive(
                    &mut result,
                    &out_grad.data,
                    0,
                    &self.original_shape,
                    &out_grad.shape,
                    strides,
                    0,
                    0,
                    &old_strides_mem,
                    &new_strides_mem,
                );
                RawTensor::new(result, &self.original_shape, false)
            }
        };

        vec![Some(grad_tensor)]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MovementGradFn {
            op: self.op.clone(),
            original_shape: self.original_shape.clone(),
        })
    }
}

// ===== MOVEMENT OPERATIONS =====
impl RawTensor {
    /// Reshape tensor to new shape (same number of elements)
    pub fn reshape(self_t: &Tensor, new_shape: &[usize]) -> Tensor {
        let (data, old_shape, req_grad, device) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        let old_size: usize = old_shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "Cannot reshape: size mismatch");

        // Reshape is a view operation - data stays in place, only shape changes
        // Use new_with_storage to preserve the device
        let out = Self::new_with_storage(data, new_shape, device, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Reshape {
                    new_shape: new_shape.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Internal permute implementation (no gradient tracking)
    ///
    /// Reorders axes according to the permutation specified by `axes`.
    /// For example, axes=[1,0] transposes a 2D matrix.
    fn permute_impl(self_t: &Tensor, axes: &[usize]) -> Tensor {
        let (data, shape, device) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };
        assert_eq!(axes.len(), shape.len(), "Axes length must match rank");

        let new_shape: Vec<usize> = axes.iter().map(|&i| shape[i]).collect();

        // Try GPU path first if available
        #[cfg(feature = "gpu")]
        if device.is_gpu()
            && let Some(storage) = RawTensor::gpu_permute(&data, &shape, &new_shape, axes)
        {
            return Self::new_with_storage(storage, &new_shape, device, false);
        }

        // CPU fallback
        let old_strides = Self::compute_strides(&shape);
        let mut new_data = vec![0.0; data.len()];

        let cpu_data = data.to_vec();

        // Helper: convert linear index to coordinates
        fn index_to_coords(idx: usize, shape: &[usize]) -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            let mut remaining = idx;
            for i in (0..shape.len()).rev() {
                coords[i] = remaining % shape[i];
                remaining /= shape[i];
            }
            coords
        }

        // Helper: convert coordinates to linear index using strides
        fn coords_to_index(coords: &[usize], strides: &[usize]) -> usize {
            coords.iter().zip(strides).map(|(c, s)| c * s).sum()
        }

        for (new_idx, val) in new_data.iter_mut().enumerate() {
            let new_coords = index_to_coords(new_idx, &new_shape);
            // Map new coordinates back to old coordinates
            let mut old_coords = vec![0; axes.len()];
            for (i, &ax) in axes.iter().enumerate() {
                old_coords[ax] = new_coords[i];
            }
            let old_idx = coords_to_index(&old_coords, &old_strides);
            *val = cpu_data[old_idx];
        }
        Self::new(new_data, &new_shape, false)
    }

    /// Permute (reorder) tensor axes
    ///
    /// # Arguments
    /// * `axes` - New ordering of axes (must be a valid permutation of 0..rank)
    pub fn permute(self_t: &Tensor, axes: &[usize]) -> Tensor {
        let req_grad = self_t.borrow().requires_grad;
        let old_shape = self_t.borrow().shape.clone();

        // Verify axes is a valid permutation
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        for (i, &ax) in sorted_axes.iter().enumerate() {
            assert_eq!(i, ax, "Invalid permutation axes");
        }

        let out = Self::permute_impl(self_t, axes);
        out.borrow_mut().requires_grad = req_grad;

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Permute {
                    axes: axes.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Expand (broadcast) tensor to larger shape
    ///
    /// Dimensions can only be expanded from size 1 to size N.
    /// Rank must remain the same.
    pub fn expand(self_t: &Tensor, new_shape: &[usize]) -> Tensor {
        let (data, old_shape, req_grad, device) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        assert_eq!(old_shape.len(), new_shape.len(), "Expand: rank must match");

        // Verify broadcasting rules: old dims must be 1 or equal to new
        for (old_d, new_d) in old_shape.iter().zip(new_shape) {
            assert!(
                *old_d == 1 || old_d == new_d,
                "Cannot expand dimension {} to {}",
                old_d,
                new_d
            );
        }

        let new_size: usize = new_shape.iter().product();
        const MAX_ALLOC: usize = 100_000_000;
        assert!(
            new_size <= MAX_ALLOC,
            "Expand would create tensor with {} elements (max: {}). Check shapes {:?} -> {:?}",
            new_size,
            MAX_ALLOC,
            old_shape,
            new_shape
        );

        // Try GPU path first if available
        #[cfg(feature = "gpu")]
        if device.is_gpu()
            && let Some(storage) = RawTensor::gpu_expand(&data, &old_shape, new_shape)
        {
            let out = Self::new_with_storage(storage, new_shape, device, req_grad);
            if req_grad {
                out.borrow_mut().parents = vec![self_t.clone()];
                out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                    op: MovementOp::Expand {
                        new_shape: new_shape.to_vec(),
                    },
                    original_shape: old_shape,
                }));
            }
            return out;
        }

        // CPU fallback
        let mut result = vec![0.0; new_size];
        let cpu_data = data.to_vec();

        // Broadcast by repeating values
        let old_strides = Self::compute_strides(&old_shape);
        let _new_strides = Self::compute_strides(new_shape);

        #[allow(clippy::needless_range_loop)]
        for i in 0..new_size {
            let mut old_idx = 0;
            let mut rem = i;
            for j in (0..new_shape.len()).rev() {
                let coord = rem % new_shape[j];
                rem /= new_shape[j];
                if old_shape[j] != 1 {
                    old_idx += coord * old_strides[j];
                }
            }
            result[i] = cpu_data[old_idx];
        }

        let out = Self::new(result, new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Expand {
                    new_shape: new_shape.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Pad tensor with zeros
    ///
    /// # Arguments
    /// * `padding` - For each dimension, (`left_pad`, `right_pad`)
    pub fn pad(self_t: &Tensor, padding: &[(usize, usize)]) -> Tensor {
        let (data, old_shape, req_grad, device) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        assert_eq!(
            padding.len(),
            old_shape.len(),
            "Padding length must match rank"
        );

        let new_shape: Vec<usize> = old_shape
            .iter()
            .zip(padding)
            .map(|(d, (l, r))| d + l + r)
            .collect();

        let new_size: usize = new_shape.iter().product();
        const MAX_ALLOC: usize = 100_000_000;
        assert!(
            new_size <= MAX_ALLOC,
            "Expand would create tensor with {} elements (max: {}). Check shapes {:?} -> {:?}",
            new_size,
            MAX_ALLOC,
            old_shape,
            new_shape
        );

        // Try GPU path first if available
        #[cfg(feature = "gpu")]
        if device.is_gpu()
            && let Some(storage) = RawTensor::gpu_pad(&data, &old_shape, &new_shape, padding)
        {
            let out = Self::new_with_storage(storage, &new_shape, device, req_grad);
            if req_grad {
                out.borrow_mut().parents = vec![self_t.clone()];
                out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                    op: MovementOp::Pad {
                        padding: padding.to_vec(),
                    },
                    original_shape: old_shape,
                }));
            }
            return out;
        }

        // CPU fallback
        let mut result = vec![0.0; new_size];
        let cpu_data = data.to_vec();

        // Copy old data into padded positions
        let old_strides = Self::compute_strides(&old_shape);
        let new_strides = Self::compute_strides(&new_shape);

        #[allow(clippy::too_many_arguments)]
        fn pad_recursive(
            result: &mut [f32],
            data: &[f32],
            dim: usize,
            old_shape: &[usize],
            _new_shape: &[usize],
            padding: &[(usize, usize)],
            old_offset: usize,
            new_offset: usize,
            old_strides: &[usize],
            new_strides: &[usize],
        ) {
            if dim == old_shape.len() {
                result[new_offset] = data[old_offset];
                return;
            }

            for i in 0..old_shape[dim] {
                let new_i = i + padding[dim].0;
                pad_recursive(
                    result,
                    data,
                    dim + 1,
                    old_shape,
                    _new_shape,
                    padding,
                    old_offset + i * old_strides[dim],
                    new_offset + new_i * new_strides[dim],
                    old_strides,
                    new_strides,
                );
            }
        }

        pad_recursive(
            &mut result,
            &cpu_data,
            0,
            &old_shape,
            &new_shape,
            padding,
            0,
            0,
            &old_strides,
            &new_strides,
        );

        let out = Self::new(result, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Pad {
                    padding: padding.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Extract a sub-region of the tensor
    ///
    /// # Arguments
    /// * `ranges` - For each dimension, (start, end) indices
    pub fn shrink(self_t: &Tensor, ranges: &[(usize, usize)]) -> Tensor {
        let (data, old_shape, req_grad, device) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        assert_eq!(
            ranges.len(),
            old_shape.len(),
            "Ranges length must match rank"
        );

        let new_shape: Vec<usize> = ranges.iter().map(|(start, end)| end - start).collect();

        // Try GPU path first if available
        #[cfg(feature = "gpu")]
        if device.is_gpu()
            && let Some(storage) = RawTensor::gpu_shrink(&data, &old_shape, &new_shape, ranges)
        {
            let out = Self::new_with_storage(storage, &new_shape, device, req_grad);
            if req_grad {
                out.borrow_mut().parents = vec![self_t.clone()];
                out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                    op: MovementOp::Shrink {
                        ranges: ranges.to_vec(),
                    },
                    original_shape: old_shape,
                }));
            }
            return out;
        }

        // CPU fallback
        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_size];
        let cpu_data = data.to_vec();

        let old_strides = Self::compute_strides(&old_shape);
        let new_strides = Self::compute_strides(&new_shape);

        #[allow(clippy::too_many_arguments)]
        fn shrink_recursive(
            result: &mut [f32],
            data: &[f32],
            dim: usize,
            old_shape: &[usize],
            new_shape: &[usize],
            ranges: &[(usize, usize)],
            old_offset: usize,
            new_offset: usize,
            old_strides: &[usize],
            new_strides: &[usize],
        ) {
            if dim == old_shape.len() {
                result[new_offset] = data[old_offset];
                return;
            }

            for i in 0..new_shape[dim] {
                let old_i = i + ranges[dim].0;
                shrink_recursive(
                    result,
                    data,
                    dim + 1,
                    old_shape,
                    new_shape,
                    ranges,
                    old_offset + old_i * old_strides[dim],
                    new_offset + i * new_strides[dim],
                    old_strides,
                    new_strides,
                );
            }
        }

        shrink_recursive(
            &mut result,
            &cpu_data,
            0,
            &old_shape,
            &new_shape,
            ranges,
            0,
            0,
            &old_strides,
            &new_strides,
        );

        let out = Self::new(result, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Shrink {
                    ranges: ranges.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Subsample tensor with specified strides
    ///
    /// Similar to slicing with step: array\[`::2`\] takes every other element
    pub fn stride_op(self_t: &Tensor, strides: &[usize]) -> Tensor {
        let (data, old_shape, req_grad, device) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        assert_eq!(
            strides.len(),
            old_shape.len(),
            "Strides length must match rank"
        );

        let new_shape: Vec<usize> = old_shape
            .iter()
            .zip(strides)
            .map(|(d, s)| d.div_ceil(*s))
            .collect();

        // Try GPU path first if available
        #[cfg(feature = "gpu")]
        if device.is_gpu()
            && let Some(storage) = RawTensor::gpu_stride(&data, &old_shape, &new_shape, strides)
        {
            let out = Self::new_with_storage(storage, &new_shape, device, req_grad);
            if req_grad {
                out.borrow_mut().parents = vec![self_t.clone()];
                out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                    op: MovementOp::Stride {
                        strides: strides.to_vec(),
                    },
                    original_shape: old_shape,
                }));
            }
            return out;
        }

        // CPU fallback
        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_size];
        let cpu_data = data.to_vec();

        let old_strides_mem = Self::compute_strides(&old_shape);
        let new_strides_mem = Self::compute_strides(&new_shape);

        #[allow(clippy::too_many_arguments)]
        fn stride_recursive(
            result: &mut [f32],
            data: &[f32],
            dim: usize,
            old_shape: &[usize],
            new_shape: &[usize],
            strides: &[usize],
            old_offset: usize,
            new_offset: usize,
            old_strides: &[usize],
            new_strides: &[usize],
        ) {
            if dim == old_shape.len() {
                result[new_offset] = data[old_offset];
                return;
            }

            for i in 0..new_shape[dim] {
                let old_i = i * strides[dim];
                stride_recursive(
                    result,
                    data,
                    dim + 1,
                    old_shape,
                    new_shape,
                    strides,
                    old_offset + old_i * old_strides[dim],
                    new_offset + i * new_strides[dim],
                    old_strides,
                    new_strides,
                );
            }
        }

        stride_recursive(
            &mut result,
            &cpu_data,
            0,
            &old_shape,
            &new_shape,
            strides,
            0,
            0,
            &old_strides_mem,
            &new_strides_mem,
        );

        let out = Self::new(result, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MovementGradFn {
                op: MovementOp::Stride {
                    strides: strides.to_vec(),
                },
                original_shape: old_shape,
            }));
        }
        out
    }

    /// Compute memory strides for row-major layout
    ///
    /// For shape [3, 4, 5], strides are [20, 5, 1]
    /// This tells us how many elements to skip to move one step in each dimension.
    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}
