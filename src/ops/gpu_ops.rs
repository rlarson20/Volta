//! GPU-accelerated tensor operations
//!
//! This module provides GPU implementations of tensor operations.
//! These are called automatically when tensors are on GPU storage.

use crate::device::Device;
#[cfg(feature = "gpu")]
use crate::device_policy::kernel_failed;
#[cfg(feature = "gpu")]
use crate::gpu::GpuKernels;
use crate::ops::BinaryOp;
use crate::ops::UnaryOp;
use crate::shape::Shape;
use crate::spec::PadValue;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

impl RawTensor {
    /// GPU-accelerated element-wise binary op (same shape; no broadcasting)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary(a: &Storage, b: &Storage, op: BinaryOp) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_op(buf_a, buf_b, op)?;

        // Return GPU storage without eager CPU transfer.
        // CPU access requires explicit to_vec() call.
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated element-wise addition
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_add(a: &Storage, b: &Storage) -> Option<Storage> {
        Self::gpu_binary(a, b, BinaryOp::Add)
    }

    /// GPU-accelerated element-wise multiplication
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mul(a: &Storage, b: &Storage) -> Option<Storage> {
        Self::gpu_binary(a, b, BinaryOp::Mul)
    }

    /// GPU-accelerated sum reduction - returns the scalar sum
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_sum_reduce(data: &Storage) -> Option<f32> {
        let buf = data.gpu_buffer()?;
        GpuKernels::sum(buf)
    }

    /// GPU-accelerated max reduction - returns `(max_value, max_index)`
    /// Uses GPU to find max value, then CPU to find the index
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_max_reduce(data: &Storage) -> Option<(f32, usize)> {
        let buf = data.gpu_buffer()?;
        let max_val = GpuKernels::max(buf)?;

        // Find the index of the max value on CPU (using cached data)
        let cpu_data = data.to_vec();
        let max_idx = cpu_data
            .iter()
            .enumerate()
            .find(|&(_, v)| (v - max_val).abs() < 1e-6)
            .map_or(0, |(i, _)| i);

        Some((max_val, max_idx))
    }

    /// GPU-accelerated mean reduction - returns the scalar mean
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mean_reduce(data: &Storage) -> Option<f32> {
        let buf = data.gpu_buffer()?;
        GpuKernels::mean(buf)
    }

    /// GPU-accelerated matrix multiplication
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_matmul(
        a: &Storage,
        b: &Storage,
        spec: &crate::spec::MatmulSpec,
    ) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::matmul(buf_a, buf_b, spec)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated unary operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_unary(input: &Storage, op: UnaryOp) -> Option<Storage> {
        let buf = input.gpu_buffer()?;

        let result = GpuKernels::unary_op(buf, op)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated unary backward operation
    ///
    /// Computes gradient for unary operations: `grad = out_grad * df/dx`
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_unary_backward(
        out_grad: &Storage,
        x: &Storage,
        op: UnaryOp,
    ) -> Option<Storage> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_x = x.gpu_buffer()?;

        let result = GpuKernels::unary_backward(buf_out, buf_x, op)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated binary backward operation for first input (a)
    ///
    /// Computes gradient with respect to first input for binary operations
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_backward_a(
        out_grad: &Storage,
        a: &Storage,
        b: &Storage,
        op: BinaryOp,
    ) -> Option<Storage> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_backward_a(buf_out, buf_a, buf_b, op)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated binary backward operation for second input (b)
    ///
    /// Computes gradient with respect to second input for binary operations
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_backward_b(
        out_grad: &Storage,
        a: &Storage,
        b: &Storage,
        op: BinaryOp,
    ) -> Option<Storage> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::binary_backward_b(buf_out, buf_a, buf_b, op)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated binary backward with RACE-FREE broadcasting support
    ///
    /// This uses a two-pass algorithm to avoid race conditions in gradient reduction.
    ///
    /// # Returns
    /// A tuple of (gradient wrt a, gradient wrt b)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_backward_broadcast_safe(
        out_grad: &Storage,
        a: &Storage,
        b: &Storage,
        op: BinaryOp,
        out_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Option<(Storage, Storage)> {
        let buf_out = out_grad.gpu_buffer()?;
        let buf_a = a.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let (a_grad, b_grad) = GpuKernels::binary_backward_broadcast_safe(
            buf_out, buf_a, buf_b, op, out_shape, a_shape, b_shape,
        )?;

        Some((
            Storage::from_gpu_buffer(a_grad),
            Storage::from_gpu_buffer(b_grad),
        ))
    }

    /// Run a binary op on the GPU when both inputs sit on the same GPU with equal shapes.
    ///
    /// Returns `(shape, storage, device)`, or `None` for the CPU path: CPU inputs, mixed
    /// devices (`edge()` panics on those), or differing shapes, since the GPU forward
    /// kernels don't broadcast (a gap, which `edge()` hands to the device policy).
    ///
    /// # Panics
    /// If the kernel runs for a supported case and returns no result.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_forward(
        self_t: &Tensor,
        other: &Tensor,
        op: BinaryOp,
    ) -> Option<(Shape, Storage, Device)> {
        let (shape_a, device_a, storage_a) = {
            let a = self_t.borrow();
            (a.shape.clone(), a.device.clone(), a.data.clone())
        };
        let (shape_b, device_b, storage_b) = {
            let b = other.borrow();
            (b.shape.clone(), b.device.clone(), b.data.clone())
        };

        if !device_a.is_gpu() || !device_b.is_gpu() || device_a != device_b {
            return None;
        }
        if shape_a != shape_b {
            return None;
        }

        let storage = Self::gpu_binary(&storage_a, &storage_b, op)
            .unwrap_or_else(|| kernel_failed(&format!("Binary({op:?})")));
        Some((shape_a, storage, device_a))
    }

    /// Not-gpu stub: the CPU path always runs.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_binary_forward(
        _self_t: &Tensor,
        _other: &Tensor,
        _op: BinaryOp,
    ) -> Option<(Shape, Storage, Device)> {
        None
    }

    /// GPU backward for a binary op, one gradient per parent.
    ///
    /// `None` means "use the CPU path": the tensors aren't all on a GPU, or the shapes
    /// differ (a gap). A partial result is never returned, since a `None` gradient for a
    /// parameter that needs one silently freezes it.
    ///
    /// # Panics
    /// If a same-shape kernel returns no result for a gradient that is required.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_binary_backward(
        op: BinaryOp,
        out_grad: &Self,
        x: &Self,
        y: &Self,
    ) -> Option<Vec<Option<Tensor>>> {
        if !(out_grad.device.is_gpu() && x.device.is_gpu() && y.device.is_gpu()) {
            return None;
        }
        // Gap until the broadcast backward bind groups are fixed (TODO.md, "GPU broadcast
        // backward is broken"): `None` sends differing shapes to the CPU path, and the
        // device policy applies when the gradient reaches its GPU parameter.
        if !(out_grad.shape == x.shape && out_grad.shape == y.shape) {
            return None;
        }
        Some(
            Self::gpu_binary_backward_same_shape(op, out_grad, x, y)
                .unwrap_or_else(|| kernel_failed(&format!("Binary({op:?})"))),
        )
    }

    /// Not-gpu stub: the CPU path always runs.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_binary_backward(
        _op: BinaryOp,
        _out_grad: &Self,
        _x: &Self,
        _y: &Self,
    ) -> Option<Vec<Option<Tensor>>> {
        None
    }

    /// Same-shape binary backward: one kernel per input that needs a gradient.
    #[cfg(feature = "gpu")]
    fn gpu_binary_backward_same_shape(
        op: BinaryOp,
        out_grad: &Self,
        x: &Self,
        y: &Self,
    ) -> Option<Vec<Option<Tensor>>> {
        let gx = if x.requires_grad {
            let storage = Self::gpu_binary_backward_a(&out_grad.data, &x.data, &y.data, op)?;
            Some(Self::new_with_storage(
                storage,
                x.shape.clone(),
                x.device.clone(),
                false,
            ))
        } else {
            None
        };
        let gy = if y.requires_grad {
            let storage = Self::gpu_binary_backward_b(&out_grad.data, &x.data, &y.data, op)?;
            Some(Self::new_with_storage(
                storage,
                y.shape.clone(),
                y.device.clone(),
                false,
            ))
        } else {
            None
        };
        Some(vec![gx, gy])
    }

    /// Broadcasting binary backward via the race-free two-pass kernels.
    #[cfg(feature = "gpu")]
    #[expect(
        dead_code,
        reason = "unreachable until the broadcast backward bind groups are fixed (TODO.md)"
    )]
    fn gpu_binary_backward_broadcast_shapes(
        op: BinaryOp,
        out_grad: &Self,
        x: &Self,
        y: &Self,
    ) -> Option<Vec<Option<Tensor>>> {
        let (grad_a, grad_b) = Self::gpu_binary_backward_broadcast_safe(
            &out_grad.data,
            &x.data,
            &y.data,
            op,
            out_grad.shape.truncate_rank(),
            x.shape.truncate_rank(),
            y.shape.truncate_rank(),
        )?;
        let gx = x
            .requires_grad
            .then(|| Self::new_with_storage(grad_a, x.shape.clone(), x.device.clone(), false));
        let gy = y
            .requires_grad
            .then(|| Self::new_with_storage(grad_b, y.shape.clone(), y.device.clone(), false));
        Some(vec![gx, gy])
    }

    /// GPU-accelerated permute operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_permute(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        axes: &[usize],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::permute(buf, old_shape, new_shape, axes)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated expand (broadcast) operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_expand(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::expand(buf, old_shape, new_shape)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated pad operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_pad(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
        value: PadValue,
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::pad(buf, old_shape, new_shape, padding, value)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated shrink operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_shrink(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        ranges: &[(usize, usize)],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::shrink(buf, old_shape, new_shape, ranges)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated stride operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_stride(
        data: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        strides: &[usize],
    ) -> Option<Storage> {
        let buf = data.gpu_buffer()?;
        let result = GpuKernels::stride(buf, old_shape, new_shape, strides)?;
        Some(Storage::from_gpu_buffer(result))
    }

    // Without the `gpu` feature there is no GPU storage, so every forward movement wrapper
    // answers "use the CPU path". They exist so `ops/movement.rs` needs no `cfg` of its own.

    /// Not-gpu stub for `gpu_permute`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_permute(
        _data: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _axes: &[usize],
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_expand`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_expand(
        _data: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_pad`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_pad(
        _data: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _padding: &[(usize, usize)],
        _value: PadValue,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_shrink`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_shrink(
        _data: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _ranges: &[(usize, usize)],
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_stride`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_stride(
        _data: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _strides: &[usize],
    ) -> Option<Storage> {
        None
    }

    /// GPU-accelerated matmul backward operation for gradient with respect to A
    ///
    /// Computes dA = grad @ B^T
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_matmul_backward_a(
        grad: &Storage,
        b: &Storage,
        spec: &crate::spec::MatmulSpec,
    ) -> Option<Storage> {
        let buf_grad = grad.gpu_buffer()?;
        let buf_b = b.gpu_buffer()?;

        let result = GpuKernels::matmul_backward_a(buf_grad, buf_b, spec)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated matmul backward operation for gradient with respect to B
    ///
    /// Computes dB = A^T @ grad
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_matmul_backward_b(
        a: &Storage,
        grad: &Storage,
        spec: &crate::spec::MatmulSpec,
    ) -> Option<Storage> {
        let buf_a = a.gpu_buffer()?;
        let buf_grad = grad.gpu_buffer()?;

        let result = GpuKernels::matmul_backward_b(buf_a, buf_grad, spec)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated sum backward operation
    ///
    /// Broadcasts scalar gradient to all elements
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_sum_backward(grad: f32, input_size: usize) -> Option<Storage> {
        let result = GpuKernels::sum_backward(grad, input_size)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated mean backward operation
    ///
    /// Broadcasts scalar gradient / count to all elements
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_mean_backward(grad: f32, input_size: usize) -> Option<Storage> {
        let result = GpuKernels::mean_backward(grad, input_size)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated max backward operation
    ///
    /// Sparse gradient - only max element receives gradient
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_max_backward(
        grad: f32,
        input_size: usize,
        max_index: usize,
    ) -> Option<Storage> {
        let result = GpuKernels::max_backward(grad, input_size, max_index)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated optimizer step operation
    ///
    /// Updates parameters in-place using gradients and optimizer state.
    /// Supports SGD, SGD with momentum, and Adam optimizers.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_optimizer_step(
        params: &mut Storage,
        grads: &Storage,
        state1: &mut Storage,
        state2: &mut Storage,
        opt_params: &crate::gpu::OptimizerStepParams,
    ) -> Option<()> {
        // The kernel writes `params`, `state1` and `state2` in place, so each goes through
        // the one GPU write path, which unshares it first.
        let buf_params = params.gpu_buffer_mut()?;
        let buf_grads = grads.gpu_buffer()?;
        let buf_state1 = state1.gpu_buffer_mut()?;
        let buf_state2 = state2.gpu_buffer_mut()?;
        GpuKernels::optimizer_step(buf_params, buf_grads, buf_state1, buf_state2, opt_params)
    }

    /// GPU-accelerated im2col transformation for convolution
    ///
    /// Transforms 4D input `(B, C, H, W)` into 2D matrix `(B*H_out*W_out, C*K_h*K_w)`.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_im2col(input: &Storage, spec: &crate::spec::Window2dSpec) -> Option<Storage> {
        let buf_input = input.gpu_buffer()?;
        let result = GpuKernels::im2col(buf_input, spec)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated col2im transformation for convolution gradient computation
    ///
    /// Transforms 2D matrix `(B*H_out*W_out, C*K_h*K_w)` back to 4D tensor `(B, C, H, W)`.
    /// This is the inverse of im2col, accumulating gradients from output positions.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_col2im(col: &Storage, spec: &crate::spec::Window2dSpec) -> Option<Storage> {
        let buf_col = col.gpu_buffer()?;
        let result = GpuKernels::col2im(buf_col, spec)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated direct convolution
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_direct_conv(
        input: &Storage,
        weight: &Storage,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        let buf_input = input.gpu_buffer()?;
        let buf_weight = weight.gpu_buffer()?;

        let result = GpuKernels::direct_conv(buf_input, buf_weight, spec)?;

        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated iGEMM convolution with tiled computation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_igemm(
        input: &Storage,
        weight: &Storage,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        let buf_input = input.gpu_buffer()?;
        let buf_weight = weight.gpu_buffer()?;

        let result = GpuKernels::igemm(buf_input, buf_weight, spec)?;

        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated convolution backward: gradient with respect to input
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_conv_backward_input(
        out_grad: &Storage,
        weight: &Storage,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        let buf_out_grad = out_grad.gpu_buffer()?;
        let buf_weight = weight.gpu_buffer()?;

        let result = crate::gpu::GpuKernels::conv_backward_input(buf_out_grad, buf_weight, spec)?;

        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated convolution backward: gradient with respect to weights
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_conv_backward_weight(
        out_grad: &Storage,
        input: &Storage,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        let buf_out_grad = out_grad.gpu_buffer()?;
        let buf_input = input.gpu_buffer()?;

        let result = crate::gpu::GpuKernels::conv_backward_weight(buf_out_grad, buf_input, spec)?;

        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated iGEMM convolution backward: gradient with respect to input
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_igemm_backward_input(
        out_grad: &Storage,
        weight: &Storage,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        let buf_out_grad = out_grad.gpu_buffer()?;
        let buf_weight = weight.gpu_buffer()?;

        let result = crate::gpu::GpuKernels::igemm_backward_input(buf_out_grad, buf_weight, spec)?;

        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated iGEMM convolution backward: gradient with respect to weights
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_igemm_backward_weight(
        out_grad: &Storage,
        input: &Storage,
        spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        let buf_out_grad = out_grad.gpu_buffer()?;
        let buf_input = input.gpu_buffer()?;

        let result = crate::gpu::GpuKernels::igemm_backward_weight(buf_out_grad, buf_input, spec)?;

        Some(Storage::from_gpu_buffer(result))
    }

    // ===== MOVEMENT BACKWARD OPERATIONS =====

    /// GPU-accelerated permute backward operation
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_permute_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        axes: &[usize],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::permute_backward(buf, old_shape, new_shape, axes)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated expand backward operation (sum over broadcast dimensions)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_expand_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::expand_backward(buf, old_shape, new_shape)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated pad backward operation (extract center region)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_pad_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::pad_backward(buf, old_shape, new_shape, padding)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated shrink backward operation (pad back to original size)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_shrink_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        ranges: &[(usize, usize)],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::shrink_backward(buf, old_shape, new_shape, ranges)?;
        Some(Storage::from_gpu_buffer(result))
    }

    /// GPU-accelerated stride backward operation (upsample gradient)
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_stride_backward(
        out_grad: &Storage,
        old_shape: &[usize],
        new_shape: &[usize],
        strides: &[usize],
    ) -> Option<Storage> {
        let buf = out_grad.gpu_buffer()?;
        let result = GpuKernels::stride_backward(buf, old_shape, new_shape, strides)?;
        Some(Storage::from_gpu_buffer(result))
    }

    // Without the `gpu` feature there is no GPU storage, so every wrapper below answers
    // "use the CPU path". They exist so call sites need no `cfg` of their own (ADDENDUM §9).

    /// Not-gpu stub for `gpu_add`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_add(_a: &Storage, _b: &Storage) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_mul`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_mul(_a: &Storage, _b: &Storage) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_sum_reduce`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_sum_reduce(_data: &Storage) -> Option<f32> {
        None
    }

    /// Not-gpu stub for `gpu_max_reduce`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_max_reduce(_data: &Storage) -> Option<(f32, usize)> {
        None
    }

    /// Not-gpu stub for `gpu_mean_reduce`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_mean_reduce(_data: &Storage) -> Option<f32> {
        None
    }

    /// Not-gpu stub for `gpu_matmul`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_matmul(
        _a: &Storage,
        _b: &Storage,
        _spec: &crate::spec::MatmulSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_unary`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_unary(_input: &Storage, _op: UnaryOp) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_unary_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_unary_backward(
        _out_grad: &Storage,
        _x: &Storage,
        _op: UnaryOp,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_matmul_backward_a`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_matmul_backward_a(
        _grad: &Storage,
        _b: &Storage,
        _spec: &crate::spec::MatmulSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_matmul_backward_b`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_matmul_backward_b(
        _a: &Storage,
        _grad: &Storage,
        _spec: &crate::spec::MatmulSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_sum_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_sum_backward(_grad: f32, _input_size: usize) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_mean_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_mean_backward(_grad: f32, _input_size: usize) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_max_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_max_backward(
        _grad: f32,
        _input_size: usize,
        _max_index: usize,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_im2col`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_im2col(
        _input: &Storage,
        _spec: &crate::spec::Window2dSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_col2im`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_col2im(_col: &Storage, _spec: &crate::spec::Window2dSpec) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_direct_conv`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_direct_conv(
        _input: &Storage,
        _weight: &Storage,
        _spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_igemm`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_igemm(
        _input: &Storage,
        _weight: &Storage,
        _spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_conv_backward_input`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_conv_backward_input(
        _out_grad: &Storage,
        _weight: &Storage,
        _spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_conv_backward_weight`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_conv_backward_weight(
        _out_grad: &Storage,
        _input: &Storage,
        _spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_igemm_backward_input`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_igemm_backward_input(
        _out_grad: &Storage,
        _weight: &Storage,
        _spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_igemm_backward_weight`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_igemm_backward_weight(
        _out_grad: &Storage,
        _input: &Storage,
        _spec: &crate::spec::Conv2dSpec,
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_permute_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_permute_backward(
        _out_grad: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _axes: &[usize],
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_expand_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_expand_backward(
        _out_grad: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_pad_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_pad_backward(
        _out_grad: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _padding: &[(usize, usize)],
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_shrink_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_shrink_backward(
        _out_grad: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _ranges: &[(usize, usize)],
    ) -> Option<Storage> {
        None
    }

    /// Not-gpu stub for `gpu_stride_backward`.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn gpu_stride_backward(
        _out_grad: &Storage,
        _old_shape: &[usize],
        _new_shape: &[usize],
        _strides: &[usize],
    ) -> Option<Storage> {
        None
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::gpu::is_gpu_available;
    use crate::storage::Storage;
    use crate::tensor::Tensor;

    #[test]
    fn gpu_add_returns_gpu_storage_without_eager_transfer() {
        if !is_gpu_available() {
            // Skip on machines without a usable GPU.
            return;
        }

        let a = Storage::gpu(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Storage::gpu(vec![5.0, 6.0, 7.0, 8.0]);

        let out = RawTensor::gpu_add(&a, &b).expect("gpu_add failed");

        // Must be GPU-backed storage.
        assert!(out.is_gpu());

        // The host copy is NOT eagerly read (lazy transfer).
        // to_vec() fetches from GPU buffer when cache is None.
        assert_eq!(out.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    /// Deterministic, non-constant, tie-free operands (`x_i = .37i + .5`, `y_j = .53j + .9`
    /// never coincide for i < 12, so Max has no ties) and positive, so Div is well-defined.
    fn operand(len: usize, scale: f32, offset: f32, shape: &[usize]) -> Tensor {
        let data = (0..len).map(|i| i as f32 * scale + offset).collect();
        RawTensor::new(data, shape, true)
    }

    /// Assert `gpu_binary_backward` returns `Some` (not a silent CPU fallback) for each
    /// shape pair, and agrees with the CPU backward.
    fn assert_gpu_backward_matches_cpu(cases: &[(&[usize], &[usize])]) {
        let Some(gpu) = Device::gpu() else {
            return;
        };
        if !is_gpu_available() {
            return;
        }
        for op in BinaryOp::ALL.iter().copied().filter(|op| !op.is_trivial()) {
            for &(shape_a, shape_b) in cases {
                let numel = |s: &[usize]| s.iter().product::<usize>();
                let out_shape = RawTensor::broadcast_shape(shape_a, shape_b);

                // CPU reference: the full graph, backward from an all-ones output grad.
                let xa = operand(numel(shape_a), 0.37, 0.5, shape_a);
                let yb = operand(numel(shape_b), 0.53, 0.9, shape_b);
                RawTensor::binary_op(&xa, &yb, op)
                    .sum()
                    .into_loss()
                    .expect("scalar loss")
                    .backward();
                let cpu_x = xa.grad().expect("cpu grad for x");
                let cpu_y = yb.grad().expect("cpu grad for y");

                // GPU: call the backward directly so a fallback shows up as `None`.
                let gx =
                    RawTensor::to_device(&operand(numel(shape_a), 0.37, 0.5, shape_a), gpu.clone());
                let gy =
                    RawTensor::to_device(&operand(numel(shape_b), 0.53, 0.9, shape_b), gpu.clone());
                let ones = RawTensor::new(vec![1.0; numel(&out_shape)], &out_shape, false);
                let out_grad = RawTensor::to_device(&ones, gpu.clone());
                let grads = RawTensor::gpu_binary_backward(
                    op,
                    &out_grad.borrow(),
                    &gx.borrow(),
                    &gy.borrow(),
                )
                .unwrap_or_else(|| panic!("{op:?} {shape_a:?}x{shape_b:?}: fell back to CPU"));

                for (name, got, want) in [
                    ("x", grads.first().and_then(Option::as_ref), &cpu_x),
                    ("y", grads.get(1).and_then(Option::as_ref), &cpu_y),
                ] {
                    let got = got
                        .unwrap_or_else(|| panic!("{op:?}: missing {name} gradient"))
                        .borrow()
                        .data
                        .to_vec();
                    assert_eq!(got.len(), want.len(), "{op:?} {name} length");
                    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                            "{op:?} {shape_a:?}x{shape_b:?} d{name}[{i}]: gpu={g} cpu={w}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gpu_binary_backward_same_shape_matches_cpu() {
        assert_gpu_backward_matches_cpu(&[(&[2, 3], &[2, 3]), (&[4, 2, 3], &[4, 2, 3])]);
    }

    #[test]
    #[ignore = "GPU broadcast backward (notes/TODO.md): the two-pass pass-1 bind group has 6 \
                bindings but wgpu's auto layout keeps only those the entry point uses (3 for Add)"]
    fn gpu_binary_backward_broadcast_matches_cpu() {
        assert_gpu_backward_matches_cpu(&[
            (&[3, 1], &[1, 4]),
            (&[2, 3], &[3]),
            (&[4, 2, 3], &[2, 1]),
        ]);
    }

    fn sgd_params(lr: f32) -> crate::gpu::OptimizerStepParams {
        crate::gpu::OptimizerStepParams {
            op: 0,
            lr,
            beta1: 0.0,
            beta2: 0.0,
            t: 0.0,
            eps: 0.0,
            weight_decay: 0.0,
            padding: 0.0,
        }
    }

    /// H3: a snapshot (a clone of the storage) keeps its value across an in-place optimizer
    /// step, the way a CPU snapshot does under copy-on-write. The snapshot is deliberately
    /// not read until after the step: clones share one GPU buffer, so before the fix the
    /// step rewrote the snapshot's data underneath it.
    #[test]
    fn gpu_optimizer_step_does_not_change_an_unread_snapshot() {
        if !is_gpu_available() {
            return;
        }
        let mut param = Storage::gpu(vec![1.0, 2.0, 3.0, 4.0]);
        let grad = Storage::gpu(vec![1.0, 1.0, 1.0, 1.0]);
        let (mut s1, mut s2) = (Storage::gpu(vec![0.0; 4]), Storage::gpu(vec![0.0; 4]));
        let snapshot = param.clone();

        RawTensor::gpu_optimizer_step(&mut param, &grad, &mut s1, &mut s2, &sgd_params(0.5))
            .expect("optimizer step");

        assert_eq!(param.to_vec(), vec![0.5, 1.5, 2.5, 3.5], "param is updated");
        assert_eq!(
            snapshot.to_vec(),
            vec![1.0, 2.0, 3.0, 4.0],
            "the snapshot must keep its value"
        );
    }

    /// The same snapshot read on the host first. This passed before the fix too, only because
    /// the snapshot's private host cache held the old values while its buffer was already
    /// overwritten: it guards the cache and the buffer staying consistent, not the bug above.
    #[test]
    fn gpu_optimizer_step_does_not_change_a_read_snapshot() {
        if !is_gpu_available() {
            return;
        }
        let mut param = Storage::gpu(vec![1.0, 2.0, 3.0, 4.0]);
        let grad = Storage::gpu(vec![1.0, 1.0, 1.0, 1.0]);
        let (mut s1, mut s2) = (Storage::gpu(vec![0.0; 4]), Storage::gpu(vec![0.0; 4]));
        let snapshot = param.clone();
        assert_eq!(snapshot.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        RawTensor::gpu_optimizer_step(&mut param, &grad, &mut s1, &mut s2, &sgd_params(0.5))
            .expect("optimizer step");

        assert_eq!(snapshot.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(param.to_vec(), vec![0.5, 1.5, 2.5, 3.5]);
    }

    /// H3: the step must also drop a host cache the parameter itself read before it.
    #[test]
    fn gpu_optimizer_step_refreshes_the_params_own_host_read() {
        if !is_gpu_available() {
            return;
        }
        let mut param = Storage::gpu(vec![1.0, 2.0]);
        let grad = Storage::gpu(vec![2.0, 2.0]);
        let (mut s1, mut s2) = (Storage::gpu(vec![0.0; 2]), Storage::gpu(vec![0.0; 2]));
        assert_eq!(param.to_vec(), vec![1.0, 2.0]);

        RawTensor::gpu_optimizer_step(&mut param, &grad, &mut s1, &mut s2, &sgd_params(0.5))
            .expect("optimizer step");

        assert_eq!(param.to_vec(), vec![0.0, 1.0]);
    }
}
