//! Rank-8 strided gather (`strided.wgsl`) and reductions along one axis (`reduce_dim.wgsl`).
//!
//! These cover the layout and reduction cases the older rank-4 movement kernels and the
//! full-tensor reductions don't: broadcasting a binary operand, `sum_dim`/`mean_dim`/
//! `max_dim` and their backward passes, and permutes above rank 4 (#67).

use super::{GpuKernels, Groups, Pass, pack_u32, run_kernel, run_passes, uniform_buffer};
use crate::gpu::{GpuBuffer, get_gpu_context};
use crate::index::{fetch, slot};

/// The highest rank the strided gather takes.
pub(crate) const MAX_GATHER_RANK: usize = 8;

/// An axis at least this long reduces with one workgroup per output; a shorter one with
/// one thread per output. Both sum pairwise, so the choice affects speed, not the result
/// beyond rounding.
const ROWS_FROM: usize = 128;

/// Uniform of `strided.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    /// Result dims, padded with 1s.
    shape: [u32; MAX_GATHER_RANK],
    /// Input strides per result axis, padded with 0s.
    strides: [u32; MAX_GATHER_RANK],
    /// Result rank.
    rank: u32,
    /// Input index of the result's first element.
    offset: u32,
    /// Result elements.
    n: u32,
    /// Pads the struct to a multiple of 16 bytes.
    pad: u32,
}

/// Uniform of `reduce_dim.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AxisParams {
    /// Product of the dims before the axis.
    outer: u32,
    /// The axis length.
    dim: u32,
    /// Product of the dims after the axis.
    inner: u32,
    /// Each sum is divided by this.
    divisor: f32,
}

/// An input viewed as `[outer, dim, inner]`, reduced over `dim`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AxisView {
    /// Product of the dims before the axis.
    pub outer: usize,
    /// The axis length.
    pub dim: usize,
    /// Product of the dims after the axis.
    pub inner: usize,
}

impl AxisView {
    /// `shape` reduced over `axis`.
    ///
    /// # Panics
    /// If `axis` is not less than the rank.
    pub(crate) fn of(shape: &[usize], axis: usize) -> Self {
        assert!(axis < shape.len(), "axis {axis} out of range for {shape:?}");
        Self {
            outer: shape.iter().take(axis).product(),
            dim: fetch(shape, axis),
            inner: shape.iter().skip(axis + 1).product(),
        }
    }

    /// Elements of the reduced output.
    const fn outputs(self) -> usize {
        self.outer * self.inner
    }

    /// Elements of the input.
    const fn inputs(self) -> usize {
        self.outer * self.dim * self.inner
    }

    /// The uniform, with `divisor`.
    fn params(self, divisor: f32) -> AxisParams {
        // The kernels index the input with u32 arithmetic.
        pack_u32(self.inputs());
        AxisParams {
            outer: pack_u32(self.outer),
            dim: pack_u32(self.dim),
            inner: pack_u32(self.inner),
            divisor,
        }
    }

    /// Whether the axis is long enough for one workgroup per output.
    const fn rows(self) -> bool {
        self.dim >= ROWS_FROM
    }
}

/// Row-major element strides of `shape`.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        *slot(&mut strides, i) = fetch(&strides, i + 1) * fetch(shape, i + 1);
    }
    strides
}

/// `in_shape`'s strides as seen from `out_shape` when `in_shape` broadcasts to it (aligned
/// on the right, as NumPy does): 0 on every axis `in_shape` lacks or holds at size 1.
///
/// # Panics
/// If `in_shape` does not broadcast to `out_shape`.
pub(crate) fn broadcast_strides(in_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    assert!(
        in_shape.len() <= out_shape.len(),
        "{in_shape:?} has more axes than {out_shape:?}"
    );
    let lead = out_shape.len() - in_shape.len();
    let own = contiguous_strides(in_shape);
    (0..out_shape.len())
        .map(|i| {
            if i < lead {
                return 0;
            }
            let (d_in, d_out) = (fetch(in_shape, i - lead), fetch(out_shape, i));
            if d_in == d_out {
                fetch(&own, i - lead)
            } else {
                assert_eq!(d_in, 1, "{in_shape:?} does not broadcast to {out_shape:?}");
                0
            }
        })
        .collect()
}

impl GpuKernels {
    /// `result[i] = input[offset + Σ coord_d(i) · strides[d]]` over the row-major
    /// coordinates of `i` in `shape`.
    ///
    /// # Panics
    /// If `shape` and `strides` differ in rank or the rank exceeds [`MAX_GATHER_RANK`], or
    /// an index of the gather falls outside `input` (a caller bug).
    #[must_use]
    pub(crate) fn gather(
        input: &GpuBuffer,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Option<GpuBuffer> {
        assert_eq!(shape.len(), strides.len(), "gather shape and strides rank");
        assert!(
            shape.len() <= MAX_GATHER_RANK,
            "gather of rank {} exceeds {MAX_GATHER_RANK}",
            shape.len()
        );
        let n: usize = shape.iter().product();
        let last = offset
            + shape
                .iter()
                .zip(strides)
                .map(|(d, s)| (d - 1) * s)
                .sum::<usize>();
        assert!(
            last < input.len(),
            "gather reads index {last} of a {}-element buffer",
            input.len()
        );
        let ctx = get_gpu_context()?;
        let mut params = GatherParams {
            shape: [1; MAX_GATHER_RANK],
            strides: [0; MAX_GATHER_RANK],
            rank: pack_u32(shape.len()),
            offset: pack_u32(offset),
            n: pack_u32(n),
            pad: 0,
        };
        for (i, (&d, &s)) in shape.iter().zip(strides).enumerate() {
            *slot(&mut params.shape, i) = pack_u32(d);
            *slot(&mut params.strides, i) = pack_u32(s);
        }
        let result = GpuBuffer::uninit_of(input.dtype(), n)?; // every element written
        run_kernel(
            "Gather",
            &ctx.pipelines_for(input.dtype()).gather,
            &[input, &result],
            bytemuck::bytes_of(&params),
            Groups::cover(n),
        )?;
        Some(result)
    }

    /// `input` (shaped `in_shape`) broadcast to `out_shape`, as a new contiguous buffer.
    ///
    /// # Panics
    /// As [`broadcast_strides`] and [`Self::gather`].
    #[must_use]
    pub(crate) fn broadcast_to(
        input: &GpuBuffer,
        in_shape: &[usize],
        out_shape: &[usize],
    ) -> Option<GpuBuffer> {
        Self::gather(input, out_shape, &broadcast_strides(in_shape, out_shape), 0)
    }

    /// `input` (shaped `old_shape`) with its axes reordered: result axis `i` is input axis
    /// `axes[i]`. Any rank up to [`MAX_GATHER_RANK`].
    ///
    /// # Panics
    /// If `axes` is not a permutation of `0..old_shape.len()`, or as [`Self::gather`].
    #[must_use]
    pub(crate) fn permute_any(
        input: &GpuBuffer,
        old_shape: &[usize],
        axes: &[usize],
    ) -> Option<GpuBuffer> {
        assert_eq!(axes.len(), old_shape.len(), "permute axes rank");
        let own = contiguous_strides(old_shape);
        let shape: Vec<usize> = axes.iter().map(|&a| fetch(old_shape, a)).collect();
        let strides: Vec<usize> = axes.iter().map(|&a| fetch(&own, a)).collect();
        Self::gather(input, &shape, &strides, 0)
    }

    /// The sum over `view`'s axis, divided by `divisor` (1 for `sum_dim`, the axis length
    /// for `mean_dim`), shaped `[outer, inner]`.
    ///
    /// # Panics
    /// If `input` does not hold `view.inputs()` elements.
    #[must_use]
    pub(crate) fn sum_dim(input: &GpuBuffer, view: AxisView, divisor: f32) -> Option<GpuBuffer> {
        Self::axis_reduce(input, view, divisor, false)
    }

    /// The maximum over `view`'s axis, as `torch.max(x, dim)`'s values, shaped
    /// `[outer, inner]`.
    ///
    /// # Panics
    /// If `input` does not hold `view.inputs()` elements.
    #[must_use]
    pub(crate) fn max_dim(input: &GpuBuffer, view: AxisView) -> Option<GpuBuffer> {
        Self::axis_reduce(input, view, 1.0, true)
    }

    /// `sum_dim` or `max_dim` forward.
    fn axis_reduce(
        input: &GpuBuffer,
        view: AxisView,
        divisor: f32,
        max: bool,
    ) -> Option<GpuBuffer> {
        assert_eq!(input.len(), view.inputs(), "axis reduction input length");
        let ctx = get_gpu_context()?;
        let p = ctx.pipelines_for(input.dtype());
        let out = GpuBuffer::uninit_of(input.dtype(), view.outputs())?; // every output written
        let (pipeline, groups) = match (max, view.rows()) {
            (false, false) => (&p.sum_dim_cols, Groups::cover(view.outputs())),
            (false, true) => (&p.sum_dim_rows, Groups::Flat(pack_u32(view.outputs()))),
            (true, false) => (&p.max_dim_cols, Groups::cover(view.outputs())),
            (true, true) => (&p.max_dim_rows, Groups::Flat(pack_u32(view.outputs()))),
        };
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&view.params(divisor)));
        run_passes(
            ctx,
            "Axis Reduce",
            &[Pass {
                pipeline,
                bindings: vec![
                    (0, input.binding()),
                    (1, out.binding()),
                    (3, params.as_entire_binding()),
                ],
                groups,
            }],
        );
        Some(out)
    }

    /// Backward of [`Self::sum_dim`]: `grad` (`[outer, inner]`) divided by `divisor` and
    /// broadcast along the axis, shaped like the input.
    ///
    /// # Panics
    /// If `grad` does not hold `view.outputs()` elements.
    #[must_use]
    pub(crate) fn sum_dim_backward(
        grad: &GpuBuffer,
        view: AxisView,
        divisor: f32,
    ) -> Option<GpuBuffer> {
        assert_eq!(
            grad.len(),
            view.outputs(),
            "sum_dim backward gradient length"
        );
        let ctx = get_gpu_context()?;
        let out = GpuBuffer::uninit_of(grad.dtype(), view.inputs())?; // every element written
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&view.params(divisor)));
        run_passes(
            ctx,
            "Sum Dim Backward",
            &[Pass {
                pipeline: &ctx.pipelines_for(grad.dtype()).sum_dim_backward,
                bindings: vec![
                    (1, out.binding()),
                    (2, grad.binding()),
                    (3, params.as_entire_binding()),
                ],
                groups: Groups::cover(view.inputs()),
            }],
        );
        Some(out)
    }

    /// Backward of [`Self::max_dim`]: each row's gradient at its maximum (recomputed from
    /// `input` by the same rule), zero elsewhere, shaped like the input.
    ///
    /// # Panics
    /// If `input` or `grad` has the wrong length for `view`, or they differ in dtype.
    #[must_use]
    pub(crate) fn max_dim_backward(
        input: &GpuBuffer,
        grad: &GpuBuffer,
        view: AxisView,
    ) -> Option<GpuBuffer> {
        super::assert_same_dtype("max_dim_backward", &[input, grad]);
        assert_eq!(input.len(), view.inputs(), "max_dim backward input length");
        assert_eq!(
            grad.len(),
            view.outputs(),
            "max_dim backward gradient length"
        );
        let ctx = get_gpu_context()?;
        let p = ctx.pipelines_for(input.dtype());
        // Every row writes all of its elements.
        let out = GpuBuffer::uninit_of(input.dtype(), view.inputs())?;
        let (pipeline, groups) = if view.rows() {
            (
                &p.max_dim_backward_rows,
                Groups::Flat(pack_u32(view.outputs())),
            )
        } else {
            (&p.max_dim_backward_cols, Groups::cover(view.outputs()))
        };
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&view.params(1.0)));
        run_passes(
            ctx,
            "Max Dim Backward",
            &[Pass {
                pipeline,
                bindings: vec![
                    (0, input.binding()),
                    (1, out.binding()),
                    (2, grad.binding()),
                    (3, params.as_entire_binding()),
                ],
                groups,
            }],
        );
        Some(out)
    }

    /// `input` (shaped `in_shape`) summed down to `out_shape`, the reverse of broadcasting
    /// `out_shape` to `in_shape`: every axis `out_shape` lacks or holds at size 1 is summed.
    /// One [`Self::sum_dim`] pass per summed axis; `out_shape` equal to `in_shape` is a copy.
    ///
    /// # Panics
    /// If `out_shape` does not broadcast to `in_shape`.
    #[must_use]
    pub(crate) fn sum_to_shape(
        input: &GpuBuffer,
        in_shape: &[usize],
        out_shape: &[usize],
    ) -> Option<GpuBuffer> {
        // Validates the broadcast.
        let _ = broadcast_strides(out_shape, in_shape);
        let lead = in_shape.len() - out_shape.len();
        let mut shape = in_shape.to_vec();
        let mut current: Option<GpuBuffer> = None;
        for axis in 0..in_shape.len() {
            let target = if axis < lead {
                1
            } else {
                fetch(out_shape, axis - lead)
            };
            if fetch(&shape, axis) == target {
                continue;
            }
            let view = AxisView::of(&shape, axis);
            let src = current.as_ref().unwrap_or(input);
            let next = Self::sum_dim(src, view, 1.0)?;
            *slot(&mut shape, axis) = 1;
            current = Some(next);
        }
        match current {
            Some(buf) => Some(buf),
            None => input.copy_region(0, input.len()),
        }
    }
}
