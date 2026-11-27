//! `MaxPool2d` (`maxpool.wgsl`), `Embedding` (`embedding.wgsl`) and `ConvTranspose2d`'s
//! column steps (`conv_transpose.wgsl`), forward and backward (#67).
//!
//! Every kernel here is a gather, one thread per element it writes, so none needs atomics
//! and each sums in the CPU's order.

use super::{GpuKernels, Groups, Pass, pack_u32, run_passes, uniform_buffer};
use crate::gpu::{GpuBuffer, get_gpu_context};
use crate::index::{fetch, slot};
use crate::spec::{ConvTranspose2dSpec, Window2dSpec};

/// Elements of `spec`'s padded input, `[B*C, padded_h, padded_w]`.
fn pool_inputs(spec: &Window2dSpec) -> usize {
    spec.batch * spec.channels * spec.padded_h() * spec.padded_w()
}

/// Elements of `spec`'s output.
const fn pool_outputs(spec: &Window2dSpec) -> usize {
    spec.batch * spec.channels * spec.out_h * spec.out_w
}

/// The uniform of `maxpool.wgsl` for `spec`: 13 fields and 3 pad words.
fn pool_params(spec: &Window2dSpec) -> [u32; 16] {
    pack_u32(pool_inputs(spec));
    let mut p = [0u32; 16];
    for (i, v) in [
        spec.batch * spec.channels,
        spec.padded_h(),
        spec.padded_w(),
        spec.pad_h,
        spec.pad_w,
        spec.in_h,
        spec.in_w,
        spec.kernel_h,
        spec.kernel_w,
        spec.stride_h,
        spec.stride_w,
        spec.out_h,
        spec.out_w,
    ]
    .into_iter()
    .enumerate()
    {
        *slot(&mut p, i) = pack_u32(v);
    }
    p
}

/// The uniform of `conv_transpose.wgsl` for `spec`, with the bias flag.
fn conv_transpose_params(spec: &ConvTranspose2dSpec, has_bias: bool) -> [u32; 16] {
    let cols = spec.out_ch * spec.kernel_h * spec.kernel_w;
    pack_u32(spec.batch * spec.in_h * spec.in_w * cols);
    pack_u32(spec.batch * spec.out_ch * spec.out_h * spec.out_w);
    let mut p = [0u32; 16];
    for (i, v) in [
        spec.batch,
        spec.out_ch,
        spec.in_h,
        spec.in_w,
        spec.kernel_h,
        spec.kernel_w,
        spec.stride_h,
        spec.stride_w,
        spec.pad_h,
        spec.pad_w,
        spec.out_h,
        spec.out_w,
        usize::from(has_bias),
    ]
    .into_iter()
    .enumerate()
    {
        *slot(&mut p, i) = pack_u32(v);
    }
    p
}

impl GpuKernels {
    /// `MaxPool2d` forward over the padded `input`, `[planes, out_h, out_w]`.
    ///
    /// # Panics
    /// If `input` does not hold `spec`'s padded input.
    #[must_use]
    pub(crate) fn maxpool_forward(input: &GpuBuffer, spec: &Window2dSpec) -> Option<GpuBuffer> {
        assert_eq!(input.len(), pool_inputs(spec), "maxpool input length");
        let ctx = get_gpu_context()?;
        let out = GpuBuffer::uninit_of(input.dtype(), pool_outputs(spec))?; // every output written
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&pool_params(spec)));
        run_passes(
            ctx,
            "MaxPool2d",
            &[Pass {
                pipeline: &ctx.pipelines_for(input.dtype()).maxpool_forward,
                bindings: vec![
                    (0, input.binding()),
                    (1, out.binding()),
                    (3, params.as_entire_binding()),
                ],
                groups: Groups::cover(pool_outputs(spec)),
            }],
        );
        Some(out)
    }

    /// `MaxPool2d` backward: the gradient with respect to the padded `input`, each
    /// window's gradient added at its winner (recomputed by the forward's rule).
    ///
    /// # Panics
    /// If `input` or `grad` has the wrong length for `spec`, or they differ in dtype.
    #[must_use]
    pub(crate) fn maxpool_backward(
        input: &GpuBuffer,
        grad: &GpuBuffer,
        spec: &Window2dSpec,
    ) -> Option<GpuBuffer> {
        super::assert_same_dtype("maxpool_backward", &[input, grad]);
        assert_eq!(
            input.len(),
            pool_inputs(spec),
            "maxpool backward input length"
        );
        assert_eq!(
            grad.len(),
            pool_outputs(spec),
            "maxpool backward gradient length"
        );
        let ctx = get_gpu_context()?;
        let out = GpuBuffer::uninit_of(input.dtype(), pool_inputs(spec))?; // every element written
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&pool_params(spec)));
        run_passes(
            ctx,
            "MaxPool2d Backward",
            &[Pass {
                pipeline: &ctx.pipelines_for(input.dtype()).maxpool_backward,
                bindings: vec![
                    (0, input.binding()),
                    (1, out.binding()),
                    (2, grad.binding()),
                    (3, params.as_entire_binding()),
                ],
                groups: Groups::cover(pool_inputs(spec)),
            }],
        );
        Some(out)
    }

    /// `Embedding` forward: row `rows[i]` of `weight` (`[vocab, dim]`) for each `i`,
    /// `[rows.len(), dim]`.
    ///
    /// # Panics
    /// If `weight`'s length is not a multiple of `dim` or a row is out of range (the layer
    /// validates indices first, so this is a caller bug).
    #[must_use]
    pub(crate) fn embedding_forward(
        weight: &GpuBuffer,
        rows: &[usize],
        dim: usize,
    ) -> Option<GpuBuffer> {
        assert_eq!(weight.len() % dim, 0, "embedding weight length");
        let vocab = weight.len() / dim;
        assert!(
            rows.iter().all(|&r| r < vocab),
            "embedding row out of range"
        );
        let ctx = get_gpu_context()?;
        let idx: Vec<u32> = rows.iter().map(|&r| pack_u32(r)).collect();
        let idx = GpuBuffer::from_u32s(&idx)?;
        let n = rows.len() * dim;
        let out = GpuBuffer::uninit_of(weight.dtype(), n)?; // every element written
        let params = [pack_u32(rows.len()), pack_u32(dim), pack_u32(vocab), 0];
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&params));
        run_passes(
            ctx,
            "Embedding",
            &[Pass {
                pipeline: &ctx.pipelines_for(weight.dtype()).embedding_forward,
                bindings: vec![
                    (0, weight.binding()),
                    (1, out.binding()),
                    (2, idx.binding()),
                    (4, params.as_entire_binding()),
                ],
                groups: Groups::cover(n),
            }],
        );
        Some(out)
    }

    /// `Embedding` backward: the weight gradient `[vocab, dim]`, each row the sum of the
    /// output-gradient rows of its occurrences in `rows`, in ascending order.
    ///
    /// # Panics
    /// If `grad` is not `[rows.len(), dim]` or a row is out of range.
    #[must_use]
    pub(crate) fn embedding_backward(
        grad: &GpuBuffer,
        rows: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Option<GpuBuffer> {
        assert_eq!(grad.len(), rows.len() * dim, "embedding gradient length");
        // CSR of occurrences: row r's positions are positions[offsets[r]..offsets[r + 1]].
        let mut offsets = vec![0u32; vocab + 1];
        for &r in rows {
            assert!(r < vocab, "embedding row {r} out of range for {vocab}");
            *slot(&mut offsets, r + 1) += 1;
        }
        for r in 0..vocab {
            *slot(&mut offsets, r + 1) += fetch(&offsets, r);
        }
        let mut fill: Vec<u32> = offsets.clone();
        let mut positions = vec![0u32; rows.len()];
        for (p, &r) in rows.iter().enumerate() {
            let at = fetch(&fill, r);
            *slot(&mut positions, usize::try_from(at).expect("u32 fits usize")) = pack_u32(p);
            *slot(&mut fill, r) = at + 1;
        }
        let ctx = get_gpu_context()?;
        let positions = GpuBuffer::from_u32s(&positions)?;
        let offsets = GpuBuffer::from_u32s(&offsets)?;
        let n = vocab * dim;
        let out = GpuBuffer::uninit_of(grad.dtype(), n)?; // every element written
        let params = [pack_u32(rows.len()), pack_u32(dim), pack_u32(vocab), 0];
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&params));
        run_passes(
            ctx,
            "Embedding Backward",
            &[Pass {
                pipeline: &ctx.pipelines_for(grad.dtype()).embedding_backward,
                bindings: vec![
                    (0, grad.binding()),
                    (1, out.binding()),
                    (2, positions.binding()),
                    (3, offsets.binding()),
                    (4, params.as_entire_binding()),
                ],
                groups: Groups::cover(n),
            }],
        );
        Some(out)
    }

    /// `ConvTranspose2d`'s fold: `col` (`[B*H_in*W_in, C_out*K_h*K_w]`) summed into the
    /// output image `[B, C_out, H_out, W_out]`, plus `bias` (`[C_out]`) per channel.
    ///
    /// # Panics
    /// If `col` or `bias` has the wrong length for `spec`, or they differ in dtype.
    #[must_use]
    pub(crate) fn conv_transpose_col2im(
        col: &GpuBuffer,
        bias: Option<&GpuBuffer>,
        spec: &ConvTranspose2dSpec,
    ) -> Option<GpuBuffer> {
        let cols = spec.out_ch * spec.kernel_h * spec.kernel_w;
        assert_eq!(
            col.len(),
            spec.batch * spec.in_h * spec.in_w * cols,
            "conv_transpose col length"
        );
        if let Some(b) = bias {
            super::assert_same_dtype("conv_transpose_col2im", &[col, b]);
            assert_eq!(b.len(), spec.out_ch, "conv_transpose bias length");
        }
        let ctx = get_gpu_context()?;
        let n = spec.batch * spec.out_ch * spec.out_h * spec.out_w;
        let out = GpuBuffer::uninit_of(col.dtype(), n)?; // every element written
        let params = conv_transpose_params(spec, bias.is_some());
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&params));
        // Without a bias the flag is off and `col` stands in at binding 2, never read.
        let bias_binding = bias.unwrap_or(col);
        run_passes(
            ctx,
            "ConvTranspose2d Col2im",
            &[Pass {
                pipeline: &ctx.pipelines_for(col.dtype()).conv_transpose_col2im,
                bindings: vec![
                    (0, col.binding()),
                    (1, out.binding()),
                    (2, bias_binding.binding()),
                    (3, params.as_entire_binding()),
                ],
                groups: Groups::cover(n),
            }],
        );
        Some(out)
    }

    /// `ConvTranspose2d` backward's unfold: `grad` (`[B, C_out, H_out, W_out]`) as the
    /// column matrix `[B*H_in*W_in, C_out*K_h*K_w]`, zero where a tap fell in the padding.
    ///
    /// # Panics
    /// If `grad` has the wrong length for `spec`.
    #[must_use]
    pub(crate) fn conv_transpose_im2col(
        grad: &GpuBuffer,
        spec: &ConvTranspose2dSpec,
    ) -> Option<GpuBuffer> {
        assert_eq!(
            grad.len(),
            spec.batch * spec.out_ch * spec.out_h * spec.out_w,
            "conv_transpose gradient length"
        );
        let ctx = get_gpu_context()?;
        let n = spec.batch * spec.in_h * spec.in_w * spec.out_ch * spec.kernel_h * spec.kernel_w;
        let out = GpuBuffer::uninit_of(grad.dtype(), n)?; // every element written
        let params = conv_transpose_params(spec, false);
        let params = uniform_buffer(ctx, bytemuck::bytes_of(&params));
        run_passes(
            ctx,
            "ConvTranspose2d Im2col",
            &[Pass {
                pipeline: &ctx.pipelines_for(grad.dtype()).conv_transpose_im2col,
                bindings: vec![
                    (0, grad.binding()),
                    (1, out.binding()),
                    (3, params.as_entire_binding()),
                ],
                groups: Groups::cover(n),
            }],
        );
        Some(out)
    }
}
