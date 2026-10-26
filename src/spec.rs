//! Validated geometry for ops whose arguments used to be positional `usize`s.
//!
//! A spec is computed once, in forward, from the input and weight [`Shape`]s.
//! Backward and the GPU kernel wrappers read the spec instead of re-deriving
//! dimensions from tensors (the B4 failure mode), so a transposed
//! `batch_size`/`channels` pair is a field-name error, not a silent one.

use crate::error::{Result, VoltaError};
use crate::shape::{Axis, Shape};

/// Reads the four dims of an NCHW-style shape, or says which tensor was not 4D.
fn dims4(shape: &Shape, what: &str) -> Result<[usize; 4]> {
    if shape.rank() != 4 {
        return Err(VoltaError::InvalidParameter(format!(
            "{what} must be 4D, got shape {:?}",
            shape.truncate_rank()
        )));
    }
    Ok([
        shape.dim(Axis(0))?,
        shape.dim(Axis(1))?,
        shape.dim(Axis(2))?,
        shape.dim(Axis(3))?,
    ])
}

fn check_stride(stride: (usize, usize)) -> Result<()> {
    if stride.0 == 0 || stride.1 == 0 {
        return Err(VoltaError::InvalidParameter(format!(
            "stride must be non-zero, got {stride:?}"
        )));
    }
    Ok(())
}

/// Output extent of a sliding window: `(padded - kernel) / stride + 1`.
fn window_out(axis: &str, input: usize, kernel: usize, stride: usize, pad: usize) -> Result<usize> {
    let padded = input + 2 * pad;
    if kernel > padded {
        return Err(VoltaError::InvalidParameter(format!(
            "kernel {axis} {kernel} is larger than padded input {axis} {padded} \
             (input {input}, padding {pad})"
        )));
    }
    Ok((padded - kernel) / stride + 1)
}

/// Validated geometry of a `Conv2d` forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Conv2dSpec {
    pub batch: usize,
    pub in_ch: usize,
    pub out_ch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl Conv2dSpec {
    /// `input` is `[B, C, H, W]`, `weight` is `[O, C, Kh, Kw]`.
    pub(crate) fn new(
        input: &Shape,
        weight: &Shape,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        let [batch, in_ch, in_h, in_w] = dims4(input, "input")?;
        let [out_ch, w_in_ch, kernel_h, kernel_w] = dims4(weight, "weight")?;
        if w_in_ch != in_ch {
            return Err(VoltaError::InvalidParameter(format!(
                "channel mismatch: input has {in_ch} channels but weight expects {w_in_ch}"
            )));
        }
        check_stride(stride)?;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let out_h = window_out("height", in_h, kernel_h, stride_h, pad_h)?;
        let out_w = window_out("width", in_w, kernel_w, stride_w, pad_w)?;
        Ok(Self {
            batch,
            in_ch,
            out_ch,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
        })
    }

    pub(crate) fn padded_h(&self) -> usize {
        self.in_h + 2 * self.pad_h
    }

    pub(crate) fn padded_w(&self) -> usize {
        self.in_w + 2 * self.pad_w
    }

    pub(crate) fn input_dims(&self) -> [usize; 4] {
        [self.batch, self.in_ch, self.in_h, self.in_w]
    }

    pub(crate) fn weight_dims(&self) -> [usize; 4] {
        [self.out_ch, self.in_ch, self.kernel_h, self.kernel_w]
    }

    pub(crate) fn output_dims(&self) -> [usize; 4] {
        [self.batch, self.out_ch, self.out_h, self.out_w]
    }

    /// The window over the zero-padded input, with no further padding.
    ///
    /// `im2col`/`col2im` run on the already-padded tensor, so they see
    /// `padded_h x padded_w` input and the same kernel and stride.
    pub(crate) fn padded_window(&self) -> Window2dSpec {
        Window2dSpec {
            batch: self.batch,
            channels: self.in_ch,
            in_h: self.padded_h(),
            in_w: self.padded_w(),
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            pad_h: 0,
            pad_w: 0,
            out_h: self.out_h,
            out_w: self.out_w,
        }
    }
}

/// Validated geometry of a `ConvTranspose2d` forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConvTranspose2dSpec {
    pub batch: usize,
    pub in_ch: usize,
    pub out_ch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl ConvTranspose2dSpec {
    /// `input` is `[B, C_in, H, W]`, `weight` is `[C_in, C_out, Kh, Kw]`.
    pub(crate) fn new(
        input: &Shape,
        weight: &Shape,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        let [batch, in_ch, in_h, in_w] = dims4(input, "input")?;
        let [w_in_ch, out_ch, kernel_h, kernel_w] = dims4(weight, "weight")?;
        if w_in_ch != in_ch {
            return Err(VoltaError::InvalidParameter(format!(
                "channel mismatch: input has {in_ch} channels but weight expects {w_in_ch}"
            )));
        }
        check_stride(stride)?;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let out_h = transpose_out("height", in_h, kernel_h, stride_h, pad_h)?;
        let out_w = transpose_out("width", in_w, kernel_w, stride_w, pad_w)?;
        Ok(Self {
            batch,
            in_ch,
            out_ch,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
        })
    }

    pub(crate) fn output_dims(&self) -> [usize; 4] {
        [self.batch, self.out_ch, self.out_h, self.out_w]
    }
}

/// Output extent of a transposed convolution: `(input - 1) * stride + kernel - 2 * pad`.
fn transpose_out(
    axis: &str,
    input: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> Result<usize> {
    let full = (input - 1) * stride + kernel;
    if full <= 2 * pad {
        return Err(VoltaError::InvalidParameter(format!(
            "transposed conv output {axis} would be non-positive \
             (input {input}, kernel {kernel}, stride {stride}, padding {pad})"
        )));
    }
    Ok(full - 2 * pad)
}

/// Validated sliding-window geometry over `[B, C, H, W]`, with no weight.
///
/// Used by `MaxPool2d`, and (with zero padding, on the already-padded tensor)
/// by `im2col`/`col2im`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Window2dSpec {
    pub batch: usize,
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl Window2dSpec {
    /// `input` is `[B, C, H, W]`.
    pub(crate) fn new(
        input: &Shape,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        let [batch, channels, in_h, in_w] = dims4(input, "input")?;
        if kernel.0 == 0 || kernel.1 == 0 {
            return Err(VoltaError::InvalidParameter(format!(
                "kernel must be positive, got {kernel:?}"
            )));
        }
        check_stride(stride)?;
        let (kernel_h, kernel_w) = kernel;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let out_h = window_out("height", in_h, kernel_h, stride_h, pad_h)?;
        let out_w = window_out("width", in_w, kernel_w, stride_w, pad_w)?;
        Ok(Self {
            batch,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
        })
    }

    pub(crate) fn input_dims(&self) -> [usize; 4] {
        [self.batch, self.channels, self.in_h, self.in_w]
    }

    pub(crate) fn padded_h(&self) -> usize {
        self.in_h + 2 * self.pad_h
    }

    pub(crate) fn padded_w(&self) -> usize {
        self.in_w + 2 * self.pad_w
    }

    pub(crate) fn output_dims(&self) -> [usize; 4] {
        [self.batch, self.channels, self.out_h, self.out_w]
    }
}

/// Validated geometry of a matrix product `[..batch, m, k] @ [..batch, k, n]`.
///
/// Both operands must have the same rank (at least 2); batch dims may broadcast
/// against each other, so `[2, 1, m, k] @ [1, 3, k, n]` gives batch `[2, 3]`.
/// Vector forms (rank 1) are not covered: `matmul` handles them separately.
/// A rank mismatch is rejected here, which is what makes B4 unreachable: the
/// backward never has to guess which operand's rank indexes which shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatmulSpec {
    pub a_batch: Vec<usize>,
    pub b_batch: Vec<usize>,
    pub out_batch: Vec<usize>,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl MatmulSpec {
    pub(crate) fn new(a: &Shape, b: &Shape) -> Result<Self> {
        let (a_dims, b_dims) = (a.truncate_rank(), b.truncate_rank());
        let (rank_a, rank_b) = (a_dims.len(), b_dims.len());
        if rank_a < 2 || rank_b < 2 {
            return Err(VoltaError::InvalidParameter(format!(
                "matmul needs rank >= 2 operands, got {a_dims:?} @ {b_dims:?}"
            )));
        }
        if rank_a != rank_b {
            return Err(VoltaError::InvalidParameter(format!(
                "matmul operands must have equal rank, got {a_dims:?} @ {b_dims:?} \
                 (rank {rank_a} vs {rank_b})"
            )));
        }
        let (a_batch, a_mat) = a_dims.split_at(rank_a - 2);
        let (b_batch, b_mat) = b_dims.split_at(rank_b - 2);
        let (&[m, k], &[k2, n]) = (a_mat, b_mat) else {
            unreachable!("split_at(rank - 2) leaves exactly two trailing dims");
        };
        if k != k2 {
            return Err(VoltaError::InvalidParameter(format!(
                "matmul dimension mismatch: {a_dims:?} @ {b_dims:?} (inner dims {k} vs {k2})"
            )));
        }
        let out_batch = a_batch
            .iter()
            .zip(b_batch)
            .map(|(&x, &y)| match (x, y) {
                (x, y) if x == y => Ok(x),
                (1, y) => Ok(y),
                (x, 1) => Ok(x),
                _ => Err(VoltaError::InvalidParameter(format!(
                    "matmul batch dims {a_batch:?} and {b_batch:?} do not broadcast"
                ))),
            })
            .collect::<Result<Vec<usize>>>()?;
        Ok(Self {
            a_batch: a_batch.to_vec(),
            b_batch: b_batch.to_vec(),
            out_batch,
            m,
            k,
            n,
        })
    }

    /// Number of independent `m x k @ k x n` products (1 when there are no batch dims).
    pub(crate) fn batch_count(&self) -> usize {
        self.out_batch.iter().product()
    }

    /// `true` when there are batch dims, i.e. the product is not a plain 2D matmul.
    pub(crate) fn is_batched(&self) -> bool {
        !self.out_batch.is_empty()
    }

    /// Dims of `a`: `[..a_batch, m, k]`.
    pub(crate) fn a_dims(&self) -> Vec<usize> {
        with_matrix(&self.a_batch, self.m, self.k)
    }

    /// Dims of `b`: `[..b_batch, k, n]`.
    pub(crate) fn b_dims(&self) -> Vec<usize> {
        with_matrix(&self.b_batch, self.k, self.n)
    }

    /// Dims of the result: `[..out_batch, m, n]`.
    pub(crate) fn out_dims(&self) -> Vec<usize> {
        with_matrix(&self.out_batch, self.m, self.n)
    }

    /// `a` broadcast to the output batch: `[..out_batch, m, k]`.
    pub(crate) fn a_expanded_dims(&self) -> Vec<usize> {
        with_matrix(&self.out_batch, self.m, self.k)
    }

    /// `b` broadcast to the output batch: `[..out_batch, k, n]`.
    pub(crate) fn b_expanded_dims(&self) -> Vec<usize> {
        with_matrix(&self.out_batch, self.k, self.n)
    }
}

/// `[..batch, rows, cols]`.
fn with_matrix(batch: &[usize], rows: usize, cols: usize) -> Vec<usize> {
    let mut dims = batch.to_vec();
    dims.extend([rows, cols]);
    dims
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(dims: &[usize]) -> Shape {
        Shape::from_dims(dims).unwrap()
    }

    fn err_text<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected Err"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn conv_output_dims_table() {
        // (in_h, in_w, k_h, k_w, stride, pad, out_h, out_w)
        let cases = [
            (8, 8, 3, 3, (1, 1), (0, 0), 6, 6),
            (8, 8, 3, 3, (1, 1), (1, 1), 8, 8),
            (8, 8, 3, 3, (2, 2), (1, 1), 4, 4),
            (9, 9, 3, 3, (3, 3), (0, 0), 3, 3),
            (8, 8, 5, 5, (2, 2), (2, 2), 4, 4),
            (7, 11, 3, 5, (1, 2), (2, 0), 9, 4),
            (4, 4, 1, 1, (1, 1), (0, 0), 4, 4),
        ];
        for (ih, iw, kh, kw, stride, pad, oh, ow) in cases {
            let spec = Conv2dSpec::new(
                &shape(&[2, 3, ih, iw]),
                &shape(&[5, 3, kh, kw]),
                stride,
                pad,
            )
            .unwrap();
            assert_eq!(
                (spec.out_h, spec.out_w),
                (oh, ow),
                "in {ih}x{iw} k {kh}x{kw} s {stride:?} p {pad:?}"
            );
        }
    }

    #[test]
    fn conv_spec_fields_come_from_the_right_axes() {
        let spec =
            Conv2dSpec::new(&shape(&[2, 3, 8, 9]), &shape(&[5, 3, 3, 2]), (2, 1), (1, 0)).unwrap();
        assert_eq!(spec.batch, 2);
        assert_eq!(spec.in_ch, 3);
        assert_eq!(spec.out_ch, 5);
        assert_eq!((spec.in_h, spec.in_w), (8, 9));
        assert_eq!((spec.kernel_h, spec.kernel_w), (3, 2));
        assert_eq!((spec.stride_h, spec.stride_w), (2, 1));
        assert_eq!((spec.pad_h, spec.pad_w), (1, 0));
        assert_eq!((spec.padded_h(), spec.padded_w()), (10, 9));
        assert_eq!(spec.input_dims(), [2, 3, 8, 9]);
        assert_eq!(spec.weight_dims(), [5, 3, 3, 2]);
        assert_eq!(spec.output_dims(), [2, 5, spec.out_h, spec.out_w]);
    }

    #[test]
    fn padded_window_is_the_zero_padding_view_of_the_padded_input() {
        let spec =
            Conv2dSpec::new(&shape(&[2, 3, 8, 9]), &shape(&[5, 3, 3, 2]), (2, 1), (1, 2)).unwrap();
        let window = spec.padded_window();
        assert_eq!(window.input_dims(), [2, 3, 10, 13]);
        assert_eq!((window.kernel_h, window.kernel_w), (3, 2));
        assert_eq!((window.stride_h, window.stride_w), (2, 1));
        assert_eq!((window.pad_h, window.pad_w), (0, 0));
        assert_eq!((window.out_h, window.out_w), (spec.out_h, spec.out_w));
    }

    #[test]
    fn conv_rejects_non_4d_input() {
        let msg = err_text(Conv2dSpec::new(
            &shape(&[3, 8, 8]),
            &shape(&[5, 3, 3, 3]),
            (1, 1),
            (0, 0),
        ));
        assert!(msg.contains("input must be 4D"), "{msg}");
    }

    #[test]
    fn conv_rejects_non_4d_weight() {
        let msg = err_text(Conv2dSpec::new(
            &shape(&[1, 3, 8, 8]),
            &shape(&[5, 3, 3]),
            (1, 1),
            (0, 0),
        ));
        assert!(msg.contains("weight must be 4D"), "{msg}");
    }

    #[test]
    fn conv_rejects_channel_mismatch() {
        let msg = err_text(Conv2dSpec::new(
            &shape(&[1, 4, 8, 8]),
            &shape(&[5, 3, 3, 3]),
            (1, 1),
            (0, 0),
        ));
        assert!(msg.contains("channel"), "{msg}");
    }

    #[test]
    fn conv_rejects_zero_stride() {
        let msg = err_text(Conv2dSpec::new(
            &shape(&[1, 3, 8, 8]),
            &shape(&[5, 3, 3, 3]),
            (0, 1),
            (0, 0),
        ));
        assert!(msg.contains("stride"), "{msg}");
    }

    #[test]
    fn conv_rejects_kernel_larger_than_padded_input() {
        let msg = err_text(Conv2dSpec::new(
            &shape(&[1, 3, 4, 4]),
            &shape(&[5, 3, 7, 3]),
            (1, 1),
            (1, 0),
        ));
        assert!(msg.contains("larger than padded input"), "{msg}");
    }

    #[test]
    fn conv_accepts_kernel_equal_to_padded_input() {
        let spec =
            Conv2dSpec::new(&shape(&[1, 3, 3, 3]), &shape(&[5, 3, 5, 5]), (1, 1), (1, 1)).unwrap();
        assert_eq!((spec.out_h, spec.out_w), (1, 1));
    }

    #[test]
    fn transpose_output_dims_table() {
        // (in, k, stride, pad, out): out = (in - 1) * stride + k - 2 * pad
        let cases = [
            (4, 3, 1, 0, 6),
            (4, 3, 2, 1, 7),
            (4, 4, 2, 1, 8),
            (1, 3, 1, 1, 1),
            (5, 2, 3, 0, 14),
        ];
        for (n, k, s, p, out) in cases {
            let spec = ConvTranspose2dSpec::new(
                &shape(&[2, 3, n, n]),
                &shape(&[3, 5, k, k]),
                (s, s),
                (p, p),
            )
            .unwrap();
            assert_eq!(
                (spec.out_h, spec.out_w),
                (out, out),
                "in {n} k {k} s {s} p {p}"
            );
        }
    }

    #[test]
    fn transpose_spec_fields_come_from_the_right_axes() {
        let spec =
            ConvTranspose2dSpec::new(&shape(&[2, 3, 4, 5]), &shape(&[3, 7, 2, 3]), (1, 2), (0, 1))
                .unwrap();
        assert_eq!((spec.batch, spec.in_ch, spec.out_ch), (2, 3, 7));
        assert_eq!((spec.in_h, spec.in_w), (4, 5));
        assert_eq!((spec.kernel_h, spec.kernel_w), (2, 3));
        // height: (4 - 1) * 1 + 2 - 0 = 5; width: (5 - 1) * 2 + 3 - 2 = 9
        assert_eq!(spec.output_dims(), [2, 7, 5, 9]);
    }

    #[test]
    fn transpose_rejects_channel_mismatch() {
        // weight axis 0 is C_in, so a [3, ...] input against a [4, ...] weight is wrong.
        let msg = err_text(ConvTranspose2dSpec::new(
            &shape(&[1, 3, 4, 4]),
            &shape(&[4, 5, 3, 3]),
            (1, 1),
            (0, 0),
        ));
        assert!(msg.contains("channel"), "{msg}");
    }

    #[test]
    fn transpose_rejects_zero_stride() {
        let msg = err_text(ConvTranspose2dSpec::new(
            &shape(&[1, 3, 4, 4]),
            &shape(&[3, 5, 3, 3]),
            (1, 0),
            (0, 0),
        ));
        assert!(msg.contains("stride"), "{msg}");
    }

    #[test]
    fn transpose_rejects_non_positive_output() {
        // (1 - 1) * 1 + 1 - 2 * 1 < 0
        let msg = err_text(ConvTranspose2dSpec::new(
            &shape(&[1, 3, 1, 1]),
            &shape(&[3, 5, 1, 1]),
            (1, 1),
            (1, 1),
        ));
        assert!(msg.contains("output"), "{msg}");
    }

    #[test]
    fn transpose_rejects_non_4d_input() {
        let msg = err_text(ConvTranspose2dSpec::new(
            &shape(&[3, 4, 4]),
            &shape(&[3, 5, 3, 3]),
            (1, 1),
            (0, 0),
        ));
        assert!(msg.contains("input must be 4D"), "{msg}");
    }

    #[test]
    fn pool_output_dims_table() {
        // (in, k, stride, pad, out)
        let cases = [
            (8, 2, 2, 0, 4),
            (8, 3, 2, 1, 4),
            (7, 3, 1, 0, 5),
            (5, 5, 1, 2, 5),
        ];
        for (n, k, s, p, out) in cases {
            let spec = Window2dSpec::new(&shape(&[2, 3, n, n]), (k, k), (s, s), (p, p)).unwrap();
            assert_eq!(
                (spec.out_h, spec.out_w),
                (out, out),
                "in {n} k {k} s {s} p {p}"
            );
        }
    }

    #[test]
    fn pool_spec_fields_come_from_the_right_axes() {
        let spec = Window2dSpec::new(&shape(&[2, 3, 8, 6]), (2, 3), (2, 1), (0, 1)).unwrap();
        assert_eq!((spec.batch, spec.channels), (2, 3));
        assert_eq!((spec.in_h, spec.in_w), (8, 6));
        assert_eq!((spec.kernel_h, spec.kernel_w), (2, 3));
        assert_eq!((spec.stride_h, spec.stride_w), (2, 1));
        assert_eq!((spec.pad_h, spec.pad_w), (0, 1));
        assert_eq!(spec.output_dims(), [2, 3, 4, 6]);
    }

    #[test]
    fn pool_rejects_non_4d_input() {
        let msg = err_text(Window2dSpec::new(&shape(&[8, 8]), (2, 2), (2, 2), (0, 0)));
        assert!(msg.contains("input must be 4D"), "{msg}");
    }

    #[test]
    fn pool_rejects_zero_stride() {
        let msg = err_text(Window2dSpec::new(
            &shape(&[1, 1, 8, 8]),
            (2, 2),
            (0, 0),
            (0, 0),
        ));
        assert!(msg.contains("stride"), "{msg}");
    }

    #[test]
    fn window_rejects_zero_kernel() {
        let msg = err_text(Window2dSpec::new(
            &shape(&[1, 1, 8, 8]),
            (0, 2),
            (1, 1),
            (0, 0),
        ));
        assert!(msg.contains("kernel must be positive"), "{msg}");
    }

    #[test]
    fn window_padded_extents_include_both_sides() {
        let spec = Window2dSpec::new(&shape(&[1, 1, 8, 6]), (2, 2), (2, 2), (1, 2)).unwrap();
        assert_eq!((spec.padded_h(), spec.padded_w()), (10, 10));
    }

    #[test]
    fn pool_rejects_kernel_larger_than_padded_input() {
        let msg = err_text(Window2dSpec::new(
            &shape(&[1, 1, 2, 2]),
            (5, 5),
            (1, 1),
            (1, 1),
        ));
        assert!(msg.contains("larger than padded input"), "{msg}");
    }

    // ===== MatmulSpec =====

    #[test]
    fn matmul_2d_has_no_batch() {
        let spec = MatmulSpec::new(&shape(&[3, 4]), &shape(&[4, 5])).unwrap();
        assert_eq!((spec.m, spec.k, spec.n), (3, 4, 5));
        assert!(!spec.is_batched());
        assert_eq!(spec.batch_count(), 1);
        assert_eq!(spec.out_dims(), vec![3, 5]);
        assert_eq!(spec.a_dims(), vec![3, 4]);
        assert_eq!(spec.b_dims(), vec![4, 5]);
    }

    #[test]
    fn matmul_3d_equal_batch() {
        let spec = MatmulSpec::new(&shape(&[2, 3, 4]), &shape(&[2, 4, 5])).unwrap();
        assert_eq!((spec.m, spec.k, spec.n), (3, 4, 5));
        assert!(spec.is_batched());
        assert_eq!(spec.batch_count(), 2);
        assert_eq!(spec.out_dims(), vec![2, 3, 5]);
    }

    #[test]
    fn matmul_batch_dims_broadcast_within_equal_rank() {
        let spec = MatmulSpec::new(&shape(&[2, 1, 3, 4]), &shape(&[1, 6, 4, 5])).unwrap();
        assert_eq!(spec.a_batch, vec![2, 1]);
        assert_eq!(spec.b_batch, vec![1, 6]);
        assert_eq!(spec.out_batch, vec![2, 6]);
        assert_eq!(spec.batch_count(), 12);
        assert_eq!(spec.a_dims(), vec![2, 1, 3, 4]);
        assert_eq!(spec.b_dims(), vec![1, 6, 4, 5]);
        assert_eq!(spec.a_expanded_dims(), vec![2, 6, 3, 4]);
        assert_eq!(spec.b_expanded_dims(), vec![2, 6, 4, 5]);
        assert_eq!(spec.out_dims(), vec![2, 6, 3, 5]);
    }

    #[test]
    fn matmul_rejects_rank_mismatch_the_b4_shape() {
        // The audit's B4 repro: [2,3,4] @ [2,1,4,6].
        let msg = err_text(MatmulSpec::new(&shape(&[2, 3, 4]), &shape(&[2, 1, 4, 6])));
        assert!(msg.contains("equal rank"), "{msg}");
    }

    #[test]
    fn matmul_rejects_2d_against_3d() {
        let msg = err_text(MatmulSpec::new(&shape(&[3, 4]), &shape(&[2, 4, 5])));
        assert!(msg.contains("equal rank"), "{msg}");
    }

    #[test]
    fn matmul_rejects_rank_below_two() {
        let msg = err_text(MatmulSpec::new(&shape(&[4]), &shape(&[4, 5])));
        assert!(msg.contains("rank >= 2"), "{msg}");
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let msg = err_text(MatmulSpec::new(&shape(&[2, 3, 4]), &shape(&[2, 5, 6])));
        assert!(msg.contains("inner dims"), "{msg}");
    }

    #[test]
    fn matmul_rejects_batch_dims_that_do_not_broadcast() {
        let msg = err_text(MatmulSpec::new(&shape(&[2, 3, 4]), &shape(&[3, 4, 5])));
        assert!(msg.contains("do not broadcast"), "{msg}");
    }
}
