use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::index::{fetch, slot};
use crate::io::{StateDict, TensorData};
use crate::nn::{Layer, Module};
use crate::shape::Axis;
use crate::spec::ConvTranspose2dSpec;
use crate::tensor::{RawTensor, Tensor};

/// Transposed 2D convolution (also called deconvolution)
///
/// Used for upsampling in generative models like GANs and VAEs.
/// The operation is the gradient of Conv2d with respect to its input.
///
/// Output size: `H_out` = (`H_in` - 1) * stride + kernel - 2*padding
pub struct ConvTranspose2d {
    weight: Tensor,       // [in_channels, out_channels, kernel_h, kernel_w]
    bias: Option<Tensor>, // [out_channels]
    stride: (usize, usize),
    padding: (usize, usize),
}

impl ConvTranspose2d {
    #[must_use]
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Self {
        // Weight shape for transpose conv: [in_channels, out_channels, K, K]
        let w = RawTensor::he_initialization(&[in_ch, out_ch, kernel, kernel]);
        w.borrow_mut().requires_grad = true;

        let b = if use_bias {
            let b = RawTensor::zeros(&[out_ch]);
            b.borrow_mut().requires_grad = true;
            Some(b)
        } else {
            None
        };

        Self {
            weight: w,
            bias: b,
            stride: (stride, stride),
            padding: (padding, padding),
        }
    }

    /// Convert column matrix back to image format
    ///
    /// This is the inverse of im2col and is used in the forward pass
    /// of transposed convolution.
    ///
    /// `col` is the column matrix `[B*H_in*W_in, C_out*K*K]`; the geometry
    /// (including the output size) comes from `spec`.
    fn col2im_transpose(col: &[f32], spec: &ConvTranspose2dSpec) -> Vec<f32> {
        let ConvTranspose2dSpec {
            batch,
            out_ch: out_channels,
            in_h,
            in_w,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
            out_h: h_out,
            out_w: w_out,
            ..
        } = *spec;

        let mut result = vec![0.0; batch * out_channels * h_out * w_out];

        // Accumulate values from column matrix into output image
        for b in 0..batch {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let row_idx = b * (in_h * in_w) + ih * in_w + iw;

                    // Starting position in output (before padding)
                    let h_start = ih * sh;
                    let w_start = iw * sw;

                    for c in 0..out_channels {
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                // Position in output
                                let h_pos = h_start + kh_idx;
                                let w_pos = w_start + kw_idx;

                                // Skip if position is in padding region
                                if h_pos < ph || h_pos >= h_out + ph {
                                    continue;
                                }
                                if w_pos < pw || w_pos >= w_out + pw {
                                    continue;
                                }

                                let h_out_pos = h_pos - ph;
                                let w_out_pos = w_pos - pw;

                                let out_idx = b * (out_channels * h_out * w_out)
                                    + c * (h_out * w_out)
                                    + h_out_pos * w_out
                                    + w_out_pos;

                                let col_idx = c * (kh * kw) + kh_idx * kw + kw_idx;
                                let cols = out_channels * kh * kw;
                                let col_data_idx = row_idx * cols + col_idx;
                                let col_val = fetch(col, col_data_idx);
                                *slot(&mut result, out_idx) += col_val;
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

/// Gradient function for `ConvTranspose2d`
///
/// The backward pass of transposed convolution is equivalent to the forward pass
/// of regular convolution with respect to the input.
#[derive(Clone)]
pub(crate) struct ConvTranspose2dGradFn {
    spec: ConvTranspose2dSpec,
    weight: Tensor,
    has_bias: bool,
}

impl ConvTranspose2dGradFn {
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // parents: [input, weight, bias] if has_bias, else [input, weight]
        let input = parents
            .first()
            .expect("ConvTranspose2dGradFn requires at least one parent (input)");

        let ConvTranspose2dSpec {
            batch,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h,
            in_w,
            kernel_h: kh,
            kernel_w: kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
        } = self.spec;

        let out_grad_tensor = Tensor::from_raw(out_grad.clone());

        // 1. Gradient w.r.t. input
        // For transposed convolution forward: output[h_out] = input @ weight where h_out = ih*stride + kh - pad
        // So gradient w.r.t. input at (ih, iw) sums out_grad at positions ih*stride + kh - pad
        // The gradient is: grad_input[ih] = sum over kh of out_grad[ih*stride + kh - pad] * weight[kh]
        let mut grad_input_data = vec![0.0; batch * in_channels * in_h * in_w];

        let out_grad_data = out_grad.data.to_vec();
        let weight_data = self.weight.borrow().data.to_vec();

        for b in 0..batch {
            for c_in in 0..in_channels {
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let mut grad_sum = 0.0;

                        for c_out in 0..out_channels {
                            for kh_idx in 0..kh {
                                for kw_idx in 0..kw {
                                    // Position in out_grad: matches forward pass col2im_transpose mapping
                                    // h_out_pos = ih * stride + kh_idx - padding
                                    let h_out =
                                        ih as i64 * stride_h as i64 + kh_idx as i64 - pad_h as i64;
                                    let w_out =
                                        iw as i64 * stride_w as i64 + kw_idx as i64 - pad_w as i64;

                                    // Check bounds
                                    if h_out >= 0
                                        && h_out < out_h as i64
                                        && w_out >= 0
                                        && w_out < out_w as i64
                                    {
                                        let out_idx = b * (out_channels * out_h * out_w)
                                            + c_out * (out_h * out_w)
                                            + h_out as usize * out_w
                                            + w_out as usize;

                                        let w_idx = c_in * (out_channels * kh * kw)
                                            + c_out * (kh * kw)
                                            + kh_idx * kw
                                            + kw_idx;

                                        grad_sum += fetch(&out_grad_data, out_idx)
                                            * fetch(&weight_data, w_idx);
                                    }
                                }
                            }
                        }

                        let in_idx =
                            b * (in_channels * in_h * in_w) + c_in * (in_h * in_w) + ih * in_w + iw;

                        *slot(&mut grad_input_data, in_idx) = grad_sum;
                    }
                }
            }
        }

        let grad_input = RawTensor::new(grad_input_data, &[batch, in_channels, in_h, in_w], true);

        let mut grads = vec![Some(grad_input)];

        // 2. Gradient w.r.t. weight
        // grad_weight[c_in, c_out, kh, kw] = sum over b, ih, iw of input[b, c_in, ih, iw] * out_grad[b, c_out, ih*stride+kh, iw*stride+kw]
        let mut grad_weight_data = vec![0.0; in_channels * out_channels * kh * kw];
        let input_data = input.borrow().data.to_vec();

        for c_in in 0..in_channels {
            for c_out in 0..out_channels {
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let mut grad_sum = 0.0;

                        for b in 0..batch {
                            for ih in 0..in_h {
                                for iw in 0..in_w {
                                    // Position in out_grad
                                    // Use i64 to avoid underflow with unsigned types
                                    let h_out =
                                        ih as i64 * stride_h as i64 + kh_idx as i64 - pad_h as i64;
                                    let w_out =
                                        iw as i64 * stride_w as i64 + kw_idx as i64 - pad_w as i64;

                                    // Check bounds
                                    if h_out >= 0
                                        && h_out < out_h as i64
                                        && w_out >= 0
                                        && w_out < out_w as i64
                                    {
                                        let in_idx = b * (in_channels * in_h * in_w)
                                            + c_in * (in_h * in_w)
                                            + ih * in_w
                                            + iw;

                                        let out_idx = b * (out_channels * out_h * out_w)
                                            + c_out * (out_h * out_w)
                                            + h_out as usize * out_w
                                            + w_out as usize;

                                        grad_sum += fetch(&input_data, in_idx)
                                            * fetch(&out_grad_data, out_idx);
                                    }
                                }
                            }
                        }

                        let w_idx = c_in * (out_channels * kh * kw)
                            + c_out * (kh * kw)
                            + kh_idx * kw
                            + kw_idx;

                        *slot(&mut grad_weight_data, w_idx) = grad_sum;
                    }
                }
            }
        }

        #[allow(
            clippy::tuple_array_conversions,
            reason = "just don't know hot to fix right now"
        )]
        let grad_weight =
            RawTensor::new(grad_weight_data, &[in_channels, out_channels, kh, kw], true);
        grads.push(Some(grad_weight));

        // 3. Gradient w.r.t. bias (if present)
        if self.has_bias && parents.len() > 2 {
            // Sum over all dimensions except the channel dimension (dim=1)
            // out_grad: [B, C_out, H_out, W_out] → [C_out]
            let grad_bias = out_grad_tensor.sum_dim(Axis(3), false); // Sum over W
            let grad_bias = grad_bias.sum_dim(Axis(2), false); // Sum over H
            let grad_bias = grad_bias.sum_dim(Axis(0), false); // Sum over B
            grads.push(Some(grad_bias));
        } else if self.has_bias {
            grads.push(None);
        }

        grads
    }
}

impl Module for ConvTranspose2d {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert("weight".to_string(), TensorData::from_tensor(&self.weight));
        if let Some(ref b) = self.bias {
            state.insert("bias".to_string(), TensorData::from_tensor(b));
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
        if let Some(w) = state.get("weight") {
            self.weight.load_weight("weight", w)?;
        }
        if let Some(b) = state.get("bias")
            && self.bias.is_some()
        {
            let bias_tensor = self.bias.as_ref().unwrap();
            bias_tensor.load_weight("bias", b)?;
        }
        Ok(())
    }
}

impl Layer for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let spec =
            ConvTranspose2dSpec::new(&x.shape(), &self.weight.shape(), self.stride, self.padding)
                .unwrap_or_else(|e| panic!("{e}"));
        let ConvTranspose2dSpec {
            batch,
            in_ch: in_channels,
            out_ch: out_channels,
            in_h,
            in_w,
            kernel_h: kh,
            kernel_w: kw,
            ..
        } = spec;

        // 1. Permute and reshape input: [B, C_in, H, W] → [B, H, W, C_in] → [B*H*W, C_in]
        // Must permute first so that the reshape groups (h,w) positions with their channel values
        let input_permuted = x.permute(&[Axis(0), Axis(2), Axis(3), Axis(1)]); // [B, H, W, C_in]
        let input_reshaped = input_permuted.reshape(&[batch * in_h * in_w, in_channels]);

        // 2. Reshape weight: [C_in, C_out, K, K] → [C_in, C_out*K*K]
        let weight_reshaped = self.weight.reshape(&[in_channels, out_channels * kh * kw]);

        // 3. Matrix multiplication: [B*H*W, C_in] @ [C_in, C_out*K*K] → [B*H*W, C_out*K*K]
        let matmul_result = input_reshaped.matmul(&weight_reshaped);

        // 4. Use col2im to convert to output shape
        let col_data = matmul_result.borrow().data.clone();
        let output_data = Self::col2im_transpose(&col_data.to_host_f32(), &spec);

        // 5. Add bias if present. The real backward for this op is the custom
        // ConvTranspose2dGradFn below (col2im isn't invertible through ordinary
        // matmul/reshape autograd), so everything up to here — including this bias
        // add — is throwaway data computation (`requires_grad: false`); only the
        // final `.data`/`.shape` are kept, and `edge()` below builds the tensor that
        // actually carries the graph.
        let mut raw_output = RawTensor::new(output_data, &spec.output_dims(), false);
        if let Some(ref bias) = self.bias {
            let bias_reshaped = bias.reshape(&[1, out_channels, 1, 1]);
            raw_output = raw_output.add(&bias_reshaped);
        }
        let (final_data, final_shape) = {
            let b = raw_output.borrow();
            (b.data.clone(), b.shape.clone())
        };

        let has_bias = self.bias.is_some();

        // `edge()` requires grad if *any* of x/weight/bias does — fixes a latent bug
        // where this used to check only `x.borrow().requires_grad`, ignoring the
        // weight and bias tensors (the same B1-class bug §1.4 targets).
        let mut inputs = vec![x.clone(), self.weight.clone()];
        if let Some(ref bias) = self.bias {
            inputs.push(bias.clone());
        }

        edge(
            inputs,
            final_data,
            final_shape,
            Device::CPU,
            GradFn::ConvTranspose2d(ConvTranspose2dGradFn {
                spec,
                weight: self.weight.clone(),
                has_bias,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Forward Pass Shape Verification Tests =====

    #[test]
    fn test_conv_transpose2d_forward_shape_basic() {
        // Input: (1, 3, 32, 32), ConvTranspose: 16 filters, 3x3, stride=1, pad=1
        // Output should be (1, 16, 32, 32)
        let conv = ConvTranspose2d::new(3, 16, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_conv_transpose2d_forward_shape_no_padding() {
        // Input: (2, 3, 8, 8), ConvTranspose: 8 filters, 3x3, stride=1, pad=0
        // Output: (2, 8, 10, 10) since (8 - 1) * 1 + 3 - 2*0 = 10
        let conv = ConvTranspose2d::new(3, 8, 3, 1, 0, false);
        let x = RawTensor::randn(&[2, 3, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![2, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose2d_forward_shape_stride2() {
        // Input: (1, 1, 8, 8), ConvTranspose: 4 filters, 3x3, stride=2, pad=1
        // Output: (1, 4, 15, 15) since (8 - 1) * 2 + 3 - 2*1 = 15
        let conv = ConvTranspose2d::new(1, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 15, 15]);
    }

    #[test]
    fn test_conv_transpose2d_forward_shape_stride2_no_padding() {
        // Input: (1, 1, 4, 4), ConvTranspose: 2 filters, 3x3, stride=2, pad=0
        // Output: (1, 2, 9, 9) since (4 - 1) * 2 + 3 - 2*0 = 9
        let conv = ConvTranspose2d::new(1, 2, 3, 2, 0, false);
        let x = RawTensor::randn(&[1, 1, 4, 4]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2, 9, 9]);
    }

    #[test]
    fn test_conv_transpose2d_1x1_kernel() {
        // 1x1 transposed convolution is equivalent to per-pixel linear transformation
        // Input: (1, 3, 16, 16), ConvTranspose: 8 filters, 1x1, stride=1, pad=0
        // Output: (1, 8, 16, 16)
        let conv = ConvTranspose2d::new(3, 8, 1, 1, 0, true);
        let x = RawTensor::randn(&[1, 3, 16, 16]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_transpose2d_5x5_kernel() {
        // Input: (1, 2, 8, 8), ConvTranspose: 4 filters, 5x5, stride=1, pad=2
        // Output: (1, 4, 8, 8) since (8 - 1) * 1 + 5 - 2*2 = 8
        let conv = ConvTranspose2d::new(2, 4, 5, 1, 2, true);
        let x = RawTensor::randn(&[1, 2, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 8, 8]);
    }

    #[test]
    fn test_conv_transpose2d_7x7_kernel() {
        // Input: (1, 1, 4, 4), ConvTranspose: 2 filters, 7x7, stride=1, pad=3
        // Output: (1, 2, 4, 4) since (4 - 1) * 1 + 7 - 2*3 = 4
        let conv = ConvTranspose2d::new(1, 2, 7, 1, 3, true);
        let x = RawTensor::randn(&[1, 1, 4, 4]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2, 4, 4]);
    }

    #[test]
    fn test_conv_transpose2d_stride3() {
        // Input: (1, 1, 4, 4), ConvTranspose: 2 filters, 3x3, stride=3, pad=1
        // Output: (1, 2, 10, 10) since (4 - 1) * 3 + 3 - 2*1 = 10
        let conv = ConvTranspose2d::new(1, 2, 3, 3, 1, true);
        let x = RawTensor::randn(&[1, 1, 4, 4]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2, 10, 10]);
    }

    #[test]
    fn test_conv_transpose2d_stride4() {
        // Input: (1, 1, 3, 3), ConvTranspose: 2 filters, 3x3, stride=4, pad=1
        // Output: (1, 2, 9, 9) since (3 - 1) * 4 + 3 - 2*1 = 9
        let conv = ConvTranspose2d::new(1, 2, 3, 4, 1, true);
        let x = RawTensor::randn(&[1, 1, 3, 3]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2, 9, 9]);
    }

    #[test]
    fn test_conv_transpose2d_padding2() {
        // Input: (1, 2, 8, 8), ConvTranspose: 4 filters, 3x3, stride=1, pad=2
        // Output: (1, 4, 6, 6) since (8 - 1) * 1 + 3 - 2*2 = 6
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 2, true);
        let x = RawTensor::randn(&[1, 2, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 6, 6]);
    }

    #[test]
    fn test_conv_transpose2d_padding3() {
        // Input: (1, 2, 10, 10), ConvTranspose: 4 filters, 5x5, stride=1, pad=3
        // Output: (1, 4, 8, 8) since (10 - 1) * 1 + 5 - 2*3 = 8
        let conv = ConvTranspose2d::new(2, 4, 5, 1, 3, true);
        let x = RawTensor::randn(&[1, 2, 10, 10]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 8, 8]);
    }

    #[test]
    fn test_conv_transpose2d_no_bias() {
        // Test transposed convolution without bias
        let conv = ConvTranspose2d::new(1, 2, 3, 1, 0, false);
        let x = RawTensor::randn(&[1, 1, 5, 5]);
        let y = conv.forward(&x);

        // Output shape: (1, 2, 7, 7) since (5 - 1) * 1 + 3 - 2*0 = 7
        assert_eq!(y.borrow().shape, vec![1, 2, 7, 7]);
        assert!(conv.bias.is_none(), "Bias should be None");
    }

    #[test]
    fn test_conv_transpose2d_different_channels() {
        // Input: (2, 4, 8, 8), ConvTranspose: 6 filters, 3x3, stride=2, pad=1
        // Output: (2, 6, 15, 15) since (8 - 1) * 2 + 3 - 2*1 = 15
        let conv = ConvTranspose2d::new(4, 6, 3, 2, 1, true);
        let x = RawTensor::randn(&[2, 4, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![2, 6, 15, 15]);
    }

    #[test]
    fn test_conv_transpose2d_multiple_batch_sizes() {
        // Test different batch sizes
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 1, true);

        for batch_size in [1, 2, 4, 8] {
            let x = RawTensor::randn(&[batch_size, 2, 16, 16]);
            let y = conv.forward(&x);

            // Output shape should maintain batch size
            assert_eq!(y.borrow().shape.dim_at(0), batch_size);
            assert_eq!(y.borrow().shape, vec![batch_size, 4, 16, 16]);
        }
    }

    #[test]
    fn test_conv_transpose2d_small_input() {
        // Test with very small input
        let conv = ConvTranspose2d::new(1, 2, 3, 1, 0, true);
        let x = RawTensor::randn(&[1, 1, 2, 2]);
        let y = conv.forward(&x);

        // Output shape: (1, 2, 4, 4) since (2 - 1) * 1 + 3 - 2*0 = 4
        assert_eq!(y.borrow().shape, vec![1, 2, 4, 4]);
    }

    #[test]
    fn test_conv_transpose2d_asymmetric_dimensions() {
        // Test with non-square input
        let conv = ConvTranspose2d::new(2, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 2, 8, 12]);
        let y = conv.forward(&x);

        // Output shape: (1, 4, 15, 23) since:
        // H: (8 - 1) * 2 + 3 - 2*1 = 15
        // W: (12 - 1) * 2 + 3 - 2*1 = 23
        assert_eq!(y.borrow().shape, vec![1, 4, 15, 23]);
    }

    #[test]
    fn test_conv_transpose2d_5x5_stride2_padding2() {
        // Input: (1, 3, 8, 8), ConvTranspose: 6 filters, 5x5, stride=2, pad=2
        // Output: (1, 6, 15, 15) since (8 - 1) * 2 + 5 - 2*2 = 15
        let conv = ConvTranspose2d::new(3, 6, 5, 2, 2, true);
        let x = RawTensor::randn(&[1, 3, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 6, 15, 15]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape_calculation() {
        // Test various output size calculations
        let test_cases = vec![
            // (in_h, in_w, kernel, stride, padding, expected_h, expected_w)
            (4, 4, 3, 1, 0, 6, 6),
            (4, 4, 3, 1, 1, 4, 4),
            (4, 4, 3, 2, 0, 9, 9),
            (4, 4, 3, 2, 1, 7, 7),
            (8, 8, 5, 1, 2, 8, 8),
            (8, 8, 5, 2, 1, 17, 17),
            (8, 8, 5, 2, 2, 15, 15),
            (16, 16, 4, 2, 1, 32, 32),
        ];

        for (in_h, in_w, kernel, stride, padding, expected_h, expected_w) in test_cases {
            let conv = ConvTranspose2d::new(2, 4, kernel, stride, padding, true);
            let x = RawTensor::randn(&[1, 2, in_h, in_w]);
            let y = conv.forward(&x);

            assert_eq!(
                y.borrow().shape,
                vec![1, 4, expected_h, expected_w],
                "Failed for in_h={in_h}, in_w={in_w}, kernel={kernel}, stride={stride}, padding={padding}"
            );
        }
    }

    // ===== Backward Pass Gradient Correctness Tests =====

    #[test]
    fn test_conv_transpose2d_backward_flow() {
        // Test that gradients flow through the entire network
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 0, true);
        let x = RawTensor::randn(&[1, 2, 5, 5]);
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        // Check that input has gradients
        assert!(x.grad().is_some(), "Input should have gradients");

        // Check that weights have gradients
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );

        // Check that bias has gradients
        if let Some(ref b) = conv.bias {
            assert!(b.grad().is_some(), "Bias should have gradients");
        }
    }

    #[test]
    fn test_conv_transpose2d_gradient() {
        // Seeded, with `eps = 0.5`, for the reason given in
        // `test_conv_transpose2d_gradient_multiple_channels`: the layer is linear in `x`,
        // so the step only affects f32 roundoff.
        crate::tensor::manual_seed(0xC7D0);
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) = RawTensor::check_gradients(
            &x,
            |t| conv.forward(t).sum(),
            0.5,  // epsilon
            2e-2, // tolerance (relaxed for numerical complexity)
        );

        assert!(
            passed,
            "ConvTranspose2d gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_stride2() {
        // Test gradient correctness with stride 2
        // Seeded, with `eps = 0.5`, for the reason given in
        // `test_conv_transpose2d_gradient_multiple_channels`: the layer is linear in `x`,
        // so the step only affects f32 roundoff.
        crate::tensor::manual_seed(0xC7D2);
        let conv = ConvTranspose2d::new(2, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 2, 4, 4]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 0.5, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d stride=2 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_no_bias() {
        // Test gradient correctness without bias
        // Seeded, with `eps = 0.5`, for the reason given in
        // `test_conv_transpose2d_gradient_multiple_channels`: the layer is linear in `x`,
        // so the step only affects f32 roundoff.
        crate::tensor::manual_seed(0xC7DB);
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 0, false);
        let x = RawTensor::randn(&[1, 2, 5, 5]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 0.5, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d no_bias gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_1x1() {
        // Test gradient correctness with 1x1 kernel
        // Seeded, with `eps = 0.5`, for the reason given in
        // `test_conv_transpose2d_gradient_multiple_channels`: the layer is linear in `x`,
        // so the step only affects f32 roundoff.
        crate::tensor::manual_seed(0xC7D1);
        let conv = ConvTranspose2d::new(2, 4, 1, 1, 0, true);
        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 0.5, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d 1x1 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_5x5() {
        // Test gradient correctness with 5x5 kernel.
        //
        // Seeded, with `eps = 0.5`, for the reason given in
        // `test_conv_transpose2d_gradient_multiple_channels`: the layer is linear in `x`, so
        // the step only affects f32 roundoff.
        crate::tensor::manual_seed(0x5B5);
        let conv = ConvTranspose2d::new(2, 4, 5, 1, 2, true);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 0.5, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d 5x5 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_multiple_channels() {
        // Test gradient correctness with multiple channels.
        //
        // Seeded because the unseeded version failed ~3% of runs. The cause was the step,
        // not the layer: the layer is linear in `x`, so the central difference has no
        // truncation error and its only error is f32 roundoff in the loss, which shrinks
        // as `eps` grows. At `eps = 5e-2`, a random input whose smallest gradient entry
        // sat just above `check_gradients`' 1e-3 absolute/relative switch put that
        // roundoff over `tol` (13 of 400 seeds); at 0.5, none of the same 400 do.
        crate::tensor::manual_seed(0xC7);
        let conv = ConvTranspose2d::new(4, 8, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 4, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 0.5, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d multiple channels gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_parameter_gradients_shape() {
        // Test that parameter gradients have the correct shape
        let conv = ConvTranspose2d::new(3, 6, 3, 2, 1, true);
        let x = RawTensor::randn(&[2, 3, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        // Check weight gradient shape
        if let Some(w_grad) = conv.weight.grad() {
            assert_eq!(
                w_grad.len(),
                conv.weight.borrow().data.len(),
                "Weight gradient should have same number of elements as weight"
            );
            assert_eq!(
                conv.weight.borrow().shape,
                vec![3, 6, 3, 3],
                "Weight shape should be [in_ch, out_ch, kernel_h, kernel_w]"
            );
        } else {
            panic!("Weight should have gradients");
        }

        // Check bias gradient shape
        if let Some(ref b) = conv.bias {
            if let Some(b_grad) = b.grad() {
                assert_eq!(
                    b_grad.len(),
                    b.borrow().data.len(),
                    "Bias gradient should have same number of elements as bias"
                );
                assert_eq!(b.borrow().shape, vec![6], "Bias shape should be [out_ch]");
            } else {
                panic!("Bias should have gradients");
            }
        }

        // Check input gradient shape
        if let Some(x_grad) = x.grad() {
            assert_eq!(
                x_grad.len(),
                x.borrow().data.len(),
                "Input gradient should have same number of elements as input"
            );
            assert_eq!(x.borrow().shape, vec![2, 3, 8, 8]);
        } else {
            panic!("Input should have gradients");
        }
    }

    #[test]
    fn test_conv_transpose2d_gradient_asymmetric_dimensions() {
        // Test gradient correctness with non-square input
        let conv = ConvTranspose2d::new(2, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 2, 6, 10]);
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        // Check gradients exist and have correct shapes
        assert!(x.grad().is_some(), "Input should have gradients");
        assert_eq!(x.borrow().shape, vec![1, 2, 6, 10]);
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );
    }

    #[test]
    fn test_conv_transpose2d_parameters_count() {
        // Test that parameters() returns correct number of tensors
        let conv_with_bias = ConvTranspose2d::new(3, 6, 3, 1, 1, true);
        assert_eq!(
            conv_with_bias.parameters().len(),
            2,
            "ConvTranspose2d with bias should have 2 parameters"
        );

        let conv_no_bias = ConvTranspose2d::new(3, 6, 3, 1, 1, false);
        assert_eq!(
            conv_no_bias.parameters().len(),
            1,
            "ConvTranspose2d without bias should have 1 parameter"
        );
    }

    #[test]
    fn test_conv_transpose2d_assert_trainable() {
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 0, true);
        let x = crate::macros::trainable_input(&[1, 2, 5, 5]);
        crate::assert_trainable!(conv, x);
    }

    /// Runs `conv.forward(x)` and returns the panic message, failing if it does not panic.
    fn forward_panic_message(conv: &ConvTranspose2d, x: &Tensor) -> String {
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| conv.forward(x)))
            .expect_err("forward should have panicked");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("panic payload should be a string")
    }

    #[test]
    fn test_conv_transpose2d_zero_stride_panics_with_spec_message() {
        let conv = ConvTranspose2d::new(2, 3, 3, 0, 0, false);
        let x = RawTensor::randn(&[1, 2, 4, 4]);
        let msg = forward_panic_message(&conv, &x);
        assert!(msg.contains("stride must be non-zero"), "{msg}");
    }

    #[test]
    fn test_conv_transpose2d_padding_that_empties_output_panics_with_spec_message() {
        // (1 - 1) * 1 + 1 - 2 * 1 would be negative.
        let conv = ConvTranspose2d::new(2, 3, 1, 1, 1, false);
        let x = RawTensor::randn(&[1, 2, 1, 1]);
        let msg = forward_panic_message(&conv, &x);
        assert!(msg.contains("non-positive"), "{msg}");
    }

    #[test]
    fn test_conv_transpose2d_channel_mismatch_panics() {
        let conv = ConvTranspose2d::new(3, 4, 3, 1, 0, false);
        let x = RawTensor::randn(&[1, 2, 5, 5]);
        let msg = forward_panic_message(&conv, &x);
        assert!(msg.contains("channel mismatch"), "{msg}");
    }

    #[test]
    fn test_conv_transpose2d_non_4d_input_panics() {
        let conv = ConvTranspose2d::new(3, 4, 3, 1, 0, false);
        let x = RawTensor::randn(&[3, 5, 5]);
        let msg = forward_panic_message(&conv, &x);
        assert!(msg.contains("input must be 4D"), "{msg}");
    }

    #[test]
    fn test_conv_transpose2d_output_shape_follows_formula() {
        // (in - 1) * stride + kernel - 2 * padding, per axis.
        let conv = ConvTranspose2d::new(2, 3, 3, 2, 1, true);
        let x = RawTensor::randn(&[2, 2, 4, 5]);
        assert_eq!(conv.forward(&x).shape(), vec![2, 3, 7, 9]);
    }
}
