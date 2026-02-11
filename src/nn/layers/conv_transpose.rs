use crate::Storage;
use crate::autograd::GradFn;
use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};
use std::cell::RefCell;
use std::rc::Rc;

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
    kernel_size: (usize, usize),
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
            kernel_size: (kernel, kernel),
        }
    }

    /// Convert column matrix back to image format
    ///
    /// This is the inverse of im2col and is used in the forward pass
    /// of transposed convolution.
    ///
    /// # Arguments
    /// * `col` - Column matrix [B*`H_in`*`W_in`, `C_out`*K*K]
    /// * `batch` - Batch size
    /// * `out_channels` - Number of output channels
    /// * `in_h`, `in_w` - Input height and width
    /// * `kernel` - Kernel size
    /// * `stride` - Stride
    /// * `padding` - Padding
    #[allow(clippy::too_many_arguments)]
    fn col2im_transpose(
        col: &[f32],
        batch: usize,
        out_channels: usize,
        in_h: usize,
        in_w: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> (Vec<f32>, Vec<usize>) {
        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        let (ph, pw) = padding;

        // Calculate output dimensions
        let h_out = (in_h - 1) * sh + kh - 2 * ph;
        let w_out = (in_w - 1) * sw + kw - 2 * pw;

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
                                if let Some(&col_val) = col.get(col_data_idx)
                                    && let Some(slot) = result.get_mut(out_idx)
                                {
                                    *slot += col_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        (result, vec![batch, out_channels, h_out, w_out])
    }
}

/// Gradient function for `ConvTranspose2d`
///
/// The backward pass of transposed convolution is equivalent to the forward pass
/// of regular convolution with respect to the input.
struct ConvTranspose2dGradFn {
    input_shape: Vec<usize>,
    weight: Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    kernel_size: (usize, usize),
    has_bias: bool,
}

impl GradFn for ConvTranspose2dGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // parents: [input, weight, bias] if has_bias, else [input, weight]
        let input = parents
            .first()
            .expect("ConvTranspose2dGradFn requires at least one parent (input)");

        // Get dimensions
        let (batch, in_channels, in_h, in_w) = (
            self.input_shape.first().copied().unwrap_or(1),
            self.input_shape.get(1).copied().unwrap_or(1),
            self.input_shape.get(2).copied().unwrap_or(1),
            self.input_shape.get(3).copied().unwrap_or(1),
        );

        let (weight_in_ch, out_channels, kh, kw) = (
            self.weight.borrow().shape.first().copied().unwrap_or(1),
            self.weight.borrow().shape.get(1).copied().unwrap_or(1),
            self.kernel_size.0,
            self.kernel_size.1,
        );

        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let out_grad_tensor = Rc::new(RefCell::new(out_grad.clone()));

        // Calculate output dimensions
        let out_h = (in_h - 1) * stride_h + kh - 2 * pad_h;
        let out_w = (in_w - 1) * stride_w + kw - 2 * pad_w;

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

                                        if let Some(&og_val) = out_grad_data.get(out_idx)
                                            && let Some(&w_val) = weight_data.get(w_idx)
                                        {
                                            grad_sum += og_val * w_val;
                                        }
                                    }
                                }
                            }
                        }

                        let in_idx =
                            b * (in_channels * in_h * in_w) + c_in * (in_h * in_w) + ih * in_w + iw;

                        if let Some(slot) = grad_input_data.get_mut(in_idx) {
                            *slot = grad_sum;
                        }
                    }
                }
            }
        }

        let grad_input = RawTensor::new(grad_input_data, &[batch, in_channels, in_h, in_w], true);

        let mut grads = vec![Some(grad_input)];

        // 2. Gradient w.r.t. weight
        // grad_weight[c_in, c_out, kh, kw] = sum over b, ih, iw of input[b, c_in, ih, iw] * out_grad[b, c_out, ih*stride+kh, iw*stride+kw]
        let mut grad_weight_data = vec![0.0; weight_in_ch * out_channels * kh * kw];
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

                                        if let Some(&in_val) = input_data.get(in_idx)
                                            && let Some(&og_val) = out_grad_data.get(out_idx)
                                        {
                                            grad_sum += in_val * og_val;
                                        }
                                    }
                                }
                            }
                        }

                        let w_idx = c_in * (out_channels * kh * kw)
                            + c_out * (kh * kw)
                            + kh_idx * kw
                            + kw_idx;

                        if let Some(slot) = grad_weight_data.get_mut(w_idx) {
                            *slot = grad_sum;
                        }
                    }
                }
            }
        }

        #[allow(
            clippy::tuple_array_conversions,
            reason = "just don't know hot to fix right now"
        )]
        let grad_weight = RawTensor::new(
            grad_weight_data,
            &[weight_in_ch, out_channels, kh, kw],
            true,
        );
        grads.push(Some(grad_weight));

        // 3. Gradient w.r.t. bias (if present)
        if self.has_bias && parents.len() > 2 {
            // Sum over all dimensions except the channel dimension (dim=1)
            // out_grad: [B, C_out, H_out, W_out] → [C_out]
            let grad_bias = out_grad_tensor.sum_dim(3, false); // Sum over W
            let grad_bias = grad_bias.sum_dim(2, false); // Sum over H
            let grad_bias = grad_bias.sum_dim(0, false); // Sum over B
            grads.push(Some(grad_bias));
        } else if self.has_bias {
            grads.push(None);
        }

        grads
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(Self {
            input_shape: self.input_shape.clone(),
            weight: self.weight.clone(),
            stride: self.stride,
            padding: self.padding,
            kernel_size: self.kernel_size,
            has_bias: self.has_bias,
        })
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, in_channels, in_h, in_w) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.shape.first().copied().unwrap_or(1),
                x_borrow.shape.get(1).copied().unwrap_or(1),
                x_borrow.shape.get(2).copied().unwrap_or(1),
                x_borrow.shape.get(3).copied().unwrap_or(1),
            )
        };

        let (weight_in_ch, out_channels, kh, kw) = {
            let w_borrow = self.weight.borrow();
            let w_ch = w_borrow.shape.first().copied().unwrap_or(1);
            assert_eq!(
                w_ch, in_channels,
                "Channel mismatch: input has {in_channels} channels but weight expects {w_ch}",
            );
            (
                w_ch,
                w_borrow.shape.get(1).copied().unwrap_or(1),
                w_borrow.shape.get(2).copied().unwrap_or(1),
                w_borrow.shape.get(3).copied().unwrap_or(1),
            )
        };

        // 1. Permute and reshape input: [B, C_in, H, W] → [B, H, W, C_in] → [B*H*W, C_in]
        // Must permute first so that the reshape groups (h,w) positions with their channel values
        let input_permuted = x.permute(&[0, 2, 3, 1]); // [B, H, W, C_in]
        let input_reshaped = input_permuted.reshape(&[batch * in_h * in_w, in_channels]);

        // 2. Reshape weight: [C_in, C_out, K, K] → [C_in, C_out*K*K]
        let weight_reshaped = self.weight.reshape(&[weight_in_ch, out_channels * kh * kw]);

        // 3. Matrix multiplication: [B*H*W, C_in] @ [C_in, C_out*K*K] → [B*H*W, C_out*K*K]
        let matmul_result = input_reshaped.matmul(&weight_reshaped);

        // 4. Use col2im to convert to output shape
        let col_data = matmul_result.borrow().data.clone();
        let (output_data, output_shape) = Self::col2im_transpose(
            &col_data,
            batch,
            out_channels,
            in_h,
            in_w,
            self.kernel_size,
            self.stride,
            self.padding,
        );

        let input_shape = x.borrow().shape.clone();
        let mut output = RawTensor::new(output_data, &output_shape, x.borrow().requires_grad);

        // 5. Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_reshaped = bias.reshape(&[1, out_channels, 1, 1]);
            output = output.add(&bias_reshaped);
        }

        // Track gradient computation graph
        if x.borrow().requires_grad {
            let mut parents = vec![x.clone(), self.weight.clone()];
            if let Some(ref bias) = self.bias {
                parents.push(bias.clone());
            }

            output.borrow_mut().parents.clone_from(&parents);
            output.borrow_mut().grad_fn = Some(Box::new(ConvTranspose2dGradFn {
                input_shape,
                weight: self.weight.clone(),
                stride: self.stride,
                padding: self.padding,
                kernel_size: self.kernel_size,
                has_bias: self.bias.is_some(),
            }));
        }

        output
    }

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

    fn load_state_dict(&mut self, state: &StateDict) {
        if let Some(w) = state.get("weight") {
            let mut t = self.weight.borrow_mut();
            t.data = Storage::cpu(w.data.clone());
            t.shape.clone_from(&w.shape);
        }
        if let Some(b) = state.get("bias")
            && self.bias.is_some()
        {
            let bias_tensor = self.bias.as_ref().unwrap();
            let mut t = bias_tensor.borrow_mut();
            t.data = Storage::cpu(b.data.clone());
            t.shape.clone_from(&b.shape);
        }
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
            assert_eq!(y.borrow().shape.first().copied().unwrap_or(0), batch_size);
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
        loss.backward();

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
        // Use smaller input for more stable numerical gradient checking
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        // Transposed convolution is affine in its inputs, but each loss evaluation sums
        // thousands of f32 multiply-adds (col2im + GEMM). Using larger epsilon to reduce
        // cancellation noise from the complex computation graph.
        let (max_err, mean_err, passed) = RawTensor::check_gradients(
            &x,
            |t| conv.forward(t).sum(),
            5e-2, // epsilon (less cancellation noise)
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
        let conv = ConvTranspose2d::new(2, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 2, 4, 4]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d stride=2 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_no_bias() {
        // Test gradient correctness without bias
        let conv = ConvTranspose2d::new(2, 4, 3, 1, 0, false);
        let x = RawTensor::randn(&[1, 2, 5, 5]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d no_bias gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_1x1() {
        // Test gradient correctness with 1x1 kernel
        let conv = ConvTranspose2d::new(2, 4, 1, 1, 0, true);
        let x = RawTensor::randn(&[1, 2, 8, 8]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d 1x1 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_5x5() {
        // Test gradient correctness with 5x5 kernel
        let conv = ConvTranspose2d::new(2, 4, 5, 1, 2, true);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

        assert!(
            passed,
            "ConvTranspose2d 5x5 gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
        );
    }

    #[test]
    fn test_conv_transpose2d_gradient_multiple_channels() {
        // Test gradient correctness with multiple channels
        let conv = ConvTranspose2d::new(4, 8, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 4, 6, 6]);
        x.borrow_mut().requires_grad = true;

        let (max_err, mean_err, passed) =
            RawTensor::check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2);

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
        loss.backward();

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
        loss.backward();

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
}
