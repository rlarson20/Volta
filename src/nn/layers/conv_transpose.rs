use crate::Storage;
use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};

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

        ConvTranspose2d {
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
    /// * `col` - Column matrix [B*H_in*W_in, C_out*K*K]
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
                                result[out_idx] += col[row_idx * cols + col_idx];
                            }
                        }
                    }
                }
            }
        }

        (result, vec![batch, out_channels, h_out, w_out])
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, in_channels, in_h, in_w) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.shape[0],
                x_borrow.shape[1],
                x_borrow.shape[2],
                x_borrow.shape[3],
            )
        };

        let (weight_in_ch, out_channels, kh, kw) = {
            let w_borrow = self.weight.borrow();
            assert_eq!(
                w_borrow.shape[0], in_channels,
                "Channel mismatch: input has {} channels but weight expects {}",
                in_channels, w_borrow.shape[0]
            );
            (
                w_borrow.shape[0],
                w_borrow.shape[1],
                w_borrow.shape[2],
                w_borrow.shape[3],
            )
        };

        // 1. Reshape input: [B, C_in, H, W] → [B*H*W, C_in]
        let input_reshaped = x.reshape(&[batch * in_h * in_w, in_channels]);

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

        let mut output = RawTensor::new(output_data, &output_shape, x.borrow().requires_grad);

        // 5. Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_reshaped = bias.reshape(&[1, out_channels, 1, 1]);
            output = output.add(&bias_reshaped);
        }

        // Track gradient computation graph
        if x.borrow().requires_grad {
            output.borrow_mut().parents.push(x.clone());
            if let Some(ref bias) = self.bias {
                output.borrow_mut().parents.push(bias.clone());
            }
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
            t.shape = w.shape.clone();
        }
        if let Some(b) = state.get("bias")
            && self.bias.is_some()
        {
            let bias_tensor = self.bias.as_ref().unwrap();
            let mut t = bias_tensor.borrow_mut();
            t.data = Storage::cpu(b.data.clone());
            t.shape = b.shape.clone();
        }
    }
}
