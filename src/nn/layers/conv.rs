// src/layers/conv.rs
use crate::Storage;
use crate::autograd::GradFn;
use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};

pub struct Conv2d {
    weight: Tensor,       // [out_channels, in_channels, kernel_h, kernel_w]
    bias: Option<Tensor>, // [out_channels]
    stride: (usize, usize),
    padding: (usize, usize),
}

/// Gradient function for im2col operation
#[derive(Clone)]
struct Im2colGradFn {
    input_shape: Vec<usize>,
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl GradFn for Im2colGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // out_grad has shape (B*H_out*W_out, C*K*K)
        // We need to convert it back to (B, C, H, W) using col2im
        let grad_data = Conv2d::col2im(&out_grad.data, &self.input_shape, self.kernel, self.stride);

        vec![Some(RawTensor::new(grad_data, &self.input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(self.clone())
    }
}

impl Conv2d {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Self {
        let w = RawTensor::he_initialization(&[out_ch, in_ch, kernel, kernel]);
        w.borrow_mut().requires_grad = true;
        let b = if use_bias {
            let b = RawTensor::zeros(&[out_ch]);
            b.borrow_mut().requires_grad = true;
            Some(b)
        } else {
            None
        };
        Conv2d {
            weight: w,
            bias: b,
            stride: (stride, stride),
            padding: (padding, padding),
        }
    }

    /// Im2col: Convert (B, C, H, W) → (B*`H_out`*`W_out`, C*K*K) matrix
    /// Each row contains a flattened receptive field
    fn im2col(x: &Tensor, kernel: (usize, usize), stride: (usize, usize)) -> Tensor {
        let (data, shape, requires_grad) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.data.clone(),
                x_borrow.shape.clone(),
                x_borrow.requires_grad,
            )
        };

        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        // Check for reasonable parameters to prevent memory issues
        assert!(height >= kh && width >= kw, "Input smaller than kernel");
        assert!(
            kh > 0 && kw > 0 && sh > 0 && sw > 0,
            "Invalid kernel/stride parameters"
        );

        // Calculate output dimensions
        let h_out = (height - kh) / sh + 1;
        let w_out = (width - kw) / sw + 1;

        // Output shape: (B*H_out*W_out, C*K*K)
        assert!(h_out > 0 && w_out > 0, "Invalid output dimensions");
        let rows = batch * h_out * w_out;
        let cols = channels * kh * kw;
        let mut result = vec![0.0; rows * cols];

        // For each output position, extract and flatten the receptive field
        for b in 0..batch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row_idx = b * (h_out * w_out) + oh * w_out + ow;

                    // Starting position in input
                    let h_start = oh * sh;
                    let w_start = ow * sw;

                    // Extract receptive field
                    for c in 0..channels {
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let h_pos = h_start + kh_idx;
                                let w_pos = w_start + kw_idx;

                                let in_idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + h_pos * width
                                    + w_pos;

                                let col_idx = c * (kh * kw) + kh_idx * kw + kw_idx;
                                result[row_idx * cols + col_idx] = data[in_idx];
                            }
                        }
                    }
                }
            }
        }

        let out = RawTensor::new(result, &[rows, cols], requires_grad);

        // Attach gradient function if input requires gradients
        if requires_grad {
            out.borrow_mut().parents = vec![x.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(Im2colGradFn {
                input_shape: shape,
                kernel,
                stride,
            }));
        }

        out
    }

    fn col2im(
        col: &[f32],
        output_shape: &[usize], // (B, C, H, W)
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Vec<f32> {
        Self::col2im_with_params(col, output_shape, kernel, stride)
    }

    /// Col2im: Inverse of im2col, used for computing input gradients
    fn col2im_with_params(
        col: &[f32],
        output_shape: &[usize], // (B, C, H, W)
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Vec<f32> {
        let (batch, channels, height, width) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        let h_out = (height - kh) / sh + 1;
        let w_out = (width - kw) / sw + 1;

        let mut result = vec![0.0; batch * channels * height * width];

        // Accumulate gradients from each receptive field
        for b in 0..batch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row_idx = b * (h_out * w_out) + oh * w_out + ow;

                    let h_start = oh * sh;
                    let w_start = ow * sw;

                    for c in 0..channels {
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let h_pos = h_start + kh_idx;
                                let w_pos = w_start + kw_idx;

                                let out_idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + h_pos * width
                                    + w_pos;

                                let col_idx = c * (kh * kw) + kh_idx * kw + kw_idx;
                                let cols = channels * kh * kw;
                                result[out_idx] += col[row_idx * cols + col_idx];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, in_channels, _height, _width) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.shape[0],
                x_borrow.shape[1],
                x_borrow.shape[2],
                x_borrow.shape[3],
            )
        };

        let (out_channels, kernel_h, kernel_w) = {
            let w_borrow = self.weight.borrow();
            assert_eq!(w_borrow.shape[1], in_channels, "Channel mismatch");
            (w_borrow.shape[0], w_borrow.shape[2], w_borrow.shape[3])
        };

        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;

        // 1. Pad input
        let x_padded = if pad_h > 0 || pad_w > 0 {
            x.pad(&[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])
        } else {
            x.clone()
        };

        let (padded_h, padded_w) = {
            let p = x_padded.borrow();
            (p.shape[2], p.shape[3])
        };

        // Calculate output dimensions
        let h_out = (padded_h - kernel_h) / stride_h + 1;
        let w_out = (padded_w - kernel_w) / stride_w + 1;

        // 2. Apply im2col: (B, C, H_pad, W_pad) → (B*H_out*W_out, C*K*K)
        let col = Self::im2col(&x_padded, (kernel_h, kernel_w), (stride_h, stride_w));

        // 3. Reshape weights: (O, C, K, K) → (C*K*K, O)
        let weight_2d = self
            .weight
            .reshape(&[out_channels, in_channels * kernel_h * kernel_w]);
        let weight_t = weight_2d.permute(&[1, 0]); // Transpose to (C*K*K, O)

        // 4. Matmul: (B*H_out*W_out, C*K*K) @ (C*K*K, O) → (B*H_out*W_out, O)
        let out_2d = col.matmul(&weight_t);

        // 5. Reshape: (B*H_out*W_out, O) → (B, H_out, W_out, O) → (B, O, H_out, W_out)
        let out_4d = out_2d.reshape(&[batch, h_out, w_out, out_channels]);
        let out = out_4d.permute(&[0, 3, 1, 2]); // (B, O, H_out, W_out)

        // 6. Add bias if present
        if let Some(ref b) = self.bias {
            let bias_reshaped = b.reshape(&[1, out_channels, 1, 1]);
            out.add(&bias_reshaped)
        } else {
            out
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
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

#[cfg(test)]
mod conv2d_tests {
    use super::*;

    #[test]
    fn test_conv2d_forward_shape() {
        // Input: (1, 3, 32, 32), Conv: 16 filters, 3x3, stride=1, pad=1
        // Output should be (1, 16, 32, 32)
        let conv = Conv2d::new(3, 16, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_forward_shape_no_padding() {
        // Input: (2, 3, 8, 8), Conv: 8 filters, 3x3, stride=1, pad=0
        // Output: (2, 8, 6, 6) since (8 - 3) / 1 + 1 = 6
        let conv = Conv2d::new(3, 8, 3, 1, 0, false);
        let x = RawTensor::randn(&[2, 3, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![2, 8, 6, 6]);
    }

    #[test]
    fn test_conv2d_forward_shape_stride2() {
        // Input: (1, 1, 8, 8), Conv: 4 filters, 3x3, stride=2, pad=1
        // Output: (1, 4, 4, 4) since (8 + 2 - 3) / 2 + 1 = 4
        let conv = Conv2d::new(1, 4, 3, 2, 1, true);
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 4, 4]);
    }

    #[test]
    //CARE: can sometimes fail as well, need to fix
    fn test_conv2d_gradient() {
        // Use smaller input for more stable numerical gradient checking
        let conv = Conv2d::new(2, 4, 3, 1, 1, true);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        x.borrow_mut().requires_grad = true;

        // Convolution is affine in its inputs, but each loss evaluation sums tens of thousands
        // of f32 multiply-adds (im2col + GEMM). With ε=1e-2 the central-difference estimator
        // was dominated by round-off (~1e-5) and produced ~3% relative error. Increasing ε
        // lowers that amplification without changing the true derivative.
        let (max_err, mean_err, passed) = RawTensor::check_gradients(
            &x,
            |t| conv.forward(t).sum(),
            5e-2, // epsilon, less cancellation noise for heavy Conv2d graphs
            2e-2, // tolerance (relaxed for conv's numerical complexity)
        );

        assert!(
            passed,
            "Conv2d gradient check failed: max_error={:.6e}, mean_error={:.6e}",
            max_err, mean_err
        );
    }

    #[test]
    fn test_conv2d_backward_flow() {
        // Test that gradients flow through the entire network
        let conv = Conv2d::new(2, 4, 3, 1, 0, true);
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
    fn test_conv2d_no_bias() {
        // Test convolution without bias
        let conv = Conv2d::new(1, 2, 3, 1, 0, false);
        let x = RawTensor::randn(&[1, 1, 5, 5]);
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2, 3, 3]);
        assert!(conv.bias.is_none(), "Bias should be None");
    }
}
