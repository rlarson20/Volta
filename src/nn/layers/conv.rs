// src/layers/conv.rs
use crate::Storage;
use crate::autograd::GradFn;
use crate::device::Device;
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
    #[must_use]
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

    /// Create a new Conv2d layer on a specific device
    ///
    /// Uses He initialization and places tensors on the specified device.
    ///
    /// # Arguments
    /// * `in_ch` - Number of input channels
    /// * `out_ch` - Number of output channels
    /// * `kernel` - Kernel size (square)
    /// * `stride` - Stride (square)
    /// * `padding` - Padding (square)
    /// * `use_bias` - Whether to include a bias term
    /// * `device` - Device to place parameters on (CPU or GPU)
    ///
    /// # Example
    /// ```no_run
    /// # use volta::{Conv2d, Device};
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// let device = Device::gpu().expect("GPU required");
    /// let layer = Conv2d::new_on_device(3, 64, 3, 1, 1, true, device);
    /// // Parameters are now on GPU
    /// # }
    /// ```
    #[must_use]
    pub fn new_on_device(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        device: Device,
    ) -> Self {
        let w = RawTensor::he_initialization(&[out_ch, in_ch, kernel, kernel]);
        w.borrow_mut().requires_grad = true;
        let w = w.to_device(device.clone());

        let b = if use_bias {
            let b = RawTensor::zeros(&[out_ch]);
            b.borrow_mut().requires_grad = true;
            Some(b.to_device(device))
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
        const MAX_ALLOC: usize = 100_000_000; // Maximum allowed allocation

        let (data, shape, requires_grad, device) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.data.clone(),
                x_borrow.shape.clone(),
                x_borrow.requires_grad,
                x_borrow.device.clone(),
            )
        };

        let (batch, channels, height, width) = (
            shape.first().copied().unwrap_or(1),
            shape.get(1).copied().unwrap_or(1),
            shape.get(2).copied().unwrap_or(1),
            shape.get(3).copied().unwrap_or(1),
        );
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

        // Use GPU path if available and input is on GPU
        #[cfg(feature = "gpu")]
        {
            if matches!(device, Device::GPU(_))
                && let Some(col_data) = crate::RawTensor::gpu_im2col(
                    &data, batch, channels, height, width, kh, kw, sh, sw, h_out, w_out,
                )
            {
                let out = RawTensor::new_with_storage(
                    col_data,
                    &[rows, cols],
                    device.clone(),
                    requires_grad,
                );
                if requires_grad {
                    out.borrow_mut().parents = vec![x.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(Im2colGradFn {
                        input_shape: shape,
                        kernel,
                        stride,
                    }));
                }
                return out;
            }
        }

        // CPU fallback
        let total_elements = rows * cols;
        assert!(
            total_elements <= MAX_ALLOC,
            "im2col would create tensor with {total_elements} elements (max: {MAX_ALLOC}). Input shape: {shape:?}, kernel: {kernel:?}, stride: {stride:?}"
        );

        let mut result = vec![0.0; total_elements];

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
                                let result_idx = row_idx * cols + col_idx;
                                let cpu_data = data.to_vec();
                                if let Some(&src_val) = cpu_data.get(in_idx)
                                    && let Some(slot) = result.get_mut(result_idx)
                                {
                                    *slot = src_val;
                                }
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

    /// Col2im: Inverse of im2col, used for computing input gradients
    fn col2im(
        col: &[f32],
        output_shape: &[usize], // (B, C, H, W)
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Vec<f32> {
        let (batch, channels, height, width) = (
            output_shape.first().copied().unwrap_or(1),
            output_shape.get(1).copied().unwrap_or(1),
            output_shape.get(2).copied().unwrap_or(1),
            output_shape.get(3).copied().unwrap_or(1),
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

        result
    }

    /// # Panics
    /// Input needs to be 4D
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, in_channels, _height, _width) = {
            let x_borrow = x.borrow();
            assert_eq!(x_borrow.shape.len(), 4, "Input must be 4D: (B, C, H, W)");
            (
                x_borrow.shape.first().copied().unwrap_or(1),
                x_borrow.shape.get(1).copied().unwrap_or(1),
                x_borrow.shape.get(2).copied().unwrap_or(1),
                x_borrow.shape.get(3).copied().unwrap_or(1),
            )
        };

        let (out_channels, kernel_h, kernel_w) = {
            let w_borrow = self.weight.borrow();
            assert_eq!(
                w_borrow.shape.get(1).copied().unwrap_or(1),
                in_channels,
                "Channel mismatch"
            );
            (
                w_borrow.shape.first().copied().unwrap_or(1),
                w_borrow.shape.get(2).copied().unwrap_or(1),
                w_borrow.shape.get(3).copied().unwrap_or(1),
            )
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
            (
                p.shape.get(2).copied().unwrap_or(1),
                p.shape.get(3).copied().unwrap_or(1),
            )
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
            // Reshape bias from (out_channels,) to (1, out_channels, 1, 1)
            let bias_reshaped = b.reshape(&[1, out_channels, 1, 1]);

            // Workaround: GPU broadcasting not yet implemented for binary ops
            // Manually expand bias to match output shape
            let bias_expanded = bias_reshaped.expand(&[batch, out_channels, h_out, w_out]);

            out.add(&bias_expanded)
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

#[cfg(all(test, feature = "gpu"))]
mod conv2d_gpu_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_conv2d_gpu_forward_shape() {
        if Device::gpu().is_none() {
            return; // Skip if no GPU available
        }

        let device = Device::gpu().unwrap();

        // Create Conv2d layer on GPU
        let conv = Conv2d::new_on_device(3, 16, 3, 1, 1, true, device.clone());
        let x = RawTensor::randn(&[1, 3, 32, 32]).to_device(device.clone());
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 16, 32, 32]);
        assert!(y.borrow().device.is_gpu(), "Output should be on GPU");
    }

    #[test]
    fn test_conv2d_gpu_forward_match_cpu() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Create CPU conv with deterministic weights (all ones)
        let conv_cpu = Conv2d::new_on_device(2, 4, 3, 1, 1, true, Device::CPU);
        // Set weights to all ones
        let weight_size = 2 * 4 * 3 * 3; // in_channels * out_channels * kernel_h * kernel_w
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; weight_size]);
        // Set bias to zeros
        conv_cpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::cpu(vec![0.0; 4]);

        // Create deterministic input (all ones)
        let x_cpu = RawTensor::new(vec![1.0; 128], &[1, 2, 8, 8], false);
        let y_cpu = conv_cpu.forward(&x_cpu);

        // Move entire conv layer to GPU
        let conv_gpu = Conv2d::new_on_device(2, 4, 3, 1, 1, true, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; weight_size]);
        conv_gpu.bias.as_ref().unwrap().borrow_mut().data =
            crate::storage::Storage::gpu(vec![0.0; 4]);

        let x_gpu = x_cpu.to_device(device.clone());
        let y_gpu = conv_gpu.forward(&x_gpu);

        // Compare shapes
        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        // Compare values (allow for floating point differences)
        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        assert_eq!(y_cpu_data.len(), y_gpu_data.len());

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Mismatch at index {}: CPU={}, GPU={}, abs_diff={}",
                i,
                cpu_val,
                gpu_val,
                abs_diff
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_backward_flow() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        let conv = Conv2d::new_on_device(2, 4, 3, 1, 0, true, device.clone());
        let x = RawTensor::randn(&[1, 2, 5, 5]).to_device(device.clone());
        x.borrow_mut().requires_grad = true;

        let y = conv.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Check that gradients exist
        assert!(x.grad().is_some(), "Input should have gradients");
        assert!(
            conv.weight.grad().is_some(),
            "Weights should have gradients"
        );
        if let Some(ref b) = conv.bias {
            assert!(b.grad().is_some(), "Bias should have gradients");
        }
    }

    #[test]
    fn test_conv2d_gpu_stride2() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        let conv = Conv2d::new_on_device(1, 4, 3, 2, 1, true, device.clone());
        let x = RawTensor::randn(&[1, 1, 8, 8]).to_device(device.clone());
        let y = conv.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 4, 4, 4]);
        assert!(y.borrow().device.is_gpu(), "Output should be on GPU");
    }

    #[test]
    fn test_conv2d_gpu_simple_values() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Create simple input with known values
        // Input: 1x1x4x4 with values 0-15
        let x_cpu = RawTensor::new(
            (0..16).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 4, 4],
            false,
        );

        // Create Conv2d with simple weights (1 output channel, 1 input channel, 2x2 kernel, stride 1, padding 0)
        // Weights: all ones
        let conv_cpu = Conv2d::new_on_device(1, 1, 2, 1, 0, false, Device::CPU);
        conv_cpu.weight.borrow_mut().data = crate::storage::Storage::cpu(vec![1.0; 4]);

        let y_cpu = conv_cpu.forward(&x_cpu);

        // Now try on GPU
        let x_gpu = x_cpu.to_device(device.clone());
        let conv_gpu = Conv2d::new_on_device(1, 1, 2, 1, 0, false, device.clone());
        conv_gpu.weight.borrow_mut().data = crate::storage::Storage::gpu(vec![1.0; 4]);

        let y_gpu = conv_gpu.forward(&x_gpu);

        // Compare shapes
        assert_eq!(y_cpu.borrow().shape, y_gpu.borrow().shape);

        // Compare values
        let y_cpu_data = y_cpu.borrow().data.to_vec();
        let y_gpu_data = y_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in y_cpu_data.iter().zip(y_gpu_data.iter()).enumerate() {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Mismatch at index {}: CPU={}, GPU={}, abs_diff={}",
                i,
                cpu_val,
                gpu_val,
                abs_diff
            );
        }
    }

    #[test]
    fn test_conv2d_gpu_padding() {
        if Device::gpu().is_none() {
            return;
        }

        let device = Device::gpu().unwrap();

        // Test with padding to ensure pad operation works on GPU
        let x_cpu = RawTensor::new(vec![1.0; 16], &[1, 1, 4, 4], false);
        let x_padded_cpu = x_cpu.pad(&[(0, 0), (0, 0), (1, 1), (1, 1)]);

        let x_gpu = x_cpu.to_device(device.clone());
        let x_padded_gpu = x_gpu.pad(&[(0, 0), (0, 0), (1, 1), (1, 1)]);

        // Check shapes
        assert_eq!(x_padded_cpu.borrow().shape, vec![1, 1, 6, 6]);
        assert_eq!(x_padded_gpu.borrow().shape, vec![1, 1, 6, 6]);

        // Check values - center should be 1.0, edges should be 0.0
        let padded_cpu_data = x_padded_cpu.borrow().data.to_vec();
        let padded_gpu_data = x_padded_gpu.borrow().data.to_vec();

        for (i, (cpu_val, gpu_val)) in padded_cpu_data
            .iter()
            .zip(padded_gpu_data.iter())
            .enumerate()
        {
            let abs_diff = (cpu_val - gpu_val).abs();
            assert!(
                abs_diff < 1e-4,
                "Padding mismatch at index {}: CPU={}, GPU={}",
                i,
                cpu_val,
                gpu_val
            );
        }
    }
}
