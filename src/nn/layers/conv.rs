// src/layers/conv.rs
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor};

#[allow(dead_code)]
pub struct Conv2d {
    weight: Tensor,       // [out_channels, in_channels, kernel_h, kernel_w]
    bias: Option<Tensor>, // [out_channels]
    stride: (usize, usize),
    padding: (usize, usize),
}

#[allow(dead_code)]
impl Conv2d {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Self {
        let w = RawTensor::randn(&[out_ch, in_ch, kernel, kernel]);
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

    #[allow(unused_variables)]
    // Im2col: Convert (B, C, H, W) → (B*H_out*W_out, C*K*K) matrix
    fn im2col(
        x: &Tensor,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        // Implementation: ~100 lines of indexing logic
        // Each row = flattened receptive field
        // ...
        // TODO: Implement im2col transformation
        // For now, return a placeholder
        let s = x.borrow();
        RawTensor::zeros(&[1, 1])
    }

    #[allow(dead_code)]
    #[allow(unused_variables)]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Pad input: (B, C, H, W) → (B, C, H+2p, W+2p)
        // 2. im2col: → (B*H_out*W_out, C*K*K)
        // 3. Reshape weight: (O, C, K, K) → (O, C*K*K) → transpose
        // 4. Matmul: (B*H_out*W_out, C*K*K) @ (C*K*K, O) → (B*H_out*W_out, O)
        // 5. Reshape: → (B, O, H_out, W_out)
        // 6. Add bias if present
        // ...
        RawTensor::zeros(&[1, 1])
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
}

#[cfg(test)]
mod conv2d_tests {
    use super::*;
    use crate::tensor::TensorOps;
    #[test]
    #[ignore] // TODO: Enable when Conv2d is fully implemented
    fn test_conv2d_forward_shape() {
        // Input: (1, 3, 32, 32), Conv: 16 filters, 3x3, stride=1, pad=1
        // Output should be (1, 16, 32, 32)
    }

    #[test]
    #[ignore] // TODO: Enable when Conv2d is fully implemented
    fn test_conv2d_gradient() {
        let conv = Conv2d::new(3, 8, 3, 1, 1, true);
        let x = RawTensor::randn(&[2, 3, 8, 8]);
        x.borrow_mut().requires_grad = true;
        let passed = RawTensor::check_gradients_simple(&x, |t| conv.forward(t).sum());
        assert!(passed);
    }
}
