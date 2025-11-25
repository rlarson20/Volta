use crate::autograd::GradFn;
use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};

/// 2D max pooling layer
///
/// Accepts tensors shaped (batch, channels, height, width) and downsamples each
/// spatial window to its maximum value, similar to `PyTorch`'s `nn.MaxPool2d`.
pub struct MaxPool2d {
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

#[derive(Clone)]
struct MaxPool2dGradFn {
    max_indices: Vec<usize>,
}

impl MaxPool2d {
    /// Square-kernel constructor for convenience
    pub fn new(kernel: usize, stride: usize, padding: usize) -> Self {
        MaxPool2d {
            kernel: (kernel, kernel),
            stride: (stride, stride),
            padding: (padding, padding),
        }
    }

    /// Arbitrary kernel/stride/padding constructor
    pub fn with_params(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        MaxPool2d {
            kernel,
            stride,
            padding,
        }
    }

    fn pool_forward(&self, x: &Tensor) -> Tensor {
        let (batch, channels, _height, _width) = {
            let x_borrow = x.borrow();
            assert_eq!(
                x_borrow.shape.len(),
                4,
                "MaxPool2d expects input shape (B, C, H, W)"
            );
            (
                x_borrow.shape[0],
                x_borrow.shape[1],
                x_borrow.shape[2],
                x_borrow.shape[3],
            )
        };

        let (pad_h, pad_w) = self.padding;
        let x_padded = if pad_h > 0 || pad_w > 0 {
            x.pad(&[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])
        } else {
            x.clone()
        };

        let (data, shape, requires_grad) = {
            let xp = x_padded.borrow();
            (xp.data.clone(), xp.shape.clone(), xp.requires_grad)
        };

        let (padded_h, padded_w) = (shape[2], shape[3]);
        let (kernel_h, kernel_w) = self.kernel;
        let (stride_h, stride_w) = self.stride;

        assert!(kernel_h > 0 && kernel_w > 0, "Kernel size must be positive");
        assert!(stride_h > 0 && stride_w > 0, "Stride must be positive");
        assert!(
            padded_h >= kernel_h && padded_w >= kernel_w,
            "Kernel larger than input"
        );

        let h_out = (padded_h - kernel_h) / stride_h + 1;
        let w_out = (padded_w - kernel_w) / stride_w + 1;

        let mut out_data = vec![0.0; batch * channels * h_out * w_out];
        let mut max_indices = vec![0usize; out_data.len()];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let out_idx = (((b * channels) + c) * h_out + oh) * w_out + ow;
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;

                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0usize;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let h_in = h_start + kh;
                                let w_in = w_start + kw;
                                let in_idx =
                                    (((b * channels + c) * padded_h) + h_in) * padded_w + w_in;
                                let val = data[in_idx];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = in_idx;
                                }
                            }
                        }

                        out_data[out_idx] = max_val;
                        max_indices[out_idx] = max_idx;
                    }
                }
            }
        }

        let out_shape = vec![batch, channels, h_out, w_out];
        let out = RawTensor::new(out_data, &out_shape, requires_grad);

        if requires_grad {
            out.borrow_mut().parents = vec![x_padded];
            out.borrow_mut().grad_fn = Some(Box::new(MaxPool2dGradFn { max_indices }));
        }

        out
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.pool_forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn load_state_dict(&mut self, _state: &StateDict) {
        // Stateless
    }
}

impl GradFn for MaxPool2dGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let input_shape = parents[0].borrow().shape.clone();
        let mut grad_input = vec![0.0; input_shape.iter().product()];

        for (idx, &max_linear_idx) in self.max_indices.iter().enumerate() {
            grad_input[max_linear_idx] += out_grad.data[idx];
        }

        vec![Some(RawTensor::new(grad_input, &input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::MaxPool2d;
    use crate::nn::Module;
    use crate::tensor::{RawTensor, TensorOps};

    #[test]
    fn test_maxpool2d_forward_shape() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[1, 3, 32, 32]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_maxpool2d_forward_values() {
        let pool = MaxPool2d::new(2, 2, 0);
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let x = RawTensor::new(data, &[1, 1, 4, 4], false);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_maxpool2d_forward_shape_with_padding() {
        let pool = MaxPool2d::new(2, 2, 1);
        let x = RawTensor::randn(&[1, 1, 3, 3]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_backward_flow() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[1, 1, 4, 4]);
        x.borrow_mut().requires_grad = true;
        let y = pool.forward(&x);
        let loss = y.sum();
        loss.backward();
        assert!(x.grad().is_some());
    }

    #[test]
    //CARE: sometimes fails grad check, need to investigate why
    fn test_maxpool2d_gradcheck() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[1, 1, 4, 4]);
        x.borrow_mut().requires_grad = true;
        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d gradient check failed");
    }
}
