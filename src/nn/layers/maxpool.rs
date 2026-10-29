use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::index::{fetch, slot};
use crate::io::StateDict;
use crate::nn::{Layer, Module};
use crate::shape::Shape;
use crate::spec::{PadValue, Window2dSpec};
use crate::storage::Storage;
use crate::tensor::{RawTensor, Tensor};

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
pub(crate) struct MaxPool2dGradFn {
    max_indices: Vec<usize>,
}

impl MaxPool2d {
    /// Square-kernel constructor for convenience
    #[must_use]
    pub const fn new(kernel: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel: (kernel, kernel),
            stride: (stride, stride),
            padding: (padding, padding),
        }
    }

    /// Arbitrary kernel/stride/padding constructor
    #[must_use]
    pub const fn with_params(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel,
            stride,
            padding,
        }
    }

    fn pool_forward(&self, x: &Tensor) -> Tensor {
        let spec = Window2dSpec::new(
            &Shape::derived(x.shape()),
            self.kernel,
            self.stride,
            self.padding,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let Window2dSpec {
            batch,
            channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: h_out,
            out_w: w_out,
            ..
        } = spec;
        // A window made only of -inf padding would output -inf and route its
        // gradient to index 0, so padding may cover at most half the kernel
        // (PyTorch's rule). Conv shares `Window2dSpec` and allows more.
        assert!(
            pad_h <= kernel_h / 2 && pad_w <= kernel_w / 2,
            "MaxPool2d padding must be at most half the kernel: padding {:?} with kernel {:?}",
            (pad_h, pad_w),
            (kernel_h, kernel_w),
        );
        let (padded_h, padded_w) = (spec.padded_h(), spec.padded_w());

        let x_padded = if pad_h > 0 || pad_w > 0 {
            x.pad_with(
                &[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)],
                PadValue::NEG_INF,
            )
        } else {
            x.clone()
        };

        let data = x_padded.borrow().data.clone();
        let data = data.to_host_f32();

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
                                let val = fetch(&data, in_idx);
                                if val > max_val {
                                    max_val = val;
                                    max_idx = in_idx;
                                }
                            }
                        }

                        *slot(&mut out_data, out_idx) = max_val;
                        *slot(&mut max_indices, out_idx) = max_idx;
                    }
                }
            }
        }

        edge(
            [x_padded],
            Storage::cpu(out_data),
            Shape::derived(spec.output_dims()),
            Device::CPU,
            GradFn::MaxPool2d(MaxPool2dGradFn { max_indices }),
        )
    }
}

impl Module for MaxPool2d {
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

impl Layer for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.pool_forward(x)
    }
}

impl MaxPool2dGradFn {
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let input_shape = parents.first().unwrap().borrow().shape.clone();
        let mut grad_input = vec![0.0; input_shape.iter().product()];

        let g_host = out_grad.data.to_host_f32();
        for (idx, &max_linear_idx) in self.max_indices.iter().enumerate() {
            let grad_val = fetch(&g_host, idx);
            *slot(&mut grad_input, max_linear_idx) += grad_val;
        }

        vec![Some(RawTensor::new(grad_input, &input_shape, false))]
    }
}

#[cfg(test)]
mod tests {
    use super::MaxPool2d;
    use crate::nn::Layer;
    use crate::tensor::RawTensor;

    // ==================== Basic Existing Tests ====================

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
        loss.into_loss().expect("scalar loss").backward();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_maxpool2d_gradcheck() {
        let pool = MaxPool2d::new(2, 2, 0);
        // Use deterministic, strictly increasing data so every pooling window
        // has a unique maximum that is well separated from the others. This
        // avoids the non-differentiable tie cases that cause the numerical
        // gradient checker to spuriously fail.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = RawTensor::new(data, &[1, 1, 4, 4], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d gradient check failed");
    }

    // ==================== Parameter Configuration Tests ====================

    #[test]
    fn test_maxpool2d_kernel_1x1() {
        let pool = MaxPool2d::new(1, 1, 0);
        let x = RawTensor::randn(&[1, 3, 8, 8]);
        let y = pool.forward(&x);
        // 1x1 kernel with stride 1: output shape equals input shape
        assert_eq!(y.borrow().shape, vec![1, 3, 8, 8]);
    }

    #[test]
    fn test_maxpool2d_kernel_3x3() {
        let pool = MaxPool2d::new(3, 2, 0);
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let y = pool.forward(&x);
        // (8 - 3) / 2 + 1 = 3.5 -> 3
        assert_eq!(y.borrow().shape, vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_maxpool2d_kernel_4x4() {
        let pool = MaxPool2d::new(4, 2, 0);
        let x = RawTensor::randn(&[2, 3, 12, 12]);
        let y = pool.forward(&x);
        // (12 - 4) / 2 + 1 = 5
        assert_eq!(y.borrow().shape, vec![2, 3, 5, 5]);
    }

    #[test]
    fn test_maxpool2d_kernel_5x5() {
        let pool = MaxPool2d::new(5, 1, 0);
        let x = RawTensor::randn(&[1, 1, 10, 10]);
        let y = pool.forward(&x);
        // (10 - 5) / 1 + 1 = 6
        assert_eq!(y.borrow().shape, vec![1, 1, 6, 6]);
    }

    #[test]
    fn test_maxpool2d_stride_1() {
        let pool = MaxPool2d::new(2, 1, 0);
        let x = RawTensor::randn(&[1, 3, 8, 8]);
        let y = pool.forward(&x);
        // (8 - 2) / 1 + 1 = 7
        assert_eq!(y.borrow().shape, vec![1, 3, 7, 7]);
    }

    #[test]
    fn test_maxpool2d_stride_3() {
        let pool = MaxPool2d::new(3, 3, 0);
        let x = RawTensor::randn(&[1, 1, 12, 12]);
        let y = pool.forward(&x);
        // (12 - 3) / 3 + 1 = 4
        assert_eq!(y.borrow().shape, vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_stride_4() {
        let pool = MaxPool2d::new(4, 4, 0);
        let x = RawTensor::randn(&[2, 2, 16, 16]);
        let y = pool.forward(&x);
        // (16 - 4) / 4 + 1 = 4
        assert_eq!(y.borrow().shape, vec![2, 2, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_padding_1() {
        let pool = MaxPool2d::new(2, 2, 1);
        let x = RawTensor::randn(&[1, 3, 8, 8]);
        let y = pool.forward(&x);
        // Padded: 8 + 2 = 10, (10 - 2) / 2 + 1 = 5
        assert_eq!(y.borrow().shape, vec![1, 3, 5, 5]);
    }

    #[test]
    fn test_maxpool2d_padding_2() {
        let pool = MaxPool2d::new(4, 2, 2);
        let x = RawTensor::randn(&[1, 1, 6, 6]);
        let y = pool.forward(&x);
        // Padded: 6 + 4 = 10, (10 - 4) / 2 + 1 = 4
        assert_eq!(y.borrow().shape, vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_padding_3() {
        let pool = MaxPool2d::new(7, 2, 3);
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let y = pool.forward(&x);
        // Padded: 8 + 6 = 14, (14 - 7) / 2 + 1 = 4
        assert_eq!(y.borrow().shape, vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_asymmetric_params() {
        let pool = MaxPool2d::with_params((2, 3), (2, 2), (1, 1));
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let y = pool.forward(&x);
        // Height: padded 8 + 2 = 10, (10 - 2) / 2 + 1 = 5
        // Width: padded 8 + 2 = 10, (10 - 3) / 2 + 1 = 4
        assert_eq!(y.borrow().shape, vec![1, 1, 5, 4]);
    }

    #[test]
    fn test_maxpool2d_multiple_channels_1() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[1, 1, 4, 4]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_multiple_channels_3() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[1, 3, 4, 4]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 3, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_multiple_channels_16() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[1, 16, 8, 8]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 16, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_batch_size_1() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[1, 3, 8, 8]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 3, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_batch_size_4() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[4, 3, 8, 8]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![4, 3, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_batch_size_8() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[8, 16, 4, 4]);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![8, 16, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_shape_formula() {
        // Test the output shape formula: h_out = (h + 2*pad - kernel) / stride + 1
        let pool = MaxPool2d::new(3, 2, 1);
        let x = RawTensor::randn(&[2, 4, 10, 10]);
        let y = pool.forward(&x);

        // Expected: (10 + 2*1 - 3) / 2 + 1 = (10 + 2 - 3) / 2 + 1 = 9 / 2 + 1 = 4 + 1 = 5
        assert_eq!(y.borrow().shape, vec![2, 4, 5, 5]);
    }

    // ==================== Gradient Correctness Tests ====================

    #[test]
    fn test_maxpool2d_gradcheck_small_kernel() {
        let pool = MaxPool2d::new(2, 2, 0);
        // Use strictly increasing data to avoid ties
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let x = RawTensor::new(data, &[1, 1, 8, 8], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d small kernel gradient check failed");
    }

    #[test]
    fn test_maxpool2d_gradcheck_large_kernel() {
        let pool = MaxPool2d::new(5, 2, 0);
        // Use strictly increasing data to avoid ties
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let x = RawTensor::new(data, &[1, 1, 16, 16], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d large kernel gradient check failed");
    }

    #[test]
    fn test_maxpool2d_gradcheck_with_padding() {
        let pool = MaxPool2d::new(3, 2, 1);
        // Use strictly increasing data to avoid ties
        let data: Vec<f32> = (0..144).map(|i| i as f32 * 0.05).collect();
        let x = RawTensor::new(data, &[1, 1, 12, 12], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d with padding gradient check failed");
    }

    #[test]
    fn test_maxpool2d_gradcheck_multiple_channels() {
        let pool = MaxPool2d::new(2, 2, 0);
        // Use strictly increasing data to avoid ties
        let data: Vec<f32> = (0..192).map(|i| i as f32 * 0.01).collect();
        let x = RawTensor::new(data, &[1, 3, 8, 8], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d multiple channels gradient check failed");
    }

    #[test]
    fn test_maxpool2d_gradcheck_large_stride() {
        let pool = MaxPool2d::new(2, 3, 0);
        // Use strictly increasing data to avoid ties
        let data: Vec<f32> = (0..144).map(|i| i as f32 * 0.02).collect();
        let x = RawTensor::new(data, &[1, 1, 12, 12], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d large stride gradient check failed");
    }

    #[test]
    fn test_maxpool2d_gradcheck_asymmetric() {
        let pool = MaxPool2d::with_params((2, 3), (2, 2), (1, 0));
        // Use strictly increasing data to avoid ties
        let data: Vec<f32> = (0..80).map(|i| i as f32 * 0.1).collect();
        let x = RawTensor::new(data, &[1, 1, 8, 10], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| pool.forward(t).sum());
        assert!(passed, "MaxPool2d asymmetric gradient check failed");
    }

    // ==================== Index Tracking Tests ====================

    #[test]
    fn test_maxpool2d_index_routing() {
        let pool = MaxPool2d::new(2, 2, 0);
        // Create data where we know which position is max in each window
        // Layout is row-major: [0,0], [0,1], [0,2], [0,3], [1,0], [1,1], [1,2], [1,3], ...
        // Window 1 (top-left, indices 0,1,4,5): [1, 2, 3, 8] -> 8 is max (index 5)
        // Window 2 (top-right, indices 2,3,6,7): [4, 5, 6, 16] -> 16 is max (index 7)
        // Window 3 (bottom-left, indices 8,9,12,13): [7, 8, 9, 14] -> 14 is max (index 13)
        // Window 4 (bottom-right, indices 10,11,14,15): [10, 11, 12, 15] -> 15 is max (index 15)
        let data = vec![
            1.0, 2.0, 4.0, 5.0, // row 0: [0,0], [0,1], [0,2], [0,3]
            3.0, 8.0, 6.0, 16.0, // row 1: [1,0], [1,1], [1,2], [1,3]
            7.0, 8.0, 10.0, 11.0, // row 2: [2,0], [2,1], [2,2], [2,3]
            9.0, 14.0, 12.0, 15.0, // row 3: [3,0], [3,1], [3,2], [3,3]
        ];
        let x = RawTensor::new(data, &[1, 1, 4, 4], true);
        let y = pool.forward(&x);
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        let grad = x.grad().unwrap();
        let grad_data = grad.clone();

        // Only max positions should receive gradient (value 1.0 from sum)
        // Window 1 max: [1,1] position, index 5 (value 8.0)
        // Window 2 max: [1,3] position, index 7 (value 16.0)
        // Window 3 max: [3,1] position, index 13 (value 14.0)
        // Window 4 max: [3,3] position, index 15 (value 15.0)

        // Verify that max positions have gradient
        assert_eq!(
            grad_data.get(5).copied().unwrap(),
            1.0,
            "Max position in window 1 should have gradient"
        );
        assert_eq!(
            grad_data.get(7).copied().unwrap(),
            1.0,
            "Max position in window 2 should have gradient"
        );
        assert_eq!(
            grad_data.get(13).copied().unwrap(),
            1.0,
            "Max position in window 3 should have gradient"
        );
        assert_eq!(
            grad_data.get(15).copied().unwrap(),
            1.0,
            "Max position in window 4 should have gradient"
        );

        // Verify that non-max positions have zero gradient
        for (i, &g) in grad_data.iter().enumerate() {
            if ![5, 7, 13, 15].contains(&i) {
                assert_eq!(
                    g, 0.0,
                    "Non-max position at index {i} should have zero gradient"
                );
            }
        }
    }

    #[test]
    fn test_maxpool2d_gradient_accumulation() {
        // Test gradient accumulation when stride=1 (overlapping windows)
        // Use a 3x3 input with 2x2 kernel stride=1 for clearer accumulation
        // Output is 2x2, with 4 overlapping windows:
        // - Window [0,0]: positions 0,1,3,4
        // - Window [0,1]: positions 1,2,4,5
        // - Window [1,0]: positions 3,4,6,7
        // - Window [1,1]: positions 4,5,7,8
        // Position 4 ([1,1]) appears in all 4 windows
        let pool = MaxPool2d::new(2, 1, 0);
        // Create data where center position is the unique max in all windows
        let data = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 100.0, 6.0, // row 1: position 4 has 100.0
            7.0, 8.0, 9.0, // row 2
        ];
        let x = RawTensor::new(data, &[1, 1, 3, 3], true);

        let y = pool.forward(&x);
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        let grad = x.grad().unwrap();
        let grad_data = grad.clone();

        // Position 4 (center) should be max in all 4 windows
        // So it should accumulate gradient of 4.0
        assert_eq!(
            grad_data.get(4).copied().unwrap(),
            4.0,
            "Position 4 should have accumulated gradient of 4.0"
        );
        // All other positions should have zero gradient
        for (i, &g) in grad_data.iter().enumerate() {
            if i != 4 {
                assert_eq!(g, 0.0, "Non-max position {i} should have zero gradient");
            }
        }
    }

    #[test]
    fn test_maxpool2d_overlap_windows() {
        // Test gradient routing with overlapping windows (stride < kernel)
        let pool = MaxPool2d::new(2, 1, 0);
        // Use strictly increasing data to avoid ties
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = RawTensor::new(data, &[1, 1, 4, 4], true);

        let y = pool.forward(&x);
        // Output shape should be 3x3 with stride 1
        assert_eq!(y.borrow().shape, vec![1, 1, 3, 3]);

        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        let grad = x.grad().unwrap();
        let grad_data = grad.clone();

        // With overlapping windows, gradients should accumulate
        // Sum of input gradients should equal number of outputs (9)
        let sum_grad: f32 = grad_data.iter().sum();
        assert_eq!(
            sum_grad, 9.0,
            "Sum of gradients should equal number of outputs"
        );
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_maxpool2d_single_pixel_1x1_kernel() {
        let pool = MaxPool2d::new(1, 1, 0);
        let data = vec![5.0];
        let x = RawTensor::new(data, &[1, 1, 1, 1], true);
        let y = pool.forward(&x);

        // 1x1 kernel with stride 1 on 1x1 input: output is 1x1 with same value
        assert_eq!(y.borrow().shape, vec![1, 1, 1, 1]);
        assert_eq!(y.item(), 5.0);

        // Gradient should flow correctly
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();
        let grad = x.grad().unwrap();
        assert_eq!(grad.clone().first().copied().unwrap(), 1.0);
    }

    #[test]
    fn test_maxpool2d_input_equals_kernel() {
        let pool = MaxPool2d::new(4, 1, 0);
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = RawTensor::new(data, &[1, 1, 4, 4], true);
        let y = pool.forward(&x);

        // 4x4 kernel with stride 1 on 4x4 input: output is 1x1 (global max pool)
        assert_eq!(y.borrow().shape, vec![1, 1, 1, 1]);
        assert_eq!(y.item(), 15.0); // Max is 15.0

        // Gradient should flow to the max position only
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();
        let grad = x.grad().unwrap();
        let grad_data = grad.clone();

        // Only the last position (value 15.0) should have gradient
        assert_eq!(
            grad_data.get(15).copied().unwrap(),
            1.0,
            "Max position should have gradient"
        );
        for (i, &g) in grad_data.iter().enumerate() {
            if i != 15 {
                assert_eq!(g, 0.0, "Non-max position {i} should have zero gradient");
            }
        }
    }

    #[test]
    fn test_maxpool2d_batch_gradient_isolation() {
        // Test that gradients for different batches don't interfere
        let pool = MaxPool2d::new(2, 2, 0);
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let x = RawTensor::new(data, &[2, 1, 4, 4], true);

        let y = pool.forward(&x);
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        let grad = x.grad().unwrap();
        let grad_data = grad.clone();

        // Each batch should have its own gradients
        // Each batch has 16 elements (1 channel * 4 * 4)
        // Verify that gradients exist for both batches
        let batch0_sum: f32 = grad_data.get(0..16).unwrap().iter().sum();
        let batch1_sum: f32 = grad_data.get(16..32).unwrap().iter().sum();

        assert!(batch0_sum > 0.0, "Batch 0 should have gradients");
        assert!(batch1_sum > 0.0, "Batch 1 should have gradients");
    }

    #[test]
    fn test_maxpool2d_channel_gradient_isolation() {
        // Test that gradients for different channels don't interfere
        let pool = MaxPool2d::new(2, 2, 0);
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let x = RawTensor::new(data, &[1, 4, 4, 4], true);

        let y = pool.forward(&x);
        let loss = y.sum();
        loss.into_loss().expect("scalar loss").backward();

        let grad = x.grad().unwrap();
        let grad_data = grad.clone();

        // Each channel should have its own gradients
        // Each channel has 16 elements (4x4)
        for c in 0..4 {
            let channel_start = c * 16;
            let channel_end = (c + 1) * 16;
            let channel_sum: f32 = grad_data
                .get(channel_start..channel_end)
                .unwrap()
                .iter()
                .sum();
            assert!(channel_sum > 0.0, "Channel {c} should have gradients");
        }
    }

    // ==================== Window geometry errors ====================

    /// Runs `pool.forward(x)` and returns the panic message, failing if it does not panic.
    fn forward_panic_message(pool: &MaxPool2d, x: &crate::tensor::Tensor) -> String {
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.forward(x)))
            .expect_err("forward should have panicked");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("panic payload should be a string")
    }

    #[test]
    fn test_maxpool2d_padding_never_beats_negative_input_h1_repro() {
        // H1: padding with 0.0 made every padded window win with 0.0.
        let pool = MaxPool2d::new(2, 2, 1);
        let x = RawTensor::new(vec![-5.0; 16], &[1, 1, 4, 4], false);
        let y = pool.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 1, 3, 3]);
        let out = y.data();
        assert!(
            out.iter().all(|&v| v == -5.0),
            "padding must not win a window of all -5.0, got {out:?}"
        );
    }

    #[test]
    fn test_maxpool2d_padding_routes_one_unit_per_output_to_real_input() {
        // Shifted negative so that a 0.0 pad would win every border window.
        let pool = MaxPool2d::new(3, 2, 1);
        let x = RawTensor::randn(&[1, 2, 6, 6]);
        let shifted: Vec<f32> = x.data().iter().map(|v| v - 10.0).collect();
        let x = RawTensor::new(shifted, &[1, 2, 6, 6], true);
        let y = pool.forward(&x);
        assert!(y.data().iter().all(|&v| v < 0.0), "output leaked the pad");
        y.sum().into_loss().expect("scalar loss").backward();
        let grad = x.grad().expect("input should receive a gradient");
        let total: f32 = grad.iter().sum();
        assert_eq!(total, y.data().len() as f32, "one unit per output window");
    }

    #[test]
    fn test_maxpool2d_padding_over_half_kernel_panics() {
        let pool = MaxPool2d::new(2, 1, 2);
        let x = RawTensor::randn(&[1, 1, 4, 4]);
        let msg = forward_panic_message(&pool, &x);
        assert!(msg.contains("at most half"), "{msg}");
    }

    #[test]
    fn test_maxpool2d_zero_stride_panics_with_spec_message() {
        let pool = MaxPool2d::new(2, 0, 0);
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let msg = forward_panic_message(&pool, &x);
        assert!(msg.contains("stride must be non-zero"), "{msg}");
    }

    #[test]
    fn test_maxpool2d_zero_kernel_panics_with_spec_message() {
        let pool = MaxPool2d::new(0, 1, 0);
        let x = RawTensor::randn(&[1, 1, 8, 8]);
        let msg = forward_panic_message(&pool, &x);
        assert!(msg.contains("kernel must be positive"), "{msg}");
    }

    #[test]
    fn test_maxpool2d_kernel_larger_than_padded_input_panics_with_spec_message() {
        let pool = MaxPool2d::new(5, 1, 0);
        let x = RawTensor::randn(&[1, 1, 3, 3]);
        let msg = forward_panic_message(&pool, &x);
        assert!(msg.contains("larger than padded input"), "{msg}");
    }

    #[test]
    fn test_maxpool2d_non_4d_input_panics_with_spec_message() {
        let pool = MaxPool2d::new(2, 2, 0);
        let x = RawTensor::randn(&[8, 8]);
        let msg = forward_panic_message(&pool, &x);
        assert!(msg.contains("input must be 4D"), "{msg}");
    }
}
