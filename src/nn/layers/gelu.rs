use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::TensorOps;
use crate::tensor::{RawTensor, Tensor};

/// Gaussian Error Linear Unit (GELU) activation function.
///
/// GELU is a smooth, non-linear activation function commonly used in transformer models
/// (BERT, GPT, etc.). It provides better performance than `ReLU` for deep networks.
///
/// # Formula
/// ```text
/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
/// ```
///
/// where erf is the error function.
///
/// # Properties
/// - Smooth and differentiable everywhere
/// - Approximates `ReLU` but with a gentle curve
/// - Zero-centered (unlike `ReLU` which is always non-negative)
/// - No learnable parameters (stateless)
///
/// # Example
/// ```
/// # use volta::GELU;
/// # use volta::Module;
/// # use volta::TensorOps;
/// # use volta::RawTensor;
/// let gelu = GELU::new();
/// let x = RawTensor::new(vec![1.0, -1.0, 0.0], &[3], false);
/// let y = gelu.forward(&x);
/// // y ≈ [0.8413, -0.1588, 0.0]
/// ```
pub struct GELU;

impl GELU {
    /// Create a new GELU activation layer
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        let sqrt2 = RawTensor::new(vec![(2.0_f32).sqrt()], &[1], false);
        let one = RawTensor::new(vec![1.0], &[1], false);
        let half = RawTensor::new(vec![0.5], &[1], false);

        x.div(&sqrt2).erf().add(&one).elem_mul(x).elem_mul(&half)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![] // No learnable parameters
    }

    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn load_state_dict(&mut self, _state: &StateDict) {
        // Stateless activation, nothing to load
    }

    fn train(&mut self, _mode: bool) {
        // Stateless, no training/eval behavior difference
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;
    use approx::assert_relative_eq;

    #[test]
    fn test_gelu_creation() {
        let gelu = GELU::new();
        // GELU is a zero-sized type, just verify it can be created
        let _ = gelu.forward(&RawTensor::new(vec![0.0], &[1], false));
    }

    #[test]
    fn test_gelu_default() {
        let gelu = GELU;
        let _ = gelu.forward(&RawTensor::new(vec![0.0], &[1], false));
    }

    #[test]
    fn test_gelu_forward() {
        let gelu = GELU::new();

        // Test gelu(0) = 0
        let x = RawTensor::new(vec![0.0], &[1], false);
        let y = gelu.forward(&x);
        assert_relative_eq!(
            y.borrow().data.first().copied().unwrap(),
            0.0,
            epsilon = 1e-6
        );

        // Test gelu(1) ≈ 0.8413
        let x = RawTensor::new(vec![1.0], &[1], false);
        let y = gelu.forward(&x);
        assert_relative_eq!(
            y.borrow().data.first().copied().unwrap(),
            0.8413,
            epsilon = 1e-4
        );

        // Test gelu(-1) ≈ -0.1588
        let x = RawTensor::new(vec![-1.0], &[1], false);
        let y = gelu.forward(&x);
        assert_relative_eq!(
            y.borrow().data.first().copied().unwrap(),
            -0.1587,
            epsilon = 1e-3
        );

        // Test gelu(0.56484) ≈ 0.4032
        // This is a test point where the tanh approximation would fail
        let x = RawTensor::new(vec![0.56484], &[1], false);
        let y = gelu.forward(&x);
        assert_relative_eq!(
            y.borrow().data.first().copied().unwrap(),
            0.4032,
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_gelu_shape() {
        let gelu = GELU::new();

        // Test 1D input
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
        let y = gelu.forward(&x);
        assert_eq!(y.borrow().shape, vec![3]);

        // Test 2D input
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = gelu.forward(&x);
        assert_eq!(y.borrow().shape, vec![2, 2]);

        // Test 3D input
        let x = RawTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
            false,
        );
        let y = gelu.forward(&x);
        assert_eq!(y.borrow().shape, vec![2, 2, 2]);
    }

    #[test]
    fn test_gelu_negative() {
        let gelu = GELU::new();
        let x = RawTensor::new(vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], &[7], false);
        let y = gelu.forward(&x);

        // GELU should preserve sign for negative inputs (unlike ReLU)
        // and be close to zero for small negative values
        let data = y.borrow().data.clone();
        assert!(
            *data.first().unwrap() < -0.04,
            "gelu(-2.0) should be significantly negative"
        );
        assert!(
            *data.get(1).unwrap() < -0.15,
            "gelu(-1.0) should be negative"
        );
        assert!(
            *data.get(2).unwrap() < -0.04,
            "gelu(-0.5) should be slightly negative"
        );
        assert_relative_eq!(*data.get(3).unwrap(), 0.0, epsilon = 1e-6);
        assert!(*data.get(4).unwrap() > 0.0, "gelu(0.5) should be positive");
        assert!(
            *data.get(5).unwrap() > 0.8,
            "gelu(1.0) should be close to 0.84"
        );
        assert!(
            *data.get(6).unwrap() > 1.9,
            "gelu(2.0) should be close to 2.0"
        );
    }

    #[test]
    fn test_gelu_state_dict() {
        let gelu = GELU::new();
        let state = gelu.state_dict();
        assert!(state.is_empty(), "GELU state dict should be empty");

        let mut gelu = GELU::new();
        gelu.load_state_dict(&state); // Should not panic
    }

    #[test]
    fn test_gelu_parameters() {
        let gelu = GELU::new();
        let params = gelu.parameters();
        assert!(
            params.is_empty(),
            "GELU should have no learnable parameters"
        );
    }

    #[test]
    fn test_gelu_train_eval() {
        let mut gelu = GELU::new();
        gelu.train(true); // Training mode
        gelu.eval(); // Eval mode

        // Should not panic and behavior should be the same
        let x = RawTensor::new(vec![1.0], &[1], false);
        let y = gelu.forward(&x);
        assert_relative_eq!(
            y.borrow().data.first().copied().unwrap(),
            0.8413,
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_gelu_gradient() {
        let gelu = GELU::new();
        let x = RawTensor::new(vec![1.0], &[1], true);
        let y = gelu.forward(&x);
        y.backward();

        // GELU gradient should exist and be non-zero
        let grad = x.grad().expect("Gradient should exist");
        assert!(!grad.is_empty(), "Gradient should not be empty");

        // At x=1, the gradient should be positive
        // The gradient of GELU at x=1 is approximately 1.08
        let grad_val = grad.first().copied().unwrap();
        assert!(grad_val > 0.0, "GELU gradient at x=1 should be positive");
        assert_relative_eq!(grad_val, 1.08, epsilon = 1e-2);
    }
}
