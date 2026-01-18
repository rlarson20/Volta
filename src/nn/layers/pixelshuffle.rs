use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::{Tensor, TensorOps};

/// `PixelShuffle`: Rearranges elements from (B, C*r², H, W) to (B, C, H*r, W*r)
///
/// This layer implements the efficient sub-pixel convolution layer described in
/// "Real-Time Single Image and Video Super-Resolution Using an Efficient
/// Sub-Pixel Convolutional Neural Network" (Shi et al., 2016).
///
/// # Arguments
/// * `upscale_factor` - Factor to increase spatial resolution by
///
/// # Shape
/// - Input: (B, C*r², H, W) where r is the `upscale_factor`
/// - Output: (B, C, H*r, W*r)
///
/// # Examples
/// ```
/// use volta::{RawTensor, PixelShuffle, nn::Module};
///
/// let layer = PixelShuffle::new(2);
/// let input = RawTensor::randn(&[1, 12, 4, 4]);  // 3 channels * 4
/// let output = layer.forward(&input);
/// assert_eq!(output.borrow().shape, vec![1, 3, 8, 8]);
/// ```
pub struct PixelShuffle {
    upscale_factor: usize,
}

impl PixelShuffle {
    /// Creates a new `PixelShuffle` layer with the given upscale factor
    ///
    /// # Arguments
    /// * `upscale_factor` - Factor to increase spatial resolution by (must be > 0)
    ///
    /// # Panics
    /// Panics if `upscale_factor` is 0
    #[must_use]
    pub fn new(upscale_factor: usize) -> Self {
        assert!(
            upscale_factor > 0,
            "PixelShuffle: upscale_factor must be positive, got {upscale_factor}",
        );
        PixelShuffle { upscale_factor }
    }
}

impl Module for PixelShuffle {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_ref = x.borrow();
        let shape = &x_ref.shape;

        // Validate input is 4D
        assert_eq!(
            shape.len(),
            4,
            "PixelShuffle: expected 4D input (B, C, H, W), got shape {shape:?}",
        );

        let batch_size = shape.first().copied().unwrap_or(1);
        let in_channels = shape.get(1).copied().unwrap_or(1);
        let height = shape.get(2).copied().unwrap_or(1);
        let width = shape.get(3).copied().unwrap_or(1);
        let r = self.upscale_factor;
        let r_squared = r * r;

        // Validate that channels are divisible by r²
        assert_eq!(
            in_channels % r_squared,
            0,
            "PixelShuffle: input channels ({in_channels}) must be divisible by upscale_factor² ({r_squared})",
        );

        let out_channels = in_channels / r_squared;
        drop(x_ref);

        // Step 1: Reshape (B, C*r², H, W) → (B, C, r, r, H, W)
        let x_reshaped = x.reshape(&[batch_size, out_channels, r, r, height, width]);

        // Step 2: Permute [0, 1, 4, 2, 5, 3] → (B, C, H, r, W, r)
        let x_permuted = x_reshaped.permute(&[0, 1, 4, 2, 5, 3]);

        // Step 3: Reshape (B, C, H, r, W, r) → (B, C, H*r, W*r)
        x_permuted.reshape(&[batch_size, out_channels, height * r, width * r])
    }

    fn parameters(&self) -> Vec<Tensor> {
        // PixelShuffle has no learnable parameters
        vec![]
    }

    fn state_dict(&self) -> StateDict {
        // No parameters to save
        StateDict::new()
    }

    fn load_state_dict(&mut self, _state: &StateDict) {
        // No parameters to load
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;

    #[test]
    fn test_pixelshuffle_shape() {
        let layer = PixelShuffle::new(2);
        let x = RawTensor::randn(&[2, 12, 4, 4]); // 3 channels * 4
        let y = layer.forward(&x);
        assert_eq!(y.borrow().shape, vec![2, 3, 8, 8]);
    }

    #[test]
    fn test_pixelshuffle_upscale_3() {
        let layer = PixelShuffle::new(3);
        let x = RawTensor::randn(&[1, 36, 5, 5]); // 4 channels * 9
        let y = layer.forward(&x);
        assert_eq!(y.borrow().shape, vec![1, 4, 15, 15]);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_pixelshuffle_invalid_channels() {
        let layer = PixelShuffle::new(2);
        let x = RawTensor::randn(&[1, 5, 4, 4]); // 5 is not divisible by 4
        layer.forward(&x);
    }

    #[test]
    #[should_panic(expected = "expected 4D input")]
    fn test_pixelshuffle_invalid_dims() {
        let layer = PixelShuffle::new(2);
        let x = RawTensor::randn(&[12, 4, 4]); // 3D input
        layer.forward(&x);
    }

    #[test]
    fn test_pixelshuffle_gradient_flow() {
        let layer = PixelShuffle::new(2);
        let x = RawTensor::randn(&[1, 4, 3, 3]);
        x.borrow_mut().requires_grad = true;

        let y = layer.forward(&x);
        let loss = y.sum();
        loss.backward();

        let grad = x.grad();
        assert!(
            grad.is_some(),
            "Gradient should flow back through PixelShuffle"
        );
        // Check gradient has correct number of elements: 1 * 4 * 3 * 3 = 36
        assert_eq!(grad.unwrap().len(), 36);
    }
}
