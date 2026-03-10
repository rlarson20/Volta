use crate::tensor::TensorOps;
use crate::tensor::{RawTensor, Tensor};

/// Generate a causal (lower-triangular) mask for autoregressive attention
///
/// In GPT-style models, each token can only attend to previous tokens (including itself).
/// This function creates a mask matrix where:
/// - Mask[i, j] = 0 if j <= i (can attend)
/// - Mask[i, j] = -inf if j > i (cannot attend to future tokens)
///
/// # Arguments
/// * `seq_len` - Sequence length
/// * `device` - Device to create the mask on (CPU or GPU)
///
/// # Returns
/// A tensor of shape [`seq_len`, `seq_len`] with -inf in the upper triangle and 0 in the lower triangle
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::causal_mask;
/// # use volta::Device;
/// let mask = causal_mask(4, Device::CPU);
/// // mask = [[0, -inf, -inf, -inf],
/// //          [0,    0, -inf, -inf],
/// //          [0,    0,    0, -inf],
/// //          [0,    0,    0,    0]]
/// ```
#[must_use]
pub fn causal_mask(seq_len: usize, device: crate::Device) -> Tensor {
    let mut mask_data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            // Upper triangle (j > i) gets -inf, lower triangle and diagonal get 0
            if j > i {
                mask_data.push(f32::NEG_INFINITY);
            } else {
                mask_data.push(0.0);
            }
        }
    }

    let mask = RawTensor::new(mask_data, &[seq_len, seq_len], false);
    mask.to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_causal_mask_shape() {
        let mask = causal_mask(4, crate::Device::CPU);
        assert_eq!(mask.borrow().shape, vec![4, 4]);
    }

    #[test]
    fn test_causal_mask_values() {
        let mask = causal_mask(3, crate::Device::CPU);
        let data = mask.borrow().data.clone();

        // Check lower triangle and diagonal (should be 0)
        assert_relative_eq!(*data.first().unwrap(), 0.0); // [0, 0]
        assert_relative_eq!(*data.get(3).unwrap(), 0.0); // [1, 0]
        assert_relative_eq!(*data.get(4).unwrap(), 0.0); // [1, 1]
        assert_relative_eq!(*data.get(6).unwrap(), 0.0); // [2, 0]
        assert_relative_eq!(*data.get(7).unwrap(), 0.0); // [2, 1]
        assert_relative_eq!(*data.get(8).unwrap(), 0.0); // [2, 2]

        // Check upper triangle (should be -inf)
        assert!(data.get(1).unwrap().is_infinite() && data.get(1).unwrap().is_sign_negative()); // [0, 1]
        assert!(data.get(2).unwrap().is_infinite() && data.get(2).unwrap().is_sign_negative()); // [0, 2]
        assert!(data.get(5).unwrap().is_infinite() && data.get(5).unwrap().is_sign_negative()); // [1, 2]
    }

    #[test]
    fn test_causal_mask_size_1() {
        let mask = causal_mask(1, crate::Device::CPU);
        assert_eq!(mask.borrow().shape, vec![1, 1]);
        assert_relative_eq!(*mask.borrow().data.first().unwrap(), 0.0);
    }

    #[test]
    fn test_causal_mask_size_2() {
        let mask = causal_mask(2, crate::Device::CPU);
        let data = mask.borrow().data.clone();

        // Row 0: [0, -inf]
        assert_relative_eq!(*data.first().unwrap(), 0.0);
        assert!(data.get(1).unwrap().is_infinite() && data.get(1).unwrap().is_sign_negative());

        // Row 1: [0, 0]
        assert_relative_eq!(*data.get(2).unwrap(), 0.0);
        assert_relative_eq!(*data.get(3).unwrap(), 0.0);
    }
}
