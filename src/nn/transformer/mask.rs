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
    let mask = RawTensor::new(mask_values(seq_len, seq_len), &[seq_len, seq_len], false);
    mask.to_device(device)
}

/// Additive mask `[q_len, kv_len]`: `-inf` where `j > i`, 0 elsewhere, on `like`'s device
/// and in its dtype. This is torch's top-left aligned `is_causal` mask
/// (`ones(L, S).tril()`); it broadcasts against any `[.., L, S]` score tensor.
pub(crate) fn causal_mask_like(q_len: usize, kv_len: usize, like: &Tensor) -> Tensor {
    like.constant_like(mask_values(q_len, kv_len), &[q_len, kv_len])
}

/// Row-major `[q_len, kv_len]` values of the causal mask: `-inf` above the diagonal
/// (`j > i`, a future key), 0 on and below it.
fn mask_values(q_len: usize, kv_len: usize) -> Vec<f32> {
    (0..q_len)
        .flat_map(|i| (0..kv_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect()
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
        let data = mask.data();

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
        assert_relative_eq!(*mask.data().first().unwrap(), 0.0);
    }

    #[test]
    fn test_causal_mask_size_2() {
        let mask = causal_mask(2, crate::Device::CPU);
        let data = mask.data();

        // Row 0: [0, -inf]
        assert_relative_eq!(*data.first().unwrap(), 0.0);
        assert!(data.get(1).unwrap().is_infinite() && data.get(1).unwrap().is_sign_negative());

        // Row 1: [0, 0]
        assert_relative_eq!(*data.get(2).unwrap(), 0.0);
        assert_relative_eq!(*data.get(3).unwrap(), 0.0);
    }
}
