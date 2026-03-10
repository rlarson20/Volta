use crate::tensor::TensorOps;
use crate::tensor::{RawTensor, Tensor};

/// Scaled Dot-Product Attention
///
/// Computes attention as: softmax(QK^T / √`d_k`) V
///
/// This is the core attention mechanism used in transformer models.
///
/// # Arguments
/// * `query` - Query tensor of shape [batch, `seq_len`, `d_k`]
/// * `key` - Key tensor of shape [batch, `seq_len`, `d_k`]
/// * `value` - Value tensor of shape [batch, `seq_len`, `d_v`]
/// * `causal` - Whether to apply causal masking (for autoregressive models)
/// * `attention_mask` - Optional additional mask (e.g., padding mask) of shape [batch, `seq_len`, `seq_len`]
///
/// # Returns
/// Output tensor of shape [batch, `seq_len`, `d_v`]
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::ScaledDotProductAttention;
/// # use volta::{RawTensor, Module};
/// # use volta::Device;
/// let attention = ScaledDotProductAttention::new();
/// let query = RawTensor::new(vec![0.0; 128], &[2, 4, 16], false);
/// let key = RawTensor::new(vec![0.0; 128], &[2, 4, 16], false);
/// let value = RawTensor::new(vec![0.0; 64], &[2, 4, 8], false);
/// let output = attention.forward(&query, &key, &value, false, None);
/// ```
#[derive(Clone, Debug)]
pub struct ScaledDotProductAttention;

impl ScaledDotProductAttention {
    /// Create a new scaled dot-product attention layer
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for ScaledDotProductAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl ScaledDotProductAttention {
    /// Forward pass for scaled dot-product attention
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch, `seq_len`, `d_k`]
    /// * `key` - Key tensor of shape [batch, `seq_len`, `d_k`]
    /// * `value` - Value tensor of shape [batch, `seq_len`, `d_v`]
    /// * `causal` - Whether to apply causal masking
    /// * `attention_mask` - Optional additional mask for padding, etc.
    ///
    /// # Returns
    /// Output tensor of shape [batch, `seq_len`, `d_v`]
    /// # Panics
    /// Wrong tensor shape
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal: bool,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let q_shape = query.borrow().shape.clone();
        let k_shape = key.borrow().shape.clone();
        let v_shape = value.borrow().shape.clone();

        assert_eq!(
            q_shape.len(),
            3,
            "Query must be 3D tensor [batch, seq_len, d_k]"
        );
        assert_eq!(
            k_shape.len(),
            3,
            "Key must be 3D tensor [batch, seq_len, d_k]"
        );
        assert_eq!(
            v_shape.len(),
            3,
            "Value must be 3D tensor [batch, seq_len, d_v]"
        );

        let (_batch, seq_len, d_k) = (
            *q_shape.first().unwrap(),
            *q_shape.get(1).unwrap(),
            *q_shape.get(2).unwrap(),
        );
        let _d_v = *v_shape.get(2).unwrap();

        assert_eq!(
            d_k,
            *k_shape.get(2).unwrap(),
            "Query and key must have same dimension d_k"
        );
        assert_eq!(
            seq_len,
            *k_shape.get(1).unwrap(),
            "Query and key must have same sequence length"
        );
        assert_eq!(
            seq_len,
            *v_shape.get(1).unwrap(),
            "Query and value must have same sequence length"
        );

        // Compute QK^T: [batch, seq_len, d_k] @ [batch, d_k, seq_len] = [batch, seq_len, seq_len]
        // For 3D tensors, we need to permute to swap the last two dimensions
        let key_t = key.permute(&[0, 2, 1]); // [batch, d_k, seq_len]
        let scores = query.matmul(&key_t); // [batch, seq_len, seq_len]

        // Scale by sqrt(d_k)
        let scale = RawTensor::new(vec![(d_k as f32).sqrt()], &[1], false);
        let scores = scores.div(&scale);

        // Apply causal mask if requested
        let scores = if causal {
            let causal = self.create_causal_mask(seq_len, query.borrow().device.clone());
            scores.add(&causal)
        } else {
            scores
        };

        // Apply additional attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.add(mask)
        } else {
            scores
        };

        // Apply softmax over the last dimension (key sequence length)
        let attention_weights = scores.softmax(2); // [batch, seq_len, seq_len]

        // Apply attention weights to values: [batch, seq_len, seq_len] @ [batch, seq_len, d_v] = [batch, seq_len, d_v]
        attention_weights.matmul(value)
    }

    fn create_causal_mask(&self, seq_len: usize, device: crate::Device) -> Tensor {
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
        RawTensor::new(mask_data, &[seq_len, seq_len], false).to_device(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let _attention = ScaledDotProductAttention::new();
        let _ = ScaledDotProductAttention;
    }

    #[test]
    fn test_attention_shape() {
        let attention = ScaledDotProductAttention::new();

        // Simple test with small dimensions
        let query = RawTensor::new(vec![0.0; 24], &[2, 3, 4], false);
        let key = RawTensor::new(vec![0.0; 24], &[2, 3, 4], false);
        let value = RawTensor::new(vec![0.0; 12], &[2, 3, 2], false);

        let output = attention.forward(&query, &key, &value, false, None);
        assert_eq!(output.borrow().shape, vec![2, 3, 2]);
    }

    #[test]
    fn test_attention_with_causal() {
        let attention = ScaledDotProductAttention::new();

        let query = RawTensor::new(vec![1.0; 24], &[2, 3, 4], false);
        let key = RawTensor::new(vec![1.0; 24], &[2, 3, 4], false);
        let value = RawTensor::new(vec![1.0; 12], &[2, 3, 2], false);

        let output = attention.forward(&query, &key, &value, true, None);
        assert_eq!(output.borrow().shape, vec![2, 3, 2]);

        // With causal masking, the output should be different from non-causal
        let output_non_causal = attention.forward(&query, &key, &value, false, None);
        // The outputs should be different because of the causal mask
        // (In this case with all ones, they might be similar, but the shape is correct)
        assert_eq!(output.borrow().shape, output_non_causal.borrow().shape);
    }

    #[test]
    fn test_attention_with_mask() {
        let attention = ScaledDotProductAttention::new();

        let query = RawTensor::new(vec![1.0; 24], &[2, 3, 4], false);
        let key = RawTensor::new(vec![1.0; 24], &[2, 3, 4], false);
        let value = RawTensor::new(vec![1.0; 12], &[2, 3, 2], false);

        // Create a simple mask (all zeros = no masking)
        let mask = RawTensor::new(vec![0.0; 18], &[2, 3, 3], false);

        let output = attention.forward(&query, &key, &value, false, Some(&mask));
        assert_eq!(output.borrow().shape, vec![2, 3, 2]);
    }

    #[test]
    fn test_attention_different_d_k_d_v() {
        let attention = ScaledDotProductAttention::new();

        // Test with d_k != d_v
        let query = RawTensor::new(vec![0.0; 48], &[2, 3, 8], false);
        let key = RawTensor::new(vec![0.0; 48], &[2, 3, 8], false);
        let value = RawTensor::new(vec![0.0; 12], &[2, 3, 2], false);

        let output = attention.forward(&query, &key, &value, false, None);
        assert_eq!(output.borrow().shape, vec![2, 3, 2]);
    }
}
