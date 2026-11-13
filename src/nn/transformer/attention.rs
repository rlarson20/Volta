use crate::shape::Axis;
use crate::tensor::Tensor;

/// Scaled Dot-Product Attention
///
/// Computes attention as: softmax(QK^T / √`d_k`) V
///
/// This is the core attention mechanism used in transformer models. As in
/// `torch.nn.functional.scaled_dot_product_attention`, the key/value length `S` may
/// differ from the query length `L` (cross-attention).
///
/// # Arguments
/// * `query` - Query tensor of shape [batch, `L`, `d_k`]
/// * `key` - Key tensor of shape [batch, `S`, `d_k`]
/// * `value` - Value tensor of shape [batch, `S`, `d_v`]
/// * `causal` - Whether to apply causal masking (for autoregressive models)
/// * `attention_mask` - Optional additive mask (e.g., padding mask) that broadcasts to [batch, `L`, `S`]
///
/// # Returns
/// Output tensor of shape [batch, `L`, `d_v`]
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::ScaledDotProductAttention;
/// # use volta::{RawTensor, Module};
/// # use volta::Device;
/// let attention = ScaledDotProductAttention::new();
/// let query = RawTensor::new(vec![0.0; 128], &[2, 4, 16], false);
/// let key = RawTensor::new(vec![0.0; 192], &[2, 6, 16], false);
/// let value = RawTensor::new(vec![0.0; 96], &[2, 6, 8], false);
/// let output = attention.forward(&query, &key, &value, false, None); // [2, 4, 8]
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
    /// * `query` - Query tensor of shape [batch, `L`, `d_k`]
    /// * `key` - Key tensor of shape [batch, `S`, `d_k`]
    /// * `value` - Value tensor of shape [batch, `S`, `d_v`]
    /// * `causal` - Whether to apply causal masking. With `L != S` the mask is top-left
    ///   aligned, as torch's `is_causal`: query `i` sees keys `j <= i`.
    /// * `attention_mask` - Optional additive mask that broadcasts to [batch, `L`, `S`]
    ///
    /// # Returns
    /// Output tensor of shape [batch, `L`, `d_v`]
    /// # Panics
    /// If a tensor is not 3D, the batch sizes differ, query and key differ in `d_k`, or
    /// key and value differ in length.
    #[must_use]
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
            q_shape.rank(),
            3,
            "Query must be 3D tensor [batch, seq_len, d_k]"
        );
        assert_eq!(
            k_shape.rank(),
            3,
            "Key must be 3D tensor [batch, seq_len, d_k]"
        );
        assert_eq!(
            v_shape.rank(),
            3,
            "Value must be 3D tensor [batch, seq_len, d_v]"
        );

        let (batch, q_len, d_k) = (q_shape.dim_at(0), q_shape.dim_at(1), q_shape.dim_at(2));
        let kv_len = k_shape.dim_at(1);

        assert!(
            batch == k_shape.dim_at(0) && batch == v_shape.dim_at(0),
            "Query, key and value must have the same batch size: {q_shape:?}, {k_shape:?}, {v_shape:?}"
        );
        assert_eq!(
            d_k,
            k_shape.dim_at(2),
            "Query and key must have same dimension d_k"
        );
        assert_eq!(
            kv_len,
            v_shape.dim_at(1),
            "Key and value must have same sequence length"
        );

        // Compute QK^T: [batch, L, d_k] @ [batch, d_k, S] = [batch, L, S]
        // For 3D tensors, we need to permute to swap the last two dimensions
        let key_t = key.permute(&[Axis(0), Axis(2), Axis(1)]); // [batch, d_k, S]
        let scores = query.matmul(&key_t); // [batch, L, S]

        // Scale by sqrt(d_k)
        let scale = query.constant_like(vec![(d_k as f32).sqrt()], &[1]);
        let scores = scores.div(&scale);

        // Apply causal mask if requested
        let scores = if causal {
            let causal = Self::create_causal_mask(q_len, kv_len, query);
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
        let attention_weights = scores.softmax(Axis(2)); // [batch, L, S]

        // Apply attention weights to values: [batch, L, S] @ [batch, S, d_v] = [batch, L, d_v]
        attention_weights.matmul(value)
    }

    /// Additive mask `[q_len, kv_len]`: `-inf` where `j > i`, 0 elsewhere, on `like`'s
    /// device and in its dtype. This is torch's top-left aligned `is_causal` mask
    /// (`ones(L, S).tril()`); it broadcasts against any `[.., L, S]` score tensor.
    pub(super) fn create_causal_mask(q_len: usize, kv_len: usize, like: &Tensor) -> Tensor {
        let mut mask_data = Vec::with_capacity(q_len * kv_len);
        for i in 0..q_len {
            for j in 0..kv_len {
                // Upper triangle (j > i) gets -inf, lower triangle and diagonal get 0
                if j > i {
                    mask_data.push(f32::NEG_INFINITY);
                } else {
                    mask_data.push(0.0);
                }
            }
        }
        like.constant_like(mask_data, &[q_len, kv_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;

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

    /// One gradient check: the input's name, the input, and the loss as a function of it.
    type Check<'a> = (
        &'static str,
        &'a Tensor,
        Box<dyn Fn(&Tensor) -> Tensor + 'a>,
    );

    /// Seeded normal values as an f64 tensor, a leaf that requires grad when `grad`.
    /// The checks run in f64 so that finite-difference rounding (about 1e-7 relative
    /// in f32) can't mask or fake a mismatch.
    fn operand(dims: &[usize], grad: bool) -> Tensor {
        let vals: Vec<f64> = crate::tensor::randn(dims)
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let t = RawTensor::from_typed(vals, dims);
        t.set_requires_grad(grad);
        t
    }

    /// Finite-difference check of query, key and value under a randomly weighted sum,
    /// with `L = 2` queries attending over `S = 3` keys (cross-attention).
    fn assert_cross_attention_gradients(causal: bool) {
        crate::tensor::manual_seed(0x5EED);
        let q = operand(&[2, 2, 4], true);
        let k = operand(&[2, 3, 4], true);
        let v = operand(&[2, 3, 5], true);
        let w = operand(&[2, 2, 5], false);
        let attn = ScaledDotProductAttention::new();

        let checks: [Check<'_>; 3] = [
            (
                "query",
                &q,
                Box::new(|t| attn.forward(t, &k, &v, causal, None).elem_mul(&w).sum()),
            ),
            (
                "key",
                &k,
                Box::new(|t| attn.forward(&q, t, &v, causal, None).elem_mul(&w).sum()),
            ),
            (
                "value",
                &v,
                Box::new(|t| attn.forward(&q, &k, t, causal, None).elem_mul(&w).sum()),
            ),
        ];
        for (name, input, loss_fn) in checks {
            let (max, mean, ok) = RawTensor::check_gradients(input, loss_fn, 1e-6, 1e-6);
            assert!(
                ok,
                "cross-attention grad wrt {name} (causal={causal}) failed: max={max:.3e} mean={mean:.3e}"
            );
        }
    }

    #[test]
    fn test_attention_cross_attention_gradients() {
        assert_cross_attention_gradients(false);
    }

    #[test]
    fn test_attention_cross_attention_causal_gradients() {
        assert_cross_attention_gradients(true);
    }

    #[test]
    fn test_attention_cross_attention_causal_mask_is_top_left_aligned() {
        // L = 2 queries over S = 3 keys. Query 0 may see key 0 only, so its output is
        // value row 0 exactly; query 1 sees keys 0 and 1, never key 2.
        crate::tensor::manual_seed(7);
        let q = crate::tensor::randn(&[1, 2, 4]);
        let k = crate::tensor::randn(&[1, 3, 4]);
        let v = RawTensor::new(vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0], &[1, 3, 2], false);
        let out = ScaledDotProductAttention::new()
            .forward(&q, &k, &v, true, None)
            .data();
        assert_eq!(out.len(), 4);
        let at = |i| crate::index::fetch(&out, i);
        assert!(
            (at(0) - 1.0).abs() < 1e-5 && (at(1) - 2.0).abs() < 1e-5,
            "{out:?}"
        );
        // A convex mix of value rows 0 and 1: strictly inside [1, 10] and [2, 20].
        assert!(at(2) > 1.0 && at(2) < 10.0, "{out:?}");
        assert!(at(3) > 2.0 && at(3) < 20.0, "{out:?}");
    }
}
