use crate::io::StateDict;
use crate::nn::{Layer, Linear, Module};
use crate::shape::Axis;
use crate::tensor::Tensor;

/// Multi-Head Attention layer
///
/// Splits the input into multiple heads, applies scaled dot-product attention to each,
/// and concatenates the results. Used in transformer architectures.
///
/// Like `torch.nn.MultiheadAttention`, the layer returns only the projected attention
/// output, `out_proj(Attn(x))`: it applies no layer norm and adds no residual. A
/// transformer block adds `x + Attn(LN(x))` itself (see `TransformerBlock`).
///
/// # Arguments
/// * `embed_dim` - Total dimension of the model
/// * `num_heads` - Number of attention heads
/// * `dropout` - Dropout probability (0.0 = no dropout)
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::MultiHeadAttention;
/// let mha = MultiHeadAttention::new(512, 8, 0.1);
/// ```
pub struct MultiHeadAttention {
    /// Model width.
    embed_dim: usize,
    /// Number of attention heads.
    num_heads: usize,
    /// Width of one head, `embed_dim / num_heads`.
    head_dim: usize,
    /// Query projection.
    q_proj: Linear,
    /// Key projection.
    k_proj: Linear,
    /// Value projection.
    v_proj: Linear,
    /// Output projection.
    out_proj: Linear,
    /// Dropout probability. Stored but not applied.
    #[allow(dead_code)]
    dropout: f32,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    ///
    /// # Arguments
    /// * `embed_dim` - Total dimension of the model (must be divisible by `num_heads`)
    /// * `num_heads` - Number of attention heads
    /// * `dropout` - Dropout probability (0.0 = no dropout)
    /// # Panics
    /// `embed_dim` must be divisible by `num_heads`
    #[must_use]
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert!(
            embed_dim.is_multiple_of(num_heads),
            "embed_dim must be divisible by num_heads"
        );

        let head_dim = embed_dim / num_heads;

        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim, false),
            k_proj: Linear::new(embed_dim, embed_dim, false),
            v_proj: Linear::new(embed_dim, embed_dim, false),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            dropout,
        }
    }

    /// Forward pass for multi-head attention
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, `seq_len`, `embed_dim`]
    /// * `causal` - Whether to apply causal masking
    /// * `attention_mask` - Optional additional mask
    ///
    /// # Returns
    /// `out_proj(Attn(x))`, of shape [batch, `seq_len`, `embed_dim`], with no residual added.
    /// # Panics
    /// unwraps for batch size etc
    #[must_use]
    pub fn forward(&self, x: &Tensor, causal: bool, attention_mask: Option<&Tensor>) -> Tensor {
        let input_shape = x.borrow().shape.clone();
        let (batch_size, seq_len, _embed_dim) = (
            input_shape.dim_at(0),
            input_shape.dim_at(1),
            input_shape.dim_at(2),
        );

        // Reshape to 2D for Linear layers: [batch, seq_len, embed_dim] -> [batch * seq_len, embed_dim]
        let x_2d = x.reshape(&[batch_size * seq_len, self.embed_dim]);

        // Project Q, K, V through Linear layers (2D)
        let q = self.q_proj.forward(&x_2d); // [batch * seq_len, embed_dim]
        let k = self.k_proj.forward(&x_2d);
        let v = self.v_proj.forward(&x_2d);

        // Reshape back to 3D: [batch * seq_len, embed_dim] -> [batch, seq_len, embed_dim]
        let q = q.reshape(&[batch_size, seq_len, self.embed_dim]);
        let k = k.reshape(&[batch_size, seq_len, self.embed_dim]);
        let v = v.reshape(&[batch_size, seq_len, self.embed_dim]);

        // Reshape for multi-head attention
        // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        let q = self.reshape_for_heads(&q, batch_size, seq_len);
        let k = self.reshape_for_heads(&k, batch_size, seq_len);
        let v = self.reshape_for_heads(&v, batch_size, seq_len);

        // Apply scaled dot-product attention
        let attn_output = self.apply_attention(&q, &k, &v, causal, attention_mask);

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        let attn_output = self.reshape_from_heads(&attn_output, batch_size, seq_len);

        // Reshape to 2D for output projection
        let attn_2d = attn_output.reshape(&[batch_size * seq_len, self.embed_dim]);
        let output_2d = self.out_proj.forward(&attn_2d);
        output_2d.reshape(&[batch_size, seq_len, self.embed_dim])
    }

    /// Splits `[batch, seq_len, embed_dim]` into heads: `[batch, num_heads, seq_len, head_dim]`.
    fn reshape_for_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
        let reshaped = x.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);

        // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        reshaped.permute(&[Axis(0), Axis(2), Axis(1), Axis(3)])
    }

    /// Inverse of `reshape_for_heads`.
    fn reshape_from_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        let permuted = x.permute(&[Axis(0), Axis(2), Axis(1), Axis(3)]);

        // [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, embed_dim]
        permuted.reshape(&[batch_size, seq_len, self.embed_dim])
    }

    /// Scaled dot-product attention over `[batch, num_heads, seq_len, head_dim]` tensors, with an optional causal mask and an optional additive mask.
    fn apply_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        // Get sequence length from query tensor shape
        let q_shape = q.borrow().shape.clone();
        let seq_len = q_shape.dim_at(2); // [batch, num_heads, seq_len, head_dim]

        // Compute scaled dot-product attention
        // QK^T: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        //      = [batch, num_heads, seq_len, seq_len]
        let k_t = k.permute(&[Axis(0), Axis(1), Axis(3), Axis(2)]); // [batch, num_heads, head_dim, seq_len]
        let scores = q.matmul(&k_t); // [batch, num_heads, seq_len, seq_len]

        // Scale by sqrt(head_dim)
        let scale = q.constant_like(vec![(self.head_dim as f32).sqrt()], &[1, 1, 1, 1]);
        let scores = scores.div(&scale);

        // Apply causal mask if requested
        let scores = if causal {
            let causal_mask = Self::create_causal_mask(seq_len, q);
            scores.add(&causal_mask)
        } else {
            scores
        };

        // Apply additional attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.add(mask)
        } else {
            scores
        };

        // Apply softmax over the last dimension
        let attention_weights = scores.softmax(Axis(3)); // [batch, num_heads, seq_len, seq_len]

        // Apply attention weights to values
        attention_weights.matmul(v) // [batch, num_heads, seq_len, head_dim]
    }

    /// Additive mask `[seq_len, seq_len]`: `-inf` above the diagonal, 0 elsewhere, on
    /// `like`'s device and in its dtype.
    fn create_causal_mask(seq_len: usize, like: &Tensor) -> Tensor {
        let mut mask_data = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data.push(f32::NEG_INFINITY);
                } else {
                    mask_data.push(0.0);
                }
            }
        }
        like.constant_like(mask_data, &[1, 1, seq_len, seq_len])
    }
}

impl Module for MultiHeadAttention {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];

        // Get parameters from each Linear layer
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());

        params
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();

        // Get state from each Linear layer
        let q_proj_state = self.q_proj.state_dict();
        for (key, value) in q_proj_state {
            state.insert(format!("q_proj.{key}"), value);
        }

        let k_proj_state = self.k_proj.state_dict();
        for (key, value) in k_proj_state {
            state.insert(format!("k_proj.{key}"), value);
        }

        let v_proj_state = self.v_proj.state_dict();
        for (key, value) in v_proj_state {
            state.insert(format!("v_proj.{key}"), value);
        }

        let out_proj_state = self.out_proj.state_dict();
        for (key, value) in out_proj_state {
            state.insert(format!("out_proj.{key}"), value);
        }

        state
    }

    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
        // Load each projection's weights by extracting prefixed sub-dicts
        let prefixes = [
            ("q_proj.", &mut self.q_proj as &mut Linear),
            ("k_proj.", &mut self.k_proj),
            ("v_proj.", &mut self.v_proj),
            ("out_proj.", &mut self.out_proj),
        ];

        for (prefix, layer) in prefixes {
            let mut sub_state = StateDict::new();
            for (key, value) in state {
                if let Some(sub_key) = key.strip_prefix(prefix) {
                    sub_state.insert(sub_key.to_string(), value.clone());
                }
            }
            if !sub_state.is_empty() {
                layer
                    .load_state_dict(&sub_state)
                    .map_err(|e| e.with_key_prefix(prefix))?;
            }
        }

        Ok(())
    }

    fn train(&mut self, _mode: bool) {
        // Notify Linear layers if they have dropout behavior
        // (currently Linear doesn't have dropout, but this is for future compatibility)
    }
}

impl Layer for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x, false, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;

    #[test]
    fn test_mha_creation() {
        let mha = MultiHeadAttention::new(512, 8, 0.1);
        assert_eq!(mha.embed_dim, 512);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.head_dim, 64);
    }

    #[test]
    fn test_mha_forward_shape() {
        let mha = MultiHeadAttention::new(128, 4, 0.0);

        // Input: [batch=2, seq_len=10, embed_dim=128]
        let x = RawTensor::new(vec![0.0; 2560], &[2, 10, 128], false);
        let output = mha.forward(&x, false, None);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_mha_forward_with_causal() {
        let mha = MultiHeadAttention::new(128, 4, 0.0);

        let x = RawTensor::new(vec![1.0; 2560], &[2, 10, 128], false);
        let output = mha.forward(&x, true, None);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_mha_assert_trainable() {
        let mha = MultiHeadAttention::new(16, 2, 0.0);
        let x = crate::macros::trainable_input(&[2, 3, 16]);
        crate::assert_trainable!(mha, x);
    }
}
