use crate::io::StateDict;
use crate::nn::{LayerNorm, Linear, Module};
use crate::tensor::TensorOps;
use crate::tensor::{RawTensor, Tensor};

/// Multi-Head Attention layer
///
/// Splits the input into multiple heads, applies scaled dot-product attention to each,
/// and concatenates the results. Used in transformer architectures.
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
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    // Q, K, V projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    // Output projection
    out_proj: Linear,
    // Optional layer norm
    norm: Option<LayerNorm>,
    // Dropout
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
            norm: None,
            dropout,
        }
    }

    /// Create a new multi-head attention layer with layer normalization
    /// # Panics
    /// `embed_dim` must be divisible by `num_heads`
    #[must_use]
    pub fn new_with_norm(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
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
            norm: Some(LayerNorm::new(vec![embed_dim])),
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
    /// Output tensor of shape [batch, `seq_len`, `embed_dim`]
    /// # Panics
    /// unwraps for batch size etc
    pub fn forward(&self, x: &Tensor, causal: bool, attention_mask: Option<&Tensor>) -> Tensor {
        let input_shape = x.borrow().shape.clone();
        let (batch_size, seq_len, _embed_dim) = (
            *input_shape.first().unwrap(),
            *input_shape.get(1).unwrap(),
            *input_shape.get(2).unwrap(),
        );

        // Apply layer norm if present
        let x = if let Some(ref norm) = self.norm {
            norm.forward(x)
        } else {
            x.clone()
        };

        // Reshape to 2D for Linear layers: [batch, seq_len, embed_dim] -> [batch * seq_len, embed_dim]
        let x_2d = x.reshape(&[batch_size * seq_len, _embed_dim]);

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
        let output = output_2d.reshape(&[batch_size, seq_len, self.embed_dim]);

        // Add residual connection
        x.add(&output)
    }

    fn reshape_for_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
        let reshaped = x.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);

        // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        reshaped.permute(&[0, 2, 1, 3])
    }

    fn reshape_from_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        let permuted = x.permute(&[0, 2, 1, 3]);

        // [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, embed_dim]
        permuted.reshape(&[batch_size, seq_len, self.embed_dim])
    }

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
        let seq_len = *q_shape.get(2).unwrap(); // [batch, num_heads, seq_len, head_dim]

        // Compute scaled dot-product attention
        // QK^T: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        //      = [batch, num_heads, seq_len, seq_len]
        let k_t = k.permute(&[0, 1, 3, 2]); // [batch, num_heads, head_dim, seq_len]
        let scores = q.matmul(&k_t); // [batch, num_heads, seq_len, seq_len]

        // Scale by sqrt(head_dim)
        let scale = RawTensor::new(vec![(self.head_dim as f32).sqrt()], &[1, 1, 1, 1], false);
        let scores = scores.div(&scale);

        // Apply causal mask if requested
        let scores = if causal {
            let causal_mask = self.create_causal_mask(seq_len, q.borrow().device.clone());
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
        let attention_weights = scores.softmax(3); // [batch, num_heads, seq_len, seq_len]

        // Apply attention weights to values
        attention_weights.matmul(v) // [batch, num_heads, seq_len, head_dim]
    }

    fn create_causal_mask(&self, seq_len: usize, device: crate::Device) -> Tensor {
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
        let mask = RawTensor::new(mask_data, &[1, 1, seq_len, seq_len], false);
        mask.to_device(device)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x, false, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];

        // Get parameters from each Linear layer
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());

        // Include norm parameters if present
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }

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

        // Include norm parameters if present
        if let Some(ref norm) = self.norm {
            let norm_state = norm.state_dict();
            for (key, value) in norm_state {
                state.insert(format!("norm.{key}"), value);
            }
        }

        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
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
                layer.load_state_dict(&sub_state);
            }
        }

        // Load norm parameters if present
        if let Some(ref mut norm) = self.norm {
            let mut norm_state = StateDict::new();
            for (key, value) in state {
                if let Some(norm_key) = key.strip_prefix("norm.") {
                    norm_state.insert(norm_key.to_string(), value.clone());
                }
            }
            if !norm_state.is_empty() {
                norm.load_state_dict(&norm_state);
            }
        }
    }

    fn train(&mut self, _mode: bool) {
        // Notify Linear layers if they have dropout behavior
        // (currently Linear doesn't have dropout, but this is for future compatibility)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mha_creation() {
        let mha = MultiHeadAttention::new(512, 8, 0.1);
        assert_eq!(mha.embed_dim, 512);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.head_dim, 64);
    }

    #[test]
    fn test_mha_creation_with_norm() {
        let mha = MultiHeadAttention::new_with_norm(256, 4, 0.1);
        assert_eq!(mha.embed_dim, 256);
        assert_eq!(mha.num_heads, 4);
        assert_eq!(mha.head_dim, 64);
        assert!(mha.norm.is_some());
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
}
