use crate::io::StateDict;
use crate::nn::{FeedForward, Layer, LayerNorm, Module, MultiHeadAttention};
use crate::tensor::Tensor;

/// Transformer Block (Pre-LN architecture)
///
/// The transformer block consists of:
/// 1. Layer Normalization
/// 2. Multi-Head Attention
/// 3. Residual Connection
/// 4. Layer Normalization
/// 5. Feed-Forward Network
/// 6. Residual Connection
///
/// This is the Pre-LN architecture used in GPT-2 and GPT-3:
/// LN → Attention → Add → LN → FFN → Add
///
/// # Arguments
/// * `embed_dim` - Embedding dimension
/// * `num_heads` - Number of attention heads
/// * `expansion_factor` - FFN expansion factor (default 4)
/// * `dropout` - Dropout probability (default 0.1)
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::TransformerBlock;
/// let block = TransformerBlock::new(512, 8, 4, 0.1);
/// ```
pub struct TransformerBlock {
    // First layer norm (before attention)
    norm1: LayerNorm,
    // Multi-head attention
    attention: MultiHeadAttention,
    // Second layer norm (before FFN)
    norm2: LayerNorm,
    // Feed-forward network
    ffn: FeedForward,
    // Dropout probability
    #[allow(dead_code)]
    dropout: f32,
    // Configuration
    #[allow(dead_code)]
    embed_dim: usize,
    #[allow(dead_code)]
    num_heads: usize,
    #[allow(dead_code)]
    expansion_factor: usize,
}

impl TransformerBlock {
    /// Create a new transformer block
    ///
    /// # Arguments
    /// * `embed_dim` - Embedding dimension
    /// * `num_heads` - Number of attention heads
    /// * `expansion_factor` - FFN expansion factor (default 4)
    /// * `dropout` - Dropout probability (default 0.1)
    #[must_use]
    pub fn new(embed_dim: usize, num_heads: usize, expansion_factor: usize, dropout: f32) -> Self {
        Self {
            norm1: LayerNorm::new(vec![embed_dim]),
            // Note: We'll use a simple attention wrapper for now
            // In a real implementation, this would be MultiHeadAttention
            attention: MultiHeadAttention::new(embed_dim, num_heads, dropout),
            norm2: LayerNorm::new(vec![embed_dim]),
            ffn: FeedForward::new(embed_dim, expansion_factor, dropout),
            dropout,
            embed_dim,
            num_heads,
            expansion_factor,
        }
    }

    /// Forward pass for the transformer block
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, `seq_len`, `embed_dim`]
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Returns
    /// Output tensor of shape [batch, `seq_len`, `embed_dim`]
    #[must_use]
    pub fn forward(&self, x: &Tensor, causal: bool) -> Tensor {
        // Pre-LN Architecture:
        // 1. LN → Attention → Add
        let x_norm1 = self.norm1.forward(x);
        let attn_out = self.attention.forward(&x_norm1, causal, None);
        let x = x.add(&attn_out);

        // 2. LN → FFN → Add
        let x_norm2 = self.norm2.forward(&x);
        let ffn_out = self.ffn.forward(&x_norm2);
        x.add(&ffn_out)
    }

    /// Get the feed-forward network
    #[must_use]
    pub const fn ffn(&self) -> &FeedForward {
        &self.ffn
    }
}

impl Module for TransformerBlock {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];

        // Layer norm parameters
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());

        // FFN parameters
        params.extend(self.ffn.parameters());

        // Attention parameters
        params.extend(self.attention.parameters());

        params
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();

        // Layer norms
        let norm1_state = self.norm1.state_dict();
        for (key, value) in norm1_state {
            state.insert(format!("norm1.{key}"), value);
        }
        let norm2_state = self.norm2.state_dict();
        for (key, value) in norm2_state {
            state.insert(format!("norm2.{key}"), value);
        }

        // FFN
        let ffn_state = self.ffn.state_dict();
        for (key, value) in ffn_state {
            state.insert(format!("ffn.{key}"), value);
        }

        // Attention
        let attn_state = self.attention.state_dict();
        for (key, value) in attn_state {
            state.insert(format!("attention.{key}"), value);
        }

        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        // Load layer norm 1
        let mut norm1_state = StateDict::new();
        for (key, value) in state {
            if let Some(sub_key) = key.strip_prefix("norm1.") {
                norm1_state.insert(sub_key.to_string(), value.clone());
            }
        }
        if !norm1_state.is_empty() {
            self.norm1.load_state_dict(&norm1_state);
        }

        // Load layer norm 2
        let mut norm2_state = StateDict::new();
        for (key, value) in state {
            if let Some(sub_key) = key.strip_prefix("norm2.") {
                norm2_state.insert(sub_key.to_string(), value.clone());
            }
        }
        if !norm2_state.is_empty() {
            self.norm2.load_state_dict(&norm2_state);
        }

        // Load FFN
        let mut ffn_state = StateDict::new();
        for (key, value) in state {
            if let Some(sub_key) = key.strip_prefix("ffn.") {
                ffn_state.insert(sub_key.to_string(), value.clone());
            }
        }
        if !ffn_state.is_empty() {
            self.ffn.load_state_dict(&ffn_state);
        }

        // Load attention
        let mut attn_state = StateDict::new();
        for (key, value) in state {
            if let Some(sub_key) = key.strip_prefix("attention.") {
                attn_state.insert(sub_key.to_string(), value.clone());
            }
        }
        if !attn_state.is_empty() {
            self.attention.load_state_dict(&attn_state);
        }
    }

    fn train(&mut self, mode: bool) {
        self.norm1.train(mode);
        self.norm2.train(mode);
        self.ffn.train(mode);
        self.attention.train(mode);
    }
}

impl Layer for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;

    #[test]
    fn test_block_creation() {
        let block = TransformerBlock::new(512, 8, 4, 0.1);
        assert_eq!(block.embed_dim, 512);
        assert_eq!(block.num_heads, 8);
        assert_eq!(block.expansion_factor, 4);
    }

    #[test]
    fn test_block_forward_shape() {
        let block = TransformerBlock::new(128, 4, 4, 0.0);

        // Input: [batch=2, seq_len=10, embed_dim=128]
        let x = RawTensor::new(vec![0.0; 2560], &[2, 10, 128], false);
        let output = block.forward(&x, false);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_block_forward_with_causal() {
        let block = TransformerBlock::new(128, 4, 4, 0.0);

        let x = RawTensor::new(vec![1.0; 2560], &[2, 10, 128], false);
        let output = block.forward(&x, true);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_block_parameters() {
        let block = TransformerBlock::new(128, 4, 4, 0.1);
        let params = block.parameters();

        // Should have at least norm parameters (gamma, beta for each of 2 norms)
        assert!(params.len() >= 4);
    }

    #[test]
    fn test_block_state_dict() {
        let block = TransformerBlock::new(128, 4, 4, 0.1);
        let state = block.state_dict();

        // Should have norm1.gamma, norm1.beta, norm2.gamma, norm2.beta, ffn parameters
        assert!(state.len() >= 4);
    }

    #[test]
    fn test_block_assert_trainable() {
        // SimpleAttention (the current placeholder attention) has no parameters of
        // its own, so this exercises norm1/norm2/ffn — everything that does.
        let block = TransformerBlock::new(16, 2, 2, 0.0);
        let x = crate::macros::trainable_input(&[2, 3, 16]);
        crate::assert_trainable!(block, x);
    }

    #[test]
    fn test_block_state_dict_includes_attention() {
        let block = TransformerBlock::new(16, 2, 2, 0.0);
        let state = block.state_dict();
        assert!(
            state.keys().any(|k| k.starts_with("attention.q_proj.")),
            "attention keys missing: {:?}",
            state.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_block_attention_state_dict_roundtrip() {
        let src = TransformerBlock::new(16, 2, 2, 0.0);
        let mut dst = TransformerBlock::new(16, 2, 2, 0.0);
        let x = crate::macros::trainable_input(&[2, 3, 16]);

        let before = dst.forward(&x, false).data();
        dst.load_state_dict(&src.state_dict());
        let after = dst.forward(&x, false).data();
        let expected = src.forward(&x, false).data();

        assert_eq!(after.len(), expected.len());
        assert_ne!(before, after, "loading should change the output");
        for (a, e) in after.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "loaded block differs: {a} vs {e}");
        }
    }

    #[test]
    fn test_block_causal_changes_output() {
        let block = TransformerBlock::new(16, 2, 2, 0.0);
        let x = crate::macros::trainable_input(&[1, 4, 16]);

        let open = block.forward(&x, false).data();
        let causal = block.forward(&x, true).data();

        assert_eq!(open.len(), causal.len());
        assert!(
            open.iter()
                .zip(causal.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "causal masking had no effect, so attention is not being applied"
        );
    }
}
