use crate::io::StateDict;
use crate::nn::{FeedForward, LayerNorm, Module};
use crate::tensor::Tensor;
use crate::tensor::TensorOps;
use std::rc::Rc;

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
    attention: Rc<dyn Module>, // Using Rc to avoid circular dependency with MHA
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
            attention: Rc::new(SimpleAttention::new(embed_dim, num_heads)),
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
    pub fn forward(&self, x: &Tensor, _causal: bool) -> Tensor {
        // Pre-LN Architecture:
        // 1. LN → Attention → Add
        let x_norm1 = self.norm1.forward(x);
        let attn_out = self.attention.forward(&x_norm1);
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
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x, false)
    }

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

        // Note: attention uses Rc<dyn Module> and Module::load_state_dict requires &mut self,
        // so we cannot call load_state_dict on the trait object through Rc.
        // When TransformerBlock is upgraded to use MultiHeadAttention directly,
        // attention state loading will work through the concrete type.
    }

    fn train(&mut self, _mode: bool) {
        // Notify sub-layers
    }
}

// Simple attention wrapper for testing
// In production, this would use MultiHeadAttention
struct SimpleAttention {
    #[allow(dead_code)]
    embed_dim: usize,
    #[allow(dead_code)]
    num_heads: usize,
}

impl SimpleAttention {
    fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embed_dim,
            num_heads,
        }
    }
}

impl Module for SimpleAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        // For now, just return the input (identity)
        // This is a placeholder for the actual attention mechanism
        x.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn load_state_dict(&mut self, _state: &StateDict) {
        // No parameters to load
    }

    fn train(&mut self, _mode: bool) {
        // No training mode difference
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
}
