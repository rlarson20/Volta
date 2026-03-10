use crate::io::StateDict;
use crate::nn::{GELU, Linear, Module};
use crate::tensor::Tensor;
use crate::tensor::TensorOps;

/// Feed-Forward Network (FFN) layer for transformer blocks
///
/// The FFN consists of two linear transformations with a GELU activation in between:
/// Linear → GELU → Dropout → Linear → Dropout
///
/// The first layer typically expands the dimension by a factor (default 4x),
/// and the second layer projects it back to the original dimension.
///
/// # Arguments
/// * `embed_dim` - Input/output dimension
/// * `expansion_factor` - Factor by which to expand the intermediate dimension (default 4)
/// * `dropout` - Dropout probability (default 0.1)
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::FeedForward;
/// let ffn = FeedForward::new(512, 4, 0.1);
/// ```
pub struct FeedForward {
    expand: Linear,
    contract: Linear,
    activation: GELU,
    #[allow(dead_code)]
    dropout: f32,
    expansion_factor: usize,
}

impl FeedForward {
    /// Create a new feed-forward network layer
    ///
    /// # Arguments
    /// * `embed_dim` - Input/output dimension
    /// * `expansion_factor` - Factor by which to expand the intermediate dimension (default 4)
    /// * `dropout` - Dropout probability (default 0.1)
    #[must_use]
    pub fn new(embed_dim: usize, expansion_factor: usize, dropout: f32) -> Self {
        let intermediate_dim = embed_dim * expansion_factor;

        Self {
            expand: Linear::new(embed_dim, intermediate_dim, true),
            contract: Linear::new(intermediate_dim, embed_dim, true),
            activation: GELU::new(),
            dropout,
            expansion_factor,
        }
    }

    /// Create a new feed-forward network with default settings
    ///
    /// Uses `expansion_factor`=4 and `dropout`=0.1
    #[must_use]
    pub fn default_config(embed_dim: usize) -> Self {
        Self::new(embed_dim, 4, 0.1)
    }

    /// Forward pass for the feed-forward network
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, `seq_len`, `embed_dim`]
    ///
    /// # Returns
    /// Output tensor of shape [batch, `seq_len`, `embed_dim`]
    /// # Panics
    /// reshape can panic
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = x.borrow().shape.clone();

        // Reshape to 2D for Linear layers: [batch, seq_len, embed_dim] -> [batch * seq_len, embed_dim]
        let batch_size = *input_shape.first().unwrap();
        let seq_len = *input_shape.get(1).unwrap();
        let embed_dim = *input_shape.get(2).unwrap();

        let x_2d = x.reshape(&[batch_size * seq_len, embed_dim]);

        // Expand: [batch * seq_len, embed_dim] -> [batch * seq_len, intermediate_dim]
        let x = self.expand.forward(&x_2d);

        // Apply GELU activation
        let x = self.activation.forward(&x);

        // Contract: [batch * seq_len, intermediate_dim] -> [batch * seq_len, embed_dim]
        let x = self.contract.forward(&x);

        // Reshape back to 3D: [batch * seq_len, embed_dim] -> [batch, seq_len, embed_dim]
        x.reshape(&[batch_size, seq_len, embed_dim])
    }

    /// Get the intermediate dimension
    #[must_use]
    pub const fn intermediate_dim(&self) -> usize {
        self.expansion_factor
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        params.extend(self.expand.parameters());
        params.extend(self.contract.parameters());
        params
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();

        // Get expand layer state
        let expand_state = self.expand.state_dict();
        for (key, value) in expand_state {
            state.insert(format!("expand.{key}"), value);
        }

        // Get contract layer state
        let contract_state = self.contract.state_dict();
        for (key, value) in contract_state {
            state.insert(format!("contract.{key}"), value);
        }

        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        // Load expand layer
        let mut expand_state = StateDict::new();
        for (key, value) in state {
            if let Some(sub_key) = key.strip_prefix("expand.") {
                expand_state.insert(sub_key.to_string(), value.clone());
            }
        }
        if !expand_state.is_empty() {
            self.expand.load_state_dict(&expand_state);
        }

        // Load contract layer
        let mut contract_state = StateDict::new();
        for (key, value) in state {
            if let Some(sub_key) = key.strip_prefix("contract.") {
                contract_state.insert(sub_key.to_string(), value.clone());
            }
        }
        if !contract_state.is_empty() {
            self.contract.load_state_dict(&contract_state);
        }
    }

    fn train(&mut self, _mode: bool) {
        // FFN has different behavior in train/eval mode (dropout)
        // For now, this is a no-op since dropout is not fully implemented
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;

    #[test]
    fn test_ffn_creation() {
        let ffn = FeedForward::new(128, 4, 0.1);
        assert_eq!(ffn.expansion_factor, 4);
    }

    #[test]
    fn test_ffn_default_config() {
        let ffn = FeedForward::default_config(256);
        assert_eq!(ffn.expansion_factor, 4);
    }

    #[test]
    fn test_ffn_forward_shape() {
        let ffn = FeedForward::new(128, 4, 0.0);

        // Input: [batch=2, seq_len=10, embed_dim=128]
        let x = RawTensor::new(vec![0.0; 2560], &[2, 10, 128], false);
        let output = ffn.forward(&x);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_ffn_forward_shape_1d() {
        let ffn = FeedForward::new(64, 3, 0.0);

        // Input: [batch=1, seq_len=5, embed_dim=64]
        let x = RawTensor::new(vec![0.0; 320], &[1, 5, 64], false);
        let output = ffn.forward(&x);

        assert_eq!(output.borrow().shape, vec![1, 5, 64]);
    }

    #[test]
    fn test_ffn_parameters() {
        let ffn = FeedForward::new(128, 4, 0.1);
        let params = ffn.parameters();

        // Should have expand (weight + bias) and contract (weight + bias) = 4 parameters total
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_ffn_state_dict() {
        let ffn = FeedForward::new(128, 4, 0.1);
        let state = ffn.state_dict();

        // Should have expand.weight, expand.bias, contract.weight, contract.bias
        assert_eq!(state.len(), 4);
    }
}
