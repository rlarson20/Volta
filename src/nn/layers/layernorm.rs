use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::storage::Storage;
use crate::tensor::{RawTensor, Tensor, TensorOps};

/// Layer Normalization layer.
///
/// Unlike `BatchNorm`, `LayerNorm` normalizes over the feature dimension (last axis)
/// rather than the batch dimension. It computes statistics per-sample, per-forward pass
/// (no running statistics).
///
/// Commonly used in Transformer architectures (BERT, GPT, etc.) and RNNs.
///
/// # Arguments
/// * `normalized_shape` - Shape of the dimensions to normalize over (must match last N dims of input)
/// * `eps` - Small value added to variance for numerical stability (default 1e-5)
///
/// # Example
/// ```no_run
/// # use volta::LayerNorm;
/// // For 2D input (B, C): normalize over C
/// let ln = LayerNorm::new(vec![64]);
///
/// // For 3D input (B, T, C): normalize over C
/// let ln = LayerNorm::new(vec![128]);
///
/// // For 4D input (B, H, W, C): normalize over H, W, C
/// let ln = LayerNorm::new(vec![32, 32, 3]);
/// ```
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f32,
    // Learnable parameters
    gamma: Tensor, // Scale parameter (initialized to 1)
    beta: Tensor,  // Shift parameter (initialized to 0)
}

impl LayerNorm {
    /// Create a new `LayerNorm` layer with default epsilon (1e-5)
    #[must_use]
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self::new_with_eps(normalized_shape, 1e-5)
    }

    /// Create a new `LayerNorm` layer with specified epsilon
    #[must_use]
    pub fn new_with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        // Calculate total size of normalized dimensions
        let num_features: usize = normalized_shape.iter().product();

        // Initialize gamma to 1, beta to 0
        let gamma = RawTensor::ones(&[num_features]);
        gamma.borrow_mut().requires_grad = true;

        let beta = RawTensor::zeros(&[num_features]);
        beta.borrow_mut().requires_grad = true;

        Self {
            normalized_shape,
            eps,
            gamma,
            beta,
        }
    }

    /// Compute the dimension index to normalize over
    ///
    /// For input shape [B, ...] where `normalized_shape` has N elements,
    /// we normalize over the last N dimensions.
    fn normalize_dim(&self, input_rank: usize) -> usize {
        input_rank - self.normalized_shape.len()
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = x.borrow().shape.clone();
        let input_rank = input_shape.len();

        // Validate that last N dimensions match normalized_shape
        let norm_rank = self.normalized_shape.len();
        assert!(
            input_rank >= norm_rank,
            "LayerNorm: input rank ({input_rank}) must be >= normalized_shape rank ({norm_rank})"
        );

        for (i, &expected_size) in self.normalized_shape.iter().enumerate() {
            let actual_size = *input_shape.get(input_rank - norm_rank + i).unwrap_or(&1);
            assert_eq!(
                actual_size,
                expected_size,
                "LayerNorm: dimension {} mismatch: expected {}, got {}",
                input_rank - norm_rank + i,
                expected_size,
                actual_size
            );
        }

        // Compute mean and variance over the last N dimensions
        let dim = self.normalize_dim(input_rank);

        // Reshape to compute stats more efficiently:
        // - Flatten leading dimensions: (B, T, ..., C) -> (B*T*..., C)
        // - Then compute mean/var over the last dimension
        let leading_dims: usize = input_shape.get(..dim).map_or(1, |s| s.iter().product());
        let trailing_dims: usize = input_shape.get(dim..).map_or(1, |s| s.iter().product());

        // Reshape to (leading_dims, trailing_dims)
        let x_reshaped = x.reshape(&[leading_dims, trailing_dims]);

        // Compute mean over dim=1 (the feature dimension)
        let mean = x_reshaped.mean_dim(1, true); // (leading_dims, 1)

        // Compute variance: E[(x - mean)^2]
        let diff = x_reshaped.sub(&mean);
        let sq_diff = diff.elem_mul(&diff);
        let var = sq_diff.mean_dim(1, true); // (leading_dims, 1)

        // Normalize: (x - mean) / sqrt(var + eps)
        let eps_tensor = RawTensor::constant(self.eps, &[1]);
        let denom = var.add(&eps_tensor).sqrt();
        let x_norm = diff.div(&denom); // (leading_dims, trailing_dims)

        // Reshape gamma and beta to broadcast correctly
        let gamma_reshaped = self.gamma.reshape(&[1, trailing_dims]);
        let beta_reshaped = self.beta.reshape(&[1, trailing_dims]);

        // Scale and shift: gamma * x_norm + beta
        let out = x_norm.elem_mul(&gamma_reshaped).add(&beta_reshaped);

        // Reshape back to original shape
        out.reshape(&input_shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert("gamma".to_string(), TensorData::from_tensor(&self.gamma));
        state.insert("beta".to_string(), TensorData::from_tensor(&self.beta));
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        let params = [("gamma", &self.gamma), ("beta", &self.beta)];

        for (key, tensor) in params {
            if let Some(t) = state.get(key) {
                let mut b = tensor.borrow_mut();
                b.data = Storage::cpu(t.data.clone());
                b.shape.clone_from(&t.shape);
            }
        }
    }

    fn train(&mut self, _mode: bool) {
        // LayerNorm has the same behavior in train and eval mode
        // (no running statistics, unlike BatchNorm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_creation() {
        let ln = LayerNorm::new(vec![64]);
        assert_eq!(ln.normalized_shape, vec![64]);
        assert_eq!(ln.gamma.borrow().shape, vec![64]);
        assert_eq!(ln.beta.borrow().shape, vec![64]);
    }

    #[test]
    fn test_layernorm_with_eps() {
        let ln = LayerNorm::new_with_eps(vec![32], 1e-6);
        assert_eq!(ln.eps, 1e-6);
    }
}
