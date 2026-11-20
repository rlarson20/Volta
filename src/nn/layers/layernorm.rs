use crate::error::OrPanic;
use crate::io::{StateDict, TensorData};
use crate::nn::hyper;
use crate::nn::{Layer, Module};
use crate::shape::Axis;
use crate::tensor::{RawTensor, Tensor};

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
/// # use volta::nn::LayerNorm;
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
    /// Trailing dims the statistics are computed over.
    normalized_shape: Vec<usize>,
    /// Added to the variance for numerical stability.
    eps: f32,
    // Learnable parameters
    /// Learnable scale, initialized to 1.
    gamma: Tensor,
    /// Learnable shift, initialized to 0.
    beta: Tensor,
}

impl LayerNorm {
    /// Create a new `LayerNorm` layer with default epsilon (1e-5)
    ///
    /// # Panics
    /// As [`Self::new_with_eps`].
    #[must_use]
    #[track_caller]
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self::new_with_eps(normalized_shape, 1e-5)
    }

    /// [`Self::new`], checking `normalized_shape` first.
    ///
    /// # Errors
    /// As [`Self::try_new_with_eps`].
    pub fn try_new(normalized_shape: Vec<usize>) -> crate::Result<Self> {
        Self::try_new_with_eps(normalized_shape, 1e-5)
    }

    /// A `LayerNorm` over the trailing dimensions `normalized_shape`, with `eps` added to
    /// the variance.
    ///
    /// # Panics
    /// If `normalized_shape` is empty or has a zero, or `eps` is not finite and positive
    /// (see [`Self::try_new_with_eps`]).
    #[must_use]
    #[track_caller]
    pub fn new_with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        Self::try_new_with_eps(normalized_shape, eps).or_panic("LayerNorm")
    }

    /// [`Self::new_with_eps`], checking the arguments first.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` naming the argument if `normalized_shape` is empty
    /// or has a zero dimension, or `eps` is zero, negative or not finite. An empty shape
    /// used to give a layer over one feature, and a negative `eps` ran (#133).
    pub fn try_new_with_eps(normalized_shape: Vec<usize>, eps: f32) -> crate::Result<Self> {
        if normalized_shape.is_empty() {
            return Err(crate::VoltaError::invalid(
                "LayerNorm",
                "normalized_shape",
                "normalized_shape must have at least one dimension, got []",
            ));
        }
        for &d in &normalized_shape {
            hyper::at_least_one("LayerNorm", "normalized_shape", d)?;
        }
        hyper::eps("LayerNorm", eps)?;
        // Calculate total size of normalized dimensions
        let num_features: usize = normalized_shape.iter().product();

        // Initialize gamma to 1, beta to 0
        let gamma = RawTensor::ones(&[num_features]);
        gamma.set_requires_grad(true);

        let beta = RawTensor::zeros(&[num_features]);
        beta.set_requires_grad(true);

        Ok(Self {
            normalized_shape,
            eps,
            gamma,
            beta,
        })
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
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert("gamma".to_string(), TensorData::from_tensor(&self.gamma));
        state.insert("beta".to_string(), TensorData::from_tensor(&self.beta));
        state
    }

    fn plan_load<'a>(&self, plan: &mut crate::nn::LoadPlan<'a>) {
        plan.param("gamma", &self.gamma);
        plan.param("beta", &self.beta);
    }

    fn train(&mut self, _mode: bool) {
        // LayerNorm has the same behavior in train and eval mode
        // (no running statistics, unlike BatchNorm)
    }
}

impl Layer for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        // A half input normalizes in f32 and returns the input's dtype, as PyTorch's
        // norm kernels do (audit A5, #316).
        crate::nn::norm_mixed(x, self.gamma.dtype(), |x, dtype| self.forward_in(x, dtype))
    }
}

impl LayerNorm {
    /// The forward pass with `x` already in `dtype`, the dtype to compute in, which the
    /// parameters are cast to ([`crate::nn::norm_mixed`]).
    fn forward_in(&self, x: &Tensor, dtype: crate::DType) -> Tensor {
        let input_shape = x.borrow().shape.clone();
        let input_rank = input_shape.rank();

        // Validate that last N dimensions match normalized_shape
        let norm_rank = self.normalized_shape.len();
        assert!(
            input_rank >= norm_rank,
            "LayerNorm: input rank ({input_rank}) must be >= normalized_shape rank ({norm_rank})"
        );

        for (i, &expected_size) in self.normalized_shape.iter().enumerate() {
            let actual_size = input_shape.dim_at(input_rank - norm_rank + i);
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
        let leading_dims = input_shape.numel_before(dim);
        let trailing_dims = input_shape.numel_from(dim);

        // Reshape to (leading_dims, trailing_dims)
        let x_reshaped = x.reshape(&[leading_dims, trailing_dims]);

        // Compute mean over dim=1 (the feature dimension)
        let mean = x_reshaped.mean_dim(Axis(1), true); // (leading_dims, 1)

        // Compute variance: E[(x - mean)^2]
        let diff = x_reshaped.sub(&mean);
        let sq_diff = diff.elem_mul(&diff);
        let var = sq_diff.mean_dim(Axis(1), true); // (leading_dims, 1)

        // Normalize: (x - mean) / sqrt(var + eps)
        let eps_tensor = x.constant_like(vec![self.eps], &[1]);
        let denom = var.add(&eps_tensor).sqrt();
        let x_norm = diff.div(&denom); // (leading_dims, trailing_dims)

        // Reshape gamma and beta to broadcast correctly
        let gamma_reshaped = self.gamma.to_dtype(dtype).reshape(&[1, trailing_dims]);
        let beta_reshaped = self.beta.to_dtype(dtype).reshape(&[1, trailing_dims]);

        // Scale and shift: gamma * x_norm + beta
        let out = x_norm.elem_mul(&gamma_reshaped).add(&beta_reshaped);

        // Reshape back to original shape
        out.reshape(input_shape.truncate_rank())
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

    #[test]
    fn test_layernorm_assert_trainable() {
        let ln = LayerNorm::new(vec![4]);
        let x = crate::macros::trainable_input(&[2, 4]);
        crate::macros::assert_trainable!(ln, x);
    }
}
