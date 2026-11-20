use crate::error::OrPanic;
use crate::io::StateDict;
use crate::nn::hyper;
use crate::nn::{Dropout, GELU, Layer, Linear, Module, insert_prefixed};
use crate::tensor::Tensor;

/// Feed-Forward Network (FFN) layer for transformer blocks
///
/// The FFN consists of two linear transformations with a GELU activation in between:
/// Linear → GELU → Dropout → Linear → Dropout
///
/// This is the feed-forward half of torch's `TransformerEncoderLayer`
/// (`dropout2(linear2(dropout(activation(linear1(x)))))`), with one probability for
/// both dropouts. Dropout is active only in training mode, the default as for every
/// torch module; call `Module::train(false)` for inference.
///
/// The first layer typically expands the dimension by a factor (default 4x),
/// and the second layer projects it back to the original dimension.
///
/// # Arguments
/// * `embed_dim` - Input/output dimension
/// * `expansion_factor` - Factor by which to expand the intermediate dimension
/// * `dropout` - Dropout probability
///
/// # Example
/// ```no_run
/// # use volta::nn::FeedForward;
/// let ffn = FeedForward::new(512, 4, 0.1);
/// ```
pub struct FeedForward {
    /// First projection, `embed_dim` to `embed_dim * expansion_factor`.
    expand: Linear,
    /// Second projection back to `embed_dim`.
    contract: Linear,
    /// Nonlinearity between the projections.
    activation: GELU,
    /// Dropout applied after the activation and again after `contract`. `Dropout` keeps
    /// no state beyond `p` and the mode, so one instance serves both places.
    dropout: Dropout,
    /// Input and output width.
    embed_dim: usize,
    /// Hidden width as a multiple of `embed_dim`.
    expansion_factor: usize,
}

impl FeedForward {
    /// Create a new feed-forward network layer
    ///
    /// # Arguments
    /// * `embed_dim` - Input/output dimension
    /// * `expansion_factor` - Factor by which to expand the intermediate dimension
    /// * `dropout` - Dropout probability
    ///
    /// # Panics
    /// If an argument is out of range (see [`Self::try_new`]).
    #[must_use]
    #[track_caller]
    pub fn new(embed_dim: usize, expansion_factor: usize, dropout: f32) -> Self {
        Self::try_new(embed_dim, expansion_factor, dropout).or_panic("FeedForward")
    }

    /// [`Self::new`], checking the arguments first.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` naming the argument if `embed_dim` or
    /// `expansion_factor` is zero, or `dropout` is not in `[0, 1]`.
    pub fn try_new(embed_dim: usize, expansion_factor: usize, dropout: f32) -> crate::Result<Self> {
        Self::check(embed_dim, expansion_factor, dropout)?;
        let intermediate_dim = embed_dim * expansion_factor;

        Ok(Self {
            expand: Linear::new(embed_dim, intermediate_dim, true),
            contract: Linear::new(intermediate_dim, embed_dim, true),
            activation: GELU::new(),
            dropout: Dropout::new(dropout),
            embed_dim,
            expansion_factor,
        })
    }

    /// `Ok` if `try_new` accepts these arguments.
    pub(crate) fn check(
        embed_dim: usize,
        expansion_factor: usize,
        dropout: f32,
    ) -> crate::Result<()> {
        hyper::at_least_one("FeedForward", "embed_dim", embed_dim)?;
        hyper::at_least_one("FeedForward", "expansion_factor", expansion_factor)?;
        hyper::unit_interval("FeedForward", "dropout", dropout)
    }

    /// Create a new feed-forward network with default settings
    ///
    /// Uses `expansion_factor`=4 and `dropout`=0.1
    #[must_use]
    #[track_caller]
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
    ///
    /// # Panics
    /// If `x` is not 3D, or its last dimension is not `embed_dim`.
    #[must_use]
    #[track_caller]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = x.borrow().shape.clone();

        // Reshape to 2D for Linear layers: [batch, seq_len, embed_dim] -> [batch * seq_len, embed_dim]
        let batch_size = input_shape.dim_at(0);
        let seq_len = input_shape.dim_at(1);
        let embed_dim = input_shape.dim_at(2);

        let x_2d = x.reshape(&[batch_size * seq_len, embed_dim]);

        // Expand: [batch * seq_len, embed_dim] -> [batch * seq_len, intermediate_dim]
        let x = self.expand.forward(&x_2d);

        // Apply GELU activation, then dropout (identity in eval mode)
        let x = self.activation.forward(&x);
        let x = self.dropout.forward(&x);

        // Contract: [batch * seq_len, intermediate_dim] -> [batch * seq_len, embed_dim]
        let x = self.contract.forward(&x);
        let x = self.dropout.forward(&x);

        // Reshape back to 3D: [batch * seq_len, embed_dim] -> [batch, seq_len, embed_dim]
        x.reshape(&[batch_size, seq_len, embed_dim])
    }

    /// Get the intermediate (hidden) dimension, `embed_dim * expansion_factor`
    #[must_use]
    pub const fn intermediate_dim(&self) -> usize {
        self.embed_dim * self.expansion_factor
    }
}

impl Module for FeedForward {
    fn parameters(&self) -> Vec<Tensor> {
        crate::nn::unique_parameters(
            self.expand
                .parameters()
                .into_iter()
                .chain(self.contract.parameters()),
        )
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        insert_prefixed(&mut state, "expand.", self.expand.state_dict());
        insert_prefixed(&mut state, "contract.", self.contract.state_dict());
        state
    }

    fn plan_load<'a>(&self, plan: &mut crate::nn::LoadPlan<'a>) {
        plan.child("expand.", &self.expand);
        plan.child("contract.", &self.contract);
    }

    fn train(&mut self, mode: bool) {
        self.dropout.train(mode);
    }
}

impl Layer for FeedForward {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
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

    #[test]
    fn test_ffn_intermediate_dim() {
        assert_eq!(FeedForward::new(16, 4, 0.0).intermediate_dim(), 64);
        assert_eq!(FeedForward::new(6, 3, 0.0).intermediate_dim(), 18);
    }

    #[test]
    fn test_ffn_dropout_identity_in_eval_active_in_training() {
        crate::tensor::manual_seed(21);
        let mut ffn = FeedForward::new(8, 4, 0.5);
        // Same weights, no dropout: what the layer must compute in eval mode.
        let mut reference = FeedForward::new(8, 4, 0.0);
        reference
            .load_state_dict(&ffn.state_dict())
            .expect("same architecture");
        let x = crate::tensor::randn(&[2, 3, 8]);
        let expected = reference.forward(&x).data();

        ffn.train(false);
        assert_eq!(ffn.forward(&x).data(), expected, "eval mode must not drop");

        ffn.train(true);
        let trained = ffn.forward(&x).data();
        assert_eq!(trained.len(), expected.len());
        assert!(
            trained
                .iter()
                .zip(expected.iter())
                .any(|(a, e)| (a - e).abs() > 1e-6),
            "training mode at p=0.5 left the output unchanged"
        );
        // The second dropout acts on the output itself: about half of it is zeroed.
        let zeros = trained.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "no output element was dropped: {trained:?}");

        // Back in eval mode the layer is deterministic again.
        ffn.train(false);
        assert_eq!(ffn.forward(&x).data(), expected);
    }

    #[test]
    fn test_ffn_assert_trainable() {
        let ffn = FeedForward::new(16, 2, 0.0);
        let x = crate::macros::trainable_input(&[2, 3, 16]);
        crate::macros::assert_trainable!(ffn, x);
    }

    /// #123: the input gradient through expand, GELU and contract, by finite differences
    /// in f64 under a random weighted loss (`sum` alone would weight every output alike).
    #[test]
    fn test_ffn_input_gradient_matches_finite_differences() {
        crate::tensor::manual_seed(0xFF01);
        let mut ffn = FeedForward::new(4, 2, 0.0);
        ffn.to_dtype(crate::DType::F64);
        let x = crate::nn::transformer::f64_operand(&[2, 3, 4], true);
        let w = crate::nn::transformer::f64_operand(&[2, 3, 4], false);
        let (max, mean, ok) =
            RawTensor::check_gradients(&x, |t| ffn.forward(t).elem_mul(&w).sum(), 1e-6, 1e-6);
        assert!(
            ok,
            "FeedForward input gradient: max={max:.3e} mean={mean:.3e}"
        );
    }
}
