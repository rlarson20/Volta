use crate::error::OrPanic;
use crate::io::StateDict;
use crate::nn::{FeedForward, Layer, LayerNorm, Module, MultiHeadAttention, insert_prefixed};
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
/// * `expansion_factor` - FFN expansion factor
/// * `dropout` - Dropout probability
///
/// # Example
/// ```no_run
/// # use volta::nn::TransformerBlock;
/// let block = TransformerBlock::new(512, 8, 4, 0.1);
/// ```
pub struct TransformerBlock {
    /// Layer norm applied before attention.
    norm1: LayerNorm,
    /// Self-attention sublayer.
    attention: MultiHeadAttention,
    /// Layer norm applied before the feed-forward network.
    norm2: LayerNorm,
    /// Feed-forward sublayer.
    ffn: FeedForward,
}

impl TransformerBlock {
    /// Create a new transformer block
    ///
    /// # Arguments
    /// * `embed_dim` - Embedding dimension
    /// * `num_heads` - Number of attention heads
    /// * `expansion_factor` - FFN expansion factor
    /// * `dropout` - Dropout probability
    ///
    /// # Panics
    /// If an argument is out of range (see [`Self::try_new`]).
    #[must_use]
    #[track_caller]
    pub fn new(embed_dim: usize, num_heads: usize, expansion_factor: usize, dropout: f32) -> Self {
        Self::try_new(embed_dim, num_heads, expansion_factor, dropout).or_panic("TransformerBlock")
    }

    /// [`Self::new`], checking every argument before building a sublayer.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` from the sublayer that would refuse the argument
    /// ([`MultiHeadAttention::try_new`], [`FeedForward::try_new`]): a zero size,
    /// `embed_dim` not a multiple of `num_heads`, or `dropout` outside `[0, 1]`.
    pub fn try_new(
        embed_dim: usize,
        num_heads: usize,
        expansion_factor: usize,
        dropout: f32,
    ) -> crate::Result<Self> {
        MultiHeadAttention::check(embed_dim, num_heads, dropout)?;
        FeedForward::check(embed_dim, expansion_factor, dropout)?;
        Ok(Self {
            norm1: LayerNorm::new(vec![embed_dim]),
            attention: MultiHeadAttention::new(embed_dim, num_heads, dropout),
            norm2: LayerNorm::new(vec![embed_dim]),
            ffn: FeedForward::new(embed_dim, expansion_factor, dropout),
        })
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
        // Pre-LN Architecture. Neither sublayer adds its own residual; the block adds both.
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
        // Layer norms, FFN, then attention; a tensor tied across them is listed once.
        crate::nn::unique_parameters(
            self.norm1
                .parameters()
                .into_iter()
                .chain(self.norm2.parameters())
                .chain(self.ffn.parameters())
                .chain(self.attention.parameters()),
        )
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        insert_prefixed(&mut state, "norm1.", self.norm1.state_dict());
        insert_prefixed(&mut state, "norm2.", self.norm2.state_dict());
        insert_prefixed(&mut state, "ffn.", self.ffn.state_dict());
        insert_prefixed(&mut state, "attention.", self.attention.state_dict());
        state
    }

    fn plan_load<'a>(&self, plan: &mut crate::nn::LoadPlan<'a>) {
        plan.child("norm1.", &self.norm1);
        plan.child("norm2.", &self.norm2);
        plan.child("ffn.", &self.ffn);
        plan.child("attention.", &self.attention);
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
        // The sublayers carry the sizes: two `LayerNorm`s (gamma, beta), the FFN's
        // `d -> 4d -> d` Linears (with bias) and four `d -> d` attention projections, of
        // which only the output projection has a bias.
        let (d, expansion) = (512, 4);
        let block = TransformerBlock::new(d, 8, expansion, 0.1);
        let numel: usize = block
            .parameters()
            .iter()
            .map(|p| p.borrow().shape.iter().product::<usize>())
            .sum();
        let hidden = expansion * d;
        let expected = 2 * 2 * d + (d * hidden + hidden) + (hidden * d + d) + (4 * d * d + d);
        assert_eq!(numel, expected);
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
        // Covers every parameter: norm1, attention's q/k/v/out projections, norm2, ffn.
        let block = TransformerBlock::new(16, 2, 2, 0.0);
        let x = crate::macros::trainable_input(&[2, 3, 16]);
        crate::macros::assert_trainable!(block, x);
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
        dst.load_state_dict(&src.state_dict())
            .expect("matching state loads");
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

    /// #123: the input gradient through both pre-LN sublayers and residuals, by finite
    /// differences in f64 under a random weighted loss, with and without the causal mask.
    #[test]
    fn test_block_input_gradient_matches_finite_differences() {
        for causal in [false, true] {
            crate::tensor::manual_seed(0xB10C);
            let mut block = TransformerBlock::new(4, 2, 2, 0.0);
            block.to_dtype(crate::DType::F64);
            let x = crate::nn::transformer::f64_operand(&[2, 3, 4], true);
            let w = crate::nn::transformer::f64_operand(&[2, 3, 4], false);
            let (max, mean, ok) = RawTensor::check_gradients(
                &x,
                |t| block.forward(t, causal).elem_mul(&w).sum(),
                1e-6,
                1e-6,
            );
            assert!(
                ok,
                "TransformerBlock input gradient (causal={causal}): max={max:.3e} mean={mean:.3e}"
            );
        }
    }
}
