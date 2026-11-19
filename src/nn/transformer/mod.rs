//! Transformer components for building GPT-style models
//!
//! This module provides the building blocks for transformer architectures:
//! - Multi-head attention mechanisms
//! - Feed-forward networks
//! - Transformer blocks (layer norm + attention + FFN + residuals)
//! - Positional encodings (sinusoidal and learned)
//! - Causal masks for autoregressive models

/// Scaled dot-product attention.
pub(crate) mod attention;
/// `TransformerBlock`.
pub(crate) mod block;
/// Position-wise feed-forward network.
pub(crate) mod feed_forward;
/// Attention masks.
pub(crate) mod mask;
/// Multi-head attention.
pub(crate) mod multi_head_attention;
/// Positional encodings.
pub(crate) mod positional_encoding;

/// Seeded normal values as an f64 tensor, a leaf that requires grad when `grad`. The
/// transformer gradient checks run in f64, so finite-difference rounding (about 1e-7
/// relative in f32) can't mask or fake a mismatch.
#[cfg(test)]
pub(crate) fn f64_operand(dims: &[usize], grad: bool) -> crate::tensor::Tensor {
    let vals: Vec<f64> = crate::tensor::randn(dims)
        .data()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let t = crate::RawTensor::from_typed(vals, dims);
    t.set_requires_grad(grad);
    t
}

pub use attention::ScaledDotProductAttention;
pub use block::TransformerBlock;
pub use feed_forward::FeedForward;
pub use mask::causal_mask;
pub use multi_head_attention::MultiHeadAttention;
pub use positional_encoding::{PositionalEncoding, PositionalEncodingType};
