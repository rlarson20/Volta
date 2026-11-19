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

pub use attention::ScaledDotProductAttention;
pub use block::TransformerBlock;
pub use feed_forward::FeedForward;
pub use mask::causal_mask;
pub use multi_head_attention::MultiHeadAttention;
pub use positional_encoding::{PositionalEncoding, PositionalEncodingType};
