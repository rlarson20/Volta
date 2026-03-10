//! Transformer components for building GPT-style models
//!
//! This module provides the building blocks for transformer architectures:
//! - Multi-head attention mechanisms
//! - Feed-forward networks
//! - Transformer blocks (layer norm + attention + FFN + residuals)
//! - Positional encodings (sinusoidal and learned)
//! - Causal masks for autoregressive models

pub mod attention;
pub mod block;
pub mod feed_forward;
pub mod mask;
pub mod multi_head_attention;
pub mod positional_encoding;

pub use attention::ScaledDotProductAttention;
pub use block::TransformerBlock;
pub use feed_forward::FeedForward;
pub use mask::causal_mask;
pub use multi_head_attention::MultiHeadAttention;
pub use positional_encoding::{PositionalEncoding, PositionalEncodingType};
