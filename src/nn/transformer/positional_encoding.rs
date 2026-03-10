use crate::Device;
use crate::io::StateDict;
use crate::nn::Module;
use crate::nn::layers::embedding::Embedding;
use crate::tensor::TensorOps;
use crate::tensor::{RawTensor, Tensor};

/// Positional encoding type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionalEncodingType {
    /// Sinusoidal (fixed, non-learnable) positional encoding
    Sinusoidal,
    /// Learned positional embeddings
    Learned,
}

/// Positional Encoding for transformer models
///
/// Adds position information to token embeddings using either:
/// - Sinusoidal encoding (fixed, based on sin/cos functions)
/// - Learned embeddings (trainable parameters)
///
/// # Arguments
/// * `embed_dim` - Embedding dimension
/// * `max_len` - Maximum sequence length
/// * `encoding_type` - Type of positional encoding
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::{PositionalEncoding, PositionalEncodingType};
/// # use volta::Device;
/// let pos_enc = PositionalEncoding::new(512, 1024, PositionalEncodingType::Sinusoidal);
/// ```
#[derive(Clone, Debug)]
pub struct PositionalEncoding {
    embed_dim: usize,
    max_len: usize,
    encoding_type: PositionalEncodingType,
    /// Sinusoidal: pre-computed fixed encoding tensor
    encoding: Option<Tensor>,
    /// Learned: trainable embedding lookup table
    learned_embedding: Option<Embedding>,
}

impl PositionalEncoding {
    /// Create a new positional encoding layer
    ///
    /// # Arguments
    /// * `embed_dim` - Embedding dimension (must be even)
    /// * `max_len` - Maximum sequence length
    /// * `encoding_type` - Type of positional encoding
    /// # Panics
    /// `embed_dim` must be even for sinusoidal positional encoding
    #[must_use]
    pub fn new(embed_dim: usize, max_len: usize, encoding_type: PositionalEncodingType) -> Self {
        assert!(
            embed_dim.is_multiple_of(2),
            "embed_dim must be even for sinusoidal positional encoding"
        );

        let (encoding, learned_embedding) = match encoding_type {
            PositionalEncodingType::Sinusoidal => (
                Some(Self::create_sinusoidal_encoding(
                    embed_dim,
                    max_len,
                    Device::CPU,
                )),
                None,
            ),
            PositionalEncodingType::Learned => (None, Some(Embedding::new(max_len, embed_dim))),
        };

        Self {
            embed_dim,
            max_len,
            encoding_type,
            encoding,
            learned_embedding,
        }
    }

    /// Create sinusoidal positional encoding
    ///
    /// Uses the formula from "Attention is All You Need":
    /// PE(pos, 2i) = sin(pos / 10000^(2i/`d_model`))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/`d_model`))
    fn create_sinusoidal_encoding(embed_dim: usize, max_len: usize, device: Device) -> Tensor {
        let mut encoding_data = Vec::with_capacity(max_len * embed_dim);

        // Create the div_term values: 10000^(2i/d_model)
        let mut div_term = Vec::with_capacity(embed_dim / 2);
        for i in 0..(embed_dim / 2) {
            let exponent = (2 * i) as f32 / embed_dim as f32;
            div_term.push(10000.0_f32.powf(-exponent));
        }

        // Compute positional encoding for each position
        for pos in 0..max_len {
            for &div_term_val in &div_term {
                let value = pos as f32 * div_term_val;
                encoding_data.push(value.sin()); // PE(pos, 2i)
                encoding_data.push(value.cos()); // PE(pos, 2i+1)
            }
        }

        let encoding = RawTensor::new(encoding_data, &[max_len, embed_dim], false);
        TensorOps::to_device(&encoding, device)
    }

    /// Add positional encoding to input embeddings
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, `seq_len`, `embed_dim`]
    ///
    /// # Returns
    /// Output tensor of shape [batch, `seq_len`, `embed_dim`] with positional encoding added
    /// # Panics
    /// Sequence length exceeds max length
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.borrow().shape.clone();
        let seq_len = *shape.get(1).unwrap();

        assert!(
            seq_len <= self.max_len,
            "Sequence length {} exceeds maximum length {}",
            seq_len,
            self.max_len
        );

        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => {
                // Get the positional encoding for the sequence
                let enc = self
                    .encoding
                    .as_ref()
                    .expect("Sinusoidal encoding should be created");

                let input_shape = x.borrow().shape.clone();
                let batch_size = *input_shape.first().unwrap();
                let seq_len = *input_shape.get(1).unwrap();
                let embed_dim = *input_shape.get(2).unwrap();

                // Slice encoding to match sequence length: [max_len, embed_dim] -> [seq_len, embed_dim]
                let enc_slice = TensorOps::shrink(enc, &[(0, seq_len), (0, embed_dim)]);

                // Add encoding to each batch element individually
                // Process each batch slice as [seq_len, embed_dim]
                let mut result_data = Vec::with_capacity(batch_size * seq_len * embed_dim);
                let x_data = x.borrow().data.to_vec();
                let enc_data = enc_slice.borrow().data.to_vec();
                let slice_size = seq_len * embed_dim;

                for b in 0..batch_size {
                    let start = b * slice_size;
                    for i in 0..slice_size {
                        let x_val = x_data.get(start + i).copied().unwrap_or(0.0);
                        let enc_val = enc_data.get(i).copied().unwrap_or(0.0);
                        result_data.push(x_val + enc_val);
                    }
                }

                RawTensor::new(result_data, &[batch_size, seq_len, embed_dim], false)
            }
            PositionalEncodingType::Learned => {
                let emb = self
                    .learned_embedding
                    .as_ref()
                    .expect("Learned embedding should be created");

                let input_shape = x.borrow().shape.clone();
                let batch_size = *input_shape.first().unwrap();
                let seq_len = *input_shape.get(1).unwrap();
                let embed_dim = *input_shape.get(2).unwrap();

                // Create position indices [0, 1, ..., seq_len-1]
                let positions: Vec<usize> = (0..seq_len).collect();
                // Look up learned embeddings: [seq_len, embed_dim]
                let pos_emb = emb.forward(&positions);

                // Add to each batch element
                let mut result_data = Vec::with_capacity(batch_size * seq_len * embed_dim);
                let x_data = x.borrow().data.to_vec();
                let emb_data = pos_emb.borrow().data.to_vec();
                let slice_size = seq_len * embed_dim;

                for b in 0..batch_size {
                    let start = b * slice_size;
                    for i in 0..slice_size {
                        let x_val = x_data.get(start + i).copied().unwrap_or(0.0);
                        let emb_val = emb_data.get(i).copied().unwrap_or(0.0);
                        result_data.push(x_val + emb_val);
                    }
                }

                RawTensor::new(result_data, &[batch_size, seq_len, embed_dim], false)
            }
        }
    }

    /// Get the encoding type
    #[must_use]
    pub const fn encoding_type(&self) -> PositionalEncodingType {
        self.encoding_type
    }

    /// Get the maximum sequence length
    #[must_use]
    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    /// Get the embedding dimension
    #[must_use]
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get trainable parameters (only for Learned encoding)
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        match &self.learned_embedding {
            Some(emb) => emb.parameters(),
            None => vec![],
        }
    }

    /// Get the state dictionary for serialization
    #[must_use]
    pub fn state_dict(&self) -> StateDict {
        match &self.learned_embedding {
            Some(emb) => {
                let inner = emb.state_dict();
                let mut state = StateDict::new();
                for (key, value) in &inner {
                    state.insert(format!("learned_embedding.{key}"), value.clone());
                }
                state
            }
            None => StateDict::new(),
        }
    }

    /// Load weights from a state dictionary
    pub fn load_state_dict(&mut self, state: &StateDict) {
        if let Some(emb) = &mut self.learned_embedding {
            let mut inner_state = StateDict::new();
            let prefix = "learned_embedding.";
            for (key, value) in state {
                if let Some(stripped) = key.strip_prefix(prefix) {
                    inner_state.insert(stripped.to_string(), value.clone());
                }
            }
            emb.load_state_dict(&inner_state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_encoding_creation() {
        let pos_enc = PositionalEncoding::new(512, 1024, PositionalEncodingType::Sinusoidal);
        assert_eq!(pos_enc.embed_dim, 512);
        assert_eq!(pos_enc.max_len, 1024);
        assert_eq!(pos_enc.encoding_type, PositionalEncodingType::Sinusoidal);
    }

    #[test]
    fn test_sinusoidal_encoding_forward_shape() {
        let pos_enc = PositionalEncoding::new(128, 512, PositionalEncodingType::Sinusoidal);

        // Input: [batch=2, seq_len=10, embed_dim=128]
        let x = RawTensor::new(vec![0.0; 2560], &[2, 10, 128], false);
        let output = pos_enc.forward(&x);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_sinusoidal_encoding_max_length() {
        let pos_enc = PositionalEncoding::new(128, 100, PositionalEncodingType::Sinusoidal);

        // Test with sequence length at max
        let x = RawTensor::new(vec![0.0; 12800], &[1, 100, 128], false);
        let output = pos_enc.forward(&x);
        assert_eq!(output.borrow().shape, vec![1, 100, 128]);

        // Test with sequence length below max
        let x = RawTensor::new(vec![0.0; 640], &[1, 5, 128], false);
        let output = pos_enc.forward(&x);
        assert_eq!(output.borrow().shape, vec![1, 5, 128]);
    }

    #[test]
    fn test_sinusoidal_encoding_odd_embed_dim_fails() {
        // This should panic because embed_dim must be even
        let result = std::panic::catch_unwind(|| {
            let _ = PositionalEncoding::new(127, 512, PositionalEncodingType::Sinusoidal);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_sinusoidal_encoding_values() {
        let pos_enc = PositionalEncoding::new(4, 10, PositionalEncodingType::Sinusoidal);

        // Create a simple input
        let x = RawTensor::new(vec![0.0; 40], &[1, 10, 4], false);
        let output = pos_enc.forward(&x);

        // The output should be different from input (positional encoding was added)
        let output_data = output.borrow().data.clone();
        let input_data = x.borrow().data.clone();

        // At least some values should be different
        let differences = output_data
            .iter()
            .zip(input_data.iter())
            .filter(|(out, inp)| (**out - **inp).abs() > 1e-6)
            .count();

        assert!(
            differences > 0,
            "Positional encoding should modify the input"
        );
    }

    #[test]
    fn test_learned_encoding_creation() {
        let pos_enc = PositionalEncoding::new(64, 128, PositionalEncodingType::Learned);
        assert_eq!(pos_enc.embed_dim, 64);
        assert_eq!(pos_enc.max_len, 128);
        assert_eq!(pos_enc.encoding_type, PositionalEncodingType::Learned);
        assert!(pos_enc.learned_embedding.is_some());
        assert!(pos_enc.encoding.is_none());
    }

    #[test]
    fn test_learned_encoding_forward_shape() {
        let pos_enc = PositionalEncoding::new(64, 128, PositionalEncodingType::Learned);

        // Input: [batch=2, seq_len=10, embed_dim=64]
        let x = RawTensor::new(vec![0.0; 1280], &[2, 10, 64], false);
        let output = pos_enc.forward(&x);

        assert_eq!(output.borrow().shape, vec![2, 10, 64]);
    }

    #[test]
    fn test_learned_encoding_modifies_input() {
        let pos_enc = PositionalEncoding::new(32, 64, PositionalEncodingType::Learned);

        let x = RawTensor::new(vec![0.0; 320], &[1, 10, 32], false);
        let output = pos_enc.forward(&x);

        let output_data = output.borrow().data.clone();
        let input_data = x.borrow().data.clone();

        let differences = output_data
            .iter()
            .zip(input_data.iter())
            .filter(|(out, inp)| (**out - **inp).abs() > 1e-6)
            .count();

        assert!(
            differences > 0,
            "Learned positional encoding should modify the input"
        );
    }

    #[test]
    fn test_learned_encoding_has_parameters() {
        let pos_enc = PositionalEncoding::new(32, 64, PositionalEncodingType::Learned);
        let params = pos_enc.parameters();
        assert_eq!(
            params.len(),
            1,
            "Learned encoding should have one parameter (weight)"
        );
        // Weight shape: [max_len, embed_dim] = [64, 32]
        assert_eq!(params.first().unwrap().borrow().shape, vec![64, 32]);
    }

    #[test]
    fn test_sinusoidal_encoding_has_no_parameters() {
        let pos_enc = PositionalEncoding::new(32, 64, PositionalEncodingType::Sinusoidal);
        let params = pos_enc.parameters();
        assert!(
            params.is_empty(),
            "Sinusoidal encoding should have no trainable parameters"
        );
    }

    #[test]
    fn test_learned_encoding_state_dict() {
        let pos_enc = PositionalEncoding::new(32, 64, PositionalEncodingType::Learned);
        let state = pos_enc.state_dict();
        assert!(
            !state.is_empty(),
            "Learned encoding should have state dict entries"
        );
        assert!(
            state
                .iter()
                .any(|(k, _)| k.starts_with("learned_embedding.")),
            "State dict keys should be prefixed with 'learned_embedding.'"
        );
    }
}
