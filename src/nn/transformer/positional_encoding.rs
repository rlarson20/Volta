use crate::Device;
use crate::io::StateDict;
use crate::nn::layers::embedding::Embedding;
use crate::nn::{Layer, Module};
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
        encoding.to_device(device)
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

    /// The `[max_len, embed_dim]` table whose first `seq_len` rows are added: the fixed
    /// sinusoids, or the learned embedding's weight. Positions are always `0..seq_len`, so
    /// the rows are a `shrink`, not an `Embedding` lookup.
    fn table(&self) -> &Tensor {
        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => self
                .encoding
                .as_ref()
                .expect("Sinusoidal encoding should be created"),
            PositionalEncodingType::Learned => {
                &self
                    .learned_embedding
                    .as_ref()
                    .expect("Learned embedding should be created")
                    .weight
            }
        }
    }
}

impl Module for PositionalEncoding {
    /// The learned embedding's weight; none for sinusoidal encoding.
    fn parameters(&self) -> Vec<Tensor> {
        match &self.learned_embedding {
            Some(emb) => emb.parameters(),
            None => vec![],
        }
    }

    fn state_dict(&self) -> StateDict {
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

    fn load_state_dict(&mut self, state: &StateDict) {
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

    /// Moves the learned weight, and also the fixed sinusoid table, which is not a parameter
    /// and so is not moved by the default.
    fn to_device(&mut self, device: Device) {
        if let Some(enc) = &self.encoding {
            self.encoding = Some(enc.to_device(device.clone()));
        }
        if let Some(emb) = &mut self.learned_embedding {
            emb.to_device(device);
        }
    }
}

impl Layer for PositionalEncoding {
    /// Adds the encoding for positions `0..seq_len` to `x` (`[batch, seq_len, embed_dim]`),
    /// built from graph ops, so the gradient reaches `x` and, in learned mode, the weight.
    ///
    /// # Panics
    /// If `x` is not rank 3, its last dim is not `embed_dim`, or `seq_len > max_len`.
    fn forward(&self, x: &Tensor) -> Tensor {
        let [batch, seq_len, embed_dim] = x.shape().fixed::<3>();
        assert_eq!(
            embed_dim, self.embed_dim,
            "PositionalEncoding: input embed_dim {embed_dim} != layer embed_dim {}",
            self.embed_dim
        );
        assert!(
            seq_len <= self.max_len,
            "Sequence length {} exceeds maximum length {}",
            seq_len,
            self.max_len
        );

        let rows = self
            .table()
            .shrink(&[(0, seq_len), (0, embed_dim)])
            .reshape(&[1, seq_len, embed_dim])
            .expand(&[batch, seq_len, embed_dim]);
        x.add(&rows)
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
            .to_host_f32()
            .iter()
            .zip(input_data.to_host_f32().iter())
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
            .to_host_f32()
            .iter()
            .zip(input_data.to_host_f32().iter())
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

    use crate::index::fetch;

    /// Seeded random `[2, 5, 6]` input that requires grad, and a random loss weight.
    fn input_and_weight() -> (Tensor, Tensor) {
        crate::tensor::manual_seed(0xB3);
        let x = RawTensor::randn(&[2, 5, 6]);
        x.set_requires_grad(true);
        (x, RawTensor::randn(&[2, 5, 6]))
    }

    #[test]
    fn test_sinusoidal_encoding_matches_formula() {
        let pe = PositionalEncoding::new(6, 10, PositionalEncodingType::Sinusoidal);
        let (x, _) = input_and_weight();
        let out = pe.forward(&x).data();
        let x_data = x.data();
        assert_eq!(out.len(), x_data.len());
        for b in 0..2 {
            for pos in 0..5 {
                for i in 0..3 {
                    let angle = pos as f32 * 10000_f32.powf(-((2 * i) as f32) / 6.0);
                    let at = (b * 5 + pos) * 6 + 2 * i;
                    for (k, expected) in [(at, angle.sin()), (at + 1, angle.cos())] {
                        let added = fetch(&out, k) - fetch(&x_data, k);
                        assert!(
                            (added - expected).abs() < 1e-5,
                            "b={b} pos={pos} i={i}: added {added}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_learned_encoding_adds_weight_rows() {
        let pe = PositionalEncoding::new(6, 10, PositionalEncodingType::Learned);
        let (x, _) = input_and_weight();
        let out = pe.forward(&x).data();
        let x_data = x.data();
        let table = pe.parameters().first().expect("learned weight").data();
        assert_eq!(out.len(), x_data.len());
        for k in 0..out.len() {
            let row_col = k % (5 * 6);
            let added = fetch(&out, k) - fetch(&x_data, k);
            assert!(
                (added - fetch(&table, row_col)).abs() < 1e-6,
                "index {k}: added {added}, expected weight[{row_col}]"
            );
        }
    }

    /// The output is `x + rows`, so for the loss `sum(out * w)` the input gradient is `w`
    /// in both modes (B3: the old forward rebuilt its output from raw data, so `x` got none).
    #[test]
    fn test_positional_encoding_input_gradient_is_loss_weight() {
        for kind in [
            PositionalEncodingType::Sinusoidal,
            PositionalEncodingType::Learned,
        ] {
            let pe = PositionalEncoding::new(6, 10, kind);
            let (x, w) = input_and_weight();
            pe.forward(&x)
                .elem_mul(&w)
                .sum()
                .into_loss()
                .expect("scalar loss")
                .backward();
            let grad = x.grad().expect("input has a gradient");
            let w = w.data();
            assert_eq!(grad.len(), w.len());
            for (k, (g, expected)) in grad.iter().zip(&w).enumerate() {
                assert!(
                    (g - expected).abs() < 1e-6,
                    "{kind:?} index {k}: {g} vs {expected}"
                );
            }
        }
    }

    /// Row `p < seq_len` of the learned weight's gradient is `sum_b w[b, p, :]`; rows at or
    /// past `seq_len` were not used and get zero.
    #[test]
    fn test_learned_encoding_weight_gradient() {
        let pe = PositionalEncoding::new(6, 10, PositionalEncodingType::Learned);
        let (x, w) = input_and_weight();
        pe.forward(&x)
            .elem_mul(&w)
            .sum()
            .into_loss()
            .expect("scalar loss")
            .backward();
        let grad = pe
            .parameters()
            .first()
            .expect("learned weight")
            .grad()
            .expect("weight has a gradient");
        let w = w.data();
        assert_eq!(grad.len(), 10 * 6);
        for (k, g) in grad.iter().enumerate() {
            let expected = if k < 5 * 6 {
                fetch(&w, k) + fetch(&w, 5 * 6 + k)
            } else {
                0.0
            };
            assert!(
                (g - expected).abs() < 1e-5,
                "weight grad {k}: {g} vs {expected}"
            );
        }
    }

    #[test]
    fn test_learned_encoding_assert_trainable() {
        let pe = PositionalEncoding::new(8, 16, PositionalEncodingType::Learned);
        let x = crate::macros::trainable_input(&[2, 4, 8]);
        crate::assert_trainable!(pe, x);
    }

    #[test]
    #[should_panic(expected = "input embed_dim 4 != layer embed_dim 6")]
    fn test_positional_encoding_rejects_wrong_embed_dim() {
        let pe = PositionalEncoding::new(6, 10, PositionalEncodingType::Sinusoidal);
        let _ = pe.forward(&RawTensor::zeros(&[1, 3, 4]));
    }
}
