use crate::autograd::GradFn;
use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, with_rng};
use rand::Rng;

/// Embedding: Maps integer indices to dense vectors
///
/// This layer creates a lookup table that maps vocabulary indices to embedding vectors.
/// It's commonly used for word embeddings, character embeddings, or any discrete token
/// representation in neural networks.
///
/// # Arguments
/// * `vocab_size` - Size of the vocabulary (number of unique tokens)
/// * `embedding_dim` - Dimension of the embedding vectors
///
/// # Shape
/// - Input: Array of indices `[num_indices]`
/// - Output: `[num_indices, embedding_dim]`
///
/// # Examples
/// ```
/// use volta::{RawTensor, Embedding, nn::Module};
///
/// let embedding = Embedding::new(100, 32);  // 100 vocab, 32-dim embeddings
/// let indices = vec![5, 12, 7];
/// let embedded = embedding.forward(&indices);
/// assert_eq!(embedded.borrow().shape, vec![3, 32]);
/// ```
pub struct Embedding {
    pub weight: Tensor,
    vocab_size: usize,
    dim: usize,
}

impl Embedding {
    /// Creates a new Embedding layer
    ///
    /// Weights are initialized uniformly in the range [-0.1, 0.1]
    #[must_use]
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        assert!(vocab_size > 0, "vocab_size must be positive");
        assert!(embedding_dim > 0, "embedding_dim must be positive");

        // Initialize with uniform distribution [-0.1, 0.1] (PyTorch default)
        let limit = 0.1;
        let data: Vec<f32> = with_rng(|rng| {
            (0..vocab_size * embedding_dim)
                .map(|_| rng.random::<f32>() * 2.0 * limit - limit)
                .collect()
        });

        let weight = RawTensor::new(data, &[vocab_size, embedding_dim], false);
        weight.borrow_mut().requires_grad = true;

        Embedding {
            weight,
            vocab_size,
            dim: embedding_dim,
        }
    }

    /// Forward pass: lookup embeddings for given indices
    ///
    /// # Arguments
    /// * `indices` - Array of vocabulary indices to look up
    ///
    /// # Returns
    /// Tensor of shape `[num_indices, embedding_dim]` containing the embeddings
    ///
    /// # Panics
    /// Panics if any index is >= `vocab_size`
    #[must_use]
    pub fn forward(&self, indices: &[usize]) -> Tensor {
        assert!(!indices.is_empty(), "indices cannot be empty");

        let weight_borrowed = self.weight.borrow();
        let weight_data = &weight_borrowed.data;

        // Gather operation: select rows from weight matrix
        let mut output_data = Vec::with_capacity(indices.len() * self.dim);

        for &idx in indices {
            assert!(
                idx < self.vocab_size,
                "Index {} out of bounds for vocab_size {}",
                idx,
                self.vocab_size
            );
            let start = idx * self.dim;
            let end = start + self.dim;
            if let Some(slice) = weight_data.get(start..end) {
                output_data.extend_from_slice(slice);
            }
        }

        drop(weight_borrowed);

        let output_shape = [indices.len(), self.dim];
        let output = RawTensor::new(output_data, &output_shape, false);

        // Attach gradient function if weight requires grad
        if self.weight.borrow().requires_grad {
            output.borrow_mut().requires_grad = true;
            output.borrow_mut().parents = vec![self.weight.clone()];
            output.borrow_mut().grad_fn = Some(Box::new(EmbeddingGradFn {
                indices: indices.to_vec(),
                vocab_size: self.vocab_size,
                embedding_dim: self.dim,
            }));
        }

        output
    }
}

impl Module for Embedding {
    fn forward(&self, _x: &Tensor) -> Tensor {
        // For compatibility with Module trait, we can't use indices directly
        // This is a workaround - in practice, use the Embedding::forward method directly
        panic!(
            "Embedding layer should be called with Embedding::forward(&indices), not Module::forward"
        );
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        let weight_data = crate::io::TensorData::from_tensor(&self.weight);
        state.insert("weight".to_string(), weight_data);
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        if let Some(weight_data) = state.get("weight") {
            // Validate shape
            assert_eq!(
                weight_data.shape,
                vec![self.vocab_size, self.dim],
                "Weight shape mismatch"
            );

            let mut weight = self.weight.borrow_mut();
            weight.data = crate::storage::Storage::cpu(weight_data.data.clone());
            weight.shape.clone_from(&weight_data.shape);
        }
    }
}

/// Custom gradient function for embedding layer
///
/// Scatters gradients from output back to the weight matrix at specific indices.
/// Handles gradient accumulation for repeated indices.
#[derive(Clone)]
struct EmbeddingGradFn {
    indices: Vec<usize>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl GradFn for EmbeddingGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // out_grad: [num_indices, embedding_dim]
        // Scatter gradients to weight: [vocab_size, embedding_dim]

        let mut grad_weight = vec![0.0; self.vocab_size * self.embedding_dim];

        // Accumulate gradients for each index
        for (i, &idx) in self.indices.iter().enumerate() {
            for d in 0..self.embedding_dim {
                let out_grad_idx = i * self.embedding_dim + d;
                let weight_idx = idx * self.embedding_dim + d;
                let grad_val = out_grad.data.get(out_grad_idx).copied().unwrap_or(0.0);
                if let Some(slot) = grad_weight.get_mut(weight_idx) {
                    *slot += grad_val;
                }
            }
        }

        let grad_tensor =
            RawTensor::new(grad_weight, &[self.vocab_size, self.embedding_dim], false);

        vec![Some(grad_tensor)]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorOps;

    #[test]
    fn test_embedding_forward_shape() {
        let embedding = Embedding::new(100, 32);
        let indices = vec![5, 12, 7, 99];
        let output = embedding.forward(&indices);

        assert_eq!(output.borrow().shape, vec![4, 32]);
    }

    #[test]
    fn test_embedding_values() {
        let embedding = Embedding::new(10, 4);
        let indices = vec![0, 5, 9];
        let output = embedding.forward(&indices);

        let weight_data = &embedding.weight.borrow().data;
        let output_data = &output.borrow().data;

        // Check that the first embedding matches the first row of weight
        for i in 0..4 {
            assert_eq!(
                output_data.get(i).copied().unwrap_or(f32::NAN),
                weight_data.get(i).copied().unwrap_or(f32::NAN)
            );
        }

        // Check that the second embedding (index 5) matches row 5 of weight
        for i in 0..4 {
            assert_eq!(
                output_data.get(4 + i).copied().unwrap_or(f32::NAN),
                weight_data.get(5 * 4 + i).copied().unwrap_or(f32::NAN)
            );
        }
    }

    #[test]
    fn test_embedding_backward_flow() {
        let embedding = Embedding::new(10, 4);
        let indices = vec![2, 7, 2]; // Note: repeated index 2
        let output = embedding.forward(&indices);

        // Sum and backward
        let loss = output.sum();
        loss.backward();

        // Check that weight has gradients
        let grad = embedding.weight.grad();
        assert!(grad.is_some(), "Weight should have gradients");

        let grad_data = grad.unwrap();
        // Gradients for index 2 should be accumulated (appears twice)
        // Each embedding contributes 1.0 per dimension (from sum)
        // So index 2 should have grad of 2.0 per dimension
        for d in 0..4 {
            let grad_at_idx2 = grad_data.get(2 * 4 + d).copied().unwrap_or(f32::NAN);
            assert!(
                (grad_at_idx2 - 2.0).abs() < 1e-5,
                "Expected grad ~2.0 for repeated index, got {}",
                grad_at_idx2
            );
        }

        // Indices 7 should have grad of 1.0 per dimension
        for d in 0..4 {
            let grad_at_idx7 = grad_data.get(7 * 4 + d).copied().unwrap_or(f32::NAN);
            assert!(
                (grad_at_idx7 - 1.0).abs() < 1e-5,
                "Expected grad ~1.0 for single index, got {}",
                grad_at_idx7
            );
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_embedding_invalid_index() {
        let embedding = Embedding::new(10, 4);
        let indices = vec![5, 15]; // 15 is out of bounds
        let _ = embedding.forward(&indices);
    }

    #[test]
    fn test_embedding_parameters() {
        let embedding = Embedding::new(50, 16);
        let params = embedding.parameters();

        assert_eq!(params.len(), 1);
        assert_eq!(params.first().unwrap().borrow().shape, vec![50, 16]);
    }
}
