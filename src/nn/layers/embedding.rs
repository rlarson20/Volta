use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::dtype::DType;
use crate::index::{fetch, slot, span};
use crate::io::StateDict;
use crate::nn::{Layer, Module};
use crate::real::{Real, dispatch_real};
use crate::shape::Shape;
use crate::storage::Storage;
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
/// - Input: I64 or I32 indices of any shape `[..]`
/// - Output: `[.., embedding_dim]`, in the weight's dtype
///
/// # Examples
/// ```
/// use volta::{Embedding, Layer, RawTensor};
///
/// let embedding = Embedding::new(100, 32); // 100 vocab, 32-dim embeddings
/// let indices = RawTensor::from_typed(vec![5i64, 12, 7, 0, 1, 2], &[2, 3]);
/// let embedded = embedding.forward(&indices);
/// assert_eq!(embedded.shape(), vec![2, 3, 32]);
/// ```
#[derive(Clone, Debug)]
pub struct Embedding {
    /// Lookup table, shape `[vocab_size, dim]`.
    pub weight: Tensor,
    /// Number of rows in the table.
    vocab_size: usize,
    /// Width of each row.
    dim: usize,
}

impl Embedding {
    /// Creates a new Embedding layer
    ///
    /// Weights are initialized uniformly in the range [-0.1, 0.1]
    /// # Panics
    /// positive vocab size, embedding dim
    #[must_use]
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        assert!(vocab_size > 0, "vocab_size must be positive");
        assert!(embedding_dim > 0, "embedding_dim must be positive");

        // Uniform [-0.1, 0.1]. This is not PyTorch's default, which is N(0, 1)
        // (docs/audits/2026-11-sweep.md, D2).
        let limit = 0.1;
        let data: Vec<f32> = with_rng(|rng| {
            (0..vocab_size * embedding_dim)
                .map(|_| (rng.random::<f32>() * 2.0).mul_add(limit, -limit))
                .collect()
        });

        let weight = RawTensor::new(data, &[vocab_size, embedding_dim], false);
        weight.set_requires_grad(true);

        Self {
            weight,
            vocab_size,
            dim: embedding_dim,
        }
    }
}

impl Embedding {
    /// The rows `indices` (an I64 or I32 tensor) selects, as `usize`, checked in range.
    ///
    /// # Panics
    /// If `indices` is not I64 or I32, or an index is negative or `>= vocab_size`.
    fn rows(&self, indices: &Tensor) -> Vec<usize> {
        let s = indices.borrow();
        let wide: Vec<i64> = match s.data.dtype() {
            DType::I64 => s.data.as_host::<i64>().expect("I64 checked").to_vec(),
            DType::I32 => s
                .data
                .as_host::<i32>()
                .expect("I32 checked")
                .iter()
                .map(|&i| i64::from(i))
                .collect(),
            other @ (DType::F16
            | DType::BF16
            | DType::F32
            | DType::F64
            | DType::U8
            | DType::Bool) => panic!(
                "Embedding: indices must be I64 or I32, got {other}; cast with \
                 to_dtype(DType::I64)"
            ),
        };
        wide.into_iter()
            .map(|i| {
                usize::try_from(i)
                    .ok()
                    .filter(|&r| r < self.vocab_size)
                    .unwrap_or_else(|| {
                        panic!(
                            "Embedding: index {i} out of range for vocab_size {}",
                            self.vocab_size
                        )
                    })
            })
            .collect()
    }
}

impl Layer for Embedding {
    /// Looks up a row of `weight` for each index in `indices`, an I64 or I32 tensor of any
    /// shape (PyTorch's `nn.Embedding`). The output has shape `[..indices.shape, dim]` and
    /// the weight's dtype; the indices carry no gradient.
    ///
    /// # Panics
    /// If `indices` is not I64 or I32, or an index is negative or `>= vocab_size`.
    fn forward(&self, indices: &Tensor) -> Tensor {
        let rows = self.rows(indices);
        let weight_borrowed = self.weight.borrow();
        let dtype = weight_borrowed.data.dtype();
        let output = dispatch_real!(dtype, T => {
            let weight_data = weight_borrowed.data.host::<T>();
            // Gather: one weight row per index.
            let mut output_data = Vec::with_capacity(rows.len() * self.dim);
            for &row in &rows {
                let start = row * self.dim;
                output_data.extend_from_slice(span(&weight_data, start..start + self.dim));
            }
            Storage::from_compute(dtype, output_data)
        });
        drop(weight_borrowed);

        let mut output_shape = indices.shape().truncate_rank().to_vec();
        output_shape.push(self.dim);

        // The indices are data, not a graph input: only the weight is a parent.
        edge(
            [self.weight.clone()],
            output,
            Shape::derived(output_shape),
            Device::CPU,
            GradFn::Embedding(EmbeddingGradFn {
                indices: rows,
                vocab_size: self.vocab_size,
                embedding_dim: self.dim,
            }),
        )
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        let weight_data = crate::io::TensorData::from_tensor(&self.weight);
        state.insert("weight".to_string(), weight_data);
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
        if let Some(weight_data) = state.get("weight") {
            self.weight.load_weight("weight", weight_data)?;
        }
        Ok(())
    }
}

/// Custom gradient function for embedding layer
///
/// Scatters gradients from output back to the weight matrix at specific indices.
/// Handles gradient accumulation for repeated indices.
#[derive(Clone)]
pub(crate) struct EmbeddingGradFn {
    /// Rows that were looked up, in output order (repeats allowed).
    indices: Vec<usize>,
    /// Number of rows in the table.
    vocab_size: usize,
    /// Width of each row.
    embedding_dim: usize,
}

impl EmbeddingGradFn {
    /// Scatters `out_grad` into a `[vocab_size, embedding_dim]` weight gradient, summing repeated indices.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        // out_grad: [num_indices, embedding_dim]
        // Scatter gradients to weight: [vocab_size, embedding_dim]

        // Accumulate gradients for each index, in the gradient's dtype
        let grad_tensor = dispatch_real!(out_grad.data.dtype(), T => {
            let mut grad_weight = vec![T::ZERO; self.vocab_size * self.embedding_dim];
            let g_host = out_grad.data.host::<T>();
            for (i, &idx) in self.indices.iter().enumerate() {
                for d in 0..self.embedding_dim {
                    let out_grad_idx = i * self.embedding_dim + d;
                    let weight_idx = idx * self.embedding_dim + d;
                    *slot(&mut grad_weight, weight_idx) += fetch(&g_host, out_grad_idx);
                }
            }
            RawTensor::from_shape_compute(out_grad.data.dtype(),
                grad_weight,
                Shape::derived(vec![self.vocab_size, self.embedding_dim]),
            )
        });

        vec![Some(grad_tensor)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-D I64 index tensor.
    fn idx(v: &[i64]) -> Tensor {
        RawTensor::from_typed(v.to_vec(), &[v.len()])
    }

    /// A negative index panics instead of wrapping (PyTorch raises `IndexError`).
    #[test]
    #[should_panic(expected = "Embedding: index -1 out of range for vocab_size 10")]
    fn negative_index_panics() {
        let _ = Embedding::new(10, 4).forward(&idx(&[-1]));
    }

    /// A float index tensor is refused, naming the cast.
    #[test]
    #[should_panic(expected = "Embedding: indices must be I64 or I32, got f32")]
    fn float_indices_panic() {
        let _ = Embedding::new(10, 4).forward(&RawTensor::new(vec![1.0], &[1], false));
    }

    /// Multi-dimensional I32 indices give `[..indices, dim]`, row by row.
    #[test]
    fn i32_indices_of_any_shape() {
        let e = Embedding::new(6, 2);
        let out = e.forward(&RawTensor::from_typed(vec![5i32, 0, 5, 3], &[2, 2]));
        assert_eq!(out.shape(), vec![2, 2, 2]);
        let w = e.weight.data();
        let o = out.data();
        for (k, &row) in [5usize, 0, 5, 3].iter().enumerate() {
            for d in 0..2 {
                assert_eq!(fetch(&o, k * 2 + d), fetch(&w, row * 2 + d));
            }
        }
    }

    #[test]
    fn test_embedding_forward_shape() {
        let embedding = Embedding::new(100, 32);
        let indices = vec![5, 12, 7, 99];
        let output = embedding.forward(&idx(&indices));

        assert_eq!(output.borrow().shape, vec![4, 32]);
    }

    #[test]
    fn test_embedding_values() {
        let embedding = Embedding::new(10, 4);
        let indices = vec![0, 5, 9];
        let output = embedding.forward(&idx(&indices));

        let weight_data = &embedding.weight.data();
        let output_data = &output.data();

        // Check that the first embedding matches the first row of weight
        for i in 0..4 {
            assert_eq!(fetch(output_data, i), fetch(weight_data, i));
        }

        // Check that the second embedding (index 5) matches row 5 of weight
        for i in 0..4 {
            assert_eq!(fetch(output_data, 4 + i), fetch(weight_data, 5 * 4 + i));
        }
    }

    #[test]
    fn test_embedding_backward_flow() {
        let embedding = Embedding::new(10, 4);
        let indices = vec![2, 7, 2]; // Note: repeated index 2
        let output = embedding.forward(&idx(&indices));

        // Sum and backward
        let loss = output.sum();
        loss.into_loss().expect("scalar loss").backward();

        // Check that weight has gradients
        let grad = embedding.weight.grad();
        assert!(grad.is_some(), "Weight should have gradients");

        let grad_data = grad.unwrap();
        // Gradients for index 2 should be accumulated (appears twice)
        // Each embedding contributes 1.0 per dimension (from sum)
        // So index 2 should have grad of 2.0 per dimension
        for d in 0..4 {
            let grad_at_idx2 = fetch(&grad_data, 2 * 4 + d);
            assert!(
                (grad_at_idx2 - 2.0).abs() < 1e-5,
                "Expected grad ~2.0 for repeated index, got {grad_at_idx2}"
            );
        }

        // Indices 7 should have grad of 1.0 per dimension
        for d in 0..4 {
            let grad_at_idx7 = fetch(&grad_data, 7 * 4 + d);
            assert!(
                (grad_at_idx7 - 1.0).abs() < 1e-5,
                "Expected grad ~1.0 for single index, got {grad_at_idx7}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "Embedding: index 15 out of range for vocab_size 10")]
    fn test_embedding_invalid_index() {
        let embedding = Embedding::new(10, 4);
        let indices = vec![5, 15]; // 15 is out of bounds
        let _ = embedding.forward(&idx(&indices));
    }

    #[test]
    fn test_embedding_parameters() {
        let embedding = Embedding::new(50, 16);
        let params = embedding.parameters();

        assert_eq!(params.len(), 1);
        assert_eq!(params.first().unwrap().borrow().shape, vec![50, 16]);
    }

    // No `assert_trainable!` test here: `Module::forward` for `Embedding` is a
    // deliberate panic stub (`Embedding::forward` takes `&[usize]`, not `&Tensor`,
    // so it can't implement the `Module` trait's signature). `assert_trainable!`
    // calls forward through that trait via UFCS, so it would always panic here.
    // `test_embedding_backward_flow` above already checks the exact per-index
    // accumulated gradient values, a strictly stronger check than "nonzero".
}
