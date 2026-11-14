use crate::io::StateDict;
use crate::nn::transformer::ScaledDotProductAttention;
use crate::nn::{Dropout, Layer, Linear, Module};
use crate::shape::Axis;
use crate::tensor::Tensor;

/// Multi-Head Attention layer
///
/// Splits the input into multiple heads, applies scaled dot-product attention to each,
/// and concatenates the results. Used in transformer architectures.
///
/// Like `torch.nn.MultiheadAttention`, the layer returns only the projected attention
/// output, `out_proj(Attn(x))`: it applies no layer norm and adds no residual. A
/// transformer block adds `x + Attn(LN(x))` itself (see `TransformerBlock`).
///
/// `forward(x, ..)` is self-attention; [`MultiHeadAttention::forward_qkv`] takes query,
/// key and value separately, for cross-attention.
///
/// Dropout, as in torch, applies to the attention weights, and only in training mode
/// (the default, as for every torch module). Call `Module::train(false)` for inference.
///
/// # Arguments
/// * `embed_dim` - Total dimension of the model
/// * `num_heads` - Number of attention heads
/// * `dropout` - Dropout probability on the attention weights (0.0 = no dropout)
///
/// # Example
/// ```no_run
/// # use volta::nn::transformer::MultiHeadAttention;
/// let mha = MultiHeadAttention::new(512, 8, 0.1);
/// ```
pub struct MultiHeadAttention {
    /// Model width.
    embed_dim: usize,
    /// Number of attention heads.
    num_heads: usize,
    /// Width of one head, `embed_dim / num_heads`.
    head_dim: usize,
    /// Query projection.
    q_proj: Linear,
    /// Key projection.
    k_proj: Linear,
    /// Value projection.
    v_proj: Linear,
    /// Output projection.
    out_proj: Linear,
    /// Dropout on the attention weights (after softmax, before they weight the values),
    /// as in `torch.nn.MultiheadAttention`. Active only in training mode.
    attn_dropout: Dropout,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    ///
    /// # Arguments
    /// * `embed_dim` - Total dimension of the model (must be divisible by `num_heads`)
    /// * `num_heads` - Number of attention heads
    /// * `dropout` - Dropout probability on the attention weights (0.0 = no dropout)
    /// # Panics
    /// `embed_dim` must be divisible by `num_heads`, and `dropout` must be in `[0, 1]`
    #[must_use]
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert!(
            embed_dim.is_multiple_of(num_heads),
            "embed_dim must be divisible by num_heads"
        );

        let head_dim = embed_dim / num_heads;

        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim, false),
            k_proj: Linear::new(embed_dim, embed_dim, false),
            v_proj: Linear::new(embed_dim, embed_dim, false),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            attn_dropout: Dropout::new(dropout),
        }
    }

    /// Forward pass for multi-head self-attention: `forward_qkv(x, x, x, ..)`.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, `seq_len`, `embed_dim`]
    /// * `causal` - Whether to apply causal masking
    /// * `attention_mask` - Optional additive mask, broadcast against the
    ///   [batch, `num_heads`, `seq_len`, `seq_len`] scores
    ///
    /// # Returns
    /// `out_proj(Attn(x))`, of shape [batch, `seq_len`, `embed_dim`], with no residual added.
    /// # Panics
    /// If `x` is not [batch, `seq_len`, `embed_dim`] (see [`MultiHeadAttention::forward_qkv`]).
    #[must_use]
    pub fn forward(&self, x: &Tensor, causal: bool, attention_mask: Option<&Tensor>) -> Tensor {
        self.forward_qkv(x, x, x, causal, attention_mask)
    }

    /// Forward pass with separate query and key/value inputs, as
    /// `torch.nn.MultiheadAttention.forward(query, key, value)`. With `key` and `value`
    /// from another sequence this is cross-attention (`L != S` is allowed).
    ///
    /// # Arguments
    /// * `query` - Tensor of shape [batch, `L`, `embed_dim`]
    /// * `key` - Tensor of shape [batch, `S`, `embed_dim`]
    /// * `value` - Tensor of shape [batch, `S`, `embed_dim`]
    /// * `causal` - Whether to apply causal masking, top-left aligned as torch's
    ///   `is_causal`: query `i` sees keys `j <= i`
    /// * `attention_mask` - Optional additive mask, broadcast against the
    ///   [batch, `num_heads`, `L`, `S`] scores
    ///
    /// # Returns
    /// `out_proj(Attn(query, key, value))`, of shape [batch, `L`, `embed_dim`].
    /// # Panics
    /// If an input is not 3D with last dimension `embed_dim`, the batch sizes differ, or
    /// `key` and `value` differ in length.
    #[must_use]
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal: bool,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let q_shape = query.borrow().shape.clone();
        let k_shape = key.borrow().shape.clone();
        let v_shape = value.borrow().shape.clone();
        for (name, shape) in [("query", &q_shape), ("key", &k_shape), ("value", &v_shape)] {
            assert!(
                shape.rank() == 3 && shape.dim_at(2) == self.embed_dim,
                "{name} must be [batch, seq_len, {}], got {shape:?}",
                self.embed_dim
            );
        }
        let (batch_size, q_len) = (q_shape.dim_at(0), q_shape.dim_at(1));
        let kv_len = k_shape.dim_at(1);
        assert!(
            batch_size == k_shape.dim_at(0) && batch_size == v_shape.dim_at(0),
            "query, key and value must have the same batch size: {q_shape:?}, {k_shape:?}, {v_shape:?}"
        );
        assert_eq!(
            kv_len,
            v_shape.dim_at(1),
            "key and value must have the same sequence length"
        );

        // Project Q, K, V through the Linear layers (2D), then split into heads:
        // [batch, len, embed_dim] -> [batch, num_heads, len, head_dim]
        let q = self.project(&self.q_proj, query, batch_size, q_len);
        let k = self.project(&self.k_proj, key, batch_size, kv_len);
        let v = self.project(&self.v_proj, value, batch_size, kv_len);

        // Apply scaled dot-product attention
        let attn_output = self.apply_attention(&q, &k, &v, causal, attention_mask);

        // Reshape back: [batch, num_heads, L, head_dim] -> [batch, L, embed_dim]
        let attn_output = self.reshape_from_heads(&attn_output, batch_size, q_len);

        // Reshape to 2D for output projection
        let attn_2d = attn_output.reshape(&[batch_size * q_len, self.embed_dim]);
        let output_2d = self.out_proj.forward(&attn_2d);
        output_2d.reshape(&[batch_size, q_len, self.embed_dim])
    }

    /// Applies `proj` to `[batch, len, embed_dim]` and splits the result into heads,
    /// `[batch, num_heads, len, head_dim]`.
    fn project(&self, proj: &Linear, x: &Tensor, batch_size: usize, len: usize) -> Tensor {
        let x_2d = x.reshape(&[batch_size * len, self.embed_dim]);
        let projected = proj
            .forward(&x_2d)
            .reshape(&[batch_size, len, self.embed_dim]);
        self.reshape_for_heads(&projected, batch_size, len)
    }

    /// Splits `[batch, seq_len, embed_dim]` into heads: `[batch, num_heads, seq_len, head_dim]`.
    fn reshape_for_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
        let reshaped = x.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);

        // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        reshaped.permute(&[Axis(0), Axis(2), Axis(1), Axis(3)])
    }

    /// Inverse of `reshape_for_heads`.
    fn reshape_from_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        let permuted = x.permute(&[Axis(0), Axis(2), Axis(1), Axis(3)]);

        // [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, embed_dim]
        permuted.reshape(&[batch_size, seq_len, self.embed_dim])
    }

    /// Scaled dot-product attention over `[batch, num_heads, len, head_dim]` tensors
    /// (query length `L`, key/value length `S`), with an optional causal mask, an
    /// optional additive mask, and dropout on the attention weights in training mode.
    fn apply_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let q_len = q.borrow().shape.dim_at(2); // [batch, num_heads, L, head_dim]
        let kv_len = k.borrow().shape.dim_at(2); // [batch, num_heads, S, head_dim]

        // Compute scaled dot-product attention
        // QK^T: [batch, num_heads, L, head_dim] @ [batch, num_heads, head_dim, S]
        //      = [batch, num_heads, L, S]
        let k_t = k.permute(&[Axis(0), Axis(1), Axis(3), Axis(2)]); // [batch, num_heads, head_dim, S]
        let scores = q.matmul(&k_t); // [batch, num_heads, L, S]

        // Scale by sqrt(head_dim)
        let scale = q.constant_like(vec![(self.head_dim as f32).sqrt()], &[1, 1, 1, 1]);
        let scores = scores.div(&scale);

        // Apply causal mask if requested
        let scores = if causal {
            // [L, S], broadcast over batch and heads.
            let causal_mask = ScaledDotProductAttention::create_causal_mask(q_len, kv_len, q);
            scores.add(&causal_mask)
        } else {
            scores
        };

        // Apply additional attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.add(mask)
        } else {
            scores
        };

        // Apply softmax over the last dimension
        let attention_weights = scores.softmax(Axis(3)); // [batch, num_heads, L, S]

        // Dropout on the attention weights, as torch does (identity in eval mode).
        let attention_weights = self.attn_dropout.forward(&attention_weights);

        // Apply attention weights to values
        attention_weights.matmul(v) // [batch, num_heads, L, head_dim]
    }
}

impl Module for MultiHeadAttention {
    fn parameters(&self) -> Vec<Tensor> {
        // Each Linear's parameters; a projection tied to another is listed once.
        crate::nn::unique_parameters(
            [&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj]
                .into_iter()
                .flat_map(Module::parameters),
        )
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();

        // Get state from each Linear layer
        let q_proj_state = self.q_proj.state_dict();
        for (key, value) in q_proj_state {
            state.insert(format!("q_proj.{key}"), value);
        }

        let k_proj_state = self.k_proj.state_dict();
        for (key, value) in k_proj_state {
            state.insert(format!("k_proj.{key}"), value);
        }

        let v_proj_state = self.v_proj.state_dict();
        for (key, value) in v_proj_state {
            state.insert(format!("v_proj.{key}"), value);
        }

        let out_proj_state = self.out_proj.state_dict();
        for (key, value) in out_proj_state {
            state.insert(format!("out_proj.{key}"), value);
        }

        state
    }

    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
        // Load each projection's weights by extracting prefixed sub-dicts
        let prefixes = [
            ("q_proj.", &mut self.q_proj as &mut Linear),
            ("k_proj.", &mut self.k_proj),
            ("v_proj.", &mut self.v_proj),
            ("out_proj.", &mut self.out_proj),
        ];

        for (prefix, layer) in prefixes {
            let mut sub_state = StateDict::new();
            for (key, value) in state {
                if let Some(sub_key) = key.strip_prefix(prefix) {
                    sub_state.insert(sub_key.to_string(), value.clone());
                }
            }
            if !sub_state.is_empty() {
                layer
                    .load_state_dict(&sub_state)
                    .map_err(|e| e.with_key_prefix(prefix))?;
            }
        }

        Ok(())
    }

    fn train(&mut self, mode: bool) {
        // The projections are Linear, which has no mode; only the dropout does.
        self.attn_dropout.train(mode);
    }
}

impl Layer for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x, false, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;

    #[test]
    fn test_mha_creation() {
        let mha = MultiHeadAttention::new(512, 8, 0.1);
        assert_eq!(mha.embed_dim, 512);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.head_dim, 64);
    }

    #[test]
    fn test_mha_forward_shape() {
        let mha = MultiHeadAttention::new(128, 4, 0.0);

        // Input: [batch=2, seq_len=10, embed_dim=128]
        let x = RawTensor::new(vec![0.0; 2560], &[2, 10, 128], false);
        let output = mha.forward(&x, false, None);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_mha_forward_with_causal() {
        let mha = MultiHeadAttention::new(128, 4, 0.0);

        let x = RawTensor::new(vec![1.0; 2560], &[2, 10, 128], false);
        let output = mha.forward(&x, true, None);

        assert_eq!(output.borrow().shape, vec![2, 10, 128]);
    }

    #[test]
    fn test_mha_assert_trainable() {
        let mha = MultiHeadAttention::new(16, 2, 0.0);
        let x = crate::macros::trainable_input(&[2, 3, 16]);
        crate::assert_trainable!(mha, x);
    }

    /// One gradient check: the input's name, the input, and the loss as a function of it.
    type Check<'a> = (
        &'static str,
        &'a Tensor,
        Box<dyn Fn(&Tensor) -> Tensor + 'a>,
    );

    /// Seeded normal values as an f64 tensor, a leaf that requires grad when `grad`.
    /// The checks run in f64 so that finite-difference rounding (about 1e-7 relative
    /// in f32) can't mask or fake a mismatch.
    fn operand(dims: &[usize], grad: bool) -> Tensor {
        let vals: Vec<f64> = crate::tensor::randn(dims)
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let t = RawTensor::from_typed(vals, dims);
        t.set_requires_grad(grad);
        t
    }

    /// Finite-difference check of query, key and value through the projections, with
    /// `L = 2` queries attending over `S = 3` keys, under a randomly weighted sum.
    fn assert_mha_cross_attention_gradients(causal: bool) {
        crate::tensor::manual_seed(0x5EED);
        let mut mha = MultiHeadAttention::new(4, 2, 0.0);
        mha.to_dtype(crate::DType::F64);
        let q = operand(&[2, 2, 4], true);
        let k = operand(&[2, 3, 4], true);
        let v = operand(&[2, 3, 4], true);
        let w = operand(&[2, 2, 4], false);

        let out = mha.forward_qkv(&q, &k, &v, causal, None);
        assert_eq!(out.shape(), vec![2, 2, 4]);

        let checks: [Check<'_>; 3] = [
            (
                "query",
                &q,
                Box::new(|t| mha.forward_qkv(t, &k, &v, causal, None).elem_mul(&w).sum()),
            ),
            (
                "key",
                &k,
                Box::new(|t| mha.forward_qkv(&q, t, &v, causal, None).elem_mul(&w).sum()),
            ),
            (
                "value",
                &v,
                Box::new(|t| mha.forward_qkv(&q, &k, t, causal, None).elem_mul(&w).sum()),
            ),
        ];
        for (name, input, loss_fn) in checks {
            let (max, mean, ok) = RawTensor::check_gradients(input, loss_fn, 1e-6, 1e-6);
            assert!(
                ok,
                "MHA cross-attention grad wrt {name} (causal={causal}) failed: max={max:.3e} mean={mean:.3e}"
            );
        }
    }

    #[test]
    fn test_mha_cross_attention_gradients() {
        assert_mha_cross_attention_gradients(false);
    }

    #[test]
    fn test_mha_cross_attention_causal_gradients() {
        assert_mha_cross_attention_gradients(true);
    }

    #[test]
    fn test_mha_self_attention_is_forward_qkv_of_x() {
        let mha = MultiHeadAttention::new(8, 2, 0.0);
        let x = crate::macros::trainable_input(&[2, 3, 8]);
        assert_eq!(
            mha.forward(&x, true, None).data(),
            mha.forward_qkv(&x, &x, &x, true, None).data()
        );
    }

    #[test]
    fn test_mha_dropout_identity_in_eval_active_in_training() {
        crate::tensor::manual_seed(22);
        let mut mha = MultiHeadAttention::new(8, 2, 0.5);
        // Same weights, no dropout: what the layer must compute in eval mode.
        let mut reference = MultiHeadAttention::new(8, 2, 0.0);
        reference
            .load_state_dict(&mha.state_dict())
            .expect("same architecture");
        let x = crate::tensor::randn(&[2, 4, 8]);
        let expected = reference.forward(&x, false, None).data();

        mha.train(false);
        assert_eq!(
            mha.forward(&x, false, None).data(),
            expected,
            "eval mode must not drop"
        );

        mha.train(true);
        let trained = mha.forward(&x, false, None).data();
        assert_eq!(trained.len(), expected.len());
        assert!(
            trained
                .iter()
                .zip(expected.iter())
                .any(|(a, e)| (a - e).abs() > 1e-6),
            "training mode at p=0.5 left the output unchanged"
        );

        mha.train(false);
        assert_eq!(mha.forward(&x, false, None).data(), expected);
    }
}
