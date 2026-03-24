// Character-Level Transformer Language Model
//
// Validates that Volta's transformer components compose correctly under backprop.
// Uses a tiny 1-layer transformer with causal attention on toy text.
// Loss should decrease visibly within 100 epochs.
//
// Architecture: Embedding -> PosEnc -> [MHA + LN -> FFN + LN] -> Linear head
//
// Usage: cargo run --example transformer_lm
//
// TODO: handle these concerns
//
// MultiHeadAttention::forward internally does x.add(&output) — so calling norm1.forward(mha.forward(x)) is Pre-LN applied to the post-residual output, which is slightly non-standard (Post-LN).
// That's fine for a smoke test but worth fixing later by refactoring MHA to not embed the residual.
//
// The positional encoding is added outside the graph (no requires_grad), which is correct for sinusoidal PE but means the model can't learn positional weights.
// If you want learned PE later, PositionalEncoding::new(embed_dim, max_len, PositionalEncodingType::Learned) is wired up.
//
// If loss doesn't move at all, the most likely culprit is the FeedForward::forward reshape — it expects exactly 3D input [batch, seq_len, embed_dim].
// A shape panic there would silently corrupt the graph on some code paths.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use volta::{
    Adam, Embedding, LayerNorm, RawTensor, TensorOps,
    nn::{
        Module,
        transformer::{FeedForward, MultiHeadAttention},
    },
    tensor::cross_entropy_loss,
};

// ===== Vocabulary =====

struct CharVocab {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
}

impl CharVocab {
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        let char_to_idx = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self {
            char_to_idx,
            idx_to_char: chars,
        }
    }

    fn size(&self) -> usize {
        self.idx_to_char.len()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }
}

// ===== Minimal Transformer Block =====
// Note: MultiHeadAttention::forward already adds the residual (x + attn_out).
// We apply LayerNorm after, then FFN + manual residual + LayerNorm.

struct TransformerLayer {
    attn: MultiHeadAttention,
    norm1: LayerNorm,
    ffn: FeedForward,
    norm2: LayerNorm,
}

impl TransformerLayer {
    fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            attn: MultiHeadAttention::new(embed_dim, num_heads, 0.0),
            norm1: LayerNorm::new(vec![embed_dim]),
            ffn: FeedForward::new(embed_dim, 4, 0.0),
            norm2: LayerNorm::new(vec![embed_dim]),
        }
    }

    fn forward(&self, x: &volta::Tensor) -> volta::Tensor {
        // Attention + residual (MHA does this internally) + LayerNorm
        let x = self.norm1.forward(&self.attn.forward(x, true, None));
        // FFN + residual + LayerNorm
        let ffn_out = self.ffn.forward(&x);
        self.norm2.forward(&x.add(&ffn_out))
    }

    fn parameters(&self) -> Vec<volta::Tensor> {
        let mut p = self.attn.parameters();
        p.extend(self.norm1.parameters());
        p.extend(self.ffn.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

// ===== Model =====
#[allow(dead_code)]
struct TinyTransformerLM {
    embedding: Embedding,
    layer: TransformerLayer,
    head: volta::nn::Linear,
    embed_dim: usize,
    vocab_size: usize,
}

impl TinyTransformerLM {
    fn new(vocab_size: usize, embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            layer: TransformerLayer::new(embed_dim, num_heads),
            head: volta::nn::Linear::new(embed_dim, vocab_size, true),
            embed_dim,
            vocab_size,
        }
    }

    fn parameters(&self) -> Vec<volta::Tensor> {
        let mut p = self.embedding.parameters();
        p.extend(self.layer.parameters());
        p.extend(self.head.parameters());
        p
    }

    // Forward pass over a sequence of token indices.
    // Returns logits of shape [seq_len, vocab_size].
    fn forward(&self, indices: &[usize]) -> volta::Tensor {
        let seq_len = indices.len();

        // Embed: [seq_len, embed_dim]
        let emb = self.embedding.forward(indices);

        // PosEnc: sinusoidal, added manually (not tracked, fine for inference quality)
        let emb = self.add_positional_encoding(emb, seq_len);

        // Reshape to [1, seq_len, embed_dim] for transformer layers
        let x = emb.reshape(&[1, seq_len, self.embed_dim]);

        // Transformer layer
        let x = self.layer.forward(&x);

        // Reshape to [seq_len, embed_dim] for head
        let x = x.reshape(&[seq_len, self.embed_dim]);

        // Linear head: [seq_len, vocab_size]
        self.head.forward(&x)
    }

    fn add_positional_encoding(&self, x: volta::Tensor, seq_len: usize) -> volta::Tensor {
        let d = self.embed_dim;
        let mut pe_data = vec![0.0f32; seq_len * d];
        for pos in 0..seq_len {
            for i in 0..(d / 2) {
                let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / d as f32);
                *pe_data.get_mut(pos * d + 2 * i).unwrap() = angle.sin();
                *pe_data.get_mut(pos * d + 2 * i + 1).unwrap() = angle.cos();
            }
        }
        let pe = RawTensor::new(pe_data, &[seq_len, d], false);
        x.add(&pe)
    }
}

// ===== Training =====

fn main() {
    volta::manual_seed(42);

    println!("=== Tiny Transformer LM ===\n");

    let text = "the quick brown fox jumps over the lazy dog. \
                to be or not to be that is the question. \
                all that glitters is not gold.";

    let vocab = CharVocab::from_text(text);
    let data = vocab.encode(text);

    println!("Vocab size:  {}", vocab.size());
    println!("Tokens:      {}", data.len());

    let embed_dim = 32;
    let num_heads = 4;
    let seq_len = 16;
    let epochs = 150;
    let lr = 3e-3;

    let model = TinyTransformerLM::new(vocab.size(), embed_dim, num_heads);
    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.borrow().data.len()).sum();
    println!("Parameters:  {total_params}");
    println!("Embed dim:   {embed_dim}, Heads: {num_heads}, SeqLen: {seq_len}\n");

    let mut optimizer = Adam::new(params, lr, (0.9, 0.999), 1e-8, 0.0);

    println!("Training...\n");

    for epoch in 1..=epochs {
        let num_batches = (data.len() - 1) / seq_len;
        let mut epoch_loss = 0.0;

        for batch in 0..num_batches {
            let start = batch * seq_len;
            let end = start + seq_len;

            let inputs = data.get(start..end).expect("inputs slice out of bounds");
            let targets = data
                .get(start + 1..end + 1)
                .expect("targets slice out of bounds");

            // Forward
            let logits = model.forward(inputs); // [seq_len, vocab_size]

            // Build one-hot targets [seq_len, vocab_size]
            let mut target_data = vec![0.0f32; seq_len * vocab.size()];
            for (t, &idx) in targets.iter().enumerate() {
                *target_data.get_mut(t * vocab.size() + idx).unwrap() = 1.0;
            }
            let target_tensor = RawTensor::new(target_data, &[seq_len, vocab.size()], false);

            let loss = cross_entropy_loss(&logits, &target_tensor);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.borrow().data.first().copied().unwrap_or(f32::NAN);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        if epoch == 1 || epoch % 25 == 0 {
            println!("Epoch {epoch:3}/{epochs}: loss = {avg_loss:.4}");
        }
    }

    println!("\n=== Done ===");
    println!("Loss should have decreased from ~3.8 toward ~2.0 or lower.");
    println!("If not, check that autograd flows through MHA and FFN.");
}
