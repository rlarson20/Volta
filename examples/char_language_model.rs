// Character-Level Language Model using LSTM
//
// This example demonstrates:
// - Embedding layer for character-to-vector mapping
// - LSTM for sequential processing
// - Character-level text generation
//
// The model predicts the next character in a sequence, learning patterns
// from a small text corpus.

use std::collections::HashMap;
use volta::{
    Adam, Dropout, Embedding, LSTMCell, Linear, Module, RawTensor, TensorOps, manual_seed,
};

/// Vocabulary for character-level language modeling
struct CharVocab {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
}

impl CharVocab {
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();

        let mut char_to_idx = HashMap::new();
        for (idx, &ch) in chars.iter().enumerate() {
            char_to_idx.insert(ch, idx);
        }

        CharVocab {
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

    fn encode_char(&self, c: char) -> usize {
        *self.char_to_idx.get(&c).unwrap()
    }

    fn decode_char(&self, idx: usize) -> char {
        *self.idx_to_char.get(idx).unwrap_or(&' ')
    }
}

/// Character-level RNN model
struct CharRNN {
    embedding: Embedding,
    lstm: LSTMCell,
    dropout: Dropout,
    decoder: Linear,
}

impl CharRNN {
    fn new(vocab_size: usize, emb_dim: usize, hidden_size: usize, dropout_p: f32) -> Self {
        CharRNN {
            embedding: Embedding::new(vocab_size, emb_dim),
            lstm: LSTMCell::new(emb_dim, hidden_size, true),
            dropout: Dropout::new(dropout_p),
            decoder: Linear::new(hidden_size, vocab_size, true),
        }
    }

    fn parameters(&self) -> Vec<volta::Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        params.extend(self.lstm.parameters());
        params.extend(self.decoder.parameters());
        params
    }

    fn train_mode(&mut self, mode: bool) {
        self.dropout.train(mode);
    }
}

/// Forward pass and compute loss for a sequence
fn forward_and_compute_loss(
    model: &CharRNN,
    inputs: &[usize],
    targets: &[usize],
    vocab_size: usize,
) -> volta::Tensor {
    let mut h_state: Option<(volta::Tensor, volta::Tensor)> = None;
    let mut all_logits = Vec::new();
    let mut all_targets = Vec::new();

    for (&input_idx, &target_idx) in inputs.iter().zip(targets) {
        // Embedding lookup
        let emb = model.embedding.forward(&[input_idx]);
        let emb_drop = model.dropout.forward(&emb);

        // LSTM step
        let (h_new, c_new) = model
            .lstm
            .forward_step(&emb_drop, h_state.as_ref().map(|(h, c)| (h, c)));

        // Decode
        let h_drop = model.dropout.forward(&h_new);
        let logits = model.decoder.forward(&h_drop);

        all_logits.push(logits);
        all_targets.push(target_idx);

        h_state = Some((h_new, c_new));
    }

    // Compute loss by accumulating cross-entropy for each timestep
    let mut total_loss: Option<volta::Tensor> = None;
    for (logits, &target_idx) in all_logits.iter().zip(&all_targets) {
        // Create one-hot target
        let mut target_one_hot = vec![0.0; vocab_size];
        *target_one_hot.get_mut(target_idx).unwrap_or(&mut 0.0) = 1.0;
        let target_tensor = RawTensor::new(target_one_hot, &[1, vocab_size], false);

        // Cross entropy loss
        let loss = volta::cross_entropy_loss(logits, &target_tensor);

        // Accumulate loss tensors (not scalars!) to maintain gradient flow
        total_loss = Some(match total_loss {
            None => loss,
            Some(prev) => prev.add(&loss),
        });
    }

    // Return average loss (divide by sequence length)
    let loss = total_loss.unwrap();
    let seq_len_tensor = RawTensor::new(vec![all_targets.len() as f32], &[1], false);
    loss.div(&seq_len_tensor)
}

/// Generate text from the model
fn generate(
    model: &mut CharRNN,
    vocab: &CharVocab,
    seed: &str,
    length: usize,
    temperature: f32,
) -> String {
    model.train_mode(false); // Disable dropout for generation

    let mut result = seed.to_string();
    let mut h_state: Option<(volta::Tensor, volta::Tensor)> = None;

    // Prime with seed text
    for c in seed.chars() {
        if let Some(&idx) = vocab.char_to_idx.get(&c) {
            let emb = model.embedding.forward(&[idx]);
            let (h_new, c_new) = model
                .lstm
                .forward_step(&emb, h_state.as_ref().map(|(h, c)| (h, c)));
            h_state = Some((h_new, c_new));
        }
    }

    // Generate new characters
    let mut current_idx = vocab.encode_char(seed.chars().last().unwrap_or('a'));

    for _ in 0..length {
        let emb = model.embedding.forward(&[current_idx]);
        let (h_new, c_new) = model
            .lstm
            .forward_step(&emb, h_state.as_ref().map(|(h, c)| (h, c)));
        let logits = model.decoder.forward(&h_new);

        // Apply temperature and sample
        let logits_data = &logits.borrow().data;
        let scaled: Vec<f32> = logits_data.iter().map(|&x| x / temperature).collect();

        // Simple argmax for deterministic generation
        let mut max_idx = 0;
        let mut max_val = *scaled.first().unwrap_or(&f32::NAN);
        for (i, &val) in scaled.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        current_idx = max_idx;
        result.push(vocab.decode_char(current_idx));
        h_state = Some((h_new, c_new));
    }

    model.train_mode(true); // Re-enable dropout
    result
}

fn main() {
    manual_seed(42);

    println!("=== Character Language Model ===\n");

    // Training data: Simple English text
    let text = "the quick brown fox jumps over the lazy dog. \
                 a journey of a thousand miles begins with a single step. \
                 to be or not to be, that is the question. \
                 all that glitters is not gold. \
                 where there is a will, there is a way.";

    // Build vocabulary
    let vocab = CharVocab::from_text(text);
    let data = vocab.encode(text);

    println!("Vocabulary size: {}", vocab.size());
    println!("Training data length: {} characters", data.len());
    println!(
        "Sample vocabulary: {:?}\n",
        &vocab.idx_to_char.get(..10.min(vocab.size())).unwrap_or(&[])
    );

    // Hyperparameters
    let emb_dim = 16;
    let hidden_size = 32;
    let seq_len = 20; // BPTT sequence length
    let epochs = 100;
    let learning_rate = 0.01;
    let dropout_p = 0.1;

    println!("Hyperparameters:");
    println!("  Embedding dimension: {}", emb_dim);
    println!("  Hidden size: {}", hidden_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", learning_rate);
    println!("  Dropout: {}\n", dropout_p);

    // Create model
    let mut model = CharRNN::new(vocab.size(), emb_dim, hidden_size, dropout_p);
    model.train_mode(true);

    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.borrow().data.len()).sum();
    println!("Total parameters: {}\n", total_params);

    // Create optimizer
    let mut optimizer = Adam::new(params, learning_rate, (0.9, 0.999), 1e-8, 0.0);

    println!("Training...\n");

    // Training loop
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let num_batches = (data.len() - 1) / seq_len;

        if num_batches == 0 {
            eprintln!("Not enough data for training");
            return;
        }

        for batch_idx in 0..num_batches {
            let start = batch_idx * seq_len;
            let end = (start + seq_len).min(data.len() - 1);

            if end <= start {
                continue;
            }

            let inputs = data.get(start..end).unwrap_or(&[]);
            let targets = data.get(start + 1..end + 1).unwrap_or(&[]);

            // Forward pass
            let loss = forward_and_compute_loss(&model, inputs, targets, vocab.size());

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.borrow().data.first().copied().unwrap_or(f32::NAN);
        }

        let avg_loss = epoch_loss / num_batches as f32;

        if epoch % 10 == 0 || epoch == 1 {
            println!("Epoch {:3}/{}: Loss = {:.4}", epoch, epochs, avg_loss);
        }
    }

    println!("\n=== Training Complete ===\n");

    // Generate text samples
    println!("Generated text samples:\n");

    let seeds = vec!["the ", "a ", "to "];
    for seed in seeds {
        let generated = generate(&mut model, &vocab, seed, 40, 1.0);
        println!("Seed: {:?}", seed);
        println!("Generated: {}\n", generated);
    }

    println!("✓ Successfully trained character language model!");
}
