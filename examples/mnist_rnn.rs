//! MNIST classification using LSTM
//!
//! This example treats each 28x28 image as a sequence of 28 timesteps,
//! where each timestep consists of 28 features (one row of pixels).
//! An LSTM processes the sequence row-by-row, and the final hidden
//! state is used for classification.
//!
//! Trains on the 60,000-image `train` split and reports accuracy on the held-out 10,000-image `t10k` split.
//!
//! Needs the MNIST files under `data/mnist`; if they are missing it prints how to download them and exits, rather than train on made-up data.
#[path = "common/mnist.rs"]
mod mnist;
use volta::nn::{Adam, BatchNorm1d, LSTMCell, Layer, Linear, Module, RecurrentLayer};
use volta::utils::ProgressBar;
use volta::{DataLoader, RawTensor};

/// Per-sample input shape: `[seq_len, features]`, each pixel row is a timestep.
const SAMPLE_SHAPE: &[usize] = &[28, 28];

/// Extract a single timestep from a batched sequence tensor.
///
/// Input shape: [batch, `seq_len`, features]
/// Output shape: [batch, features]
fn extract_timestep(batch: &volta::Tensor, t: usize) -> volta::Tensor {
    let shape = batch.shape();
    let &[batch_size, seq_len, features] = shape.truncate_rank() else {
        panic!("extract_timestep expects [batch, seq_len, features], got {shape:?}");
    };

    assert!(
        t < seq_len,
        "Timestep {t} out of bounds (seq_len={seq_len})"
    );

    let mut data = Vec::with_capacity(batch_size * features);
    let batch_data = batch.data();

    for b in 0..batch_size {
        let start = b * seq_len * features + t * features;
        let end = start + features;
        data.extend_from_slice(
            batch_data
                .get(start..end)
                .expect("timestep slice lies within the batch"),
        );
    }

    RawTensor::new(data, &[batch_size, features], false)
}

fn main() {
    println!("=== MNIST RNN (LSTM) Training ===\n");

    let mnist_dir = "data/mnist";
    println!("Loading MNIST data from {mnist_dir}...");
    let train = mnist::load_split(mnist_dir, "train");
    // Held-out set: accuracy is measured on these 10,000 images, never trained on.
    let test = mnist::load_split(mnist_dir, "t10k");
    let num_samples = train.len;
    println!(
        "Loaded {num_samples} training and {} test samples",
        test.len
    );

    // Create DataLoader
    // Shape: [28, 28] - treating image as 28 timesteps of 28 features
    let batch_size = 64;
    let mut dataloader = DataLoader::new(
        train.images,
        train.labels,
        SAMPLE_SHAPE,
        &[10], // 10-class one-hot labels
        batch_size,
        true, // shuffle
    );

    // Build model components
    let input_size = 28; // Features per timestep (pixels per row)
    let hidden_size = 128; // LSTM hidden dimension
    let num_classes = 10;

    let lstm = LSTMCell::new(input_size, hidden_size, true);
    let mut bn = BatchNorm1d::new(hidden_size);
    let classifier = Linear::new(hidden_size, num_classes, true);

    // Collect all parameters
    let mut params = lstm.parameters();
    params.extend(bn.parameters());
    params.extend(classifier.parameters());

    let mut optimizer = Adam::new(params, 0.001, (0.9, 0.999), 1e-8, 0.0);

    println!("\nArchitecture:");
    println!("  Input: 28×28 image → 28 timesteps of 28 features");
    println!("  LSTM({input_size} → {hidden_size} hidden)");
    println!("  BatchNorm1d({hidden_size})");
    println!("  Linear({hidden_size} → {num_classes})");
    println!(
        "\nTraining with {} batches per epoch...\n",
        num_samples.div_ceil(batch_size)
    );

    // Training
    let num_epochs = 10;
    // The loader yields the last, partial batch too.
    let total_batches = num_samples.div_ceil(batch_size);
    let seq_len = 28;

    for epoch in 0..num_epochs {
        dataloader.reset();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        bn.train(true);

        let mut progress = ProgressBar::new(
            total_batches,
            &format!("Epoch {:2}/{}", epoch + 1, num_epochs),
        );

        for (batch_x, batch_y) in &mut dataloader {
            optimizer.zero_grad();

            // Process sequence through LSTM
            // batch_x shape: [batch, 28, 28]
            let batch = batch_x.shape().dim(volta::Axis(0)).expect("batch dim");
            let mut state = lstm.initial_state(batch);

            for t in 0..seq_len {
                // Extract timestep t: [batch, 28]
                let x_t = extract_timestep(&batch_x, t);
                let _ = lstm.step(&x_t, &mut state);
            }

            // Use final hidden state for classification
            let (h_final, _) = state;

            // BatchNorm + Classification
            let h_norm = bn.forward(&h_final);
            let logits = classifier.forward(&h_norm);

            // Loss and backprop
            let loss = RawTensor::cross_entropy_loss(&logits, &batch_y);
            loss.clone().into_loss().expect("scalar loss").backward();
            optimizer.step();

            epoch_loss += loss.item();
            num_batches += 1;
            progress.update(num_batches);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        println!(" Loss = {avg_loss:.6}");
    }

    // Test accuracy
    println!("\n=== Testing ===");
    bn.train(false);

    // A separate, unshuffled loader over the t10k split.
    let mut test_loader = DataLoader::new(
        test.images,
        test.labels,
        SAMPLE_SHAPE,
        &[10],
        batch_size,
        false,
    );
    let mut total_correct = 0;
    let mut total_samples = 0;
    let mut progress = ProgressBar::new(test.len.div_ceil(batch_size), "Testing");

    for (test_x, test_y) in &mut test_loader {
        // Process sequence through LSTM
        let batch = test_x.shape().dim(volta::Axis(0)).expect("batch dim");
        let mut state = lstm.initial_state(batch);

        for t in 0..seq_len {
            let x_t = extract_timestep(&test_x, t);
            let _ = lstm.step(&x_t, &mut state);
        }

        let (h_final, _) = state;
        let h_norm = bn.forward(&h_final);
        let logits = classifier.forward(&h_norm);

        let pred_data = logits.data();
        let target_data = test_y.data();
        let batch_size = test_y
            .shape()
            .dim(volta::Axis(0))
            .expect("test labels have a batch dim");

        for i in 0..batch_size {
            let pred_class = (0..10)
                .max_by(|&a, &b| {
                    pred_data
                        .get(i * 10 + a)
                        .copied()
                        .expect("index in range")
                        .partial_cmp(&pred_data.get(i * 10 + b).copied().expect("index in range"))
                        .unwrap()
                })
                .unwrap();

            let true_class = (0..10)
                .position(|j| {
                    target_data
                        .get(i * 10 + j)
                        .copied()
                        .expect("index in range")
                        > 0.5
                })
                .unwrap();

            if pred_class == true_class {
                total_correct += 1;
            }
            total_samples += 1;
        }
        progress.inc();
    }
    println!();

    let accuracy = total_correct as f32 / total_samples as f32 * 100.0;
    assert_eq!(total_samples, test.len, "every test sample is scored once");
    println!("Test Accuracy (t10k): {total_correct}/{total_samples} ({accuracy:.2}%)");
}
