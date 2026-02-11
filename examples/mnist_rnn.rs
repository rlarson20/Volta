//! MNIST classification using LSTM
//!
//! This example treats each 28x28 image as a sequence of 28 timesteps,
//! where each timestep consists of 28 features (one row of pixels).
//! An LSTM processes the sequence row-by-row, and the final hidden
//! state is used for classification.

use std::path::Path;
use volta::{
    Adam, BatchNorm1d, DataLoader, LSTMCell, Linear, Module, ProgressBar, RawTensor, TensorOps,
    load_mnist_images, load_mnist_labels, to_one_hot,
};

/// Extract a single timestep from a batched sequence tensor.
///
/// Input shape: [batch, `seq_len`, features]
/// Output shape: [batch, features]
fn extract_timestep(batch: &volta::Tensor, t: usize) -> volta::Tensor {
    let borrowed = batch.borrow();
    let batch_size = *borrowed.shape.first().unwrap_or(&1);
    let seq_len = *borrowed.shape.get(1).unwrap_or(&1);
    let features = *borrowed.shape.get(2).unwrap_or(&1);

    assert!(
        t < seq_len,
        "Timestep {t} out of bounds (seq_len={seq_len})"
    );

    let mut data = Vec::with_capacity(batch_size * features);

    for b in 0..batch_size {
        let start = b * seq_len * features + t * features;
        let end = start + features;
        if let Some(slice) = borrowed.data.get(start..end) {
            data.extend_from_slice(slice);
        }
    }

    RawTensor::new(data, &[batch_size, features], false)
}

fn main() {
    println!("=== MNIST RNN (LSTM) Training ===\n");

    // Try to load MNIST data
    let mnist_dir = "data/mnist";
    let train_images_path = format!("{mnist_dir}/train-images-idx3-ubyte");
    let train_labels_path = format!("{mnist_dir}/train-labels-idx1-ubyte");

    let (image_data, label_data, num_samples) = if Path::new(&train_images_path).exists()
        && Path::new(&train_labels_path).exists()
    {
        println!("Loading MNIST data from {mnist_dir}...");

        let images = load_mnist_images(&train_images_path).expect("Failed to load MNIST images");
        let labels = load_mnist_labels(&train_labels_path).expect("Failed to load MNIST labels");

        let num_samples = labels.len();
        let label_data = to_one_hot(&labels, 10);

        println!("Loaded {num_samples} training samples");
        (images, label_data, num_samples)
    } else {
        println!("MNIST data not found. Using synthetic data for demonstration.");
        println!("\nTo download MNIST data:");
        println!("  mkdir -p data/mnist");
        println!("  cd data/mnist");
        println!("  curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
        println!("  curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
        println!("  gunzip *.gz");
        println!();

        // Generate synthetic data
        let num_samples = 1024;
        let image_data: Vec<f32> = (0..num_samples * 28 * 28)
            .map(|i| (i % 255) as f32 / 255.0)
            .collect();
        let labels: Vec<u8> = (0..num_samples).map(|i| (i % 10) as u8).collect();
        let label_data = to_one_hot(&labels, 10);

        println!("Generated {num_samples} synthetic samples");
        (image_data, label_data, num_samples)
    };

    // Create DataLoader
    // Shape: [28, 28] - treating image as 28 timesteps of 28 features
    let batch_size = 64;
    let mut dataloader = DataLoader::new(
        image_data,
        label_data,
        &[28, 28], // [seq_len, features] - each row is a timestep
        &[10],     // 10-class one-hot labels
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
        num_samples / batch_size
    );

    // Training
    let num_epochs = 10;
    let total_batches = num_samples / batch_size;
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
            let mut h_state: Option<(volta::Tensor, volta::Tensor)> = None;

            for t in 0..seq_len {
                // Extract timestep t: [batch, 28]
                let x_t = extract_timestep(&batch_x, t);

                let (h_new, c_new) = match &h_state {
                    Some((h, c)) => lstm.forward_step(&x_t, Some((h, c))),
                    None => lstm.forward_step(&x_t, None),
                };

                h_state = Some((h_new, c_new));
            }

            // Use final hidden state for classification
            let (h_final, _) = h_state.unwrap();

            // BatchNorm + Classification
            let h_norm = bn.forward(&h_final);
            let logits = classifier.forward(&h_norm);

            // Loss and backprop
            let loss = RawTensor::cross_entropy_loss(&logits, &batch_y);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.borrow().data.first().copied().unwrap_or(f32::NAN);
            num_batches += 1;
            progress.update(num_batches);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        println!(" Loss = {avg_loss:.6}");
    }

    // Test accuracy
    println!("\n=== Testing ===");
    bn.train(false);
    dataloader.reset();

    let mut total_correct = 0;
    let mut total_samples = 0;
    let mut progress = ProgressBar::new(total_batches, "Testing");

    for (test_x, test_y) in &mut dataloader {
        // Process sequence through LSTM
        let mut h_state: Option<(volta::Tensor, volta::Tensor)> = None;

        for t in 0..seq_len {
            let x_t = extract_timestep(&test_x, t);

            let (h_new, c_new) = match &h_state {
                Some((h, c)) => lstm.forward_step(&x_t, Some((h, c))),
                None => lstm.forward_step(&x_t, None),
            };

            h_state = Some((h_new, c_new));
        }

        let (h_final, _) = h_state.unwrap();
        let h_norm = bn.forward(&h_final);
        let logits = classifier.forward(&h_norm);

        let pred_data = &logits.borrow().data;
        let target_data = &test_y.borrow().data;
        let batch_size = *test_y.borrow().shape.first().unwrap_or(&1);

        for i in 0..batch_size {
            let pred_class = (0..10)
                .max_by(|&a, &b| {
                    pred_data
                        .get(i * 10 + a)
                        .copied()
                        .unwrap_or(f32::NAN)
                        .partial_cmp(&pred_data.get(i * 10 + b).copied().unwrap_or(f32::NAN))
                        .unwrap()
                })
                .unwrap();

            let true_class = (0..10)
                .position(|j| target_data.get(i * 10 + j).copied().unwrap_or(f32::NAN) > 0.5)
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
    println!("Training Set Accuracy: {total_correct}/{total_samples} ({accuracy:.2}%)");
}
