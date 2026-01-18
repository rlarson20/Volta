// Super Resolution using Efficient Sub-Pixel Convolution
//
// This example demonstrates the PixelShuffle layer for image upscaling,
// implementing the ESPCN architecture from:
// "Real-Time Single Image and Video Super-Resolution Using an Efficient
// Sub-Pixel Convolutional Neural Network" (Shi et al., 2016)
//
// Since Volta doesn't have image I/O yet, we use synthetic data to demonstrate
// the architecture and training process.

use volta::{
    Adam, Conv2d, Module, PixelShuffle, RawTensor, ReLU, Sequential, TensorOps, manual_seed,
    mse_loss,
};

fn main() {
    manual_seed(42);

    println!("=== Super Resolution with PixelShuffle ===\n");

    // Hyperparameters
    let upscale_factor = 2; // 2x upscaling
    let batch_size = 4;
    let low_res_size = 8; // 8x8 low-resolution input
    let high_res_size = low_res_size * upscale_factor; // 16x16 high-resolution target
    let learning_rate = 0.001;
    let num_epochs = 100;

    println!("Configuration:");
    println!("  Upscale factor: {}x", upscale_factor);
    println!("  Low-res size: {}x{}", low_res_size, low_res_size);
    println!("  High-res size: {}x{}", high_res_size, high_res_size);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}\n", num_epochs);

    // Build model: ESPCN architecture
    // Conv layers with increasing then decreasing channels, final layer outputs r² channels
    let model = Sequential::new(vec![
        Box::new(Conv2d::new(1, 64, 5, 1, 2, true)), // (1, 64, kernel=5, padding=2)
        Box::new(ReLU),
        Box::new(Conv2d::new(64, 64, 3, 1, 1, true)), // (64, 64, kernel=3, padding=1)
        Box::new(ReLU),
        Box::new(Conv2d::new(64, 32, 3, 1, 1, true)), // (64, 32, kernel=3, padding=1)
        Box::new(ReLU),
        Box::new(Conv2d::new(
            32,
            upscale_factor * upscale_factor,
            3,
            1,
            1,
            true,
        )), // (32, r², kernel=3, padding=1)
        Box::new(PixelShuffle::new(upscale_factor)), // Rearrange to high-res
    ]);

    println!("Model architecture:");
    println!("  Conv2d(1 → 64, kernel=5x5, padding=2) + ReLU");
    println!("  Conv2d(64 → 64, kernel=3x3, padding=1) + ReLU");
    println!("  Conv2d(64 → 32, kernel=3x3, padding=1) + ReLU");
    println!(
        "  Conv2d(32 → {}, kernel=3x3, padding=1)",
        upscale_factor * upscale_factor
    );
    println!("  PixelShuffle(upscale_factor={})", upscale_factor);
    println!();

    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.borrow().data.len()).sum();
    println!("Total parameters: {}\n", total_params);

    // Create optimizer (lr, betas, eps, weight_decay)
    let mut optimizer = Adam::new(params, learning_rate, (0.9, 0.999), 1e-8, 0.0);

    println!("Training...\n");

    // Training loop
    for epoch in 1..=num_epochs {
        // Generate synthetic training data
        // Low-res: smooth gradients, High-res: sharper version with more detail
        let (low_res, high_res) = generate_synthetic_data(batch_size, low_res_size, high_res_size);

        // Forward pass
        let output = model.forward(&low_res);

        // Compute MSE loss
        let loss_tensor = mse_loss(&output, &high_res);
        let loss_value = loss_tensor
            .borrow()
            .data
            .first()
            .copied()
            .unwrap_or(f32::NAN);

        // Backward pass
        optimizer.zero_grad();
        loss_tensor.backward();
        optimizer.step();

        // Print progress
        if epoch % 10 == 0 || epoch == 1 {
            println!("Epoch {:3}/{}: Loss = {:.6}", epoch, num_epochs, loss_value);
        }
    }

    println!("\n=== Training Complete ===\n");

    // Demonstrate upscaling on a test pattern
    println!("Testing on a simple checkerboard pattern...");
    let test_input = create_test_pattern(1, low_res_size);
    let test_output = model.forward(&test_input);

    println!("  Input shape:  {:?}", test_input.borrow().shape);
    println!("  Output shape: {:?}", test_output.borrow().shape);

    // Show a slice of the output to verify reasonable values
    let out_data = &test_output.borrow().data;
    println!(
        "  Output range: [{:.3}, {:.3}]",
        out_data.iter().cloned().fold(f32::INFINITY, f32::min),
        out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    println!("\n✓ Successfully demonstrated super-resolution with PixelShuffle!");
}

/// Generate synthetic training data
///
/// Creates smooth low-resolution images and corresponding high-resolution targets
/// with added detail/frequency content to simulate the super-resolution task
fn generate_synthetic_data(
    batch_size: usize,
    low_res_size: usize,
    high_res_size: usize,
) -> (volta::Tensor, volta::Tensor) {
    // Low-res: smooth patterns (e.g., gentle gradients, blobs)
    let low_res = RawTensor::rand(&[batch_size, 1, low_res_size, low_res_size]);

    // High-res: start with upsampled version of low-res (via simple repetition),
    // then add high-frequency detail
    let lr_data = low_res.borrow().data.to_vec();
    let mut hr_data = Vec::with_capacity(batch_size * high_res_size * high_res_size);

    let scale = high_res_size / low_res_size;

    for b in 0..batch_size {
        for y in 0..high_res_size {
            for x in 0..high_res_size {
                // Map high-res coordinate back to low-res
                let lr_y = y / scale;
                let lr_x = x / scale;
                let lr_idx = b * low_res_size * low_res_size + lr_y * low_res_size + lr_x;
                let base_value = lr_data.get(lr_idx).copied().unwrap_or(0.0);

                // Add some high-frequency detail using a deterministic pattern
                // This simulates texture/edges that should be learned
                let detail = ((x as f32 * 0.5 + y as f32 * 0.3).sin() * 0.05).abs();
                hr_data.push(base_value + detail);
            }
        }
    }

    let high_res = RawTensor::new(
        hr_data,
        &[batch_size, 1, high_res_size, high_res_size],
        false,
    );

    (low_res, high_res)
}

/// Create a test checkerboard pattern for visualization
fn create_test_pattern(batch_size: usize, size: usize) -> volta::Tensor {
    let mut data = Vec::with_capacity(batch_size * size * size);

    for _b in 0..batch_size {
        for y in 0..size {
            for x in 0..size {
                // Simple checkerboard: alternating 0.2 and 0.8
                let value = if (x + y) % 2 == 0 { 0.2 } else { 0.8 };
                data.push(value);
            }
        }
    }

    RawTensor::new(data, &[batch_size, 1, size, size], false)
}
