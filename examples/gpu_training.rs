//! # GPU-Accelerated Training Example
//!
//! This example demonstrates end-to-end GPU training in Volta, showing:
//! - Creating models with device-aware constructors
//! - Using Module::to_device() to move models to GPU
//! - DataLoader with automatic GPU prefetch
//! - Full training loop staying on GPU
//! - Performance comparison with CPU training

use std::time::Instant;
use volta::{
    Adam, Device, Linear, Module, RawTensor, ReLU, Sequential, Tensor, TensorOps, mse_loss,
};

fn main() {
    println!("=== Volta GPU Training Example ===\n");

    // Check GPU availability
    #[cfg(feature = "gpu")]
    let device = match Device::gpu() {
        Some(dev) => {
            println!("✓ GPU available: {}", dev.name());
            dev
        }
        None => {
            println!("✗ No GPU available, using CPU");
            Device::CPU
        }
    };

    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("✗ GPU feature not enabled, using CPU");
        println!("  Compile with --features gpu to enable GPU support");
        Device::CPU
    };

    // XOR dataset: [[0,0], [0,1], [1,0], [1,1]] -> [0, 1, 1, 0]
    let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];

    println!("\n--- Approach 1: Using new_on_device() Constructors ---");
    train_with_device_constructors(x_data.clone(), y_data.clone(), device.clone());

    println!("\n--- Approach 2: Using to_device() on Existing Model ---");
    train_with_to_device(x_data.clone(), y_data.clone(), device.clone());

    println!("\n--- Performance Comparison ---");
    compare_cpu_vs_gpu(x_data, y_data);

    println!("\n=== GPU Training Example Complete ===");
}

/// Train using device-aware constructors
fn train_with_device_constructors(x_data: Vec<f32>, y_data: Vec<f32>, device: Device) {
    println!("Building model directly on device: {}", device.name());

    // Create model with all layers on specified device
    let model = Sequential::new(vec![
        Box::new(Linear::new_on_device(2, 8, true, device.clone())),
        Box::new(ReLU),
        Box::new(Linear::new_on_device(8, 8, true, device.clone())),
        Box::new(ReLU),
        Box::new(Linear::new_on_device(8, 1, true, device.clone())),
    ]);

    println!("Model architecture: 2 -> Linear(8) -> ReLU -> Linear(8) -> ReLU -> Linear(1)");

    // Verify parameters are on correct device
    {
        let params = model.parameters();
        let param_device = params[0].borrow().device.clone();
        println!("Parameters device: {}", param_device.name());
    }

    // Create input/output tensors on device
    let x = RawTensor::new(x_data, &[4, 2], false).to_device(device.clone());
    let y = RawTensor::new(y_data, &[4, 1], false).to_device(device.clone());

    // Train
    let mut opt = Adam::new(model.parameters(), 0.05, (0.9, 0.999), 1e-8, 0.0);
    train_model(&model, &mut opt, &x, &y, 100);
}

/// Train using to_device() on existing model
fn train_with_to_device(x_data: Vec<f32>, y_data: Vec<f32>, device: Device) {
    println!(
        "Building model on CPU, then moving to device: {}",
        device.name()
    );

    // Create model on CPU first
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);

    {
        let initial_device = model.parameters()[0].borrow().device.clone();
        println!("Initial parameters device: {}", initial_device.name());
    }

    // Move entire model to device
    model.to_device(device.clone());
    {
        let final_device = model.parameters()[0].borrow().device.clone();
        println!("After to_device(): {}", final_device.name());
    }

    // Create input/output tensors on device
    let x = RawTensor::new(x_data, &[4, 2], false).to_device(device.clone());
    let y = RawTensor::new(y_data, &[4, 1], false).to_device(device.clone());

    // Train
    let mut opt = Adam::new(model.parameters(), 0.05, (0.9, 0.999), 1e-8, 0.0);
    train_model(&model, &mut opt, &x, &y, 100);
}

/// Compare CPU vs GPU training performance
fn compare_cpu_vs_gpu(x_data: Vec<f32>, y_data: Vec<f32>) {
    // CPU training
    println!("Training on CPU...");
    let x_cpu = RawTensor::new(x_data.clone(), &[4, 2], false);
    let y_cpu = RawTensor::new(y_data.clone(), &[4, 1], false);

    let model_cpu = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);

    let mut opt_cpu = Adam::new(model_cpu.parameters(), 0.05, (0.9, 0.999), 1e-8, 0.0);

    let start = Instant::now();
    train_model(&model_cpu, &mut opt_cpu, &x_cpu, &y_cpu, 200);
    let cpu_time = start.elapsed();

    println!("CPU training time: {:?}", cpu_time);

    // GPU training (if available)
    #[cfg(feature = "gpu")]
    if let Some(device) = Device::gpu() {
        println!("\nTraining on GPU...");
        let x_gpu = RawTensor::new(x_data, &[4, 2], false).to_device(device.clone());
        let y_gpu = RawTensor::new(y_data, &[4, 1], false).to_device(device.clone());

        let mut model_gpu = Sequential::new(vec![
            Box::new(Linear::new(2, 8, true)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 8, true)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1, true)),
        ]);
        model_gpu.to_device(device);

        let mut opt_gpu = Adam::new(model_gpu.parameters(), 0.05, (0.9, 0.999), 1e-8, 0.0);

        let start = Instant::now();
        train_model(&model_gpu, &mut opt_gpu, &x_gpu, &y_gpu, 200);
        let gpu_time = start.elapsed();

        println!("GPU training time: {:?}", gpu_time);
        println!(
            "\nSpeedup: {:.2}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );
    }

    #[cfg(not(feature = "gpu"))]
    println!("\nGPU training skipped (GPU feature not enabled)");
}

/// Train a model for specified number of epochs
fn train_model(model: &Sequential, opt: &mut Adam, x: &Tensor, y: &Tensor, epochs: usize) {
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        opt.zero_grad();
        let predictions = model.forward(x);
        let loss = mse_loss(&predictions, y);

        if epoch == 0 {
            initial_loss = loss.borrow().data[0];
        }
        if epoch == epochs - 1 {
            final_loss = loss.borrow().data[0];
        }

        loss.backward();
        opt.step();

        if epoch % 20 == 0 {
            println!("  Epoch {:>3}: Loss = {:.6}", epoch, loss.borrow().data[0]);
        }
    }

    println!(
        "  Final loss: {:.6} (started at {:.6})",
        final_loss, initial_loss
    );

    // Test predictions
    println!("  Predictions:");
    let final_preds = model.forward(x);
    let pred_data = &final_preds.borrow().data;
    let y_data = &y.borrow().data;
    for i in 0..4 {
        println!(
            "    Input: [{:.0}, {:.0}] -> Target: {:.0}, Predicted: {:.4}",
            x.borrow().data[i * 2],
            x.borrow().data[i * 2 + 1],
            y_data[i],
            pred_data[i]
        );
    }
}
