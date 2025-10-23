# Volta ⚡

[![Build Status](https://img.shields.io/github/actions/workflow/status/rlarson20/volta/rust.yml?branch=main)](https://github.com/rlarson20/volta/actions)
[![Crates.io](https://img.shields.io/crates/v/volta.svg)](https://crates.io/crates/volta)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Volta is a minimal deep learning and automatic differentiation library built from scratch in pure Rust, heavily inspired by PyTorch. It provides a dynamic computation graph, NumPy-style broadcasting, and common neural network primitives.

This project is an educational endeavor to demystify the inner workings of modern autograd engines. It prioritizes correctness, clarity, and a clean API over raw performance, while still providing hooks for hardware acceleration.

## Key Features

- **Dynamic Computation Graph:** Build and backpropagate through graphs on the fly, just like PyTorch.
- **Reverse-Mode Autodiff:** A powerful `backward()` method for efficient end-to-end gradient calculation.
- **Rich Tensor Operations:** A comprehensive set of unary, binary, reduction, and matrix operations via an ergonomic `TensorOps` trait.
- **NumPy-style Broadcasting:** Sophisticated support for operations on tensors with different shapes.
- **Neural Network Primitives:** High-level `nn::Module` trait with `Linear`, `ReLU`, and `Sequential` layers for building models.
- **Classic Optimizers:** Includes `SGD` (with momentum) and `Adam` for model training.
- **Model Persistence:** Save and load `Linear` layer weights to a compact binary format using `bincode`.
- **BLAS Acceleration (macOS):** Optional performance boost for matrix multiplication via Apple's Accelerate framework.
- **Validation-Focused:** Includes a robust numerical gradient checker to ensure the correctness of all implemented operations.

## Project Status

This library is **training-ready for small to medium-sized feedforward neural networks (MLPs)**. It has a correct and well-tested autograd engine.

- ✅ **What's Working:** Full autograd engine, MLPs, optimizers, `DataLoader`, loss functions, and saving/loading of individual `Linear` layers. The test suite includes over 65 tests validating gradient correctness.
- ⚠️ **What's in Progress:** Performance is not yet a primary focus. While BLAS acceleration is available for macOS matrix multiplication, most operations use naive loops.
- ❌ **What's Missing:**
  - **GPU Support:** Currently CPU-only.
  - **Convolutional Layers:** `Conv2d` is not yet implemented, blocking CNN-based tasks.
  - **Full Model Serialization:** Only individual `Linear` layers can be saved/loaded, not entire `Sequential` models.

## Installation

Add Volta to your `Cargo.toml`:

```toml
[dependencies]
volta = "0.1.0" # Replace with the latest version
```

### Enabling BLAS on macOS

For a significant performance boost in matrix multiplication on macOS, enable the `accelerate` feature:

```toml
[dependencies]
volta = { version = "0.1.0", features = ["accelerate"] }
```

## Quick Start: Training an MLP

Here's how to define a simple Multi-Layer Perceptron (MLP), train it on synthetic data, and save a layer's weights.

```rust
use volta::{nn::*, tensor::*, Adam, Sequential, TensorOps};

fn main() {
    // 1. Define a simple model: 2 -> 8 -> 1
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);

    // 2. Create synthetic data
    let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x = new_tensor::new(x_data, &[4, 2], false);

    let y_data = vec![0.0, 1.0, 1.0, 0.0];
    let y = new_tensor::new(y_data, &[4, 1], false);

    // 3. Set up the optimizer
    let params = model.parameters();
    let mut optimizer = Adam::new(params, 0.1, (0.9, 0.999), 1e-8);

    // 4. Training loop
    println!("Training a simple MLP to learn XOR...");
    for epoch in 0..=100 {
        optimizer.zero_grad();

        let pred = model.forward(&x);
        let loss = mse_loss(&pred, &y);

        if epoch % 20 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss.borrow().data[0]);
        }

        loss.backward();
        optimizer.step();
    }
    println!("Training complete.");

    // 5. Save the weights of the first linear layer
    // Note: We need to downcast the trait object to access the concrete type's methods.
    if let Some(first_layer) = model.layers[0]
        .as_any()
        .downcast_ref::<Linear>()
    {
        println!("\nSaving first layer weights to 'layer1.bin'...");
        first_layer.save("layer1.bin").unwrap();
    }

    // 6. Load the weights into a new layer
    let mut new_layer = Linear::new(2, 8, true);
    println!("Loading weights into a new layer from 'layer1.bin'.");
    new_layer.load("layer1.bin").unwrap();

    // Verify forward pass is the same
    let y1 = model.layers[0].forward(&x);
    let y2 = new_layer.forward(&x);
    assert_eq!(y1.borrow().data, y2.borrow().data);
    println!("Verification successful: Loaded layer produces identical output.");
}
```

## API Overview

The library is designed around a few core concepts:

- **`Tensor`**: The central data structure, an `Rc<RefCell<RawTensor>>`, which holds data, shape, and gradient information. It allows for a mutable, shared structure to build the computation graph.
- **`TensorOps`**: A trait implemented for `Tensor` that provides the ergonomic, user-facing API for all operations (e.g., `tensor.add(&other)`, `tensor.matmul(&weights)`).
- **`nn::Module`**: A trait for building neural network layers (`Linear`, `ReLU`) and composing them into larger models (`Sequential`). It standardizes the `forward()` pass and parameter collection.
- **Optimizers (`Adam`, `SGD`)**: Structures that take a list of model parameters and update their weights based on computed gradients during `step()`.

## Running the Test Suite

Volta has an extensive test suite that validates the correctness of every operation and its gradient. To run the tests:

```bash
cargo test -- --nocapture
```

To run tests with BLAS acceleration enabled (on macOS):

```bash
cargo test --features accelerate -- --nocapture
```

_Note: One test, `misc_tests::test_adam_vs_sgd`, is known to be flaky as it depends on the random seed and convergence speed. It may occasionally fail._

## Roadmap

The next major steps for Volta are focused on expanding its capabilities to handle more complex models and improving performance.

1.  **Full Model Serialization:** Implement `save`/`load` for `Sequential` containers to persist entire models, not just individual layers.
2.  **Vision Support:** Implement `Conv2d` and `MaxPool2d` layers to unlock the ability to build and train Convolutional Neural Networks (CNNs).
3.  **GPU Acceleration:** Integrate a backend for GPU computation (e.g., `wgpu` for cross-platform support or direct `metal` bindings for macOS) to drastically speed up training.
4.  **Performance Optimization:** Implement SIMD for element-wise operations and further integrate optimized BLAS routines.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/volta/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
