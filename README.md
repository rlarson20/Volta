# Volta ⚡

<small>A PyTorch-like deep learning framework in pure Rust</small>

[![Build Status](https://img.shields.io/github/actions/workflow/status/rlarson20/volta/ci.yml?branch=main)](https://github.com/rlarson20/volta/actions)
[![Crates.io](https://img.shields.io/crates/v/volta.svg)](https://crates.io/crates/volta)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Volta is a minimal deep learning and automatic differentiation library built from scratch in pure Rust, heavily inspired by PyTorch. It provides a dynamic computation graph, NumPy-style broadcasting, and common neural network primitives.

This project is an educational endeavor to demystify the inner workings of modern autograd engines. It prioritizes correctness, clarity, and a clean API over raw performance, while still providing hooks for hardware acceleration.

## Key Features

- **Dynamic Computation Graph:** Build and backpropagate through graphs on the fly, just like PyTorch.
- **Reverse-Mode Autodiff:** Efficient reverse-mode automatic differentiation with topological sorting.
- **Rich Tensor Operations:** A comprehensive set of unary, binary, reduction, and matrix operations via an ergonomic `TensorOps` trait.
- **Broadcasting:** Full NumPy-style broadcasting support for arithmetic operations.
- **Neural Network Layers:** `Linear`, `Conv2d`, `MaxPool2d`, `Flatten`, `ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `BatchNorm2d`.
- **Optimizers:** `SGD` (momentum + weight decay), `Adam` (bias-corrected + weight decay), and experimental `Muon`.
- **IO System:** Save and load model weights (state dicts) via `bincode`.
- **BLAS Acceleration (macOS):** Optional acceleration for matrix multiplication via Apple's Accelerate framework.
- **Validation-Focused:** Includes a robust numerical gradient checker to ensure the correctness of all implemented operations.

## Project Status

This library is functional for training MLPs and CNNs on CPU. It features a verified autograd engine and correctly implemented `im2col` convolutions.

- ✅ **What's Working:** Autograd, Conv2d/Linear layers, Optimizers (including Muon), DataLoaders, Serialization.
- ⚠️ **What's in Progress:** Performance is not yet a primary focus. While BLAS acceleration is available for macOS matrix multiplication, most operations use naive loops.
- ⚠️ **GPU Support:** Experimental GPU support (WIP) via the "gpu" feature. The main API remains CPU-only and using GPU device operations may panic or fall back to CPU with warnings.
- ❌ **What's Missing:**
  - Production-ready GPU integration, distributed training, learning-rate schedulers, recurrent/transformer layers.

## Installation

Add Volta to your `Cargo.toml`:

```toml
[dependencies]
volta = "0.2.0"
```

### Enabling BLAS on macOS

For a significant performance boost in matrix multiplication on macOS, enable the `accelerate` feature:

```toml
[dependencies]
volta = { version = "0.2.0", features = ["accelerate"] }
```

## Examples:

### Training an MLP

Here's how to define a simple Multi-Layer Perceptron (MLP), train it on synthetic data, and save the model.

```rust
use volta::{nn::*, tensor::*, Adam, Sequential, TensorOps, io};

fn main() {
    // 1. Define a simple model: 2 -> 8 -> 1
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);

    // 2. Create synthetic data
    let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x = RawTensor::new(x_data, &[4, 2], false); // Batch size 4, 2 features

    let y_data = vec![0.0, 1.0, 1.0, 0.0];
    let y = RawTensor::new(y_data, &[4], false); // Flattened targets

    // 3. Set up the optimizer
    let params = model.parameters();
    let mut optimizer = Adam::new(params, 0.1, (0.9, 0.999), 1e-8, 0.0);

    // 4. Training loop
    println!("Training a simple MLP to learn XOR...");
    for epoch in 0..=300 {
        optimizer.zero_grad();

        let pred = model.forward(&x).reshape(&[4]); //alignment
        let loss = mse_loss(&pred, &y);

        if epoch % 20 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss.borrow().data[0]);
        }

        loss.backward();
        optimizer.step();
    }
    // 5. Save and Load State Dict
    let state = model.state_dict();
    io::save_state_dict(&state, "model.bin").expect("Failed to save");

    // Verify loading
    let mut new_model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);
    let loaded_state = io::load_state_dict("model.bin").expect("Failed to load");
    new_model.load_state_dict(&loaded_state);

}
```

### LeNet-style CNN training on CPU

The following utilizes the current API to define a training-ready CNN.

```rust
use volta::{Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Adam};
use volta::nn::Module;
use volta::TensorOps;

fn main() {
    // 1. Define Model
    let model = Sequential::new(vec![
        // Input: 1x28x28
        Box::new(Conv2d::new(1, 6, 5, 1, 2, true)), // Padding 2
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2, 2, 0)),
        // Feature map size here: 6x14x14
        Box::new(Flatten::new()),
        Box::new(Linear::new(6 * 14 * 14, 10, true)),
    ]);

    // 2. Data & Optimizer
    let input = volta::randn(&[4, 1, 28, 28]); // Batch 4
    let target = volta::randn(&[4, 10]);       // Dummy targets
    let params = model.parameters();
    let mut optim = Adam::new(params, 1e-3, (0.9, 0.999), 1e-8, 0.0);

    // 3. Training Step
    optim.zero_grad();
    let output = model.forward(&input);
    let loss = volta::mse_loss(&output, &target);
    loss.backward();
    optim.step();

    println!("Loss: {:?}", loss.borrow().data[0]);
}
```

## API Overview

The library is designed around a few core concepts:

- **`Tensor`**: The central data structure, an `Rc<RefCell<RawTensor>>`, which holds data, shape, and gradient information. It allows for a mutable, shared structure to build the computation graph.
- **`TensorOps`**: A trait implemented for `Tensor` that provides the ergonomic, user-facing API for all operations (e.g., `tensor.add(&other)`, `tensor.matmul(&weights)`).
- **`nn::Module`**: A trait for building neural network layers (`Linear`, `ReLU`) and composing them into larger models (`Sequential`). It standardizes the `forward()` pass and parameter collection.
- **Optimizers (`Adam`, `SGD`, `Muon`)**: Structures that take a list of model parameters and update their weights based on computed gradients during `step()`.
- **Vision Support:** `Conv2d` uses im2col + GEMM, `MaxPool2d` with gradient support, plus `BatchNorm2d` and `Dropout`.

## Running the Test Suite

Volta has an extensive test suite that validates the correctness of every operation and its gradient. To run the tests:

```bash
cargo test -- --nocapture
```

To run tests with BLAS acceleration enabled (on macOS):

```bash
cargo test --features accelerate -- --nocapture
```

## Roadmap

The next major steps for Volta are focused on expanding its capabilities to handle more complex models and improving performance.

1.  **Performance Optimization:** Implement SIMD for element-wise operations and further integrate optimized BLAS routines.

### Outstanding Issues

- **Serialization Fragility**: `Sequential` relies on string-key matching for `state_dict` (e.g., "0.weight"). Renaming layers or changing architecture depth will break loading without helpful error messages.
- **Performance**: `im2col` implementation in `src/nn/layers/conv.rs` materializes the entire matrix in memory. Large batch sizes or high-resolution images will easily OOM even on high-end machines.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/rlarson20/volta/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
