# Volta ⚡

<small>A PyTorch-like deep learning framework in pure Rust</small>

[![Build Status](https://img.shields.io/github/actions/workflow/status/rlarson20/volta/ci.yml?branch=main)](https://github.com/rlarson20/volta/actions)
[![Crates.io](https://img.shields.io/crates/v/volta.svg)](https://crates.io/crates/volta)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rlarson20/Volta)

Volta is a minimal deep learning and automatic differentiation library built from scratch in pure Rust, heavily inspired by PyTorch. It provides a dynamic computation graph, NumPy-style broadcasting, and common neural network primitives.

This project is an educational endeavor to demystify the inner workings of modern autograd engines. It prioritizes correctness, clarity, and a clean API over raw performance, while still providing hooks for hardware acceleration.

## Key Features

- **Dynamic Computation Graph:** Build and backpropagate through graphs on the fly, just like PyTorch.
- **Reverse-Mode Autodiff:** Efficient reverse-mode automatic differentiation with topological sorting.
- **Rich Tensor Operations:** A comprehensive set of unary, binary, reduction, and matrix operations via an ergonomic `TensorOps` trait.
- **Broadcasting:** Full NumPy-style broadcasting support for arithmetic operations.
- **Neural Network Layers:** `Linear`, `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `Embedding`, `LSTMCell`, `PixelShuffle`, `Flatten`, `ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `BatchNorm1d`, `BatchNorm2d`.
- **Optimizers:** `SGD` (momentum + weight decay), `Adam` (bias-corrected + weight decay), and experimental `Muon`.
- **External Model Loading:** Load weights from PyTorch, HuggingFace, and other frameworks via `StateDictMapper` with automatic weight transposition and key remapping. Supports SafeTensors format.
- **Named Layers:** Human-readable state dict keys with `Sequential::builder()` pattern for robust serialization.
- **Multi-dtype Support:** Initial support for f16, bf16, f32, f64, i32, i64, u8, and bool tensors.
- **IO System:** Save and load model weights (state dicts) via `bincode` or SafeTensors format.
- **BLAS Acceleration (macOS):** Optional acceleration for matrix multiplication via Apple's Accelerate framework.
- **GPU Acceleration:** Experimental WGPU-based GPU support for core tensor operations (elementwise, matmul, reductions, movement ops) with automatic backward pass on GPU.
- **Validation-Focused:** Includes a robust numerical gradient checker to ensure the correctness of all implemented operations.

## Project Status

This library is functional for training MLPs, CNNs, RNNs, GANs, VAEs, and other architectures on CPU. It features a verified autograd engine and correctly implemented `im2col` convolutions.

- ✅ **What's Working:**
  - **Core Autograd:** All operations verified with numerical gradient checking
  - **Layers:** Linear, Conv2d, ConvTranspose2d, MaxPool2d, Embedding, LSTMCell, PixelShuffle, BatchNorm1d/2d, Dropout
  - **Optimizers:** SGD (with momentum), Adam, Muon
  - **External Loading:** PyTorch/HuggingFace model weights via SafeTensors with automatic transposition
  - **Named Layers:** Robust serialization with human-readable state dict keys
  - **Loss Functions:** MSE, Cross-Entropy, NLL, BCE, KL Divergence
  - **Examples:** MNIST, CIFAR, character LM, VAE, DCGAN, super-resolution, LSTM time series
  - **GPU Training Pipeline:** GPU-accelerated forward pass for Conv2d with device-aware layers and GPU optimizer state storage
  - **Benchmarking Suite:** Comprehensive Criterion benchmarks with 3 categories (tensor_ops, neural_networks, gpu_comparison) and HTML reports
  - **Enhanced GPU Safety:** GPU buffer pooling, command queue throttling, CPU cache invalidation, and early warning system
  - **Code Quality:** All `indexing_slicing` clippy errors resolved; ~400+ pedantic lints reduced to ~223 remaining

- ⚠️ **What's in Progress:**
  - **Performance:** Comprehensive benchmarking suite for performance tracking with `just bench` commands
  - **GPU Support:** Experimental WGPU-based acceleration via `gpu` feature:
    - ✅ Core ops on GPU: elementwise (unary/binary), matmul, reductions (sum/max/mean), movement ops (permute/expand/pad/shrink/stride)
    - ✅ GPU backward pass for autograd with lazy CPU↔GPU transfers
    - ✅ GPU-accelerated forward pass implemented for Conv2d
    - ⚠️ Neural network layer backward passes still being ported to GPU
    - ⚠️ Broadcasting preprocessing happens on CPU before GPU dispatch

- ❌ **What's Missing:**
  - Production-ready GPU integration, distributed training, learning-rate schedulers, attention/transformer layers

## Installation

Add Volta to your `Cargo.toml`:

```toml
[dependencies]
volta = "0.3.0"
```

### Enabling BLAS on macOS

For a significant performance boost in matrix multiplication on macOS, enable the `accelerate` feature:

```toml
[dependencies]
volta = { version = "0.3.0", features = ["accelerate"] }
```

### Enabling GPU Support

For experimental GPU acceleration via WGPU, enable the `gpu` feature:

```toml
[dependencies]
volta = { version = "0.3.0", features = ["gpu"] }
```

Or combine both for maximum performance:

```toml
[dependencies]
volta = { version = "0.3.0", features = ["accelerate", "gpu"] }
```

## Examples:

### Training an MLP

Here's how to define a simple Multi-Layer Perceptron (MLP) with named layers, train it on synthetic data, and save the model.

```rust
use volta::{nn::*, tensor::*, Adam, Sequential, TensorOps, io};

fn main() {
    // 1. Define a simple model with named layers: 2 -> 8 -> 1
    let model = Sequential::builder()
        .add_named("fc1", Box::new(Linear::new(2, 8, true)))
        .add_unnamed(Box::new(ReLU))
        .add_named("fc2", Box::new(Linear::new(8, 1, true)))
        .build();

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

    // 5. Save and Load State Dict (human-readable keys: "fc1.weight", "fc1.bias", etc.)
    let state = model.state_dict();
    io::save_state_dict(&state, "model.bin").expect("Failed to save");

    // Verify loading
    let mut new_model = Sequential::builder()
        .add_named("fc1", Box::new(Linear::new(2, 8, true)))
        .add_unnamed(Box::new(ReLU))
        .add_named("fc2", Box::new(Linear::new(8, 1, true)))
        .build();
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

### Loading External PyTorch Models

Volta can load weights from PyTorch, HuggingFace, and other frameworks using SafeTensors format with automatic weight mapping and transposition.

```rust
use volta::{
    Linear, Module, ReLU, Sequential,
    io::{load_safetensors, mapping::StateDictMapper},
};

fn main() {
    // 1. Build matching architecture with named layers
    let mut model = Sequential::builder()
        .add_named("fc1", Box::new(Linear::new(784, 128, true)))
        .add_unnamed(Box::new(ReLU))
        .add_named("fc2", Box::new(Linear::new(128, 10, true)))
        .build();

    // 2. Load PyTorch weights with automatic transposition
    // PyTorch Linear stores weights as [out, in], Volta uses [in, out]
    let pytorch_state = load_safetensors("pytorch_model.safetensors")
        .expect("Failed to load SafeTensors");

    let mapper = StateDictMapper::new()
        .transpose("fc1.weight")  // [128,784] → [784,128]
        .transpose("fc2.weight"); // [10,128] → [128,10]

    let volta_state = mapper.map(pytorch_state);

    // 3. Load into model
    model.load_state_dict(&volta_state);

    // 4. Run inference
    let input = volta::randn(&[1, 784]);
    let output = model.forward(&input);
    println!("Output shape: {:?}", output.borrow().shape);
}
```

**Weight Mapping Features:**

- `rename(from, to)` - Rename individual keys
- `rename_prefix(old, new)` - Rename all keys with prefix
- `strip_prefix(prefix)` - Remove prefix from keys
- `transpose(key)` - Transpose 2D weight matrices (PyTorch compatibility)
- `transpose_pattern(pattern)` - Transpose all matching keys
- `select_keys(keys)` / `exclude_keys(keys)` - Filter state dict

See `examples/load_external_mnist.rs` for a complete end-to-end example with validation.

### GPU Acceleration Example

```rust
use volta::{Device, TensorOps, randn};

fn main() {
    // Create tensors on CPU
    let a = randn(&[1024, 1024]);
    let b = randn(&[1024, 1024]);

    // Move to GPU
    let device = Device::gpu().expect("GPU required");
    let a_gpu = a.to_device(device.clone());
    let b_gpu = b.to_device(device.clone());

    // Operations execute on GPU automatically
    let c_gpu = a_gpu.matmul(&b_gpu);  // GPU matmul
    let sum_gpu = c_gpu.sum();          // GPU reduction

    // Gradients computed on GPU when possible
    sum_gpu.backward();
    println!("Gradient shape: {:?}", a_gpu.borrow().grad.as_ref().unwrap().shape());
}
```

## API Overview

The library is designed around a few core concepts:

- **`Tensor`**: The central data structure, an `Rc<RefCell<RawTensor>>`, which holds data, shape, gradient information, and device location. Supports multiple data types (f32, f16, bf16, f64, i32, i64, u8, bool).
- **`TensorOps`**: A trait implemented for `Tensor` that provides the ergonomic, user-facing API for all operations (e.g., `tensor.add(&other)`, `tensor.matmul(&weights)`).
- **`nn::Module`**: A trait for building neural network layers and composing them into larger models. Provides `forward()`, `parameters()`, `state_dict()`, `load_state_dict()`, and `to_device()` methods.
- **`Sequential::builder()`**: Builder pattern for composing layers with named parameters for robust serialization. Supports both `add_named()` for human-readable state dict keys and `add_unnamed()` for activation layers.
- **Optimizers (`Adam`, `SGD`, `Muon`)**: Structures that take a list of model parameters and update their weights based on computed gradients during `step()`.
- **`Device`**: Abstraction for CPU/GPU compute. Tensors can be moved between devices with `to_device()`, and operations automatically dispatch to GPU kernels when available.
- **External Model Loading:** `StateDictMapper` provides transformations (rename, transpose, prefix handling) to load weights from PyTorch, HuggingFace, and other frameworks via SafeTensors format.
- **Vision Support:** `Conv2d`, `ConvTranspose2d` (for GANs/VAEs), `MaxPool2d`, `PixelShuffle` (for super-resolution), `BatchNorm1d/2d`, and `Dropout`.
- **Sequence Support:** `Embedding` layers for discrete inputs, `LSTMCell` for recurrent architectures.

## Running the Test Suite

Volta has an extensive test suite that validates the correctness of every operation and its gradient. To run the tests:

```bash
cargo test -- --nocapture
```

To run tests with BLAS acceleration enabled (on macOS):

```bash
cargo test --features accelerate -- --nocapture
```

To run tests with GPU support:

```bash
cargo test --features gpu -- --nocapture
```

Run specific test categories:

```bash
cargo test core          # Core tensor tests
cargo test grad_check    # Numerical gradient validation
cargo test broadcasting  # Broadcasting rules
cargo test neural        # Neural network layers
cargo test optimizer     # Optimizer convergence
```

## Available Examples

The `examples/` directory contains complete working examples demonstrating various capabilities:

```bash
# Basic examples
cargo run --example readme1                    # Simple MLP training
cargo run --example readme2                    # LeNet-style CNN
cargo run --example showcase                   # Feature showcase

# Vision tasks
cargo run --example mnist_cnn                  # MNIST digit classification
cargo run --example super_resolution           # Image upscaling with PixelShuffle
cargo run --example dcgan                      # Deep Convolutional GAN

# Generative models
cargo run --example vae                        # Variational Autoencoder

# Sequence models
cargo run --example char_lm                    # Character-level language model
cargo run --example lstm_time_series           # Time series prediction

# External model loading
cargo run --example load_external_mnist        # Load PyTorch weights via SafeTensors

# GPU acceleration
cargo run --example gpu --features gpu         # GPU tensor operations
cargo run --example gpu_training --features gpu # GPU-accelerated training

# Regression
cargo run --example polynomial_regression      # Polynomial curve fitting
```

## Roadmap

The next major steps for Volta are focused on expanding its capabilities to handle more complex models and improving performance.

1. **Complete GPU Integration:** Port remaining neural network layers (Linear, Conv2d) to GPU, optimize GEMM kernels with shared memory tiling.
2. **Performance Optimization:** Implement SIMD for element-wise operations, optimize broadcasting on GPU, kernel fusion for composite operations.
3. **Transformer Support:** Add attention mechanisms, positional encodings, layer normalization.
4. **Learning Rate Schedulers:** Cosine annealing, step decay, warmup schedules.

### Outstanding Issues

- **Conv2d Memory Inefficiency**: `im2col` implementation in `src/nn/layers/conv.rs` materializes the entire matrix in memory. Large batch sizes or high-resolution images will easily OOM even on high-end machines.
- **GPU Kernel Efficiency**: Current GPU matmul uses naive implementation without shared memory tiling. Significant performance gains possible with optimized GEMM kernels.
- **Multi-dtype Completeness**: While storage supports multiple dtypes (f16, bf16, f64, etc.), most operations still assume f32. Full dtype support requires operation kernels for each type.
- **Single-threaded**: Uses `Rc<RefCell>` instead of `Arc<Mutex>`, limiting to single-threaded execution on CPU.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/rlarson20/volta/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
