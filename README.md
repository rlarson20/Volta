# Volta ⚡

<small>A PyTorch-like deep learning framework in pure Rust</small>

[![Build Status](https://img.shields.io/github/actions/workflow/status/rlarson20/volta/ci.yml?branch=main)](https://github.com/rlarson20/volta/actions)
[![Crates.io](https://img.shields.io/crates/v/volta.svg)](https://crates.io/crates/volta)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rlarson20/Volta)

Volta is a minimal deep learning and automatic differentiation library for Rust, heavily inspired by PyTorch.
Built from scratch to expose the internals - DAG linearization, `GradFn` dispatch, and SafeTensors interop - that production frameworks abstract away.
It provides a dynamic computation graph, NumPy-style broadcasting, and common neural network primitives.

This project is an educational endeavor to demystify the inner workings of modern autograd engines. It prioritizes correctness, clarity, and a clean API over raw performance, while still providing hooks for hardware acceleration.

> [!CAUTION]
> This repo is pre-v1.0, and currently... not in a spot that I'm proud of.
> Expect MAJOR refactors in order to get it into a more polished spot.

## Key Features

- **Dynamic Computation Graph:** Build and backpropagate through graphs on the fly, just like PyTorch.
- **Reverse-Mode Autodiff:** Efficient reverse-mode automatic differentiation with topological sorting.
- **Rich Tensor Operations:** A comprehensive set of unary, binary, reduction, and matrix operations as methods on `Tensor`.
- **Broadcasting:** Full NumPy-style broadcasting support for arithmetic operations.
- **Neural Network Layers:** `Linear`, `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `Embedding`, `LSTMCell`, `PixelShuffle`, `LayerNorm`, `Flatten`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Dropout`, `BatchNorm1d`, `BatchNorm2d`.
- **Optimizers:** `SGD` (momentum + weight decay), `Adam` (bias-corrected + weight decay), and experimental `Muon`.
- **External Model Loading:** Load weights from PyTorch, HuggingFace, and other frameworks via `StateDictMapper` with automatic weight transposition and key remapping. Supports SafeTensors format.
- **Named Layers:** Human-readable state dict keys with `Sequential::builder()` pattern for robust serialization.
- **SafeTensors dtypes:** f16, bf16, f64, i32, i64, u8, and bool files load (converted to f32); every op, layer, and optimizer computes in f32, f64, f16 or bf16 on the CPU and in f32, f16 or bf16 on the GPU (`Module::to_dtype` converts a model); `amp::autocast` and `GradScaler` give mixed precision (`docs/plans/ADDENDUM.md` §11).
- **IO System:** Save and load model weights (state dicts) via `bincode` or SafeTensors format.
- **BLAS Acceleration (macOS):** Optional acceleration for matrix multiplication via Apple's Accelerate framework.
- **GPU Acceleration:** Experimental WGPU-based GPU support for core tensor operations (elementwise, matmul, reductions, movement ops) with automatic backward pass on GPU.
- **Validation-Focused:** Includes a robust numerical gradient checker to ensure the correctness of all implemented operations.

## Project Status

This library trains MLPs, CNNs, RNNs, GANs, VAEs, and other architectures on CPU. Known defects are tracked as [GitHub issues](https://github.com/rlarson20/Volta/issues); the `merge-gate` label marks the ones that block a release.

- ✅ **What's Working:**
  - **Core Autograd:** Operations are tested against numerical gradients, and the core ops and layers against PyTorch reference values (`tests/torch_ref_*.rs`)
  - **Layers:** Linear, Conv2d, ConvTranspose2d, MaxPool2d, Embedding, LSTMCell, PixelShuffle, LayerNorm, BatchNorm1d/2d, Dropout, MultiHeadAttention, TransformerBlock, PositionalEncoding
  - **Optimizers:** SGD (with momentum), Adam, Muon
  - **External Loading:** PyTorch/HuggingFace model weights via SafeTensors with automatic transposition
  - **Named Layers:** Robust serialization with human-readable state dict keys
  - **Loss Functions:** MSE, Cross-Entropy, NLL, BCE, KL Divergence
  - **Examples:** MNIST (CNN, LSTM, bf16 autocast), character LMs (LSTM, Transformer), VAE, DCGAN, super-resolution, LSTM sine-wave prediction, polynomial regression; see [Available Examples](#available-examples)
  - **GPU Training Pipeline:** Fully GPU-accelerated training for Conv2d layers with device-aware constructors and GPU optimizer state storage
  - **Benchmarking Suite:** Criterion benchmarks in 4 targets (tensor_ops, neural_networks, gpu_comparison, conv_algorithms) with HTML reports
  - **Enhanced GPU Safety:** GPU buffer pooling, command queue throttling, CPU cache invalidation, and early warning system
  - **Code Quality:** `cargo clippy --all-targets -- -D warnings` (with the pedantic set configured in `Cargo.toml`) is clean with and without default features, and runs in CI
  - **GPU Convolution:** Fully GPU-accelerated Conv2d with Direct, im2col, and iGEMM algorithms
    - All algorithms support both forward and backward passes on GPU
    - Auto-selection chooses optimal algorithm based on input size and device
    - Memory-efficient alternatives (Direct, iGEMM) prevent OOM on large inputs

- ⚠️ **What's in Progress:**
  - **Performance:** Comprehensive benchmarking suite for performance tracking with `just bench` commands
  - **Benchmarks:** `just bench-gpu` completes a full run. The GPU is slower than the CPU at every size measured, because each op pays about 1.5 ms of fixed overhead; see `dev-docs/BENCHMARKING.md`
  - **GPU Support:** Experimental WGPU-based acceleration via `gpu` feature:
    - ✅ Core ops on GPU: elementwise (unary/binary), matmul, reductions (sum/max/mean), movement ops (permute/expand/pad/shrink/stride)
    - ✅ GPU backward pass for autograd with lazy CPU↔GPU transfers
    - ⚠️ Broadcasting preprocessing happens on CPU before GPU dispatch

- ❌ **What's Missing:**
  - Production-ready GPU integration, distributed training, learning-rate schedulers (planned; see the roadmap)

## Installation

Add Volta to your `Cargo.toml`:

```toml
[dependencies]
volta = "0.3.0"
```

Both optional backends are default features: `gpu` (experimental WGPU acceleration, which compiles `wgpu`) and `accelerate` (Apple's Accelerate framework for matrix multiplication on macOS; a no-op on other platforms).

### CPU-only build

To build without the GPU backend, turn the default features off:

```toml
[dependencies]
volta = { version = "0.3.0", default-features = false }
```

Add back what you want, for example `features = ["accelerate"]` for BLAS on macOS without the GPU backend.

## Architecture

Volta is built around a dynamic computation graph that tracks operations for reverse-mode automatic differentiation.

### Computation Graph & DAG Linearization

When operations are performed on tensors requiring gradients, Volta constructs a Directed Acyclic Graph (DAG) of the computation. During the backward pass, this graph is linearized via topological sorting to compute gradients efficiently without deep recursion.

```text
    [Input x] (requires_grad=true)             [Weights w] (requires_grad=true)
          \                                         /
           \                                       /
            v                                     v
         [matmul + add] <--- (GradFn::Matmul, GradFn::Binary) --
               \
                v
            [ReLU] <--- (GradFn::Unary)
               |
               v
            [Loss] <--- (the GradFns of the ops the loss is built from)
```

### `GradFn` Dispatch

The core of the autograd engine is the `GradFn` enum (`src/autograd.rs`). The output of each differentiable operation stores a `GradFn` variant holding what its backward pass needs. During the backward pass, the topological sort yields a linearized sequence of nodes; the engine visits them iteratively and calls each node's `GradFn::backward`, a `match` over the variants, which maps the gradient of the output to the gradients of the inputs. This iterative approach avoids call stack exhaustion on deep, complex graphs.

### Serialization: Why SafeTensors?

Volta saves and loads state dicts in two formats: `bincode` (`io::save_state_dict` / `io::load_state_dict`, used by the MLP example below) and HuggingFace's SafeTensors (`io::save_safetensors` / `io::load_safetensors`). SafeTensors is the interchange format, for these reasons:

- **Simple, checked layout**: a JSON header followed by raw tensor bytes. The format allows memory-mapping, but Volta does not use it: `load_safetensors` reads the whole file into memory and copies each tensor out of it.
- **Safety**: Unlike Pickle, SafeTensors cannot execute arbitrary code.
- **Interoperability**: It enables Volta to natively read weights from models trained in PyTorch or JAX.

### Compute Backends

Volta provides multiple backends for computation:

- **CPU Backend**: Matrix multiplication uses Apple's Accelerate framework on macOS (the `accelerate` feature) and the single-threaded `matrixmultiply` crate elsewhere. Everything else runs in plain Rust loops on one thread.
- **GPU Backend (Experimental)**: Uses `wgpu` to dispatch operations to the GPU. Tensors can seamlessly move between devices using `Tensor::to_device()`.

## Benchmarks

Criterion benchmarks run through `just bench` (CPU) and `just bench-gpu`. GPU results and how to read them are in `dev-docs/BENCHMARKING.md`; the GPU does not beat the CPU yet.

## Examples:

### Training an MLP

Here's how to define a simple Multi-Layer Perceptron (MLP) with named layers, train it on synthetic data, and save the model.

```rust,no_run
use volta::{
    Adam, Sequential, io,
    nn::{Layer, Linear, Module, ReLU},
    tensor::{RawTensor, mse_loss},
};

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
            println!("Epoch {}: loss = {:.6}", epoch, loss.item());
        }

        loss.into_loss().expect("loss is scalar").backward();
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
    new_model.load_state_dict(&loaded_state).expect("Failed to load state");
}
```

### LeNet-style CNN training on CPU

The following utilizes the current API to define a training-ready CNN.

```rust
use volta::{Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Adam};
use volta::nn::{Layer, Module};

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
    loss.clone().into_loss().expect("loss is scalar").backward();
    optim.step();

    println!("Loss: {}", loss.item());
}
```

### Loading External PyTorch Models

Volta can load weights from PyTorch, HuggingFace, and other frameworks using SafeTensors format with automatic weight mapping and transposition.

```rust,no_run
use volta::{
    Layer, Linear, Module, ReLU, Sequential,
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
    model.load_state_dict(&volta_state).expect("converted weights should fit the model");

    // 4. Run inference
    let input = volta::randn(&[1, 784]);
    let output = model.forward(&input);
    println!("Output shape: {:?}", output.shape());
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

```rust,no_run
use volta::{Device, randn};

fn main() {
    let device = Device::gpu().expect("GPU required");

    // Create tensors on the CPU and move them to the GPU. `randn` tensors don't require
    // grad, so mark `a_gpu` (a leaf) as one, or backward has nothing to reach.
    let a_gpu = randn(&[1024, 1024]).to_device(device.clone());
    a_gpu.set_requires_grad(true);
    let b_gpu = randn(&[1024, 1024]).to_device(device);

    // Operations execute on GPU automatically
    let c_gpu = a_gpu.matmul(&b_gpu); // GPU matmul
    let sum_gpu = c_gpu.sum(); // GPU reduction

    // Gradients computed on GPU when possible
    sum_gpu.into_loss().expect("loss is scalar").backward();
    let grad = a_gpu.grad().expect("a_gpu requires grad");
    println!("Gradient elements: {}", grad.len());
}
```

## API Overview

The library is designed around a few core concepts:

- **`Tensor`**: The central data structure, a handle to a shared, reference-counted `RawTensor` (data, shape, dtype, gradient, and device). Its fields are private; read them with methods such as `data()`, `item()`, `shape()`, `grad()` and `dtype()`. The Dtypes note under Outstanding Issues says which dtypes compute.
- **Tensor methods**: `Tensor` has inherent methods for all operations (e.g., `tensor.add(&other)`, `tensor.matmul(&weights)`); there is no trait to import.
- **`nn::Module`**: The base trait for anything with learnable parameters. Provides `parameters()`, `state_dict()`, `load_state_dict()`, `zero_grad()`, `to_device()`, and `train()`/`eval()`.
- **`nn::Layer`**: A `Module` that maps one tensor to one tensor with `forward()`. Only `Layer`s can go in a `Sequential`. Implement it to write your own layer.
- **`nn::RecurrentLayer`** (`LSTMCell`): a sealed trait for layers that carry state between steps (`initial_state`/`step`). `Embedding` is a plain `Layer` whose input is an I64 or I32 index tensor, as in PyTorch.
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

The default features include `gpu` and `accelerate`, so this already builds the GPU backend (GPU tests skip when no adapter is found). For a CPU-only run:

```bash
cargo test --no-default-features
```

`cargo test <filter>` runs the tests whose names contain `<filter>`; there are no category modules. To run one test file, name it:

```bash
cargo test --no-default-features --test torch_ref_layers   # layers against PyTorch values
cargo test --no-default-features --lib conv                # library tests with "conv" in the name
```

`just ci` runs the full gate (fmt, clippy, tests and docs with and without default features, and the examples).

## Available Examples

The `examples/` directory contains complete programs demonstrating various capabilities. Those in `just examples-smoke` run in CI on every change; the list and the reason for each one left out are in `dev-docs/DEVELOPMENT.md` ("Examples smoke run").

The MNIST examples (`mnist_cnn`, `mnist_rnn`, `mnist_autocast`, `vae`, `dcgan`) need the dataset in `data/mnist/`: `just mnist-data` downloads and checks it. Build anything that trains on MNIST with `--release`.

```bash
# Basic examples
cargo run --example readme1                    # Simple MLP training
cargo run --example readme2                    # LeNet-style CNN
cargo run --example showcase                   # Feature showcase
cargo run --example layernorm_demo             # LayerNorm forward and gradients

# Vision tasks
cargo run --release --example mnist_cnn        # MNIST digit classification
cargo run --release --example mnist_autocast   # MNIST under bf16 autocast vs f32
cargo run --example super_resolution           # Image upscaling with PixelShuffle
cargo run --release --example dcgan -- 1       # Deep Convolutional GAN (argument: epochs, default 5)

# Generative models
cargo run --release --example vae              # Variational Autoencoder

# Sequence models
cargo run --example char_language_model        # Character-level LSTM language model
cargo run --example char_transformer_model     # Character-level Transformer language model
cargo run --release --example mnist_rnn        # MNIST rows as a sequence, LSTM classifier
cargo run --release --example time_sequence    # LSTM sine-wave prediction (trains to NaN today, #88)

# External model loading
cargo run --example load_external_mnist        # Load PyTorch weights via SafeTensors

# GPU acceleration (the `gpu` feature is on by default)
cargo run --example gpu                        # GPU tensor operations
cargo run --example gpu_training               # GPU-accelerated training

# Regression
cargo run --example regression                 # Polynomial curve fitting
```

## Roadmap

The next major steps for Volta are focused on expanding its capabilities to handle more complex models and improving performance.

1. **Complete GPU Integration:** Port the remaining layers to the GPU (`ConvTranspose2d` has no GPU path), optimize GEMM kernels with shared memory tiling.
2. **Performance Optimization:** Implement SIMD for element-wise operations, optimize broadcasting on GPU, kernel fusion for composite operations.
3. **Learning Rate Schedulers:** Cosine annealing, step decay, warmup schedules.

### Outstanding Issues

- **Conv2d Memory Inefficiency**: `im2col` implementation in `src/nn/layers/conv.rs` materializes the entire matrix in memory. Large batch sizes or high-resolution images will easily OOM even on high-end machines. Somewhat mitigated through algorithm selection in convolutional layer.
- **GPU Kernel Efficiency**: Current GPU matmul uses naive implementation without shared memory tiling. Significant performance gains possible with optimized GEMM kernels.
- **Dtypes**: tensors of every dtype can be built, read back (`to_vec_as`), and cast (`RawTensor::from_typed`, `to_dtype`). Ops compute in f32, f64, f16 and bf16 on the CPU (half types in f32, rounding each output) and in f32, f16 and bf16 on the GPU (bf16, and f16 without `SHADER_F16`, as packed 16-bit emulation); integer and bool tensors are data (movement, comparisons, `where`, indices, class targets), and integer arithmetic panics. Mixed operands promote as in PyTorch unless `DTypePolicy::Strict`. A non-f32 model cannot be saved yet (`state_dict` is f32).
- **Single-threaded**: Uses `Rc<RefCell>` instead of `Arc<Mutex>`, limiting to single-threaded execution on CPU.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/rlarson20/volta/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
