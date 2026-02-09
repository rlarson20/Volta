# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Instructions

Whenever you're corrected on something, or you learn something about the codebase, either add it to the relevant spot in this file, or add it under the Learned Instructions if it's too general for any other spot in this file.

## Learned Instructions

## Project Overview

Volta is a PyTorch-like deep learning framework written in pure Rust. It's an educational project designed to demystify automatic differentiation engines and neural network computation graphs. The project is currently functional for CPU-based training of MLPs, CNNs, and RNNs with an extensive verified autograd engine.

## Development Commands

### Rust (Core Library)

```bash
# Build and test
cargo check
cargo build
cargo test -- --nocapture

# Build with BLAS acceleration (macOS only)
cargo build --features accelerate
cargo test --features accelerate -- --nocapture

# GPU-specific commands (default features include gpu and accelerate)
cargo build --features gpu                    # Build with GPU support
cargo test --features gpu -- --nocapture     # Run tests with GPU
cargo run --example gpu --features gpu        # Run GPU example

# Run examples
cargo run --example readme1                    # Simple MLP training
cargo run --example readme2                    # LeNet-style CNN
cargo run --example showcase                   # Feature showcase
cargo run --example load_external_mnist        # PyTorch → Volta loading
cargo run --example polynomial_regression      # Polynomial regression demo
cargo run --example lstm_time_series           # Time series prediction
cargo run --example char_language_model        # Character-level LSTM language modeling

# Quick verification (runs check, build, and test)
just check
```

### Testing

```bash
# Run all tests with output
cargo test -- --nocapture

# Run specific test categories
cargo test core                # Core tensor tests
cargo test grad_check          # Numerical gradient validation
cargo test broadcasting        # Broadcasting rules
cargo test neural              # Neural network layers
cargo test optimizer           # Optimizer convergence

# Run a specific test
cargo test test_name -- --nocapture
```

### Benchmarking

**⚠️ WARNING:** The `gpu_comparison` benchmark is NOT SAFE and will crash your computer.
DO NOT run `just bench-gpu` or `cargo bench --bench gpu_comparison`.

The new `conv_algorithms` benchmark is SAFE and designed to stay under 16GB memory limit.

```bash
# Run all benchmarks (except gpu_comparison)
just bench

# Run convolution algorithm comparison (SAFE)
just bench-conv

# Run specific benchmark
just bench-name tensor_ops
just bench-name neural_networks
just bench-name gpu_comparison # NOT FOR USE

# CPU-only benchmarks (no acceleration)
just bench-cpu

# With BLAS acceleration (macOS)
just bench-accel

# GPU comparison benchmarks
just bench-gpu

# Save baseline for comparison
just bench-save

# Compare against saved baseline
just bench-compare

# Open HTML report
just bench-report
```

### LLM Integration (justfile)

The justfile includes LLM-assisted development commands (ONLY for the maintainer's use):

```bash
just ask <model>              # Ask an LLM for recommendations
just ask-gpu <model>          # Ask an LLM for GPU-specific help
just ask-err <model>          # Ask an LLM to diagnose build/test errors
just ask-status <model>       # Get project status report from LLM
```

### Convolution Algorithm Benchmarks

The `conv_algorithms` benchmark compares Direct vs im2col+GEMM convolution algorithms:

**Benchmark Groups:**

1. **`conv_algorithm_comparison`**: Compares algorithms across various scenarios
   - Tiny/small/medium/large inputs
   - Different kernel sizes (1x1, 3x3, 5x5, 7x7)
   - ResNet-style layers

2. **`conv_kernel_comparison`**: Varies kernel size (1-7)
   - Fixed input: 8x64x56x56
   - Shows how kernel size affects performance

3. **`conv_batch_comparison`**: Varies batch size (1-32)
   - Fixed input: 64x56x56
   - Shows scaling with batch size

4. **`conv_spatial_comparison`**: Different spatial dimensions
   - MNIST (28x28), CIFAR (32x32)
   - ImageNet variants (56x56, 112x112, 224x224)

5. **`conv_auto_mode`**: Tests automatic algorithm selection
   - Verifies Auto mode chooses correctly

6. **`conv_memory_scaling`**: Shows memory usage patterns
   - Scales channels, spatial dimensions, and batch size
   - All benchmarks stay under 16GB limit

**Memory Safety:**

- All scenarios calculate im2col memory usage beforehand
- Maximum memory used: ~1GB (very_large scenario)
- Prints memory usage for each benchmark
- Asserts <16GB limit before running

**Example output:**

```
tiny: B=1, C=3→16, H=W=16, K=3x3, im2col memory: 0.02 MB
small: B=4, C=3→16, H=W=32, K=3x3, im2col memory: 0.41 MB
medium: B=8, C=64→128, H=W=28, K=3x3, im2col memory: 67.5 MB
```

### Benchmarking Guidelines

For benchmark changes:

- Always implement resource cleanup (buffer pools, cache invalidation) after benchmark groups
- Use size-bucketed buffer pools to prevent exhaustion
- Add hard sync timeouts for GPU operations

## Architecture

### Core Design Philosophy

Volta is built around a few fundamental components that work together to provide a PyTorch-like deep learning experience:

**Tensor Design (`src/tensor.rs`):**

```rust
pub type Tensor = Rc<RefCell<RawTensor>>;

pub struct RawTensor {
    pub data: Storage,              // CPU Vec<u8> or GPU buffer
    pub shape: Vec<usize>,          // Dimensions: [B, C, H, W]
    pub grad: Option<Storage>,      // Gradient accumulation
    pub requires_grad: bool,        // Track gradients?
    pub grad_fn: Option<Box<dyn GradFn>>, // Backprop function
    pub parents: Vec<Tensor>,       // Dependency graph
    pub device: Device,             // CPU/GPU
}
```

Key design choices:

- **Rc<RefCell>** for dynamic computation graphs (single-threaded)
- **Gradient functions as trait objects** for polymorphic backpropagation
- **Storage abstraction** for CPU/GPU portability
- **Row-major flat vectors** for efficient memory layout

**Storage System (`src/storage.rs`):**

```rust
pub enum Storage {
    Cpu { data: Vec<u8>, dtype: DType },
    #[cfg(feature = "gpu")]
    Gpu { buffer: Arc<GpuBuffer>, dtype: DType, cpu_cache: RefCell<Option<Vec<u8>>> }
}
```

This byte-level storage design enables:

- **Runtime dtype flexibility**: Data stored as `Vec<u8>` with separate `dtype` tag
- **Supported dtypes**: f16, bf16, f32, f64, i32, i64, u8, bool
- **Efficient GPU↔CPU transfers**: Lazy caching via `cpu_cache` in GPU variant
- **SafeTensors compatibility**: Direct mapping to SafeTensors format dtype specification

**Error Handling (`src/error.rs`):**

Recent refactor introduced `VoltaError` types using `thiserror` for comprehensive error handling:

- **VoltaError** enum with specific error variants (BroadcastError, ShapeError, DeviceError, etc.)
- **Result types** instead of panics where possible
- **Error propagation** with context throughout the codebase
- All operations now fallibly handle edge cases (bounds checking, device compatibility, etc.)

**Autograd Engine (`src/autograd.rs`):**

- Reverse-mode (backpropagation) via topological DFS
- Topological sorting prevents recomputation in diamond graphs
- Gradient accumulation in parent tensors
- All operations verified with numerical gradient checking

**Operation Organization (`src/ops/`):**

- `unary.rs`: Element-wise ops (sigmoid, relu, sqrt, exp, log, sin, cos, tanh)
- `binary.rs`: Binary ops (add, sub, elem_mul, div, max_elem, pow)
- `matmul.rs`: Matrix multiplication with broadcasting
- `reduce.rs`: Reduction ops (sum, mean, max_reduce)
- `ternary.rs`: Three-operand ops (mulacc, where)
- `movement.rs`: Shape manipulation (reshape, permute, expand, transpose, pad, shrink, stride)

**Neural Network Layers (`src/nn/layers/`):**

- `linear.rs`: Fully connected layer
- `conv.rs`: 2D convolution with algorithm selection (Direct, im2col + GEMM, Auto)
  - **Direct convolution**: Memory-efficient, computes each output pixel by iterating over kernel
  - **im2col + GEMM**: Faster for large kernels, materializes intermediate matrix
  - **Auto-selection**: Chooses algorithm based on input size, kernel size, batch size, and device
  - **Full gradient support**: Both algorithms support forward and backward passes
- `conv_transpose.rs`: Transposed convolution (for GANs/VAEs)
- `maxpool.rs`: Max pooling with gradient support
- `batchnorm.rs`: Batch normalization (1d and 2d)
- `dropout.rs`: Stochastic regularization
- `relu.rs`, `sigmoid.rs`, `tanh.rs`: Activation functions
- `sequential.rs`: Container for composing layers
- `sequential_builder.rs`: Builder pattern for named layers
- `flatten.rs`: Reshape to 1D
- `embedding.rs`: Embedding layer for discrete inputs
- `lstm.rs`: LSTM cell for recurrent architectures
- `pixelshuffle.rs`: Pixel shuffle for super-resolution

**Optimizers (`src/nn/optim/`):**

- `sgd.rs`: Stochastic gradient descent with momentum and weight decay
- `adam.rs`: Adam optimizer with bias correction and weight decay
- `muon.rs`: Experimental Muon optimizer

### GPU Support (`src/gpu/`)

WGPU-based acceleration for core tensor operations:

**✅ GPU-accelerated operations:**

- **Binary ops**: add, sub, mul, div, max, mod, cmplt
- **Unary ops**: neg, exp, log, relu, sigmoid, tanh, sqrt, recip, exp2, log2, sin, cos
- **Reduction ops**: sum, max, mean
- **Matrix multiplication**: matmul with configurable workgroup sizes
- **Movement ops**: permute, expand, pad, shrink, stride
- **Backward pass**: GPU-accelerated gradients for core operations
- **Conv2d**: GPU-accelerated convolution (im2col and direct algorithms)

**❌ Still CPU-only:**

- Neural network layer backward passes (Conv2d backward, Linear backward)
- Broadcasting preprocessing
- Loss functions

**GPU Safety Infrastructure:**

- GPU buffer pooling (64 buffers) to prevent allocation exhaustion
- Command queue throttling to prevent system hangs
- CPU cache invalidation for GPU benchmarks
- Early warning system for resource exhaustion

**GPU Development Guidelines:**

When implementing GPU operations:

1. Ensure both forward and backward passes are GPU-resident
2. Check for CPU fallback paths in binary ops (especially `to_f32_vec()`)
3. Verify bind group layouts match between forward/backward
4. Test with actual GPU tensors, not just CPU tensors

## Pre-commit Hooks

The project uses pre-commit hooks configured in `.pre-commit-config.yaml`:

- **fmt**: Runs `cargo fmt` to format code
- **cargo-check**: Validates code compiles
- **clippy**: Lints with exceptions for `needless_range_loop` and `too_many_arguments`
- **cargo test**: Runs full test suite
- **General hooks**: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files, check-merge-conflict

To install hooks: `pre-commit install`
Run hooks before committing (almost always right): `pre-commit run`

## Defensive Programming with Clippy

Volta uses a defensive linting approach inspired by [corrode.dev](https://corrode.dev/blog/defensive-programming/). The following Clippy lints are configured in `Cargo.toml`:

```toml
[lints.clippy]
indexing_slicing = "deny"          # Prevents direct indexing into slices
fallible_impl_from = "deny"        # Warns about panic-prone From impls
wildcard_enum_match_arm = "deny"   # Disallows wildcard _ patterns
unneeded_field_pattern = "deny"    # Identifies ignored struct fields
fn_params_excessive_bools = "deny" # Warns about too many bool params
must_use_candidate = "deny"        # Suggests #[must_use] attributes
```

**Pre-commit exceptions:**

- `clippy::needless_range_loop`: Allowed for readability in certain cases
- `clippy::too_many_arguments`: Allowed for functions with many parameters

**Key principles:**

- **No direct indexing**: Use `.get()`, iterators, or safe wrappers instead of `[]` indexing
- **No panicking From impls**: Use `TryFrom` for fallible conversions
- **Exhaustive matching**: Always handle all enum variants explicitly
- **Explicit field patterns**: Don't ignore struct fields with `..` without reason
- **Result-oriented**: Return `Result` types instead of panicking on errors

**Recent work:**

- All `indexing_slicing` errors resolved
- Comprehensive `VoltaError` types introduced for error handling
- ~400+ pedantic lints reduced to ~223 remaining
- Continual improvement toward full defensive compliance

## Code Quality

**Clippy Linting:**

Run `cargo clippy --all-targets` (not just `cargo clippy --all`) to catch linting errors in tests and examples. Fix all clippy warnings before committing.

## Testing

The test suite (tests in `src/lib.rs`) validates:

- **Core operations**: Basic ops, chain rule, tensor shapes
- **Gradient correctness**: All operations verified with numerical gradient checking
- **Broadcasting**: NumPy-style broadcasting rules
- **Neural networks**: Linear, Sequential, Conv2d, ConvTranspose2d, MaxPool2d layers
- **Optimizers**: Adam, SGD convergence tests
- **Edge cases**: Numerical edge cases, empty tensors, scalar tensors, shape operations

**Recent testing improvements:**

- Comprehensive Conv2d layer test coverage
- Comprehensive MaxPool2d layer test coverage
- Scalar tensor and empty tensor edge case tests
- Numerical edge case tests with improved error handling
- Gradient checks for missing unary operations

**Workflow Conventions:**

- Always run `cargo test` before committing changes
- If tests fail, fix them before proceeding with commit

### Gradient Checks

When implementing gradients, follow these guidelines:

- Verify gradient accumulation works correctly
- Avoid operations that produce non-deterministic results (e.g., argmax/argmin on ties)
- Ensure gradient shapes match input tensor shapes

## Important Implementation Details

### Serialization (State Dicts)

#### Named Layer Support

Sequential now supports optional layer names for more robust model serialization:

```rust
// Create Sequential with named layers
let model = Sequential::builder()
    .add_named("encoder", Box::new(Linear::new(784, 128, true)))
    .add_unnamed(Box::new(ReLU))  // Unnamed layers still work, but now called add_unnamed
    .add_named("decoder", Box::new(Linear::new(128, 10, true)))
    .build();

// State dict uses human-readable keys
let state = model.state_dict();
// Keys: "encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias"
```

**Benefits:**

- Human-readable state dict keys
- More robust to architecture changes (rename layers without breaking loading)
- Can retrieve layers by name: `model.get_named("encoder")`
- Full backward compatibility with numeric indices

**Backward Compatibility**: Old numeric format still works:

```rust
// Old style (still supported)
let model = Sequential::new(vec![
    Box::new(Linear::new(2, 8, true)),
    Box::new(ReLU),
]);
// Keys: "0.weight", "0.bias"

// Named models can load old numeric state dicts
let mut named_model = Sequential::builder()
    .add_named("layer1", Box::new(Linear::new(2, 8, true)))
    .add(Box::new(ReLU))
    .build();
named_model.load_state_dict(&old_numeric_state);  // Works!
```

#### External Model Loading

Load weights from PyTorch, HuggingFace, and other frameworks using weight mapping:

```rust
use volta::io::{load_safetensors, mapping::StateDictMapper};

// Load PyTorch model
let pytorch_state = load_safetensors("pytorch_model.safetensors")?;

// Create mapper: rename keys + transpose weights
let mapper = StateDictMapper::new()
    .rename("fc1.weight", "encoder.weight")
    .rename("fc1.bias", "encoder.bias")
    .transpose("encoder.weight")  // PyTorch: [out,in] → Volta: [in,out]
    .strip_prefix("model.");

let volta_state = mapper.map(pytorch_state);
model.load_state_dict(&volta_state);
```

**StateDictMapper transformations:**

- `rename(from, to)` - Rename single key
- `rename_prefix(old, new)` - Rename all keys with prefix
- `strip_prefix(prefix)` - Remove prefix from keys
- `add_prefix(prefix)` - Add prefix to keys
- `transpose(key)` - Transpose 2D weight matrix
- `transpose_pattern(pattern)` - Transpose all matching keys
- `select_keys(keys)` - Keep only specified keys
- `exclude_keys(keys)` - Remove specified keys
- `transform(fn)` - Custom transformation

**Why transpose?** PyTorch Linear stores weights as `[out_features, in_features]` and applies transpose during forward pass (`y = x @ W^T`). Volta stores weights as `[in_features, out_features]` and uses direct matmul (`y = x @ W`). When loading PyTorch weights into Volta, transpose is required.

See `examples/load_external_mnist.rs` for a complete end-to-end example.

### Conv2d Implementation

Uses algorithm selection in `src/nn/layers/conv.rs` with three available algorithms:

**1. Direct Convolution:**

- Computes each output pixel by iterating over kernel directly
- **Memory efficient**: No intermediate allocations
- **Best for**: Small inputs/kernels, memory-constrained scenarios
- **Trade-off**: Slower than im2col/iGEMM for larger inputs
- **✅ Full gradient support**: Input and weight gradients computed directly

**2. im2col + GEMM:**

- Materializes intermediate matrix (B*H_out*W_out, C*K*K) then uses GEMM
- **Fast**: Leverages optimized matrix multiplication
- **Best for**: Large kernels, general training
- **Trade-off**: High memory usage (can OOM on large inputs)
- **✅ Full gradient support**: Gradients via col2im and GEMM

**3. Implicit GEMM (iGEMM):**

- Performs GEMM without materializing the im2col matrix
- **Balanced**: Good trade-off between Direct and im2col
- **Best for**: Medium-to-large inputs where memory is a concern
- **Approach**: Tiled computation for cache efficiency
- **✅ Full gradient support**: Tiled gradient computation matches forward pass
- **🔧 Extensible**: Architecture supports future variants (Winograd, FFT, Direct-to-GEMM)

**Algorithm Selection:**

```rust
// Auto-select based on input characteristics
let conv = Conv2d::new(3, 16, 3, 1, 1, true);
conv.set_algo(ConvAlgo::Auto); // Default behavior

// Force specific algorithm
conv.set_algo(ConvAlgo::Direct);  // Memory efficient
conv.set_algo(ConvAlgo::Im2col);  // Faster but uses more memory
conv.set_algo(ConvAlgo::IGEMM);   // Balanced approach
```

**Auto-selection logic:**

- GPU: Always uses im2col (GPU-accelerated)
- CPU small inputs: Uses direct for small kernels (≤3x3) and small batches (≤4)
- CPU large inputs (>5M im2col elements): Uses direct to save memory
- CPU medium inputs: Uses iGEMM for balanced performance/memory
- **All algorithms support training** with full gradient computation

**iGEMM Architecture:**
The iGEMM implementation is designed with extensibility for future optimizations:

- **Tiled (current)**: Cache-friendly blocking with configurable tile sizes
- **Winograd (future)**: Transform-domain convolution for 3x3 kernels
- **FFT (future)**: Frequency-domain convolution for large kernels (audio)
- **Direct-to-GEMM (future)**: Hand-tuned micro-kernels for maximum performance

To add a new iGEMM variant:

1. Add variant to `IGEMMVariant` enum in `src/nn/layers/conv.rs`
2. Implement forward/backward in `Conv2d::igemm_forward`
3. Update `Conv2d::igemm_backward` to handle the variant
4. Add tests and benchmarks

**Recent improvements:**

- ✅ Direct convolution now supports full gradient computation
- ✅ iGEMM implementation with tiled forward and backward passes
- ✅ Extensible architecture for future convolution algorithms
- ✅ Memory-efficient training enabled for large inputs

### Performance Characteristics

- **Not optimized for speed (yet)**: Educational focus prioritizes correctness and clarity
- **BLAS acceleration available**: macOS can use Accelerate framework for matmul (`--features accelerate`)
- **Naive implementations**: Most operations use simple loops rather than SIMD or optimized kernels
- **Single-threaded**: Uses Rc instead of Arc, no parallelism

## Known Issues and Limitations

- **Single-threaded only**: Uses Rc<RefCell> instead of Arc<Mutex>
- **No distributed training**
- **No learning rate schedulers**
- **No RNN/Transformer layers** (LSTMCell exists but no full RNN/Transformer)
- **im2col memory inefficiency**: Addressed - direct convolution available for training with gradients
- **GPU direct convolution gradients**: GPU backward pass still CPU-only (future work)
- **Incomplete GPU support**: Some operations still CPU-only

## Code Conventions

When working with this codebase:

1. **All operations must have verified gradients**: Add numerical gradient check tests in `src/lib.rs`
2. **Use TensorOps trait**: User-facing API for all tensor operations
3. **Module trait for layers**: Implement `forward()`, `parameters()`, `state_dict()`, `load_state_dict()`
4. **GradFn for backprop**: Each operation implements the GradFn trait for reverse-mode autodiff
5. **Shape semantics**: Follow PyTorch conventions (NCHW for conv layers)
6. **Error handling**: Return `VoltaError` results instead of panicking where possible
7. **Defensive indexing**: Use `.get()`, iterators, or safe wrappers; avoid direct `[]` indexing

## Dependencies

### Rust

- `rand`, `rand_distr`: Random number generation
- `approx`: Numerical approximations for gradient checking
- `bincode`: Binary serialization for model weights
- `matrixmultiply`: GEMM operations
- `cblas-sys`: BLAS interface
- `blas-src` (optional): macOS Accelerate framework
- `wgpu` (optional): GPU compute via WebGPU
- `half`, `bytemuck`: Dtype support (f16, bf16)
- `safetensors`: SafeTensors format support for model loading
- `thiserror`: Error handling
- `serde_json`: for future config parsing
- `criterion`: Benchmarking framework (dev dependency)
