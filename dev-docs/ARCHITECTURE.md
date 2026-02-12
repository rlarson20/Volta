# Architecture Documentation

This document provides detailed architecture documentation for the Volta deep learning framework.

## Core Design Philosophy

Volta is built around a few fundamental components that work together to provide a PyTorch-like deep learning experience.

### Tensor Design (`src/tensor.rs`)

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

**Key design choices:**

- **Rc<RefCell>** for dynamic computation graphs (single-threaded)
- **Gradient functions as trait objects** for polymorphic backpropagation
- **Storage abstraction** for CPU/GPU portability
- **Row-major flat vectors** for efficient memory layout

### Storage System (`src/storage.rs`)

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

### Device Abstraction (`src/device.rs`)

```rust
pub enum Device {
    CPU,
    GPU(String),  // GPU device identifier (e.g., "CUDA", "Metal")
}
```

Tensors can be moved between devices using `to_device()`, and operations automatically dispatch based on tensor device location. The `Module` trait also supports `to_device()` for moving all parameters at once.

### Error Handling (`src/error.rs`)

Recent refactor introduced `VoltaError` types using `thiserror` for comprehensive error handling:

- **VoltaError** enum with specific error variants (BroadcastError, ShapeError, DeviceError, etc.)
- **Result types** instead of panics where possible
- **Error propagation** with context throughout the codebase
- All operations now fallibly handle edge cases (bounds checking, device compatibility, etc.)

### Autograd Engine (`src/autograd.rs`)

- Reverse-mode (backpropagation) via topological DFS
- Topological sorting prevents recomputation in diamond graphs
- Gradient accumulation in parent tensors
- All operations verified with numerical gradient checking

**Key Design Pattern - Computation Graph:**

```rust
// Each tensor stores its computation graph dependencies
pub struct RawTensor {
    pub parents: Vec<Tensor>,           // Input tensors to this operation
    pub grad_fn: Option<Box<dyn GradFn>>, // How to compute gradients backward
    // ...
}

// Backward traversal uses post-order DFS to build topological order
// This ensures all gradients from a node's consumers are accumulated
// before computing gradients for that node's parents
```

## Operation Organization (`src/ops/`)

- **unary.rs**: Element-wise ops (sigmoid, relu, sqrt, exp, log, sin, cos, tanh)
- **binary.rs**: Binary ops (add, sub, elem_mul, div, max_elem, pow)
- **matmul.rs**: Matrix multiplication with broadcasting
- **reduce.rs**: Reduction ops (sum, mean, max_reduce)
- **ternary.rs**: Three-operand ops (mulacc, where)
- **movement.rs**: Shape manipulation (reshape, permute, expand, transpose, pad, shrink, stride)
- **gpu_ops.rs**: GPU-specific operation implementations

**Operation Pattern:**

Each operation type follows a consistent pattern:

1. **Enum definition** (e.g., `BinaryOp::Add`, `UnaryOp::Sigmoid`)
2. **Forward implementation** in operation-specific functions
3. **GradFn implementation** for backward pass
4. **TensorOps wrapper** for user-facing API
5. **GPU kernel** (in `src/gpu/kernels.rs`) for GPU acceleration

## Neural Network Layers (`src/nn/layers/`)

- **linear.rs**: Fully connected layer
- **conv.rs**: 2D convolution with algorithm selection (Direct, im2col + GEMM, Auto)
  - **Direct convolution**: Memory-efficient, computes each output pixel by iterating over kernel
  - **im2col + GEMM**: Faster for large kernels, materializes intermediate matrix
  - **Auto-selection**: Chooses algorithm based on input size, kernel size, batch size, and device
  - **Full gradient support**: Both algorithms support forward and backward passes
- **conv_transpose.rs**: Transposed convolution (for GANs/VAEs)
- **maxpool.rs**: Max pooling with gradient support
- **batchnorm.rs**: Batch normalization (1d and 2d)
- **dropout.rs**: Stochastic regularization
- **relu.rs**, **sigmoid.rs**, **tanh.rs**: Activation functions
- **sequential.rs**: Container for composing layers
- **sequential_builder.rs**: Builder pattern for named layers
- **flatten.rs**: Reshape to 1D
- **embedding.rs**: Embedding layer for discrete inputs
- **lstm.rs**: LSTM cell for recurrent architectures
- **pixelshuffle.rs**: Pixel shuffle for super-resolution

## Optimizers (`src/nn/optim/`)

- **sgd.rs**: Stochastic gradient descent with momentum and weight decay
- **adam.rs**: Adam optimizer with bias correction and weight decay
- **muon.rs**: Experimental Muon optimizer

## GPU Support (`src/gpu/`)

WGPU-based acceleration for core tensor operations.

### GPU Architecture

```rust
// Global singleton GPU context (lazy initialization)
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

// GPU buffers are reference-counted for efficient sharing
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: usize,
}
```

### GPU Safety Systems

- **Buffer Pooling**: 64-buffer pool to prevent allocation exhaustion
- **Command Queue Throttling**: Prevents system hangs from queue buildup
- **CPU Cache Invalidation**: Lazy caching with automatic invalidation
- **Early Warning System**: Monitors GPU health and resource usage
- **Staging Buffer Pool**: Efficient CPU↔GPU transfers

### GPU-accelerated Operations

**✅ Fully GPU-accelerated:**

- **Binary ops**: add, sub, mul, div, max, mod, cmplt
- **Unary ops**: neg, exp, log, relu, sigmoid, tanh, sqrt, recip, exp2, log2, sin, cos
- **Reduction ops**: sum, max, mean
- **Matrix multiplication**: matmul with configurable workgroup sizes
- **Movement ops**: permute, expand, pad, shrink, stride
- **Backward pass**: GPU-accelerated gradients for core operations
- **Conv2d**: Fully GPU-accelerated convolution (Direct, im2col, and iGEMM algorithms)
  - Forward passes for all three algorithms
  - Backward passes (input and weight gradients) for all three algorithms
  - Auto-selection prefers iGEMM on GPU for inputs >1M elements

**❌ Still CPU-only:**

- Linear backward pass (gradient computation for Linear layers)
- Broadcasting preprocessing
- Loss functions

### GPU Development Guidelines

When implementing GPU operations:

1. Ensure both forward and backward passes are GPU-resident
2. Check for CPU fallback paths in binary ops (especially `to_f32_vec()`)
3. Verify bind group layouts match between forward/backward
4. Test with actual GPU tensors, not just CPU tensors

## Performance Characteristics

- **Not optimized for speed (yet)**: Educational focus prioritizes correctness and clarity
- **BLAS acceleration available**: macOS can use Accelerate framework for matmul (`--features accelerate`)
- **Naive implementations**: Most operations use simple loops rather than SIMD or optimized kernels
- **Single-threaded**: Uses Rc instead of Arc, no parallelism
