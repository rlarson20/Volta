# Architecture Documentation

This document provides detailed architecture documentation for the Volta deep learning framework.

TODO: UPDATE THIS WITH PHASE 1 CHANGES

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
    pub(crate) grad_fn: Option<GradFn>, // Backprop function (closed enum)
    pub parents: Vec<Tensor>,       // Dependency graph
    pub device: Device,             // CPU/GPU
}
```

**Key design choices:**

- **Rc<RefCell>** for dynamic computation graphs (single-threaded)
- **A closed `GradFn` enum** for backpropagation (one variant per op; `edge()` is the only constructor of graph edges)
- **Storage abstraction** for CPU/GPU portability
- **Row-major flat vectors** for efficient memory layout

### Storage System (`src/storage.rs`)

```rust
pub enum Storage {
    Cpu(HostBuf), // F16(Arc<[f16]>) | BF16(..) | F32(..) | F64(..) | I32(..) | I64(..) | U8(..) | Bool(..)
    #[cfg(feature = "gpu")]
    Gpu(Arc<GpuStorage>) // GpuStorage { buffer: GpuBuffer, host: OnceLock<HostBuf> }; F32, F16, BF16 or Bool
}
```

This design enables:

- **Typed CPU buffers**: one shared, copy-on-write `Arc<[T]>` per dtype (f16, bf16, f32, f64, i32, i64, u8, bool), so element type and alignment are part of the type. Tensors of any of these dtypes can be built (`RawTensor::from_typed`), and cast (`to_dtype`). CPU kernels (ops, layers, optimizers) are generic over `real::Real` (f32, f64) and dispatch on the dtype; f16 and bf16 run the f32 instance and round each output; integer and bool tensors are data (movement copies bits, comparisons, `where`, indices). `Module::to_dtype` converts a model, buffers included. On the GPU, every shader is expanded per element type (`gpu::shader_template`): f32, native f16, and f16/bf16 packed two per `u32`.
- **Efficient GPU↔CPU transfers**: Lazy, write-once host copy (`OnceLock`) shared by clones of a GPU storage; writes copy first
- **SafeTensors compatibility**: the eight dtypes map one-to-one; any other SafeTensors dtype is a load error

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
    pub(crate) grad_fn: Option<GradFn>, // How to compute gradients backward
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
3. **`GradFn` variant** wrapping a per-op struct with a `backward` method, plus an arm in the `GradFn::backward` and `GradFn::is_trivial` matches (`src/autograd.rs`)
4. **`Tensor` method** (a one-line forward in `src/tensor.rs`) for the user-facing API
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

**✅ GPU kernels** (in f32, f16 and bf16; Bool for comparisons and `where`):

- **Binary ops**: add, sub, mul, div, max, mod (equal shapes; broadcasting is a device-policy gap)
- **Comparisons and `where`**: lt, le, gt, ge, eq, ne, `where` (equal shapes)
- **Casts**: `to_dtype` between f32, f16, bf16 and Bool
- **Unary ops**: neg, exp, log, relu, sigmoid, tanh, sqrt, recip, exp2, log2, sin, cos
- **Reduction ops**: sum, max, mean
- **Matrix multiplication**: matmul with configurable workgroup sizes
- **Movement ops**: permute, expand, pad, shrink, stride
- **Backward pass**: GPU-accelerated gradients for core operations
- **Conv2d**: Fully GPU-accelerated convolution (Direct, im2col, and iGEMM algorithms)
  - Forward passes for all three algorithms
  - Backward passes (input and weight gradients) for all three algorithms
  - Auto-selection prefers iGEMM on GPU for inputs >1M elements

**❌ Gaps** (a panic under `DevicePolicy::Strict`, CPU under `PreferGpu`; the full list is in `notes/TODO.md`, "Kernel gaps that `Strict` rejects"):

- Broadcasting binary ops, forward and backward (`Linear`'s bias add, the norm layers, GELU's constants)
- `sum_dim`/`max_dim` (so softmax and the norms), batched and 1-D `matmul`, movement above rank 4
- `Embedding`, `MaxPool2d`, `ConvTranspose2d` (which also panics with a bias: audit A1)

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
