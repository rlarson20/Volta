# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Instructions

Whenever you're corrected on something, or you learn something about the codebase, either add it to the relevant spot in this file, or add it under the Learned Instructions if it's too general for any other spot in this file.

## Learned Instructions

### Recent Refactoring Work (2025-2026)

**Conv2d Algorithm Improvements (COMPLETED):**

- Direct convolution implemented with full gradient support (memory-efficient)
- iGEMM (Implicit GEMM) implemented with tiled computation (balanced memory/performance)
- im2col+GEMM remains available but is memory-intensive
- Auto-selection chooses algorithm based on input size, kernel size, batch size, and device
- All three algorithms support both forward and backward passes on GPU with zero CPU fallback

**See `docs/plans/ADDENDUM.md` for ongoing and planned refactoring work, and the current prioritized bug/improvement roadmap.**

**Feature flags gotcha:** `Cargo.toml` sets `default = ["gpu", "accelerate"]`, so plain `cargo build`/`cargo test`/`cargo clippy` compile WITH the GPU backend (contrary to the README's opt-in framing, and CI's "no features" step labels). Use `--no-default-features` for a CPU-only build.

**ADDENDUM.md Phase 1 (opaque `Tensor`, `edge()` constructor, `NoGrad`, `assert_trainable!`) — COMPLETED:**

- `Tensor` is now `struct Tensor { inner: Rc<RefCell<RawTensor>> }` (was a bare type alias). `.borrow()`/`.borrow_mut()`/`.as_ptr()` are `pub(crate)` forwarding methods, not renamed — this keeps every _internal_ `t.borrow()` call site compiling unchanged (method resolution is by name, not by whether `Tensor` derefs to `RefCell`), while making the fields genuinely unreachable from outside the crate.
- Because of that `pub(crate)` boundary, `tests/*.rs` and `examples/*.rs` (which only see the crate's public API, same as any downstream user) needed a small public API added to `Tensor` before they'd compile again: `shape()`, `data()`, `device()`, `grad_storage()`, `detach()`, a now-`pub` `set_requires_grad()`, and `set_grad(&self, grad: Storage)` (the second sanctioned writer of `grad` outside init, alongside `accumulate_grad`, added specifically so optimizer tests can seed a synthetic gradient without a real `backward()` pass). All call sites in `tests/` and `examples/` were rewritten to use it instead of raw `.borrow()`/`.borrow_mut()`.
- `NO_GRAD` (in `src/autograd.rs`) is a `thread_local!` reentrant depth counter (`Cell<u32>`), not a plain bool — needed so a `no_grad()` scope can nest inside another (e.g. a layer's own `no_grad()` running inside `check_gradients`'s wrapping one) without the inner guard's `Drop` prematurely re-enabling grad tracking.
- `edge()` (in `src/autograd.rs`) is now the only code that attaches `parents`/`grad_fn` to a tensor; all of `src/ops/*`, `src/nn/layers/conv.rs`, `conv_transpose.rs`, `embedding.rs`, and `maxpool.rs` go through it. Along the way this fixed a latent bug shared by `conv.rs`'s `direct_conv`/`igemm_forward` and `conv_transpose.rs`'s `forward`: they used to gate graph attachment on only the input tensor's `requires_grad`, silently ignoring the weight/bias tensors.

**Logging (Phase 0 "log facade", COMPLETED):** the library never prints to stdout/stderr. All diagnostics go through the `log` facade (`log::{error,warn,info,debug,trace}!`). It's silent unless a downstream binary installs a logger (`env_logger`, `tracing-subscriber`, …) — e.g. `RUST_LOG=volta=debug` or `RUST_LOG=volta::gpu=trace`. `#![cfg_attr(not(test), deny(clippy::print_stdout, clippy::print_stderr))]` in `lib.rs` enforces this; `src/utils/progress.rs` (interactive progress bar) is the one sanctioned exception via a module `#[allow]`. Don't add `println!`/`eprintln!` to `src/` — use `log`.

## Project Overview

Volta is a PyTorch-like deep learning framework written in pure Rust. It's an educational project designed to demystify automatic differentiation engines and neural network computation graphs.

**Key Characteristics:**

- **Educational focus**: Prioritizes correctness and clarity over performance
- **PyTorch-like API**: Familiar patterns for ML practitioners
- **Pure Rust**: No Python dependencies, suitable for Rust-native ML
- **GPU acceleration**: Experimental WGPU-based support for core operations
- **Extensible**: Clean architecture for adding new operations and layers

## Development Commands

### Quick Start

```bash
# Build and test
cargo check
cargo build
cargo test -- --nocapture

# Quick verification (runs check, build, and test)
just check

# Run examples
cargo run --example readme1                    # Simple MLP training
cargo run --example readme2                    # LeNet-style CNN
cargo run --example showcase                   # Feature showcase
cargo run --example load_external_mnist        # PyTorch → Volta loading
```

### Testing

```bash
# Run all tests with output
cargo test -- --nocapture

# Run specific test modules (organized by category in src/lib.rs)
cargo test core                # Core tensor operations
cargo test grad_check          # Numerical gradient validation
cargo test broadcasting        # Broadcasting rules
cargo test neural              # Neural network layers
cargo test optimizer           # Optimizer convergence tests
cargo test axis_reduce_tests   # Dimension reduction operations
cargo test shape_tests         # Shape manipulation tests

# Run a specific test
cargo test test_name -- --nocapture
```

## Testing Overview

**Test Organization:**

- **core**: Basic tensor operations, chain rule, shape manipulation
- **grad_check**: Numerical gradient validation for all operations
- **broadcasting**: NumPy-style broadcasting rules
- **neural**: Neural network layer tests (Linear, Conv2d, MaxPool2d, etc.)
- **optimizer**: Optimizer convergence tests (Adam, SGD)
- **axis_reduce_tests**: Dimension reduction operations (sum_dim, max_dim, softmax)
- **shape_tests**: Shape manipulation operations (reshape, permute, transpose)

**Test Coverage:**

- Core operations, gradient correctness, broadcasting
- Neural networks (Linear, Sequential, Conv2d, ConvTranspose2d, MaxPool2d)
- Optimizers (Adam, SGD)
- Edge cases (numerical edge cases, empty tensors, scalar tensors)

**Workflow:**

- Always run `cargo test` before committing changes
- Use `check_gradients_simple()` for basic gradient validation
- Use full `check_gradients()` for comprehensive validation

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

Volta uses a defensive linting approach. The following Clippy lints are configured in `Cargo.toml`:

```toml
[lints.clippy]
indexing_slicing = "deny"          # Prevents direct indexing into slices
fallible_impl_from = "deny"        # Warns about panic-prone From impls
wildcard_enum_match_arm = "deny"   # Disallows wildcard _ patterns
unneeded_field_pattern = "deny"    # Identifies ignored struct fields
fn_params_excessive_bools = "deny" # Warns about too many bool params
must_use_candidate = "deny"        # Suggests #[must_use] attributes
```

**Key principles:**

- **No direct indexing**: Use `.get()`, iterators, or safe wrappers instead of `[]` indexing
- **No panicking From impls**: Use `TryFrom` for fallible conversions
- **Exhaustive matching**: Always handle all enum variants explicitly
- **Explicit field patterns**: Don't ignore struct fields with `..` without reason
- **Result-oriented**: Return `Result` types instead of panicking on errors

## Code Quality

Run `cargo clippy --all-targets` (not just `cargo clippy --all`) to catch linting errors in tests and examples. Fix all clippy warnings before committing.

## Serialization (State Dicts)

The framework supports multiple serialization formats for model weights:

**Supported Formats:**

- **bincode**: Binary format for Volta-native model saving/loading
- **SafeTensors**: Industry-standard format for interoperability with PyTorch, HuggingFace, etc.

**Key Types:**

- `StateDict`: `BTreeMap<String, TensorData>` - Human-readable parameter names to tensor data
- `TensorData`: Serializable struct containing `data: Vec<f32>` and `shape: Vec<usize>`
- `StateDictDiff`: Debugging tool for comparing expected vs loaded state dicts

**Key Functions:**

- `save_state_dict()`: Save model weights to bincode format
- `load_state_dict()`: Load model weights from bincode format
- `load_safetensors()`: Load from SafeTensors format
- `save_safetensors()`: Save to SafeTensors format
- `load_safetensors_with_mapping()`: Load with automatic key/weight transformations

**Named Layer Support:**

Sequential now supports optional layer names for more robust model serialization. See `examples/load_external_mnist.rs` for a complete end-to-end example.

**External Model Loading:**

Load weights from PyTorch, HuggingFace, and other frameworks using weight mapping with `StateDictMapper`.

## Known Issues and Limitations

**Current Limitations:**

- **Single-threaded only**: Uses Rc<RefCell> instead of Arc<Mutex>
- **No distributed training**
- **No learning rate schedulers**
- **No full RNN layer** (LSTMCell exists; transformer components — `MultiHeadAttention`, `TransformerBlock`, `PositionalEncoding` — ARE implemented in `src/nn/transformer/`)
- **Incomplete GPU support**: Some operations still CPU-only (check `src/gpu/mod.rs` for current status)

## Code Conventions

When working with this codebase:

1. **All operations must have verified gradients**: Add numerical gradient check tests in `src/lib.rs`
2. **Module trait for layers**: Implement `forward()`, `parameters()`, `state_dict()`, `load_state_dict()`
3. **GradFn for backprop**: Each operation implements the GradFn trait for reverse-mode autodiff
4. **Shape semantics**: Follow PyTorch conventions (NCHW for conv layers)
5. **Error handling**: Return `VoltaError` results instead of panicking where possible
6. **Defensive indexing**: Use `.get()`, iterators, or safe wrappers; avoid direct `[]` indexing

## Performance Characteristics

- **Not optimized for speed (yet)**: Educational focus prioritizes correctness and clarity
- **BLAS acceleration available**: macOS can use Accelerate framework for matmul (`--features accelerate`)
- **Naive implementations**: Most operations use simple loops rather than SIMD or optimized kernels
- **Single-threaded**: Uses Rc instead of Arc, no parallelism

## Dependencies

### Core Dependencies

- `rand`, `rand_distr`: Random number generation
- `approx`: Numerical approximations for gradient checking
- `bincode`: Binary serialization for model weights
- `matrixmultiply`: GEMM operations
- `half`, `bytemuck`: Dtype support (f16, bf16)
- `safetensors`: SafeTensors format support for model loading
- `thiserror`: Error handling

### Optional Dependencies

- `blas-src` (macOS only): BLAS interface for Accelerate framework
- `wgpu`: GPU compute via WebGPU
- `pollster`: Async executor for GPU operations

### Development Dependencies

- `criterion`: Benchmarking framework

## Reference Documentation

For detailed information on specific topics, see:

- **Architecture**: `dev-docs/ARCHITECTURE.md` - Core design philosophy, tensor design, storage system, autograd engine, operation organization, neural network layers, GPU support
- **Benchmarking**: `dev-docs/BENCHMARKING.md` - Performance testing, convolution algorithm benchmarks, benchmarking guidelines
- **Development**: `dev-docs/DEVELOPMENT.md` - Adding new operations, adding new neural network layers, testing workflow conventions, gradient checks
- **Conv2D Implementation**: `dev-docs/CONV2D_IMPLEMENTATION.md` - Algorithm overview (Direct, im2col, iGEMM), algorithm selection, iGEMM architecture, recent improvements
- **Refactoring**: `notes/REFACTOR_SUGGESTIONS.md` - Ongoing and planned refactoring work
