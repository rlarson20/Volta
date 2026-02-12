# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Instructions

Whenever you're corrected on something, or you learn something about the codebase, either add it to the relevant spot in this file, or add it under the Learned Instructions if it's too general for any other spot in this file.

## Learned Instructions

### Recent Refactoring Work (2025-2026)

**TensorOps Duplication Removal (COMPLETED):**
- Eliminated 107 lines of boilerplate from tensor operations
- TensorOps implementations now call helper functions directly (`binary_op()`, `unary_op()`, `reduce_op()`, `ternary_op()`)
- When adding new operations, follow the pattern in `src/ops/` - implement the operation logic in the appropriate file, then add a thin wrapper in TensorOps trait

**Conv2d Algorithm Improvements (COMPLETED):**
- Direct convolution implemented with full gradient support (memory-efficient)
- iGEMM (Implicit GEMM) implemented with tiled computation (balanced memory/performance)
- im2col+GEMM remains available but is memory-intensive
- Auto-selection chooses algorithm based on input size, kernel size, batch size, and device
- All three algorithms support both forward and backward passes on GPU with zero CPU fallback

**See `notes/REFACTOR_SUGGESTIONS.md` for ongoing and planned refactoring work.**

## Project Overview

Volta is a PyTorch-like deep learning framework written in pure Rust. It's an educational project designed to demystify automatic differentiation engines and neural network computation graphs.

**Key Characteristics:**
- **Educational focus**: Prioritizes correctness and clarity over performance
- **PyTorch-like API**: Familiar patterns for ML practitioners
- **Pure Rust**: No Python dependencies, suitable for Rust-native ML
- **Verified gradients**: All operations validated with numerical gradient checking
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

### GPU Support

```bash
# Build with GPU support
cargo build --features gpu
cargo test --features gpu -- --nocapture
cargo run --example gpu --features gpu

# Build with BLAS acceleration (macOS only)
cargo build --features accelerate
cargo test --features accelerate -- --nocapture
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
- **No RNN/Transformer layers** (LSTMCell exists but no full RNN/Transformer)
- **Incomplete GPU support**: Some operations still CPU-only (check `src/gpu/mod.rs` for current status)

**Known Anti-Patterns (from notes/REFACTOR_SUGGESTIONS.md):**
1. **Excessive Clone Operations**: 424+ clone operations across 45 files due to Rc<RefCell> design
2. **Gradient Function Boilerplate**: Each operation requires ~30 lines of repetitive GradFn code
3. **Recursive Movement Ops**: Deep nesting in movement operations (pad, shrink, stride)

## Code Conventions

When working with this codebase:

1. **All operations must have verified gradients**: Add numerical gradient check tests in `src/lib.rs`
2. **Use TensorOps trait**: User-facing API for all tensor operations
3. **Module trait for layers**: Implement `forward()`, `parameters()`, `state_dict()`, `load_state_dict()`
4. **GradFn for backprop**: Each operation implements the GradFn trait for reverse-mode autodiff
5. **Shape semantics**: Follow PyTorch conventions (NCHW for conv layers)
6. **Error handling**: Return `VoltaError` results instead of panicking where possible
7. **Defensive indexing**: Use `.get()`, iterators, or safe wrappers; avoid direct `[]` indexing

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

## Development Notes

- **scrapboard.md**: Contains development notes, TODOs, and clippy linting progress
- **notes/REFACTOR_SUGGESTIONS.md**: Detailed analysis of completed and planned refactoring work
- **responses/**: Directory containing LLM-assisted development recommendations
- **sys-prompts/**: System prompts for external LLM assistance (maintainer use only)

## Reference Documentation

For detailed information on specific topics, see:

- **Architecture**: `dev-docs/ARCHITECTURE.md` - Core design philosophy, tensor design, storage system, autograd engine, operation organization, neural network layers, GPU support
- **Benchmarking**: `dev-docs/BENCHMARKING.md` - Performance testing, convolution algorithm benchmarks, benchmarking guidelines
- **Development**: `dev-docs/DEVELOPMENT.md` - Adding new operations, adding new neural network layers, testing workflow conventions, gradient checks
- **Conv2D Implementation**: `dev-docs/CONV2D_IMPLEMENTATION.md` - Algorithm overview (Direct, im2col, iGEMM), algorithm selection, iGEMM architecture, recent improvements
- **Refactoring**: `notes/REFACTOR_SUGGESTIONS.md` - Ongoing and planned refactoring work
