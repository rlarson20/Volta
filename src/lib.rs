//! # Volta
//!
//! A minimal automatic differentiation library implementing PyTorch-like tensor operations
//! from scratch in pure Rust. This library provides:
//! - Dynamic computation graphs for automatic differentiation
//! - Broadcasting support for tensor operations
//! - Common neural network operations (matmul, activations, etc.)
//! - Numerical gradient checking for validation
//!
//! ## Architecture
//!
//! The library uses reference-counted interior mutability (`Rc<RefCell<RawTensor>>`) to build
//! dynamic computation graphs. Each tensor operation creates new tensors and stores gradient
//! functions that know how to backpropagate through that operation.
// A library must not write to stdout/stderr unprompted — diagnostics go through
// the `log` facade. Scoped to non-test builds of the library crate: examples and
// integration tests are separate crates (unaffected), and unit-test modules may
// print freely. `utils::progress` is the one legitimate exception and opts out
// with a module-level `#[allow]`.
#![cfg_attr(not(test), deny(clippy::print_stdout, clippy::print_stderr))]
#[cfg(feature = "gpu")]
pub mod gpu;

pub mod dtype;
/// Error type and `Result` alias.
pub mod error;
pub(crate) mod index;
/// The float types kernels compute in (`f32`, `f64`) and the dtype dispatch.
pub(crate) mod real;
pub mod shape;
pub(crate) mod spec;
pub mod storage;

// Add to re-exports:
pub use autograd::ScalarLoss;
pub use dtype::{DType, DTypePolicy, DTypePolicyGuard, dtype_policy, with_dtype_policy};
pub use error::{Result, VoltaError};
pub use shape::{Axis, Shape, Strides};
pub use spec::PadValue;
pub use storage::Storage;

#[cfg(feature = "gpu")]
pub use gpu::{
    GpuBuffer, GpuContext, get_gpu_context, gpu_cleanup, gpu_compact, gpu_pending_count,
    gpu_pool_stats, gpu_sync, gpu_sync_threshold, is_gpu_available,
};

/// Automatic mixed precision: `autocast` and `GradScaler`.
pub mod amp;
/// Reverse-mode automatic differentiation: the graph edges and `backward`.
pub mod autograd;
/// Dataset loading and preprocessing.
pub mod data;
/// The `Device` a tensor lives on.
pub mod device;
pub mod device_policy;
/// Saving, loading, and remapping state dicts.
pub mod io;
/// Internal macros.
mod macros;
/// Layers, containers, optimizers, and transformer blocks.
pub mod nn;
/// Differentiable tensor operations.
pub mod ops;
/// The `Tensor` type.
pub mod tensor;
/// Small utilities (progress reporting).
pub mod utils;

// Re-export main types for easy access
pub use amp::{GradScaler, autocast};
pub use device::Device;
pub use device_policy::{
    DevicePolicy, DevicePolicyGuard, FallbackEvent, Pass, device_policy, take_fallback_events,
    with_device_policy,
};
pub use nn::layers::Dropout;
pub use nn::layers::flatten::Flatten;
pub use nn::optim::Optimizer;
pub use nn::{
    Adam, BatchNorm1d, BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, Embedding, GELU, LSTMCell,
    Layer, LayerNorm, Linear, MaxPool2d, Module, Muon, PixelShuffle, ReLU, RecurrentLayer, SGD,
    Sequential, SequentialBuilder, Sigmoid, Tanh,
};
pub use tensor::{RawTensor, Tensor};

// Main entry points

pub use tensor::{
    DataLoader, bce_loss, bce_with_logits_loss, check_gradients, check_gradients_simple,
    cross_entropy_loss, kl_divergence_gaussian, manual_seed, max_dim, mse_loss, new_tensor,
    nll_loss, ones, rand, randn, randn_like, softmax, sum_dim, zeros,
};

pub use data::{load_mnist_images, load_mnist_labels, normalize, to_one_hot};
pub use io::{
    TypedTensorData, load_safetensors, load_safetensors_raw, load_safetensors_with_mapping,
    load_state_dict_with_mapping, mapping, save_safetensors, save_safetensors_typed,
};
pub use utils::ProgressBar;

pub use ops::{BinaryOp, MovementOp, ReduceOp, TernaryOp, UnaryOp};

// ===== TESTS =====
//
// The crate-level test modules live in `src/lib_tests/`, one file each. `#[path]` keeps
// each module at the crate root, so test names (`unary_tests::..`) are unchanged.
//
// The test suite validates:
// - Basic operations (add, mul, etc.)
// - Gradient correctness (chain rule, broadcasting)
// - Complex scenarios (neural networks, matmul variants)
// - Numerical gradient checking (validates all gradients)

#[cfg(test)]
#[path = "lib_tests/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lib_tests/unary_tests.rs"]
mod unary_tests;

#[cfg(test)]
#[path = "lib_tests/binary_tests.rs"]
mod binary_tests;

#[cfg(test)]
#[path = "lib_tests/reduce_tests.rs"]
mod reduce_tests;

#[cfg(test)]
#[path = "lib_tests/ternary_tests.rs"]
mod ternary_tests;

#[cfg(test)]
#[path = "lib_tests/movement_tests.rs"]
mod movement_tests;

#[cfg(test)]
#[path = "lib_tests/misc_tests.rs"]
mod misc_tests;

// ===== EDGE CASE AND ERROR HANDLING TESTS =====
// This module tests that invalid operations panic with appropriate error messages.
// These tests document the validation behavior of shape operations.
#[cfg(test)]
#[path = "lib_tests/edge_case_tests.rs"]
mod edge_case_tests;

#[cfg(test)]
#[path = "lib_tests/axis_reduce_tests.rs"]
mod axis_reduce_tests;

#[cfg(test)]
#[path = "lib_tests/gpu_tests.rs"]
mod gpu_tests;

// ===== EXPANDED COVERAGE (P1) =====
//
// The following modules add direct unit tests for foundational ops and core
// neural-network layers that previously had only indirect coverage. They
// are declared the same way as the modules above and rely on
// `RawTensor::check_gradients_simple` for backward-pass validation.

#[cfg(test)]
#[path = "lib_tests/binary_ops_tests.rs"]
mod binary_ops_tests;

#[cfg(test)]
#[path = "lib_tests/unary_ops_tests.rs"]
mod unary_ops_tests;

#[cfg(test)]
#[path = "lib_tests/reduce_ops_tests.rs"]
mod reduce_ops_tests;

#[cfg(test)]
#[path = "lib_tests/matmul_tests.rs"]
mod matmul_tests;

#[cfg(test)]
#[path = "lib_tests/linear_tests.rs"]
mod linear_tests;

#[cfg(test)]
#[path = "lib_tests/activation_tests.rs"]
mod activation_tests;

#[cfg(test)]
#[path = "lib_tests/dropout_tests.rs"]
mod dropout_tests;

#[cfg(test)]
#[path = "lib_tests/batchnorm_tests.rs"]
mod batchnorm_tests;

#[cfg(test)]
#[path = "lib_tests/f64_tests.rs"]
mod f64_tests;
