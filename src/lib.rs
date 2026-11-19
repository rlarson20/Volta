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
// Library code states the broken contract in an `expect` instead of a bare `unwrap()`
// (post-Phase-5 audit A9). Tests may unwrap.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(not(feature = "gpu"))]
#[path = "gpu_stub.rs"]
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

// ===== ROOT RE-EXPORTS (#341; DECISIONS.md, "One path per item; the root re-exports the core") =====
//
// Every public item has one canonical path, in the module that owns it, and at most one root
// re-export. The root re-exports every public item of the eight core modules (`tensor`,
// `autograd`, `amp`, `dtype`, `device`, `device_policy`, `shape`, `error`) and nothing
// else. The domain modules are reached through their module only, with private leaf
// modules: `nn` (layers, containers, transformer blocks, optimizers), `ops` (op enums),
// `data`, `io` (and `io::mapping`), `utils`, `storage`, `gpu`. So `volta::Tensor` and
// `volta::nn::Conv2d`, as in `torch.Tensor` and `torch.nn.Conv2d`.

pub use amp::{AutocastGuard, GradScaler, autocast, autocast_dtype};
pub use autograd::{NoGradGuard, ScalarLoss, no_grad};
pub use device::Device;
pub use device_policy::{
    DevicePolicy, DevicePolicyGuard, FallbackEvent, MAX_FALLBACK_EVENTS, Pass, device_policy,
    take_fallback_events, with_device_policy,
};
pub use dtype::{DType, DTypePolicy, DTypePolicyGuard, dtype_policy, with_dtype_policy};
pub use error::{Result, VoltaError};
pub use shape::{Axis, MAX_NUMEL, Shape, Strides};
pub use tensor::{
    DataLoader, PadValue, RawTensor, Tensor, bce_loss, bce_with_logits_loss, check_gradients,
    check_gradients_simple, cross_entropy_loss, kl_divergence_gaussian, manual_seed, max_dim,
    mse_loss, nll_loss, ones, rand, randn, randn_like, softmax, sum_dim, zeros,
};

/// The README's Rust snippets, compiled as doctests so they keep up with the public API (S23).
/// Exists only under `cargo test --doc`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

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
#[path = "lib_tests/loss_tests.rs"]
mod loss_tests;

#[cfg(test)]
#[path = "lib_tests/gpu_tests.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "lib_tests/gpu_half_tests.rs"]
mod gpu_half_tests;

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
