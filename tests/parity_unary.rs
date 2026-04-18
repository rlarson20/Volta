//! CPU/GPU parity for unary tensor ops.
//!
//! Each test runs the op on a fixed CPU tensor and a GPU copy of the same
//! tensor and asserts that outputs (and gradients, where applicable) match
//! within tolerance. Tests early-exit if no GPU is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_BWD_GPU, TOL_FWD_GPU, assert_unary_parity, assert_unary_parity_backward};
use volta::tensor::TensorOps;

const SHAPE_2D: &[usize] = &[4, 5];

#[test]
fn neg_parity() {
    assert_unary_parity_backward(
        "neg",
        |t| t.neg(),
        SHAPE_2D,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn recip_parity() {
    // Avoid 0; keep magnitudes well above eps to keep gradients well-conditioned.
    assert_unary_parity_backward(
        "recip",
        |t| t.recip(),
        SHAPE_2D,
        (0.5, 5.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn sqrt_parity() {
    assert_unary_parity_backward(
        "sqrt",
        |t| t.sqrt(),
        SHAPE_2D,
        (0.01, 5.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn exp_parity() {
    assert_unary_parity_backward(
        "exp",
        |t| t.exp(),
        SHAPE_2D,
        (-3.0, 3.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn log_parity() {
    assert_unary_parity_backward(
        "log",
        |t| t.log(),
        SHAPE_2D,
        (0.1, 5.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn exp2_parity() {
    assert_unary_parity_backward(
        "exp2",
        |t| t.exp2(),
        SHAPE_2D,
        (-3.0, 3.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn log2_parity() {
    assert_unary_parity_backward(
        "log2",
        |t| t.log2(),
        SHAPE_2D,
        (0.1, 5.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn sin_parity() {
    assert_unary_parity_backward(
        "sin",
        |t| t.sin(),
        SHAPE_2D,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn cos_parity() {
    assert_unary_parity_backward(
        "cos",
        |t| t.cos(),
        SHAPE_2D,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn tanh_parity() {
    assert_unary_parity_backward(
        "tanh",
        |t| t.tanh(),
        SHAPE_2D,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn sigmoid_parity() {
    assert_unary_parity_backward(
        "sigmoid",
        |t| t.sigmoid(),
        SHAPE_2D,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn relu_parity() {
    // Skip 0 to keep finite differences well-defined; otherwise sign at 0 is ambiguous.
    assert_unary_parity_backward(
        "relu",
        |t| t.relu(),
        SHAPE_2D,
        (-1.5, 1.5),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn erf_parity() {
    assert_unary_parity("erf", |t| t.erf(), SHAPE_2D, (-1.5, 1.5), TOL_FWD_GPU);
}

#[test]
fn unary_parity_3d_shape() {
    // Sanity check: helper handles a non-2D shape.
    assert_unary_parity(
        "sigmoid 3d",
        |t| t.sigmoid(),
        &[2, 3, 4],
        (-2.0, 2.0),
        TOL_FWD_GPU,
    );
}
