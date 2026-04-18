//! CPU/GPU parity for binary tensor ops.
//!
//! Differentiable ops are checked forward + backward; non-differentiable ops
//! (modulo, cmplt) are checked forward only. All shapes are matched — Volta
//! does not support broadcasting on the GPU path.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_BWD_GPU, TOL_FWD_GPU, assert_binary_parity, assert_binary_parity_backward};
use volta::tensor::TensorOps;

const SHAPE: &[usize] = &[3, 4];

#[test]
fn add_parity() {
    assert_binary_parity_backward(
        "add",
        |a, b| a.add(b),
        SHAPE,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn sub_parity() {
    assert_binary_parity_backward(
        "sub",
        |a, b| a.sub(b),
        SHAPE,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn elem_mul_parity() {
    assert_binary_parity_backward(
        "elem_mul",
        |a, b| a.elem_mul(b),
        SHAPE,
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn div_parity() {
    // Keep divisor away from 0 to keep gradient magnitudes well-conditioned.
    assert_binary_parity_backward(
        "div",
        |a, b| a.div(b),
        SHAPE,
        (0.5, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn max_elem_parity_forward_only() {
    // Backward of max_elem can route gradient differently when ties occur —
    // the make_input ranges don't guarantee strict inequality between a/b.
    // Forward parity is the reliable invariant to check here.
    assert_binary_parity(
        "max_elem",
        |a, b| a.max_elem(b),
        SHAPE,
        (-2.0, 2.0),
        TOL_FWD_GPU,
    );
}

#[test]
fn modulo_parity_forward() {
    assert_binary_parity("modulo", |a, b| a.modulo(b), SHAPE, (1.0, 5.0), TOL_FWD_GPU);
}

#[test]
fn cmplt_parity_forward() {
    assert_binary_parity("cmplt", |a, b| a.cmplt(b), SHAPE, (-2.0, 2.0), TOL_FWD_GPU);
}
