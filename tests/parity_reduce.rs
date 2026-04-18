//! CPU/GPU parity for reduction ops (forward only).
//!
//! Backward parity for reductions is exercised end-to-end in the unary and
//! binary parity tests (which call `.sum()` to produce a scalar loss). Here
//! we verify forward equivalence across whole-tensor and axis variants.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_FWD_GPU, assert_reduce_parity};
use volta::tensor::TensorOps;

#[test]
fn sum_whole_tensor_parity() {
    assert_reduce_parity("sum", |t| t.sum(), &[4, 5], TOL_FWD_GPU);
}

#[test]
fn mean_whole_tensor_parity() {
    assert_reduce_parity("mean", |t| t.mean(), &[4, 5], TOL_FWD_GPU);
}

#[test]
fn max_reduce_whole_tensor_parity() {
    assert_reduce_parity("max_reduce", |t| t.max_reduce(), &[4, 5], TOL_FWD_GPU);
}

#[test]
fn sum_dim_keepdim_parity() {
    assert_reduce_parity(
        "sum_dim keepdim",
        |t| t.sum_dim(1, true),
        &[4, 5],
        TOL_FWD_GPU,
    );
}

#[test]
fn sum_dim_no_keepdim_parity() {
    assert_reduce_parity("sum_dim", |t| t.sum_dim(0, false), &[4, 5], TOL_FWD_GPU);
}

#[test]
fn mean_dim_keepdim_parity() {
    assert_reduce_parity(
        "mean_dim keepdim",
        |t| t.mean_dim(1, true),
        &[4, 5],
        TOL_FWD_GPU,
    );
}

#[test]
fn mean_dim_no_keepdim_parity() {
    assert_reduce_parity("mean_dim", |t| t.mean_dim(0, false), &[4, 5], TOL_FWD_GPU);
}

#[test]
fn max_dim_keepdim_parity() {
    assert_reduce_parity(
        "max_dim keepdim",
        |t| t.max_dim(1, true),
        &[4, 5],
        TOL_FWD_GPU,
    );
}

#[test]
fn sum_3d_parity() {
    // Sanity: helper handles a non-2D shape.
    assert_reduce_parity("sum 3d", |t| t.sum(), &[2, 3, 4], TOL_FWD_GPU);
}
