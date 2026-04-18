//! CPU/GPU parity for matmul across the supported shape matrix.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_MATMUL_BWD, TOL_MATMUL_FWD, assert_matmul_parity};

#[test]
fn matmul_2d_2d_parity() {
    assert_matmul_parity(&[3, 4], &[4, 5], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
}

#[test]
fn matmul_2d_2d_square_parity() {
    assert_matmul_parity(&[8, 8], &[8, 8], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
}

#[test]
fn matmul_2d_1d_parity() {
    assert_matmul_parity(&[4, 3], &[3], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
}

#[test]
fn matmul_1d_2d_parity() {
    assert_matmul_parity(&[3], &[3, 5], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
}

#[test]
fn matmul_1d_1d_dot_parity() {
    assert_matmul_parity(&[6], &[6], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
}
