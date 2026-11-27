//! CPU/GPU parity for matmul across the supported shape matrix.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_MATMUL_BWD, TOL_MATMUL_FWD, assert_matmul_parity, expect_gaps};

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
    expect_gaps(&[], || {
        assert_matmul_parity(&[4, 3], &[3], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
    });
}

#[test]
fn matmul_1d_2d_parity() {
    expect_gaps(&[], || {
        assert_matmul_parity(&[3], &[3, 5], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
    });
}

#[test]
fn matmul_1d_1d_dot_parity() {
    expect_gaps(&[], || {
        assert_matmul_parity(&[6], &[6], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
    });
}

/// Past one 16x16 tile and not a multiple of 16 on any side: the multi-tile `k` loop and the
/// partial-tile edges run against the CPU (GEMM report §4).
#[test]
fn matmul_2d_2d_multi_tile_partial_edges_parity() {
    assert_matmul_parity(&[37, 19], &[19, 23], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
}

/// Non-square, `k` larger than a tile, so a transposed operand cannot cancel out.
#[test]
fn matmul_2d_2d_tall_skinny_parity() {
    assert_matmul_parity(&[5, 41], &[41, 3], TOL_MATMUL_FWD, TOL_MATMUL_BWD);
}

/// Batched products, equal and broadcast batch dims, run on the GPU in one launch; a
/// broadcast operand's gradient is summed back over the repeated batch dims.
#[test]
fn matmul_batched_parity() {
    expect_gaps(&[], || {
        for (a, b) in [
            (&[2, 3, 4][..], &[2, 4, 5][..]),
            (&[1, 3, 4], &[3, 4, 5]),
            (&[2, 1, 3, 17], &[1, 3, 17, 5]),
            (&[2, 2, 2, 2, 2, 3, 4], &[2, 1, 2, 1, 2, 4, 3]),
        ] {
            assert_matmul_parity(a, b, TOL_MATMUL_FWD, TOL_MATMUL_BWD);
        }
    });
}
