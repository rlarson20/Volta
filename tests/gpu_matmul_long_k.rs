//! GPU matmul over a long contraction (#319): the forward and both backward kernels used to
//! keep one running f32 sum over the whole shared dimension, which loses about `k * 2^-24`
//! relative (4% low at `k = 2^22`, and stuck at 2^24 for a sum of ones past that). Each
//! kernel now sums each 16-wide tile into a partial and the partials pairwise, so the error
//! grows with `log2(k)`. Each test contracts copies of `0.1f32` and compares with the exact
//! sum in f64: `2^22` of them for the forward; 1,000,000 for the backward, whose long
//! dimension is also a dimension of the forward's output, and a launch over more than
//! 65535 * 16 output columns is past wgpu's dispatch limit (#305). Runs on any adapter;
//! lavapipe in a cloud session.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{gpu_device, gpu_leaf, skip_if_no_gpu};
use volta::{Device, RawTensor};

/// The forward's shared dimension: 2^22.
const K: usize = 1 << 22;

/// The backward tests' shared dimension, under the dispatch limit on the forward's
/// output (#305). A single chain is about 1% off here.
const K_BWD: usize = 1_000_000;

/// Relative error allowed against the exact sum: pairwise over 2^18 tile partials is a
/// few times 2^-24 * 18; the old single chain was 4e-2.
const REL_TOL: f64 = 1e-5;

/// `0.1f32 * k`, summed exactly (every term is the same f32, so the sum is a product).
fn exact(k: usize) -> f64 {
    f64::from(0.1f32) * f64::from(u32::try_from(k).expect("k fits u32"))
}

/// Asserts `got` is within `REL_TOL` of the exact sum of `k` terms.
fn assert_exact(label: &str, k: usize, got: f32) {
    let want = exact(k);
    let rel = ((f64::from(got) - want) / want).abs();
    assert!(
        rel < REL_TOL,
        "{label}: GPU {got}, exact {want}, relative error {rel:e}"
    );
}

#[test]
fn forward_long_k() {
    if skip_if_no_gpu() {
        return;
    }
    // (1, K) @ (K, 1): one output, contracted over K.
    let a = RawTensor::new(vec![0.1; K], &[1, K], false).to_device(gpu_device());
    let b = RawTensor::new(vec![1.0; K], &[K, 1], false).to_device(gpu_device());
    let c = a.matmul(&b).to_device(Device::CPU).item();
    assert_exact("forward (1,K)@(K,1)", K, c);
}

#[test]
fn backward_a_long_contraction() {
    if skip_if_no_gpu() {
        return;
    }
    // C = A @ B with A (1, 1) and B (1, K_BWD); loss sum(C). dA = 1 @ B^T sums B.
    let a = gpu_leaf(&RawTensor::new(vec![1.0], &[1, 1], true));
    let b = RawTensor::new(vec![0.1; K_BWD], &[1, K_BWD], false).to_device(gpu_device());
    a.matmul(&b)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    let da = a.grad_storage().expect("dA").to_vec();
    assert_eq!(da.len(), 1);
    assert_exact("backward dA = grad @ B^T", K_BWD, da[0]);
}

#[test]
fn backward_b_long_contraction() {
    if skip_if_no_gpu() {
        return;
    }
    // C = A @ B with A (K_BWD, 1) and B (1, 1); loss sum(C). dB = A^T @ 1 sums A.
    let a = RawTensor::new(vec![0.1; K_BWD], &[K_BWD, 1], false).to_device(gpu_device());
    let b = gpu_leaf(&RawTensor::new(vec![1.0], &[1, 1], true));
    a.matmul(&b)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    let db = b.grad_storage().expect("dB").to_vec();
    assert_eq!(db.len(), 1);
    assert_exact("backward dB = A^T @ grad", K_BWD, db[0]);
}
