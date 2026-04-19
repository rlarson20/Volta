//! CPU/GPU parity for movement tensor ops.
//!
//! Each test runs the op on a fixed CPU tensor and a GPU copy of the same
//! tensor and asserts that outputs (and gradients, where applicable) match
//! within tolerance. Tests early-exit if no GPU is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{
    TOL_BWD_GPU, TOL_FWD_GPU, assert_close, assert_unary_parity, assert_unary_parity_backward,
    gpu_device, make_input, skip_if_no_gpu,
};
use volta::Device;
use volta::tensor::TensorOps;

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

#[test]
fn transpose_parity() {
    assert_unary_parity(
        "transpose",
        |t| t.transpose(),
        &[4, 5],
        (-2.0, 2.0),
        TOL_FWD_GPU,
    );
}

#[test]
fn transpose_backward_parity() {
    assert_unary_parity_backward(
        "transpose",
        |t| t.transpose(),
        &[4, 5],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

// ---------------------------------------------------------------------------
// reshape
// ---------------------------------------------------------------------------

#[test]
fn reshape_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], false, (-2.0, 2.0));
    let x_gpu = x_cpu.to_device(gpu_device());

    let y_cpu = x_cpu.reshape(&[6, 4]);
    let y_gpu = x_gpu.reshape(&[6, 4]).to_device(Device::CPU);

    assert_close(
        "reshape fwd",
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.borrow().data.to_vec(),
        TOL_FWD_GPU,
    );
}

#[test]
fn reshape_backward_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], true, (-2.0, 2.0));
    let x_gpu = x_cpu.to_device(gpu_device());

    let y_cpu = x_cpu.reshape(&[6, 4]);
    y_cpu.sum().backward();

    let y_gpu = x_gpu.reshape(&[6, 4]);
    y_gpu.sum().backward();

    assert_close(
        "reshape fwd",
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.to_device(Device::CPU).borrow().data.to_vec(),
        TOL_FWD_GPU,
    );

    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU gradient missing")
        .to_vec();
    assert_close("reshape bwd", &g_cpu, &g_gpu, TOL_BWD_GPU);
}

// ---------------------------------------------------------------------------
// permute
// ---------------------------------------------------------------------------

#[test]
fn permute_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], false, (-2.0, 2.0));
    let x_gpu = x_cpu.to_device(gpu_device());

    // [2,3,4] permute [1,2,0] -> [3,4,2]
    let y_cpu = x_cpu.permute(&[1, 2, 0]);
    let y_gpu = x_gpu.permute(&[1, 2, 0]).to_device(Device::CPU);

    assert_close(
        "permute fwd",
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.borrow().data.to_vec(),
        TOL_FWD_GPU,
    );
}

#[test]
fn permute_backward_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], true, (-2.0, 2.0));
    let x_gpu = x_cpu.to_device(gpu_device());

    let y_cpu = x_cpu.permute(&[1, 2, 0]);
    y_cpu.sum().backward();

    let y_gpu = x_gpu.permute(&[1, 2, 0]);
    y_gpu.sum().backward();

    assert_close(
        "permute fwd",
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.to_device(Device::CPU).borrow().data.to_vec(),
        TOL_FWD_GPU,
    );

    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU gradient missing")
        .to_vec();
    assert_close("permute bwd", &g_cpu, &g_gpu, TOL_BWD_GPU);
}

// ---------------------------------------------------------------------------
// flatten  (via reshape, since TensorOps has no standalone flatten method)
// ---------------------------------------------------------------------------

#[test]
fn flatten_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], false, (-2.0, 2.0));
    let x_gpu = x_cpu.to_device(gpu_device());

    // Flatten dims 1.. : [2, 3, 4] -> [2, 12]
    let y_cpu = x_cpu.reshape(&[2, 12]);
    let y_gpu = x_gpu.reshape(&[2, 12]).to_device(Device::CPU);

    assert_close(
        "flatten fwd",
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.borrow().data.to_vec(),
        TOL_FWD_GPU,
    );
}

#[test]
fn flatten_backward_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], true, (-2.0, 2.0));
    let x_gpu = x_cpu.to_device(gpu_device());

    let y_cpu = x_cpu.reshape(&[2, 12]);
    y_cpu.sum().backward();

    let y_gpu = x_gpu.reshape(&[2, 12]);
    y_gpu.sum().backward();

    assert_close(
        "flatten fwd",
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.to_device(Device::CPU).borrow().data.to_vec(),
        TOL_FWD_GPU,
    );

    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU gradient missing")
        .to_vec();
    assert_close("flatten bwd", &g_cpu, &g_gpu, TOL_BWD_GPU);
}

// ---------------------------------------------------------------------------
// pow — not available in TensorOps; no `pow` method exists on Tensor.
// Skipping as the operation is not implemented in the framework.
// ---------------------------------------------------------------------------
