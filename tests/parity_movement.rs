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
    expect_gaps, gpu_leaf, make_input, skip_if_no_gpu,
};
use volta::Axis;
use volta::Device;

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

#[test]
fn transpose_parity() {
    expect_gaps(&["Transpose"], || {
        assert_unary_parity(
            "transpose",
            volta::Tensor::transpose,
            &[4, 5],
            (-2.0, 2.0),
            TOL_FWD_GPU,
        );
    });
}

#[test]
fn transpose_backward_parity() {
    expect_gaps(&["Transpose"], || {
        assert_unary_parity_backward(
            "transpose",
            volta::Tensor::transpose,
            &[4, 5],
            (-2.0, 2.0),
            TOL_FWD_GPU,
            TOL_BWD_GPU,
        );
    });
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
    let x_gpu = gpu_leaf(&x_cpu);

    let y_cpu = x_cpu.reshape(&[6, 4]);
    let y_gpu = x_gpu.reshape(&[6, 4]).to_device(Device::CPU);

    assert_close("reshape fwd", &y_cpu.data(), &y_gpu.data(), TOL_FWD_GPU);
}

#[test]
fn reshape_backward_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], true, (-2.0, 2.0));
    let x_gpu = gpu_leaf(&x_cpu);

    let y_cpu = x_cpu.reshape(&[6, 4]);
    y_cpu.sum().into_loss().expect("scalar loss").backward();

    let y_gpu = x_gpu.reshape(&[6, 4]);
    y_gpu.sum().into_loss().expect("scalar loss").backward();

    assert_close(
        "reshape fwd",
        &y_cpu.data(),
        &y_gpu.to_device(Device::CPU).data(),
        TOL_FWD_GPU,
    );

    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu.grad_storage().expect("GPU gradient missing").to_vec();
    assert_close("reshape bwd", &g_cpu, &g_gpu, TOL_BWD_GPU);
}

// ---------------------------------------------------------------------------
// contiguous
// ---------------------------------------------------------------------------

#[test]
fn contiguous_backward_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], true, (-2.0, 2.0));
    let x_gpu = gpu_leaf(&x_cpu);
    // Distinct weights, so a gradient routed to the wrong element cannot pass.
    let w_cpu = make_input(&[2, 3, 4], false, (0.5, 3.0));
    let w_gpu = gpu_leaf(&w_cpu);

    let y_cpu = volta::RawTensor::contiguous(&x_cpu);
    let y_gpu = volta::RawTensor::contiguous(&x_gpu);
    assert!(
        y_gpu.device().is_gpu(),
        "contiguous moved a GPU tensor to {}",
        y_gpu.device()
    );

    y_cpu
        .elem_mul(&w_cpu)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    y_gpu
        .elem_mul(&w_gpu)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();

    assert_close(
        "contiguous fwd",
        &y_cpu.data(),
        &y_gpu.to_device(Device::CPU).data(),
        TOL_FWD_GPU,
    );
    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu.grad_storage().expect("GPU gradient missing").to_vec();
    assert_close("contiguous bwd", &g_cpu, &g_gpu, TOL_BWD_GPU);
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
    let x_gpu = gpu_leaf(&x_cpu);

    // [2,3,4] permute [1,2,0] -> [3,4,2]
    let y_cpu = x_cpu.permute(&[Axis(1), Axis(2), Axis(0)]);
    let y_gpu = x_gpu
        .permute(&[Axis(1), Axis(2), Axis(0)])
        .to_device(Device::CPU);

    assert_close("permute fwd", &y_cpu.data(), &y_gpu.data(), TOL_FWD_GPU);
}

#[test]
fn permute_backward_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], true, (-2.0, 2.0));
    let x_gpu = gpu_leaf(&x_cpu);

    let y_cpu = x_cpu.permute(&[Axis(1), Axis(2), Axis(0)]);
    y_cpu.sum().into_loss().expect("scalar loss").backward();

    let y_gpu = x_gpu.permute(&[Axis(1), Axis(2), Axis(0)]);
    y_gpu.sum().into_loss().expect("scalar loss").backward();

    assert_close(
        "permute fwd",
        &y_cpu.data(),
        &y_gpu.to_device(Device::CPU).data(),
        TOL_FWD_GPU,
    );

    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu.grad_storage().expect("GPU gradient missing").to_vec();
    assert_close("permute bwd", &g_cpu, &g_gpu, TOL_BWD_GPU);
}

// ---------------------------------------------------------------------------
// flatten  (via reshape, since Tensor has no standalone flatten method)
// ---------------------------------------------------------------------------

#[test]
fn flatten_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], false, (-2.0, 2.0));
    let x_gpu = gpu_leaf(&x_cpu);

    // Flatten dims 1.. : [2, 3, 4] -> [2, 12]
    let y_cpu = x_cpu.reshape(&[2, 12]);
    let y_gpu = x_gpu.reshape(&[2, 12]).to_device(Device::CPU);

    assert_close("flatten fwd", &y_cpu.data(), &y_gpu.data(), TOL_FWD_GPU);
}

#[test]
fn flatten_backward_parity() {
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(&[2, 3, 4], true, (-2.0, 2.0));
    let x_gpu = gpu_leaf(&x_cpu);

    let y_cpu = x_cpu.reshape(&[2, 12]);
    y_cpu.sum().into_loss().expect("scalar loss").backward();

    let y_gpu = x_gpu.reshape(&[2, 12]);
    y_gpu.sum().into_loss().expect("scalar loss").backward();

    assert_close(
        "flatten fwd",
        &y_cpu.data(),
        &y_gpu.to_device(Device::CPU).data(),
        TOL_FWD_GPU,
    );

    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu.grad_storage().expect("GPU gradient missing").to_vec();
    assert_close("flatten bwd", &g_cpu, &g_gpu, TOL_BWD_GPU);
}

// ---------------------------------------------------------------------------
// pow — not available; no `pow` method exists on Tensor.
// Skipping as the operation is not implemented in the framework.
// ---------------------------------------------------------------------------

/// Rank 5 is past the GPU permute kernels (rank <= 4), so forward and backward are gaps
/// and the backward runs its CPU path on a GPU gradient. That path once relabeled the
/// GPU storage as CPU instead of copying it to the host, and panicked.
#[test]
fn permute_rank5_backward_gap_parity() {
    if skip_if_no_gpu() {
        return;
    }
    expect_gaps(&["Movement(Permute)"], || {
        let shape = [2, 1, 3, 2, 2];
        let axes = [Axis(4), Axis(2), Axis(0), Axis(3), Axis(1)];
        let x_cpu = make_input(&shape, true, (-2.0, 2.0));
        let w_cpu = make_input(&[2, 3, 2, 2, 1], false, (0.5, 3.0));
        x_cpu
            .permute(&axes)
            .elem_mul(&w_cpu)
            .sum()
            .into_loss()
            .expect("scalar")
            .backward();

        let x_gpu = gpu_leaf(&x_cpu);
        let w_gpu = gpu_leaf(&w_cpu);
        let out = x_gpu.permute(&axes);
        assert!(
            out.device().is_gpu(),
            "PreferGpu moves the gap's result back"
        );
        out.elem_mul(&w_gpu)
            .sum()
            .into_loss()
            .expect("scalar")
            .backward();

        assert_close(
            "permute rank-5 grad",
            &x_cpu.grad().expect("CPU grad"),
            &x_gpu.grad().expect("GPU grad"),
            TOL_BWD_GPU,
        );
    });
}
