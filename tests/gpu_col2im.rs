//! GPU `col2im` (the Im2col algorithm's input gradient) against the CPU, for window
//! geometries where an input position is covered by zero, one, or several output windows
//! (#309). The kernel now visits only the output positions whose window covers each input
//! position; these cases check the window bounds it computes: stride above the kernel
//! (positions no window covers), stride below it (overlapping windows), a large kernel with
//! padding, a non-square input, and batch > 1. Runs on any adapter; lavapipe in a cloud
//! session.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{assert_close, gpu_device, gpu_leaf, skip_if_no_gpu};
use volta::{Conv2d, ConvAlgo, Device, Module, RawTensor, manual_seed};

/// Input-gradient parity for an Im2col `Conv2d` with these settings, under a randomly
/// weighted loss so no gradient is uniform.
fn check(
    label: &str,
    (in_ch, out_ch, kernel, stride, padding): (usize, usize, usize, usize, usize),
    input: &[usize],
) {
    if skip_if_no_gpu() {
        return;
    }
    manual_seed(0x0309);
    let mut conv_cpu = Conv2d::new(in_ch, out_ch, kernel, stride, padding, true);
    conv_cpu.set_algo(ConvAlgo::Im2col);
    let mut conv_gpu = Conv2d::new(in_ch, out_ch, kernel, stride, padding, true);
    conv_gpu
        .load_state_dict(&conv_cpu.state_dict())
        .expect("same architecture");
    conv_gpu.to_device(gpu_device());
    conv_gpu.set_algo(ConvAlgo::Im2col);

    let x_cpu = RawTensor::randn(input);
    x_cpu.set_requires_grad(true);
    let x_gpu = gpu_leaf(&x_cpu);

    let y_cpu = conv_cpu.forward(&x_cpu);
    let y_gpu = conv_gpu.forward(&x_gpu);
    let w_cpu = RawTensor::randn(y_cpu.shape().truncate_rank());
    let w_gpu = gpu_leaf(&w_cpu.detach());
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

    let g_cpu = x_cpu.grad().expect("CPU input gradient");
    let g_gpu = x_gpu
        .grad_storage()
        .expect("GPU input gradient")
        .to_device(&Device::CPU)
        .to_vec();
    assert!(
        g_cpu.iter().any(|v| v.abs() > 1e-3),
        "{label}: degenerate gradient"
    );
    assert_close(&format!("{label} dX"), &g_cpu, &g_gpu, 1e-4);
}

#[test]
fn stride_above_kernel_leaves_uncovered_positions() {
    // kernel 2, stride 3: every third row and column is in no window; its gradient is 0.
    check("k2 s3", (2, 3, 2, 3, 0), &[1, 2, 11, 11]);
}

#[test]
fn overlapping_windows_stride_one() {
    check("k3 s1", (3, 4, 3, 1, 0), &[2, 3, 9, 7]);
}

#[test]
fn large_kernel_with_padding_and_stride_two() {
    check("k5 s2 p2", (2, 3, 5, 2, 2), &[2, 2, 13, 10]);
}

#[test]
fn kernel_equals_stride() {
    check("k2 s2", (1, 2, 2, 2, 0), &[1, 1, 8, 6]);
}
