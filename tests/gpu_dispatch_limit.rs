//! GPU 16x16 grid kernels past wgpu's 65535-workgroups-per-dimension limit (#305), and
//! iGEMM computing each output once (#307).
//!
//! The conv and matmul kernels used to launch an `(x, y, 1)` grid with one 16-wide tile per
//! workgroup on each axis, so an axis over 65535 * 16 = 1,048,560 elements failed wgpu
//! validation and panicked with wgpu's text. They now launch a flat list of tiles
//! (`Groups::Tiles`), which has no per-axis cap. Each test here crosses that size on the
//! axis named, and compares the GPU forward and gradients with the CPU. The small-shape
//! cases check the tile mapping with more than one tile on each axis. Runs on any adapter;
//! lavapipe in a cloud session.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{assert_close, gpu_device, gpu_leaf, make_input, skip_if_no_gpu};
use volta::{Conv2d, ConvAlgo, Device, Module, RawTensor, Tensor, manual_seed};

/// One more than the widest axis the old grid could launch (65535 tiles of 16).
const PAST_LIMIT: usize = 65_535 * 16 + 1;

/// A GPU tensor's values on the host.
fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::CPU).data()
}

/// Forward, input-gradient and weight-gradient parity for a `Conv2d` with these settings
/// and `algo`, under a randomly weighted loss so no gradient is uniform.
fn check_conv(
    label: &str,
    algo: ConvAlgo,
    (in_ch, out_ch, kernel, stride, padding): (usize, usize, usize, usize, usize),
    input: &[usize],
) {
    if skip_if_no_gpu() {
        return;
    }
    manual_seed(0x0305);
    let mut conv_cpu = Conv2d::new(in_ch, out_ch, kernel, stride, padding, false);
    conv_cpu.set_algo(algo);
    let mut conv_gpu = Conv2d::new(in_ch, out_ch, kernel, stride, padding, false);
    conv_gpu
        .load_state_dict(&conv_cpu.state_dict())
        .expect("same architecture");
    conv_gpu.to_device(gpu_device());
    conv_gpu.set_algo(algo);

    let x_cpu = RawTensor::randn(input);
    x_cpu.set_requires_grad(true);
    let x_gpu = gpu_leaf(&x_cpu);

    let y_cpu = conv_cpu.forward(&x_cpu);
    let y_gpu = conv_gpu.forward(&x_gpu);
    assert!(
        matches!(y_gpu.device(), Device::GPU(_)),
        "{label}: the GPU conv fell back to the CPU"
    );
    assert_close(&format!("{label} fwd"), &y_cpu.data(), &host(&y_gpu), 1e-4);

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

    let dx_cpu = x_cpu.grad().expect("CPU input gradient");
    let dx_gpu = x_gpu
        .grad_storage()
        .expect("GPU input gradient")
        .to_device(&Device::CPU)
        .to_vec();
    assert!(
        dx_cpu.iter().any(|v| v.abs() > 1e-3),
        "{label}: degenerate input gradient"
    );
    assert_close(&format!("{label} dX"), &dx_cpu, &dx_gpu, 1e-4);

    let params_cpu = conv_cpu.parameters();
    let params_gpu = conv_gpu.parameters();
    assert_eq!(params_cpu.len(), params_gpu.len());
    for (p_cpu, p_gpu) in params_cpu.iter().zip(&params_gpu) {
        let g_cpu = p_cpu.grad().expect("CPU weight gradient");
        let g_gpu = p_gpu
            .grad_storage()
            .expect("GPU weight gradient")
            .to_device(&Device::CPU)
            .to_vec();
        // Each weight gradient sums over a million positions: compare relative to its size.
        assert_close_rel(&format!("{label} dW"), &g_cpu, &g_gpu);
    }
}

/// Relative tolerance for the matmul checks. Every case here also sums over about a
/// million terms on some side (the long axis is a dimension of both an output and a
/// contraction), where the CPU and GPU summation orders differ by a few ulps of the
/// result; the shared absolute `TOL_MATMUL_*` cannot hold for sums near 1e5.
const MATMUL_REL_TOL: f32 = 1e-5;

/// `assert_close` with the tolerance scaled by the largest magnitude in `cpu`.
fn assert_close_rel(label: &str, cpu: &[f32], gpu: &[f32]) {
    let scale = cpu.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    assert_close(label, cpu, gpu, MATMUL_REL_TOL * scale);
}

/// Forward and both gradients of `a @ b` on the GPU against the CPU, under a randomly
/// weighted loss (`.sum()` would make dC all ones).
fn check_matmul(a_shape: &[usize], b_shape: &[usize]) {
    if skip_if_no_gpu() {
        return;
    }
    let label = format!("matmul {a_shape:?} @ {b_shape:?}");
    let a_cpu = make_input(a_shape, true, (-1.0, 1.0));
    let b_cpu = make_input(b_shape, true, (-1.0, 1.0));
    let (a_gpu, b_gpu) = (gpu_leaf(&a_cpu), gpu_leaf(&b_cpu));
    let c_cpu = a_cpu.matmul(&b_cpu);
    let c_gpu = a_gpu.matmul(&b_gpu);
    assert!(
        matches!(c_gpu.device(), Device::GPU(_)),
        "{label}: the GPU matmul fell back to the CPU"
    );
    assert_close_rel(&format!("{label} fwd"), &c_cpu.data(), &host(&c_gpu));

    manual_seed(0x0305);
    let w_cpu = RawTensor::randn(c_cpu.shape().truncate_rank());
    let w_gpu = gpu_leaf(&w_cpu);
    for (c, w) in [(&c_cpu, &w_cpu), (&c_gpu, &w_gpu)] {
        c.elem_mul(w)
            .sum()
            .into_loss()
            .expect("scalar loss")
            .backward();
    }
    for (name, cpu, gpu) in [("dA", &a_cpu, &a_gpu), ("dB", &b_cpu, &b_gpu)] {
        let g_cpu = cpu.grad().expect("CPU gradient");
        let g_gpu = gpu
            .grad_storage()
            .expect("GPU gradient")
            .to_device(&Device::CPU)
            .to_vec();
        assert_close_rel(&format!("{label} {name}"), &g_cpu, &g_gpu);
    }
}

/// `[1, 1, 1025, 1024]` through a 1x1 kernel: `B * H_out * W_out` (the y axis of the
/// forward and input-gradient grids) is 1,049,600.
const TALL: [usize; 4] = [1, 1, 1025, 1024];

#[test]
fn direct_conv_spatial_axis_past_limit() {
    check_conv("direct 1x1", ConvAlgo::Direct, (1, 2, 1, 1, 0), &TALL);
}

#[test]
fn igemm_conv_spatial_axis_past_limit() {
    check_conv("igemm 1x1", ConvAlgo::IGEMM, (1, 2, 1, 1, 0), &TALL);
}

#[test]
fn im2col_conv_spatial_axis_past_limit() {
    // The im2col matrix has 1,049,600 rows (the matmul's y axis), and col2im's y axis is
    // `B * H * W`, also 1,049,600.
    check_conv("im2col 1x1", ConvAlgo::Im2col, (1, 2, 1, 1, 0), &TALL);
}

#[test]
fn conv_tile_mapping_with_several_tiles_per_axis() {
    // 20 output channels (2 channel tiles), 2 * 34 * 34 inputs and 2 * 32 * 32 outputs
    // (hundreds of spatial tiles), 3 input channels for the input-gradient grid's x axis.
    for algo in [ConvAlgo::Direct, ConvAlgo::IGEMM, ConvAlgo::Im2col] {
        check_conv(
            &format!("{algo:?} 3x3"),
            algo,
            (3, 20, 3, 1, 0),
            &[2, 3, 34, 34],
        );
        check_conv(
            &format!("{algo:?} 3x3 s2 p1"),
            algo,
            (17, 5, 3, 2, 1),
            &[2, 17, 9, 11],
        );
    }
}

#[test]
fn matmul_output_columns_past_limit() {
    // Forward x axis is n; matmul_backward_b's output is (k, n), x axis n.
    check_matmul(&[2, 3], &[3, PAST_LIMIT]);
}

#[test]
fn matmul_shared_dimension_past_limit() {
    // matmul_backward_a's output is (m, k), x axis k; matmul_backward_b's is (k, n), y axis k.
    // The forward sums a million products; the tolerance is relative for that reason.
    check_matmul(&[2, PAST_LIMIT], &[PAST_LIMIT, 2]);
}

#[test]
fn matmul_output_rows_past_limit() {
    // Forward y axis is m; matmul_backward_a's output is (m, k), y axis m.
    check_matmul(&[PAST_LIMIT, 3], &[3, 2]);
}
