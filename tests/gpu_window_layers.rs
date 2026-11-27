//! `MaxPool2d`, `Embedding` and `ConvTranspose2d` on the GPU (#464, #67), forward and
//! backward, against the CPU under `DevicePolicy::Strict`.
//!
//! The kernels are `maxpool.wgsl` (backward recomputes each window's winner), `embedding.wgsl`
//! (backward sums each row's occurrences from a host-built index list) and
//! `conv_transpose.wgsl` (the fold and unfold around two matmuls). Each case covers more
//! geometry than the gate test (`gpu_train_step.rs`): padding, overlapping and rectangular
//! windows, strides above the kernel, repeated indices, batch above one, and no bias.
//! `MaxPool2d`'s tie and NaN rules are checked bit for bit.
//!
//! Every loss is a randomly weighted sum, so a gradient routed to the wrong element fails.
//! Skips cleanly when no GPU adapter is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_MATMUL_BWD, assert_close, gpu_device, gpu_leaf, skip_if_no_gpu};
use volta::nn::{ConvTranspose2d, Embedding, Layer, MaxPool2d, Module};
use volta::{DType, DevicePolicy, RawTensor, Tensor, manual_seed, with_device_policy};

/// `out` weighted by `w` and summed, backpropagated.
fn backprop(out: &Tensor, w: &Tensor) {
    out.elem_mul(w).sum().into_loss().unwrap().backward();
}

/// Copy `cpu`'s weights into `gpu` and move it to the GPU.
fn mirror(cpu: &dyn Module, gpu: &mut dyn Module) {
    gpu.load_state_dict(&cpu.state_dict())
        .expect("same architecture");
    gpu.to_device(gpu_device());
}

/// Run `cpu` and `gpu` (a mirrored copy) on `x`, and compare the output, the input gradient
/// (when `x` requires grad) and every parameter gradient.
fn check_layer(what: &str, cpu: &dyn Layer, gpu: &dyn Layer, x: &Tensor, tol: f32) {
    // Integer indices stay on the CPU (the GPU holds no integer dtype); the layer's weight
    // decides the device.
    let gx = if x.requires_grad() {
        gpu_leaf(x)
    } else if x.dtype().is_float() {
        x.detach().to_device(gpu_device())
    } else {
        x.clone()
    };
    let out = cpu.forward(x);
    let out_gpu = gpu.forward(&gx);
    assert!(out_gpu.device().is_gpu(), "{what}: output left the GPU");
    assert_eq!(out.shape(), out_gpu.shape(), "{what}: shape");
    assert_close(what, &out.data(), &out_gpu.data(), tol);
    let w = RawTensor::randn(out.shape().truncate_rank());
    backprop(&out, &w);
    backprop(&out_gpu, &gpu_leaf(&w));
    if x.requires_grad() {
        assert!(
            gx.grad_storage().unwrap().is_gpu(),
            "{what}: grad left the GPU"
        );
        assert_close(
            &format!("{what} input grad"),
            &x.grad().unwrap(),
            &gx.grad().unwrap(),
            tol,
        );
    } else {
        assert!(gx.grad().is_none(), "{what}: a frozen input got a gradient");
    }
    for (i, (p, g)) in cpu.parameters().iter().zip(&gpu.parameters()).enumerate() {
        assert!(
            g.grad_storage().unwrap().is_gpu(),
            "{what}: param {i} grad left the GPU"
        );
        assert_close(
            &format!("{what} param {i} grad"),
            &p.grad().unwrap(),
            &g.grad().unwrap(),
            tol,
        );
    }
}

/// A seeded random leaf of `shape` that requires grad.
fn leaf(shape: &[usize]) -> Tensor {
    let x = RawTensor::randn(shape);
    x.set_requires_grad(true);
    x
}

/// A `MaxPool2d` case: kernel, stride, padding, input shape.
type PoolCase = (
    (usize, usize),
    (usize, usize),
    (usize, usize),
    &'static [usize],
);

/// A `ConvTranspose2d` case: in and out channels, kernel, stride, padding, bias, input shape.
type ConvTCase = (usize, usize, usize, usize, usize, bool, &'static [usize]);

/// `MaxPool2d` over padding, overlapping and rectangular windows, and a stride larger
/// than the kernel (some inputs belong to no window and get zero).
#[test]
fn maxpool_geometries_match_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    let cases: [PoolCase; 5] = [
        ((2, 2), (2, 2), (0, 0), &[2, 3, 8, 8]),
        ((3, 3), (2, 2), (1, 1), &[2, 2, 7, 9]),
        ((3, 3), (1, 1), (1, 1), &[1, 3, 5, 5]),
        ((2, 3), (1, 2), (1, 1), &[3, 2, 6, 7]),
        ((2, 2), (3, 3), (0, 0), &[2, 1, 8, 8]),
    ];
    for (kernel, stride, padding, shape) in cases {
        manual_seed(0x464);
        let pool = MaxPool2d::with_params(kernel, stride, padding);
        let what = format!("MaxPool2d k{kernel:?} s{stride:?} p{padding:?} {shape:?}");
        check_layer(&what, &pool, &pool, &leaf(shape), 1e-6);
    }
}

/// `MaxPool2d` follows torch on the GPU: a tie goes to the first element in scan order, a
/// NaN beats every number and the last NaN wins, and each window's whole gradient goes to
/// its one winner. Overlapping windows make one element win several of them.
#[test]
fn maxpool_ties_and_nan_match_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    let mut v = vec![1.0f32; 36];
    v[7] = 5.0;
    v[8] = 5.0;
    v[14] = 5.0;
    v[20] = f32::NAN;
    v[21] = f32::NAN;
    v[33] = f32::NEG_INFINITY;
    let pool = MaxPool2d::with_params((3, 3), (1, 1), (1, 1));
    let x = RawTensor::new(v, &[1, 1, 6, 6], true);
    let gx = gpu_leaf(&x);
    let out = pool.forward(&x);
    let out_gpu = pool.forward(&gx);
    let bits = |t: &Tensor| t.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&out), bits(&out_gpu), "MaxPool2d tie/NaN values");
    out.sum().into_loss().unwrap().backward();
    out_gpu.sum().into_loss().unwrap().backward();
    assert_eq!(
        x.grad().unwrap(),
        gx.grad().unwrap(),
        "MaxPool2d tie/NaN grad"
    );
}

/// `MaxPool2d` in half dtypes: the backward writes interleaved elements of packed words.
#[test]
fn maxpool_in_half_dtypes_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    for dtype in [DType::F16, DType::BF16] {
        manual_seed(0x16);
        let x = RawTensor::randn(&[2, 3, 7, 7]).to_dtype(dtype);
        x.set_requires_grad(true);
        let pool = MaxPool2d::with_params((3, 3), (2, 2), (1, 1));
        check_layer(&format!("MaxPool2d {dtype:?}"), &pool, &pool, &x, 2e-2);
    }
}

/// `Embedding` with repeated, unused and multi-dimensional indices, in f32 and half dtypes.
#[test]
fn embedding_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    let idx = RawTensor::from_typed(vec![4i64, 0, 4, 9, 4, 2, 0, 11], &[2, 4]);
    for (dtype, tol) in [(DType::F32, 1e-6), (DType::F16, 2e-2), (DType::BF16, 2e-2)] {
        manual_seed(0xE);
        let mut cpu = Embedding::new(12, 5);
        cpu.to_dtype(dtype);
        let mut gpu = Embedding::new(12, 5);
        gpu.to_dtype(dtype);
        mirror(&cpu, &mut gpu);
        check_layer(&format!("Embedding {dtype:?}"), &cpu, &gpu, &idx, tol);
    }
}

/// `ConvTranspose2d` over strides 1 to 3, padding 0 to 2, rectangular inputs, batch above
/// one, with and without a bias, and with a frozen input (only the weight's gradient).
#[test]
fn conv_transpose_geometries_match_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    let cases: [ConvTCase; 5] = [
        (3, 4, 3, 2, 1, true, &[2, 3, 5, 5]),
        (2, 3, 4, 1, 0, false, &[3, 2, 4, 6]),
        (4, 2, 3, 3, 2, true, &[1, 4, 6, 3]),
        (1, 5, 1, 1, 0, true, &[2, 1, 3, 3]),
        (3, 3, 5, 2, 2, false, &[2, 3, 4, 4]),
    ];
    for (in_ch, out_ch, k, stride, pad, bias, shape) in cases {
        manual_seed(0x7C);
        let cpu = ConvTranspose2d::new(in_ch, out_ch, k, stride, pad, bias);
        let mut gpu = ConvTranspose2d::new(in_ch, out_ch, k, stride, pad, bias);
        mirror(&cpu, &mut gpu);
        let what =
            format!("ConvTranspose2d {in_ch}->{out_ch} k{k} s{stride} p{pad} b{bias} {shape:?}");
        check_layer(&what, &cpu, &gpu, &leaf(shape), TOL_MATMUL_BWD);
        let frozen = RawTensor::randn(shape);
        check_layer(
            &format!("{what} frozen"),
            &cpu,
            &gpu,
            &frozen,
            TOL_MATMUL_BWD,
        );
    }
}
