//! Broadcasting binary ops and axis reductions on the GPU (#67, #60), forward and backward,
//! against the CPU under `DevicePolicy::Strict`.
//!
//! Binary ops broadcast by gathering each repeating operand to the result's shape (the
//! rank-8 strided gather), and their gradients are summed back to each operand's shape.
//! `sum_dim`, `mean_dim` and `max_dim` run on `reduce_dim.wgsl`, which has a one-thread-per-
//! output kernel for short axes and a one-workgroup-per-output kernel for axes of 128 or
//! more; both are covered, including `max_dim`'s tie and NaN rules.
//!
//! Every loss is a randomly weighted sum, so a gradient routed to the wrong element fails.
//! Skips cleanly when no GPU adapter is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_BWD_GPU, TOL_FWD_GPU, assert_close, gpu_leaf, skip_if_no_gpu};
use volta::{Axis, DType, DevicePolicy, RawTensor, Tensor, manual_seed, with_device_policy};

/// A binary op under test.
type BinOp = fn(&Tensor, &Tensor) -> Tensor;

/// The binary ops with GPU kernels, and the value range each is tested on.
const BINARY: [(&str, BinOp, (f32, f32)); 5] = [
    ("add", Tensor::add, (-2.0, 2.0)),
    ("sub", Tensor::sub, (-2.0, 2.0)),
    ("elem_mul", Tensor::elem_mul, (-2.0, 2.0)),
    ("div", Tensor::div, (0.5, 2.0)),
    ("modulo", Tensor::modulo, (1.0, 5.0)),
];

/// Shape pairs that broadcast: one side, both sides, unequal ranks, rank 6.
const PAIRS: [(&[usize], &[usize]); 6] = [
    (&[2, 3], &[3]),
    (&[3, 1], &[1, 4]),
    (&[2, 1, 4], &[3, 1]),
    (&[4], &[2, 3, 4]),
    (&[2, 3, 4], &[1]),
    (&[2, 1, 3, 1, 2, 2], &[3, 4, 1, 1]),
];

/// A seeded uniform leaf on `[lo, hi)` that requires grad.
fn uniform(shape: &[usize], (lo, hi): (f32, f32)) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = RawTensor::rand(&[n])
        .data()
        .iter()
        .map(|u| lo + u * (hi - lo))
        .collect();
    RawTensor::new(data, shape, true)
}

/// `out` weighted by `w` and summed, backpropagated.
fn backprop(out: &Tensor, w: &Tensor) {
    out.elem_mul(w).sum().into_loss().unwrap().backward();
}

/// Every binary op over every broadcasting pair matches the CPU, values and both gradients.
#[test]
fn broadcast_binary_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    for (name, op, range) in BINARY {
        for (sa, sb) in PAIRS {
            manual_seed(0x67);
            let (a, b) = (uniform(sa, range), uniform(sb, range));
            let (ga, gb) = (gpu_leaf(&a), gpu_leaf(&b));
            let out = op(&a, &b);
            let w = RawTensor::randn(out.shape().truncate_rank());
            let out_gpu = op(&ga, &gb);
            assert!(
                out_gpu.device().is_gpu(),
                "{name} {sa:?} {sb:?} left the GPU"
            );
            let what = format!("{name} {sa:?} {sb:?}");
            assert_close(&what, &out.data(), &out_gpu.data(), TOL_FWD_GPU);
            backprop(&out, &w);
            backprop(&out_gpu, &gpu_leaf(&w));
            assert!(
                ga.grad_storage().unwrap().is_gpu(),
                "{what}: a grad left the GPU"
            );
            assert!(
                gb.grad_storage().unwrap().is_gpu(),
                "{what}: b grad left the GPU"
            );
            assert_close(
                &format!("{what} da"),
                &a.grad().unwrap(),
                &ga.grad().unwrap(),
                TOL_BWD_GPU,
            );
            assert_close(
                &format!("{what} db"),
                &b.grad().unwrap(),
                &gb.grad().unwrap(),
                TOL_BWD_GPU,
            );
        }
    }
}

/// An axis reduction under test.
type AxisOp = fn(&Tensor, Axis, bool) -> Tensor;

/// The axis reductions.
const AXIS_OPS: [(&str, AxisOp); 3] = [
    ("sum_dim", Tensor::sum_dim),
    ("mean_dim", Tensor::mean_dim),
    ("max_dim", Tensor::max_dim),
];

/// Run `op` over `axis` of `x` on both devices, keepdim or not, and compare the values and
/// the input gradient under a random weighting. `tol` scales with the axis length.
fn check_axis(name: &str, op: AxisOp, x: &Tensor, axis: usize, keepdim: bool, tol: f32) {
    x.zero_grad();
    let gx = gpu_leaf(x);
    let what = format!("{name} {:?} axis {axis} keepdim {keepdim}", x.shape());
    let out = op(x, Axis(axis), keepdim);
    let out_gpu = op(&gx, Axis(axis), keepdim);
    assert!(out_gpu.device().is_gpu(), "{what} left the GPU");
    assert_eq!(out.shape(), out_gpu.shape(), "{what}: shape");
    let w = RawTensor::randn(out.shape().truncate_rank());
    assert_close(&what, &out.data(), &out_gpu.data(), tol);
    backprop(&out, &w);
    backprop(&out_gpu, &gpu_leaf(&w));
    assert!(
        gx.grad_storage().unwrap().is_gpu(),
        "{what}: grad left the GPU"
    );
    assert_close(
        &format!("{what} grad"),
        &x.grad().unwrap(),
        &gx.grad().unwrap(),
        tol,
    );
}

/// Every axis of a rank-3 tensor, both launch shapes (a 300-long axis takes the
/// workgroup-per-output kernel), keepdim and not.
#[test]
fn axis_reductions_match_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    for shape in [&[3usize, 4, 5][..], &[2, 300, 3], &[1000, 2]] {
        manual_seed(0x464);
        let x = uniform(shape, (-3.0, 3.0));
        for (name, op) in AXIS_OPS {
            for axis in 0..shape.len() {
                for keepdim in [false, true] {
                    check_axis(name, op, &x, axis, keepdim, 1e-4);
                }
            }
        }
    }
}

/// `max_dim` follows `torch.max(x, dim)` on both kernels: a tie goes to the first index, a
/// NaN wins over every number, and the first NaN wins over later ones, so the whole
/// gradient goes to that one element.
#[test]
fn max_dim_ties_and_nan_match_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    for len in [6usize, 300] {
        let mut ties = vec![1.0f32; len];
        *ties.get_mut(2).unwrap() = 5.0;
        *ties.get_mut(len - 1).unwrap() = 5.0;
        let mut nans = vec![0.5f32; len];
        *nans.get_mut(1).unwrap() = 9.0;
        *nans.get_mut(3).unwrap() = f32::NAN;
        *nans.get_mut(len - 2).unwrap() = f32::NAN;
        for row in [ties, nans] {
            let x = RawTensor::new(row, &[1, len], true);
            let gx = gpu_leaf(&x);
            let out = x.max_dim(Axis(1), false);
            let out_gpu = gx.max_dim(Axis(1), false);
            assert_eq!(
                out.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out_gpu
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "max_dim value, len {len}"
            );
            out.sum().into_loss().unwrap().backward();
            out_gpu.sum().into_loss().unwrap().backward();
            assert_eq!(
                x.grad().unwrap(),
                gx.grad().unwrap(),
                "max_dim grad, len {len}"
            );
        }
    }
}

/// Half dtypes reduce in f32 and round once, on both devices: `mean_dim` and `max_dim`
/// (whose backward writes interleaved elements of packed 16-bit words) match the CPU.
#[test]
fn axis_reductions_in_half_dtypes_match_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let _strict = with_device_policy(DevicePolicy::Strict);
    for dtype in [DType::F16, DType::BF16] {
        manual_seed(0x16);
        let x = uniform(&[3, 5, 7], (-2.0, 2.0)).detach().to_dtype(dtype);
        x.set_requires_grad(true);
        for (name, op) in [AXIS_OPS[1], AXIS_OPS[2]] {
            for axis in 0..3 {
                check_axis(name, op, &x, axis, false, 2e-2);
            }
        }
    }
}
