//! GPU `exp2` uses WGSL's `exp2` builtin, not `pow(2.0, x)` (#311), and stays within the
//! accuracy WGSL specifies for it: `3 + 2 * |x|` ulps (WGSL § "Floating Point Accuracy").
//!
//! The parity tests' tolerance (1e-5) would hide an error of a few ulps, so these compare in
//! ulps, at fractional exponents with normal results. Subnormal results are out of scope:
//! WGSL lets an implementation flush them to zero, and lavapipe does (`exp2(-140)` is 0
//! there with either form). Runs on any adapter; lavapipe in a cloud session, where
//! `exp2(0.5)` is 2 ulps off with either form, inside the bound.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{gpu_device, gpu_leaf, skip_if_no_gpu};
use volta::{Device, RawTensor};

/// The exponents: fractional, with normal f32 results.
const XS: [f32; 8] = [0.5, -0.5, 0.25, 1.5, 3.3, -7.75, 10.1, 0.1];

/// WGSL's accuracy bound for `exp2(x)`, in ulps.
fn wgsl_exp2_ulps(x: f32) -> f64 {
    3.0 + 2.0 * f64::from(x.abs())
}

/// Distance in ulps between two finite, same-sign, normal f32 values.
fn ulps(a: f32, b: f32) -> u32 {
    a.to_bits().abs_diff(b.to_bits())
}

/// The forward is within WGSL's bound of the CPU's `exp2` (correctly rounded to within an
/// ulp, so the bound is the GPU's error plus at most one).
#[test]
fn exp2_forward_at_fractional_exponents() {
    if skip_if_no_gpu() {
        return;
    }
    let xs = XS.to_vec();
    let cpu = RawTensor::new(xs.clone(), &[xs.len()], false).exp2().data();
    let gpu = RawTensor::new(xs.clone(), &[xs.len()], false)
        .to_device(gpu_device())
        .exp2()
        .to_device(Device::CPU)
        .data();
    assert_eq!(cpu.len(), xs.len());
    assert_eq!(gpu.len(), xs.len());
    for ((x, c), g) in xs.iter().zip(&cpu).zip(&gpu) {
        assert!(
            f64::from(ulps(*c, *g)) <= wgsl_exp2_ulps(*x) + 1.0,
            "exp2({x}): CPU {c}, GPU {g}, {} ulps",
            ulps(*c, *g)
        );
    }
}

/// The backward, `g * exp2(x) * ln 2`, at the same exponents: the forward's bound plus two
/// more roundings.
#[test]
fn exp2_backward_at_fractional_exponents() {
    if skip_if_no_gpu() {
        return;
    }
    let xs = XS.to_vec();
    let x_cpu = RawTensor::new(xs.clone(), &[xs.len()], true);
    let x_gpu = gpu_leaf(&x_cpu);
    x_cpu
        .exp2()
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    x_gpu
        .exp2()
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    let g_cpu = x_cpu.grad().expect("CPU gradient");
    let g_gpu = x_gpu.grad_storage().expect("GPU gradient").to_vec();
    assert_eq!(g_cpu.len(), xs.len());
    assert_eq!(g_gpu.len(), xs.len());
    for ((x, c), g) in xs.iter().zip(&g_cpu).zip(&g_gpu) {
        assert!(
            f64::from(ulps(*c, *g)) <= wgsl_exp2_ulps(*x) + 3.0,
            "d exp2({x}): CPU {c}, GPU {g}, {} ulps",
            ulps(*c, *g)
        );
    }
}
