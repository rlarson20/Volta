//! GPU `modulo` against the CPU's floored remainder (#302), pinning the contract
//! `BinaryOp::Mod` documents:
//! - an infinite divisor with a finite dividend, and the NaN cases (zero divisor, infinite
//!   dividend, NaN operand), match the CPU exactly;
//! - for `|a / b| < 2^23` the GPU result is in the floored range (zero or b's sign,
//!   `|r| < |b|`) and within `(|a| + |b|) * 2^-22` of the CPU's exact remainder, counted
//!   modulo `b`. Past `2^23` the GPU result is not the remainder (an exact kernel is #355),
//!   so no case here goes past it.
//!
//! Runs on any adapter; lavapipe in a cloud session (where WGSL's `fma` is not fused, so
//! the error bound is what is exercised, not exactness).

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{gpu_device, skip_if_no_gpu};
use volta::{Device, RawTensor};

/// `a mod b` elementwise on the CPU and on the GPU, as `(cpu, gpu)`.
fn both(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (ta, tb) = (
        RawTensor::new(a.to_vec(), &[a.len()], false),
        RawTensor::new(b.to_vec(), &[b.len()], false),
    );
    let cpu = ta.modulo(&tb).data();
    let gpu = ta
        .to_device(gpu_device())
        .modulo(&tb.to_device(gpu_device()))
        .to_device(Device::CPU)
        .data();
    (cpu, gpu)
}

/// Equal as values: the same bits, or both NaN, or both zero (the sign of a zero result
/// may differ between the backends).
fn same(c: f32, g: f32) -> bool {
    c.to_bits() == g.to_bits() || (c.is_nan() && g.is_nan()) || (c == 0.0 && g == 0.0)
}

/// Asserts the CPU gives `expected` and the GPU matches the CPU exactly, pair by pair.
fn assert_exact(a: &[f32], b: &[f32], expected: &[f32]) {
    let (cpu, gpu) = both(a, b);
    assert_eq!(cpu.len(), a.len());
    assert_eq!(gpu.len(), a.len());
    assert_eq!(expected.len(), a.len());
    let rows = a.iter().zip(b).zip(cpu.iter().zip(&gpu)).zip(expected);
    for (((&x, &y), (&c, &g)), &want) in rows {
        assert!(same(c, want), "CPU {x} mod {y}: {c} (expected {want})");
        assert!(same(c, g), "{x} mod {y}: CPU {c:e}, GPU {g:e}");
    }
}

/// Asserts the documented bound for finite pairs with `|a / b| < 2^23`: the GPU result is
/// in the floored range, and within `(|a| + |b|) * 2^-22` of the CPU's, modulo `b`.
fn assert_within_bound(a: &[f32], b: &[f32]) {
    let (cpu, gpu) = both(a, b);
    assert_eq!(cpu.len(), a.len());
    assert_eq!(gpu.len(), a.len());
    for ((&x, &y), (&c, &g)) in a.iter().zip(b).zip(cpu.iter().zip(&gpu)) {
        assert!(
            (f64::from(x) / f64::from(y)).abs() < f64::from(1u32 << 23),
            "{x} mod {y}: test input past the documented quotient bound"
        );
        assert!(
            g == 0.0 || ((g < 0.0) == (y < 0.0) && g.abs() < y.abs()),
            "{x} mod {y}: GPU {g:e} is outside the floored range"
        );
        let diff = (f64::from(c) - f64::from(g)).abs();
        let modular = diff.min(f64::from(y).abs() - diff).abs();
        let tol = (f64::from(x).abs() + f64::from(y).abs()) * 2f64.powi(-22);
        assert!(
            modular <= tol,
            "{x} mod {y}: CPU {c:e}, GPU {g:e}, distance mod |b| {modular:e} > {tol:e}"
        );
    }
}

/// Infinite divisors and the NaN cases are exact: before the fix the GPU gave NaN for every
/// infinite divisor (`inf * trunc(a / inf) = inf * 0`).
#[test]
fn special_divisors_match_cpu_exactly() {
    if skip_if_no_gpu() {
        return;
    }
    let inf = f32::INFINITY;
    let nan = f32::NAN;
    let rows: &[(f32, f32, f32)] = &[
        (5.0, inf, 5.0),
        (-5.0, inf, inf),
        (5.0, -inf, -inf),
        (-5.0, -inf, -5.0),
        (0.0, inf, 0.0),
        (inf, 3.0, nan),
        (-inf, inf, nan),
        (5.0, 0.0, nan),
        (nan, 1.0, nan),
        (1.0, nan, nan),
        (-7.0, 2.0, 1.0),
        (7.0, -3.0, -2.0),
        (5.5, 2.0, 1.5),
        (-0.0, 3.0, 0.0),
    ];
    let a: Vec<f32> = rows.iter().map(|r| r.0).collect();
    let b: Vec<f32> = rows.iter().map(|r| r.1).collect();
    let want: Vec<f32> = rows.iter().map(|r| r.2).collect();
    assert_exact(&a, &b, &want);
}

/// A deterministic sweep over both signs and quotients from below 1 up to just under 2^23:
/// each `b` is a non-round value and each `a` is `q * b` plus a fraction of `b`, so the
/// remainder is far from trivial. Every element must be within the documented bound.
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the sweep builds each pair in f64 and rounds it to the f32 operand on purpose"
)]
fn sweep_stays_within_the_bound_below_2_pow_23() {
    if skip_if_no_gpu() {
        return;
    }
    // A small LCG, so the sweep needs no RNG dependency and is the same on every run.
    let mut state: u32 = 0x2545_f491;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(state >> 8) / f64::from(1u32 << 24)
    };
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..4096 {
        // Quotient magnitude 2^e for e in [-4, 22], so |a / b| < 2^23.
        let e = -4.0 + 26.0 * next();
        let q = 2f64.powf(e);
        let divisor = (0.01 + 100.0 * next()) * if i % 2 == 0 { 1.0 } else { -1.0 };
        let dividend = q * divisor * if i % 3 == 0 { -1.0 } else { 1.0 };
        a.push(dividend as f32);
        b.push(divisor as f32);
    }
    assert_within_bound(&a, &b);
}

/// Exact multiples and near-multiples, where `a / b` rounds across an integer and the
/// truncated quotient is one off: the result must stay in the floored range and within the
/// bound, counted modulo `b` (a remainder just above 0 may come back just below `|b|`).
/// Includes the audit's `1000.3 mod 0.1` and `0.3 mod 0.1`, which the GPU gave as 0.
#[test]
fn near_integer_quotients_stay_within_the_bound() {
    if skip_if_no_gpu() {
        return;
    }
    let (mut a, mut b) = (vec![1000.3f32, 0.3], vec![0.1f32, 0.1]);
    for &divisor in &[0.1f32, 0.3, 0.7, 1.1, 3.0, -0.1, -2.5] {
        for k in [1.0f32, 2.0, 3.0, 7.0, 10.0, 1000.0, 65_536.0, 1_000_000.0] {
            let center = k * divisor;
            for dividend in [
                center,
                f32::from_bits(center.to_bits() + 1),
                f32::from_bits(center.to_bits() - 1),
                -center,
            ] {
                a.push(dividend);
                b.push(divisor);
            }
        }
    }
    assert_within_bound(&a, &b);
}
