//! GELU's negative tail on the GPU (#323, audit N11): `x/2 * (1 + erf(x / sqrt 2))`
//! cancels in `1 + erf` for negative `x`, so an absolute error in the GPU `erf` becomes a
//! large relative error in GELU. The decision on #323 is to keep the PyTorch formula and
//! make the GPU `erf` as accurate as the CPU's `libm::erff`; these tests pin that.
//!
//! Each test prints `GPU-RAN <name>` before its assertions, because a GPU test with no
//! adapter returns early and still reports `ok`.

#![cfg(feature = "gpu")]

mod common;

use common::{expect_gaps, gpu_leaf, skip_if_no_gpu};
use volta::nn::{GELU, Layer};
use volta::{RawTensor, Tensor};

/// Skip (returning `true`) when there is no GPU; otherwise print the marker for `name`.
fn skip_or_mark(name: &str) -> bool {
    if skip_if_no_gpu() {
        println!("GPU-SKIPPED {name}");
        true
    } else {
        println!("GPU-RAN {name}");
        false
    }
}

/// Distance in f32 units in the last place between two finite values of the same sign.
fn ulps(a: f32, b: f32) -> u32 {
    assert_eq!(
        a.is_sign_negative(),
        b.is_sign_negative(),
        "{a} and {b} differ in sign"
    );
    a.to_bits().abs_diff(b.to_bits())
}

/// The CPU `[n]` tensor of `xs`, without grad.
fn cpu(xs: &[f32]) -> Tensor {
    RawTensor::new(xs.to_vec(), &[xs.len()], false)
}

/// The GPU `erf` is within one ulp of the CPU's through the range where GELU's tail
/// cancels (|x| from 1 to 6, i.e. GELU inputs down to about -8.5), so `1 + erf` keeps the
/// CPU's accuracy. With Abramowitz-Stegun 7.1.26 the gap was 2 ulps at -3/sqrt(2).
#[test]
fn gpu_erf_is_within_one_ulp_of_the_cpu_in_the_tail() {
    if skip_or_mark("gpu_erf_is_within_one_ulp_of_the_cpu_in_the_tail") {
        return;
    }
    let mut xs = Vec::new();
    let mut x = 1.0f32;
    while x < 6.0 {
        xs.extend([x, -x]);
        x += 0.01;
    }
    xs.push(-3.0 / 2.0f32.sqrt());
    let c = cpu(&xs).erf().data();
    let g = gpu_leaf(&cpu(&xs)).erf().data();
    assert_eq!(c.len(), g.len());
    let mut worst = (0, 0.0f32);
    for ((&x, &c), &g) in xs.iter().zip(&c).zip(&g) {
        let d = ulps(c, g);
        if d > worst.0 {
            worst = (d, x);
        }
    }
    println!("erf worst ulps {} at x = {}", worst.0, worst.1);
    assert!(
        worst.0 <= 1,
        "erf {} ulps from the CPU at x = {}",
        worst.0,
        worst.1
    );
}

/// GELU at x = -3, the audit's probe point: the GPU result matches the CPU's (which is
/// bit-identical to PyTorch 2.15 f32 there) far closer than the 4.4e-5 relative error the
/// old `erf` gave. One ulp of `erf` at -3/sqrt(2) is 2.2e-5 of `1 + erf`, so the bound
/// allows that much (on lavapipe the results are identical).
#[test]
fn gpu_gelu_matches_cpu_at_minus_three() {
    if skip_or_mark("gpu_gelu_matches_cpu_at_minus_three") {
        return;
    }
    let xs = [-3.0f32, -4.0, -5.0];
    let want = GELU::new().forward(&cpu(&xs)).data();
    let mut got = Vec::new();
    // GELU's scalar constants are `[1]` tensors, and broadcasting is a GPU gap, so only
    // `erf` runs on the device; the binary ops run on the CPU under `PreferGpu`.
    expect_gaps(&["Binary(Add)", "Binary(Div)", "Binary(Mul)"], || {
        got = GELU::new().forward(&gpu_leaf(&cpu(&xs))).data();
    });
    assert_eq!(want.len(), got.len());
    for ((&x, &w), &g) in xs.iter().zip(&want).zip(&got) {
        let rel = ((g - w) / w).abs();
        println!("GELU({x}): cpu {w:e} gpu {g:e} rel {rel:e}");
        assert!(rel <= 2.5e-5, "GELU({x}): cpu {w} gpu {g}, relative {rel}");
    }
}
