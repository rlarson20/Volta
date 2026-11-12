//! GPU full-tensor reductions (`sum`, `mean`, `max_reduce`): the multi-pass kernel of
//! `reduce.wgsl` against the CPU and an f64 reference, at sizes either side of each pass
//! boundary (a workgroup folds 1024 values, so 1024 and 1025 values take one and two
//! passes, and 1024 * 1024 + 1 takes three), and their backward passes with an upstream
//! gradient other than 1 (the gradient is read on the device, never fetched to the host).
//!
//! Each test prints `GPU-RAN <name>` before its assertions, because a GPU test with no
//! adapter returns early and still reports `ok`; check the log for the marker.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{gpu_leaf, skip_if_no_gpu};
use volta::{RawTensor, Tensor};

/// Sizes at and around the pass boundaries of a 256-thread, 4-values-per-thread workgroup.
const SIZES: [usize; 12] = [
    1, 2, 255, 256, 257, 1023, 1024, 1025, 65_536, 65_537, 1_048_576, 1_048_577,
];

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

/// `n` reproducible values in `[-1, 1)` (xorshift; the tests need no external RNG).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // The top 24 bits are exact in f32.
            let unit = (state >> 40) as f32 / 16_777_216.0;
            unit.mul_add(2.0, -1.0)
        })
        .collect()
}

/// A CPU leaf and an independent GPU leaf with the same values, both requiring grad.
fn leaves(data: Vec<f32>) -> (Tensor, Tensor) {
    let n = data.len();
    let cpu = RawTensor::new(data, &[n], true);
    let gpu = gpu_leaf(&cpu);
    (cpu, gpu)
}

/// A one-element constant on the same device as `like`.
fn scalar_like(like: &Tensor, v: f32) -> Tensor {
    RawTensor::new(vec![v], &[1], false).to_device(like.device())
}

#[test]
fn gpu_sum_and_mean_match_f64_at_every_pass_boundary() {
    if skip_or_mark("gpu_sum_and_mean_match_f64_at_every_pass_boundary") {
        return;
    }
    for (k, &n) in SIZES.iter().enumerate() {
        let data = values(n, 0x9E37_79B9 + k as u64);
        let exact: f64 = data.iter().map(|&v| f64::from(v)).sum();
        let magnitude: f64 = data.iter().map(|&v| f64::from(v).abs()).sum();
        // A tree of depth ~log n keeps the error near eps * log n * sum|x|.
        let tol = 1e-5 * magnitude.max(1.0);
        let (_, gpu) = leaves(data);

        let sum = f64::from(gpu.sum().item());
        assert!(
            (sum - exact).abs() <= tol,
            "sum n={n}: gpu {sum} f64 {exact}"
        );

        let mean = f64::from(gpu.mean().item());
        let want = exact / n as f64;
        assert!(
            (mean - want).abs() <= tol / n as f64,
            "mean n={n}: gpu {mean} f64 {want}"
        );
    }
}

#[test]
fn gpu_max_and_its_gradient_match_cpu_at_every_pass_boundary() {
    if skip_or_mark("gpu_max_and_its_gradient_match_cpu_at_every_pass_boundary") {
        return;
    }
    for (k, &n) in SIZES.iter().enumerate() {
        let (cpu, gpu) = leaves(values(n, 0xC0FF_EE00 + k as u64));
        let cpu_max = cpu.max_reduce();
        let gpu_max = gpu.max_reduce();
        assert_eq!(cpu_max.item(), gpu_max.item(), "max n={n}");

        // Weighted by 3 so the gradient that reaches the kernel is not 1.
        let loss = |max: &Tensor| max.elem_mul(&scalar_like(max, 3.0));
        loss(&cpu_max).into_loss().expect("scalar").backward();
        loss(&gpu_max).into_loss().expect("scalar").backward();
        let (want, got) = (cpu.grad().expect("cpu grad"), gpu.grad().expect("gpu grad"));
        assert_eq!(want.len(), got.len(), "max grad length n={n}");
        assert_eq!(want, got, "max grad n={n}");
        assert_eq!(want.iter().filter(|&&g| g != 0.0).count(), 1, "n={n}");
    }
}

#[test]
fn gpu_sum_and_mean_backward_use_the_upstream_gradient() {
    if skip_or_mark("gpu_sum_and_mean_backward_use_the_upstream_gradient") {
        return;
    }
    let n = 3000;
    let (_, sum_in) = leaves(values(n, 1));
    let sum = sum_in.sum();
    sum.elem_mul(&scalar_like(&sum, 3.0))
        .into_loss()
        .expect("scalar")
        .backward();
    let grad = sum_in.grad().expect("grad");
    assert_eq!(grad.len(), n);
    assert!(grad.iter().all(|&g| g == 3.0), "sum backward");

    let (_, mean_in) = leaves(values(n, 2));
    let mean = mean_in.mean();
    mean.elem_mul(&scalar_like(&mean, 3.0))
        .into_loss()
        .expect("scalar")
        .backward();
    let grad = mean_in.grad().expect("grad");
    assert_eq!(grad.len(), n);
    let want = 3.0 / n as f32;
    assert!(
        grad.iter().all(|&g| (g - want).abs() <= 1e-7),
        "mean backward: want {want}"
    );
}

#[test]
fn gpu_and_cpu_agree_on_an_exact_tie() {
    if skip_or_mark("gpu_and_cpu_agree_on_an_exact_tie") {
        return;
    }
    // Which of the tied elements gets the gradient is #40's rule (torch splits it evenly);
    // until then the CPU and the GPU must at least pick the same one, including when the
    // tie straddles two workgroups and two passes.
    let mut long = vec![0.0; 1_100_000];
    long[7] = 9.0;
    long[1_050_000] = 9.0;
    let mut mid = vec![0.0; 3000];
    mid[5] = 4.0;
    mid[2000] = 4.0;
    for data in [vec![2.0, 5.0, 5.0, 1.0], mid, long] {
        let n = data.len();
        let (cpu, gpu) = leaves(data);
        cpu.max_reduce().into_loss().expect("scalar").backward();
        gpu.max_reduce().into_loss().expect("scalar").backward();
        assert_eq!(cpu.grad().expect("cpu"), gpu.grad().expect("gpu"), "n={n}");
    }
}

#[test]
fn gpu_max_of_negative_values_and_a_trailing_partial_group() {
    if skip_or_mark("gpu_max_of_negative_values_and_a_trailing_partial_group") {
        return;
    }
    // Every value negative, and the maximum in the last, partly filled workgroup.
    let n = 2049;
    let mut data = vec![-5.0; n];
    data[n - 1] = -1.5;
    let (_, gpu) = leaves(data);
    let max = gpu.max_reduce();
    assert_eq!(max.item(), -1.5);
    max.into_loss().expect("scalar").backward();
    let grad = gpu.grad().expect("grad");
    assert_eq!(grad.iter().position(|&g| g == 1.0), Some(n - 1));
}

#[test]
fn gpu_mean_of_2_pow_25_ones_is_one() {
    if skip_or_mark("gpu_mean_of_2_pow_25_ones_is_one") {
        return;
    }
    let n = 1usize << 25;
    let (_, gpu) = leaves(vec![1.0; n]);
    assert_eq!(gpu.mean().item(), 1.0);
}
