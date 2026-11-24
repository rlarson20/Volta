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

/// Reductions on several threads at once give the same sums as one thread (#244): every
/// thread shares one device and one buffer pool, so a pass's partials must stay out of the
/// pool until the command buffer that reads them has been submitted.
#[test]
fn gpu_concurrent_sums_match_f64() {
    if skip_or_mark("gpu_concurrent_sums_match_f64") {
        return;
    }
    // Two sizes whose passes leave 8-byte and 4100-byte partials, and a three-pass size
    // that drops both while its own encoder is still open.
    let sizes = [2usize, 1025, 1_048_577];
    std::thread::scope(|scope| {
        for t in 0..6u64 {
            scope.spawn(move || {
                for round in 0..20u64 {
                    for (k, &n) in sizes.iter().enumerate() {
                        let data = values(n, 0xA5A5_0000 + t * 1000 + round * 10 + k as u64);
                        let exact: f64 = data.iter().map(|&v| f64::from(v)).sum();
                        let magnitude: f64 = data.iter().map(|&v| f64::from(v).abs()).sum();
                        let (_, gpu) = leaves(data);
                        let sum = f64::from(gpu.sum().item());
                        assert!(
                            (sum - exact).abs() <= 1e-5 * magnitude.max(1.0),
                            "thread {t} round {round} sum n={n}: gpu {sum} f64 {exact}"
                        );
                    }
                }
            });
        }
    });
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
fn gpu_and_cpu_split_the_gradient_of_an_exact_tie_evenly() {
    if skip_or_mark("gpu_and_cpu_split_the_gradient_of_an_exact_tie_evenly") {
        return;
    }
    // `torch.max()` splits the gradient over every tied element (S4). Include a tie that
    // straddles two workgroups and two passes, and an upstream gradient other than 1.
    let mut long = vec![0.0; 1_100_000];
    long[7] = 9.0;
    long[1_050_000] = 9.0;
    let mut mid = vec![0.0; 3000];
    mid[5] = 4.0;
    mid[2000] = 4.0;
    let ties = [
        (vec![2.0, 5.0, 5.0, 1.0], [1, 2]),
        (mid, [5, 2000]),
        (long, [7, 1_050_000]),
    ];
    for (data, winners) in ties {
        let n = data.len();
        let (cpu, gpu) = leaves(data);
        for t in [&cpu, &gpu] {
            let max = t.max_reduce();
            max.elem_mul(&scalar_like(&max, 3.0))
                .into_loss()
                .expect("scalar")
                .backward();
        }
        let (cpu_grad, gpu_grad) = (cpu.grad().expect("cpu"), gpu.grad().expect("gpu"));
        assert_eq!(cpu_grad, gpu_grad, "n={n}");
        for (i, g) in gpu_grad.iter().enumerate() {
            let want = if winners.contains(&i) { 1.5 } else { 0.0 };
            assert_eq!(*g, want, "n={n}, element {i}");
        }
    }
}

#[test]
fn gpu_max_propagates_nan_and_splits_the_gradient_over_the_nans() {
    if skip_or_mark("gpu_max_propagates_nan_and_splits_the_gradient_over_the_nans") {
        return;
    }
    // A NaN anywhere makes the maximum NaN (torch.max), including a NaN in a later pass,
    // and the gradient goes evenly to every NaN, not to the numbers.
    let mut long = values(1_100_000, 3);
    long[7] = f32::NAN;
    long[1_050_000] = f32::NAN;
    let mut mid = values(3000, 4);
    mid[2000] = f32::NAN;
    for (data, nans) in [
        (vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0], vec![1, 3]),
        (mid, vec![2000]),
        (long, vec![7, 1_050_000]),
    ] {
        let n = data.len();
        let (cpu, gpu) = leaves(data);
        for t in [&cpu, &gpu] {
            let max = t.max_reduce();
            assert!(max.item().is_nan(), "n={n}: the maximum is NaN");
            max.into_loss().expect("scalar").backward();
        }
        let gpu_grad = gpu.grad().expect("gpu");
        assert_eq!(cpu.grad().expect("cpu"), gpu_grad, "n={n}");
        let want = 1.0 / nans.len() as f32;
        for (i, g) in gpu_grad.iter().enumerate() {
            assert_eq!(
                *g,
                if nans.contains(&i) { want } else { 0.0 },
                "n={n}, element {i}"
            );
        }
    }
}

#[test]
fn gpu_max_of_all_neg_inf_splits_over_every_element() {
    if skip_or_mark("gpu_max_of_all_neg_inf_splits_over_every_element") {
        return;
    }
    // Every element is the maximum, as torch: a `-inf` that equals the maximum is a tie.
    for n in [3, 3000] {
        let (_, gpu) = leaves(vec![f32::NEG_INFINITY; n]);
        let max = gpu.max_reduce();
        assert_eq!(max.item(), f32::NEG_INFINITY, "n={n}");
        max.into_loss().expect("scalar").backward();
        let want = 1.0 / n as f32;
        let grad = gpu.grad().expect("grad");
        assert!(
            grad.iter().all(|&g| g == want),
            "n={n}: every element gets {want}"
        );
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
