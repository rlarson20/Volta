//! GPU probes for defects in `docs/audits/2026-11-sweep.md`: full-tensor `max`, `modulo`,
//! accuracy of `erf`, saturation of `tanh` and `sigmoid`, the `sum` of a large tensor, and
//! optimizer state after `to_device`.
//!
//! Each test prints `GPU-RAN <name>` before its assertions, because a GPU test with no
//! adapter returns early and still reports `ok`; check the log for the marker. Defects are
//! `#[ignore = "S<n>: ..."]`; run them with
//! `cargo test --test gpu_sweep -- --ignored --nocapture`.

#![cfg(feature = "gpu")]

use volta::gpu::is_gpu_available;
use volta::{Device, Linear, Module, RawTensor, SGD, Storage, Tensor};

/// Skip (returning `true`) when there is no GPU; otherwise print the marker for `name`.
fn skip_or_mark(name: &str) -> bool {
    if is_gpu_available() {
        println!("GPU-RAN {name}");
        false
    } else {
        println!("GPU-SKIPPED {name}");
        true
    }
}

/// An independent GPU copy of `data`.
fn on_gpu(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
    let t = RawTensor::new(data, shape, false).to_device(Device::gpu().expect("gpu"));
    t.set_requires_grad(requires_grad);
    t
}

#[test]
#[ignore = "S21: the GPU max_reduce kernel starts its running maximum at -1e9, so the max of values below that is wrong"]
fn gpu_max_of_values_below_minus_1e9() {
    if skip_or_mark("gpu_max_of_values_below_minus_1e9") {
        return;
    }
    let x = on_gpu(vec![-2.0e9, -3.0e9], &[2], false);
    assert_eq!(x.max_reduce().item(), -2.0e9);
}

#[test]
#[ignore = "S21: the GPU max index is found by |v - max| < 1e-6, so a near-tie sends the gradient to the first of two values 2.4e-7 apart"]
fn gpu_max_gradient_goes_to_the_true_maximum_on_a_near_tie() {
    if skip_or_mark("gpu_max_gradient_goes_to_the_true_maximum_on_a_near_tie") {
        return;
    }
    // 3.0000002 is the next f32 above 3.0, so the maximum is index 1.
    let hi = f32::from_bits(3.0f32.to_bits() + 1);
    let x = on_gpu(vec![3.0, hi, 1.0], &[3], true);
    x.max_reduce().into_loss().expect("scalar").backward();
    assert_eq!(x.grad().expect("grad"), vec![0.0, 1.0, 0.0]);
}

#[test]
fn modulo_agrees_between_cpu_and_gpu() {
    if skip_or_mark("modulo_agrees_between_cpu_and_gpu") {
        return;
    }
    // Mixed signs: the floored result takes the divisor's sign, as torch.remainder.
    let (a, b) = (vec![-1.0, 5.5, -7.0, 7.0], vec![3.0, 2.0, 2.0, -3.0]);
    let (cpu_a, cpu_b) = (
        RawTensor::new(a.clone(), &[4], true),
        RawTensor::new(b.clone(), &[4], true),
    );
    let (gpu_a, gpu_b) = (on_gpu(a, &[4], true), on_gpu(b, &[4], true));
    let cpu = cpu_a.modulo(&cpu_b);
    let gpu = gpu_a.modulo(&gpu_b);
    assert_eq!(cpu.data(), vec![2.0, 1.5, 1.0, -2.0]);
    assert_eq!(cpu.data(), gpu.data());

    cpu.sum().into_loss().expect("scalar").backward();
    gpu.sum().into_loss().expect("scalar").backward();
    // d/da = 1 and d/db = -floor(a/b).
    assert_eq!(cpu_a.grad().expect("grad"), vec![1.0; 4]);
    assert_eq!(cpu_b.grad().expect("grad"), vec![1.0, -2.0, 4.0, 3.0]);
    assert_eq!(cpu_a.grad(), gpu_a.grad());
    assert_eq!(cpu_b.grad(), gpu_b.grad());
}

#[test]
#[ignore = "S22: the GPU erf is Abramowitz-Stegun 7.1.26 (abs error 1.5e-7), so its relative error near 0 is about 1e-4"]
fn gpu_erf_is_relatively_accurate_near_zero() {
    if skip_or_mark("gpu_erf_is_relatively_accurate_near_zero") {
        return;
    }
    let xs = vec![1.0e-3, 1.0e-2, 0.1, 1.0];
    let cpu = RawTensor::new(xs.clone(), &[4], false).erf().data();
    let gpu = on_gpu(xs, &[4], false).erf().data();
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(((c - g) / c).abs() < 1e-5, "erf[{i}]: cpu {c} gpu {g}");
    }
}

#[test]
fn gpu_tanh_saturates_without_nan() {
    if skip_or_mark("gpu_tanh_saturates_without_nan") {
        return;
    }
    // 43.8 to 44.3 is the band where Metal's tanh returned 0, 44.4 and up where it was NaN.
    let xs = vec![
        -100.0, -50.0, -44.0, 43.8, 44.0, 44.3, 44.4, 45.0, 50.0, 100.0,
    ];
    let want: Vec<f32> = xs.iter().map(|x: &f32| x.signum()).collect();
    let n = xs.len();
    assert_eq!(on_gpu(xs, &[n], false).tanh().data(), want);
}

#[test]
fn gpu_tanh_agrees_with_cpu_in_the_range_that_does_not_saturate() {
    if skip_or_mark("gpu_tanh_agrees_with_cpu_in_the_range_that_does_not_saturate") {
        return;
    }
    let xs = vec![-9.5, -3.0, -0.01, 1.0e-4, 0.7, 9.5];
    let n = xs.len();
    let cpu = RawTensor::new(xs.clone(), &[n], false).tanh().data();
    let gpu = on_gpu(xs, &[n], false).tanh().data();
    assert_eq!(cpu.len(), gpu.len());
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!((c - g).abs() <= 1e-6, "tanh[{i}]: cpu {c} gpu {g}");
    }
}

#[test]
fn gpu_tanh_backward_saturates_without_nan() {
    if skip_or_mark("gpu_tanh_backward_saturates_without_nan") {
        return;
    }
    let xs = vec![-100.0, -50.0, -1.5, 0.3, 43.8, 44.0, 50.0, 100.0];
    let n = xs.len();
    let cpu = RawTensor::new(xs.clone(), &[n], true);
    cpu.tanh().sum().into_loss().expect("scalar").backward();
    let want = cpu.grad().expect("cpu grad");
    let gpu = on_gpu(xs, &[n], true);
    gpu.tanh().sum().into_loss().expect("scalar").backward();
    let got = gpu.grad().expect("gpu grad");
    assert_eq!(want.len(), got.len());
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!(
            g.is_finite() && (w - g).abs() <= 1e-6,
            "tanh grad[{i}]: cpu {w} gpu {g}"
        );
    }
}

#[test]
fn gpu_tanh_propagates_nan() {
    if skip_or_mark("gpu_tanh_propagates_nan") {
        return;
    }
    // Clamping the input must not turn a NaN into a number.
    let x = on_gpu(vec![f32::NAN, 1.0], &[2], true);
    let y = x.tanh();
    let out = y.data();
    assert!(out[0].is_nan(), "tanh(NaN) on the GPU is {}", out[0]);
    y.sum().into_loss().expect("scalar").backward();
    let grad = x.grad().expect("grad");
    assert!(grad[0].is_nan(), "tanh'(NaN) on the GPU is {}", grad[0]);
}

#[test]
fn gpu_sigmoid_saturates_without_nan() {
    if skip_or_mark("gpu_sigmoid_saturates_without_nan") {
        return;
    }
    let xs = vec![-100.0, -50.0, 50.0, 100.0];
    let cpu = RawTensor::new(xs.clone(), &[4], false).sigmoid().data();
    let gpu = on_gpu(xs, &[4], false).sigmoid().data();
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            g.is_finite() && (c - g).abs() <= 1e-6,
            "sigmoid[{i}]: cpu {c} gpu {g}"
        );
    }
}

#[test]
#[ignore = "S6: the single-thread GPU sum_reduce accumulates sequentially in f32, so the sum of 2^25 ones stalls at 2^24"]
fn gpu_sum_of_2_pow_25_ones_is_exact() {
    if skip_or_mark("gpu_sum_of_2_pow_25_ones_is_exact") {
        return;
    }
    let n = 1usize << 25;
    let x = on_gpu(vec![1.0; n], &[n], false);
    assert_eq!(x.sum().item(), 33_554_432.0);
}

#[test]
#[ignore = "S17: SGD allocates its velocity on the parameter's device in new(), so moving the model to the GPU afterwards makes step() panic"]
fn optimizer_built_before_to_device_still_steps() {
    if skip_or_mark("optimizer_built_before_to_device_still_steps") {
        return;
    }
    let mut layer = Linear::new(2, 2, false);
    let params = layer.parameters();
    let mut opt = SGD::new(params.clone(), 0.1, 0.9, 0.0);
    layer.to_device(Device::gpu().expect("gpu"));
    for p in &params {
        p.set_grad(Storage::cpu(vec![1.0; 4]));
    }
    opt.step();
}
