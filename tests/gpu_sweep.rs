//! GPU probes for defects in `docs/audits/2026-11-sweep.md`: full-tensor `max`, `modulo`,
//! accuracy of `erf`, the `sum` of a large tensor, and optimizer state after `to_device`.
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
#[ignore = "S5: GPU modulo is floored (-1 mod 3 = 2), CPU modulo is truncated (-1)"]
fn modulo_agrees_between_cpu_and_gpu() {
    if skip_or_mark("modulo_agrees_between_cpu_and_gpu") {
        return;
    }
    let (a, b) = (vec![-1.0, 5.5, -7.0, 7.0], vec![3.0, 2.0, 2.0, -3.0]);
    let cpu =
        RawTensor::new(a.clone(), &[4], false).modulo(&RawTensor::new(b.clone(), &[4], false));
    let gpu = on_gpu(a, &[4], false).modulo(&on_gpu(b, &[4], false));
    assert_eq!(cpu.data(), gpu.data());
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
#[ignore = "S24: GPU tanh(x) is NaN for x >= 45 (44 is fine; measured on Apple M2), consistent with exp(2x) overflowing f32; torch gives 1"]
fn gpu_tanh_saturates_without_nan() {
    if skip_or_mark("gpu_tanh_saturates_without_nan") {
        return;
    }
    let xs = vec![-100.0, -50.0, 50.0, 100.0];
    assert_eq!(
        on_gpu(xs, &[4], false).tanh().data(),
        vec![-1.0, -1.0, 1.0, 1.0]
    );
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
