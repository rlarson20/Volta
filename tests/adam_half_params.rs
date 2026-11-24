//! Adam refuses half (f16, bf16) parameters (#317): its moment estimates live in the
//! parameter's dtype, where the f16 second moment underflows to 0 and the step is finite
//! but wrong (`lr * sqrt(t)` on the CPU, `lr * m / eps` on the GPU). The supported path,
//! f32 parameters, takes the issue's probe to 0.999, 0.998, 0.997 on both backends.

use volta::error::VoltaError;
use volta::nn::{Adam, Linear, Module};
use volta::{DType, Device, RawTensor, Tensor};

/// The issue's probe parameter: `w = 1`, a leaf of `dtype` on `device`.
fn probe_param(dtype: DType, device: Device) -> Tensor {
    let w = RawTensor::new(vec![1.0], &[1], false)
        .to_dtype(dtype)
        .to_device(device)
        .detach();
    w.set_requires_grad(true);
    w
}

/// Asserts `try_new` refuses `w` with the #317 message naming `dtype`.
fn assert_refused(w: Tensor, dtype: DType) {
    match Adam::try_new(vec![w], 1e-3, (0.9, 0.999), 1e-8, 0.0) {
        Err(VoltaError::InvalidParameter(m)) => {
            assert!(
                m.message.contains(&format!("parameter 0 is {dtype}"))
                    && m.message.contains("#317")
                    && m.message.contains("amp::autocast"),
                "{m}"
            );
        }
        Err(e) => panic!("{dtype}: wrong error {e:?}"),
        Ok(_) => panic!("{dtype}: Adam accepted a half parameter"),
    }
}

/// Three Adam steps of `loss = sum(w * 1e-3)` from `w = 1`; the values of `w` after each.
fn probe_steps(w: &Tensor) -> Vec<f32> {
    let mut opt = Adam::new(vec![w.clone()], 1e-3, (0.9, 0.999), 1e-8, 0.0);
    (0..3)
        .map(|_| {
            opt.zero_grad();
            w.elem_mul(&RawTensor::new(vec![1e-3], &[1], false).to_device(w.device()))
                .sum()
                .into_loss()
                .expect("scalar")
                .backward();
            opt.step();
            w.item()
        })
        .collect()
}

/// Asserts the probe's trajectory is 0.999, 0.998, 0.997 (each step is `lr`, since the
/// bias-corrected `m / sqrt(v)` is 1 for a constant gradient).
fn assert_probe(label: &str, got: &[f32]) {
    let want = [0.999f32, 0.998, 0.997];
    assert_eq!(got.len(), want.len());
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-6, "{label} step {k}: {g} vs {w}");
    }
}

#[test]
fn try_new_refuses_half_parameters_on_the_cpu() {
    for dtype in [DType::F16, DType::BF16] {
        assert_refused(probe_param(dtype, Device::CPU), dtype);
    }
}

/// A list with one half parameter among f32 ones is refused, naming its index.
#[test]
fn try_new_names_the_half_parameter() {
    let params = vec![
        probe_param(DType::F32, Device::CPU),
        probe_param(DType::F16, Device::CPU),
    ];
    match Adam::try_new(params, 1e-3, (0.9, 0.999), 1e-8, 0.0) {
        Err(VoltaError::InvalidParameter(m)) => {
            assert!(m.message.contains("parameter 1 is f16"), "{m}");
        }
        Err(e) => panic!("wrong error {e:?}"),
        Ok(_) => panic!("Adam accepted a half parameter"),
    }
}

/// A model cast to f16 after its optimizer was built is refused at `step`, before any
/// parameter is updated.
#[test]
#[should_panic(expected = "Adam: parameter 0 is f16")]
fn step_refuses_a_parameter_cast_to_half_later() {
    let mut layer = Linear::new(2, 1, true);
    let mut opt = Adam::new(layer.parameters(), 1e-3, (0.9, 0.999), 1e-8, 0.0);
    layer.to_dtype(DType::F16);
    opt.step();
}

#[test]
fn f32_probe_steps_by_lr_on_the_cpu() {
    assert_probe("cpu", &probe_steps(&probe_param(DType::F32, Device::CPU)));
}

/// GPU half parameters are refused, and the f32 probe matches the CPU's trajectory.
/// Skipped without an adapter; on CI's GPU lanes `gpu_smoke_test` fails a missing one.
#[cfg(feature = "gpu")]
#[test]
fn gpu_refuses_half_parameters_and_steps_the_f32_probe_by_lr() {
    if !volta::gpu::is_gpu_available() {
        return;
    }
    let gpu = Device::gpu().expect("an adapter");
    for dtype in [DType::F16, DType::BF16] {
        assert_refused(probe_param(dtype, gpu.clone()), dtype);
    }
    let w = probe_param(DType::F32, gpu);
    assert!(w.device().is_gpu());
    assert_probe("gpu", &probe_steps(&w));
}
