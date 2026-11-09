//! Repros for the defects found by the post-Phase-5 audit (`docs/audits/2026-11-post-phase5.md`).
//!
//! Each test asserts the correct behaviour (PyTorch's, where it has one) and carries
//! `#[ignore = "A<n>: ..."]` (or the sweep ID it confirms) until the fix lands; un-ignore it
//! with the fix. GPU tests print `GPU-RAN <name>` before asserting, because without an
//! adapter they return early and report `ok`. Run them with
//! `cargo test --test audit_repros -- --ignored --nocapture`.

#[cfg(feature = "gpu")]
use volta::Device;
use volta::{DType, RawTensor, SGD, Storage};

/// The GPU, or `None` (and a `GPU-SKIPPED` line) without one; prints `GPU-RAN` otherwise.
#[cfg(feature = "gpu")]
fn gpu_or_skip(name: &str) -> Option<Device> {
    let gpu = Device::gpu().filter(|_| volta::is_gpu_available());
    if gpu.is_some() {
        println!("GPU-RAN {name}");
    } else {
        println!("GPU-SKIPPED {name}");
    }
    gpu
}

/// A1 (fixed): `ConvTranspose2d` with a bias built its pre-bias output on the CPU and
/// added the GPU bias to it, so a GPU model panicked with "expected all inputs on one
/// device" under every device policy. Under `PreferGpu` a layer with no GPU kernel must
/// still run (on the CPU, moved back) and match the CPU.
#[cfg(feature = "gpu")]
#[test]
fn conv_transpose2d_with_bias_runs_on_the_gpu_under_prefer_gpu() {
    use volta::{ConvTranspose2d, DevicePolicy, Layer, Module, with_device_policy};
    let Some(gpu) = gpu_or_skip("conv_transpose2d_with_bias_runs_on_the_gpu_under_prefer_gpu")
    else {
        return;
    };
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    volta::manual_seed(1);
    let cpu = ConvTranspose2d::new(2, 3, 3, 2, 1, true);
    volta::manual_seed(1);
    let mut on_gpu = ConvTranspose2d::new(2, 3, 3, 2, 1, true);
    on_gpu.to_device(gpu.clone());
    let x = RawTensor::randn(&[1, 2, 3, 3]);
    let want = cpu.forward(&x).data();
    let out = on_gpu.forward(&x.to_device(gpu));
    assert!(out.device().is_gpu());
    let got = out.data();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "gpu {g} cpu {w}");
    }
}

/// S13 (confirmed by this audit, fixed): `Sequential` overrides `to_dtype` to reach each child
/// but not `to_device`, so the default moves only parameters and `PositionalEncoding`'s
/// sinusoid table (a buffer) stays on the CPU: `forward` on a GPU input panics.
#[cfg(feature = "gpu")]
#[test]
fn sequential_to_device_moves_child_buffers() {
    use volta::nn::{PositionalEncoding, PositionalEncodingType};
    use volta::{Layer, Module, Sequential};
    let Some(gpu) = gpu_or_skip("sequential_to_device_moves_child_buffers") else {
        return;
    };
    let mut s = Sequential::new(vec![Box::new(PositionalEncoding::new(
        4,
        8,
        PositionalEncodingType::Sinusoidal,
    ))]);
    s.to_device(gpu.clone());
    let out = s.forward(&RawTensor::randn(&[1, 3, 4]).to_device(gpu));
    assert!(out.device().is_gpu());
}

/// A2 (fixed): elementwise GPU kernels dispatched `len / 256` workgroups with no check against
/// wgpu's 65535 limit, so a tensor of 16,776,961 elements or more (a 4096 x 4096 matrix
/// is 16,777,216) aborts with an opaque `wgpu error: Validation Error`; the cast, compare,
/// where and unscale kernels refuse the same sizes with a message that names Muon. The
/// kernels should cover any size (a 2-D dispatch or a strided loop).
#[cfg(feature = "gpu")]
#[test]
fn gpu_elementwise_and_cast_cover_a_4096_square() {
    let Some(gpu) = gpu_or_skip("gpu_elementwise_and_cast_cover_a_4096_square") else {
        return;
    };
    let n = 4096 * 4096;
    // Distinct values near the end, so a wrong flat index shows up as a wrong value.
    let data: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 * 0.25).collect();
    let a = RawTensor::new(data.clone(), &[4096, 4096], false).to_device(gpu);
    let last = n - 1;
    let want = |i: usize| (i % 1000) as f32 * 0.25;

    let sum = a.add(&a).data();
    assert_eq!(sum[last], 2.0 * want(last));
    assert_eq!(sum[n - 300_000], 2.0 * want(n - 300_000));
    let exp = a.relu().data();
    assert_eq!(exp[last], want(last));
    let half = a.to_dtype(DType::F16).data();
    assert_eq!(half[last], want(last));
    let lt = a.lt(&a.add(&a)).to_vec_as::<bool>().expect("bool");
    assert_eq!(lt[last], want(last) > 0.0);
    let picked = a.lt(&a).where_op(&a, &a.neg()).data();
    assert_eq!(picked[last], -want(last));
    let permuted = a.permute(&[volta::Axis(1), volta::Axis(0)]).data();
    assert_eq!(permuted[last], want(last));

    // Backward through a GPU unary op over the same size.
    let x = a.detach();
    x.set_requires_grad(true);
    x.elem_mul(&x).sum().into_loss().expect("scalar").backward();
    let g = x.grad().expect("grad");
    assert_eq!(g[last], 2.0 * want(last));
}

/// A3 (fixed): a negative momentum was silently ignored on the CPU (the momentum branch tests
/// `> 0`) and panics on an `unwrap()` on the GPU (no velocity was allocated). PyTorch
/// refuses it at construction: `ValueError: Invalid momentum value: -0.5`.
#[test]
#[should_panic(expected = "momentum")]
fn sgd_refuses_a_negative_momentum() {
    let p = RawTensor::new(vec![1.0], &[1], true);
    let _ = SGD::new(vec![p], 0.1, -0.5, 0.0);
}

/// A4 (fixed): `GradScaler` unscaled f16 gradients; PyTorch refuses (`ValueError: Attempting to
/// unscale FP16 gradients.`), because the unscaled gradient of an f16 parameter can
/// underflow or overflow in f16 itself. Loss scaling is for autocast's f32 parameters.
#[test]
#[should_panic(expected = "FP16")]
fn grad_scaler_refuses_f16_gradients() {
    use volta::GradScaler;
    let p = RawTensor::new(vec![1.0], &[1], false)
        .to_dtype(DType::F16)
        .detach();
    p.set_requires_grad(true);
    p.set_grad(Storage::cpu(vec![1.0]).cast(DType::F16));
    let mut opt = SGD::new(vec![p], 0.1, 0.0, 0.0);
    let _ = GradScaler::new().step(&mut opt);
}

/// A3's rule for every optimizer: a negative (or NaN) hyperparameter, or an Adam beta
/// outside `[0, 1)`, is refused by `try_new` with PyTorch's message, before any state is
/// allocated; valid ones build.
#[test]
fn optimizers_validate_hyperparameters_like_torch() {
    use volta::{Adam, Muon, VoltaError};
    let p = || vec![RawTensor::new(vec![1.0], &[1], true)];
    let msg = |r: volta::Result<()>| match r {
        Err(VoltaError::InvalidParameter(m)) => m,
        other => panic!("expected InvalidParameter, got {other:?}"),
    };
    assert_eq!(
        msg(SGD::try_new(p(), -0.1, 0.0, 0.0).map(drop)),
        "Invalid learning rate value: -0.1"
    );
    assert_eq!(
        msg(SGD::try_new(p(), 0.1, -0.5, 0.0).map(drop)),
        "Invalid momentum value: -0.5"
    );
    assert_eq!(
        msg(SGD::try_new(p(), 0.1, f32::NAN, 0.0).map(drop)),
        "Invalid momentum value: NaN"
    );
    assert_eq!(
        msg(SGD::try_new(p(), 0.1, 0.0, -1.0).map(drop)),
        "Invalid weight_decay value: -1"
    );
    assert_eq!(
        msg(Adam::try_new(p(), 0.1, (1.0, 0.999), 1e-8, 0.0).map(drop)),
        "Invalid beta parameter at index 0: 1"
    );
    assert_eq!(
        msg(Adam::try_new(p(), 0.1, (0.9, -0.1), 1e-8, 0.0).map(drop)),
        "Invalid beta parameter at index 1: -0.1"
    );
    assert_eq!(
        msg(Adam::try_new(p(), 0.1, (0.9, 0.999), -1e-8, 0.0).map(drop)),
        "Invalid epsilon value: -0.00000001"
    );
    assert_eq!(
        msg(Muon::try_new(p(), 0.02, -0.95, true, 5).map(drop)),
        "Invalid momentum value: -0.95"
    );
    assert!(SGD::try_new(p(), 0.1, 0.9, 1e-4).is_ok());
    assert!(Adam::try_new(p(), 1e-3, (0.9, 0.999), 1e-8, 0.0).is_ok());
    assert!(Muon::try_new(p(), 0.02, 0.95, true, 5).is_ok());
}

/// A4's boundary: bf16 gradients are accepted, as in PyTorch (only f16 is refused).
#[test]
fn grad_scaler_accepts_bf16_gradients() {
    use volta::GradScaler;
    let p = RawTensor::new(vec![1.0], &[1], false)
        .to_dtype(DType::BF16)
        .detach();
    p.set_requires_grad(true);
    p.set_grad(Storage::cpu(vec![65536.0]).cast(DType::BF16));
    let mut opt = SGD::new(vec![p.clone()], 0.5, 0.0, 0.0);
    assert!(GradScaler::new().step(&mut opt));
    assert_eq!(p.data(), vec![0.5]);
}
