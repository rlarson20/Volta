//! Adam's bias correction is computed once per step on the host in f64 (#304), so the
//! update on each backend is within a few f32 roundings of an f64 reference.
//!
//! The GPU kernel used to evaluate `1 / (1 - pow(beta, t))` per element in f32. At `t = 1`
//! `1 - beta2^t` is `1e-3`, so `pow`'s error became 2.4e-4 relative on lavapipe: the same
//! factor on every element, which an absolute tolerance on the parameters does not see.
//!
//! The parameter starts at zero and gets its gradient through `set_grad`, so its value
//! after each step is the total update, with no rounding against a large starting value.
//! The GPU test returns early without an adapter; run it with `VOLTA_REQUIRE_GPU=1` where
//! one exists (`just test-gpu`).

use volta::nn::Adam;
use volta::storage::Storage;
use volta::{RawTensor, Tensor};

const LR: f32 = 1e-3;
const BETAS: (f32, f32) = (0.9, 0.999);
const EPS: f32 = 1e-8;
/// The gradient fed at every step (the audit's `2 * p` for `p = [1, -2, 0.5, 3]`).
const GRAD: [f32; 4] = [2.0, -4.0, 1.0, 6.0];
/// Steps checked: the first, where the correction is largest, and two later ones.
const CHECKED: [usize; 3] = [1, 2, 10];
/// CPU relative tolerance on the total update: eight f32 roundings.
const CPU_REL_TOL: f64 = 8.0 * f32::EPSILON as f64;
/// GPU relative tolerance: 32 f32 roundings. lavapipe stays under the CPU bound, but CI's
/// Apple GPU (`macos (stable)`) measured 1.9e-6 (16 eps) at step 10 while passing steps 1
/// and 2. The scales are host constants now, so that residual is in the kernel's moment
/// updates, division or `sqrt`; Metal's fast-math is the likely cause (#312), not verified.
/// The f32 `pow` error this test pins was 2.4e-4 at step 1 and 4.8e-5 at step 10 (12x
/// this bound).
#[cfg(feature = "gpu")]
const GPU_REL_TOL: f64 = 32.0 * f32::EPSILON as f64;

/// The parameter after each of `CHECKED.last()` steps, in f64, with the f32 hyperparameters
/// widened (so the test checks the step, not how `0.999` rounds to f32; see #321).
fn reference() -> Vec<Vec<f64>> {
    let (b1, b2) = (f64::from(BETAS.0), f64::from(BETAS.1));
    let (lr, eps) = (f64::from(LR), f64::from(EPS));
    let mut p = [0.0f64; 4];
    let (mut m, mut v) = ([0.0f64; 4], [0.0f64; 4]);
    let last = *CHECKED.last().expect("steps to check");
    (1..=last)
        .map(|t| {
            let t_i = i32::try_from(t).expect("small step count");
            let (s1, s2) = (1.0 / (1.0 - b1.powi(t_i)), 1.0 / (1.0 - b2.powi(t_i)));
            for (((p, m), v), g) in p.iter_mut().zip(&mut m).zip(&mut v).zip(GRAD) {
                let g = f64::from(g);
                *m = b1 * *m + (1.0 - b1) * g;
                *v = b2 * *v + (1.0 - b2) * g * g;
                *p -= lr * (*m * s1) / ((*v * s2).sqrt() + eps);
            }
            p.to_vec()
        })
        .collect()
}

/// Steps a zero parameter on `param`'s device and checks it against [`reference`] after
/// each step in `CHECKED`, at relative tolerance `rel_tol`.
fn check_against_reference(param: &Tensor, backend: &str, rel_tol: f64) {
    let expected = reference();
    let mut opt = Adam::new(vec![param.clone()], LR, BETAS, EPS, 0.0);
    for (step, want) in (1..=expected.len()).zip(&expected) {
        param.set_grad(Storage::cpu(GRAD.to_vec()));
        opt.step();
        if !CHECKED.contains(&step) {
            continue;
        }
        let got = param.data();
        assert_eq!(got.len(), want.len(), "{backend} step {step}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let rel = ((f64::from(*g) - w) / w).abs();
            assert!(
                rel <= rel_tol,
                "{backend} step {step} [{i}]: got {g}, f64 reference {w}, relative error {rel:e} > {rel_tol:e}"
            );
        }
    }
}

/// A trainable CPU parameter of zeros, one per gradient entry.
fn zero_param() -> Tensor {
    let p = RawTensor::new(vec![0.0; GRAD.len()], &[GRAD.len()], false);
    p.set_requires_grad(true);
    p
}

/// The CPU step, which shares the host scales with the GPU one.
#[test]
fn cpu_adam_update_matches_an_f64_reference() {
    check_against_reference(&zero_param(), "cpu", CPU_REL_TOL);
}

/// The GPU kernel step, which used an f32 `pow` per element before #304.
#[cfg(feature = "gpu")]
#[test]
fn gpu_adam_update_matches_an_f64_reference() {
    use volta::Device;
    use volta::gpu::is_gpu_available;
    if !is_gpu_available() {
        return;
    }
    let p = RawTensor::new(vec![0.0; GRAD.len()], &[GRAD.len()], false)
        .to_device(Device::gpu().expect("adapter checked above"));
    p.set_requires_grad(true);
    check_against_reference(&p, "gpu", GPU_REL_TOL);
}
