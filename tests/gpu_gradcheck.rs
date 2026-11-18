//! `check_gradients` on GPU tensors (#124): each perturbed input lives on the GPU, so the
//! finite differences and the analytic gradient both come from the GPU kernels. One check
//! per backward kernel family that has a GPU kernel: elementwise unary and binary, the
//! full `sum` every loss here ends in, matmul and movement. The default
//! `DevicePolicy::Strict` makes a CPU fallback a panic, so a pass means the GPU kernels
//! ran. `mean_dim` (`SumDim`) and `softmax` (`MaxDim`) are left out: they have no GPU
//! kernel and panic under `Strict`. Only built with the `gpu` feature; skips without a GPU.
#![cfg(feature = "gpu")]

use volta::gpu::is_gpu_available;
use volta::{Axis, Device, RawTensor, Tensor};

const EPS: f32 = 1e-2;
const TOL: f32 = 1e-2;

/// A leaf on the GPU with `data` and `shape`, requiring grad.
fn gpu_leaf(data: Vec<f32>, shape: &[usize]) -> Tensor {
    let dev = Device::gpu().expect("GPU availability checked by the caller");
    let t = RawTensor::new(data, shape, false).to_device(dev);
    t.set_requires_grad(true);
    t
}

/// Deterministic non-degenerate values: distinct, mixed signs, away from zero.
fn values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = 0.37 * i as f32 - 0.9;
            if v.abs() < 0.1 { v + 0.25 } else { v }
        })
        .collect()
}

/// `check_gradients` on `x`, asserting every input `loss_fn` sees is on the GPU.
fn assert_gradcheck(name: &str, x: &Tensor, loss_fn: impl Fn(&Tensor) -> Tensor) {
    let (max_err, mean_err, passed) = RawTensor::check_gradients(
        x,
        |t| {
            assert!(
                t.device().is_gpu(),
                "{name}: perturbed input on {}",
                t.device()
            );
            let loss = loss_fn(t);
            assert!(loss.device().is_gpu(), "{name}: loss on {}", loss.device());
            loss
        },
        EPS,
        TOL,
    );
    assert!(
        passed,
        "{name}: GPU gradcheck failed, max {max_err:.3e}, mean {mean_err:.3e}"
    );
    assert!(x.device().is_gpu(), "{name}: input left the GPU");
}

#[test]
fn gradcheck_runs_on_gpu_unary() {
    if !is_gpu_available() {
        return;
    }
    let x = gpu_leaf(values(6), &[2, 3]);
    assert_gradcheck("tanh", &x, |t| t.tanh().sum());
    assert_gradcheck("sigmoid", &x, |t| t.sigmoid().sum());
    assert_gradcheck("exp", &x, |t| t.exp().sum());
    assert_gradcheck("sin", &x, |t| t.sin().sum());
}

#[test]
fn gradcheck_runs_on_gpu_binary() {
    if !is_gpu_available() {
        return;
    }
    // The issue's repro: x * x on a GPU tensor panicked with "cast to f64".
    let x = gpu_leaf(vec![0.5, 1.5, -0.7], &[3]);
    assert_gradcheck("mul self", &x, |t| t.elem_mul(t).sum());

    let x = gpu_leaf(values(6), &[2, 3]);
    let y = gpu_leaf(vec![1.2, -0.8, 2.0, 0.6, -1.5, 0.9], &[2, 3]);
    assert_gradcheck("mul", &x, |t| t.elem_mul(&y).sum());
    assert_gradcheck("div numerator", &x, |t| t.div(&y).sum());
    assert_gradcheck("div denominator", &y, |t| x.div(t).sum());
    assert_gradcheck("sub", &x, |t| y.sub(t).elem_mul(t).sum());
}

#[test]
fn gradcheck_runs_on_gpu_matmul() {
    if !is_gpu_available() {
        return;
    }
    let a = gpu_leaf(values(6), &[2, 3]);
    let b = gpu_leaf(vec![0.4, -1.3, 0.8, 1.1, -0.6, 0.2], &[3, 2]);
    assert_gradcheck("matmul lhs", &a, |t| t.matmul(&b).tanh().sum());
    assert_gradcheck("matmul rhs", &b, |t| a.matmul(t).tanh().sum());
}

#[test]
fn gradcheck_runs_on_gpu_movement() {
    if !is_gpu_available() {
        return;
    }
    let x = gpu_leaf(values(6), &[2, 3]);
    let w = gpu_leaf(vec![0.5, -1.0, 1.5, 2.0, -0.5, 1.0], &[3, 2]);
    assert_gradcheck("permute", &x, |t| {
        t.permute(&[Axis(1), Axis(0)]).elem_mul(&w).sum()
    });
    assert_gradcheck("reshape", &x, |t| t.reshape(&[3, 2]).elem_mul(&w).sum());
}
