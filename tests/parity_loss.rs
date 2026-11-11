//! CPU/GPU parity for the losses that #42 changed: `bce_loss` (the fused `Bce` op, whose
//! forward and backward run as device ops) and `cross_entropy_loss` with class-index
//! targets (a `where` select), forward and input gradient. Both pin their device-policy
//! gaps, so an op that silently drops to the CPU fails the test.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_BWD_GPU, TOL_FWD_GPU, assert_close, expect_gaps, gpu_leaf};
use volta::{DType, RawTensor, Tensor};

/// Forward value and input gradient of `loss(x)`.
fn value_and_grad(x: &Tensor, loss: impl Fn(&Tensor) -> Tensor) -> (f32, Vec<f32>) {
    let out = loss(x);
    let value = out.item();
    out.into_loss().expect("scalar loss").backward();
    (value, x.grad().expect("input gradient"))
}

#[test]
fn bce_loss_parity_including_the_clamp() {
    // Interior soft targets plus p = 0 and p = 1, where the log floor and the
    // gradient's 1e-12 floor apply. All on the GPU: no gaps.
    let p_vals = vec![0.0, 1.0, 0.0, 1.0, 0.3, 0.05, 0.5, 0.95];
    let y_vals = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.2, 0.7, 0.9];
    expect_gaps(&[], || {
        let p_cpu = RawTensor::new(p_vals.clone(), &[2, 4], true);
        let y_cpu = RawTensor::new(y_vals.clone(), &[2, 4], false);
        let (p_gpu, y_gpu) = (gpu_leaf(&p_cpu), gpu_leaf(&y_cpu));
        let (v_cpu, g_cpu) = value_and_grad(&p_cpu, |p| RawTensor::bce_loss(p, &y_cpu));
        let (v_gpu, g_gpu) = value_and_grad(&p_gpu, |p| RawTensor::bce_loss(p, &y_gpu));
        assert!(p_gpu.device().is_gpu(), "pred left the GPU");
        assert_close(
            "bce forward",
            &[v_cpu],
            &[v_gpu],
            TOL_FWD_GPU * v_cpu.abs().max(1.0),
        );
        // The clamped entries are about 2.5e11; compare relative to each magnitude.
        assert_eq!(g_cpu.len(), g_gpu.len());
        for (i, (c, g)) in g_cpu.iter().zip(&g_gpu).enumerate() {
            assert!(
                (c - g).abs() <= TOL_BWD_GPU * c.abs().max(1.0),
                "bce grad[{i}]: cpu {c} vs gpu {g}"
            );
        }
    });
}

#[test]
fn cross_entropy_index_targets_parity_with_a_neg_inf_logit() {
    // Class indices stay on the CPU (the GPU holds no integer dtype); the mask they
    // become is moved to the logits' device. Pinned gaps: `log_softmax`'s per-axis
    // reductions and its broadcast subtract, and `where`'s backward, which runs on the
    // host (its forward has a kernel).
    let logits = vec![1.0, f32::NEG_INFINITY, 2.0, 0.5, 0.2, f32::NEG_INFINITY];
    expect_gaps(&["Binary(Sub)", "MaxDim", "SumDim", "Where"], || {
        let z_cpu = RawTensor::new(logits.clone(), &[2, 3], true);
        let z_gpu = gpu_leaf(&z_cpu);
        let classes = RawTensor::new(vec![0.0, 1.0], &[2], false).to_dtype(DType::I64);
        let (v_cpu, g_cpu) = value_and_grad(&z_cpu, |z| RawTensor::cross_entropy_loss(z, &classes));
        let (v_gpu, g_gpu) = value_and_grad(&z_gpu, |z| RawTensor::cross_entropy_loss(z, &classes));
        assert!(v_cpu.is_finite(), "cpu loss {v_cpu}");
        assert_close("ce forward", &[v_cpu], &[v_gpu], TOL_FWD_GPU);
        assert_close("ce grad", &g_cpu, &g_gpu, TOL_BWD_GPU);
    });
}
