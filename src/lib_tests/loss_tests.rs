//! Fused binary cross entropy (`GradFn::Bce`): its gradient against finite differences,
//! and the panics it shares with PyTorch's `binary_cross_entropy`.

use super::*;
use crate::autograd::no_grad;

/// Interior probabilities in `[0.05, 0.95]`, where PyTorch's BCE gradient is the
/// derivative of the loss, so finite differences can check it.
fn interior_probs(shape: &[usize]) -> Vec<f32> {
    RawTensor::rand(shape)
        .data()
        .into_iter()
        .map(|u| 0.05 + 0.9 * u)
        .collect()
}

#[test]
fn bce_gradient_matches_finite_differences_with_soft_targets() {
    manual_seed(0xB7);
    let shape = [3, 4];
    let p = RawTensor::new(interior_probs(&shape), &shape, true);
    let y = RawTensor::rand(&shape);
    let (max_err, mean_err, passed) =
        RawTensor::check_gradients(&p, |t| RawTensor::bce_loss(t, &y), 1e-3, 1e-2);
    assert!(
        passed,
        "bce_loss gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
    );
}

#[test]
fn bce_gradient_matches_finite_differences_in_f64() {
    manual_seed(0xB8);
    let shape = [2, 5];
    let p = RawTensor::new(interior_probs(&shape), &shape, false).to_dtype(DType::F64);
    p.set_requires_grad(true);
    let y = RawTensor::rand(&shape).to_dtype(DType::F64);
    let (max_err, mean_err, passed) =
        RawTensor::check_gradients(&p, |t| RawTensor::bce_loss(t, &y), 1e-6, 1e-6);
    assert!(
        passed,
        "f64 bce_loss gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
    );
}

#[test]
fn bce_gradient_is_torchs_formula_at_the_clamp() {
    // p = 0 with y = 1: the clamped forward is flat, but PyTorch's gradient is
    // (p - y) / max(p (1 - p), 1e-12) / N = -1e12 / 2.
    let p = RawTensor::new(vec![0.0, 0.5], &[2], true);
    let y = RawTensor::new(vec![1.0, 0.5], &[2], false);
    let loss = RawTensor::bce_loss(&p, &y);
    assert!((loss.item() - (100.0 + std::f32::consts::LN_2) / 2.0).abs() < 1e-4);
    loss.into_loss().expect("scalar loss").backward();
    let g = p.grad().expect("pred has a gradient");
    assert_eq!(g.len(), 2);
    assert!((g[0] / -5e11 - 1.0).abs() < 1e-5, "grad[0] = {}", g[0]);
    assert!(g[1].abs() < 1e-7, "grad[1] = {}", g[1]);
}

#[test]
#[should_panic(expected = "derivative for target is not implemented")]
fn bce_rejects_a_target_that_requires_grad() {
    let p = RawTensor::new(vec![0.2, 0.7], &[2], true);
    let y = RawTensor::new(vec![0.0, 1.0], &[2], true);
    let _ = RawTensor::bce_loss(&p, &y);
}

#[test]
fn bce_accepts_a_grad_target_under_no_grad() {
    let p = RawTensor::new(vec![0.2, 0.7], &[2], true);
    let y = RawTensor::new(vec![0.0, 1.0], &[2], true);
    let _guard = no_grad();
    let expected = -(0.8f32.ln() + 0.7f32.ln()) / 2.0;
    assert!((RawTensor::bce_loss(&p, &y).item() - expected).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "every pred must be in [0, 1], got 1.5 at flat index 1")]
fn bce_rejects_a_prediction_above_one() {
    let p = RawTensor::new(vec![0.2, 1.5], &[2], false);
    let y = RawTensor::new(vec![0.0, 1.0], &[2], false);
    let _ = RawTensor::bce_loss(&p, &y);
}

#[test]
#[should_panic(expected = "every pred must be in [0, 1], got NaN at flat index 0")]
fn bce_rejects_a_nan_prediction() {
    let p = RawTensor::new(vec![f32::NAN, 0.5], &[2], false);
    let y = RawTensor::new(vec![0.0, 1.0], &[2], false);
    let _ = RawTensor::bce_loss(&p, &y);
}

#[test]
#[should_panic(expected = "target shape must match pred")]
fn bce_rejects_a_broadcast_target() {
    let p = RawTensor::new(vec![0.2, 0.7], &[2, 1], false);
    let y = RawTensor::new(vec![0.0, 1.0], &[1, 2], false);
    let _ = RawTensor::bce_loss(&p, &y);
}

#[test]
fn cross_entropy_index_targets_gradient_matches_finite_differences() {
    // The class-index path selects with `where`, not a one-hot product.
    manual_seed(0xC8);
    let logits = RawTensor::randn(&[3, 4]);
    logits.set_requires_grad(true);
    let classes = RawTensor::new(vec![2.0, 0.0, 3.0], &[3], false).to_dtype(DType::I64);
    let (max_err, mean_err, passed) = RawTensor::check_gradients(
        &logits,
        |t| RawTensor::cross_entropy_loss(t, &classes),
        1e-3,
        1e-2,
    );
    assert!(
        passed,
        "cross_entropy_loss gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
    );
}
