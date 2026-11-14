//! Fused binary cross entropy (`GradFn::Bce`): its gradient against finite differences,
//! and the panics it shares with PyTorch's `binary_cross_entropy`. Cross entropy and NLL:
//! the index-target gradient, and the target shapes they reject (#94).

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

#[test]
#[should_panic(
    expected = "cross_entropy_loss: float targets must match the logits' shape [3, 3], got [3] (class indices must be an I64/I32 tensor)"
)]
fn cross_entropy_rejects_float_class_indices_when_n_equals_c() {
    // #94: float [N] indices broadcast against [N, C] when N == C and gave 10.04 here,
    // where the same indices as I64 give 0.0134.
    let logits = RawTensor::new(
        vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
        &[3, 3],
        true,
    );
    let float_idx = RawTensor::new(vec![0.0, 1.0, 2.0], &[3], false);
    let _ = RawTensor::cross_entropy_loss(&logits, &float_idx);
}

#[test]
#[should_panic(
    expected = "nll_loss: float targets must match the logits' shape [1, 4], got [1] (class indices must be an I64/I32 tensor)"
)]
fn nll_rejects_float_class_index_for_a_batch_of_one() {
    let log_probs = RawTensor::log_softmax(&RawTensor::randn(&[1, 4]), Axis(1));
    let float_idx = RawTensor::new(vec![2.0], &[1], false);
    let _ = RawTensor::nll_loss(&log_probs, &float_idx);
}

#[test]
#[should_panic(expected = "cross_entropy_loss: float targets must match the logits' shape")]
fn cross_entropy_rejects_broadcast_probability_targets() {
    // A [1, C] probability row would broadcast over the batch; PyTorch rejects it too.
    let logits = RawTensor::randn(&[2, 3]);
    let probs = RawTensor::new(vec![0.2, 0.3, 0.5], &[1, 3], false);
    let _ = RawTensor::cross_entropy_loss(&logits, &probs);
}

#[test]
fn cross_entropy_same_shape_probabilities_match_i64_indices() {
    // One-hot float targets of the logits' shape still work, and agree with the same
    // classes as I64 and I32 indices.
    manual_seed(0x94);
    let logits = RawTensor::randn(&[3, 3]);
    let one_hot = RawTensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        false,
    );
    let i64_idx = RawTensor::from_typed(vec![0i64, 1, 2], &[3]);
    let i32_idx = RawTensor::from_typed(vec![0i32, 1, 2], &[3]);
    let from_probs = RawTensor::cross_entropy_loss(&logits, &one_hot).item();
    let from_i64 = RawTensor::cross_entropy_loss(&logits, &i64_idx).item();
    let from_i32 = RawTensor::cross_entropy_loss(&logits, &i32_idx).item();
    assert!(
        (from_probs - from_i64).abs() < 1e-6,
        "one-hot {from_probs} vs I64 {from_i64}"
    );
    assert!(
        (from_i32 - from_i64).abs() < 1e-6,
        "I32 {from_i32} vs I64 {from_i64}"
    );
    // Non-degenerate: -mean(log_softmax picked) on random logits is positive and finite.
    assert!(from_i64.is_finite() && from_i64 > 0.0, "loss {from_i64}");
}

#[test]
fn cross_entropy_soft_targets_gradient_matches_finite_differences() {
    // In f64: in f32 a 1e-3 central difference of a loss near 1 is mostly rounding.
    manual_seed(0x95);
    let logits = RawTensor::randn(&[3, 4]).to_dtype(DType::F64);
    logits.set_requires_grad(true);
    let soft = RawTensor::softmax(&RawTensor::randn(&[3, 4]), Axis(1))
        .detach()
        .to_dtype(DType::F64);
    let (max_err, mean_err, passed) = RawTensor::check_gradients(
        &logits,
        |t| RawTensor::cross_entropy_loss(t, &soft),
        1e-3,
        1e-4,
    );
    assert!(
        passed,
        "soft-target cross_entropy_loss gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
    );
}

/// #95: `[N, 1]` against `[N]` used to broadcast to an `[N, N]` matrix and return its mean.
#[test]
#[should_panic(expected = "mse_loss: pred and target must have the same shape")]
fn mse_loss_rejects_mismatched_shapes() {
    let p = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4, 1], true);
    let y = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], false);
    let _ = RawTensor::mse_loss(&p, &y);
}

#[test]
#[should_panic(expected = "bce_with_logits_loss: logits and target must have the same shape")]
fn bce_with_logits_loss_rejects_mismatched_shapes() {
    let p = RawTensor::new(vec![0.5; 4], &[4, 1], true);
    let y = RawTensor::new(vec![1.0, 0.0, 1.0, 0.0], &[4], false);
    let _ = RawTensor::bce_with_logits_loss(&p, &y);
}

#[test]
#[should_panic(expected = "kl_divergence_gaussian: mu and logvar must have the same shape")]
fn kl_divergence_gaussian_rejects_mismatched_shapes() {
    let mu = RawTensor::new(vec![0.0; 4], &[2, 2], true);
    let logvar = RawTensor::new(vec![0.0; 2], &[2], true);
    let _ = RawTensor::kl_divergence_gaussian(&mu, &logvar);
}

/// Equal shapes still work: identical tensors give an MSE of exactly 0.
#[test]
fn mse_loss_of_identical_tensors_is_zero() {
    let p = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4, 1], true);
    let y = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4, 1], false);
    assert_eq!(RawTensor::mse_loss(&p, &y).item(), 0.0);
}
