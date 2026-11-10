//! `BatchNorm1d` and `BatchNorm2d`: running statistics, normalization, eval mode, and gradients.

use super::*;
use crate::index::fetch;
use approx::assert_relative_eq;

fn running_stats(state: &io::StateDict) -> (Vec<f32>, Vec<f32>) {
    let rm = state
        .get("running_mean")
        .expect("state dict missing running_mean")
        .data
        .clone();
    let rv = state
        .get("running_var")
        .expect("state dict missing running_var")
        .data
        .clone();
    (rm, rv)
}

// ---- BatchNorm1d ----

#[test]
fn bn1d_forward_shape_preserved() {
    let layer = BatchNorm1d::new(4);
    let x = RawTensor::new(vec![1.0; 8 * 4], &[8, 4], true);
    let y = <BatchNorm1d as Layer>::forward(&layer, &x);
    assert_eq!(y.borrow().shape, vec![8, 4]);
}

#[test]
fn bn1d_train_normalizes_to_zero_mean_unit_var() {
    let layer = BatchNorm1d::new(2);
    // Per-feature, batch-of-4: column 0 = {1, 2, 3, 4}, column 1 = {-1, 0, 1, 2}
    let x = RawTensor::new(
        vec![1.0, -1.0, 2.0, 0.0, 3.0, 1.0, 4.0, 2.0],
        &[4, 2],
        false,
    );
    let y = <BatchNorm1d as Layer>::forward(&layer, &x);
    let out = y.borrow().data.to_vec();
    // Output mean per column should be ~0 and stddev ~1 (gamma=1, beta=0).
    let col0: Vec<f32> = (0..4).map(|i| fetch(&out, i * 2)).collect();
    let mean0: f32 = col0.iter().sum::<f32>() / 4.0;
    let var0: f32 = col0.iter().map(|v| (v - mean0).powi(2)).sum::<f32>() / 4.0;
    assert_relative_eq!(mean0, 0.0, epsilon = 1e-5);
    assert_relative_eq!(var0, 1.0, max_relative = 1e-3);
}

#[test]
fn bn1d_training_updates_running_stats() {
    let layer = BatchNorm1d::new(3);
    // Capture initial state (running_mean=0, running_var=1).
    let (rm0, rv0) = running_stats(&<BatchNorm1d as Module>::state_dict(&layer));
    assert_eq!(rm0, vec![0.0, 0.0, 0.0]);
    assert_eq!(rv0, vec![1.0, 1.0, 1.0]);

    // Forward with a batch whose stats clearly differ from defaults.
    let x = RawTensor::new(
        vec![
            1.0, 10.0, -5.0, 2.0, 12.0, -3.0, 3.0, 14.0, -1.0, 4.0, 16.0, 1.0,
        ],
        &[4, 3],
        false,
    );
    let _ = <BatchNorm1d as Layer>::forward(&layer, &x);

    let (rm1, rv1) = running_stats(&<BatchNorm1d as Module>::state_dict(&layer));
    // Running mean should have moved toward batch mean for each feature.
    // Batch means: col0 = 2.5, col1 = 13, col2 = -2.0
    // With momentum=0.1: rm = 0.9*0 + 0.1*batch_mean
    assert_relative_eq!(fetch(&rm1, 0), 0.25, epsilon = 1e-4);
    assert_relative_eq!(fetch(&rm1, 1), 1.3, epsilon = 1e-4);
    assert_relative_eq!(fetch(&rm1, 2), -0.2, epsilon = 1e-4);
    // Variance updates should differ from the initial 1.0 for at least one feature.
    let any_var_changed = rv1.iter().any(|v| (v - 1.0).abs() > 1e-3);
    assert!(
        any_var_changed,
        "running_var should have moved after training step"
    );
}

#[test]
fn bn1d_eval_uses_running_stats() {
    let mut layer = BatchNorm1d::new(2);
    // Pre-load known running stats via load_state_dict.
    let mut state = <BatchNorm1d as Module>::state_dict(&layer);
    state.insert(
        "running_mean".to_string(),
        io::TensorData {
            data: vec![5.0, -5.0],
            shape: vec![2],
        },
    );
    state.insert(
        "running_var".to_string(),
        io::TensorData {
            data: vec![4.0, 4.0],
            shape: vec![2],
        },
    );
    <BatchNorm1d as Module>::load_state_dict(&mut layer, &state).expect("state loads");
    layer.train(false);

    // Input == running_mean => normalized output should be ~0 (then * gamma=1 + beta=0).
    let x = RawTensor::new(vec![5.0, -5.0, 5.0, -5.0], &[2, 2], false);
    let y = <BatchNorm1d as Layer>::forward(&layer, &x);
    for v in y.borrow().data.to_vec() {
        assert!(
            v.abs() < 1e-4,
            "eval-mode normalized output should be ~0, got {v}"
        );
    }
}

#[test]
fn bn1d_gradcheck_input() {
    let layer = BatchNorm1d::new(2);
    let x = RawTensor::new(vec![1.0, -1.0, 2.0, 0.0, 3.0, 1.0, 4.0, 2.0], &[4, 2], true);
    // Use stricter epsilon on BN since it's mean-centered (small grads).
    let (_max_e, _mean_e, passed) = RawTensor::check_gradients(
        &x,
        |t| <BatchNorm1d as Layer>::forward(&layer, t).sum(),
        1e-2,
        1e-2,
    );
    assert!(passed, "BatchNorm1d input gradient check failed");
}

#[test]
fn bn1d_assert_trainable_all_params_get_gradient() {
    let layer = BatchNorm1d::new(3);
    let x = RawTensor::new(
        vec![
            1.0, 10.0, -5.0, 2.0, 12.0, -3.0, 3.0, 14.0, -1.0, 4.0, 16.0, 1.0,
        ],
        &[4, 3],
        true,
    );
    crate::assert_trainable!(layer, x);
}

// ---- BatchNorm2d ----

#[test]
fn bn2d_forward_shape_preserved() {
    let layer = BatchNorm2d::new(3);
    // (B=2, C=3, H=2, W=2)
    let x = RawTensor::new(vec![1.0; 2 * 3 * 2 * 2], &[2, 3, 2, 2], true);
    let y = <BatchNorm2d as Layer>::forward(&layer, &x);
    assert_eq!(y.borrow().shape, vec![2, 3, 2, 2]);
}

#[test]
fn bn2d_training_updates_running_stats() {
    let layer = BatchNorm2d::new(2);
    let (rm0, rv0) = running_stats(&<BatchNorm2d as Module>::state_dict(&layer));
    assert_eq!(rm0, vec![0.0, 0.0]);
    assert_eq!(rv0, vec![1.0, 1.0]);

    // Channel 0 = all 1.0, Channel 1 = all 5.0  =>  batch_mean = [1, 5]
    let mut data = Vec::with_capacity(2 * 2 * 2 * 2);
    for _b in 0..2 {
        for c in 0..2 {
            for _ in 0..(2 * 2) {
                data.push(if c == 0 { 1.0 } else { 5.0 });
            }
        }
    }
    let x = RawTensor::new(data, &[2, 2, 2, 2], false);
    let _ = <BatchNorm2d as Layer>::forward(&layer, &x);

    let (rm1, _rv1) = running_stats(&<BatchNorm2d as Module>::state_dict(&layer));
    // momentum=0.1: rm = 0.9*0 + 0.1*[1, 5] = [0.1, 0.5]
    assert_relative_eq!(fetch(&rm1, 0), 0.1, epsilon = 1e-5);
    assert_relative_eq!(fetch(&rm1, 1), 0.5, epsilon = 1e-5);
}

#[test]
fn bn2d_gradcheck_input() {
    let layer = BatchNorm2d::new(2);
    // Distinct values so BN actually computes a non-trivial normalization.
    let mut data = Vec::with_capacity(2 * 2 * 2 * 2);
    for i in 0..(2 * 2 * 2 * 2) {
        data.push((i as f32) * 0.1 + 1.0);
    }
    let x = RawTensor::new(data, &[2, 2, 2, 2], true);
    let (_max_e, _mean_e, passed) = RawTensor::check_gradients(
        &x,
        |t| <BatchNorm2d as Layer>::forward(&layer, t).sum(),
        1e-2,
        1e-2,
    );
    assert!(passed, "BatchNorm2d input gradient check failed");
}

#[test]
fn bn2d_assert_trainable_all_params_get_gradient() {
    let layer = BatchNorm2d::new(2);
    let x = crate::macros::trainable_input(&[2, 2, 2, 2]);
    crate::assert_trainable!(layer, x);
}

// ---- one value per channel (S12) ----

#[test]
#[should_panic(expected = "more than 1 value per channel")]
fn bn1d_one_sample_in_training_panics() {
    let layer = BatchNorm1d::new(2);
    let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], false);
    let _ = <BatchNorm1d as Layer>::forward(&layer, &x);
}

#[test]
#[should_panic(expected = "more than 1 value per channel")]
fn bn2d_one_value_per_channel_in_training_panics() {
    let layer = BatchNorm2d::new(3);
    let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3, 1, 1], false);
    let _ = <BatchNorm2d as Layer>::forward(&layer, &x);
}

#[test]
fn rejected_batch_leaves_running_stats_untouched() {
    let layer = BatchNorm1d::new(2);
    let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], false);
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        <BatchNorm1d as Layer>::forward(&layer, &x)
    }));
    assert!(rejected.is_err());
    let (rm, rv) = running_stats(&<BatchNorm1d as Module>::state_dict(&layer));
    assert_eq!(rm, vec![0.0, 0.0]);
    assert_eq!(rv, vec![1.0, 1.0]);
}

#[test]
fn bn2d_one_sample_with_spatial_extent_trains() {
    // 4 values per channel: torch accepts this.
    let layer = BatchNorm2d::new(2);
    let x = RawTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 6.0, 9.0],
        &[1, 2, 2, 2],
        false,
    );
    let _ = <BatchNorm2d as Layer>::forward(&layer, &x);
    let (_, rv) = running_stats(&<BatchNorm2d as Module>::state_dict(&layer));
    assert!(rv.iter().all(|v| v.is_finite()), "running_var: {rv:?}");
}

#[test]
fn bn1d_one_sample_in_eval_is_finite() {
    let mut layer = BatchNorm1d::new(2);
    layer.train(false);
    let x = RawTensor::new(vec![0.5, 1.0], &[1, 2], false);
    let y = <BatchNorm1d as Layer>::forward(&layer, &x).data();
    // Default running stats (mean 0, var 1), gamma 1, beta 0: y = x / sqrt(1 + eps).
    let scale = 1.0 / (1.0f32 + 1e-5).sqrt();
    assert_relative_eq!(fetch(&y, 0), 0.5 * scale, epsilon = 1e-6);
    assert_relative_eq!(fetch(&y, 1), 1.0 * scale, epsilon = 1e-6);
}
