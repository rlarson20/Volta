use volta::nn::{Adam, SGD};
use volta::{RawTensor, Storage, TensorOps};

/// Helper: read the first (scalar) value from a tensor.
fn scalar(t: &volta::Tensor) -> f32 {
    t.borrow().data.first().copied().unwrap_or(f32::NAN)
}

// ---------------------------------------------------------------------------
// Adam tests
// ---------------------------------------------------------------------------

#[test]
fn test_adam_bias_correction() {
    // Single parameter = 1.0, constant gradient = 0.5
    // After one step, verify the bias-corrected update matches hand computation.
    let lr: f32 = 0.01;
    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let eps: f32 = 1e-8;
    let grad_val: f32 = 0.5;
    let init_param: f32 = 1.0;

    let t = RawTensor::new(vec![init_param], &[1], true);
    t.borrow_mut().grad = Some(Storage::cpu(vec![grad_val]));

    let mut opt = Adam::new(vec![t.clone()], lr, (beta1, beta2), eps, 0.0);
    opt.step();

    // Hand-compute expected value after step 1 (t=1):
    // m = beta1*0 + (1-beta1)*grad = 0.1 * 0.5 = 0.05
    // v = beta2*0 + (1-beta2)*grad^2 = 0.001 * 0.25 = 0.00025
    // m_hat = m / (1 - beta1^1) = 0.05 / 0.1 = 0.5
    // v_hat = v / (1 - beta2^1) = 0.00025 / 0.001 = 0.25
    // delta = lr * m_hat / (sqrt(v_hat) + eps) = 0.01 * 0.5 / (0.5 + 1e-8)
    let m = (1.0 - beta1) * grad_val;
    let v = (1.0 - beta2) * grad_val.powi(2);
    let m_hat = m / (1.0 - beta1);
    let v_hat = v / (1.0 - beta2);
    let delta = lr * m_hat / (v_hat.sqrt() + eps);
    let expected = init_param - delta;

    let actual = scalar(&t);
    assert!(
        (actual - expected).abs() < 1e-7,
        "Adam bias correction: expected {expected}, got {actual}"
    );
}

#[test]
fn test_adam_mv_accumulation_multi_step() {
    // 3 steps with the same gradient = 0.5, verify the full trajectory.
    let lr: f32 = 0.01;
    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let eps: f32 = 1e-8;
    let grad_val: f32 = 0.5;
    let init_param: f32 = 2.0;

    let t = RawTensor::new(vec![init_param], &[1], true);
    t.borrow_mut().grad = Some(Storage::cpu(vec![grad_val]));

    let mut opt = Adam::new(vec![t.clone()], lr, (beta1, beta2), eps, 0.0);

    // Simulate 3 steps, tracking expected m, v, and parameter.
    let mut m: f32 = 0.0;
    let mut v: f32 = 0.0;
    let mut expected_param = init_param;

    for step in 1..=3_i32 {
        // Re-inject gradient before each step (step does not zero grad,
        // but we want a known, fresh gradient each time).
        t.borrow_mut().grad = Some(Storage::cpu(vec![grad_val]));
        opt.step();

        // EMA updates
        m = beta1 * m + (1.0 - beta1) * grad_val;
        v = beta2 * v + (1.0 - beta2) * grad_val.powi(2);
        let m_hat = m / (1.0 - beta1.powi(step));
        let v_hat = v / (1.0 - beta2.powi(step));
        expected_param -= lr * m_hat / (v_hat.sqrt() + eps);

        let actual = scalar(&t);
        assert!(
            (actual - expected_param).abs() < 1e-6,
            "Adam step {step}: expected {expected_param}, got {actual}"
        );
    }
}

#[test]
fn test_adam_weight_decay() {
    // With weight_decay=0.01, zero gradient, param=2.0:
    // The effective gradient becomes wd * param = 0.01 * 2.0 = 0.02,
    // so the parameter should move. Compare against wd=0.0 (no movement).
    let lr: f32 = 0.01;
    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let eps: f32 = 1e-8;
    let init_param: f32 = 2.0;

    // With weight decay
    let t_wd = RawTensor::new(vec![init_param], &[1], true);
    t_wd.borrow_mut().grad = Some(Storage::cpu(vec![0.0]));
    let mut opt_wd = Adam::new(vec![t_wd.clone()], lr, (beta1, beta2), eps, 0.01);
    opt_wd.step();
    let val_wd = scalar(&t_wd);

    // Without weight decay
    let t_no = RawTensor::new(vec![init_param], &[1], true);
    t_no.borrow_mut().grad = Some(Storage::cpu(vec![0.0]));
    let mut opt_no = Adam::new(vec![t_no.clone()], lr, (beta1, beta2), eps, 0.0);
    opt_no.step();
    let val_no = scalar(&t_no);

    // Without weight decay the param should stay at init (grad=0 => no update)
    assert!(
        (val_no - init_param).abs() < 1e-8,
        "Adam no-wd: expected {init_param}, got {val_no}"
    );

    // With weight decay the param should have moved toward zero
    assert!(
        val_wd < init_param,
        "Adam with wd should decrease param: got {val_wd}"
    );

    // Verify the exact value via hand computation:
    // effective_grad = 0.0 + 0.01 * 2.0 = 0.02
    let eff_grad: f32 = 0.01 * init_param;
    let m = (1.0 - beta1) * eff_grad;
    let v = (1.0 - beta2) * eff_grad.powi(2);
    let m_hat = m / (1.0 - beta1);
    let v_hat = v / (1.0 - beta2);
    let expected = init_param - lr * m_hat / (v_hat.sqrt() + eps);
    assert!(
        (val_wd - expected).abs() < 1e-7,
        "Adam wd exact: expected {expected}, got {val_wd}"
    );
}

#[test]
fn test_adam_convergence_convex() {
    // Minimize f(x) = (x - 3)^2 via Adam, 200 steps.
    let x = RawTensor::new(vec![0.0], &[1], true);
    let mut opt = Adam::new(vec![x.clone()], 0.05, (0.9, 0.999), 1e-8, 0.0);

    for _ in 0..500 {
        opt.zero_grad();
        let target = RawTensor::new(vec![3.0], &[1], false);
        let diff = x.sub(&target);
        let loss = diff.elem_mul(&diff).sum();
        loss.backward();
        opt.step();
    }

    let final_val = scalar(&x);
    assert!(
        (final_val - 3.0).abs() < 0.05,
        "Adam convex: expected ~3.0, got {final_val}"
    );
}

// ---------------------------------------------------------------------------
// SGD tests
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_no_momentum_simple() {
    // momentum=0.0, verify exact theta -= lr * grad
    let lr: f32 = 0.1;
    let init_param: f32 = 5.0;
    let grad_val: f32 = 2.0;

    let t = RawTensor::new(vec![init_param], &[1], true);
    t.borrow_mut().grad = Some(Storage::cpu(vec![grad_val]));

    let mut opt = SGD::new(vec![t.clone()], lr, 0.0, 0.0);
    opt.step();

    let expected = init_param - lr * grad_val; // 5.0 - 0.2 = 4.8
    let actual = scalar(&t);
    assert!(
        (actual - expected).abs() < 1e-7,
        "SGD simple: expected {expected}, got {actual}"
    );

    // Second step with same persistent gradient
    opt.step();
    let expected2 = expected - lr * grad_val; // 4.8 - 0.2 = 4.6
    let actual2 = scalar(&t);
    assert!(
        (actual2 - expected2).abs() < 1e-7,
        "SGD simple step2: expected {expected2}, got {actual2}"
    );
}

#[test]
fn test_sgd_weight_decay() {
    // weight_decay=0.1, no momentum.
    // effective_grad = grad + wd * param
    // new_param = param - lr * effective_grad
    let lr: f32 = 0.1;
    let wd: f32 = 0.1;
    let init_param: f32 = 4.0;
    let grad_val: f32 = 1.0;

    let t = RawTensor::new(vec![init_param], &[1], true);
    t.borrow_mut().grad = Some(Storage::cpu(vec![grad_val]));

    let mut opt = SGD::new(vec![t.clone()], lr, 0.0, wd);
    opt.step();

    // effective_grad = 1.0 + 0.1 * 4.0 = 1.4
    // new_param = 4.0 - 0.1 * 1.4 = 4.0 - 0.14 = 3.86
    let eff_grad = grad_val + wd * init_param;
    let expected = init_param - lr * eff_grad;
    let actual = scalar(&t);
    assert!(
        (actual - expected).abs() < 1e-7,
        "SGD weight decay: expected {expected}, got {actual}"
    );
}

#[test]
fn test_sgd_convergence_convex() {
    // Minimize f(x) = (x - 3)^2 via SGD with momentum=0.9, 500 steps, lr=0.05
    let x = RawTensor::new(vec![0.0], &[1], true);
    let mut opt = SGD::new(vec![x.clone()], 0.05, 0.9, 0.0);

    for _ in 0..500 {
        opt.zero_grad();
        let target = RawTensor::new(vec![3.0], &[1], false);
        let diff = x.sub(&target);
        let loss = diff.elem_mul(&diff).sum();
        loss.backward();
        opt.step();
    }

    let final_val = scalar(&x);
    assert!(
        (final_val - 3.0).abs() < 0.05,
        "SGD convex: expected ~3.0, got {final_val}"
    );
}
