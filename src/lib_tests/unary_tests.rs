//! Unary ops: `neg`, `sqrt`, `exp2`/`log2`, and `erf`, forward and backward.

use super::*;
use approx::assert_relative_eq;

#[test]
fn test_neg_forward_backward() {
    let x = RawTensor::new(vec![2.0, -3.0], &[2], true);
    let y = x.neg();

    // Forward
    assert_eq!(y.borrow().data, vec![-2.0, 3.0]);

    // Backward: ∂(-x)/∂x = -1
    y.sum().into_loss().expect("scalar loss").backward();
    assert_eq!(x.grad(), Some(vec![-1.0, -1.0]));
}

#[test]
fn test_sqrt_chain() {
    let x = RawTensor::new(vec![4.0], &[1], true);
    let y = x.sqrt(); // y = 2.0
    let z = y.elem_mul(&y); // z = 4.0
    z.into_loss().expect("scalar loss").backward();

    // ∂z/∂x = ∂z/∂y * ∂y/∂x = 2y * 1/(2√x) = 2*2 * 1/4 = 1.0
    assert_relative_eq!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("tensor data should exist"),
        1.0,
        epsilon = 1e-6
    );
}

#[test]
fn test_exp2_log2_inverse() {
    let x = RawTensor::new(vec![2.0], &[1], true);
    let y = x.exp2().log2(); // should recover x
    y.clone().into_loss().expect("scalar loss").backward();

    assert_relative_eq!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        2.0,
        epsilon = 1e-6
    );
    // Chain rule: ∂(log2(2^x))/∂x = 1
    assert_relative_eq!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("tensor data should exist"),
        1.0,
        epsilon = 1e-6
    );
}

#[test]
fn test_erf_forward() {
    // Test erf(0) = 0
    let x = RawTensor::new(vec![0.0], &[1], false);
    let y = x.erf();
    assert_relative_eq!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        0.0,
        epsilon = 1e-6
    );

    // Test erf(1) ≈ 0.8427
    let x = RawTensor::new(vec![1.0], &[1], false);
    let y = x.erf();
    assert_relative_eq!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        0.8427,
        epsilon = 1e-4
    );

    // Test erf(-1) ≈ -0.8427
    let x = RawTensor::new(vec![-1.0], &[1], false);
    let y = x.erf();
    assert_relative_eq!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        -0.8427,
        epsilon = 1e-4
    );
}

#[test]
fn test_erf_gradient() {
    // Test gradient: d(erf(x))/dx = 2/sqrt(pi) * exp(-x^2)
    let x = RawTensor::new(vec![1.0], &[1], true);
    let y = x.erf();
    y.into_loss().expect("scalar loss").backward();

    // Expected gradient: 2/sqrt(pi) * exp(-1) ≈ 0.4151
    let expected_grad = 2.0 / std::f32::consts::PI.sqrt() * (-1.0_f32).exp();
    assert_relative_eq!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("tensor data should exist"),
        expected_grad,
        epsilon = 1e-6
    );
}
