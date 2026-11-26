//! Unary ops: forward values and backward passes, `neg`, `sqrt`, `exp2`/`log2` and `erf` in detail, and a gradient check per op.

use super::*;
use crate::index::fetch;
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

fn shaped(data: Vec<f32>) -> Tensor {
    let n = data.len();
    RawTensor::new(data, &[n], true)
}

// Forward correctness ------------------------------------------------

#[test]
fn neg_forward() {
    let x = shaped(vec![1.0, -2.0, 0.0]);
    assert_eq!(x.neg().borrow().data.to_vec(), vec![-1.0, 2.0, 0.0]);
}

#[test]
fn recip_forward() {
    let x = shaped(vec![1.0, 2.0, 4.0]);
    let out = x.recip().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 0.5, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 2), 0.25, epsilon = 1e-6);
}

#[test]
fn sqrt_forward() {
    let x = shaped(vec![1.0, 4.0, 9.0]);
    let out = x.sqrt().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 2.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 2), 3.0, epsilon = 1e-6);
}

#[test]
fn exp_forward() {
    let x = shaped(vec![0.0, 1.0]);
    let out = x.exp().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), std::f32::consts::E, epsilon = 1e-5);
}

#[test]
fn log_forward() {
    let x = shaped(vec![1.0, std::f32::consts::E]);
    let out = x.log().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 0.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 1.0, epsilon = 1e-5);
}

#[test]
fn exp2_forward() {
    let x = shaped(vec![0.0, 1.0, 3.0]);
    let out = x.exp2().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 2.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 2), 8.0, epsilon = 1e-5);
}

#[test]
fn log2_forward() {
    let x = shaped(vec![1.0, 2.0, 8.0]);
    let out = x.log2().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 0.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 2), 3.0, epsilon = 1e-5);
}

#[test]
fn sin_forward() {
    let x = shaped(vec![0.0, std::f32::consts::FRAC_PI_2]);
    let out = x.sin().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 0.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 1.0, epsilon = 1e-6);
}

#[test]
fn cos_forward() {
    let x = shaped(vec![0.0, std::f32::consts::PI]);
    let out = x.cos().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), -1.0, epsilon = 1e-5);
}

#[test]
fn tanh_forward() {
    let x = shaped(vec![0.0, 100.0, -100.0]);
    let out = x.tanh().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 0.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 2), -1.0, epsilon = 1e-6);
}

#[test]
fn sigmoid_forward() {
    let x = shaped(vec![0.0, 100.0, -100.0]);
    let out = x.sigmoid().borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 0.5, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 1.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 2), 0.0, epsilon = 1e-6);
}

#[test]
fn relu_forward() {
    let x = shaped(vec![1.0, -2.0, 0.0, 3.5]);
    assert_eq!(x.relu().borrow().data.to_vec(), vec![1.0, 0.0, 0.0, 3.5]);
}

// Backward correctness via gradient check ----------------------------

#[test]
fn neg_gradcheck() {
    let x = shaped(vec![1.0, -1.0, 0.5]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.neg().sum()));
}

#[test]
fn recip_gradcheck() {
    let x = shaped(vec![0.5, 1.0, 2.0, 4.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.recip().sum()));
}

#[test]
fn sqrt_gradcheck() {
    let x = shaped(vec![0.5, 1.0, 4.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.sqrt().sum()));
}

#[test]
fn exp_gradcheck() {
    let x = shaped(vec![-1.0, 0.0, 1.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.exp().sum()));
}

#[test]
fn log_gradcheck() {
    let x = shaped(vec![0.5, 1.0, 2.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.log().sum()));
}

#[test]
fn exp2_gradcheck() {
    let x = shaped(vec![-1.0, 0.0, 1.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.exp2().sum()));
}

#[test]
fn log2_gradcheck() {
    let x = shaped(vec![0.5, 1.0, 2.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.log2().sum()));
}

#[test]
fn sin_gradcheck() {
    let x = shaped(vec![-1.0, 0.0, 1.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.sin().sum()));
}

#[test]
fn cos_gradcheck() {
    let x = shaped(vec![-1.0, 0.0, 1.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.cos().sum()));
}

#[test]
fn tanh_gradcheck() {
    let x = shaped(vec![-1.0, 0.0, 1.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.tanh().sum()));
}

#[test]
fn sigmoid_gradcheck() {
    let x = shaped(vec![-1.0, 0.0, 1.0]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.sigmoid().sum()));
}

#[test]
fn relu_gradcheck() {
    // Avoid 0 to keep the gradient well-defined under finite differences.
    let x = shaped(vec![-1.5, -0.5, 0.5, 1.5]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.relu().sum()));
}

#[test]
fn erf_gradcheck() {
    let x = shaped(vec![-0.5, 0.0, 0.5]);
    assert!(RawTensor::check_gradients_simple(&x, |t| t.erf().sum()));
}
