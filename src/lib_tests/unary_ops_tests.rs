//! Direct unit tests for the unary ops: forward values and gradient checks.

use super::*;
use crate::index::fetch;
use approx::assert_relative_eq;

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
