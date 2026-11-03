//! Direct unit tests for the binary ops: forward values and gradient checks.

use super::*;
use crate::index::fetch;

fn make(shape: &[usize], data: Vec<f32>) -> Tensor {
    RawTensor::new(data, shape, true)
}

#[test]
fn add_forward_value_and_shape() {
    let a = make(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make(&[2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let c = a.add(&b);
    assert_eq!(c.borrow().shape, vec![2, 2]);
    assert_eq!(c.borrow().data.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn add_gradcheck() {
    let x = make(&[3], vec![0.5, -1.0, 2.0]);
    let y = RawTensor::new(vec![0.25, 0.75, -0.5], &[3], false);
    let passed = RawTensor::check_gradients_simple(&x, |t| t.add(&y).sum());
    assert!(passed, "add gradient check failed");
}

#[test]
fn sub_forward_value() {
    let a = make(&[3], vec![10.0, 5.0, 0.0]);
    let b = make(&[3], vec![3.0, 4.0, -1.0]);
    let c = a.sub(&b);
    assert_eq!(c.borrow().data.to_vec(), vec![7.0, 1.0, 1.0]);
}

#[test]
fn sub_gradcheck() {
    let x = make(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let y = RawTensor::new(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], false);
    let passed = RawTensor::check_gradients_simple(&x, |t| t.sub(&y).sum());
    assert!(passed, "sub gradient check failed");
}

#[test]
fn elem_mul_forward_value() {
    let a = make(&[3], vec![2.0, 3.0, 4.0]);
    let b = make(&[3], vec![5.0, 6.0, 7.0]);
    let c = a.elem_mul(&b);
    assert_eq!(c.borrow().data.to_vec(), vec![10.0, 18.0, 28.0]);
}

#[test]
fn elem_mul_gradcheck() {
    let x = make(&[3], vec![1.5, -2.0, 0.5]);
    let y = RawTensor::new(vec![0.5, 1.0, -1.5], &[3], false);
    let passed = RawTensor::check_gradients_simple(&x, |t| t.elem_mul(&y).sum());
    assert!(passed, "elem_mul gradient check failed");
}

#[test]
fn div_forward_value() {
    let a = make(&[3], vec![10.0, 9.0, 1.0]);
    let b = make(&[3], vec![2.0, 3.0, 4.0]);
    let c = a.div(&b);
    let out = c.borrow().data.to_vec();
    assert!((fetch(&out, 0) - 5.0).abs() < 1e-6);
    assert!((fetch(&out, 1) - 3.0).abs() < 1e-6);
    assert!((fetch(&out, 2) - 0.25).abs() < 1e-6);
}

#[test]
fn div_gradcheck() {
    let x = make(&[3], vec![2.0, -3.0, 5.0]);
    let y = RawTensor::new(vec![1.5, 2.0, 0.5], &[3], false);
    let passed = RawTensor::check_gradients_simple(&x, |t| t.div(&y).sum());
    assert!(passed, "div gradient check failed");
}

#[test]
fn max_elem_forward_value() {
    let a = make(&[4], vec![1.0, 5.0, 2.0, 8.0]);
    let b = make(&[4], vec![3.0, 4.0, 7.0, 6.0]);
    let c = a.max_elem(&b);
    assert_eq!(c.borrow().data.to_vec(), vec![3.0, 5.0, 7.0, 8.0]);
}

#[test]
fn max_elem_gradient_routes_to_winner() {
    // Use distinct, unambiguous values to avoid tie-handling ambiguity.
    let a = make(&[3], vec![5.0, 1.0, 9.0]);
    let b = make(&[3], vec![2.0, 7.0, 4.0]);
    let z = a.max_elem(&b);
    z.sum().into_loss().expect("scalar loss").backward();
    // a wins at indices 0 and 2.
    assert_eq!(a.grad(), Some(vec![1.0, 0.0, 1.0]));
    assert_eq!(b.grad(), Some(vec![0.0, 1.0, 0.0]));
}

#[test]
fn modulo_forward_value() {
    let a = RawTensor::new(vec![7.0, 8.0, 9.0], &[3], false);
    let b = RawTensor::new(vec![3.0, 3.0, 3.0], &[3], false);
    let c = a.modulo(&b);
    let out = c.borrow().data.to_vec();
    assert!((fetch(&out, 0) - 1.0).abs() < 1e-6);
    assert!((fetch(&out, 1) - 2.0).abs() < 1e-6);
    assert!((fetch(&out, 2) - 0.0).abs() < 1e-6);
}

#[test]
fn lt_forward_value_is_bool() {
    let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
    let b = RawTensor::new(vec![3.0, 1.0, 3.0, 5.0], &[4], false);
    let c = a.lt(&b);
    assert_eq!(c.dtype(), crate::DType::Bool);
    assert!(!c.requires_grad(), "a Bool output has no gradient");
    // 1<3=true, 2<1=false, 3<3=false, 4<5=true
    assert_eq!(
        c.to_vec_as::<bool>().expect("bool"),
        vec![true, false, false, true]
    );
}

#[test]
fn add_then_mul_chain_gradcheck() {
    // (x + y) * z, gradient w.r.t. x should be z.
    let x = make(&[3], vec![1.0, 2.0, 3.0]);
    let y = RawTensor::new(vec![4.0, 5.0, 6.0], &[3], false);
    let z = RawTensor::new(vec![0.5, -0.5, 1.0], &[3], false);
    let passed = RawTensor::check_gradients_simple(&x, |t| t.add(&y).elem_mul(&z).sum());
    assert!(passed, "add+mul chain gradient check failed");
}
