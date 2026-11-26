//! Binary ops: forward values, the backward passes of `add`, `elem_mul`, `div` and `max`, gradient checks, the chain rule, and a device-name check.

use super::*;
use crate::index::fetch;

#[test]
fn test_div_backward() {
    let x = RawTensor::new(vec![6.0], &[1], true);
    let y = RawTensor::new(vec![2.0], &[1], true);
    let z = x.div(&y); // z = 3.0
    z.into_loss().expect("scalar loss").backward();

    // ∂(x/y)/∂x = 1/y = 0.5
    assert_eq!(x.grad(), Some(vec![0.5]));
    // ∂(x/y)/∂y = -x/y² = -6/4 = -1.5
    assert_eq!(y.grad(), Some(vec![-1.5]));
}

#[test]
fn test_max_backward() {
    let x = RawTensor::new(vec![3.0, 1.0], &[2], true);
    let y = RawTensor::new(vec![2.0, 4.0], &[2], true);
    let z = x.max_elem(&y);
    let loss = z.sum();
    loss.into_loss().expect("scalar loss").backward();

    // max picks [3.0, 4.0], so grads flow to x[0] and y[1]
    assert_eq!(x.grad(), Some(vec![1.0, 0.0]));
    assert_eq!(y.grad(), Some(vec![0.0, 1.0]));
}

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
fn modulo_is_floored_for_negative_operands() {
    let a = RawTensor::new(vec![-1.0, 7.0], &[2], false);
    let b = RawTensor::new(vec![3.0, -3.0], &[2], false);
    let out = a.modulo(&b).data();
    assert_eq!(out, vec![2.0, -2.0]);
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

#[test]
fn test_basic_add_backward() {
    let a = RawTensor::new(vec![2.0], &[1], true);
    let b = RawTensor::new(vec![3.0], &[1], true);
    let c = a.add(&b);
    c.into_loss().expect("scalar loss").backward();

    assert_eq!(a.grad(), Some(vec![1.0]));
    assert_eq!(b.grad(), Some(vec![1.0]));
}

#[test]
fn test_enhanced_device_safety() {
    let a = RawTensor::new(vec![2.0], &[1], true);
    let b = RawTensor::new(vec![3.0], &[1], true);
    let c = a.add(&b);
    c.into_loss().expect("scalar loss").backward();

    // Test device handling safety
    let cpu_device = Device::CPU;
    assert!(cpu_device.is_cpu());
    assert!(!cpu_device.is_gpu());
    assert_eq!(cpu_device.name(), "CPU");

    let gpu_device = Device::GPU("CUDA".to_string());
    assert!(!gpu_device.is_cpu());
    assert!(gpu_device.is_gpu());
    assert_eq!(gpu_device.name(), "CUDA");

    // Test that tensor operations still work
    assert_eq!(a.grad(), Some(vec![1.0]));
    assert_eq!(b.grad(), Some(vec![1.0]));
}

#[test]
fn test_multiply_backward() {
    let a = RawTensor::new(vec![3.0], &[1], true);
    let b = RawTensor::new(vec![4.0], &[1], true);
    let c = a.elem_mul(&b);
    c.into_loss().expect("scalar loss").backward();

    assert_eq!(a.grad(), Some(vec![4.0]));
    assert_eq!(b.grad(), Some(vec![3.0]));
}

#[test]
fn test_chain_rule() {
    let a = RawTensor::new(vec![2.0], &[1], true);
    let b = RawTensor::new(vec![3.0], &[1], true);
    let c = a.add(&b);
    let d = c.elem_mul(&a);
    d.into_loss().expect("scalar loss").backward();

    assert_eq!(a.grad(), Some(vec![7.0]));
    assert_eq!(b.grad(), Some(vec![2.0]));
}

#[test]
fn test_multidim_ops() {
    let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let b = RawTensor::new(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], true);
    let c = a.elem_mul(&b);
    let loss = c.sum();
    loss.into_loss().expect("scalar loss").backward();

    assert_eq!(a.grad(), Some(vec![0.5, 0.5, 0.5, 0.5]));
    assert_eq!(b.grad(), Some(vec![1.0, 2.0, 3.0, 4.0]));
}
