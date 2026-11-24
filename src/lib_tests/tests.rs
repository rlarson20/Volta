//! Basic ops and the chain rule: add, multiply, and sum with their backward passes, multidimensional ops, and a device-safety check.

use super::*;

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
fn test_sum_backward() {
    let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
    let loss = a.sum();
    loss.into_loss().expect("scalar loss").backward();

    assert_eq!(a.grad(), Some(vec![1.0, 1.0, 1.0]));
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
