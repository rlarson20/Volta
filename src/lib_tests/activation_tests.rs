//! Activation layers (`ReLU`, `Sigmoid`, `Tanh`): forward against the op, and gradient checks.

use super::*;
use crate::index::fetch;

#[test]
fn relu_layer_forward_matches_op() {
    let layer = ReLU;
    let x = RawTensor::new(vec![-1.0, 0.0, 2.0, -3.0], &[4], true);
    let y = <ReLU as Layer>::forward(&layer, &x);
    assert_eq!(y.borrow().data.to_vec(), vec![0.0, 0.0, 2.0, 0.0]);
    assert!(layer.parameters().is_empty());
}

#[test]
fn relu_layer_gradcheck() {
    let layer = ReLU;
    let x = RawTensor::new(vec![-1.5, -0.5, 0.5, 1.5], &[4], true);
    assert!(RawTensor::check_gradients_simple(&x, |t| {
        <ReLU as Layer>::forward(&layer, t).sum()
    }));
}

#[test]
fn sigmoid_layer_forward_matches_op() {
    let layer = Sigmoid;
    let x = RawTensor::new(vec![0.0, 100.0, -100.0], &[3], true);
    let y = <Sigmoid as Layer>::forward(&layer, &x);
    let out = y.borrow().data.to_vec();
    assert!((fetch(&out, 0) - 0.5).abs() < 1e-6);
    assert!((fetch(&out, 1) - 1.0).abs() < 1e-6);
    assert!(fetch(&out, 2).abs() < 1e-6);
}

#[test]
fn sigmoid_layer_gradcheck() {
    let layer = Sigmoid;
    let x = RawTensor::new(vec![-1.0, 0.0, 1.0], &[3], true);
    assert!(RawTensor::check_gradients_simple(&x, |t| {
        <Sigmoid as Layer>::forward(&layer, t).sum()
    }));
}

#[test]
fn tanh_layer_forward_matches_op() {
    let layer = Tanh;
    let x = RawTensor::new(vec![0.0, 100.0, -100.0], &[3], true);
    let y = <Tanh as Layer>::forward(&layer, &x);
    let out = y.borrow().data.to_vec();
    assert!(fetch(&out, 0).abs() < 1e-6);
    assert!((fetch(&out, 1) - 1.0).abs() < 1e-6);
    assert!((fetch(&out, 2) + 1.0).abs() < 1e-6);
}

#[test]
fn tanh_layer_gradcheck() {
    let layer = Tanh;
    let x = RawTensor::new(vec![-1.0, 0.0, 1.0], &[3], true);
    assert!(RawTensor::check_gradients_simple(&x, |t| {
        <Tanh as Layer>::forward(&layer, t).sum()
    }));
}
