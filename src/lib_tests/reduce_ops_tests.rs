//! Direct unit tests for the reductions: forward values, gradient routing, and gradient checks.

use super::*;
use crate::index::fetch;
use approx::assert_relative_eq;

#[test]
fn sum_forward() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], false);
    assert_relative_eq!(x.sum().item(), 10.0, epsilon = 1e-6);
}

#[test]
fn sum_backward_is_ones() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
    x.sum().into_loss().expect("scalar loss").backward();
    assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0]));
}

#[test]
fn mean_forward() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], false);
    assert_relative_eq!(x.mean().item(), 2.5, epsilon = 1e-6);
}

#[test]
fn mean_backward_is_uniform() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
    x.mean().into_loss().expect("scalar loss").backward();
    // dL/dx_i = 1/N
    let g = x.grad().unwrap();
    for v in g {
        assert_relative_eq!(v, 0.25, epsilon = 1e-6);
    }
}

#[test]
fn max_reduce_routes_grad_to_argmax() {
    let x = RawTensor::new(vec![1.0, 7.0, 3.0, 2.0], &[4], true);
    x.max_reduce().into_loss().expect("scalar loss").backward();
    assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0, 0.0]));
}

#[test]
fn sum_dim_keepdim_shape() {
    // (2, 3) -> sum over dim 1 with keepdim -> (2, 1)
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let s = x.sum_dim(Axis(1), true);
    assert_eq!(s.borrow().shape, vec![2, 1]);
    assert_eq!(s.borrow().data.to_vec(), vec![6.0, 15.0]);
}

#[test]
fn sum_dim_no_keepdim_shape() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let s = x.sum_dim(Axis(0), false);
    // (2, 3) reduced over dim 0 -> (3,)
    assert_eq!(s.borrow().shape, vec![3]);
    assert_eq!(s.borrow().data.to_vec(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn sum_dim_gradcheck() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    assert!(RawTensor::check_gradients_simple(&x, |t| t
        .sum_dim(Axis(1), true)
        .sum()));
}

#[test]
fn mean_dim_keepdim_value() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let m = x.mean_dim(Axis(1), true);
    assert_eq!(m.borrow().shape, vec![2, 1]);
    let out = m.borrow().data.to_vec();
    assert_relative_eq!(fetch(&out, 0), 2.0, epsilon = 1e-6);
    assert_relative_eq!(fetch(&out, 1), 5.0, epsilon = 1e-6);
}

#[test]
fn mean_dim_gradcheck() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    assert!(RawTensor::check_gradients_simple(&x, |t| t
        .mean_dim(Axis(0), true)
        .sum()));
}

#[test]
fn max_dim_value_and_shape() {
    // distinct values per row to avoid tie ambiguity
    let x = RawTensor::new(vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0], &[2, 3], false);
    let m = x.max_dim(Axis(1), true);
    assert_eq!(m.borrow().shape, vec![2, 1]);
    assert_eq!(m.borrow().data.to_vec(), vec![5.0, 9.0]);
}

#[test]
fn max_dim_routes_grad_to_argmax_per_row() {
    let x = RawTensor::new(vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0], &[2, 3], true);
    x.max_dim(Axis(1), true)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    // argmax row 0 -> col 1; row 1 -> col 0
    assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]));
}
