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

/// 1.0 then many terms below half an ulp of 1.0: a running sum never moves off 1.0, an error
/// of 3e-3 for 100000 terms (S6). Pairwise summation is off by under 1e-5 (the first leaf run
/// of 128 still adds in sequence).
fn one_then_tiny(n: usize) -> Vec<f32> {
    let mut v = vec![3.0e-8f32; n];
    *crate::index::slot(&mut v, 0) = 1.0;
    v
}

#[test]
fn sum_dim_adds_a_long_strided_axis_pairwise() {
    // Axis 0 of (n, 2): column 0 is `one_then_tiny`, column 1 is zero, so the sum strides
    // by 2 through the data (inner_size > 1).
    let n = 100_001;
    let data: Vec<f32> = one_then_tiny(n)
        .into_iter()
        .flat_map(|v| [v, 0.0])
        .collect();
    let x = RawTensor::new(data, &[n, 2], false);
    let s = x.sum_dim(Axis(0), false).data();
    let want = 1.0 + 100_000.0 * 3.0e-8_f64;
    assert!(
        (f64::from(fetch(&s, 0)) - want).abs() < 1e-5,
        "col 0: {}, exact {want}",
        fetch(&s, 0)
    );
    assert_eq!(fetch(&s, 1), 0.0);
}

#[test]
fn sum_dim_of_2_pow_25_ones_is_exact() {
    let n = 1usize << 25;
    let x = RawTensor::new(vec![1.0; n], &[1, n], false);
    assert_eq!(x.sum_dim(Axis(1), false).data(), vec![33_554_432.0]);
}

#[test]
fn mean_dim_of_2_pow_25_ones_is_one() {
    let n = 1usize << 25;
    let x = RawTensor::new(vec![1.0; n], &[1, n], false);
    assert_eq!(x.mean_dim(Axis(1), false).data(), vec![1.0]);
}

#[test]
fn sum_and_mean_add_pairwise() {
    let x = RawTensor::new(one_then_tiny(100_001), &[100_001], false);
    let want = 1.0 + 100_000.0 * 3.0e-8_f64;
    assert!((f64::from(x.sum().item()) - want).abs() < 1e-5);
    assert!((f64::from(x.mean().item()) - want / 100_001.0).abs() < 1e-9);
}
