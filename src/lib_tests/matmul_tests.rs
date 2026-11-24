//! Matrix multiplication for 2D, 1D, and batched operands: values, shapes, and gradient checks.

use super::*;
use approx::assert_relative_eq;

#[test]
fn matmul_2d_2d_value() {
    // (2,3) @ (3,2) -> (2,2). Use simple integer-valued data.
    let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let b = RawTensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2], false);
    let c = a.matmul(&b);
    assert_eq!(c.borrow().shape, vec![2, 2]);
    // row 0: [1+0+3, 0+2+3] = [4, 5]
    // row 1: [4+0+6, 0+5+6] = [10, 11]
    assert_eq!(c.borrow().data.to_vec(), vec![4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn matmul_2d_2d_gradcheck() {
    let a = RawTensor::new(vec![0.5, -1.0, 0.25, 1.0], &[2, 2], true);
    let b = RawTensor::new(vec![1.0, 0.5, -0.5, 1.0], &[2, 2], false);
    assert!(RawTensor::check_gradients_simple(&a, |t| t
        .matmul(&b)
        .sum()));
}

#[test]
fn matmul_2d_1d_value() {
    // (3,2) @ (2,) -> (3,)
    let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
    let v = RawTensor::new(vec![1.0, -1.0], &[2], false);
    let z = a.matmul(&v);
    assert_eq!(z.borrow().shape, vec![3]);
    // [1-2, 3-4, 5-6] = [-1, -1, -1]
    assert_eq!(z.borrow().data.to_vec(), vec![-1.0, -1.0, -1.0]);
}

#[test]
fn matmul_2d_1d_gradcheck() {
    let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let v = RawTensor::new(vec![0.5, -0.5], &[2], false);
    assert!(RawTensor::check_gradients_simple(&a, |t| t
        .matmul(&v)
        .sum()));
}

#[test]
fn matmul_1d_2d_value() {
    // (3,) @ (3,2) -> (2,)
    let v = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let m = RawTensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2], false);
    let z = v.matmul(&m);
    assert_eq!(z.borrow().shape, vec![2]);
    // row dot col: col0 = 1*1+2*0+3*1 = 4, col1 = 1*0+2*1+3*1 = 5
    assert_eq!(z.borrow().data.to_vec(), vec![4.0, 5.0]);
}

#[test]
fn matmul_1d_2d_gradcheck() {
    let v = RawTensor::new(vec![0.5, -0.5, 1.0], &[3], true);
    let m = RawTensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2], false);
    assert!(RawTensor::check_gradients_simple(&v, |t| t
        .matmul(&m)
        .sum()));
}

#[test]
fn matmul_1d_1d_dot_value() {
    let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let b = RawTensor::new(vec![4.0, -1.0, 2.0], &[3], false);
    let z = a.matmul(&b);
    // 4 - 2 + 6 = 8; result is scalar shape
    assert_relative_eq!(z.item(), 8.0, epsilon = 1e-6);
}

#[test]
fn matmul_1d_1d_dot_gradcheck() {
    let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
    let b = RawTensor::new(vec![0.5, -0.5, 1.0], &[3], false);
    assert!(RawTensor::check_gradients_simple(&a, |t| t
        .matmul(&b)
        .sum()));
}

#[test]
fn matmul_batched_value_and_shape() {
    // (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2). All-ones inputs => entries = 3.
    let a = RawTensor::ones(&[2, 2, 3]);
    let b = RawTensor::ones(&[2, 3, 2]);
    let c = a.matmul(&b);
    assert_eq!(c.borrow().shape, vec![2, 2, 2]);
    for v in c.borrow().data.to_vec() {
        assert_relative_eq!(v, 3.0, epsilon = 1e-6);
    }
}

#[test]
fn matmul_batched_gradcheck() {
    // Small batched matmul gradient check on `a`.
    let a = RawTensor::new(
        vec![
            0.5, -0.5, 1.0, 1.5, -1.0, 0.5, 0.25, 0.75, -0.25, 1.0, 0.5, -0.5,
        ],
        &[2, 2, 3],
        true,
    );
    let b = RawTensor::new(
        vec![
            0.5, 0.25, 1.0, -0.5, 0.0, 1.0, -1.0, 0.5, 0.25, 1.0, 0.5, -0.5,
        ],
        &[2, 3, 2],
        false,
    );
    assert!(RawTensor::check_gradients_simple(&a, |t| t
        .matmul(&b)
        .sum()));
}

#[test]
fn transpose_2d_shape_and_value() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let t = x.transpose();
    assert_eq!(t.borrow().shape, vec![3, 2]);
    // permuted: original (i,j) -> (j,i). Row-major: [1,4, 2,5, 3,6]
    assert_eq!(t.borrow().data.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
