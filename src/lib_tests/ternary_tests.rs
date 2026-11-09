//! Ternary ops: the `mul_acc` and `where` backward passes, including `where` with broadcasting.

use super::*;

#[test]
fn test_mulacc_backward() {
    // z = x*y + w
    let x = RawTensor::new(vec![2.0], &[1], true);
    let y = RawTensor::new(vec![3.0], &[1], true);
    let w = RawTensor::new(vec![1.0], &[1], true);
    let z = x.mulacc(&y, &w); // z = 7.0
    z.into_loss().expect("scalar loss").backward();

    assert_eq!(x.grad(), Some(vec![3.0])); // ∂z/∂x = y
    assert_eq!(y.grad(), Some(vec![2.0])); // ∂z/∂y = x
    assert_eq!(w.grad(), Some(vec![1.0])); // ∂z/∂w = 1
}

#[test]
fn test_where_backward() {
    let cond = RawTensor::from_typed(vec![true, false], &[2]);
    let x = RawTensor::new(vec![10.0, 20.0], &[2], true);
    let y = RawTensor::new(vec![30.0, 40.0], &[2], true);
    let z = cond.where_op(&x, &y); // picks [10.0, 40.0]
    z.sum().into_loss().expect("scalar loss").backward();

    assert_eq!(x.grad(), Some(vec![1.0, 0.0])); // grad flows where cond=1
    assert_eq!(y.grad(), Some(vec![0.0, 1.0])); // grad flows where cond=0
}

#[test]
fn test_where_broadcast_backward() {
    // condition shape (2,1), true branch (2,3), false branch (1,3)
    let cond = RawTensor::from_typed(vec![true, false], &[2, 1]);
    let true_branch = RawTensor::new(vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0], &[2, 3], true);
    let false_branch = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], true);
    let out = cond.where_op(&true_branch, &false_branch);
    let loss = out.sum();
    loss.into_loss().expect("scalar loss").backward();

    // Gradients: first row picks true branch, second row picks false branch
    assert_eq!(true_branch.grad(), Some(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]));
    assert_eq!(false_branch.grad(), Some(vec![1.0, 1.0, 1.0]));
}
