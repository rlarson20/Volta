//! Binary ops: the `div` and `max` backward passes.

use super::*;

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
