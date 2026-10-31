//! Movement ops: the `reshape` and `permute` backward passes.

use super::*;

#[test]
fn test_reshape_backward() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
    let y = x.reshape(&[2, 2]);
    let loss = y.sum();
    loss.into_loss().expect("scalar loss").backward();

    // Gradient reshapes back to [4]
    assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
}

#[test]
fn test_permute_backward() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let y = x.permute(&[Axis(1), Axis(0)]); // transpose
    let loss = y.sum();
    loss.into_loss().expect("scalar loss").backward();

    // Gradient permutes back
    assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
}
