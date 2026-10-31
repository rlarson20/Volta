//! Reductions: the max-reduce backward pass.

use super::*;

#[test]
fn test_reduce_max_backward() {
    let x = RawTensor::new(vec![1.0, 5.0, 3.0], &[3], true);
    let y = x.max_reduce(); // finds 5.0 at index 1
    y.into_loss().expect("scalar loss").backward();

    // Only max element gets gradient
    assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0]));
}
