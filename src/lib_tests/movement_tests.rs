//! Movement ops: the `reshape`, `contiguous` and `permute` backward passes.

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
fn test_contiguous_backward_reaches_the_input() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    // Distinct weights, so a gradient routed to the wrong element cannot pass.
    let w = RawTensor::new(
        vec![1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0],
        &[2, 3],
        false,
    );
    let y = RawTensor::contiguous(&x);
    assert_eq!(y.data(), x.data());
    y.elem_mul(&w)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();

    assert_eq!(x.grad(), Some(w.data()));
}

#[test]
fn test_contiguous_gradcheck() {
    let x = RawTensor::new(vec![0.3, -1.2, 2.5, 0.7, -0.4, 1.9], &[2, 3], true);
    assert!(RawTensor::check_gradients_simple(&x, |t| {
        RawTensor::contiguous(t).sum()
    }));
}

#[test]
fn test_contiguous_of_a_scalar_and_of_a_constant() {
    let s = RawTensor::new(vec![5.0], &[], true);
    RawTensor::contiguous(&s)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    assert_eq!(s.grad(), Some(vec![1.0]));

    // An input that needs no gradient gives an output that needs none.
    let c = RawTensor::new(vec![1.0, 2.0], &[2], false);
    assert!(!RawTensor::contiguous(&c).requires_grad());
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
