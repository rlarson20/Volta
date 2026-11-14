//! A weight shared between layers (#101): containers list it once, so an optimizer steps
//! it once, and an optimizer handed it twice refuses.
use volta::nn::{Layer, Linear, Module, Sequential};
use volta::{Adam, Muon, RawTensor, SGD, Tensor, VoltaError};

/// Two bias-free `Linear(2, 2)` layers sharing one weight, in a `Sequential`.
fn tied_model() -> (Sequential, Tensor) {
    let l1 = Linear::new(2, 2, false);
    let mut l2 = Linear::new(2, 2, false);
    l2.weight = l1.weight.clone();
    let w = l1.weight.clone();
    (Sequential::new(vec![Box::new(l1), Box::new(l2)]), w)
}

#[test]
fn ptr_eq_is_identity_not_equality() {
    let a = RawTensor::new(vec![1.0, 2.0], &[2], false);
    let b = RawTensor::new(vec![1.0, 2.0], &[2], false);
    assert!(a.ptr_eq(&a.clone()));
    assert!(!a.ptr_eq(&b), "equal values are not the same tensor");
}

#[test]
fn sequential_lists_a_tied_weight_once() {
    let (seq, w) = tied_model();
    let ps = seq.parameters();
    assert_eq!(ps.len(), 1);
    assert!(ps[0].ptr_eq(&w));
}

/// The #101 repro: the SGD step on a tied weight is `lr * grad`, not twice that.
#[test]
fn tied_weight_takes_one_sgd_step() {
    let (seq, w) = tied_model();
    let w0 = w.data();
    let lr = 0.1;
    let mut opt = SGD::new(seq.parameters(), lr, 0.0, 0.0);
    let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], false);
    seq.forward(&x)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    let g = w.grad().expect("the tied weight has a gradient");
    assert!(g.iter().any(|v| v.abs() > 1e-3), "degenerate gradient");
    opt.step();
    let w1 = w.data();
    assert_eq!(w0.len(), 4);
    for i in 0..w0.len() {
        let expected = w0[i] - lr * g[i];
        assert!(
            (w1[i] - expected).abs() < 1e-6,
            "element {i}: got {}, want {expected} (a doubled step would be {})",
            w1[i],
            w0[i] - 2.0 * lr * g[i]
        );
    }
}

#[test]
fn optimizers_reject_the_same_tensor_twice() {
    let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let dup = || vec![x.clone(), x.clone()];
    for (name, result) in [
        ("SGD", SGD::try_new(dup(), 0.1, 0.0, 0.0).map(drop)),
        (
            "Adam",
            Adam::try_new(dup(), 0.1, (0.9, 0.999), 1e-8, 0.0).map(drop),
        ),
        ("Muon", Muon::try_new(dup(), 0.1, 0.9, true, 5).map(drop)),
    ] {
        match result {
            Err(VoltaError::InvalidParameter(msg)) => assert!(
                msg.contains("parameter 1 is the same tensor as parameter 0"),
                "{name}: {msg}"
            ),
            other => panic!("{name}: expected InvalidParameter, got {other:?}"),
        }
    }
}

#[test]
fn distinct_tensors_with_equal_values_are_accepted() {
    let a = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let b = RawTensor::new(vec![1.0, 2.0], &[2], true);
    assert!(SGD::try_new(vec![a, b], 0.1, 0.0, 0.0).is_ok());
}
