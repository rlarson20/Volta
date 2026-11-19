//! Optimizers reject non-leaf parameters and accept (and skip) frozen leaves (#102).

use volta::error::VoltaError;
use volta::nn::{Adam, Muon, SGD};
use volta::{RawTensor, Tensor};

fn non_leaf() -> (Tensor, Tensor) {
    let w = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let nl = w.elem_mul(&w);
    (w, nl)
}

fn assert_non_leaf_error<T>(r: volta::Result<T>, name: &str) {
    match r {
        Err(VoltaError::InvalidParameter(m)) => {
            assert!(
                m.contains("non-leaf") && m.contains("parameter 1"),
                "{name}: {m}"
            );
        }
        Err(e) => panic!("{name}: wrong error {e:?}"),
        Ok(_) => panic!("{name}: a non-leaf parameter was accepted"),
    }
}

#[test]
fn optimizers_reject_a_non_leaf_parameter() {
    let (w, nl) = non_leaf();
    let params = || vec![w.clone(), nl.clone()];
    assert_non_leaf_error(SGD::try_new(params(), 0.1, 0.0, 0.0), "SGD");
    assert_non_leaf_error(
        Adam::try_new(params(), 0.1, (0.9, 0.999), 1e-8, 0.0),
        "Adam",
    );
    assert_non_leaf_error(Muon::try_new(params(), 0.1, 0.9, false, 5), "Muon");
}

#[test]
#[should_panic(expected = "non-leaf")]
fn sgd_new_panics_on_a_non_leaf_parameter() {
    let (_w, nl) = non_leaf();
    let _ = SGD::new(vec![nl], 0.1, 0.0, 0.0);
}

/// A frozen leaf is accepted (as in PyTorch), gets no gradient, and is left alone by `step`
/// while the trainable parameter next to it moves.
#[test]
fn frozen_leaf_is_accepted_and_not_stepped() {
    let frozen = RawTensor::new(vec![1.0, 2.0], &[2], false);
    let trained = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let mut opt = SGD::try_new(vec![frozen.clone(), trained.clone()], 0.1, 0.0, 0.0)
        .expect("a frozen leaf is a valid parameter");
    frozen
        .elem_mul(&trained)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    opt.step();
    assert_eq!(frozen.data(), vec![1.0, 2.0]);
    assert!(frozen.grad().is_none());
    // d/dt sum(f * t) = f, so t -= 0.1 * f.
    let t = trained.data();
    assert!(
        (t[0] - 0.9).abs() < 1e-6 && (t[1] - 1.8).abs() < 1e-6,
        "{t:?}"
    );
}
